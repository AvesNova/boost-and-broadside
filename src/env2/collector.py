"""
Asynchronous data collection and storage for the vectorized environment.
"""

import queue
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

import h5py
import numpy as np
import torch


@torch.jit.script
def compile_tokens(
    position: torch.Tensor,
    velocity: torch.Tensor,
    team_id: torch.Tensor,
    health: torch.Tensor,
    power: torch.Tensor,
    attitude: torch.Tensor,
    ang_vel: torch.Tensor,
    is_shooting: torch.Tensor,
) -> torch.Tensor:
    """JIT-compiled function to assemble token tensor on GPU.

    Args:
        position: Ship positions (complex).
        velocity: Ship velocities (complex).
        team_id: Ship team IDs.
        health: Ship health.
        power: Ship power.
        attitude: Ship orientation (complex).
        ang_vel: Ship angular velocity.
        is_shooting: Boolean indicating if shooting.

    Returns:
        A tensor of shape (Batch, NumShips, 15) containing the compiled tokens.
    """
    batch_size, num_ships = position.shape
    device = position.device
    tokens = torch.zeros((batch_size, num_ships, 15), dtype=torch.float32, device=device)

    # 0: Team
    tokens[..., 0] = team_id.float()
    # 1: Health
    tokens[..., 1] = health
    # 2: Power
    tokens[..., 2] = power
    # 3,4: Pos (Raw)
    tokens[..., 3] = position.real
    tokens[..., 4] = position.imag
    tokens[..., 5] = velocity.real
    tokens[..., 6] = velocity.imag
    tokens[..., 9] = ang_vel
    tokens[..., 10] = attitude.real
    tokens[..., 11] = attitude.imag
    tokens[..., 12] = is_shooting.float()

    return tokens


class AsyncCollector:
    """Collects data from Vectorized Environment and writes to HDF5 asynchronously.

    Buffers episodes in memory until completion, then clones and processes them
    using a worker pool before queuing for HDF5 writing.

    Attributes:
        data_path: Path to the output HDF5 file.
        num_envs: Number of parallel environments.
        max_ships: Maximum number of ships per environment.
        device: Torch device.
        save_interval: Number of episodes to buffer before writing to disk.
        max_steps: Maximum episode length limit for pre-allocation.
    """

    def __init__(
        self,
        data_path: str,
        num_envs: int,
        max_ships: int,
        device: torch.device,
        save_interval: int = 100,
        max_steps: int = 2000,
        num_workers: int = 4,
        total_steps: Optional[int] = None,
    ):
        """Initializes the AsyncCollector.

        Args:
            data_path: Path to output HDF5.
            num_envs: Number of parallel environments.
            max_ships: Maximum number of ships per environment.
            device: Torch device.
            save_interval: Flush interval (in episodes).
            max_steps: Maximum steps per episode (corresponds to buffer_steps).
            num_workers: Number of processing worker threads.
            total_steps: Optional total transition target for pre-allocation.
        """
        self.data_path = Path(data_path)
        self.num_envs = num_envs
        self.max_ships = max_ships
        self.device = device
        self.save_interval = save_interval
        self.max_steps = max_steps

        # Pinned CPU buffers for current episodes (pre-allocated)
        buffer_shape = (num_envs, self.max_steps, max_ships)
        self.pos_buffer = torch.zeros((*buffer_shape, 2), dtype=torch.float32, pin_memory=True)
        self.vel_buffer = torch.zeros((*buffer_shape, 2), dtype=torch.float16, pin_memory=True)
        self.health_buffer = torch.zeros(buffer_shape, dtype=torch.float16, pin_memory=True)
        self.power_buffer = torch.zeros(buffer_shape, dtype=torch.float16, pin_memory=True)
        self.ang_vel_buffer = torch.zeros(buffer_shape, dtype=torch.float16, pin_memory=True)
        self.attitude_buffer = torch.zeros(
            (*buffer_shape, 2), dtype=torch.float16, pin_memory=True
        )
        self.is_shooting_buffer = torch.zeros(buffer_shape, dtype=torch.uint8, pin_memory=True)
        self.team_buffer = torch.zeros(buffer_shape, dtype=torch.uint8, pin_memory=True)

        self.action_buffer = torch.zeros((*buffer_shape, 3), dtype=torch.uint8, pin_memory=True)
        self.expert_action_buffer = torch.zeros(
            (*buffer_shape, 3), dtype=torch.uint8, pin_memory=True
        )
        self.reward_buffer = torch.zeros(buffer_shape, dtype=torch.float32, pin_memory=True)
        self.skill_buffer = torch.zeros(buffer_shape, dtype=torch.float32, pin_memory=True)

        self.step_counts = torch.zeros(num_envs, dtype=torch.long)

        self.write_queue = queue.Queue()
        self.running = True

        self._init_h5(total_steps=total_steps)
        self.total_episodes_written = 0
        self.total_transitions_completed = 0  # Track transitions queued for writing

        # Background Pool for cloning and processing (returns, etc.)
        self.process_pool = ThreadPoolExecutor(max_workers=num_workers)

        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()

    def _init_h5(self, total_steps: Optional[int] = None):
        """Creates HDF5 file and initial datasets.

        Args:
            total_steps: Optional total transition target for pre-allocation.
        """
        with h5py.File(self.data_path, "w") as h5_file:
            # Metadata
            h5_file.attrs["max_ships"] = self.max_ships
            h5_file.attrs["token_dim"] = 9  # Kept for compatibility if we reconstruct

            init_size = total_steps if total_steps is not None else 0
            ep_init_size = 0  # We don't know episode count in advance

            h5_file.create_dataset(
                "episode_lengths", (ep_init_size,), maxshape=(None,), dtype="i8", compression="lzf"
            )

            chunk_size = 1024 * 10
            
            def create_feature_ds(name, shape_suffix, dtype, chunks_suffix):
                return h5_file.create_dataset(
                    name,
                    (init_size, *shape_suffix),
                    maxshape=(None, *shape_suffix),
                    dtype=dtype,
                    chunks=(chunk_size, *chunks_suffix),
                    compression="lzf",
                )

            # --- Feature Datasets ---
            create_feature_ds("position", (self.max_ships, 2), "f4", (self.max_ships, 2))
            create_feature_ds("velocity", (self.max_ships, 2), "f2", (self.max_ships, 2))
            create_feature_ds("health", (self.max_ships,), "f2", (self.max_ships,))
            create_feature_ds("power", (self.max_ships,), "f2", (self.max_ships,))
            create_feature_ds("attitude", (self.max_ships, 2), "f2", (self.max_ships, 2))
            create_feature_ds("ang_vel", (self.max_ships,), "f2", (self.max_ships,))
            create_feature_ds("is_shooting", (self.max_ships,), "u1", (self.max_ships,))
            create_feature_ds("team_ids", (self.max_ships,), "u1", (self.max_ships,))

            # --- Standard Datasets ---
            create_feature_ds("actions", (self.max_ships, 3), "u1", (self.max_ships, 3))
            create_feature_ds("expert_actions", (self.max_ships, 3), "u1", (self.max_ships, 3))
            h5_file.create_dataset(
                "episode_ids",
                (init_size,),
                maxshape=(None,),
                dtype="i8",
                chunks=(chunk_size,),
                compression="lzf",
            )
            create_feature_ds("agent_skills", (self.max_ships,), "f4", (self.max_ships,))
            create_feature_ds("rewards", (self.max_ships,), "f4", (self.max_ships,))
            create_feature_ds("returns", (self.max_ships,), "f4", (self.max_ships,))
            create_feature_ds("action_masks", (self.max_ships,), "bool", (self.max_ships,))

    def _writer_loop(self):
        """Background thread loop to write to HDF5."""
        batch = []

        try:
            with h5py.File(self.data_path, "a") as h5_file:
                while self.running or not self.write_queue.empty():
                    try:
                        # Pull one item to wait if empty
                        item = self.write_queue.get(timeout=0.1)
                        batch = [item]

                        # GREEDILY pull all available items from the queue for bulk writing
                        while not self.write_queue.empty():
                            try:
                                batch.append(self.write_queue.get_nowait())
                            except queue.Empty:
                                break

                        if batch:
                            self._flush_batch_to_handle(h5_file, batch)
                            batch = []

                    except queue.Empty:
                        continue
                    except Exception as e:
                        print(f"Writer Loop Item Error: {e}")

                if batch:
                    self._flush_batch_to_handle(h5_file, batch)

        except Exception as e:
            print(f"Writer Thread Fatal Error: {e}")

    def _flush_batch_to_handle(self, h5_file: h5py.File, batch: List[Dict[str, torch.Tensor]]):
        """Writes a batch of episodes to the HDF5 file in a single bulk operation.

        Args:
            h5_file: Open HDF5 file handle.
            batch: List of episode dictionaries.
        """
        if not batch:
            return

        total_new_steps = sum(item["length"] for item in batch)
        # Check current size of a reliable dataset
        current_size = h5_file["actions"].shape[0]
        current_episodes = h5_file["episode_lengths"].shape[0]

        # Target total size
        target_size = current_size + total_new_steps
        
        # Grow if necessary (if pre-allocation was smaller or not used)
        if h5_file["actions"].shape[0] < target_size:
            for key in ["position", "velocity", "health", "power", "attitude", "ang_vel", 
                        "is_shooting", "team_ids", "actions", "expert_actions", "rewards", 
                        "returns", "episode_ids", "agent_skills", "action_masks"]:
                h5_file[key].resize((target_size, *h5_file[key].shape[1:]))

        h5_file["episode_lengths"].resize((current_episodes + len(batch),))

        # Bulk concatenate all data in the batch
        # We process each key to form a single large numpy array for the write
        def aggregate(key):
            if key == "episode_ids":
                # Special handling for episode_ids: assign sequential IDs
                ids = []
                for i, item in enumerate(batch):
                    ids.append(np.full((item["length"],), current_episodes + i, dtype=np.int64))
                return np.concatenate(ids, axis=0)
            elif key == "action_masks":
                return np.ones((total_new_steps, self.max_ships), dtype=bool)
            elif key == "agent_skills":
                return np.concatenate([item.get("agent_skills", torch.ones((item["length"], self.max_ships))) for item in batch], axis=0)
            elif key == "expert_actions":
                return np.concatenate([item.get("expert_actions", item["actions"]) for item in batch], axis=0)
            else:
                return np.concatenate([item[key] for item in batch], axis=0)

        # Write each field in one call
        for key in ["position", "velocity", "health", "power", "attitude", "ang_vel", 
                    "is_shooting", "team_ids", "actions", "expert_actions", "rewards", 
                    "returns", "episode_ids", "agent_skills", "action_masks"]:
            h5_file[key][current_size:target_size] = aggregate(key)

        # Write episode lengths
        h5_file["episode_lengths"][current_episodes : current_episodes + len(batch)] = [
            item["length"] for item in batch
        ]

        self.total_episodes_written += len(batch)

    def step(
        self,
        obs_dict: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        expert_actions: Optional[torch.Tensor] = None,
        agent_skills: Optional[torch.Tensor] = None,
    ):
        """Records a single step of data from all environments.

        Args:
            obs_dict: Observation dictionary at time T.
            actions: Actions taken at time T.
            rewards: Rewards received.
            dones: Termination flag.
            expert_actions: (Optional) Optimal actions.
            agent_skills: (Optional) Skill levels for each ship.
        """
        # Deconstruct Obs
        pos = obs_dict["position"]
        vel = obs_dict["velocity"]
        health = obs_dict["health"]
        power = obs_dict["power"]
        att = obs_dict["attitude"]
        ang_vel = obs_dict["ang_vel"]
        is_shooting = obs_dict["is_shooting"]
        team_id = obs_dict["team_id"]

        num_envs = self.num_envs

        # Prepare CPU Tensors (Non-blocking)
        # Note: We cast here for buffer compatibility
        cpu_pos = torch.stack([pos.real, pos.imag], dim=-1).to(
            "cpu", dtype=torch.float32, non_blocking=True
        )
        cpu_vel = torch.stack([vel.real, vel.imag], dim=-1).to(
            "cpu", dtype=torch.float16, non_blocking=True
        )
        cpu_health = health.to("cpu", dtype=torch.float16, non_blocking=True)
        cpu_power = power.to("cpu", dtype=torch.float16, non_blocking=True)
        cpu_att = torch.stack([att.real, att.imag], dim=-1).to(
            "cpu", dtype=torch.float16, non_blocking=True
        )
        cpu_ang_vel = ang_vel.to("cpu", dtype=torch.float16, non_blocking=True)
        cpu_shooting = is_shooting.to("cpu", dtype=torch.uint8, non_blocking=True)
        cpu_team = team_id.to("cpu", dtype=torch.uint8, non_blocking=True)

        cpu_actions = actions.to("cpu", dtype=torch.uint8, non_blocking=True)
        cpu_rewards = rewards.to("cpu", non_blocking=True)

        if expert_actions is not None:
            cpu_expert = expert_actions.to("cpu", dtype=torch.uint8, non_blocking=True)
        else:
            cpu_expert = cpu_actions

        if agent_skills is not None:
            cpu_skills = agent_skills.to("cpu", dtype=torch.float32, non_blocking=True)
        else:
            cpu_skills = torch.ones((num_envs, self.max_ships), dtype=torch.float32)

        indices = torch.arange(num_envs)
        steps = self.step_counts

        # Buffer data
        write_steps = torch.clamp(steps, max=self.max_steps - 1)

        self.pos_buffer[indices, write_steps] = cpu_pos
        self.vel_buffer[indices, write_steps] = cpu_vel
        self.health_buffer[indices, write_steps] = cpu_health
        self.power_buffer[indices, write_steps] = cpu_power
        self.attitude_buffer[indices, write_steps] = cpu_att
        self.ang_vel_buffer[indices, write_steps] = cpu_ang_vel
        self.is_shooting_buffer[indices, write_steps] = cpu_shooting
        self.team_buffer[indices, write_steps] = cpu_team

        self.action_buffer[indices, write_steps] = cpu_actions
        self.expert_action_buffer[indices, write_steps] = cpu_expert
        self.reward_buffer[indices, write_steps] = cpu_rewards
        self.skill_buffer[indices, write_steps] = cpu_skills

        self.step_counts += 1

        # Handle terminations
        if dones.any():
            done_indices = torch.nonzero(dones, as_tuple=True)[0]

            for idx in done_indices.tolist():
                length = self.step_counts[idx].item()
                if length > self.max_steps:
                    length = self.max_steps

                # Offload cloning and processing to worker pool
                self.process_pool.submit(self._finalize_episode, idx, length)
                self.step_counts[idx] = 0

    def _finalize_episode(self, env_idx: int, length: int):
        """Processes and clones a finished episode from the buffer.

        Runs in a worker thread.
        """
        try:
            episode = {
                "position": self.pos_buffer[env_idx, :length].clone(),
                "velocity": self.vel_buffer[env_idx, :length].clone(),
                "health": self.health_buffer[env_idx, :length].clone(),
                "power": self.power_buffer[env_idx, :length].clone(),
                "attitude": self.attitude_buffer[env_idx, :length].clone(),
                "ang_vel": self.ang_vel_buffer[env_idx, :length].clone(),
                "is_shooting": self.is_shooting_buffer[env_idx, :length].clone(),
                "team_ids": self.team_buffer[env_idx, :length].clone(),
                "actions": self.action_buffer[env_idx, :length].clone(),
                "expert_actions": self.expert_action_buffer[env_idx, :length].clone(),
                "rewards": self.reward_buffer[env_idx, :length].clone(),
                "agent_skills": self.skill_buffer[env_idx, :length].clone(),
                "returns": self._compute_returns(
                    self.reward_buffer[env_idx, :length], gamma=0.99
                ),
                "length": length,
            }
            self.total_transitions_completed += length
            self.write_queue.put(episode)
        except Exception as e:
            print(f"Error finalizing episode from env {env_idx}: {e}")

    def _compute_returns(self, rewards: torch.Tensor, gamma: float) -> torch.Tensor:
        """Computes returns-to-go using vectorized operations.

        G_t = R_t + gamma * G_{t+1}
        """
        # Note: input rewards is (L, N)
        seq_len = rewards.shape[0]
        if seq_len == 0:
            return torch.empty_like(rewards)

        # Vectorized returns computation:
        # G_t = sum_{j=t}^{L-1} gamma^{j-t} R_j
        # G_t = (sum_{j=t}^{L-1} gamma^j R_j) / gamma^t

        device = rewards.device
        gamma_powers = torch.pow(gamma, torch.arange(seq_len, device=device, dtype=torch.float32))

        # Y_j = gamma^j * R_j
        # Use unsqueeze to multiply across the ships dimension (dim 1)
        weighted_rewards = rewards * gamma_powers.unsqueeze(1)

        # Suffix sums of weighted_rewards
        # torch.cumsum is prefix sum. Suffix sum = TotalSum - PrefixSum(t-1)
        # Or more easily: flip -> cumsum -> flip
        suffix_sums = torch.flip(torch.cumsum(torch.flip(weighted_rewards, dims=[0]), dim=0), dims=[0])

        # G_t = suffix_sums / gamma^t
        returns = suffix_sums / gamma_powers.unsqueeze(1)

        return returns

    def close(self):
        """Signals shutdown and joins writer thread."""
        self.running = False
        self.process_pool.shutdown(wait=True)
        self.writer_thread.join()
