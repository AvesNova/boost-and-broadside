"""
Asynchronous data collection and storage for the vectorized environment.
"""

import queue
import threading
from pathlib import Path
from typing import Dict, List

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
    # ... (Truncated logic for brevity in replace, but I must provide FULL content if requested or careful chunks)
    # Wait, I am using replace_file_content, so I must provide the FULL replacement.
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

    Buffers episodes in memory until completion, then queues for writing.

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
    ):
        """Initializes the AsyncCollector.

        Args:
            data_path: Path to output HDF5.
            num_envs: Number of parallel environments.
            max_ships: Maximum number of ships per environment.
            device: Torch device.
            save_interval: Flush interval (in episodes).
        """
        self.data_path = Path(data_path)
        self.num_envs = num_envs
        self.max_ships = max_ships
        self.device = device
        self.save_interval = save_interval

        # Max steps per episode safety margin
        self.max_steps = 2000

        # Pinned CPU buffers for current episodes (pre-allocated)
        # We split 'tokens' into granular features
        self.pos_buffer = torch.zeros(
            (num_envs, self.max_steps, max_ships, 2), dtype=torch.float32, pin_memory=True
        )
        self.vel_buffer = torch.zeros(
            (num_envs, self.max_steps, max_ships, 2), dtype=torch.float16, pin_memory=True
        )
        self.health_buffer = torch.zeros(
            (num_envs, self.max_steps, max_ships), dtype=torch.float16, pin_memory=True
        )
        self.power_buffer = torch.zeros(
            (num_envs, self.max_steps, max_ships), dtype=torch.float16, pin_memory=True
        )
        self.ang_vel_buffer = torch.zeros(
            (num_envs, self.max_steps, max_ships), dtype=torch.float16, pin_memory=True
        )
        self.attitude_buffer = torch.zeros(
            (num_envs, self.max_steps, max_ships, 2), dtype=torch.float16, pin_memory=True
        )
        self.is_shooting_buffer = torch.zeros(
            (num_envs, self.max_steps, max_ships), dtype=torch.uint8, pin_memory=True
        )
        self.team_buffer = torch.zeros(
            (num_envs, self.max_steps, max_ships), dtype=torch.uint8, pin_memory=True
        )

        self.action_buffer = torch.zeros(
            (num_envs, self.max_steps, max_ships, 3), dtype=torch.uint8, pin_memory=True
        )
        self.expert_action_buffer = torch.zeros(
            (num_envs, self.max_steps, max_ships, 3), dtype=torch.uint8, pin_memory=True
        )
        self.reward_buffer = torch.zeros(
            (num_envs, self.max_steps, max_ships), dtype=torch.float32, pin_memory=True
        )
        self.skill_buffer = torch.zeros(
            (num_envs, self.max_steps, max_ships), dtype=torch.float32, pin_memory=True
        )
        self.mask_buffer = torch.ones(
            (num_envs, self.max_steps, max_ships), dtype=torch.bool, pin_memory=True
        )

        self.step_counts = torch.zeros(num_envs, dtype=torch.long)

        self.write_queue = queue.Queue()
        self.running = True

        self._init_h5()
        self.total_episodes_written = 0

        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()

    def _init_h5(self):
        """Creates HDF5 file and initial datasets."""
        with h5py.File(self.data_path, "w") as h5_file:
            # Metadata
            h5_file.attrs["max_ships"] = self.max_ships
            h5_file.attrs["token_dim"] = 15 # Kept for compatibility if we reconstruct
            
            h5_file.create_dataset("episode_lengths", (0,), maxshape=(None,), dtype="i8")

            chunk_size = 1024 * 10
            
            # --- Feature Datasets ---
            h5_file.create_dataset(
                "position",
                (0, self.max_ships, 2),
                maxshape=(None, self.max_ships, 2),
                dtype="f4",
                chunks=(chunk_size, self.max_ships, 2),
            )
            h5_file.create_dataset(
                "velocity",
                (0, self.max_ships, 2),
                maxshape=(None, self.max_ships, 2),
                dtype="f2", # float16
                chunks=(chunk_size, self.max_ships, 2),
            )
            h5_file.create_dataset(
                "health",
                (0, self.max_ships),
                maxshape=(None, self.max_ships),
                dtype="f2",
                chunks=(chunk_size, self.max_ships),
            )
            h5_file.create_dataset(
                "power",
                (0, self.max_ships),
                maxshape=(None, self.max_ships),
                dtype="f2",
                chunks=(chunk_size, self.max_ships),
            )
            h5_file.create_dataset(
                "attitude",
                (0, self.max_ships, 2),
                maxshape=(None, self.max_ships, 2),
                dtype="f2",
                chunks=(chunk_size, self.max_ships, 2),
            )
            h5_file.create_dataset(
                "ang_vel",
                (0, self.max_ships),
                maxshape=(None, self.max_ships),
                dtype="f2",
                chunks=(chunk_size, self.max_ships),
            )
            h5_file.create_dataset(
                "is_shooting",
                (0, self.max_ships),
                maxshape=(None, self.max_ships),
                dtype="u1", # bool/uint8
                chunks=(chunk_size, self.max_ships),
            )
            h5_file.create_dataset(
                "team_ids",
                (0, self.max_ships),
                maxshape=(None, self.max_ships),
                dtype="u1", # int8/uint8 - 0 or 1
                chunks=(chunk_size, self.max_ships),
            )

            # --- Standard Datasets ---
            h5_file.create_dataset(
                "actions",
                (0, self.max_ships, 3),
                maxshape=(None, self.max_ships, 3),
                dtype="u1",
                chunks=(chunk_size, self.max_ships, 3),
            )

            h5_file.create_dataset(
                "expert_actions",
                (0, self.max_ships, 3),
                maxshape=(None, self.max_ships, 3),
                dtype="u1",
                chunks=(chunk_size, self.max_ships, 3),
            )
            h5_file.create_dataset("episode_ids", (0,), maxshape=(None,), dtype="i8", chunks=(chunk_size,))
            h5_file.create_dataset(
                "agent_skills",
                (0, self.max_ships),
                maxshape=(None, self.max_ships),
                dtype="f4",
                chunks=(chunk_size, self.max_ships),
            )
            h5_file.create_dataset(
                "rewards",
                (0, self.max_ships),
                maxshape=(None, self.max_ships),
                dtype="f4",
                chunks=(chunk_size, self.max_ships),
            )
            h5_file.create_dataset(
                "returns",
                (0, self.max_ships),
                maxshape=(None, self.max_ships),
                dtype="f4",
                chunks=(chunk_size, self.max_ships),
            )
            h5_file.create_dataset(
                "action_masks",
                (0, self.max_ships),
                maxshape=(None, self.max_ships),
                dtype="bool",
                chunks=(chunk_size, self.max_ships),
            )

    def _writer_loop(self):
        """Background thread loop to write to HDF5."""
        batch = []

        try:
            with h5py.File(self.data_path, "a") as h5_file:
                while self.running or not self.write_queue.empty():
                    try:
                        item = self.write_queue.get(timeout=0.1)
                        batch.append(item)

                        if len(batch) >= self.save_interval or (not self.running and len(batch) > 0):
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
        """Writes a batch of episodes to the HDF5 file.

        Args:
            h5_file: Open HDF5 file handle.
            batch: List of episode dictionaries.
        """
        if not batch:
            return

        total_steps = sum(item["length"] for item in batch)
        # Check current size of a reliable dataset
        current_size = h5_file["actions"].shape[0]
        current_episodes = h5_file["episode_lengths"].shape[0]

        # Resize Datasets
        # Features
        h5_file["position"].resize((current_size + total_steps, self.max_ships, 2))
        h5_file["velocity"].resize((current_size + total_steps, self.max_ships, 2))
        h5_file["health"].resize((current_size + total_steps, self.max_ships))
        h5_file["power"].resize((current_size + total_steps, self.max_ships))
        h5_file["attitude"].resize((current_size + total_steps, self.max_ships, 2))
        h5_file["ang_vel"].resize((current_size + total_steps, self.max_ships))
        h5_file["is_shooting"].resize((current_size + total_steps, self.max_ships))
        h5_file["team_ids"].resize((current_size + total_steps, self.max_ships))
        
        # Standard
        h5_file["actions"].resize((current_size + total_steps, self.max_ships, 3))
        h5_file["expert_actions"].resize((current_size + total_steps, self.max_ships, 3))
        h5_file["rewards"].resize((current_size + total_steps, self.max_ships))
        h5_file["returns"].resize((current_size + total_steps, self.max_ships))
        h5_file["episode_ids"].resize((current_size + total_steps,))
        h5_file["agent_skills"].resize((current_size + total_steps, self.max_ships))
        h5_file["action_masks"].resize((current_size + total_steps, self.max_ships))

        h5_file["episode_lengths"].resize((current_episodes + len(batch),))

        # Write Data
        write_idx = current_size
        ep_idx = current_episodes

        for item in batch:
            length = item["length"]

            # Features
            h5_file["position"][write_idx : write_idx + length] = item["position"].numpy()
            h5_file["velocity"][write_idx : write_idx + length] = item["velocity"].numpy()
            h5_file["health"][write_idx : write_idx + length] = item["health"].numpy()
            h5_file["power"][write_idx : write_idx + length] = item["power"].numpy()
            h5_file["attitude"][write_idx : write_idx + length] = item["attitude"].numpy()
            h5_file["ang_vel"][write_idx : write_idx + length] = item["ang_vel"].numpy()
            h5_file["is_shooting"][write_idx : write_idx + length] = item["is_shooting"].numpy()
            h5_file["team_ids"][write_idx : write_idx + length] = item["team_ids"].numpy()
            
            # Standard
            h5_file["actions"][write_idx : write_idx + length] = item["actions"].numpy()

            if "expert_actions" in item:
                h5_file["expert_actions"][write_idx : write_idx + length] = item["expert_actions"].numpy()
            else:
                h5_file["expert_actions"][write_idx : write_idx + length] = item["actions"].numpy()

            h5_file["rewards"][write_idx : write_idx + length] = item["rewards"].numpy()
            h5_file["returns"][write_idx : write_idx + length] = item["returns"].numpy()

            h5_file["episode_ids"][write_idx : write_idx + length] = ep_idx

            if "agent_skills" in item:
                h5_file["agent_skills"][write_idx : write_idx + length] = item["agent_skills"].numpy()
            else:
                h5_file["agent_skills"][write_idx : write_idx + length] = 1.0

            h5_file["action_masks"][write_idx : write_idx + length] = True
            h5_file["episode_lengths"][ep_idx] = length

            write_idx += length
            ep_idx += 1

        self.total_episodes_written += len(batch)

    def step(
        self,
        obs_dict: Dict[str, torch.Tensor],
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        expert_actions: torch.Tensor | None = None,
        agent_skills: torch.Tensor | None = None,
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
        cpu_pos = torch.stack([pos.real, pos.imag], dim=-1).to("cpu", dtype=torch.float32, non_blocking=True)
        cpu_vel = torch.stack([vel.real, vel.imag], dim=-1).to("cpu", dtype=torch.float16, non_blocking=True)
        cpu_health = health.to("cpu", dtype=torch.float16, non_blocking=True)
        cpu_power = power.to("cpu", dtype=torch.float16, non_blocking=True)
        cpu_att = torch.stack([att.real, att.imag], dim=-1).to("cpu", dtype=torch.float16, non_blocking=True)
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

                self.write_queue.put(
                    {
                        "position": self.pos_buffer[idx, :length].clone(),
                        "velocity": self.vel_buffer[idx, :length].clone(),
                        "health": self.health_buffer[idx, :length].clone(),
                        "power": self.power_buffer[idx, :length].clone(),
                        "attitude": self.attitude_buffer[idx, :length].clone(),
                        "ang_vel": self.ang_vel_buffer[idx, :length].clone(),
                        "is_shooting": self.is_shooting_buffer[idx, :length].clone(),
                        "team_ids": self.team_buffer[idx, :length].clone(),
                        "actions": self.action_buffer[idx, :length].clone(),
                        "expert_actions": self.expert_action_buffer[idx, :length].clone(),
                        "rewards": self.reward_buffer[idx, :length].clone(),
                        "agent_skills": self.skill_buffer[idx, :length].clone(),
                        "returns": self._compute_returns(
                            self.reward_buffer[idx, :length], gamma=0.99
                        ),
                        "length": length,
                    }
                )
                self.step_counts[idx] = 0

    def _tokenize(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Assembles ship tokens."""
        ang_vel = obs.get("ang_vel", torch.zeros_like(obs["position"].real))
        return compile_tokens(
            obs["position"],
            obs["velocity"],
            obs["team_id"],
            obs["health"],
            obs["power"],
            obs["attitude"],
            ang_vel,
            obs["is_shooting"],
        )

    def _compute_returns(self, rewards: torch.Tensor, gamma: float) -> torch.Tensor:
        """Computes returns-to-go."""
        returns = torch.zeros_like(rewards)
        ret = torch.zeros(rewards.shape[1], dtype=torch.float32)
        for t in reversed(range(len(rewards))):
            ret = rewards[t] + gamma * ret
            returns[t] = ret
        return returns

    def close(self):
        """Signals shutdown and joins writer thread."""
        self.running = False
        self.writer_thread.join()
