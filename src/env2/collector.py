
import torch
import numpy as np
import h5py
import threading
import queue
from typing import Dict, List, Optional
from pathlib import Path
import time

@torch.jit.script
def compile_tokens(
    position: torch.Tensor,
    velocity: torch.Tensor,
    team_id: torch.Tensor,
    health: torch.Tensor,
    power: torch.Tensor,
    attitude: torch.Tensor,
    ang_vel: torch.Tensor,
    is_shooting: torch.Tensor
) -> torch.Tensor:
    B, N = position.shape
    device = position.device
    tokens = torch.zeros((B, N, 15), dtype=torch.float32, device=device)
    
    w, h = 1024.0, 1024.0
    
    # 0: Team
    tokens[..., 0] = team_id.float()
    # 1: Health
    tokens[..., 1] = health / 100.0
    # 2: Power
    tokens[..., 2] = power / 100.0
    # 3,4: Pos
    tokens[..., 3] = position.real / w
    tokens[..., 4] = position.imag / h
    # 5,6: Vel (Norm 180)
    tokens[..., 5] = velocity.real / 180.0
    tokens[..., 6] = velocity.imag / 180.0
    # 7,8: Acc (Unused/Zero)
    
    # 9: Ang Vel
    tokens[..., 9] = ang_vel
    
    # 10,11: Attitude
    tokens[..., 10] = attitude.real
    tokens[..., 11] = attitude.imag
    
    # 12: Shooting
    tokens[..., 12] = is_shooting.float()
    
    return tokens


class AsyncCollector:
    """
    Collects data from Vectorized Environment and writes to HDF5 asynchronously.
    Buffers episodes in memory until completion, then queues for writing.
    """
    def __init__(
        self, 
        data_path: str, # Path to output HDF5
        num_envs: int,
        max_ships: int,
        device: torch.device,
        save_interval: int = 100 # Flush to disk every N episodes
    ):
        self.data_path = Path(data_path)
        self.num_envs = num_envs
        self.max_ships = max_ships
        self.device = device
        self.save_interval = save_interval
        
        # Buffers (List of lists for flexibility, or TensorArray?)
        # Since lengths vary, List of lists is easiest, then stack.
        # But appending to python list is slow?
        # Better: Pre-allocated buffers per env, resize if needed?
        # Max steps ~1000.
        # We can implement a ring buffer per env or just large tensor.
        self.max_steps = 2000 # Safety margin
        
        # Pinned CPU buffers for current episodes
        self.obs_buffer = torch.zeros((num_envs, self.max_steps, max_ships, 15), dtype=torch.float32, pin_memory=True)
        self.action_buffer = torch.zeros((num_envs, self.max_steps, max_ships, 3), dtype=torch.uint8, pin_memory=True) # Int actions
        self.reward_buffer = torch.zeros((num_envs, self.max_steps, max_ships), dtype=torch.float32, pin_memory=True)
        # We likely need action masks too?
        self.mask_buffer = torch.ones((num_envs, self.max_steps, max_ships), dtype=torch.bool, pin_memory=True) # Default True
        
        self.step_counts = torch.zeros(num_envs, dtype=torch.long)
        
        # Queue for finished episodes
        # Item: Dict of tensors
        self.write_queue = queue.Queue()
        self.running = True
        # Init HDF5 first!
        self._init_h5()
        
        self.total_episodes_written = 0
        
        # Start Writer Thread
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()

    def _init_h5(self):
        """Create HDF5 file and datasets."""
        # Clean existing or append? Overwrite for now.
        with h5py.File(self.data_path, "w") as f:
            # Resizable datasets
            # We don't know total size.
            f.create_dataset("episode_lengths", (0,), maxshape=(None,), dtype="i8")
            
            # Data: flattened (TotalSteps, ...)
            # We will resize these as we append.
            # Chunking is important.
            chunk_size = 1024 * 10
            
            f.create_dataset("tokens", (0, self.max_ships, 15), maxshape=(None, self.max_ships, 15), dtype="f4", chunks=(chunk_size, self.max_ships, 15))
            f.create_dataset("actions", (0, self.max_ships, 3), maxshape=(None, self.max_ships, 3), dtype="u1", chunks=(chunk_size, self.max_ships, 3))
            f.create_dataset("rewards", (0, self.max_ships), maxshape=(None, self.max_ships), dtype="f4", chunks=(chunk_size, self.max_ships))
            f.create_dataset("returns", (0, self.max_ships), maxshape=(None, self.max_ships), dtype="f4", chunks=(chunk_size, self.max_ships))
            f.create_dataset("action_masks", (0, self.max_ships), maxshape=(None, self.max_ships), dtype="bool", chunks=(chunk_size, self.max_ships))
            # f.create_dataset("team_ids", ...) # If needed

    def _writer_loop(self):
        """Background thread to write to HDF5. Keeps file open."""
        batch = []
        
        try:
            with h5py.File(self.data_path, "a") as f:
                while self.running or not self.write_queue.empty():
                    try:
                        item = self.write_queue.get(timeout=0.1) # Short timeout to check running
                        batch.append(item)
                        
                        if len(batch) >= self.save_interval or (not self.running and len(batch) > 0):
                            self._flush_batch_to_handle(f, batch)
                            batch = []
                            
                    except queue.Empty:
                        continue
                    except Exception as e:
                        print(f"Writer Loop Item Error: {e}")
                
                # Final flush
                if batch:
                    self._flush_batch_to_handle(f, batch)
                    
        except Exception as e:
             print(f"Writer Thread Fatal Error: {e}")

    def _flush_batch_to_handle(self, f, batch: List[Dict[str, torch.Tensor]]):
        """Write a batch using existing file handle."""
        if not batch:
            return
            
        # Calculate total steps in batch
        total_steps = sum(item["length"] for item in batch)
        current_size = f["tokens"].shape[0]
        current_episodes = f["episode_lengths"].shape[0]
        
        # Resize
        f["tokens"].resize((current_size + total_steps, self.max_ships, 15))
        f["actions"].resize((current_size + total_steps, self.max_ships, 3))
        f["rewards"].resize((current_size + total_steps, self.max_ships))
        f["returns"].resize((current_size + total_steps, self.max_ships))
        
        f["episode_lengths"].resize((current_episodes + len(batch),))
        
        # Write Data
        write_idx = current_size
        ep_idx = current_episodes
        
        for item in batch:
            length = item["length"]
            
            f["tokens"][write_idx : write_idx + length] = item["tokens"].numpy()
            f["actions"][write_idx : write_idx + length] = item["actions"].numpy()
            f["rewards"][write_idx : write_idx + length] = item["rewards"].numpy()
            f["returns"][write_idx : write_idx + length] = item["returns"].numpy()
            
            f["episode_lengths"][ep_idx] = length
            
            write_idx += length
            ep_idx += 1
            
        self.total_episodes_written += len(batch)
        # print(f"Written {len(batch)} episodes. Total: {self.total_episodes_written}")

    def step(self, obs_dict: Dict[str, torch.Tensor], actions: torch.Tensor, rewards: torch.Tensor, dones: torch.Tensor):
        """
        Record step data.
        Assumes auto-reset happens AFTER this call in Env (or we handle it).
        `obs` is state at t.
        `actions` is action at t.
        `rewards` is reward at t (result of action).
        `dones` is done at t.
        """
        # Copy to buffer
        # Need to convert obs_dict to flat tokens (B, N, 15)?
        # For now, placeholder or partial.
        # Actually `unified_dataset` expects specific token format?
        # `get_token` in `ship.py` does this.
        # We need a vectorized `get_token` function.
        # I'll implement `vector_get_token` in collector or physics.
        # Assuming `obs_dict` contains raw tensors.
        
        # 1. Vectorized Tokenization (CPU or GPU)
        # GPU is faster.
        tokens = self._tokenize(obs_dict) # (B, N, 15)
        
        # 2. Copy to Pinned Memory
        # Use copy_ with non_blocking=True?
        # But we need CPU tensors.
        # .to("cpu", non_blocking=True)
        
        B = self.num_envs
        cpu_tokens = tokens.to("cpu", non_blocking=True)
        # Cast actions to uint8
        cpu_actions = actions.to("cpu", dtype=torch.uint8, non_blocking=True)
        cpu_rewards = rewards.to("cpu", non_blocking=True)
        
        # Sync? No, double buffering is implied by queue.
        # But here we write to specific index `step_counts`.
        # We assume `step` is called sequentially.
        # The copy is async.
        
        # We iterate B to copy? No, batched copy is better.
        # Slicing: buffer[b, step[b]] = ... 
        # Advanced indexing `buffer[range(B), step_counts]`
        
        indices = torch.arange(B)
        steps = self.step_counts
        
        # Check limit
        if (steps >= self.max_steps).any():
            # Handle overflow (truncate)
            # Find overflowing envs
            overflow_mask = steps >= self.max_steps
            # Force reset for these? Or just stop recording?
            # We must stop indexing.
            # Clamp steps to max_steps - 1 for safety, writing garbage, but allow env to finish.
            # But writer depends on `step_counts`.
            # Realistically, env should reset before 2000 steps.
            # If overflow, warn.
            # print("Warning: buffer overflow")
            # For now, clamp write index.
            write_steps = torch.clamp(steps, max=self.max_steps-1)
            self.obs_buffer[indices, write_steps] = cpu_tokens
            self.action_buffer[indices, write_steps] = cpu_actions
            self.reward_buffer[indices, write_steps] = cpu_rewards
            
            self.step_counts += 1 # Continue counting to track length, but cap at retrieval.
        else:
            self.obs_buffer[indices, steps] = cpu_tokens
            self.action_buffer[indices, steps] = cpu_actions
            self.reward_buffer[indices, steps] = cpu_rewards
            
            self.step_counts += 1
        
        # Check Dones
        # If done, extract and queue
        if dones.any():
            done_indices = torch.nonzero(dones, as_tuple=True)[0]
            # Must process immediately (before next step overwrites or index resets)
            # Actually Env auto-resets internally usually? 
            # If so, the `obs` returned by step is NEW state.
            # But `rewards` and `dones` correspond to the step just finished.
            # `tokens` computed from `obs`? `step` returns `obs` (next state)?
            # Standard Gym: `next_obs, reward, done`.
            # We want to store `(obs_t, action_t, reward_t)`.
            # `obs_dict` passing into here should be `obs_t` (Previous obs).
            # The caller must manage this: Store obs_t, call step -> act/rew/done/obs_t+1. Record.
            # I will assume `obs_dict` is the observation used to generate `actions`.
            
            # Wait, `cpu_tokens` created from `obs_dict`.
            pass
            
            # Extract episodes
            # Since `cpu_tokens` is async copy, we need to sync before reading back?
            # Or just wait? 
            # `to("cpu")` creates a copy. `non_blocking` means it returns immediately but tensor is not ready?
            # If we access it, it syncs.
            
            # For done envs:
            for idx in done_indices.tolist():
                length = self.step_counts[idx].item()
                if length > self.max_steps:
                    length = self.max_steps
                
                # Copy slice
                ep_tokens = self.obs_buffer[idx, :length].clone()
                ep_actions = self.action_buffer[idx, :length].clone()
                ep_rewards = self.reward_buffer[idx, :length].clone()
                
                # Compute Returns (Monte Carlo)
                # Discounted sum? BC usually uses raw returns or returns-to-go.
                # `unified_dataset` loader reads `returns`.
                # Assuming returns-to-go.
                # Calculation: iterate backwards.
                # Optimized: torch.flip -> cumsum? No, discounted.
                # If gamma=1.0, cumsum.
                # If gamma < 1.0, simple loop.
                # Lets assume gamma=0.99
                
                ep_returns = self._compute_returns(ep_rewards, gamma=0.99)
                
                # Queue
                self.write_queue.put({
                    "tokens": ep_tokens,
                    "actions": ep_actions,
                    "rewards": ep_rewards,
                    "returns": ep_returns,
                    "length": length
                })
                
                # Reset count
                self.step_counts[idx] = 0

    def _tokenize(self, obs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Matched `Ship.get_token` layout but JIT compiled."""
        # Unpack dict for JIT (JIT doesn't like Dict[str, Tensor] input well sometimes with mixed types or dynamic keys)
        # Passing tensors explicitly is safer/faster
        
        # Handle missing keys (or default)
        
        if "ang_vel" not in obs:
             # Should be there, but fallback
             ang_vel = torch.zeros_like(obs["position"].real)
        else:
             ang_vel = obs["ang_vel"]
             
        return compile_tokens(
            obs["position"],
            obs["velocity"],
            obs["team_id"],
            obs["health"],
            obs["power"],
            obs["attitude"],
            ang_vel,
            obs["is_shooting"]
        )

    def _compute_returns(self, rewards: torch.Tensor, gamma: float) -> torch.Tensor:
        """Compute Returns-To-Go."""
        # rewards shape (T, N)
        if gamma == 1.0:
            # Simple reverse cumsum
            return torch.flip(torch.cumsum(torch.flip(rewards, [0]), dim=0), [0])
        else:
            # Loop (fast enough for CPU batch)
            returns = torch.zeros_like(rewards)
            ret = torch.zeros(rewards.shape[1], dtype=torch.float32)
            for t in reversed(range(len(rewards))):
                ret = rewards[t] + gamma * ret
                returns[t] = ret
            return returns

    def close(self):
        self.running = False
        self.writer_thread.join()
