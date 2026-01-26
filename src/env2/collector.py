
import os
import time
import queue
import threading
import torch
import numpy as np
import h5py
from pathlib import Path

class AsyncCollector:
    """
    Collects data from TensorEnv (GPU) and writes to HDF5 (Disk) asynchronously.
    Uses pinned CPU memory for efficient GPU->CPU transfer.
    """
    
    def __init__(
        self,
        output_dir: str,
        buffer_size: int, # Number of steps to buffer before writing
        batch_size: int,  # Env batch size
        max_ships: int,
        token_dim: int = 15,
        action_dim: int = 3,
        num_teams: int = 2,
        device: str = "cuda"
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.buffer_size = buffer_size
        self.B = batch_size
        self.N = max_ships
        self.output_path = self.output_dir / f"data_{int(time.time())}.h5"
        
        # Pinned Memory Buffers
        # We use strict structure matching unified_dataset
        self.cpu_buffer = {
            "tokens": torch.zeros((buffer_size, batch_size, max_ships, token_dim), dtype=torch.float32).pin_memory(),
            "actions": torch.zeros((buffer_size, batch_size, max_ships, action_dim), dtype=torch.long).pin_memory(),
            "rewards": torch.zeros((buffer_size, batch_size, max_ships), dtype=torch.float32).pin_memory(), # or per team? Usually N.
            "dones": torch.zeros((buffer_size, batch_size), dtype=torch.bool).pin_memory(),
            # Additional metadata if needed
            "team_ids": torch.zeros((buffer_size, batch_size, max_ships), dtype=torch.long).pin_memory()
        }
        
        self.cursor = 0
        self.lock = threading.Lock()
        
        # Background Writer
        self.write_queue = queue.Queue()
        self.running = True
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
        
    def add(
        self,
        tokens: torch.Tensor,   # (B, N, D)
        actions: torch.Tensor,  # (B, N, 3)
        rewards: torch.Tensor,  # (B, 2) or (B, N)
        dones: torch.Tensor,    # (B,)
        team_ids: torch.Tensor  # (B, N)
    ):
        """
        Add a batch of data. Non-blocking GPU->CPU copy.
        """
        # If full, flush (swap buffers logic or block?)
        # Simple for now: copy to pinned buffer. If full, submit to write queue.
        # Ideally we have double buffering (2 huge buffers). 
        # For simplicity: Use one buffer, if full, clone and send to writer? 
        # Cloning is fast enough compared to disk IO.
        
        idx = self.cursor
        
        # Async copy
        # Ensure input is detached
        
        # Handle Rewards shape
        # Input rewards might be (B, 2) [Team Rewards].
        # Output expects (B, N) [Agent Rewards].
        # Broadcast/Map team rewards to agents.
        if rewards.dim() == 2 and rewards.shape[1] == 2:
            # Map based on team_ids
            # team_ids: (B, N)
            # rewards: (B, 2)
            # (B, N)
            r0 = rewards[:, 0].unsqueeze(1)
            r1 = rewards[:, 1].unsqueeze(1)
            mapped_rewards = torch.where(team_ids == 0, r0, r1)
        else:
            mapped_rewards = rewards
            
        self.cpu_buffer["tokens"][idx].copy_(tokens, non_blocking=True)
        self.cpu_buffer["actions"][idx].copy_(actions, non_blocking=True)
        self.cpu_buffer["rewards"][idx].copy_(mapped_rewards, non_blocking=True)
        self.cpu_buffer["dones"][idx].copy_(dones, non_blocking=True)
        self.cpu_buffer["team_ids"][idx].copy_(team_ids, non_blocking=True)
        
        self.cursor += 1
        
        if self.cursor >= self.buffer_size:
            self._flush()
            
    def _flush(self):
        """Submit current buffer to writer and reset."""
        # We need to ensure GPU sync before assuming CPU buffer is valid?
        # copy_(non_blocking=True) requires synchronization if we read immediately.
        # But we are submitting to a queue. The writer thread will read it.
        # Synchronization happens naturally or we force iter?
        # Safe bet: torch.cuda.synchronize() or stream sync.
        # This blocks the main thread slightly but ensures data safety.
        # Better: use current stream.
        
        # Clone the buffer data to send to thread (so we can overwrite main buffer)
        # Deepcopy of tensor data?
        # Actually, standard way: Double Buffer.
        # Swap `self.cpu_buffer` with a free one.
        pass # Only one buffer implemented implies blocking flush or overwrite risk.
        
        # Let's implement simplified swap.
        # We can allocate a new buffer? Expensive.
        # Pre-allocate 2 sets.
        
        # Refactor init to have buffers = [dict, dict]
        # But let's stick to cloning for MVP robustness if buffer_size isn't huge.
        # Cloning (Buffer * B * N) might be heavy.
        
        # Sending:
        chunk = {
            k: v.clone() for k, v in self.cpu_buffer.items()
        }
        self.write_queue.put(chunk)
        self.cursor = 0
        
    def _writer_loop(self):
        """Background thread to write HDF5."""
        
        # Open HDF5 (append mode)
        # We keep file open or open/close?
        # Default: keep open.
        
        with h5py.File(self.output_path, "w") as f:
            # Create datasets
            # (Total, B, N, ...)
            
            dsets = {}
            
            while self.running or not self.write_queue.empty():
                try:
                    chunk = self.write_queue.get(timeout=1.0)
                except queue.Empty:
                    continue
                    
                # Organize dims: (Time, Batch, ...) -> (Time*Batch, ...)
                # Flatten first two dims for Dataset
                
                # Iterate keys
                for k, v in chunk.items():
                    # v: (Buffer, B, ...)
                    # Flatten to (Buffer*B, ...)
                    flat_data = v.flatten(0, 1).numpy() 
                    
                    if k not in dsets:
                        # Create
                        shape = list(flat_data.shape)
                        maxshape = list(shape)
                        maxshape[0] = None
                        dsets[k] = f.create_dataset(
                            k, data=flat_data, maxshape=tuple(maxshape), chunks=True
                        )
                    else:
                        # Resize
                        dset = dsets[k]
                        old_len = dset.shape[0]
                        add_len = flat_data.shape[0]
                        dset.resize(old_len + add_len, axis=0)
                        dset[old_len:] = flat_data
                        
                f.flush()
                self.write_queue.task_done()
                
    def close(self):
        if self.cursor > 0:
            self._flush()
        self.running = False
        self.writer_thread.join()
        print(f"Data saved to {self.output_path}")

