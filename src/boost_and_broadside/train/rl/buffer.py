
import torch
import numpy as np
from typing import Optional, Generator, Dict

class GPUBuffer:
    """
    A GPU-resident replay buffer for PPO.
    
    Stores all rollout data as pre-allocated tensors on the device to avoid
    CPU-GPU transfer overhead. Implements GAE computation natively.
    Supports Dictionary Observations for complex environments.
    """
    
    def __init__(
        self,
        num_steps: int,
        num_envs: int,
        obs_shapes: Dict[str, tuple], # Dict of shapes for each key
        action_shape: tuple,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: torch.device = torch.device("cuda")
    ):
        self.num_steps = num_steps
        self.num_envs = num_envs
        self.num_ships = obs_shapes['state'][0] # Assuming dim 0 is num_ships
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device
        
        # Calculate buffer size
        self.batch_size = num_steps * num_envs
        
        # Storage initialization
        # Obs is a dictionary of tensors
        self.obs = {}
        for key, shape in obs_shapes.items():
            self.obs[key] = torch.zeros((num_steps, num_envs, *shape), device=device, dtype=torch.float32)
        
        # Actions: (T, B, N, 3)
        # action_shape passed in is (N, 3)
        self.actions = torch.zeros((num_steps, num_envs, *action_shape), device=device, dtype=torch.float32)
        
        # Per-Ship Scalars
        self.logprobs = torch.zeros((num_steps, num_envs, self.num_ships), device=device, dtype=torch.float32)
        self.rewards = torch.zeros((num_steps, num_envs, self.num_ships), device=device, dtype=torch.float32)
        self.dones = torch.zeros((num_steps, num_envs), device=device, dtype=torch.float32) # Dones are per-env
        self.values = torch.zeros((num_steps, num_envs, self.num_ships), device=device, dtype=torch.float32)
        self.advantages = torch.zeros((num_steps, num_envs, self.num_ships), device=device, dtype=torch.float32)
        self.returns = torch.zeros((num_steps, num_envs, self.num_ships), device=device, dtype=torch.float32)
        
        # Mamba States Storage (Initial per rollout)
        self.initial_conv_state = {}
        self.initial_ssm_state = {}
        
        self.ptr = 0
        
    def reset(self):
        self.ptr = 0
        self.initial_conv_state = {}
        self.initial_ssm_state = {}
        
    def add(
        self, 
        obs: Dict[str, torch.Tensor], 
        action: torch.Tensor, 
        logprob: torch.Tensor, 
        reward: torch.Tensor, 
        done: torch.Tensor, 
        value: torch.Tensor
    ):
        """
        Add a step to the buffer. All inputs should be GPU tensors.
        """
        if self.ptr >= self.num_steps:
            raise IndexError("Buffer is full")
            
        for key, val in obs.items():
            if key in self.obs:
                self.obs[key][self.ptr] = val
        
        self.actions[self.ptr] = action
        self.logprobs[self.ptr] = logprob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = done
        self.values[self.ptr] = value
        
        self.ptr += 1
        
    def store_initial_state(self, mamba_state: Dict):
        """Stores the Mamba state at the beginning of the rollout."""
        # mamba_state is {layer_idx: (conv, ssm)}
        for layer_idx, (conv, ssm) in mamba_state.items():
            self.initial_conv_state[layer_idx] = conv.clone()
            self.initial_ssm_state[layer_idx] = ssm.clone()

    def compute_gae(self, next_value: torch.Tensor, next_done: torch.Tensor):
        """
        Computes GAE separately.
        next_value: (num_envs, num_ships)
        next_done: (num_envs,)
        """
        with torch.no_grad():
            lastgaelam = 0
            for t in reversed(range(self.num_steps)):
                if t == self.num_steps - 1:
                    nextnonterminal = 1.0 - next_done # (B,)
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - self.dones[t + 1] # (B,)
                    nextvalues = self.values[t + 1]
                
                # Broadcast non-terminal to ships (B, 1)
                nextnonterminal = nextnonterminal.unsqueeze(-1)
                
                delta = self.rewards[t] + self.gamma * nextvalues * nextnonterminal - self.values[t]
                self.advantages[t] = lastgaelam = delta + self.gamma * self.gae_lambda * nextnonterminal * lastgaelam
                
            self.returns = self.advantages + self.values

    def get_minibatch_iterator(self, num_minibatches: int) -> Generator:
        """
        Yields minibatches of (T, N_sub, ...).
        """
        envs_per_batch = self.num_envs // num_minibatches
        env_indices = np.arange(self.num_envs)
        np.random.shuffle(env_indices)
        
        for start in range(0, self.num_envs, envs_per_batch):
            end = start + envs_per_batch
            mb_env_inds = env_indices[start:end]
            
            # Slice Dict Obs
            mb_obs = {}
            for key, val in self.obs.items():
                mb_obs[key] = val[:, mb_env_inds]
            
            # Next Obs (Targets)
            # Obs buffer is usually (T, B, ...).
            # We want obs[t+1] for each t.
            # But standard buffer only stores obs[t].
            # PPO usually discards the very last obs of rollout unless we store T+1 obs.
            # However, for efficiency, people often just shift:
            # target[t] = obs[t+1].
            # We need to handle the boundary.
            # Wait, `self.obs` has shape `num_steps`.
            # Where is the "next" observation?
            # In PPO, we usually overwrite obs.
            # If we want next_obs targets, we need to store them OR shift.
            # But the buffer is circular or fixed length.
            # Let's check `add` method. It stores `obs` (current).
            # So `self.obs[t]` is state at time t.
            # We want `self.obs[t+1]` as target for state at t.
            # But we only have `num_steps` storage.
            # We need to peek at the NEXT batch or use the `next_obs` from the last step of the rollout?
            # Actually, CleanerRL stores `next_obs` separately or uses `obs` with offset.
            # Since `self.obs` is (T, B, ...), and we yield (T, B, ...), getting `obs[t+1]` is tricky inside the batch because T is the full sequence.
            # Solution: We can create `mb_next_obs` by rolling `mb_obs` by -1 and filling the last step with the bootstrapping `next_obs` if we had saved it?
            # But we don't save the final `next_obs` of the rollout in the buffer class persistently.
            # Alternative: Just use `mb_obs` as target but shifted:
            # target[t] = mb_obs[t+1]
            # target[T-1] = ??? (Need the observation AFTER the rollout).
            # The bootstrap phase has `next_obs`. We should modify `buffer.compute_gae` or `add` to optionally store `next_obs`?
            # Or pass `last_next_obs` to `get_minibatch_iterator`?
            
            # For now, let's implement a "Same-Batch Shift" with a zero-pad/duplicate for the last step, or simpler:
            # We can't perfectly predict T+1 without storage.
            # Let's Modify `add` to store `next_obs`? Too expensive?
            # Actually, `self.obs` is `num_steps`.
            # If we want State Loss, we really should be storing (obs, next_obs) pairs or (obs sequence of length T+1).
            # Let's use `mb_obs` rolled by 1 as an approximation, but that's wrong (first step predicts second).
            # mb_next_obs[t] = mb_obs[t+1].
            # For the last step `mb_next_obs[T-1]`, we don't have the data in `mb_obs`.
            # Valid approach: Mask the loss for the last step?
            # Or better: We can modify `add` to overwrite `obs` at `step+1`? No `obs` is inputs.
            # Let's assume we Mask the last step for now.
            
            mb_next_obs = {}
            for key, val in self.obs.items():
                 # Shift by 1 time step
                 shifted = torch.roll(val[:, mb_env_inds], shifts=-1, dims=0)
                 # The last element is now the first element (wrapped), which is WRONG.
                 # We must mask it in the loss or fill it if possible.
                 # Let's rely on masking in the Trainer (set loss_mask[T-1] = 0).
                 mb_next_obs[key] = shifted

            mb_actions = self.actions[:, mb_env_inds]
            mb_logprobs = self.logprobs[:, mb_env_inds]
            mb_advantages = self.advantages[:, mb_env_inds]
            mb_returns = self.returns[:, mb_env_inds]
            mb_values = self.values[:, mb_env_inds]
            mb_dones = self.dones[:, mb_env_inds]
            mb_rewards = self.rewards[:, mb_env_inds]
            
            # Initial states for these envs
            mb_mamba_state = {}
            for layer_idx in self.initial_conv_state:
                conv = self.initial_conv_state[layer_idx][mb_env_inds]
                ssm = self.initial_ssm_state[layer_idx][mb_env_inds]
                mb_mamba_state[layer_idx] = (conv, ssm)
            
            yield (
                mb_obs, mb_next_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns, mb_values, mb_dones, mb_rewards,
                mb_mamba_state
            )
