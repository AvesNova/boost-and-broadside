
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
            
            mb_actions = self.actions[:, mb_env_inds]
            mb_logprobs = self.logprobs[:, mb_env_inds]
            mb_advantages = self.advantages[:, mb_env_inds]
            mb_returns = self.returns[:, mb_env_inds]
            mb_values = self.values[:, mb_env_inds]
            
            # Initial states for these envs
            mb_mamba_state = {}
            for layer_idx in self.initial_conv_state:
                conv = self.initial_conv_state[layer_idx][mb_env_inds]
                ssm = self.initial_ssm_state[layer_idx][mb_env_inds]
                mb_mamba_state[layer_idx] = (conv, ssm)
            
            yield (
                mb_obs, mb_actions, mb_logprobs, mb_advantages, mb_returns, mb_values,
                mb_mamba_state
            )
