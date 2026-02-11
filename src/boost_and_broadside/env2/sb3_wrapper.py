import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig

from boost_and_broadside.env2.coordinator_wrapper import TensorEnvWrapper
from boost_and_broadside.agents.tokenizer import observation_to_tokens
from boost_and_broadside.core.constants import NUM_POWER_ACTIONS, NUM_TURN_ACTIONS, NUM_SHOOT_ACTIONS, STATE_DIM

class SB3Wrapper(gym.Env):
    """
    Gymnasium wrapper for TensorEnvWrapper compatible with Stable Baselines 3.
    
    Converts TensorEnv observations to tokenized tensors and adapts the action space.
    Controls Team 0 by default.
    """
    def __init__(self, env: TensorEnvWrapper, config: DictConfig):
        self.env = env
        self.config = config
        self.max_ships = config.environment.max_ships
        self.token_dim = STATE_DIM # Defined in tokenizer.py docstring
        
        # Observation Space
        # SB3 policies will expect a Dict observation if using MultiInputPolicy
        # containing the key "tokens".
        self.observation_space = gym.spaces.Dict({
            "tokens": gym.spaces.Box(
                low=-np.inf, high=np.inf, 
                shape=(self.max_ships, self.token_dim), 
                dtype=np.float32
            )
        })
        
        # Action Space
        # Multi-Agent control: Flattened MultiDiscrete
        # shape: (MaxShips * 3) where each triplet is [Power, Turn, Shoot]
        # Gym MultiDiscrete expects a 1D array of dimension sizes
        nvec = [NUM_POWER_ACTIONS, NUM_TURN_ACTIONS, NUM_SHOOT_ACTIONS] * self.max_ships
        self.action_space = gym.spaces.MultiDiscrete(np.array(nvec, dtype=np.int64))
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self.env.reset()
        return self._transform_obs(obs), info
        
    def step(self, action):
        """
        Step the environment with the given action.
        
        Args:
            action: Flattened array of actions for all ships.
        """
        # Reshape action to (MaxShips, 3)
        # action is likely a numpy array from SB3
        action_reshaped = action.reshape(self.max_ships, 3)
        
        actions_dict = {}
        for i in range(self.max_ships):
            ship_id = i 
            # Convert to tensor for TensorEnv
            # Note: TensorEnvWrapper expect {ship_id: tensor([p, t, s])}
            actions_dict[ship_id] = torch.tensor(action_reshaped[i], device=self.env.device)
            
        obs, rewards_dict, terminated, truncated, info = self.env.step(actions_dict)
        
        # Return reward for Team 0
        reward = float(rewards_dict[0])
        
        return self._transform_obs(obs), reward, terminated, truncated, info
        
    def _transform_obs(self, obs_dict):
        """Convert raw TensorEnv observation to tokenized dictionary."""
        # tokenizer expects pure tensors in dict
        tokens = observation_to_tokens(
            obs_dict, 
            perspective=0, 
            world_size=self.env.world_size
        ) # Returns (1, N, 15)
        
        return {"tokens": tokens.squeeze(0).cpu().numpy()}
