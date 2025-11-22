import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from omegaconf import DictConfig

from agents.agents import create_agent
from env.env import Environment
from env.constants import Actions


class SB3Wrapper(gym.Wrapper):
    """
    Gymnasium wrapper to adapt the Environment for Stable Baselines3.
    
    - Flattens the multi-agent action interface to a single-agent interface.
    - Handles the opponent agent (Team 1) internally.
    - Exposes Team 0's observation and reward.
    """

    def __init__(self, env: Environment, config: DictConfig):
        super().__init__(env)
        self.config = config
        self.max_ships = env.max_ships
        self.num_actions = len(Actions)

        # Define Action Space: MultiBinary for all potential ships
        # Shape: (max_ships * num_actions,)
        # We use MultiBinary because the underlying actions are discrete/binary flags
        self.action_space = spaces.MultiBinary(self.max_ships * self.num_actions)

        # Define Observation Space
        # We pass the raw tokens to the model
        # Token dim is 12 based on tokenizer.py (13 actually? let's check tokenizer.py again)
        # tokenizer.py:
        # torch.stack([...], dim=1)
        # 1 (team_eq) + 1 (health) + 1 (power) + 2 (pos_sin/cos) + 2 (pos_sin/cos) + 2 (vel) + 2 (att) + 1 (shooting) = 12?
        # Let's verify token_dim from config or code.
        # Config says token_dim: 12.
        self.token_dim = config.train.model.transformer.token_dim
        
        self.observation_space = spaces.Dict({
            "tokens": spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.max_ships, self.token_dim),
                dtype=np.float32
            )
        })

        # Initialize opponent agent
        # We'll use the agent configured for team2 in the config, or default to scripted
        opponent_type = config.get("team2", "scripted")
        opponent_config = config.agents.get(opponent_type, {}).get("agent_config", {})
        
        # If it's a scripted agent, we need to make sure it has the right config
        if opponent_type == "scripted":
             # Ensure world_size is in config if needed by ScriptedAgent
             if "world_size" not in opponent_config:
                 opponent_config["world_size"] = env.world_size

        self.opponent = create_agent(opponent_type, opponent_config)
        self.opponent_team_id = 1

    def reset(self, seed=None, options=None):
        # Reset the environment
        # We can pass options to control the game mode (e.g. "1v1", "2v2")
        game_mode = options.get("game_mode", "1v1") if options else "1v1"
        
        obs, info = self.env.reset(game_mode=game_mode)
        
        # Convert observation to tokens for Team 0
        from agents.tokenizer import observation_to_tokens
        tokens = observation_to_tokens(obs, perspective=0) # (1, max_ships, token_dim)
        
        # Remove batch dim for SB3
        tokens = tokens.squeeze(0).numpy()
        
        return {"tokens": tokens}, info

    def step(self, action):
        # action is a flat numpy array of shape (max_ships * num_actions,)
        
        # 1. Parse Team 0 actions
        team_0_actions = {}
        
        # Reshape action to (max_ships, num_actions)
        action_reshaped = action.reshape(self.max_ships, self.num_actions)
        
        # Identify which ships belong to Team 0
        # We need to look at the current state to know which ships are alive and on Team 0
        # But the action vector covers ALL potential ship IDs.
        # We can just extract actions for all IDs that map to Team 0.
        
        # In the current env implementation, ship_id is the index.
        # We can iterate through all ships in the state.
        if self.env.state:
            for ship_id, ship in self.env.state.ships.items():
                if ship.team_id == 0 and ship.alive:
                    # Extract action for this ship
                    ship_action = action_reshaped[ship_id]
                    # Convert to tensor as expected by env
                    team_0_actions[ship_id] = torch.tensor(ship_action, dtype=torch.int64)

        # 2. Get Team 1 (Opponent) actions
        # The opponent needs the observation from its perspective
        obs = self.env.get_observation()
        
        # Identify Team 1 ships
        team_1_ship_ids = []
        if self.env.state:
            for ship_id, ship in self.env.state.ships.items():
                if ship.team_id == self.opponent_team_id and ship.alive:
                    team_1_ship_ids.append(ship_id)
        
        team_1_actions = {}
        if team_1_ship_ids:
            # Opponent agent expects dict observation and list of ship IDs
            # Some agents might expect specific format, but create_agent returns a callable
            # that generally matches the interface.
            # ScriptedAgent takes (obs, ship_ids)
            team_1_actions = self.opponent(obs, team_1_ship_ids)

        # 3. Combine actions
        all_actions = {**team_0_actions, **team_1_actions}
        
        # 4. Step environment
        next_obs, rewards, terminated, truncated, info = self.env.step(all_actions)
        
        # 5. Process return for Team 0
        # Reward is a dict {team_id: reward}
        reward = rewards.get(0, 0.0)
        
        # Convert next observation to tokens
        from agents.tokenizer import observation_to_tokens
        tokens = observation_to_tokens(next_obs, perspective=0)
        tokens = tokens.squeeze(0).numpy()
        
        return {"tokens": tokens}, reward, terminated, truncated, info
