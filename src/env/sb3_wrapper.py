"""
Stable Baselines3 wrapper for the Environment.

Adapts the multi-agent Environment to a single-agent interface compatible
with Stable Baselines3, handling the opponent agent internally.
"""

from typing import Any

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from omegaconf import DictConfig

from agents.agents import create_agent
from agents.tokenizer import observation_to_tokens
from env.env import Environment
from env.constants import HumanActions


class SB3Wrapper(gym.Wrapper):
    """
    Gymnasium wrapper to adapt the Environment for Stable Baselines3.

    Features:
    - Flattens the multi-agent action interface to a single-agent interface.
    - Handles the opponent agent (Team 1) internally.
    - Exposes Team 0's observation and reward.
    """

    def __init__(self, env: Environment, config: DictConfig):
        """
        Initialize the SB3 wrapper.

        Args:
            env: The base Environment instance.
            config: Configuration dictionary.
        """
        super().__init__(env)
        self.config = config
        self.max_ships = env.max_ships
        self.num_actions = len(HumanActions)

        # Define Action Space: MultiBinary for all potential ships
        # Shape: (max_ships * num_actions,)
        self.action_space = spaces.MultiBinary(self.max_ships * self.num_actions)

        # Define Observation Space
        self.token_dim = config.train.model.transformer.token_dim

        self.observation_space = spaces.Dict(
            {
                "tokens": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.max_ships, self.token_dim),
                    dtype=np.float32,
                )
            }
        )

        # Initialize opponent agent
        opponent_type = config.team2
        opponent_config_dict = config.agents.get(opponent_type, {})
        opponent_config = opponent_config_dict.get("agent_config", {})

        # If it's a scripted agent, ensure world_size is in config
        if opponent_type == "scripted":
            if "world_size" not in opponent_config:
                opponent_config["world_size"] = env.world_size

        self.opponent = create_agent(opponent_type, opponent_config)
        self.opponent_team_id = 1

    def reset(
        self, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """
        Reset the environment.

        Args:
            seed: Random seed.
            options: Additional options (e.g., game_mode).

        Returns:
            Tuple of (observation, info).
        """
        game_mode = options.get("game_mode", "1v1") if options else "1v1"

        obs, info = self.env.reset(game_mode=game_mode)

        # Convert observation to tokens for Team 0
        tokens = observation_to_tokens(obs, perspective=0)  # (1, max_ships, token_dim)

        # Remove batch dim for SB3
        tokens = tokens.squeeze(0).numpy()

        return {"tokens": tokens}, info

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Flat action array of shape (max_ships * num_actions,).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        # 1. Parse Team 0 actions
        team_0_actions = {}

        # Reshape action to (max_ships, num_actions)
        action_reshaped = action.reshape(self.max_ships, self.num_actions)

        # Extract actions for Team 0 ships
        if self.env.state:
            for ship_id, ship in self.env.state.ships.items():
                if ship.team_id == 0 and ship.alive:
                    ship_action = action_reshaped[ship_id]
                    team_0_actions[ship_id] = torch.tensor(
                        ship_action, dtype=torch.int64
                    )

        # 2. Get Team 1 (Opponent) actions
        obs = self.env.get_observation()

        # Identify Team 1 ships
        team_1_ship_ids = []
        if self.env.state:
            for ship_id, ship in self.env.state.ships.items():
                if ship.team_id == self.opponent_team_id and ship.alive:
                    team_1_ship_ids.append(ship_id)

        team_1_actions = {}
        if team_1_ship_ids:
            team_1_actions = self.opponent(obs, team_1_ship_ids)

        # 3. Combine actions
        all_actions = {**team_0_actions, **team_1_actions}

        # 4. Step environment
        next_obs, rewards, terminated, truncated, info = self.env.step(all_actions)

        # 5. Process return for Team 0
        reward = rewards.get(0, 0.0)

        # Convert next observation to tokens
        tokens = observation_to_tokens(next_obs, perspective=0)
        tokens = tokens.squeeze(0).numpy()

        return {"tokens": tokens}, reward, terminated, truncated, info
