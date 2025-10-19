from typing import Any
import numpy as np
import torch
from omegaconf import DictConfig
from wandb import agent

from agents.agents import create_agent
from agents.tokenizer import observation_to_tokens
from env.event import GameEvent
from env.state import State
from env.env import Environment


class GameCoordinator:
    """
    Central coordinator that orchestrates the game between environment and agents
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the game coordinator

        Args:
            config: Configuration dictionary containing all setup
        """
        self.config = config

        self.env = Environment(**config.environment)

        self.agents = {
            agent_name: create_agent(**agent_config)
            for agent_name, agent_config in config.agents.items()
        }

        self.obs_history: list[State] = []

    def reset(self, game_mode: str):
        obs, _ = self.env.reset(game_mode=game_mode)

        self.obs_history.append(obs)
        self.all_tokens = {
            0: observation_to_tokens(obs=obs, perspective=0),
            1: observation_to_tokens(obs=obs, perspective=1),
        }

    def step(self):
        terminated = False
        teams = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}

        while not terminated:
            # TODO: We need to handle configurable agents

            team_actions = {
                team_id: self.agents["scripted"](self.obs_history[-1], ship_ids)
                for team_id, ship_ids in teams.items()
            }

            # Flatten team actions to ship actions
            actions = {}
            for team_id, ship_ids in teams.items():
                ship_actions = team_actions[team_id]
                actions.update(ship_actions)

            obs, rewards, terminated, _, info = self.env.step(actions=actions)

            self.obs_history.append(obs)
            self.all_tokens[0] = torch.cat(
                [self.all_tokens[0], observation_to_tokens(obs=obs, perspective=0)],
                dim=0,
            )
            self.all_tokens[1] = torch.cat(
                [self.all_tokens[1], observation_to_tokens(obs=obs, perspective=1)],
                dim=0,
            )
            # TODO: We need to have a data collector that can collect a lot of data that can be quickly loaded.
            # The data loader should save the time_step, episode_id, tokens, actions and rewards
            # All the data should be aggregated into a single tensor for all attrributes across all episodes

    def close(self):
        """Clean up resources"""
        if hasattr(self.env, "close"):
            self.env.close()
