from typing import Any
import numpy as np
import torch
from omegaconf import DictConfig
from wandb import agent

from agents.agents import create_agent
from components.components import create_component
from env.event import GameEvent
from env.state import State
from .env.env import Environment


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

        self.components = {
            components_name: create_component(**components_config)
            for components_name, components_config in config.components.items()
        }

        self.obs_history: list[State] = []

    def reset(self, game_mode: str):
        obs, _ = self.env.reset(game_mode=game_mode)

        self.obs_history.append(obs)

    def step(self):
        # actions = {
        #     ship_id: torch.zeros((self.env.max_ships,), dtype=torch.float32)
        #     for ship_id in range(self.env.max_ships)
        # }
        terminated = False
        teams = {0: [0, 1, 2, 3], 1: [4, 5, 6, 7]}

        while not terminated:
            actions = {
                team_id: self.agents["scripted"](self.obs_history[-1], ship_ids)
                for team_id, ship_ids in teams.items()
            }

            obs, rewards, terminated, _, info = self.env.step(actions=actions)

            self.obs_history.append(obs)
