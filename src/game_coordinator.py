from typing import Any
import numpy as np
import torch
from omegaconf import DictConfig
from wandb import agent

from agents.agents import create_agent
from components.components import create_component
from env.event import GameEvent
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

    def reset(self, game_mode: str):
        self.env.reset(game_mode=game_mode)

    def step(self):
        actions = {
            ship_id: torch.zeros((self.env.max_ships,), dtype=torch.float32)
            for ship_id in range(self.env.max_ships)
        }
        terminated = False

        while not terminated:
            obs, rewards, terminated, _, info = self.env.step(actions=actions)
