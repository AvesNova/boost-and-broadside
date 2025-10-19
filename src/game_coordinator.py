from typing import Any
import numpy as np
import torch
from omegaconf import DictConfig
from wandb import agent

from agents.agents import create_agent
from components.components import create_component
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

        def reset(self): ...

        def step(self): ...
