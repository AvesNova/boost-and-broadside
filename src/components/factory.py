from typing import Any

from .base import Component
from .renderer import Renderer
from .data_collector import DataCollector
from .episode_controller import EpisodeController


class ComponentFactory:
    """Factory for creating components from configuration"""

    @staticmethod
    def create_component(component_config: dict[str, Any]) -> Component:
        """
        Create a component from configuration

        Args:
            component_config: Dictionary containing component configuration

        Returns:
            Component instance
        """
        component_type = component_config["type"]

        if component_type == "renderer":
            return Renderer(component_config)
        elif component_type == "data_collector":
            return DataCollector(component_config)
        elif component_type == "episode_controller":
            return EpisodeController(component_config)
        else:
            raise ValueError(f"Unknown component type: {component_type}")
