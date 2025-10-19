from typing import Any

from .base import Component
from .renderer import Renderer
from .data_collector import DataCollector
from .episode_controller import EpisodeController


def create_component(
    component_type: str, component_config: dict[str, Any]
) -> Component:
    match component_type:
        case "renderer":
            return Renderer(component_config)
        case "data_collector":
            return DataCollector(component_config)
        case "episode_controller":
            return EpisodeController(component_config)
        case _:
            raise ValueError(f"Unknown component type: {component_config["type"]}")
