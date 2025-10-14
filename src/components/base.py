from abc import ABC, abstractmethod
from typing import Any


class Component(ABC):
    """Base class for all components in the game"""

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize component

        Args:
            config: Component-specific configuration
        """
        self.config = config or {}

    def on_episode_start(self, coordinator: Any) -> None:
        """
        Called when an episode starts

        Args:
            coordinator: GameCoordinator instance
        """
        pass

    def on_step(
        self, coordinator: Any, obs: dict, actions: dict, rewards: dict, info: dict
    ) -> None:
        """
        Called after each environment step

        Args:
            coordinator: GameCoordinator instance
            obs: Observation dictionary
            actions: Actions taken this step
            rewards: Rewards received this step
            info: Additional info from environment
        """
        pass

    def on_episode_end(self, coordinator: Any) -> None:
        """
        Called when an episode ends

        Args:
            coordinator: GameCoordinator instance
        """
        pass

    def close(self) -> None:
        """Cleanup resources"""
        pass

    @property
    def enabled(self) -> bool:
        """Check if component is enabled"""
        return self.config.get("enabled", True)
