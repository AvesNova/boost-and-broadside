from typing import Any

from .base import Component


class Renderer(Component):
    """Component for rendering the game state"""

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize renderer

        Args:
            config: Renderer configuration
        """
        super().__init__(config)

        self.render_mode = config.get("render_mode", "human")
        self.target_fps = config.get("target_fps", 60)
        self.renderer = None

    def on_episode_start(self, coordinator: Any) -> None:
        """Initialize renderer when episode starts"""
        if self.render_mode == "human" and coordinator.env:
            # Set environment render mode if not already set
            if coordinator.env.render_mode != "human":
                # Create new environment with human rendering
                import sys
                import os

                sys.path.append(
                    os.path.join(os.path.dirname(__file__), "..", "..", "..", "src")
                )
                from env import Environment

                env_config = {
                    "world_size": coordinator.env.world_size,
                    "max_ships": coordinator.env.max_ships,
                    "agent_dt": coordinator.env.agent_dt,
                    "physics_dt": coordinator.env.physics_dt,
                    "render_mode": "human",
                }
                coordinator.env = Environment(**env_config)

            self.renderer = coordinator.env.renderer

            # Register human agents with renderer
            for agent in coordinator.agent_manager.agents.values():
                if agent.get_agent_type() == "human":
                    if hasattr(agent, "set_renderer"):
                        agent.set_renderer(self.renderer)
                    else:
                        # For backward compatibility
                        agent.renderer = self.renderer

    def on_step(
        self, coordinator: Any, obs: dict, actions: dict, rewards: dict, info: dict
    ) -> None:
        """Render after each step"""
        if self.render_mode == "human" and coordinator.env and coordinator.env.state:
            # Handle events and render
            if not coordinator.env.renderer.handle_events():
                # User closed window
                coordinator._should_terminate = True

            coordinator.env.render(coordinator.env.state[-1])

    def on_episode_end(self, coordinator: Any) -> None:
        """Cleanup after episode ends"""
        pass

    def close(self) -> None:
        """Cleanup renderer resources"""
        if self.renderer:
            self.renderer.close()
            self.renderer = None
