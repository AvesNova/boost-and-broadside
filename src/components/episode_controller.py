from typing import Any
import time

from .base import Component


class EpisodeController(Component):
    """Component for controlling episode flow and termination"""

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize episode controller

        Args:
            config: Episode controller configuration
        """
        super().__init__(config)

        self.max_steps = config.get("max_steps", 10000)
        self.time_limit = config.get("time_limit", None)  # In seconds
        self.enable_early_termination = config.get("enable_early_termination", True)

        # Timing state
        self.start_time = None
        self.current_step = 0

    def on_episode_start(self, coordinator: Any) -> None:
        """Initialize timing when episode starts"""
        self.start_time = time.time()
        self.current_step = 0

        # Set coordinator state
        coordinator._current_step = 0
        coordinator._should_terminate = False

    def on_step(
        self, coordinator: Any, obs: dict, actions: dict, rewards: dict, info: dict
    ) -> None:
        """Check termination conditions after each step"""
        self.current_step += 1
        coordinator._current_step = self.current_step

        # Check step limit
        if self.current_step >= self.max_steps:
            coordinator._should_terminate = True
            coordinator._truncated = True
            return

        # Check time limit
        if self.time_limit and self.start_time:
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= self.time_limit:
                coordinator._should_terminate = True
                coordinator._truncated = True
                return

        # Check early termination conditions
        if self.enable_early_termination:
            self._check_early_termination(coordinator, obs, info)

    def on_episode_end(self, coordinator: Any) -> None:
        """Finalize episode state"""
        coordinator._terminated = (
            not coordinator._truncated if hasattr(coordinator, "_truncated") else False
        )

        # Print episode summary
        if hasattr(coordinator, "_current_step"):
            print(f"Episode ended after {coordinator._current_step} steps")
            if self.start_time:
                elapsed_time = time.time() - self.start_time
                print(f"Duration: {elapsed_time:.2f} seconds")

    def _check_early_termination(self, coordinator: Any, obs: dict, info: dict) -> None:
        """Check for early termination conditions"""
        # Check if all ships on a team are dead
        if "alive" in obs and "team_id" in obs:
            alive_teams = set()

            for ship_id in range(obs["alive"].shape[0]):
                if obs["alive"][ship_id, 0].item() > 0:
                    team_id = obs["team_id"][ship_id, 0].item()
                    alive_teams.add(team_id)

            # If only one or zero teams have alive ships, terminate
            if len(alive_teams) <= 1:
                coordinator._should_terminate = True
                coordinator._terminated = True

    def get_progress(self) -> dict[str, Any]:
        """Get current progress information"""
        progress = {
            "current_step": self.current_step,
            "max_steps": self.max_steps,
            "progress_percentage": min(
                100.0, (self.current_step / self.max_steps) * 100
            ),
        }

        if self.time_limit and self.start_time:
            elapsed_time = time.time() - self.start_time
            progress.update(
                {
                    "elapsed_time": elapsed_time,
                    "time_limit": self.time_limit,
                    "time_progress_percentage": min(
                        100.0, (elapsed_time / self.time_limit) * 100
                    ),
                }
            )

        return progress

    def should_continue(self) -> bool:
        """Check if the episode should continue"""
        if self.current_step >= self.max_steps:
            return False

        if self.time_limit and self.start_time:
            elapsed_time = time.time() - self.start_time
            if elapsed_time >= self.time_limit:
                return False

        return True
