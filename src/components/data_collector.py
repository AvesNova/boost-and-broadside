from typing import Any
import os
import pickle
import gzip
from datetime import datetime

from .base import Component


class DataCollector(Component):
    """Component for collecting episode data for training"""

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize data collector

        Args:
            config: Data collector configuration
        """
        super().__init__(config)

        self.output_path = config.get("output_path", "data/episodes")
        self.compress = config.get("compress", True)
        self.collect_rewards = config.get("collect_rewards", True)
        self.collect_info = config.get("collect_info", False)

        # Episode data storage
        self.episode_data = None
        self.episode_count = 0

    def on_episode_start(self, coordinator: Any) -> None:
        """Initialize data collection when episode starts"""
        self.episode_data = {
            "game_mode": getattr(coordinator, "_current_game_mode", "unknown"),
            "team_ids": coordinator.agent_manager.get_team_ids(),
            "agent_types": {
                agent_id: agent.get_agent_type()
                for agent_id, agent in coordinator.agent_manager.agents.items()
            },
            "team_assignments": {
                agent_id: {"team_id": agent.team_id, "squad": agent.squad}
                for agent_id, agent in coordinator.agent_manager.agents.items()
            },
            "observations": [],
            "actions": [],
            "rewards": [] if self.collect_rewards else None,
            "info": [] if self.collect_info else None,
            "episode_length": 0,
            "terminated": False,
            "truncated": False,
            "timestamp": datetime.now().isoformat(),
        }

    def on_step(
        self, coordinator: Any, obs: dict, actions: dict, rewards: dict, info: dict
    ) -> None:
        """Collect data after each step"""
        if self.episode_data is not None:
            # Store observation
            self.episode_data["observations"].append(obs)

            # Store actions
            self.episode_data["actions"].append(actions)

            # Store rewards if enabled
            if self.collect_rewards and rewards is not None:
                self.episode_data["rewards"].append(rewards)

            # Store info if enabled
            if self.collect_info and info is not None:
                self.episode_data["info"].append(info)

    def on_episode_end(self, coordinator: Any) -> None:
        """Finalize and save episode data"""
        if self.episode_data is not None:
            # Update episode metadata
            self.episode_data["episode_length"] = (
                coordinator._current_step
                if hasattr(coordinator, "_current_step")
                else 0
            )
            self.episode_data["terminated"] = (
                coordinator._terminated
                if hasattr(coordinator, "_terminated")
                else False
            )
            self.episode_data["truncated"] = (
                coordinator._truncated if hasattr(coordinator, "_truncated") else False
            )

            # Save episode data
            self._save_episode_data()

            self.episode_count += 1

    def _save_episode_data(self):
        """Save episode data to file"""
        if not self.output_path:
            return

        # Create output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)

        # Generate filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"episode_{self.episode_count:04d}_{timestamp}.pkl"

        if self.compress:
            filename += ".gz"

        filepath = os.path.join(self.output_path, filename)

        # Save data
        try:
            if self.compress:
                with gzip.open(filepath, "wb") as f:
                    pickle.dump(self.episode_data, f)
            else:
                with open(filepath, "wb") as f:
                    pickle.dump(self.episode_data, f)

            print(f"Saved episode data to {filepath}")
        except Exception as e:
            print(f"Error saving episode data: {e}")

    def get_episode_data(self) -> dict:
        """Get the current episode data"""
        return self.episode_data

    def get_episode_count(self) -> int:
        """Get the number of episodes collected"""
        return self.episode_count
