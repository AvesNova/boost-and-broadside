"""
Replay agent for playing back recorded episodes.

Allows replaying previously recorded game episodes by feeding stored actions
back into the environment.
"""

from typing import Any
import torch
import torch.nn as nn


class ReplayAgent(nn.Module):
    """
    Agent that replays actions from recorded episode data.

    This agent plays back pre-recorded actions, useful for debugging,
    visualization, or analysis of past games.
    """

    def __init__(
        self,
        agent_id: str,
        team_id: int,
        squad: list[int],
        replay_data: dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        """
        Initialize replay agent.

        Args:
            agent_id: Unique identifier for this agent.
            team_id: Which team this agent belongs to (not used for replay).
            squad: List of ship IDs this agent controls.
            replay_data: Dictionary containing recorded episode data.
            **kwargs: Ignored additional arguments.
        """
        super().__init__()
        self.agent_id = agent_id
        self.team_id = team_id
        self.squad = squad

        self.replay_data = replay_data
        self.current_step = 0
        self.episode_actions: list[dict[int, torch.Tensor]] = []
        self.episode_length = 0

        # Load actions from replay data
        if replay_data and "actions" in replay_data:
            self._load_actions()

    def _load_actions(self) -> None:
        """
        Load actions from replay data.

        Replay data structure:
        {
            "actions": {
                "team_id": [step_0_actions, step_1_actions, ...],
                ...
            },
            "episode_length": N
        }
        """
        # Combine all team actions into a single sequence
        self.episode_actions = []

        if "actions" in self.replay_data:
            # Find the maximum number of steps
            max_steps = 0
            for team_actions in self.replay_data["actions"].values():
                max_steps = max(max_steps, len(team_actions))

            # Create combined actions for each step
            for step in range(max_steps):
                step_actions = {}

                # Collect actions from all teams for this step
                for team_id, team_actions in self.replay_data["actions"].items():
                    if step < len(team_actions):
                        # team_actions[step] is a dict mapping ship_id to action
                        step_actions.update(team_actions[step])

                self.episode_actions.append(step_actions)

        self.episode_length = len(self.episode_actions)

    def forward(
        self, obs_dict: dict[str, torch.Tensor], ship_ids: list[int]
    ) -> dict[int, torch.Tensor]:
        """
        Forward pass for the agent.

        Args:
            obs_dict: Observation dictionary from environment.
            ship_ids: List of ship IDs to control.

        Returns:
            Dictionary mapping ship_id to action tensor.
        """
        actions: dict[int, torch.Tensor] = {}

        if self.current_step < self.episode_length:
            step_actions = self.episode_actions[self.current_step]

            # Convert actions to tensors and filter for our squad
            for ship_id, action in step_actions.items():
                if ship_id in ship_ids:
                    if isinstance(action, torch.Tensor):
                        actions[ship_id] = action
                    else:
                        # Convert to tensor if needed
                        actions[ship_id] = torch.tensor(action, dtype=torch.float32)
        else:
            # No more recorded actions, return zero actions
            for ship_id in ship_ids:
                actions[ship_id] = torch.zeros(6, dtype=torch.float32)

        # Increment step for next call
        self.current_step += 1

        return actions

    def reset(self) -> None:
        """Reset agent state for new episode."""
        self.current_step = 0

    def is_finished(self) -> bool:
        """Check if replay has finished."""
        return self.current_step >= self.episode_length

    def get_progress(self) -> tuple[int, int]:
        """
        Get current progress through replay.

        Returns:
            Tuple of (current_step, total_steps).
        """
        return (self.current_step, self.episode_length)

    def set_replay_data(self, replay_data: dict[str, Any]) -> None:
        """
        Set new replay data.

        Args:
            replay_data: New replay data dictionary.
        """
        self.replay_data = replay_data
        self.current_step = 0
        self._load_actions()
