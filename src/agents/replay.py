import torch
import torch.nn as nn


class ReplayAgent(nn.Module):
    """Agent that controls all ships from recorded episode data"""

    def __init__(
        self,
        agent_id: str,
        team_id: int,
        squad: list[int],
        replay_data: dict | None = None,
    ):
        """
        Initialize replay agent

        Args:
            agent_id: Unique identifier for this agent
            team_id: Which team this agent belongs to (not used for replay)
            squad: List of ship IDs this agent controls (should be all ships)
            replay_data: Dictionary containing recorded episode data
        """
        super().__init__(agent_id, team_id, squad)

        self.replay_data = replay_data
        self.current_step = 0
        self.episode_actions = None
        self.episode_length = 0

        # Load actions from replay data
        if replay_data and "actions" in replay_data:
            self._load_actions()

    def _load_actions(self):
        """Load actions from replay data"""
        # Replay data structure:
        # {
        #     "actions": {
        #         "team_id": [step_0_actions, step_1_actions, ...],
        #         ...
        #     },
        #     "episode_length": N
        # }

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

    def get_actions(self, obs_dict: dict[str, torch.Tensor]) -> dict[int, torch.Tensor]:
        """
        Get actions from recorded episode data

        Args:
            obs_dict: Observation dictionary from environment

        Returns:
            Dictionary mapping ship_id to action tensor
        """
        actions: dict[int, torch.Tensor] = {}

        if self.current_step < self.episode_length:
            step_actions = self.episode_actions[self.current_step]

            # Convert actions to tensors and filter for our squad
            for ship_id, action in step_actions.items():
                if ship_id in self.squad:
                    if isinstance(action, torch.Tensor):
                        actions[ship_id] = action
                    else:
                        # Convert to tensor if needed
                        actions[ship_id] = torch.tensor(action, dtype=torch.float32)
        else:
            # No more recorded actions, return zero actions
            for ship_id in self.squad:
                actions[ship_id] = torch.zeros(6, dtype=torch.float32)

        # Increment step for next call
        self.current_step += 1

        return actions

    def get_agent_type(self) -> str:
        """Return agent type for logging/debugging"""
        return "replay"

    def reset(self) -> None:
        """Reset agent state for new episode"""
        self.current_step = 0

    def is_finished(self) -> bool:
        """Check if replay has finished"""
        return self.current_step >= self.episode_length

    def get_progress(self) -> tuple[int, int]:
        """Get current progress through replay"""
        return (self.current_step, self.episode_length)

    def set_replay_data(self, replay_data: dict) -> None:
        """Set new replay data"""
        self.replay_data = replay_data
        self.current_step = 0
        self._load_actions()
