"""
Playback Agent - replays saved episode data
"""

import torch
from .agents import Agent


class PlaybackAgent(Agent):
    """Agent that replays actions from saved episode data"""

    def __init__(self, episode_data: dict, team_id: int):
        self.episode_data = episode_data
        self.team_id = team_id
        self.current_step = 0

        # Extract team actions from episode data
        self.team_actions = episode_data.get("actions", {}).get(team_id, [])
        self.episode_length = len(self.team_actions)

        print(f"PlaybackAgent initialized for team {team_id}")
        print(f"Episode length: {self.episode_length} steps")

    def get_actions(
        self, obs_dict: dict, ship_ids: list[int]
    ) -> dict[int, torch.Tensor]:
        """Return recorded actions for current timestep"""
        actions = {}

        # Return zero actions if we're beyond the recorded episode
        if self.current_step >= self.episode_length:
            for ship_id in ship_ids:
                actions[ship_id] = torch.zeros(6, dtype=torch.float32)
            return actions

        # Get recorded actions for this timestep
        step_actions = self.team_actions[self.current_step]

        # Provide actions for requested ships, zero for missing ones
        for ship_id in ship_ids:
            if ship_id in step_actions:
                action = step_actions[ship_id]
                # Ensure action is a tensor
                if isinstance(action, torch.Tensor):
                    actions[ship_id] = action.clone()
                else:
                    actions[ship_id] = torch.tensor(action, dtype=torch.float32)
            else:
                actions[ship_id] = torch.zeros(6, dtype=torch.float32)

        # Advance timestep for next call
        self.current_step += 1

        return actions

    def get_agent_type(self) -> str:
        return "playback"

    def reset_playback(self):
        """Reset to beginning of episode"""
        self.current_step = 0

    def is_finished(self) -> bool:
        """Check if playback has reached the end"""
        return self.current_step >= self.episode_length

    def get_progress(self) -> tuple[int, int]:
        """Get current progress (current_step, total_steps)"""
        return self.current_step, self.episode_length


def create_playback_agent(episode_data: dict, team_id: int) -> PlaybackAgent:
    """Factory function for playback agents"""
    return PlaybackAgent(episode_data, team_id)
