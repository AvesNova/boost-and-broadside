"""
Unified RL Training Wrapper - integrates with the unified agent system
"""

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from collections import deque

from .env import Environment
from .agents import (
    Agent,
    create_scripted_agent,
    create_rl_agent,
    create_selfplay_agent,
)
from .team_transformer_model import TeamController


class UnifiedRLWrapper(gym.Env):
    """
    RL training wrapper that uses the unified agent system.
    Supports all opponent types: scripted, self-play, mixed.
    """

    def __init__(
        self,
        env_config: dict | None = None,
        learning_team_id: int = 0,
        team_assignments: dict[int, list[int]] | None = None,
        opponent_config: dict | None = None,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        super().__init__()
        self.rng = rng

        # Environment setup
        env_config = env_config or {}
        self.base_env = Environment(**env_config)

        # Team configuration
        self.learning_team_id = learning_team_id
        self.team_assignments = team_assignments or {0: [0, 1], 1: [2, 3]}
        self.controlled_ships = self.team_assignments[learning_team_id]
        self.team_controller = TeamController(self.team_assignments)

        # Opponent configuration
        opponent_config = opponent_config or {}
        self.opponent_type = opponent_config.get("type", "scripted")
        self.scripted_mix_ratio = opponent_config.get("scripted_mix_ratio", 0.3)

        # Initialize opponents
        self._setup_opponents(opponent_config)

        # Observation and action spaces
        self._setup_spaces()

        # Episode tracking
        self.episode_count = 0
        self.wins = 0
        self.losses = 0
        self.current_episode_type = "scripted"
        self.current_opponent: Agent | None = None

        # Self-play memory management
        self.steps_since_opponent_update = 0
        self.opponent_update_freq = opponent_config.get("opponent_update_freq", 10000)

    def _setup_opponents(self, config: dict):
        """Setup different opponent types"""
        world_size = self.base_env.world_size

        # Scripted opponent
        scripted_config = config.get(
            "scripted_config",
            {
                "max_shooting_range": 500.0,
                "angle_threshold": 5.0,
                "bullet_speed": 500.0,
                "target_radius": 10.0,
                "radius_multiplier": 1.5,
            },
        )
        self.scripted_opponent = create_scripted_agent(world_size, scripted_config)

        # Self-play opponent
        memory_size = config.get("selfplay_memory_size", 50)
        self.selfplay_opponent = create_selfplay_agent(
            self.team_controller, memory_size
        )

        # Start with scripted opponent
        self.current_opponent = self.scripted_opponent

    def _setup_spaces(self):
        """Setup observation and action spaces for SB3"""
        max_ships = self.base_env.max_ships
        token_dim = 10

        # Observation space: token matrix for all ships
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(max_ships, token_dim), dtype=np.float32
        )

        # Action space: actions for our controlled ships only
        num_controlled_ships = len(self.controlled_ships)
        num_actions_per_ship = 6

        self.action_space = spaces.MultiBinary(
            num_controlled_ships * num_actions_per_ship
        )

    def _choose_episode_type(self) -> str:
        """Choose opponent type for this episode"""
        if self.opponent_type == "scripted":
            return "scripted"
        elif self.opponent_type == "self_play":
            return "self_play"
        elif self.opponent_type == "mixed":
            return (
                "scripted"
                if self.rng.random() < self.scripted_mix_ratio
                else "self_play"
            )
        else:
            return "scripted"

    def _get_opponent_actions(self, obs_dict: dict) -> dict[int, torch.Tensor]:
        """Get opponent actions using current opponent agent"""
        opponent_team_ids = [
            tid for tid in self.team_assignments.keys() if tid != self.learning_team_id
        ]

        all_opponent_actions = {}
        for team_id in opponent_team_ids:
            ship_ids = self.team_assignments[team_id]
            team_actions = self.current_opponent.get_actions(obs_dict, ship_ids)
            all_opponent_actions.update(team_actions)

        return all_opponent_actions

    def _unflatten_actions(
        self, flattened_actions: np.ndarray
    ) -> dict[int, torch.Tensor]:
        """Convert flattened SB3 actions back to ship action dict"""
        actions = {}
        # Ensure flattened_actions is not empty
        if flattened_actions.size == 0:
            # Return default actions (all zeros) for controlled ships
            for ship_id in sorted(self.controlled_ships):
                actions[ship_id] = torch.zeros(6, dtype=torch.float32)
            return actions

        for i, ship_id in enumerate(sorted(self.controlled_ships)):
            start_idx = i * 6
            end_idx = start_idx + 6
            # Ensure we don't go out of bounds
            if end_idx <= flattened_actions.size:
                ship_action = flattened_actions[start_idx:end_idx]
                actions[ship_id] = torch.from_numpy(ship_action).float()
            else:
                # Return default action if not enough values
                actions[ship_id] = torch.zeros(6, dtype=torch.float32)
        return actions

    def reset(self, **kwargs):
        """Reset environment and choose opponent type"""
        self.episode_count += 1
        self.current_episode_type = self._choose_episode_type()

        # Set current opponent
        if self.current_episode_type == "scripted":
            self.current_opponent = self.scripted_opponent
        elif self.current_episode_type == "self_play":
            self.current_opponent = self.selfplay_opponent
        else:
            self.current_opponent = self.scripted_opponent

        # Reset base environment (nvn gives variety across game modes)
        obs_dict, info = self.base_env.reset(game_mode="nvn")

        # Update team assignments from actual environment state
        self._update_team_assignments_from_env()

        observation = obs_dict["tokens"].numpy()

        # Add episode info
        info.update(
            {
                "episode_type": self.current_episode_type,
                "episode_count": self.episode_count,
                "win_rate": self.get_win_rate(),
                "opponent_type": self.current_opponent.get_agent_type(),
            }
        )

        return observation, info

    def _update_team_assignments_from_env(self):
        """Update team assignments based on actual environment state"""
        if not self.base_env.state:
            return

        current_state = self.base_env.state[-1]
        actual_teams = {}

        for ship_id, ship in current_state.ships.items():
            team_id = ship.team_id
            if team_id not in actual_teams:
                actual_teams[team_id] = []
            actual_teams[team_id].append(ship_id)

        # Update team assignments and controlled ships
        self.team_assignments = actual_teams
        if self.learning_team_id in actual_teams:
            self.controlled_ships = actual_teams[self.learning_team_id]

        # Update team controller
        self.team_controller = TeamController(self.team_assignments)

    def step(self, action):
        """Step environment with unified agent system"""
        # Get our team's actions
        team_actions = self._unflatten_actions(action)

        # Get opponent actions
        obs_dict = self.base_env.get_observation()
        opponent_actions = self._get_opponent_actions(obs_dict)

        # Combine and step
        all_actions = {**team_actions, **opponent_actions}
        obs_dict, _, terminated, truncated, info = self.base_env.step(all_actions)

        # Calculate team reward
        current_state = self.base_env.state[-1]
        team_reward = self.base_env._calculate_team_reward(
            current_state, self.learning_team_id, episode_ended=terminated
        )

        # Track wins/losses
        if terminated:
            outcome_reward = self.base_env._calculate_outcome_rewards(
                current_state, self.learning_team_id
            )

            if outcome_reward > 0.5:
                self.wins += 1
            elif outcome_reward < -0.5:
                self.losses += 1

        # Update self-play opponent periodically
        self.steps_since_opponent_update += 1
        if (
            self.steps_since_opponent_update >= self.opponent_update_freq
            and self.current_episode_type == "self_play"
        ):
            self._maybe_update_selfplay_opponent()
            self.steps_since_opponent_update = 0

        # Prepare return values
        observation = obs_dict["tokens"].numpy()

        info.update(
            {
                "episode_type": self.current_episode_type,
                "team_reward": team_reward,
                "opponent_type": self.current_opponent.get_agent_type(),
                "controlled_ships_alive": sum(
                    1
                    for ship_id in self.controlled_ships
                    if ship_id < obs_dict["alive"].shape[0]
                    and obs_dict["alive"][ship_id, 0].item() > 0
                ),
            }
        )

        return observation, team_reward, terminated, truncated, info

    def _maybe_update_selfplay_opponent(self):
        """Update self-play opponent if we have models in memory"""
        if len(self.selfplay_opponent.model_memory) > 0:
            # This would need the model class and config - implement based on your setup
            pass

    def add_model_to_selfplay_memory(self, model):
        """Add a model to self-play memory"""
        self.selfplay_opponent.add_model_to_memory(model)
        print(
            f"Added model to self-play memory (size: {len(self.selfplay_opponent.model_memory)})"
        )

    def update_selfplay_opponent(self, model_class, model_config: dict):
        """Update the self-play opponent"""
        self.selfplay_opponent.update_opponent(model_class, model_config)
        print("Updated self-play opponent")

    def get_win_rate(self) -> float:
        """Get current win rate"""
        if self.episode_count <= 1:
            return 0.5
        total_games = self.wins + self.losses
        if total_games == 0:
            return 0.5
        return self.wins / total_games

    def close(self):
        """Close environment"""
        self.base_env.close()


def create_unified_rl_env(
    env_config: dict | None = None,
    learning_team_id: int = 0,
    team_assignments: dict[int, list[int]] | None = None,
    opponent_config: dict | None = None,
) -> UnifiedRLWrapper:
    """Factory function to create unified RL environment"""

    # Default configurations
    default_env_config = {
        "world_size": (1200, 800),
        "max_ships": 8,
        "agent_dt": 0.04,
        "physics_dt": 0.02,
    }

    default_opponent_config = {
        "type": "mixed",  # scripted, self_play, mixed
        "scripted_mix_ratio": 0.3,
        "selfplay_memory_size": 50,
        "opponent_update_freq": 10000,
        "scripted_config": {
            "max_shooting_range": 500.0,
            "angle_threshold": 5.0,
            "bullet_speed": 500.0,
            "target_radius": 10.0,
            "radius_multiplier": 1.5,
        },
    }

    default_team_assignments = {0: [0, 1], 1: [2, 3]}  # Will be updated by nvn mode

    # Merge with user configs
    if env_config:
        # Filter out unsupported parameters
        filtered_env_config = {
            k: v for k, v in env_config.items() if k != "max_episode_steps"
        }
        default_env_config.update(filtered_env_config)
    if opponent_config:
        default_opponent_config.update(opponent_config)
    if team_assignments:
        default_team_assignments.update(team_assignments)

    return UnifiedRLWrapper(
        env_config=default_env_config,
        learning_team_id=learning_team_id,
        team_assignments=default_team_assignments,
        opponent_config=default_opponent_config,
    )
