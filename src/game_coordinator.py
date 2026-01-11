import torch
from omegaconf import DictConfig

from agents.agents import create_agent
from agents.tokenizer import observation_to_tokens
from env.env import Environment


class GameCoordinator:
    """
    Central coordinator that orchestrates the game between environment and agents.

    This class manages the game loop, agent interactions, and data collection
    for a single episode. It handles the mapping between team-based agents
    and individual ships.
    """

    def __init__(self, config: DictConfig, render_mode: str | None = None) -> None:
        """
        Initialize the game coordinator.

        Args:
            config: Configuration dictionary containing all setup parameters.
            render_mode: Override render mode (e.g., "none" for data collection).
        """
        self.config = config

        env_config = dict(config.environment)
        if render_mode is not None:
            env_config["render_mode"] = render_mode

        self.env = Environment(**env_config)

        self.agents = {
            agent_name: create_agent(
                agent_type=agent_config.agent_type,
                agent_config=agent_config.agent_config,
            )
            for agent_name, agent_config in config.agents.items()
        }

        self.obs_history: list[dict] = []
        # Initialize other attributes to satisfy type checkers
        self.all_tokens: dict[int, torch.Tensor] = {}
        self.all_actions: dict[int, list[torch.Tensor]] = {}
        self.all_action_masks: dict[int, list[float]] = {}
        self.all_rewards: dict[int, list[float]] = {}

    def reset(self, game_mode: str) -> None:
        """
        Reset environment for a new episode.

        Args:
            game_mode: The game mode to initialize (e.g., "1v1", "nvn").
        """
        obs, _ = self.env.reset(game_mode=game_mode)

        self.obs_history = [obs]
        self.all_tokens = {
            0: observation_to_tokens(obs=obs, perspective=0),
            1: observation_to_tokens(obs=obs, perspective=1),
        }
        self.all_actions = {
            ship_id: [] for ship_id in range(self.config.environment.max_ships)
        }
        self.all_action_masks = {
            ship_id: [] for ship_id in range(self.config.environment.max_ships)
        }
        self.all_rewards = {0: [], 1: []}

    def step(self) -> tuple[float, bool]:
        """
        Run the episode until termination.

        Returns:
            Tuple of (episode_sim_time, terminated).
        """
        terminated = False
        obs = self.obs_history[-1]

        team_names = self.config.collect.teams
        max_episode_length = self.config.collect.max_episode_length
        step_count = 0

        # Initialize RNG if not present (could be moved to init)
        if not hasattr(self, "_rng"):
            import numpy as np
            self._rng = np.random.default_rng()
        
        random_prob = self.config.collect.get("random_action_prob", 0.0)

        while not terminated:
            # 1. Determine which ships belong to which team
            teams = self._get_teams_from_obs(obs)

            # 2. Get actions from agents for their respective teams
            # We fetch all actions first, then override with random ones
            team_actions = {
                team_id: self.agents[team_names[team_id]](
                    self.obs_history[-1], ship_ids
                )
                for team_id, ship_ids in teams.items()
            }

            # 3. Flatten actions into a single dictionary and apply randomization
            actions = {}
            action_masks = {} # 1.0 for expert, 0.0 for random
            
            for team_id, ship_ids in teams.items():
                ship_actions = team_actions[team_id]
                
                for ship_id in ship_ids:
                    action = ship_actions[ship_id]
                    mask = 1.0
                    
                    if self._rng.random() < random_prob:
                        # Generate random action
                        # Power: 0-2, Turn: 0-6, Shoot: 0-1
                        # Action tensor shape (3,) floats
                        rand_power = float(self._rng.integers(0, 3))
                        rand_turn = float(self._rng.integers(0, 7))
                        rand_shoot = float(self._rng.integers(0, 2))
                        action = torch.tensor([rand_power, rand_turn, rand_shoot], dtype=torch.float32)
                        mask = 0.0
                    
                    actions[ship_id] = action
                    action_masks[ship_id] = mask

            # 4. Record actions and masks for history
            for ship_id, action in actions.items():
                self.all_actions[ship_id].append(action)
                self.all_action_masks[ship_id].append(action_masks[ship_id])

            # 5. Step the environment
            obs, rewards, terminated, _, info = self.env.step(actions=actions)

            # 6. Record rewards and observations
            for team_id, reward in rewards.items():
                self.all_rewards[team_id].append(reward)

            self.obs_history.append(obs)
            self._update_tokens(obs)

            step_count += 1
            if step_count >= max_episode_length:
                terminated = True

        episode_sim_time = info.get("current_time", 0.0)
        return episode_sim_time, terminated

    def _update_tokens(self, obs: dict) -> None:
        """Helper to update token history."""
        self.all_tokens[0] = torch.cat(
            [self.all_tokens[0], observation_to_tokens(obs=obs, perspective=0)],
            dim=0,
        )
        self.all_tokens[1] = torch.cat(
            [self.all_tokens[1], observation_to_tokens(obs=obs, perspective=1)],
            dim=0,
        )

    def _get_teams_from_obs(self, obs: dict) -> dict[int, list[int]]:
        """
        Extract team assignments from observation.

        Args:
            obs: Observation dictionary.

        Returns:
            Dictionary mapping team_id to list of ship_ids.
        """
        teams: dict[int, list[int]] = {}
        for ship_id in range(len(obs["ship_id"])):
            if obs["alive"][ship_id].item():
                team_id = obs["team_id"][ship_id].item()
                if team_id not in teams:
                    teams[team_id] = []
                teams[team_id].append(ship_id)
        return teams

    def close(self) -> None:
        """Clean up resources."""
        if hasattr(self.env, "close"):
            self.env.close()
