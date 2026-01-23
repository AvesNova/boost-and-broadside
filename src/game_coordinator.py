import torch
from omegaconf import DictConfig
from dataclasses import dataclass
import numpy as np

from agents.agents import create_agent
from agents.tokenizer import observation_to_tokens
from env.env import Environment


@dataclass
class StickyActionState:
    """Tracks the state of sticky actions for a single ship."""
    
    # Current active action for each head
    power: int = 0
    turn: int = 0
    shoot: int = 0
    
    # Remaining steps for the current sticky choice
    power_steps: int = 0
    turn_steps: int = 0
    shoot_steps: int = 0
    
    # Whether the current period is "expert" (True) or "random" (False)
    power_expert: bool = True
    turn_expert: bool = True
    shoot_expert: bool = True


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
        
        # Inject number of teams from collect config
        env_config["num_teams"] = len(config.collect.teams)

        self.env = Environment(**env_config)

        self.agents = {}
        for agent_name, agent_config_node in config.agents.items():
            # Convert config node to dict to allow modification
            agent_cfg_dict = dict(agent_config_node.agent_config)

            # Inject world_size if not present
            if "world_size" not in agent_cfg_dict:
                # OmegaConf list to tuple for compatibility
                agent_cfg_dict["world_size"] = tuple(self.config.environment.world_size)

            # Create agent
            self.agents[agent_name] = create_agent(
                agent_type=agent_config_node.agent_type,
                agent_config=agent_cfg_dict,
            )

        self.obs_history: list[dict] = []
        # Initialize other attributes to satisfy type checkers
        self.all_tokens: dict[int, torch.Tensor] = {}
        self.all_actions: dict[int, list[torch.Tensor]] = {}
        self.all_expert_actions: dict[int, list[torch.Tensor]] = {} # New: Save expert actions
        self.all_action_masks: dict[int, list[float]] = {}
        self.all_rewards: dict[int, list[float]] = {}
        
        # Sticky Action States
        self.sticky_states: dict[int, StickyActionState] = {}
        self._rng = np.random.default_rng()

    def reset(
        self, game_mode: str, team_skills: dict[int, float] | None = None
    ) -> None:
        """
        Reset environment for a new episode.

        Args:
            game_mode: The game mode to initialize (e.g., "1v1", "nvn").
            team_skills: Dictionary mapping team_id to skill level (0.0 to 1.0).
                         1.0 = Expert, 0.0 = Random. dict[int, float].
                         If None, defaults to 1.0 (Expert).
        """
        obs, _ = self.env.reset(game_mode=game_mode)

        num_teams = len(self.config.collect.teams)
        if team_skills is None:
            self.team_skills = {i: 1.0 for i in range(num_teams)}
        else:
            self.team_skills = team_skills

        self.obs_history = [obs]
        # Initialize token history for each team
        # num_teams already defined above
        self.all_tokens = {
            team_id: observation_to_tokens(
                obs=obs,
                perspective=team_id,
                world_size=tuple(self.config.environment.world_size),
            )
            for team_id in range(num_teams)
        }
        self.all_actions = {
            ship_id: [] for ship_id in range(self.config.environment.max_ships)
        }
        self.all_expert_actions = {
            ship_id: [] for ship_id in range(self.config.environment.max_ships)
        }
        self.all_action_masks = {
            ship_id: [] for ship_id in range(self.config.environment.max_ships)
        }
        self.all_rewards = {team_id: [] for team_id in range(num_teams)}
        
        # Reset sticky states
        self.sticky_states = {
            ship_id: StickyActionState() 
            for ship_id in range(self.config.environment.max_ships)
        }

    def _sample_sticky_duration(self) -> int:
        """
        Sample a duration for a sticky action from a Beta distribution.
        Target mean is 16 steps.
        Beta(2, 6) * 64 roughly gives a nice distribution with mean ~16.
        """
        # Beta(alpha=2, beta=6) has mean 0.25.
        # 0.25 * 64 = 16.
        val = self._rng.beta(2, 6)
        steps = int(val * 64)
        return max(1, steps)

    def _update_sticky_state(
        self, state: StickyActionState, expert_action: torch.Tensor, skill_level: float
    ) -> torch.Tensor:
        """
        Update sticky state and determine the action to take.
        
        Args:
            state: StickyActionState for the ship.
            expert_action: (3,) tensor [Power, Turn, Shoot] from expert.
            skill_level: Probability of choosing 'Expert' profile.
            
        Returns:
            Taken action as (3,) tensor.
        """
        # Expert action values
        exp_p = int(expert_action[0].item())
        exp_t = int(expert_action[1].item())
        exp_s = int(expert_action[2].item())
        
        # --- POWER ---
        if state.power_steps <= 0:
            # Resample profile
            state.power_steps = self._sample_sticky_duration()
            if self._rng.random() < skill_level:
                state.power_expert = True
            else:
                state.power_expert = False
                # Sample random action to stick to
                state.power = self._rng.integers(0, 3) # 0, 1, 2
        
        state.power_steps -= 1
        
        # Determine output
        final_p = exp_p if state.power_expert else state.power
        
        # --- TURN ---
        if state.turn_steps <= 0:
            state.turn_steps = self._sample_sticky_duration()
            if self._rng.random() < skill_level:
                state.turn_expert = True
            else:
                state.turn_expert = False
                state.turn = self._rng.integers(0, 7)
                
        state.turn_steps -= 1
        final_t = exp_t if state.turn_expert else state.turn
        
        # --- SHOOT ---
        if state.shoot_steps <= 0:
            state.shoot_steps = self._sample_sticky_duration()
            if self._rng.random() < skill_level:
                state.shoot_expert = True
            else:
                state.shoot_expert = False
                state.shoot = self._rng.integers(0, 2)
                
        state.shoot_steps -= 1
        final_s = exp_s if state.shoot_expert else state.shoot
        
        return torch.tensor([final_p, final_t, final_s], dtype=torch.uint8)

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

        while not terminated:
            # 1. Determine which ships belong to which team
            teams = self._get_teams_from_obs(obs)

            # 2. Get actions from agents for their respective teams
            # Expert actions
            team_actions = {
                team_id: self.agents[team_names[team_id]](
                    self.obs_history[-1], ship_ids
                )
                for team_id, ship_ids in teams.items()
            }

            # 3. Apply Sticky/Random Logic
            actions = {}
            expert_actions_dict = {}
            action_masks = {}

            for team_id, ship_ids in teams.items():
                ship_actions_expert = team_actions[team_id]
                skill_level = self.team_skills.get(team_id, 1.0)
                
                for ship_id in ship_ids:
                    expert_tensor = ship_actions_expert[ship_id]
                    expert_actions_dict[ship_id] = expert_tensor.to(dtype=torch.uint8)
                    
                    # Update Sticky State
                    sticky_state = self.sticky_states[ship_id]
                    taken_tensor = self._update_sticky_state(
                        sticky_state, expert_tensor, skill_level
                    )
                    
                    actions[ship_id] = taken_tensor
                    
                    # Mask isn't really used same way anymore, but we can track expert usage ratio?
                    # Let's just store 1.0 if Fully Expert this step?
                    # Or maybe just average of the 3 heads.
                    is_expert = float(sticky_state.power_expert and sticky_state.turn_expert and sticky_state.shoot_expert)
                    action_masks[ship_id] = is_expert

            # 4. Record actions and masks for history
            for ship_id, action in actions.items():
                self.all_actions[ship_id].append(action)
                self.all_expert_actions[ship_id].append(expert_actions_dict[ship_id])
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
        # Update tokens for all tracked teams
        for team_id in self.all_tokens.keys():
            self.all_tokens[team_id] = torch.cat(
                [
                    self.all_tokens[team_id],
                    observation_to_tokens(
                        obs=obs,
                        perspective=team_id,
                        world_size=tuple(self.config.environment.world_size),
                    ),
                ],
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
