from typing import Any
import numpy as np
import torch
from omegaconf import DictConfig
from wandb import agent

from agents.agents import create_agent

from .env.env import Environment
from .agent_manager import AgentManager
from .components import ComponentFactory


class GameCoordinator:
    """
    Central coordinator that orchestrates the game between environment and agents
    """

    def __init__(self, config: DictConfig):
        """
        Initialize the game coordinator

        Args:
            config: Configuration dictionary containing all setup
        """
        self.config = config

        self.env = Environment(**config.environment)

        self.agents = {
            create_agent(agent_id, **agent_config)
            for agent_id, agent_config in config.agents.items()
        }

        # Initialize components
        components_config = config.get("components", {})
        self.components = ComponentFactory.create_components_from_config(
            dict(components_config)
        )

        # Game state
        self.current_obs = None
        self.current_step = 0
        self.episode_count = 0
        self.should_terminate = False
        self.terminated = False  # Natural termination (victory/defeat)
        self.truncated = False  # Early termination (time limit, etc.)

        # Statistics
        self.stats = {
            "episodes_completed": 0,
            "total_steps": 0,
            "team_wins": {},
            "episode_rewards": {},
        }

    def reset(self) -> dict[str, np.ndarray]:
        """
        Reset the environment and all agents for a new episode

        Returns:
            Initial observation
        """
        # Reset environment
        obs, _ = self.env.reset()
        self.current_obs = obs

        # Reset agents
        self.agent_manager.reset()

        # Reset game state
        self.current_step = 0
        self.should_terminate = False
        self.terminated = False
        self.truncated = False

        # Call component hooks
        for component in self.components.values():
            component.on_episode_start(self)

        return self.current_obs

    def step(
        self,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[int, np.ndarray],
        dict[int, float],
        dict[str, Any],
        bool,
        bool,
    ]:
        """
        Execute one step of the game

        Returns:
            Tuple of (observation, actions, rewards, info, terminated, truncated)
        """
        if self.should_terminate:
            return self.current_obs, {}, {}, {}, self.terminated, self.truncated

        # Get actions from all active agents
        actions_by_ship, agent_info = self.agent_manager.get_actions(self.current_obs)

        # Convert actions to environment format
        env_actions = self._convert_actions_to_env_format(actions_by_ship)

        # Step environment
        self.current_obs, rewards, terminated, truncated, info = self.env.step(
            env_actions
        )

        # Convert rewards to ship-based format
        rewards_by_ship = self._convert_rewards_to_ship_format(rewards)

        # Update game state
        self.current_step += 1

        # Check victory conditions
        game_over, winner = self.agent_manager.check_victory(self.current_obs)
        if game_over:
            self.terminated = True
            self.should_terminate = True
            info["winner"] = winner
            info["victory"] = True

        # Call component hooks
        for component in self.components.values():
            component.on_step(
                self, self.current_obs, actions_by_ship, rewards_by_ship, info
            )

        # Check if components want to terminate
        if not self.should_terminate:
            for component in self.components.values():
                if hasattr(
                    component, "should_terminate"
                ) and component.should_terminate(self):
                    self.should_terminate = True
                    self.truncated = True
                    break

        return (
            self.current_obs,
            actions_by_ship,
            rewards_by_ship,
            info,
            self.terminated,
            self.truncated,
        )

    def run_episode(self, num_steps: int = 1000) -> dict[str, Any]:
        """
        Run a complete episode from start to finish

        Args:
            num_steps: Maximum number of steps to run

        Returns:
            Episode summary statistics
        """
        # Reset for new episode
        obs = self.reset()

        # Episode tracking
        episode_rewards: dict[int, float] = {}
        episode_steps = 0
        episode_data = {
            "observations": [obs],
            "actions": [],
            "rewards": [],
            "infos": [],
        }

        # Run episode loop
        while not self.should_terminate and episode_steps < num_steps:
            obs, actions, rewards, info, terminated, truncated = self.step()
            episode_steps += 1

            # Track rewards
            for ship_id, reward in rewards.items():
                if ship_id not in episode_rewards:
                    episode_rewards[ship_id] = 0
                episode_rewards[ship_id] += reward

            # Store step data (optional, based on config)
            if self.config.get("store_episode_data", False):
                episode_data["actions"].append(actions)
                episode_data["rewards"].append(rewards)
                episode_data["infos"].append(info)
                if not terminated and not truncated:
                    episode_data["observations"].append(obs)

            # Check if we've reached the step limit
            if episode_steps >= num_steps:
                self.truncated = True
                self.should_terminate = True

        # Finalize episode
        episode_summary = self._finalize_episode(
            episode_rewards, episode_steps, episode_data
        )

        # Call component hooks
        for component in self.components.values():
            component.on_episode_end(self)

        # Update statistics
        self._update_stats(episode_summary)

        return episode_summary

    def run(self, num_episodes: int = 1) -> list[dict[str, Any]]:
        """
        Run multiple episodes

        Args:
            num_episodes: Number of episodes to run

        Returns:
            List of episode summaries
        """
        episode_summaries = []

        for episode in range(num_episodes):
            print(f"Running episode {episode + 1}/{num_episodes}")
            summary = self.run_episode()
            episode_summaries.append(summary)
            self.episode_count += 1

        return episode_summaries

    def _convert_actions_to_env_format(
        self, actions_by_ship: dict[int, torch.Tensor]
    ) -> dict[int, torch.Tensor]:
        """Convert ship-based actions to environment format"""
        if not actions_by_ship:
            return {}

        # Convert to the format expected by Environment.step()
        env_actions = {}
        for ship_id, action in actions_by_ship.items():
            env_actions[ship_id] = action

        return env_actions

    def _convert_rewards_to_ship_format(self, rewards: dict) -> dict[int, float]:
        """Convert environment rewards to ship-based format"""
        rewards_by_ship: dict[int, float] = {}

        # If rewards is already a dictionary, just convert values to float
        if isinstance(rewards, dict):
            for ship_id, reward in rewards.items():
                rewards_by_ship[ship_id] = float(reward)
        else:
            # If it's an array, convert to dictionary
            for i, reward in enumerate(rewards):
                rewards_by_ship[i] = float(reward)

        return rewards_by_ship

    def _finalize_episode(
        self, episode_rewards: dict[int, float], episode_steps: int, episode_data: dict
    ) -> dict[str, Any]:
        """Create episode summary"""
        # Get team status
        team_status = self.agent_manager.get_team_status(self.current_obs)

        # Determine winner
        winner = None
        for team_id, status in team_status.items():
            if status["active"]:
                winner = team_id
                break

        # Aggregate rewards by team
        team_rewards: dict[int, float] = {}
        for ship_id, reward in episode_rewards.items():
            team = self.agent_manager.get_team_for_ship(ship_id)
            if team:
                team_id = team.team_id
                if team_id not in team_rewards:
                    team_rewards[team_id] = 0
                team_rewards[team_id] += reward

        return {
            "episode_id": self.episode_count + 1,
            "steps": episode_steps,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "winner": winner,
            "team_status": team_status,
            "team_rewards": team_rewards,
            "ship_rewards": episode_rewards,
            "data": (
                episode_data if self.config.get("store_episode_data", False) else None
            ),
        }

    def _update_stats(self, episode_summary: dict[str, Any]) -> None:
        """Update overall statistics"""
        self.stats["episodes_completed"] += 1
        self.stats["total_steps"] += episode_summary["steps"]

        # Track wins
        winner = episode_summary.get("winner")
        if winner is not None:
            if winner not in self.stats["team_wins"]:
                self.stats["team_wins"][winner] = 0
            self.stats["team_wins"][winner] += 1

        # Track rewards
        for team_id, reward in episode_summary["team_rewards"].items():
            if team_id not in self.stats["episode_rewards"]:
                self.stats["episode_rewards"][team_id] = []
            self.stats["episode_rewards"][team_id].append(reward)

    def get_stats(self) -> dict[str, Any]:
        """Get current statistics"""
        return self.stats.copy()

    def render(self, mode: str = "human") -> np.ndarray | None:
        """Render the current state"""
        # Check if renderer component is available
        if "renderer" in self.components:
            return self.components["renderer"].render(self.current_obs, mode)

        # Fall back to environment rendering
        return self.env.render(mode)

    def close(self) -> None:
        """Clean up resources"""
        # Close environment
        if hasattr(self.env, "close"):
            self.env.close()

        # Close components
        for component in self.components.values():
            if hasattr(component, "close"):
                component.close()
