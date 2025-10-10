"""
Parallel RL Training Wrapper - manages multiple environments for parallel experience collection
"""

import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces
from collections import deque
from typing import Dict, List, Any, Optional, Tuple
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

from rl_wrapper import UnifiedRLWrapper
from .parallel_utils import WorkerProcess, WorkerPool


class ParallelRLWorker(WorkerProcess):
    """
    Worker process for parallel RL environment simulation
    """

    def __init__(self, worker_id: int, config: Dict[str, Any], **kwargs):
        super().__init__(worker_id, config, **kwargs)
        self.env_config = kwargs.get("env_config", {})
        self.opponent_config = kwargs.get("opponent_config", {})
        self.learning_team_id = kwargs.get("learning_team_id", 0)
        self.team_assignments = kwargs.get("team_assignments", {0: [0, 1], 1: [2, 3]})
        self.timesteps_to_collect = kwargs.get("timesteps_to_collect", 10000)
        self.checkpoint_freq = kwargs.get("checkpoint_freq", 2500)
        self.output_dir = kwargs.get("output_dir", f"rl_worker_{worker_id}")

    def execute(self):
        """Execute RL experience collection"""
        self.log_info(
            f"Starting RL experience collection for {self.timesteps_to_collect} timesteps"
        )

        # Setup output directory
        from pathlib import Path
        import pickle

        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Create environment
        env = UnifiedRLWrapper(
            env_config=self.env_config,
            learning_team_id=self.learning_team_id,
            team_assignments=self.team_assignments,
            opponent_config=self.opponent_config,
        )

        # Collect experiences
        all_experiences = []
        timesteps_collected = 0
        episode_count = 0

        try:
            while (
                timesteps_collected < self.timesteps_to_collect
                and not self._should_stop
            ):
                obs, info = env.reset()
                done = False
                episode_experiences = []
                episode_timesteps = 0

                while not done and episode_timesteps < 1000:  # Max episode length
                    # Random action for now (would be replaced by model actions)
                    action = env.action_space.sample()

                    next_obs, reward, terminated, truncated, info = env.step(action)
                    done = terminated or truncated

                    # Store experience
                    experience = {
                        "obs": obs,
                        "action": action,
                        "reward": reward,
                        "next_obs": next_obs,
                        "done": done,
                        "info": info,
                    }
                    episode_experiences.append(experience)

                    obs = next_obs
                    episode_timesteps += 1
                    timesteps_collected += 1

                all_experiences.extend(episode_experiences)
                episode_count += 1

                # Periodic checkpointing
                if (
                    episode_count % 10 == 0
                    or timesteps_collected >= self.timesteps_to_collect
                ):
                    checkpoint_path = (
                        output_path / f"checkpoint_timesteps_{timesteps_collected}.pkl"
                    )
                    with open(checkpoint_path, "wb") as f:
                        pickle.dump(
                            {
                                "experiences": all_experiences[
                                    -min(1000, len(all_experiences)) :
                                ],
                                "timesteps_collected": timesteps_collected,
                                "episode_count": episode_count,
                                "worker_id": self.worker_id,
                            },
                            f,
                        )

                    self.log_info(
                        f"Checkpoint: {timesteps_collected} timesteps, {episode_count} episodes"
                    )

                # Check if we should stop
                if self._should_stop:
                    break

        except Exception as e:
            self.log_error(f"Error during RL collection: {e}")
            return 1

        finally:
            env.close()

        # Save final data
        final_path = output_path / "final_experiences.pkl"
        with open(final_path, "wb") as f:
            pickle.dump(
                {
                    "experiences": all_experiences,
                    "timesteps_collected": timesteps_collected,
                    "episode_count": episode_count,
                    "worker_id": self.worker_id,
                },
                f,
            )

        self.log_info(
            f"Completed collection: {timesteps_collected} timesteps, {episode_count} episodes"
        )

        return {
            "worker_id": self.worker_id,
            "timesteps_collected": timesteps_collected,
            "episode_count": episode_count,
            "output_dir": str(output_path),
        }


class ParallelRLWrapper:
    """
    Wrapper for managing multiple RL environments in parallel
    """

    def __init__(
        self,
        env_config: Dict[str, Any],
        num_envs: int = 4,
        learning_team_id: int = 0,
        team_assignments: Optional[Dict[int, List[int]]] = None,
        opponent_config: Optional[Dict[str, Any]] = None,
    ):
        self.env_config = env_config
        self.num_envs = num_envs
        self.learning_team_id = learning_team_id
        self.team_assignments = team_assignments or {0: [0, 1], 1: [2, 3]}
        self.opponent_config = opponent_config or {}

        # Create individual environments
        self.envs = []
        for i in range(num_envs):
            env = UnifiedRLWrapper(
                env_config=env_config,
                learning_team_id=learning_team_id,
                team_assignments=team_assignments,
                opponent_config=opponent_config,
            )
            self.envs.append(env)

        # Setup spaces (same as individual environments)
        self.observation_space = self.envs[0].observation_space
        self.action_space = self.envs[0].action_space

        # Tracking
        self.episode_count = 0
        self.total_timesteps = 0

    def reset(self) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset all environments"""
        observations = []
        infos = []

        for env in self.envs:
            obs, info = env.reset()
            observations.append(obs)
            infos.append(info)

        self.episode_count += self.num_envs

        return np.array(observations), {"infos": infos}

    def step(
        self, actions: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        """Step all environments"""
        observations = []
        rewards = []
        terminateds = []
        truncateds = []
        infos = []

        for env, action in zip(self.envs, actions):
            obs, reward, terminated, truncated, info = env.step(action)
            observations.append(obs)
            rewards.append(reward)
            terminateds.append(terminated)
            truncateds.append(truncated)
            infos.append(info)

        self.total_timesteps += self.num_envs

        return (
            np.array(observations),
            np.array(rewards),
            np.array(terminateds),
            np.array(truncateds),
            {"infos": infos},
        )

    def collect_experiences(
        self,
        model,
        timesteps: int,
        checkpoint_freq: int = 25000,
        output_dir: str = "rl_experiences",
    ) -> Dict[str, Any]:
        """Collect experiences using the provided model"""
        from pathlib import Path
        import pickle

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        all_experiences = []
        timesteps_collected = 0
        episode_count = 0

        obs, infos = self.reset()

        while timesteps_collected < timesteps:
            # Get actions from model
            if hasattr(model, "predict"):
                actions, _ = model.predict(obs, deterministic=False)
            else:
                # Fallback to random actions
                actions = np.array(
                    [self.action_space.sample() for _ in range(self.num_envs)]
                )

            # Step environments
            next_obs, rewards, terminateds, truncateds, step_infos = self.step(actions)
            dones = terminateds | truncateds

            # Store experiences
            for i in range(self.num_envs):
                experience = {
                    "obs": obs[i],
                    "action": actions[i],
                    "reward": rewards[i],
                    "next_obs": next_obs[i],
                    "done": dones[i],
                    "info": step_infos["infos"][i],
                }
                all_experiences.append(experience)

            obs = next_obs
            timesteps_collected += self.num_envs

            # Reset environments that are done
            for i, done in enumerate(dones):
                if done:
                    obs[i], _ = self.envs[i].reset()
                    episode_count += 1

            # Periodic checkpointing
            if timesteps_collected % checkpoint_freq == 0:
                checkpoint_path = (
                    output_path / f"checkpoint_timesteps_{timesteps_collected}.pkl"
                )
                with open(checkpoint_path, "wb") as f:
                    pickle.dump(
                        {
                            "experiences": all_experiences[
                                -min(10000, len(all_experiences)) :
                            ],
                            "timesteps_collected": timesteps_collected,
                            "episode_count": episode_count,
                        },
                        f,
                    )

                print(
                    f"Checkpoint: {timesteps_collected} timesteps, {episode_count} episodes"
                )

        # Save final experiences
        final_path = output_path / "final_experiences.pkl"
        with open(final_path, "wb") as f:
            pickle.dump(
                {
                    "experiences": all_experiences,
                    "timesteps_collected": timesteps_collected,
                    "episode_count": episode_count,
                },
                f,
            )

        return {
            "experiences": all_experiences,
            "timesteps_collected": timesteps_collected,
            "episode_count": episode_count,
            "output_path": str(output_path),
        }

    def close(self):
        """Close all environments"""
        for env in self.envs:
            env.close()


def create_parallel_rl_env(
    env_config: Dict[str, Any],
    num_envs: int = 4,
    learning_team_id: int = 0,
    team_assignments: Optional[Dict[int, List[int]]] = None,
    opponent_config: Optional[Dict[str, Any]] = None,
) -> ParallelRLWrapper:
    """Factory function to create parallel RL environment"""

    default_env_config = {
        "world_size": (1200, 800),
        "max_ships": 8,
        "agent_dt": 0.04,
        "physics_dt": 0.02,
    }

    default_opponent_config = {
        "type": "mixed",
        "scripted_mix_ratio": 0.3,
        "selfplay_memory_size": 50,
        "opponent_update_freq": 10000,
    }

    default_team_assignments = {0: [0, 1], 1: [2, 3]}

    # Merge with defaults
    if env_config:
        default_env_config.update(env_config)
    if opponent_config:
        default_opponent_config.update(opponent_config)
    if team_assignments:
        default_team_assignments.update(team_assignments)

    return ParallelRLWrapper(
        env_config=default_env_config,
        num_envs=num_envs,
        learning_team_id=learning_team_id,
        team_assignments=default_team_assignments,
        opponent_config=default_opponent_config,
    )
