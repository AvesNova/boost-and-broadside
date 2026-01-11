"""Data collector for behavioral cloning training data."""

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from omegaconf import DictConfig


@dataclass
class EpisodeData:
    """Data for a single episode"""

    tokens_team_0: torch.Tensor
    tokens_team_1: torch.Tensor
    actions: dict[int, list[torch.Tensor]]
    action_masks: dict[int, list[float]]
    rewards: dict[int, list[float]]
    episode_length: int
    sim_time: float


class DataCollector:
    """Collects and saves episode data for behavioral cloning"""

    def __init__(self, config: DictConfig, worker_id: int, run_timestamp: str):
        """
        Initialize data collector

        Args:
            config: Configuration dictionary with collect settings
            worker_id: Unique identifier for this worker process
            run_timestamp: Timestamp string for this run
        """
        self.config = config
        self.worker_id = worker_id
        self.run_timestamp = run_timestamp

        run_dir = Path(config.collect.output_dir) / run_timestamp
        self.output_dir = run_dir / f"worker_{worker_id}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.save_frequency = config.collect.save_frequency
        self.token_dim = config.train.model.transformer.token_dim
        self.max_ships = config.train.model.transformer.max_ships
        self.num_actions = 3

        self.episodes: list[EpisodeData] = []
        self.total_steps = 0
        self.total_sim_time = 0.0
        self.episodes_collected = 0

    def add_episode(
        self,
        tokens_team_0: torch.Tensor,
        tokens_team_1: torch.Tensor,
        actions: dict[int, list[torch.Tensor]],
        action_masks: dict[int, list[float]],
        rewards: dict[int, list[float]],
        sim_time: float,
    ) -> None:
        """
        Add an episode to the collection

        Args:
            tokens_team_0: Token sequence for team 0 (T, max_ships, token_dim)
            tokens_team_1: Token sequence for team 1 (T, max_ships, token_dim)
            actions: Dictionary mapping ship_id to list of action tensors
            rewards: Dictionary mapping team_id to list of rewards per timestep
            sim_time: Total simulation time for this episode
        """
        episode_length = tokens_team_0.shape[0]

        episode = EpisodeData(
            tokens_team_0=tokens_team_0,
            tokens_team_1=tokens_team_1,
            actions=actions,
            action_masks=action_masks,
            rewards=rewards,
            episode_length=episode_length,
            sim_time=sim_time,
        )

        self.episodes.append(episode)
        self.total_steps += episode_length
        self.total_sim_time += sim_time
        self.episodes_collected += 1

        if len(self.episodes) >= self.save_frequency:
            self.save()

    def _aggregate_episodes(self) -> dict[str, Any]:
        """
        Aggregate all episode data into single tensors organized by team

        Returns:
            Dictionary containing aggregated data organized by team
        """
        if not self.episodes:
            return {}

        num_episodes = len(self.episodes)

        episode_lengths = torch.tensor(
            [ep.episode_length for ep in self.episodes], dtype=torch.int64
        )

        total_timesteps = episode_lengths.sum().item()

        # Calculate simulation time only for the current batch of episodes
        batch_sim_time = sum(ep.sim_time for ep in self.episodes)

        tokens_team_0 = torch.zeros(
            (total_timesteps, self.max_ships, self.token_dim), dtype=torch.float32
        )
        tokens_team_1 = torch.zeros(
            (total_timesteps, self.max_ships, self.token_dim), dtype=torch.float32
        )

        actions_team_0 = torch.zeros(
            (total_timesteps, self.max_ships, self.num_actions), dtype=torch.float32
        )
        actions_team_1 = torch.zeros(
            (total_timesteps, self.max_ships, self.num_actions), dtype=torch.float32
        )

        actions_mask_team_0 = torch.zeros(
            (total_timesteps, self.max_ships), dtype=torch.float32
        )
        actions_mask_team_1 = torch.zeros(
            (total_timesteps, self.max_ships), dtype=torch.float32
        )

        rewards_team_0 = torch.zeros((total_timesteps,), dtype=torch.float32)
        rewards_team_1 = torch.zeros((total_timesteps,), dtype=torch.float32)

        episode_ids = torch.zeros((total_timesteps,), dtype=torch.int64)

        current_idx = 0
        for episode_id, episode in enumerate(self.episodes):
            ep_len = episode.episode_length
            end_idx = current_idx + ep_len

            tokens_team_0[current_idx:end_idx] = episode.tokens_team_0
            tokens_team_1[current_idx:end_idx] = episode.tokens_team_1

            for ship_id, ship_actions in episode.actions.items():
                for t, action in enumerate(ship_actions):
                    if ship_id < self.max_ships // 2:
                        actions_team_0[current_idx + t, ship_id] = action
                    else:
                        actions_team_1[
                            current_idx + t, ship_id - self.max_ships // 2
                        ] = action

            for ship_id, ship_masks in episode.action_masks.items():
                for t, mask in enumerate(ship_masks):
                    if ship_id < self.max_ships // 2:
                        actions_mask_team_0[current_idx + t, ship_id] = mask
                    else:
                        actions_mask_team_1[
                            current_idx + t, ship_id - self.max_ships // 2
                        ] = mask


            if 0 in episode.rewards and len(episode.rewards[0]) > 0:
                reward_tensor_0 = torch.tensor(episode.rewards[0], dtype=torch.float32)
                if len(reward_tensor_0) < ep_len:
                    reward_tensor_0 = torch.nn.functional.pad(
                        reward_tensor_0, (0, ep_len - len(reward_tensor_0))
                    )
                elif len(reward_tensor_0) > ep_len:
                    reward_tensor_0 = reward_tensor_0[:ep_len]
                rewards_team_0[current_idx:end_idx] = reward_tensor_0

            if 1 in episode.rewards and len(episode.rewards[1]) > 0:
                reward_tensor_1 = torch.tensor(episode.rewards[1], dtype=torch.float32)
                if len(reward_tensor_1) < ep_len:
                    reward_tensor_1 = torch.nn.functional.pad(
                        reward_tensor_1, (0, ep_len - len(reward_tensor_1))
                    )
                elif len(reward_tensor_1) > ep_len:
                    reward_tensor_1 = reward_tensor_1[:ep_len]
                rewards_team_1[current_idx:end_idx] = reward_tensor_1

            episode_ids[current_idx:end_idx] = episode_id

            current_idx = end_idx

        return {
            "team_0": {
                "tokens": tokens_team_0,
                "actions": actions_team_0,
                "action_masks": actions_mask_team_0,
                "rewards": rewards_team_0,
            },
            "team_1": {
                "tokens": tokens_team_1,
                "actions": actions_team_1,
                "action_masks": actions_mask_team_1,
                "rewards": rewards_team_1,
            },
            "episode_ids": episode_ids,
            "episode_lengths": episode_lengths,
            "metadata": {
                "num_episodes": num_episodes,
                "total_timesteps": total_timesteps,
                "total_sim_time": batch_sim_time,
                "worker_id": self.worker_id,
                "run_timestamp": self.run_timestamp,
                "max_ships": self.max_ships,
                "token_dim": self.token_dim,
                "num_actions": self.num_actions,
            },
        }

    def save(self) -> None:
        """Save collected data to disk and clear memory"""
        if not self.episodes:
            return

        data = self._aggregate_episodes()

        save_path = self.output_dir / f"data_checkpoint_{self.episodes_collected}.pkl"

        with open(save_path, "wb") as f:
            pickle.dump(data, f)

        # Clear episodes to free memory
        num_saved = len(self.episodes)
        self.episodes = []

        print(f"Worker {self.worker_id}: Saved {num_saved} episodes " f"to {save_path}")

    def finalize(self) -> None:
        """Final save when collection is complete"""
        if self.episodes:
            self.save()
