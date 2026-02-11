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

    features_team_0: dict[str, torch.Tensor]
    features_team_1: dict[str, torch.Tensor]
    actions: dict[int, list[torch.Tensor]]
    expert_actions: dict[int, list[torch.Tensor]] = None 
    action_masks: dict[int, list[float]] = None
    rewards: dict[int, list[float]] = None
    episode_length: int = 0
    sim_time: float = 0.0
    agent_skills: dict[int, float] = None
    team_ids: dict[int, int] = None


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
        # robust config access
        if "model" in config and "d_model" in config.model:
             self.token_dim = config.model.d_model
        elif "train" in config and "model" in config.train:
             # Legacy path
             if "transformer" in config.train.model:
                  self.token_dim = config.train.model.transformer.token_dim
             else:
                  self.token_dim = config.train.model.d_model
        else:
             # Fallback
             self.token_dim = 64

        if "environment" in config and "max_ships" in config.environment:
             self.max_ships = config.environment.max_ships
        elif "train" in config and "model" in config.train and "transformer" in config.train.model:
              self.max_ships = config.train.model.transformer.max_ships
        else:
              self.max_ships = 8
        self.num_actions = 3

        self.episodes: list[EpisodeData] = []
        self.total_steps = 0
        self.total_sim_time = 0.0
        self.episodes_collected = 0

    def add_episode(
        self,
        features_team_0: dict[str, torch.Tensor],
        features_team_1: dict[str, torch.Tensor],
        actions: dict[int, list[torch.Tensor]],
        action_masks: dict[int, list[float]],
        rewards: dict[int, list[float]],
        sim_time: float,
        agent_skills: dict[int, float],
        team_ids: dict[int, int],
        expert_actions: dict[int, list[torch.Tensor]] = None,
    ) -> None:
        """
        Add an episode to the collection

        Args:
            features_team_0: Dictionary of feature tensors for team 0
            features_team_1: Dictionary of feature tensors for team 1
            actions: Dictionary mapping ship_id to list of action tensors
            rewards: Dictionary mapping team_id to list of rewards per timestep
            sim_time: Total simulation time for this episode
        """
        # Pick any feature to get length (e.g., position)
        if "position" in features_team_0:
            episode_length = features_team_0["position"].shape[0]
        else:
             # Fallback if empty features? Should not happen.
            episode_length = 0

        episode = EpisodeData(
            features_team_0=features_team_0,
            features_team_1=features_team_1,
            actions=actions,
            expert_actions=expert_actions, 
            action_masks=action_masks,
            rewards=rewards,
            episode_length=episode_length,
            sim_time=sim_time,
            agent_skills=agent_skills,
            team_ids=team_ids,
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

        # Prepare aggregated feature storage
        # We need to scan keys from first episode
        first_ep = self.episodes[0]
        feature_keys = list(first_ep.features_team_0.keys())
        
        features_team_0 = {
            k: torch.zeros((total_timesteps, *first_ep.features_team_0[k].shape[1:]), dtype=first_ep.features_team_0[k].dtype)
            for k in feature_keys
        }
        features_team_1 = {
             k: torch.zeros((total_timesteps, *first_ep.features_team_1[k].shape[1:]), dtype=first_ep.features_team_1[k].dtype)
            for k in feature_keys
        }

        # Actions are now uint8
        actions_team_0 = torch.zeros(
            (total_timesteps, self.max_ships, self.num_actions), dtype=torch.uint8
        )
        actions_team_1 = torch.zeros(
            (total_timesteps, self.max_ships, self.num_actions), dtype=torch.uint8
        )
        
        # Expert Actions
        expert_actions_team_0 = torch.zeros(
            (total_timesteps, self.max_ships, self.num_actions), dtype=torch.uint8
        )
        expert_actions_team_1 = torch.zeros(
            (total_timesteps, self.max_ships, self.num_actions), dtype=torch.uint8
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

        # New fields
        agent_skills_team_0 = torch.zeros((total_timesteps,), dtype=torch.float32)
        agent_skills_team_1 = torch.zeros((total_timesteps,), dtype=torch.float32)
        team_ids_team_0 = torch.zeros((total_timesteps,), dtype=torch.int64)
        team_ids_team_1 = torch.zeros((total_timesteps,), dtype=torch.int64)

        current_idx = 0
        for episode_id, episode in enumerate(self.episodes):
            ep_len = episode.episode_length
            end_idx = current_idx + ep_len

            # Aggregating features
            for k in feature_keys:
                features_team_0[k][current_idx:end_idx] = episode.features_team_0[k]
                features_team_1[k][current_idx:end_idx] = episode.features_team_1[k]

            # Actions
            for ship_id, ship_actions in episode.actions.items():
                for t, action in enumerate(ship_actions):
                    # Action is likely float tensor from coordination, cast to uint8
                    action_u8 = action.to(dtype=torch.uint8)
                    if ship_id < self.max_ships // 2:
                        actions_team_0[current_idx + t, ship_id] = action_u8
                    else:
                        actions_team_1[
                            current_idx + t, ship_id - self.max_ships // 2
                        ] = action_u8
                        
            # Expert Actions
            if episode.expert_actions:
                for ship_id, ship_actions in episode.expert_actions.items():
                    for t, action in enumerate(ship_actions):
                        action_u8 = action.to(dtype=torch.uint8)
                        if ship_id < self.max_ships // 2:
                            expert_actions_team_0[current_idx + t, ship_id] = action_u8
                        else:
                            expert_actions_team_1[
                                current_idx + t, ship_id - self.max_ships // 2
                            ] = action_u8

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

            # Fill skill and team ID
            # Team 0
            agent_skills_team_0[current_idx:end_idx] = episode.agent_skills.get(0, 1.0)
            team_ids_team_0[current_idx:end_idx] = episode.team_ids.get(0, 0)

            # Team 1
            agent_skills_team_1[current_idx:end_idx] = episode.agent_skills.get(1, 1.0)
            team_ids_team_1[current_idx:end_idx] = episode.team_ids.get(1, 1)

            current_idx = end_idx

        return {
            "team_0": {
                "features": features_team_0,
                "actions": actions_team_0,
                "expert_actions": expert_actions_team_0, # New
                "action_masks": actions_mask_team_0,
                "rewards": rewards_team_0,
                "agent_skills": agent_skills_team_0,
                "team_ids": team_ids_team_0,
            },
            "team_1": {
                "features": features_team_1,
                "actions": actions_team_1,
                "expert_actions": expert_actions_team_1, # New
                "action_masks": actions_mask_team_1,
                "rewards": rewards_team_1,
                "agent_skills": agent_skills_team_1,
                "team_ids": team_ids_team_1,
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

        print(f"Worker {self.worker_id}: Saved {num_saved} episodes to {save_path}")

    def finalize(self) -> None:
        """Final save when collection is complete"""
        if self.episodes:
            self.save()
