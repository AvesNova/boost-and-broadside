"""
Behavioral Cloning data loader.

Handles loading, preprocessing, and splitting of BC training data from
collected episodes.
"""
import pickle
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from omegaconf import DictConfig


class BCDataLoader:
    """
    Data loader for behavioral cloning training.
    Handles loading, preprocessing, and splitting of BC training data.
    """

    def __init__(self, cfg: DictConfig):
        """
        Initialize the BC data loader.

        Args:
            cfg: Configuration dictionary containing data paths and settings
        """
        self.cfg = cfg
        self.data_path = cfg.train.bc_data_path
        self.validation_split = cfg.train.bc.validation_split
        self.gamma = cfg.train.bc.get("gamma", 0.99)

        # Load the data
        self.data = self._load_data()

        # Process the data
        self.tokens, self.actions, self.returns = self._process_data()

        # Get data info
        self.data_info = self._get_data_info()

    def _load_data(self) -> dict[str, Any]:
        """
        Load BC training data from the specified path.

        Returns:
            Dictionary containing the loaded BC data
        """
        if self.data_path is None:
            base_path = Path("data/bc_pretraining")

            # Find the latest folder that has aggregated_data.pkl
            latest_folder = None
            for d in sorted(base_path.iterdir(), key=lambda d: d.name, reverse=True):
                if d.is_dir() and (d / "aggregated_data.pkl").exists():
                    latest_folder = d
                    break

            if latest_folder is None:
                raise FileNotFoundError("No folder with aggregated_data.pkl found")

            file_path = latest_folder / "aggregated_data.pkl"
        else:
            file_path = Path(self.data_path)

        with open(file_path, "rb") as f:
            data = pickle.load(f)

        return data

    def _process_data(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Process the loaded data into tensors suitable for training.

        Returns:
            Tuple of (tokens, actions, returns) tensors
        """
        # Since play is symmetric, we can combine both teams' data for training
        team_0 = self.data["team_0"]
        team_1 = self.data["team_1"]

        tokens = torch.cat([team_0["tokens"], team_1["tokens"]], dim=0)
        actions = torch.cat([team_0["actions"], team_1["actions"]], dim=0)
        rewards = torch.cat([team_0["rewards"], team_1["rewards"]], dim=0)
        episode_lengths = torch.cat(
            [self.data["episode_lengths"], self.data["episode_lengths"]], dim=0
        )

        # Compute discounted returns for value function training
        returns = self._compute_discounted_returns(
            rewards, episode_lengths, gamma=self.gamma
        )

        # Reshape returns to match (batch, max_ships) format for value function training
        # rewards is (T,) but we need returns to be (T, max_ships) for each ship
        batch_size = tokens.shape[0]
        max_ships = tokens.shape[1]
        returns_expanded = returns.unsqueeze(1).expand(-1, max_ships)

        return tokens, actions, returns_expanded

    def _compute_discounted_returns(
        self,
        all_rewards: torch.Tensor,
        all_episode_lengths: torch.Tensor,
        gamma: float = 0.99,
    ) -> torch.Tensor:
        """
        Compute discounted returns for each timestep.

        Args:
            all_rewards: Tensor of rewards for all timesteps
            all_episode_lengths: Tensor of episode lengths
            gamma: Discount factor

        Returns:
            Tensor of discounted returns
        """
        device = all_rewards.device
        max_len = all_episode_lengths.max().item()
        num_episodes = all_episode_lengths.shape[0]

        # Create padded episode tensor
        episodes = torch.zeros(num_episodes, max_len, device=device)

        # Fill in episodes
        start_idx = 0
        for i, length in enumerate(all_episode_lengths):
            ep_len = length.item()
            episodes[i, :ep_len] = all_rewards[start_idx : start_idx + ep_len]
            start_idx += ep_len

        # Create discount matrix: [1, gamma, gamma^2, ..., gamma^(max_len-1)]
        discounts = gamma ** torch.arange(max_len, device=device)

        # Compute returns using convolution-like operation
        returns_padded = torch.zeros_like(episodes)
        for i in range(max_len):
            # For position i, sum rewards[i:] * discounts[:len-i]
            remaining = max_len - i
            returns_padded[:, i] = (episodes[:, i:] * discounts[:remaining]).sum(dim=1)

        # Flatten back to original shape
        returns = torch.zeros_like(all_rewards)
        start_idx = 0
        for i, length in enumerate(all_episode_lengths):
            ep_len = length.item()
            returns[start_idx : start_idx + ep_len] = returns_padded[i, :ep_len]
            start_idx += ep_len

        return returns

    def _get_data_info(self) -> dict[str, Any]:
        """
        Get information about the loaded data.

        Returns:
            Dictionary containing data information
        """
        return {
            "total_samples": self.tokens.shape[0],
            "token_dim": self.tokens.shape[2],
            "max_ships": self.tokens.shape[1],
            "num_actions": self.actions.shape[2],
            "data_shape": {
                "tokens": self.tokens.shape,
                "actions": self.actions.shape,
                "returns": self.returns.shape,
            },
        }

    def get_dataloaders(
        self, batch_size: int, num_workers: int = 4
    ) -> tuple[DataLoader, DataLoader]:
        """
        Create train and validation data loaders.

        Args:
            batch_size: Batch size for training
            num_workers: Number of worker processes for data loading

        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Create dataset
        dataset = TensorDataset(self.tokens, self.actions, self.returns)

        # Split into train and validation
        val_size = int(self.validation_split * len(dataset))
        train_size = len(dataset) - val_size
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        return train_loader, val_loader

    def get_data_info(self) -> dict[str, Any]:
        """
        Get information about the loaded data.

        Returns:
            Dictionary containing data information
        """
        return self.data_info


def create_bc_data_loader(cfg: DictConfig) -> BCDataLoader:
    """
    Create a BC data loader from configuration.

    Args:
        cfg: Configuration dictionary

    Returns:
        BCDataLoader instance
    """
    return BCDataLoader(cfg)
