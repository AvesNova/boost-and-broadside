import pickle
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch


def get_latest_data_path() -> str:
    """Find the latest aggregated data file."""
    base_path = Path("data/bc_pretraining")
    
    # Find the latest folder that has aggregated_data.pkl
    latest_folder = None
    for d in sorted(base_path.iterdir(), key=lambda d: d.name, reverse=True):
        if d.is_dir() and (d / "aggregated_data.pkl").exists():
            latest_folder = d
            break

    if latest_folder is None:
        raise FileNotFoundError("No folder with aggregated_data.pkl found")

    return str(latest_folder / "aggregated_data.pkl")


def load_bc_data(data_path: str = None) -> dict:
    """
    Load BC training data from the latest aggregated data file.

    Args:
        data_path: Optional path to specific data file. If None, uses latest.

    Returns:
        Dictionary containing the loaded BC data
    """
    if data_path is None:
        file_path = Path(get_latest_data_path())
    else:
        file_path = Path(data_path)

    with open(file_path, "rb") as f:
        data = pickle.load(f)

    return data


def create_bc_data_loader(
    data: dict,
    batch_size: int,
    gamma: float = 0.99,
    validation_split: float = 0.2,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """Create BC training data loaders from team_0 perspective only.

    Args:
        data: Dictionary containing team data, episode lengths, etc.
        batch_size: Batch size for training.
        gamma: Discount factor for returns computation.
        validation_split: Fraction of data for validation.
        num_workers: Number of worker processes for data loading.

    Returns:
        Tuple of (train_loader, val_loader).
    """
    # Load only team_0 data
    team_0 = data["team_0"]

    tokens = team_0["tokens"]
    actions = team_0["actions"]
    rewards = team_0["rewards"]
    episode_lengths = data["episode_lengths"]

    returns = _compute_discounted_returns(rewards, episode_lengths, gamma=gamma)

    dataset = TensorDataset(tokens, actions, returns)

    # Split into train and validation
    total_size = len(dataset)
    val_size = int(total_size * validation_split)
    train_size = total_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

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


def _compute_discounted_returns(
    all_rewards: torch.Tensor,
    all_episode_lengths: torch.Tensor,
    gamma: float = 0.99,
) -> torch.Tensor:
    """Compute discounted returns for each timestep.

    Args:
        all_rewards: Tensor of rewards for all timesteps (T,).
        all_episode_lengths: Tensor of episode lengths (NumEpisodes,).
        gamma: Discount factor.

    Returns:
        Tensor of discounted returns (T,).
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


from train.unified_dataset import UnifiedEpisodeDataset, ShortView, LongView


def create_unified_data_loaders(
    data: dict,
    short_batch_size: int,
    long_batch_size: int,
    short_batch_len: int = 32,
    long_batch_len: int = 128,
    batch_ratio: int = 4,
    validation_split: float = 0.2,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Create data loaders for unified pool mixed batch training.

    Episodes are stored once in UnifiedEpisodeDataset.
    ShortView and LongView provide access to the same underlying data.

    Args:
        data: Dictionary containing team tokens, actions, and episode lengths.
        short_batch_size: Batch size for short sequences.
        long_batch_size: Batch size for long sequences.
        short_batch_len: Sequence length for short batches.
        long_batch_len: Sequence length for long batches.
        batch_ratio: Ratio of short to long batches (e.g., 4 = 4 short : 1 long).
        validation_split: Fraction of data for validation.
        num_workers: Number of worker processes for data loading.

    Returns:
        Tuple of (train_short_loader, train_long_loader, val_short_loader, val_long_loader).
    """
    # Load only team_0 data
    team_0 = data["team_0"]
    tokens = team_0["tokens"]
    actions = team_0["actions"]
    rewards = team_0["rewards"]
    episode_lengths = data["episode_lengths"]

    # Compute returns
    returns = _compute_discounted_returns(rewards, episode_lengths)

    # Load action masks or default to ones
    action_masks = team_0.get("action_masks", None)
    if action_masks is None:
        # Default to ones (TotalTimesteps, MaxShips)
        action_masks = torch.ones(actions.shape[0], actions.shape[1], dtype=torch.float32)

    # Initialize Unified Dataset (One copy of tensors)
    unified_dataset = UnifiedEpisodeDataset(
        tokens, actions, returns, episode_lengths, action_masks
    )

    # Create Indices
    num_episodes = len(episode_lengths)
    all_indices = torch.randperm(num_episodes).tolist()

    # Split Indices for Train/Val
    val_size = int(num_episodes * validation_split)
    train_size = num_episodes - val_size
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:]

    # Filter for Long Views
    # We only include episodes capable of supporting long sequences
    train_long_indices = [
        i for i in train_indices if unified_dataset.get_length(i) >= long_batch_len
    ]
    val_long_indices = [
        i for i in val_indices if unified_dataset.get_length(i) >= long_batch_len
    ]

    # Create Views
    # Short Views use ALL episodes in their respective split
    train_short_view = ShortView(
        unified_dataset, train_indices, seq_len=short_batch_len
    )
    val_short_view = ShortView(unified_dataset, val_indices, seq_len=short_batch_len)

    # Long Views only use eligible episodes
    warmup_len = long_batch_len - 96  # As per previous logic (128 - 96 = 32)
    train_long_view = LongView(
        unified_dataset,
        train_long_indices,
        seq_len=long_batch_len,
        warmup_len=warmup_len,
    )
    val_long_view = LongView(
        unified_dataset, val_long_indices, seq_len=long_batch_len, warmup_len=warmup_len
    )

    # Create Loaders
    train_short_loader = DataLoader(
        train_short_view,
        batch_size=short_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_short_loader = DataLoader(
        val_short_view,
        batch_size=short_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    train_long_loader = DataLoader(
        train_long_view,
        batch_size=long_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_long_loader = DataLoader(
        val_long_view,
        batch_size=long_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_short_loader, train_long_loader, val_short_loader, val_long_loader
