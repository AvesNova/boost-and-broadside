import pickle
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset
import torch


def load_bc_data(data_path: str = None) -> dict:
    """
    Load BC training data from the latest aggregated data file.

    Args:
        data_path: Optional path to specific data file. If None, uses latest.

    Returns:
        Dictionary containing the loaded BC data
    """
    if data_path is None:
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
    # Since play is symmetric, we can combine both teams' data for training
    team_0 = data["team_0"]
    team_1 = data["team_1"]

    tokens = torch.cat([team_0["tokens"], team_1["tokens"]], dim=0)
    actions = torch.cat([team_0["actions"], team_1["actions"]], dim=0)
    rewards = torch.cat([team_0["rewards"], team_1["rewards"]], dim=0)
    episode_lengths = torch.cat(
        [data["episode_lengths"], data["episode_lengths"]], dim=0
    )
    returns = compute_discounted_returns(rewards, episode_lengths, gamma=gamma)

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


def compute_discounted_returns(all_rewards, all_episode_lengths, gamma=0.99):
    """Vectorized version - faster but uses more memory."""
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


def get_latest_data_path() -> str:
    """
    Get the path to the latest aggregated data file.

    Returns:
        Path string to the latest aggregated_data.pkl file
    """
    base_path = Path("data/bc_pretraining")

    # Find the latest folder
    latest_folder = max(
        (d for d in base_path.iterdir() if d.is_dir()), key=lambda d: d.name
    )

    file_path = latest_folder / "aggregated_data.pkl"
    return str(file_path)


if __name__ == "__main__":
    data = load_bc_data()
    data_loader = create_bc_data_loader(data, batch_size=512)
    12
