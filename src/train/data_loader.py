from pathlib import Path
from torch.utils.data import DataLoader
import torch

from train.unified_dataset import UnifiedEpisodeDataset, ShortView, LongView

def get_latest_data_path() -> str:
    """Find the latest aggregated HDF5 data file."""
    base_path = Path("data/bc_pretraining")
    
    # Find the latest folder that has aggregated_data.h5
    latest_folder = None
    for d in sorted(base_path.iterdir(), key=lambda d: d.name, reverse=True):
        if d.is_dir() and (d / "aggregated_data.h5").exists():
            latest_folder = d
            break

    if latest_folder is None:
        raise FileNotFoundError("No folder with aggregated_data.h5 found")

    return str(latest_folder / "aggregated_data.h5")


def load_bc_data(data_path: str = None) -> str:
    """
    Resolve the path to the BC data.
    
    Args:
        data_path: Optional path. If None, uses latest.
        
    Returns:
        Path string to the HDF5 file.
    """
    if data_path is None:
        return get_latest_data_path()
    return data_path


def create_unified_data_loaders(
    data_path: str,
    short_batch_size: int,
    long_batch_size: int,
    short_batch_len: int = 32,
    long_batch_len: int = 128,
    batch_ratio: int = 4, # Unused effectively, controlled by caller
    validation_split: float = 0.2,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Create data loaders for unified pool mixed batch training using HDF5.
    """
    # Initialize Dataset (Lightweight, just loads lengths)
    unified_dataset = UnifiedEpisodeDataset(data_path)
    episode_lengths = unified_dataset.episode_lengths
    num_episodes = len(episode_lengths)

    # Create Indices
    all_indices = torch.randperm(num_episodes).tolist()

    # Split Indices for Train/Val
    val_size = int(num_episodes * validation_split)
    train_size = num_episodes - val_size
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:]

    # Filter for Long Views
    train_long_indices = [
        i for i in train_indices if unified_dataset.get_length(i) >= long_batch_len
    ]
    val_long_indices = [
        i for i in val_indices if unified_dataset.get_length(i) >= long_batch_len
    ]

    # Create Views
    train_short_view = ShortView(
        unified_dataset, train_indices, seq_len=short_batch_len
    )
    val_short_view = ShortView(unified_dataset, val_indices, seq_len=short_batch_len)

    warmup_len = long_batch_len - 96
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
    # Note: persistent_workers=True is recommended if num_workers > 0
    # but requires lifecycle management.
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": (num_workers > 0), 
        "prefetch_factor": 2 if num_workers > 0 else None
    }

    train_short_loader = DataLoader(
        train_short_view, batch_size=short_batch_size, shuffle=True, **kwargs
    )
    val_short_loader = DataLoader(
        val_short_view, batch_size=short_batch_size, shuffle=False, **kwargs
    )

    train_long_loader = DataLoader(
        train_long_view, batch_size=long_batch_size, shuffle=True, **kwargs
    )
    val_long_loader = DataLoader(
        val_long_view, batch_size=long_batch_size, shuffle=False, **kwargs
    )

    return train_short_loader, train_long_loader, val_short_loader, val_long_loader


def create_bc_data_loader(
    data_path: str,
    batch_size: int,
    gamma: float = 0.99, # Deprecated/Unused, returns are precomputed
    validation_split: float = 0.2,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader]:
    """
    Create BC training data loaders (HDF5 backed, Sequence Length=1).
    """
    # Use ShortView with seq_len=1 to simulate timestep sampling
    # It samples random timesteps from random episodes.
    
    unified_dataset = UnifiedEpisodeDataset(data_path)
    episode_lengths = unified_dataset.episode_lengths
    num_episodes = len(episode_lengths)
    
    all_indices = torch.randperm(num_episodes).tolist()
    val_size = int(num_episodes * validation_split)
    train_size = num_episodes - val_size
    train_indices = all_indices[:train_size]
    val_indices = all_indices[train_size:]
    
    train_view = ShortView(unified_dataset, train_indices, seq_len=1)
    val_view = ShortView(unified_dataset, val_indices, seq_len=1)
    
    kwargs = {
        "num_workers": num_workers,
        "pin_memory": True,
        "persistent_workers": (num_workers > 0),
        "prefetch_factor": 2 if num_workers > 0 else None
    }
    
    train_loader = DataLoader(
        train_view, batch_size=batch_size, shuffle=True, **kwargs
    )
    val_loader = DataLoader(
        val_view, batch_size=batch_size, shuffle=False, **kwargs
    )
    
    return train_loader, val_loader
