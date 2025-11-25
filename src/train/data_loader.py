import pickle
from pathlib import Path
from torch.utils.data import DataLoader, TensorDataset, Dataset
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


class SequenceDataset(Dataset):
    def __init__(self, tokens, actions, episode_lengths, context_len: int = 128):
        """
        Args:
            tokens: (TotalTimesteps, N, F) tensor
            actions: (TotalTimesteps, N, A) tensor
            episode_lengths: (NumEpisodes,) tensor
            context_len: Length of sequences to return
        """
        print(f"DEBUG: SequenceDataset init tokens: {tokens.shape}, actions: {actions.shape}")
        self.tokens = tokens
        self.actions = actions
        self.context_len = context_len
        
        # Pre-process actions (shift right by 1, pad with 0 at start of each episode)
        # But since data is flattened, we need to be careful.
        # Actually, we can just shift the whole tensor and then fix the boundaries?
        # Or better: handle shifting in __getitem__.
        # If we return tokens[t:t+L] and actions[t:t+L], 
        # we want input actions to be actions[t-1:t+L-1].
        # So we need access to t-1.
        # If t=0 (start of episode), action is 0.
        
        # Let's build an index mapping
        self.indices = []
        current_idx = 0
        for length in episode_lengths:
            l = length.item()
            if l >= context_len:
                # We can take windows starting from 0 to l - context_len
                # For each window starting at 'start', we need tokens[start:start+context_len]
                # And actions[start-1:start+context_len-1]
                # If start=0, we need a zero action.
                for start in range(l - context_len + 1):
                    self.indices.append((current_idx, start))
            current_idx += l
            
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        base_idx, start_offset = self.indices[idx]
        
        # Tokens: [start : start + context_len]
        abs_start = base_idx + start_offset
        abs_end = abs_start + self.context_len
        
        seq_tokens = self.tokens[abs_start:abs_end]
        
        # Actions: [start-1 : start + context_len - 1]
        # If start_offset == 0, we pad with zeros for the first action
        if start_offset == 0:
            # First action is 0, then actions[base_idx : base_idx + context_len - 1]
            # But we need context_len actions.
            # So [0, action[0], action[1], ... action[L-2]]
            
            # Get actions from data
            # We need context_len - 1 actions from data
            if self.context_len > 1:
                data_actions = self.actions[abs_start : abs_end - 1]
                # Prepend zeros
                zeros = torch.zeros(1, *data_actions.shape[1:], dtype=data_actions.dtype, device=data_actions.device)
                seq_actions = torch.cat([zeros, data_actions], dim=0)
            else:
                seq_actions = torch.zeros(1, *self.actions.shape[1:], dtype=self.actions.dtype, device=self.actions.device)
        else:
            # We can just take slice [abs_start - 1 : abs_end - 1]
            # Ensure indices are valid
            slice_start = abs_start - 1
            slice_end = abs_end - 1
            seq_actions = self.actions[slice_start : slice_end]

        # Verify shapes
        if seq_tokens.shape[0] != self.context_len:
            # Pad if necessary (should not happen if logic is correct)
            pass
        
        if seq_actions.shape[0] != self.context_len:
            # Fix shape
            if seq_actions.shape[0] < self.context_len:
                pad_len = self.context_len - seq_actions.shape[0]
                zeros = torch.zeros(pad_len, *seq_actions.shape[1:], dtype=seq_actions.dtype, device=seq_actions.device)
                seq_actions = torch.cat([seq_actions, zeros], dim=0)
            elif seq_actions.shape[0] > self.context_len:
                seq_actions = seq_actions[:self.context_len]

        return seq_tokens, seq_actions


class AlternatingBatchLengthDataset(Dataset):
    """
    Dataset that supports both short and long batch lengths.
    Each sample randomly chooses between short_batch_len and long_batch_len.
    
    This enables true per-iteration alternation as described in the paper:
    "alternate training on many short batches and occasional long batches"
    
    Key concept: batch_len >= context_len prevents overfitting to start-of-sequence tokens.
    """
    def __init__(
        self, 
        tokens, 
        actions, 
        episode_lengths, 
        context_len: int = 96,
        short_batch_len: int = 32,
        long_batch_len: int = 128,
        long_batch_ratio: float = 0.2,
    ):
        """
        Args:
            tokens: (TotalTimesteps, N, F) tensor
            actions: (TotalTimesteps, N, A) tensor
            episode_lengths: (NumEpisodes,) tensor
            context_len: Model's context length (window size)
            short_batch_len: Length of short batches (should be >= context_len)
            long_batch_len: Length of long batches (should be >= context_len)
            long_batch_ratio: Probability of using long batch length per sample
        """
        self.tokens = tokens
        self.actions = actions
        self.context_len = context_len
        self.short_batch_len = short_batch_len
        self.long_batch_len = long_batch_len
        self.long_batch_ratio = long_batch_ratio
        
        # Build indices for both short and long batch lengths
        # We'll create indices for the maximum batch length (long)
        # and truncate at runtime if we choose short
        self.episodes = []
        current_idx = 0
        for length in episode_lengths:
            l = length.item()
            # We need at least long_batch_len timesteps to support both lengths
            if l >= long_batch_len:
                # Can sample multiple windows from this episode
                for start in range(l - long_batch_len + 1):
                    self.episodes.append((current_idx, start, l))
            current_idx += l
            
    def __len__(self):
        return len(self.episodes)
    
    def __getitem__(self, idx):
        base_idx, start_offset, episode_length = self.episodes[idx]
        
        # Randomly choose batch length for THIS sample
        if torch.rand(1).item() < self.long_batch_ratio:
            batch_len = self.long_batch_len
        else:
            batch_len = self.short_batch_len
        
        # Ensure we don't exceed episode boundaries
        # start_offset was calculated for long_batch_len, so we might need to adjust
        max_start_for_batch_len = episode_length - batch_len
        actual_start_offset = min(start_offset, max_start_for_batch_len)
        
        # Get a batch of batch_len timesteps
        abs_start = base_idx + actual_start_offset
        abs_end = abs_start + batch_len
        
        batch_tokens = self.tokens[abs_start:abs_end]
        
        # Actions: [start-1 : start + batch_len - 1]
        if actual_start_offset == 0:
            # First action is 0
            if batch_len > 1:
                data_actions = self.actions[abs_start : abs_end - 1]
                zeros = torch.zeros(1, *data_actions.shape[1:], dtype=data_actions.dtype)
                batch_actions = torch.cat([zeros, data_actions], dim=0)
            else:
                batch_actions = torch.zeros(1, *self.actions.shape[1:], dtype=self.actions.dtype)
        else:
            slice_start = abs_start - 1
            slice_end = abs_end - 1
            batch_actions = self.actions[slice_start : slice_end]
        
        return batch_tokens, batch_actions


def collate_variable_length_batches(batch):
    """
    Custom collate function for batches with variable lengths.
    Pads all samples to the maximum length in the batch.
    
    Args:
        batch: List of (tokens, actions) tuples with potentially different lengths
        
    Returns:
        Tuple of (padded_tokens, padded_actions) tensors
    """
    # Find max length in this batch
    max_len = max(tokens.shape[0] for tokens, _ in batch)
    
    # Get shapes from first sample
    first_tokens, first_actions = batch[0]
    token_shape = first_tokens.shape[1:]  # (N, F)
    action_shape = first_actions.shape[1:]  # (N, A)
    
    # Pad all samples to max_len
    padded_tokens_list = []
    padded_actions_list = []
    
    for tokens, actions in batch:
        current_len = tokens.shape[0]
        if current_len < max_len:
            # Pad
            pad_len = max_len - current_len
            token_pad = torch.zeros(pad_len, *token_shape, dtype=tokens.dtype)
            action_pad = torch.zeros(pad_len, *action_shape, dtype=actions.dtype)
            tokens = torch.cat([tokens, token_pad], dim=0)
            actions = torch.cat([actions, action_pad], dim=0)
        
        padded_tokens_list.append(tokens)
        padded_actions_list.append(actions)
    
    # Stack into batches
    batched_tokens = torch.stack(padded_tokens_list, dim=0)
    batched_actions = torch.stack(padded_actions_list, dim=0)
    
    return batched_tokens, batched_actions




def create_world_model_data_loader(
    data: dict,
    batch_size: int,
    context_len: int = 96,
    validation_split: float = 0.2,
    num_workers: int = 4,
    use_alternating_lengths: bool = True,
    short_batch_len: int = 32,
    long_batch_len: int = 128,
    long_batch_ratio: float = 0.2,
) -> tuple[DataLoader, DataLoader]:
    """
    Create data loaders for world model training.
    
    Args:
        data: Dictionary containing team tokens, actions, and episode lengths
        batch_size: Batch size for training
        context_len: Model's context length (window size)
        validation_split: Fraction of data to use for validation
        num_workers: Number of worker processes for data loading
        use_alternating_lengths: If True, use AlternatingBatchLengthDataset
        short_batch_len: Number of timesteps in short batches (should be >= context_len)
        long_batch_len: Number of timesteps in long batches (should be >= context_len)
        long_batch_ratio: Probability of using long batch length per sample
    """
    # Combine teams
    team_0 = data["team_0"]
    team_1 = data["team_1"]

    tokens = torch.cat([team_0["tokens"], team_1["tokens"]], dim=0)
    actions = torch.cat([team_0["actions"], team_1["actions"]], dim=0)
    
    # Episode lengths need to be duplicated for both teams
    episode_lengths = torch.cat([data["episode_lengths"], data["episode_lengths"]], dim=0)
    
    print(f"DEBUG: tokens shape before dataset: {tokens.shape}")
    print(f"DEBUG: actions shape before dataset: {actions.shape}")
    
    # Choose dataset type
    if use_alternating_lengths:
        print(f"Using alternating batch lengths: short={short_batch_len}, long={long_batch_len}, ratio={long_batch_ratio}")
        dataset = AlternatingBatchLengthDataset(
            tokens, 
            actions, 
            episode_lengths, 
            context_len=context_len,
            short_batch_len=short_batch_len,
            long_batch_len=long_batch_len,
            long_batch_ratio=long_batch_ratio,
        )
        collate_fn = collate_variable_length_batches
    else:
        dataset = SequenceDataset(tokens, actions, episode_lengths, context_len=context_len)
        collate_fn = None  # Use default collate

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
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
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
