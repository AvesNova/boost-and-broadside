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


class ShortSequenceDataset(Dataset):
    """Dataset for short sequences with padding support.
    
    Extracts fixed-length sequences from episodes, padding shorter episodes
    and masking padded positions from loss computation.
    
    Attributes:
        tokens: Flattened token tensor (TotalTimesteps, N, F).
        actions: Flattened action tensor (TotalTimesteps, N, A).
        episode_lengths: Length of each episode (NumEpisodes,).
        seq_len: Target sequence length.
        episode_indices: List of (start_idx, length) tuples for each episode.
    """
    
    def __init__(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        episode_lengths: torch.Tensor,
        seq_len: int = 32
    ) -> None:
        """Initialize short sequence dataset.
        
        Args:
            tokens: Flattened token tensor (TotalTimesteps, N, F).
            actions: Flattened action tensor (TotalTimesteps, N, A).
            episode_lengths: Length of each episode (NumEpisodes,).
            seq_len: Target sequence length for all samples.
        """
        self.tokens = tokens
        self.actions = actions
        self.episode_lengths = episode_lengths
        self.seq_len = seq_len
        
        self.episode_indices: list[tuple[int, int]] = []
        current_idx = 0
        for length in episode_lengths:
            episode_len = length.item()
            self.episode_indices.append((current_idx, episode_len))
            current_idx += episode_len
            
    def __len__(self) -> int:
        """Return number of episodes."""
        return len(self.episode_indices)
    
    def _get_shifted_actions(
        self,
        abs_start: int,
        abs_end: int,
        start_offset: int
    ) -> torch.Tensor:
        """Get actions with proper shifting for causal modeling.
        
        Actions are shifted so that action[t] corresponds to the action
        taken BEFORE observing state[t]. For episode start (offset=0),
        the first action is zero.
        
        Args:
            abs_start: Absolute start index in flattened tensor.
            abs_end: Absolute end index in flattened tensor.
            start_offset: Offset from episode start.
            
        Returns:
            Shifted action tensor (seq_len, N, A).
        """
        actual_len = abs_end - abs_start
        
        if start_offset == 0:
            if actual_len > 1:
                data_actions = self.actions[abs_start : abs_end - 1]
                zeros = torch.zeros(
                    1, *data_actions.shape[1:], dtype=data_actions.dtype
                )
                return torch.cat([zeros, data_actions], dim=0)
            else:
                return torch.zeros(
                    1, *self.actions.shape[1:], dtype=self.actions.dtype
                )
        else:
            return self.actions[abs_start - 1 : abs_end - 1]
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single sequence sample.
        
        Args:
            idx: Episode index.
            
        Returns:
            Tuple of (tokens, actions, loss_mask) where:
                - tokens: (seq_len, N, F)
                - actions: (seq_len, N, A)
                - loss_mask: (seq_len,) boolean mask (True = compute loss)
        """
        base_idx, length = self.episode_indices[idx]
        
        if length <= self.seq_len:
            start_offset = 0
            actual_len = length
        else:
            start_offset = torch.randint(0, length - self.seq_len + 1, (1,)).item()
            actual_len = self.seq_len
            
        abs_start = base_idx + start_offset
        abs_end = abs_start + actual_len
        
        seq_tokens = self.tokens[abs_start:abs_end]
        seq_actions = self._get_shifted_actions(abs_start, abs_end, start_offset)
            
        if actual_len < self.seq_len:
            pad_len = self.seq_len - actual_len
            
            token_pad = torch.zeros(
                pad_len, *seq_tokens.shape[1:], dtype=seq_tokens.dtype
            )
            seq_tokens = torch.cat([seq_tokens, token_pad], dim=0)
            
            action_pad = torch.zeros(
                pad_len, *seq_actions.shape[1:], dtype=seq_actions.dtype
            )
            seq_actions = torch.cat([seq_actions, action_pad], dim=0)
            
            loss_mask = torch.ones(self.seq_len, dtype=torch.bool)
            loss_mask[actual_len:] = False
        else:
            loss_mask = torch.ones(self.seq_len, dtype=torch.bool)
            
        return seq_tokens, seq_actions, loss_mask



class LongSequenceDataset(Dataset):
    """Dataset for long sequences with warm-up period.
    
    Extracts fixed-length sequences from long episodes only. The first
    warmup_len tokens serve as context (KV cache priming) and are excluded
    from loss computation.
    
    Attributes:
        tokens: Flattened token tensor (TotalTimesteps, N, F).
        actions: Flattened action tensor (TotalTimesteps, N, A).
        seq_len: Target sequence length.
        warmup_len: Number of initial tokens to exclude from loss.
        indices: List of (start_idx, length) tuples for eligible episodes.
    """
    
    def __init__(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        episode_lengths: torch.Tensor,
        seq_len: int = 128,
        warmup_len: int = 32
    ) -> None:
        """Initialize long sequence dataset.
        
        Args:
            tokens: Flattened token tensor (TotalTimesteps, N, F).
            actions: Flattened action tensor (TotalTimesteps, N, A).
            episode_lengths: Length of each episode (NumEpisodes,).
            seq_len: Target sequence length for all samples.
            warmup_len: Number of initial tokens for warm-up (excluded from loss).
        """
        self.tokens = tokens
        self.actions = actions
        self.seq_len = seq_len
        self.warmup_len = warmup_len
        
        self.indices: list[tuple[int, int]] = []
        current_idx = 0
        for length in episode_lengths:
            episode_len = length.item()
            if episode_len >= seq_len:
                self.indices.append((current_idx, episode_len))
            current_idx += episode_len
            
    def __len__(self) -> int:
        """Return number of eligible long episodes."""
        return len(self.indices)
    
    def _get_shifted_actions(
        self,
        abs_start: int,
        abs_end: int,
        start_offset: int
    ) -> torch.Tensor:
        """Get actions with proper shifting for causal modeling.
        
        Args:
            abs_start: Absolute start index in flattened tensor.
            abs_end: Absolute end index in flattened tensor.
            start_offset: Offset from episode start.
            
        Returns:
            Shifted action tensor (seq_len, N, A).
        """
        if start_offset == 0:
            data_actions = self.actions[abs_start : abs_end - 1]
            zeros = torch.zeros(1, *data_actions.shape[1:], dtype=data_actions.dtype)
            return torch.cat([zeros, data_actions], dim=0)
        else:
            return self.actions[abs_start - 1 : abs_end - 1]
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get a single long sequence sample.
        
        Args:
            idx: Episode index.
            
        Returns:
            Tuple of (tokens, actions, loss_mask) where:
                - tokens: (seq_len, N, F)
                - actions: (seq_len, N, A)
                - loss_mask: (seq_len,) boolean mask (True = compute loss)
        """
        base_idx, length = self.indices[idx]
        
        max_start = length - self.seq_len
        start_offset = torch.randint(0, max_start + 1, (1,)).item()
        
        abs_start = base_idx + start_offset
        abs_end = abs_start + self.seq_len
        
        seq_tokens = self.tokens[abs_start:abs_end]
        seq_actions = self._get_shifted_actions(abs_start, abs_end, start_offset)
            
        loss_mask = torch.ones(self.seq_len, dtype=torch.bool)
        loss_mask[:self.warmup_len] = False
        
        return seq_tokens, seq_actions, loss_mask



def _extract_episode_subset(
    tokens: torch.Tensor,
    actions: torch.Tensor,
    episode_lengths: torch.Tensor,
    indices: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Extract subset of episodes from flattened tensors.
    
    Args:
        tokens: Flattened token tensor (TotalTimesteps, N, F).
        actions: Flattened action tensor (TotalTimesteps, N, A).
        episode_lengths: Length of each episode (NumEpisodes,).
        indices: Indices of episodes to extract.
        
    Returns:
        Tuple of (subset_tokens, subset_actions, subset_lengths).
    """
    subset_lengths = episode_lengths[indices]
    
    cum_lengths = torch.cumsum(
        torch.cat([torch.tensor([0]), episode_lengths]), dim=0
    )
    
    subset_tokens_list = []
    subset_actions_list = []
    
    for idx in indices:
        start = cum_lengths[idx].item()
        end = cum_lengths[idx + 1].item()
        subset_tokens_list.append(tokens[start:end])
        subset_actions_list.append(actions[start:end])
        
    subset_tokens = torch.cat(subset_tokens_list, dim=0)
    subset_actions = torch.cat(subset_actions_list, dim=0)
    
    return subset_tokens, subset_actions, subset_lengths


def _split_dataset(
    dataset: Dataset,
    val_split: float
) -> tuple[Dataset, Dataset]:
    """Split dataset into train and validation sets.
    
    Args:
        dataset: Dataset to split.
        val_split: Fraction of data for validation.
        
    Returns:
        Tuple of (train_dataset, val_dataset).
    """
    total = len(dataset)
    val_size = int(total * val_split)
    train_size = total - val_size
    return torch.utils.data.random_split(dataset, [train_size, val_size])


def create_dual_pool_data_loaders(
    data: dict,
    short_batch_size: int,
    long_batch_size: int,
    short_batch_len: int = 32,
    long_batch_len: int = 128,
    batch_ratio: int = 4,
    validation_split: float = 0.2,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Create data loaders for dual-pool mixed batch training.
    
    Episodes are randomly assigned to short or long pools at each epoch.
    The pool ratio is calculated to match the desired batch ratio during training.
    
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
    team_0 = data["team_0"]
    team_1 = data["team_1"]

    tokens = torch.cat([team_0["tokens"], team_1["tokens"]], dim=0)
    actions = torch.cat([team_0["actions"], team_1["actions"]], dim=0)
    episode_lengths = torch.cat(
        [data["episode_lengths"], data["episode_lengths"]], dim=0
    )
    
    pool_ratio = batch_ratio * (long_batch_len / short_batch_len)
    
    long_eligible_indices = (episode_lengths >= long_batch_len).nonzero().squeeze()
    short_only_indices = (episode_lengths < long_batch_len).nonzero().squeeze()
    
    if long_eligible_indices.ndim == 0:
        long_eligible_indices = long_eligible_indices.unsqueeze(0)
    if short_only_indices.ndim == 0:
        short_only_indices = short_only_indices.unsqueeze(0)
        
    num_long_eligible = len(long_eligible_indices)
    
    target_long_count = int(len(episode_lengths) / (pool_ratio + 1))
    actual_long_count = min(target_long_count, num_long_eligible)
    
    perm = torch.randperm(num_long_eligible)
    selected_long_indices = long_eligible_indices[perm[:actual_long_count]]
    remaining_long_indices = long_eligible_indices[perm[actual_long_count:]]
    
    short_pool_indices = torch.cat([short_only_indices, remaining_long_indices])
    
    long_tokens, long_actions, long_lengths = _extract_episode_subset(
        tokens, actions, episode_lengths, selected_long_indices
    )
    short_tokens, short_actions, short_lengths = _extract_episode_subset(
        tokens, actions, episode_lengths, short_pool_indices
    )
    
    warmup_len = long_batch_len - 96
    long_dataset = LongSequenceDataset(
        long_tokens, long_actions, long_lengths,
        seq_len=long_batch_len, warmup_len=warmup_len
    )
    short_dataset = ShortSequenceDataset(
        short_tokens, short_actions, short_lengths,
        seq_len=short_batch_len
    )
    
    train_long, val_long = _split_dataset(long_dataset, validation_split)
    train_short, val_short = _split_dataset(short_dataset, validation_split)
    
    train_long_loader = DataLoader(
        train_long, batch_size=long_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_long_loader = DataLoader(
        val_long, batch_size=long_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    train_short_loader = DataLoader(
        train_short, batch_size=short_batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_short_loader = DataLoader(
        val_short, batch_size=short_batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    
    return train_short_loader, train_long_loader, val_short_loader, val_long_loader



if __name__ == "__main__":
    data = load_bc_data()
    # Test dual pool creation
    ts, tl, vs, vl = create_dual_pool_data_loaders(
        data, 
        short_batch_size=128, 
        long_batch_size=32,
        short_batch_len=32,
        long_batch_len=128,
        batch_ratio=4
    )
    print(f"Short Train Batches: {len(ts)}")
    print(f"Long Train Batches: {len(tl)}")

