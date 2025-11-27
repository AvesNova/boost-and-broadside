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
    """
    Dataset for short sequences (e.g., 32 tokens).
    Handles padding for episodes shorter than the sequence length.
    """
    def __init__(self, tokens, actions, episode_lengths, seq_len: int = 32):
        self.tokens = tokens
        self.actions = actions
        self.episode_lengths = episode_lengths
        self.seq_len = seq_len
        
        # Create indices for all episodes
        # We can sample from any episode
        self.episode_indices = []
        current_idx = 0
        for i, length in enumerate(episode_lengths):
            l = length.item()
            self.episode_indices.append((current_idx, l))
            current_idx += l
            
    def __len__(self):
        return len(self.episode_indices)
    
    def __getitem__(self, idx):
        base_idx, length = self.episode_indices[idx]
        
        # Determine start position
        if length <= self.seq_len:
            start = 0
            actual_len = length
        else:
            # Random start
            start = torch.randint(0, length - self.seq_len + 1, (1,)).item()
            actual_len = self.seq_len
            
        # Get tokens and actions
        abs_start = base_idx + start
        abs_end = abs_start + actual_len
        
        seq_tokens = self.tokens[abs_start:abs_end]
        
        # Actions need to be shifted? 
        # In the previous code, actions were handled carefully.
        # Let's stick to the convention: input actions are previous actions.
        # If start=0, first action is 0.
        
        if start == 0:
            if actual_len > 1:
                data_actions = self.actions[abs_start : abs_end - 1]
                zeros = torch.zeros(1, *data_actions.shape[1:], dtype=data_actions.dtype)
                seq_actions = torch.cat([zeros, data_actions], dim=0)
            else:
                seq_actions = torch.zeros(1, *self.actions.shape[1:], dtype=self.actions.dtype)
        else:
            seq_actions = self.actions[abs_start - 1 : abs_end - 1]
            
        # Pad if necessary
        if actual_len < self.seq_len:
            pad_len = self.seq_len - actual_len
            
            # Pad tokens
            token_pad = torch.zeros(pad_len, *seq_tokens.shape[1:], dtype=seq_tokens.dtype)
            seq_tokens = torch.cat([seq_tokens, token_pad], dim=0)
            
            # Pad actions
            action_pad = torch.zeros(pad_len, *seq_actions.shape[1:], dtype=seq_actions.dtype)
            seq_actions = torch.cat([seq_actions, action_pad], dim=0)
            
            # Create loss mask
            # Valid tokens are 1, padded are 0
            loss_mask = torch.ones(self.seq_len, dtype=torch.bool)
            loss_mask[actual_len:] = False
        else:
            loss_mask = torch.ones(self.seq_len, dtype=torch.bool)
            
        return seq_tokens, seq_actions, loss_mask


class LongSequenceDataset(Dataset):
    """
    Dataset for long sequences (e.g., 128 tokens).
    Includes warm-up period where loss is not computed.
    Only includes episodes long enough for the sequence.
    """
    def __init__(self, tokens, actions, episode_lengths, seq_len: int = 128, warmup_len: int = 32):
        self.tokens = tokens
        self.actions = actions
        self.seq_len = seq_len
        self.warmup_len = warmup_len
        
        # Filter episodes that are long enough
        self.indices = []
        current_idx = 0
        for length in episode_lengths:
            l = length.item()
            if l >= seq_len:
                # We can sample from this episode
                # Store (base_idx, length)
                self.indices.append((current_idx, l))
            current_idx += l
            
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        base_idx, length = self.indices[idx]
        
        # Random start position
        # We need seq_len tokens
        max_start = length - self.seq_len
        start = torch.randint(0, max_start + 1, (1,)).item()
        
        abs_start = base_idx + start
        abs_end = abs_start + self.seq_len
        
        seq_tokens = self.tokens[abs_start:abs_end]
        
        # Actions
        if start == 0:
            data_actions = self.actions[abs_start : abs_end - 1]
            zeros = torch.zeros(1, *data_actions.shape[1:], dtype=data_actions.dtype)
            seq_actions = torch.cat([zeros, data_actions], dim=0)
        else:
            seq_actions = self.actions[abs_start - 1 : abs_end - 1]
            
        # Loss mask
        # First warmup_len tokens are 0 (ignore), rest are 1 (compute loss)
        loss_mask = torch.ones(self.seq_len, dtype=torch.bool)
        loss_mask[:self.warmup_len] = False
        
        return seq_tokens, seq_actions, loss_mask


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
    """
    Create data loaders for dual-pool training strategy.
    Returns (train_short_loader, train_long_loader, val_short_loader, val_long_loader).
    """
    # Combine teams
    team_0 = data["team_0"]
    team_1 = data["team_1"]

    tokens = torch.cat([team_0["tokens"], team_1["tokens"]], dim=0)
    actions = torch.cat([team_0["actions"], team_1["actions"]], dim=0)
    episode_lengths = torch.cat([data["episode_lengths"], data["episode_lengths"]], dim=0)
    
    # Calculate pool split ratio
    # ratio = batch_ratio * (long_batch_len / short_batch_len)
    # This is the ratio of short episodes to long episodes
    pool_ratio = batch_ratio * (long_batch_len / short_batch_len)
    
    # Identify eligible episodes for long pool (>= long_batch_len)
    long_eligible_indices = (episode_lengths >= long_batch_len).nonzero().squeeze()
    short_only_indices = (episode_lengths < long_batch_len).nonzero().squeeze()
    
    if long_eligible_indices.ndim == 0:
        long_eligible_indices = long_eligible_indices.unsqueeze(0)
    if short_only_indices.ndim == 0:
        short_only_indices = short_only_indices.unsqueeze(0)
        
    num_long_eligible = len(long_eligible_indices)
    
    # We want N_short / N_long ≈ pool_ratio
    # N_long = Total / (pool_ratio + 1)
    # But we are constrained by num_long_eligible
    
    target_long_count = int(len(episode_lengths) / (pool_ratio + 1))
    actual_long_count = min(target_long_count, num_long_eligible)
    
    # Randomly select episodes for long pool
    perm = torch.randperm(num_long_eligible)
    selected_long_indices = long_eligible_indices[perm[:actual_long_count]]
    remaining_long_indices = long_eligible_indices[perm[actual_long_count:]]
    
    # Short pool gets everything else
    short_pool_indices = torch.cat([short_only_indices, remaining_long_indices])
    
    # Create datasets
    # We need to reconstruct tokens/actions/lengths for each pool
    # This is expensive to do by copying, so we'll pass the full tensors 
    # and a list of allowed episode indices?
    # Actually, the Dataset classes take full tensors and iterate over episode_lengths.
    # We can just pass the subset of episode_lengths and corresponding tokens/actions?
    # No, tokens are flattened. We need to slice them.
    
    # Helper to extract subset
    def extract_subset(indices):
        subset_lengths = episode_lengths[indices]
        
        # We need to find the start/end in the flattened tensors for each episode
        # This requires a cumulative sum of lengths
        cum_lengths = torch.cumsum(torch.cat([torch.tensor([0]), episode_lengths]), dim=0)
        
        subset_tokens_list = []
        subset_actions_list = []
        
        for idx in indices:
            start = cum_lengths[idx].item()
            end = cum_lengths[idx+1].item()
            subset_tokens_list.append(tokens[start:end])
            subset_actions_list.append(actions[start:end])
            
        subset_tokens = torch.cat(subset_tokens_list, dim=0)
        subset_actions = torch.cat(subset_actions_list, dim=0)
        
        return subset_tokens, subset_actions, subset_lengths

    # Extract pools
    long_tokens, long_actions, long_lengths = extract_subset(selected_long_indices)
    short_tokens, short_actions, short_lengths = extract_subset(short_pool_indices)
    
    # Create Datasets
    long_dataset = LongSequenceDataset(
        long_tokens, long_actions, long_lengths, 
        seq_len=long_batch_len, warmup_len=long_batch_len - 96 # Assuming 96 context
    )
    short_dataset = ShortSequenceDataset(
        short_tokens, short_actions, short_lengths, 
        seq_len=short_batch_len
    )
    
    # Split into train/val
    def split_dataset(dataset, val_split):
        total = len(dataset)
        val_size = int(total * val_split)
        train_size = total - val_size
        return torch.utils.data.random_split(dataset, [train_size, val_size])
        
    train_long, val_long = split_dataset(long_dataset, validation_split)
    train_short, val_short = split_dataset(short_dataset, validation_split)
    
    # Create Loaders
    train_long_loader = DataLoader(train_long, batch_size=long_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_long_loader = DataLoader(val_long, batch_size=long_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
    train_short_loader = DataLoader(train_short, batch_size=short_batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_short_loader = DataLoader(val_short, batch_size=short_batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    
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

