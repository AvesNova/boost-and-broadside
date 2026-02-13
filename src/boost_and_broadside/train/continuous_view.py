import torch
from torch.utils.data import Dataset
from typing import Optional, Dict, Any
from boost_and_broadside.train.unified_dataset import UnifiedEpisodeDataset, BaseView

class ContinuousView(BaseView):
    """
    A view of the dataset as a continuous stream of tokens.
    Used for MambaBB training where we don't pad, but treat data as one long sequence.
    """
    def __init__(
        self,
        dataset: UnifiedEpisodeDataset,
        indices: list[int], # List of GLOBAL START TIMESTEPS
        seq_len: int = 1024,
        reward_config: Optional[dict] = None
    ):
        super().__init__(dataset, indices, seq_len, reward_config)
        
        # Check if episode_ids is available (Critical for Mamba)
        if not self.dataset.has_dataset("episode_ids"):
            raise ValueError("ContinuousView requires 'episode_ids' in HDF5 for handling state resets.")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        # idx translates to a start position in the global stream
        # In this view, self.indices are actual timestep indices.
        global_start = self.indices[idx]
        
        # 1. Fetch Data
        tokens = self.dataset.get_cross_episode_slice("tokens", global_start, self.seq_len)
        actions = self.dataset.get_cross_episode_slice("actions", global_start, self.seq_len)
        episode_ids = self.dataset.get_cross_episode_slice("episode_ids", global_start, self.seq_len)
        
        if self.dataset.has_dataset("team_ids"):
             team_ids = self.dataset.get_cross_episode_slice("team_ids", global_start, self.seq_len)
        else:
             team_ids = torch.zeros(self.seq_len, dtype=torch.int64)

        # Masks if they exist
        if self.dataset.has_dataset("action_masks"):
             action_masks = self.dataset.get_cross_episode_slice("action_masks", global_start, self.seq_len)
        else:
             action_masks = torch.ones_like(actions)

        # 2. Process Indices
        seq_idx = episode_ids.int()
        
        # 3. Compute Reset Semantic Mask
        if global_start == 0:
             prev_id = -1
        else:
             prev_id_tensor = self.dataset.get_cross_episode_slice("episode_ids", global_start - 1, 1)
             prev_id = prev_id_tensor[0].int().item()
        
        full_ids = torch.cat([torch.tensor([prev_id], dtype=torch.int32), seq_idx], dim=0)
        diff = full_ids[1:] != full_ids[:-1]
        reset_mask = diff # (L,) boolean
        
        # 4. Loss Masking
        loss_mask = ~reset_mask
        
        # Fetch Position and Attitude
        pos = self.dataset.get_cross_episode_slice("position", global_start, self.seq_len)
        att = self.dataset.get_cross_episode_slice("attitude", global_start, self.seq_len)
        
        # Rewards and Returns
        rewards_raw = self.dataset.get_cross_episode_slice("rewards", global_start, self.seq_len)
        returns = self.dataset.get_cross_episode_slice("returns", global_start, self.seq_len)
        
        # Compute Rewards On-The-Fly if configured
        if self.reward_registry:
             rewards = self._compute_aligned_rewards(tokens, team_ids, rewards_raw, seq_pos=pos, seq_att=att)
        else:
             rewards = rewards_raw

        return {
            "states": tokens,
            "actions": actions,
            "team_ids": team_ids,
            "seq_idx": seq_idx,
            "reset_mask": reset_mask,
            "loss_mask": loss_mask,
            "action_masks": action_masks,
            "pos": pos,
            "rewards": rewards,
            "returns": returns
        }
