
import torch
from torch.utils.data import Dataset
from train.unified_dataset import UnifiedEpisodeDataset

class ContinuousView(Dataset):
    """
    A view of the dataset as a continuous stream of tokens.
    Used for MambaBB training where we don't pad, but treat data as one long sequence.
    """
    def __init__(
        self,
        dataset: UnifiedEpisodeDataset,
        indices: list[int], # List of GLOBAL START TIMESTEPS
        seq_len: int = 1024,
    ):
        self.dataset = dataset
        self.indices = indices
        self.seq_len = seq_len
        
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
        # We need [T] tokens, [T] actions, [T] episode_ids
        # Note: Input at T is State_T. Target is State_{T+1} (handled by causal shifting in model or here?)
        # Standard: 
        # - Input: State_0 ... State_{L-1}
        # - Target: State_1 ... State_L
        # - Action Input: Action_0 ... Action_{L-1} (Teacher Forcing)
        
        # So we actually need L+1 length to get targets?
        # Or we just predict next token.
        # Let's read L+1 to be safe and slice in collate or model if needed.
        # Actually standard causal modeling: Read L. Input = data[:-1], Target = data[1:].
        # But here we have interleaved State/Action.
        # Let's read exactly seq_len.
        
        tokens = self.dataset.get_cross_episode_slice("tokens", global_start, self.seq_len)
        actions = self.dataset.get_cross_episode_slice("actions", global_start, self.seq_len)
        episode_ids = self.dataset.get_cross_episode_slice("episode_ids", global_start, self.seq_len)
        
        # Masks if they exist
        if self.dataset.has_dataset("action_masks"):
             action_masks = self.dataset.get_cross_episode_slice("action_masks", global_start, self.seq_len)
        else:
             action_masks = torch.ones_like(actions)

        # 2. Process Indices
        # Mamba needs int32 usually for seq_idx
        seq_idx = episode_ids.int()
        
        # 3. Compute Reset Semantic Mask
        # Where does a NEW episode start?
        # It starts where seq_idx changes relative to the PREVIOUS step.
        # For the very first step in this chunk, we don't know if it's new unless we check t-1.
        # But usually we just say: if we cut in the middle of episode, it's NOT a reset.
        # IF we cut exactly at start, it IS.
        # The Mamba kernel handles the hidden state reset based on seq_idx.
        # We need `reset_mask` for the Additive Reset Vector in the Embedding layer.
        
        # Logic:
        # mask[t] = 1 if seq_idx[t] != seq_idx[t-1]
        # For t=0, we look at global_start - 1.
        
        # Fetch one extra previous int for robust boundary check
        if global_start == 0:
             prev_id = -1
        else:
             prev_id_tensor = self.dataset.get_cross_episode_slice("episode_ids", global_start - 1, 1)
             prev_id = prev_id_tensor[0].int().item()
        
        # Concatenate prev to current
        full_ids = torch.cat([torch.tensor([prev_id], dtype=torch.int32), seq_idx], dim=0)
        
        # Diff
        diff = full_ids[1:] != full_ids[:-1]
        reset_mask = diff # (L,) boolean
        
        # 4. Loss Masking
        # We might want to mask loss at the transition boundary?
        # Spec: "Mask (Drop Loss) when: Transition Frame"
        # Since at transition, S_{t+1} belongs to new episode, predicting it from A_t (old episode) is invalid.
        loss_mask = ~reset_mask
        
        # Also strictly mask dead ships? 
        # Spec: "Dead Entity: HP <= 0 in the target step"
        # We assume tokens include HP. 
        # But we can leave that to the Model/Loss function effectively if tokens are passed.
        # Or we can compute it here.
        # Let's stick to returning raw data and a 'validity' mask.
        
        return {
            "states": tokens,
            "actions": actions,
            "seq_idx": seq_idx,
            "reset_mask": reset_mask,
            "loss_mask": loss_mask,
            "action_masks": action_masks
        }
