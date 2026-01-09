import torch
from torch.utils.data import Dataset
import logging

log = logging.getLogger(__name__)


class UnifiedEpisodeDataset:
    """
    Central storage for all episode data.
    Avoids duplicating tensors by holding them once.
    Views (ShortView, LongView) will reference this dataset.
    """

    def __init__(
        self,
        tokens: torch.Tensor,
        actions: torch.Tensor,
        episode_lengths: torch.Tensor,
    ):
        """
        Args:
            tokens: (TotalTimesteps, N, F)
            actions: (TotalTimesteps, N, A)
            episode_lengths: (NumEpisodes,)
        """
        self.tokens = tokens
        self.actions = actions
        self.episode_lengths = episode_lengths

        # Precompute start indices for O(1) access
        self.episode_starts = torch.zeros_like(episode_lengths)
        self.episode_starts[1:] = torch.cumsum(episode_lengths[:-1], dim=0)

    def get_episode(self, episode_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns full (tokens, actions) for a given episode index."""
        start = self.episode_starts[episode_idx].item()
        length = self.episode_lengths[episode_idx].item()
        end = start + length
        return self.tokens[start:end], self.actions[start:end]

    def get_length(self, episode_idx: int) -> int:
        return self.episode_lengths[episode_idx].item()


class BaseView(Dataset):
    def __init__(
        self, dataset: UnifiedEpisodeDataset, indices: list[int], seq_len: int
    ):
        self.dataset = dataset
        self.indices = indices
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.indices)

    def _get_shifted_actions(
        self, full_actions: torch.Tensor, start_offset: int, target_len: int
    ) -> torch.Tensor:
        """
        Get actions shifted for causal modeling.
        If start_offset == 0, prepend zeros (no previous action).
        Else, take previous action.
        """
        # full_actions is the slice of the episode relevant to the context
        # But wait, we need the *previous* action for the first step if offset > 0.
        # So we should probably slice from the main dataset with care.
        pass  # Implemented in subclasses or helper using absolute indices

    def _get_shifted_actions_from_full(
        self, abs_start: int, abs_end: int, start_offset: int
    ) -> torch.Tensor:
        """Helper to extract shifted actions from the main dataset tensor."""
        if start_offset == 0:
            # First action is 0, then actions[0...T-1]
            data_actions = self.dataset.actions[abs_start : abs_end - 1]
            zeros = torch.zeros(
                1,
                *data_actions.shape[1:],
                dtype=data_actions.dtype,
                device=data_actions.device,
            )
            return torch.cat([zeros, data_actions], dim=0)
        else:
            # Previous action exists
            return self.dataset.actions[abs_start - 1 : abs_end - 1]


class ShortView(BaseView):
    def __init__(
        self, dataset: UnifiedEpisodeDataset, indices: list[int], seq_len: int = 32
    ):
        super().__init__(dataset, indices, seq_len)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        episode_idx = self.indices[idx]
        length = self.dataset.get_length(episode_idx)
        base_idx = self.dataset.episode_starts[episode_idx].item()

        # Sampling logic
        if length <= self.seq_len:
            start_offset = 0
            actual_len = length
        else:
            start_offset = torch.randint(0, length - self.seq_len + 1, (1,)).item()
            actual_len = self.seq_len

        abs_start = base_idx + start_offset
        abs_end = abs_start + actual_len

        seq_tokens = self.dataset.tokens[abs_start:abs_end]
        seq_actions = self._get_shifted_actions_from_full(
            abs_start, abs_end, start_offset
        )

        # Padding
        if actual_len < self.seq_len:
            pad_len = self.seq_len - actual_len

            # Create padding (on same device/dtype)
            token_pad = torch.zeros(
                pad_len,
                *seq_tokens.shape[1:],
                dtype=seq_tokens.dtype,
                device=seq_tokens.device,
            )
            seq_tokens = torch.cat([seq_tokens, token_pad], dim=0)

            action_pad = torch.zeros(
                pad_len,
                *seq_actions.shape[1:],
                dtype=seq_actions.dtype,
                device=seq_actions.device,
            )
            seq_actions = torch.cat([seq_actions, action_pad], dim=0)

            loss_mask = torch.ones(
                self.seq_len, dtype=torch.bool, device=seq_tokens.device
            )
            loss_mask[actual_len:] = False
        else:
            loss_mask = torch.ones(
                self.seq_len, dtype=torch.bool, device=seq_tokens.device
            )

        return seq_tokens, seq_actions, loss_mask


class LongView(BaseView):
    def __init__(
        self,
        dataset: UnifiedEpisodeDataset,
        indices: list[int],
        seq_len: int = 128,
        warmup_len: int = 32,
    ):
        super().__init__(dataset, indices, seq_len)
        self.warmup_len = warmup_len

        # Verify indices are valid for this seq_len (sanity check)
        # In production, we assume caller filtered correctly.

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        episode_idx = self.indices[idx]
        length = self.dataset.get_length(episode_idx)
        base_idx = self.dataset.episode_starts[episode_idx].item()

        if length < self.seq_len:
            # Should not happen if indices are filtered correctly
            raise ValueError(f"Episode length {length} < seq_len {self.seq_len}")

        max_start = length - self.seq_len
        start_offset = torch.randint(0, max_start + 1, (1,)).item()

        abs_start = base_idx + start_offset
        abs_end = abs_start + self.seq_len

        seq_tokens = self.dataset.tokens[abs_start:abs_end]
        seq_actions = self._get_shifted_actions_from_full(
            abs_start, abs_end, start_offset
        )

        loss_mask = torch.ones(self.seq_len, dtype=torch.bool, device=seq_tokens.device)
        loss_mask[: self.warmup_len] = False

        return seq_tokens, seq_actions, loss_mask
