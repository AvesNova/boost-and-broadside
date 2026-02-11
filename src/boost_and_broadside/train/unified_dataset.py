import torch
from torch.utils.data import Dataset
import logging
import h5py
from boost_and_broadside.core.constants import (
    NORM_VELOCITY,
    NORM_ANGULAR_VELOCITY,
    NORM_HEALTH,
    NORM_POWER,
    StateFeature,
    STATE_DIM,
)



log = logging.getLogger(__name__)


class UnifiedEpisodeDataset:
    """
    Central storage for all episode data, backed by HDF5.
    Avoids duplicating large tensors in memory.
    Views (ShortView, LongView) will reference this dataset.
    """

    def __init__(self, h5_path: str, world_size: tuple[float, float] = (1024.0, 1024.0)):
        self.h5_path = h5_path
        self.world_size = world_size
        self._h5_file = None

        # Load metadata immediately (lightweight)
        # We need episode_lengths to build indices
        with h5py.File(h5_path, "r") as f:
            self.episode_lengths = torch.from_numpy(f["episode_lengths"][:])
            self.available_keys = set(f.keys())

        # Precompute start indices for O(1) access
        self.episode_starts = torch.zeros_like(self.episode_lengths)
        self.episode_starts[1:] = torch.cumsum(self.episode_lengths[:-1], dim=0)
        
        # Calculate total timesteps
        if len(self.episode_lengths) > 0:
            self.total_timesteps = self.episode_starts[-1].item() + self.episode_lengths[-1].item()
        else:
            self.total_timesteps = 0
            
    @property
    def num_encodings(self):
        return self.total_timesteps

    def has_dataset(self, name: str) -> bool:
        return name in self.available_keys

    @property
    def h5_file(self):
        # Lazy loading to support multiprocessing (pickling)
        if self._h5_file is None:
            self._h5_file = h5py.File(self.h5_path, "r", swmr=True, libver="latest")
        return self._h5_file

    def get_length(self, episode_idx: int) -> int:
        return self.episode_lengths[episode_idx].item()

    def get_episode_mean_skill(self, episode_idx: int) -> float:
        """
        Get the mean skill of agents in this episode.
        Returns 0.0 if agent_skills is not present.
        """
        if "agent_skills" not in self.available_keys:
            return 1.0 # Default to high skill (include everything if no skills found)
            
        start = self.episode_starts[episode_idx].item()
        # Just sample the first timestep's skill to be fast
        # Assuming skill is constant per episode or close enough
        # Shape: (MaxShips,) or (N, MaxShips)
        # We read just 1 timestep
        skills = self.h5_file["agent_skills"][start:start+1] 
        return float(skills.mean())

    def get_slice(self, dataset_name: str, start: int, end: int) -> torch.Tensor:
        """Helper to slice from HDF5 and convert to Tensor."""
        if dataset_name == "tokens":
            return self._assemble_tokens(start, end)
        
        data = self.h5_file[dataset_name][start:end]
        return torch.from_numpy(data)

    def _assemble_tokens(self, start: int, end: int) -> torch.Tensor:
        """Assembles the monolithic token tensor from granular features."""
        # Load components (cast to float32 immediately then to bf16)
        # We access h5_file directly to avoid recursion
        f = self.h5_file
        
        # Features with shapes (T, N, 2) or (T, N)
        # We DO NOT load position here anymore for tokens
        vel = torch.from_numpy(f["velocity"][start:end]).float()
        health = torch.from_numpy(f["health"][start:end]).float()
        power = torch.from_numpy(f["power"][start:end]).float()
        att = torch.from_numpy(f["attitude"][start:end]).float()
        ang = torch.from_numpy(f["ang_vel"][start:end]).float()
        shoot = torch.from_numpy(f["is_shooting"][start:end]).float()
        team = torch.from_numpy(f["team_ids"][start:end]).float()
        
        # Allocate tokens (T, N, STATE_DIM) - bf16
        T, N, _ = vel.shape
        tokens = torch.zeros((T, N, STATE_DIM), dtype=torch.bfloat16)
        
        # 0: Health
        tokens[..., StateFeature.HEALTH] = health / NORM_HEALTH
        # 1: Power
        tokens[..., StateFeature.POWER] = power / NORM_POWER
        
        # 2-3: Vel
        tokens[..., StateFeature.VX] = vel[..., 0] / NORM_VELOCITY
        tokens[..., StateFeature.VY] = vel[..., 1] / NORM_VELOCITY
        
        # 4: Ang Vel
        tokens[..., StateFeature.ANG_VEL] = ang / NORM_ANGULAR_VELOCITY
        
        return tokens

        
    def get_cross_episode_slice(self, dataset_name: str, global_start: int, length: int) -> torch.Tensor:
        """
        Slice a dataset effectively ignoring episode boundaries.
        The HDF5 datasets (tokens, actions, etc) are already concatenated.
        So we just read [global_start : global_start + length].
        
        However, the HDF5 file might be split or we conceptually think of it as episodes.
        But physically, our h5 structure described in README matches: "All datasets are aligned along the first dimension (Total Timesteps)"
        So simple slicing works! 
        
        We just need to check bounds.
        """
        # Bounds check
        if dataset_name == "tokens":
            total_len = self.total_timesteps
        else:
            total_len = self.h5_file[dataset_name].shape[0]
            
        if global_start + length > total_len:
            # Wrap around or pad?
            # For simplicity in this project (massive data), we often just clip or wrap.
            # But "Continuous Stream" usually implies we just stop or wrap.
            # Let's Implement WRAPPING for infinite streaming if we hit the end.
            
            remaining = total_len - global_start
            part1 = self.get_slice(dataset_name, global_start, total_len)
            part2 = self.get_slice(dataset_name, 0, length - remaining)
            return torch.cat([part1, part2], dim=0)
            
        return self.get_slice(dataset_name, global_start, global_start + length)

    def __del__(self):
        if self._h5_file is not None:
            self._h5_file.close()


class BaseView(Dataset):
    def __init__(
        self, dataset: UnifiedEpisodeDataset, indices: list[int], seq_len: int
    ):
        self.dataset = dataset
        self.indices = indices
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.indices)

    def _get_shifted_actions_from_full(
        self, abs_start: int, abs_end: int, start_offset: int
    ) -> torch.Tensor:
        """
        Helper to extract input actions (SHIFTED right).
        Input at time t is Action_{t-1}.
        """
        if start_offset == 0:
            # First action is 0, then actions[0...T-1]
            # Slicing end-1 from HDF5
            data_actions = self.dataset.get_slice("actions", abs_start, abs_end - 1)

            # Create zeros with same shape (except dim 0 is 1)
            zeros = torch.zeros(1, *data_actions.shape[1:], dtype=data_actions.dtype)
            return torch.cat([zeros, data_actions], dim=0)
        else:
            # Previous action exists
            return self.dataset.get_slice("actions", abs_start - 1, abs_end - 1)
            
    def _get_target_actions(self, abs_start: int, abs_end: int) -> torch.Tensor:
        """
        Helper to extract target actions (Current timestep).
        Target at time t is ExpertAction_{t}.
        """
        if self.dataset.has_dataset("expert_actions"):
            return self.dataset.get_slice("expert_actions", abs_start, abs_end)
        else:
            # Fallback to standard actions if expert actions not available
            return self.dataset.get_slice("actions", abs_start, abs_end)

    def _get_shifted_masks_from_full(
        self, abs_start: int, abs_end: int, start_offset: int
    ) -> torch.Tensor:
        """Helper to extract shifted action masks from the main dataset."""
        if start_offset == 0:
            # First action is 0 (expert/dummy), then masks[0...T-1]
            data_masks = self.dataset.get_slice("action_masks", abs_start, abs_end - 1)

            # Prepend 1.0 (Expert) for the zero-action at t=0
            ones = torch.ones(1, *data_masks.shape[1:], dtype=data_masks.dtype)
            return torch.cat([ones, data_masks], dim=0)
        else:
            # Previous action exists
            return self.dataset.get_slice("action_masks", abs_start - 1, abs_end - 1)


class ShortView(BaseView):
    def __init__(
        self, dataset: UnifiedEpisodeDataset, indices: list[int], seq_len: int = 32
    ):
        super().__init__(dataset, indices, seq_len)

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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

        # Slice using helper
        seq_tokens = self.dataset.get_slice("tokens", abs_start, abs_end)
        seq_pos = self.dataset.get_slice("position", abs_start, abs_end)
        seq_input_actions = self._get_shifted_actions_from_full(
            abs_start, abs_end, start_offset
        )
        seq_target_actions = self._get_target_actions(abs_start, abs_end)
        
        # Convert to long if they are uint8 (byte)
        if seq_input_actions.dtype == torch.uint8:
            seq_input_actions = seq_input_actions.to(torch.int64)
        if seq_target_actions.dtype == torch.uint8:
            seq_target_actions = seq_target_actions.to(torch.int64)
        
        seq_masks = self._get_shifted_masks_from_full(abs_start, abs_end, start_offset)
        seq_masks = self._get_shifted_masks_from_full(abs_start, abs_end, start_offset)
        seq_returns = self.dataset.get_slice("returns", abs_start, abs_end)
        seq_rewards = self.dataset.get_slice("rewards", abs_start, abs_end)
        
        # Optional Features


        if self.dataset.has_dataset("agent_skills"):
            seq_skills = self.dataset.get_slice("agent_skills", abs_start, abs_end)
        else:
            seq_skills = torch.ones(actual_len, dtype=torch.float32)

        if self.dataset.has_dataset("team_ids"):
            seq_team_ids = self.dataset.get_slice("team_ids", abs_start, abs_end)
        else:
            seq_team_ids = torch.zeros(actual_len, dtype=torch.int64)

        # Padding
        if actual_len < self.seq_len:
            pad_len = self.seq_len - actual_len

            # Create padding (on cpu, as data is on cpu)
            token_pad = torch.zeros(
                pad_len, *seq_tokens.shape[1:], dtype=seq_tokens.dtype
            )
            seq_tokens = torch.cat([seq_tokens, token_pad], dim=0)

            pos_pad = torch.zeros(
                pad_len, *seq_pos.shape[1:], dtype=seq_pos.dtype
            )
            seq_pos = torch.cat([seq_pos, pos_pad], dim=0)

            action_pad = torch.zeros(
                pad_len, *seq_input_actions.shape[1:], dtype=seq_input_actions.dtype
            )
            seq_input_actions = torch.cat([seq_input_actions, action_pad], dim=0)
            
            target_action_pad = torch.zeros(
                pad_len, *seq_target_actions.shape[1:], dtype=seq_target_actions.dtype
            )
            seq_target_actions = torch.cat([seq_target_actions, target_action_pad], dim=0)

            # Pad masks
            mask_pad = torch.ones(pad_len, *seq_masks.shape[1:], dtype=seq_masks.dtype)
            seq_masks = torch.cat([seq_masks, mask_pad], dim=0)

            return_pad = torch.zeros(
                pad_len, *seq_returns.shape[1:], dtype=seq_returns.dtype
            )
            seq_returns = torch.cat([seq_returns, return_pad], dim=0)
            
            reward_pad = torch.zeros(
                pad_len, *seq_rewards.shape[1:], dtype=seq_rewards.dtype
            )
            seq_rewards = torch.cat([seq_rewards, reward_pad], dim=0)
            
            # Pad skills (pad with 1.0 - expert)
            skill_pad = torch.ones(
                pad_len, *seq_skills.shape[1:], dtype=seq_skills.dtype
            )
            seq_skills = torch.cat([seq_skills, skill_pad], dim=0)

            # Pad team_ids
            tid_pad = torch.zeros(
                pad_len, *seq_team_ids.shape[1:], dtype=seq_team_ids.dtype
            )
            seq_team_ids = torch.cat([seq_team_ids, tid_pad], dim=0)

            loss_mask = torch.ones(self.seq_len, dtype=torch.bool)
            loss_mask[actual_len:] = False
        else:
            loss_mask = torch.ones(self.seq_len, dtype=torch.bool)

        return (
            seq_tokens,
            seq_input_actions,
            seq_target_actions, 
            seq_returns,
            seq_rewards,
            loss_mask,
            seq_masks,
            seq_skills,
            seq_team_ids,
            seq_pos,
        )


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

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        episode_idx = self.indices[idx]
        length = self.dataset.get_length(episode_idx)
        base_idx = self.dataset.episode_starts[episode_idx].item()

        if length < self.seq_len:
            raise ValueError(f"Episode length {length} < seq_len {self.seq_len}")

        max_start = length - self.seq_len
        start_offset = torch.randint(0, max_start + 1, (1,)).item()

        abs_start = base_idx + start_offset
        abs_end = abs_start + self.seq_len

        seq_tokens = self.dataset.get_slice("tokens", abs_start, abs_end)
        seq_pos = self.dataset.get_slice("position", abs_start, abs_end)
        seq_input_actions = self._get_shifted_actions_from_full(
            abs_start, abs_end, start_offset
        )
        seq_target_actions = self._get_target_actions(abs_start, abs_end)
        
        # Convert to long
        if seq_input_actions.dtype == torch.uint8:
            seq_input_actions = seq_input_actions.to(torch.int64)
        if seq_target_actions.dtype == torch.uint8:
            seq_target_actions = seq_target_actions.to(torch.int64)

        seq_masks = self._get_shifted_masks_from_full(abs_start, abs_end, start_offset)
        seq_returns = self.dataset.get_slice("returns", abs_start, abs_end)
        seq_rewards = self.dataset.get_slice("rewards", abs_start, abs_end)
        


        if self.dataset.has_dataset("agent_skills"):
            seq_skills = self.dataset.get_slice("agent_skills", abs_start, abs_end)
        else:
            seq_skills = torch.ones(self.seq_len, dtype=torch.float32)

        if self.dataset.has_dataset("team_ids"):
            seq_team_ids = self.dataset.get_slice("team_ids", abs_start, abs_end)
        else:
            seq_team_ids = torch.zeros(self.seq_len, dtype=torch.int64)

        loss_mask = torch.ones(self.seq_len, dtype=torch.bool)
        loss_mask[: self.warmup_len] = False

        return (
            seq_tokens,
            seq_input_actions,
            seq_target_actions,
            seq_returns,
            seq_rewards,
            loss_mask,
            seq_masks,
            seq_skills,
            seq_team_ids,
            seq_pos,
        )

