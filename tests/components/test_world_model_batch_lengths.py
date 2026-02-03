
"""Tests for world model batch length alternation."""

import torch
import h5py
from train.unified_dataset import UnifiedEpisodeDataset, ShortView, LongView
from train.data_loader import create_unified_data_loaders


def save_dummy_data_to_h5(
    path, num_episodes=10, episode_len=200, num_ships=8, token_dim=9, action_dim=3
):
    """Create dummy data and save to HDF5."""
    total_timesteps = num_episodes * episode_len

    # Granular features
    # NOTE: UnifiedEpisodeDataset hardcodes assembly to 15 dims
    pos = torch.randn(total_timesteps, num_ships, 2)
    vel = torch.randn(total_timesteps, num_ships, 2)
    health = torch.rand(total_timesteps, num_ships)
    power = torch.rand(total_timesteps, num_ships)
    attitude = torch.randn(total_timesteps, num_ships, 2)
    ang_vel = torch.randn(total_timesteps, num_ships)
    is_shooting = torch.randint(0, 2, (total_timesteps, num_ships)).float()
    team_ids = torch.randint(0, 2, (total_timesteps, num_ships)).float()

    actions = torch.randn(total_timesteps, num_ships, action_dim)
    episode_lengths = torch.tensor([episode_len] * num_episodes, dtype=torch.int64)
    returns = torch.zeros(total_timesteps)
    action_masks = torch.ones(total_timesteps, num_ships)

    # New fields
    agent_skills = torch.rand(total_timesteps)

    with h5py.File(path, "w") as f:
        f.create_dataset("position", data=pos.numpy())
        f.create_dataset("velocity", data=vel.numpy())
        f.create_dataset("health", data=health.numpy())
        f.create_dataset("power", data=power.numpy())
        f.create_dataset("attitude", data=attitude.numpy())
        f.create_dataset("ang_vel", data=ang_vel.numpy())
        f.create_dataset("is_shooting", data=is_shooting.numpy())
        f.create_dataset("team_ids", data=team_ids.numpy())

        f.create_dataset("actions", data=actions.numpy())
        f.create_dataset("returns", data=returns.numpy())
        f.create_dataset("action_masks", data=action_masks.numpy())
        f.create_dataset("episode_lengths", data=episode_lengths.numpy())
        f.create_dataset("agent_skills", data=agent_skills.numpy())
        f.attrs["token_dim"] = token_dim

    return episode_lengths


class TestShortView:
    """Tests for ShortView."""

    def test_view_creation(self, tmp_path):
        """Test that view can be created."""
        h5_path = tmp_path / "test_data.h5"
        episode_lengths = save_dummy_data_to_h5(h5_path)

        dataset = UnifiedEpisodeDataset(str(h5_path))
        view = ShortView(dataset, list(range(len(episode_lengths))), seq_len=32)

        assert len(view) > 0

    def test_batch_structure(self, tmp_path):
        """Test that batches have correct structure and padding."""
        h5_path = tmp_path / "test_data.h5"
        episode_lengths = save_dummy_data_to_h5(h5_path, num_episodes=1, episode_len=20)

        dataset = UnifiedEpisodeDataset(str(h5_path))
        view = ShortView(dataset, list(range(len(episode_lengths))), seq_len=32)

        batch_tokens, batch_input_actions, batch_target_actions, batch_returns, loss_mask, batch_masks, _, _, _ = view[
            0
        ]

        assert batch_tokens.shape[0] == 32
        assert batch_input_actions.shape[0] == 32
        assert loss_mask.shape[0] == 32

        # Check padding (last 12 should be padded)
        # Note: 20 unmasked, 12 padded
        assert not loss_mask[20:].any()
        assert loss_mask[:20].all()


class TestLongView:
    """Tests for LongView."""

    def test_view_creation(self, tmp_path):
        """Test that view can be created."""
        h5_path = tmp_path / "test_data.h5"
        episode_lengths = save_dummy_data_to_h5(h5_path)

        dataset = UnifiedEpisodeDataset(str(h5_path))
        # Only use episodes longer than seq_len
        valid_indices = [i for i, l in enumerate(episode_lengths) if l >= 128]

        view = LongView(dataset, valid_indices, seq_len=128, warmup_len=32)

        assert len(view) > 0

    def test_batch_structure(self, tmp_path):
        """Test that batches have correct structure and warmup masking."""
        h5_path = tmp_path / "test_data.h5"
        episode_lengths = save_dummy_data_to_h5(
            h5_path, num_episodes=1, episode_len=200
        )

        dataset = UnifiedEpisodeDataset(str(h5_path))
        view = LongView(
            dataset, list(range(len(episode_lengths))), seq_len=128, warmup_len=32
        )

        batch_tokens, batch_input_actions, batch_target_actions, batch_returns, loss_mask, batch_masks, _, _, _ = view[
            0
        ]

        assert batch_tokens.shape[0] == 128
        assert batch_input_actions.shape[0] == 128
        assert loss_mask.shape[0] == 128

        # Check warmup masking
        assert not loss_mask[:32].any()
        assert loss_mask[32:].all()


class TestUnifiedDataLoaders:
    """Integration tests for unified data loaders."""

    def test_loader_creation(self, tmp_path):
        """Test that loaders are created correctly."""
        h5_path = tmp_path / "test_data.h5"
        save_dummy_data_to_h5(h5_path, num_episodes=40, episode_len=200)

        ts, tl, vs, vl = create_unified_data_loaders(
            str(h5_path),
            short_batch_size=4,
            long_batch_size=2,
            short_batch_len=32,
            long_batch_len=128,
            batch_ratio=4,
            validation_split=0.2,
            num_workers=0,
        )

        assert len(ts) > 0
        assert len(tl) > 0

        # Verify batch shapes
        s_batch = next(iter(ts))
        s_tokens = s_batch["states"]
        assert s_tokens.shape == (4, 32, 8, 9)

        l_batch = next(iter(tl))
        l_tokens = l_batch["states"]
        assert l_tokens.shape == (2, 128, 8, 9)
