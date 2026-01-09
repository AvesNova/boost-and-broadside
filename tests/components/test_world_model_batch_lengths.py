"""Tests for world model batch length alternation."""

import torch
import pytest
from train.unified_dataset import UnifiedEpisodeDataset, ShortView, LongView
from train.data_loader import create_unified_data_loaders


def create_dummy_data(
    num_episodes=10, episode_len=200, num_ships=8, token_dim=12, action_dim=6
):
    """Create dummy data for testing."""
    total_timesteps = num_episodes * episode_len

    tokens = torch.randn(total_timesteps, num_ships, token_dim)
    actions = torch.randn(total_timesteps, num_ships, action_dim)
    episode_lengths = torch.tensor([episode_len] * num_episodes, dtype=torch.int64)

    data = {
        "team_0": {
            "tokens": tokens,
            "actions": actions,
            "rewards": torch.zeros(total_timesteps),
        },
        "team_1": {
            "tokens": tokens.clone(),
            "actions": actions.clone(),
            "rewards": torch.zeros(total_timesteps),
        },
        "episode_lengths": episode_lengths,
        "episode_ids": torch.arange(total_timesteps) // episode_len,
    }

    return data


class TestShortView:
    """Tests for ShortView."""

    def test_view_creation(self):
        """Test that view can be created."""
        data = create_dummy_data()
        tokens = data["team_0"]["tokens"]
        actions = data["team_0"]["actions"]
        episode_lengths = data["episode_lengths"]

        dataset = UnifiedEpisodeDataset(tokens, actions, episode_lengths)
        view = ShortView(dataset, list(range(len(episode_lengths))), seq_len=32)

        assert len(view) > 0

    def test_batch_structure(self):
        """Test that batches have correct structure and padding."""
        data = create_dummy_data(num_episodes=1, episode_len=20)  # Shorter than seq_len
        tokens = data["team_0"]["tokens"]
        actions = data["team_0"]["actions"]
        episode_lengths = data["episode_lengths"]

        dataset = UnifiedEpisodeDataset(tokens, actions, episode_lengths)
        view = ShortView(dataset, list(range(len(episode_lengths))), seq_len=32)

        batch_tokens, batch_actions, loss_mask = view[0]

        assert batch_tokens.shape[0] == 32
        assert batch_actions.shape[0] == 32
        assert loss_mask.shape[0] == 32

        # Check padding (last 12 should be padded)
        # Note: 20 unmasked, 12 padded
        assert not loss_mask[20:].any()
        assert loss_mask[:20].all()


class TestLongView:
    """Tests for LongView."""

    def test_view_creation(self):
        """Test that view can be created."""
        data = create_dummy_data()
        tokens = data["team_0"]["tokens"]
        actions = data["team_0"]["actions"]
        episode_lengths = data["episode_lengths"]

        dataset = UnifiedEpisodeDataset(tokens, actions, episode_lengths)
        # Only use episodes longer than seq_len
        valid_indices = [
            i for i, l in enumerate(episode_lengths) if l >= 128
        ]
        
        view = LongView(
            dataset, valid_indices, seq_len=128, warmup_len=32
        )

        assert len(view) > 0

    def test_batch_structure(self):
        """Test that batches have correct structure and warmup masking."""
        data = create_dummy_data(num_episodes=1, episode_len=200)
        tokens = data["team_0"]["tokens"]
        actions = data["team_0"]["actions"]
        episode_lengths = data["episode_lengths"]

        dataset = UnifiedEpisodeDataset(tokens, actions, episode_lengths)
        view = LongView(
            dataset, list(range(len(episode_lengths))), seq_len=128, warmup_len=32
        )

        batch_tokens, batch_actions, loss_mask = view[0]

        assert batch_tokens.shape[0] == 128
        assert batch_actions.shape[0] == 128
        assert loss_mask.shape[0] == 128

        # Check warmup masking
        assert not loss_mask[:32].any()
        assert loss_mask[32:].all()


class TestUnifiedDataLoaders:
    """Integration tests for unified data loaders."""

    def test_loader_creation(self):
        """Test that loaders are created correctly."""
        data = create_dummy_data(num_episodes=40, episode_len=200)

        ts, tl, vs, vl = create_unified_data_loaders(
            data,
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
        s_tokens, s_actions, s_mask = next(iter(ts))
        assert s_tokens.shape == (4, 32, 8, 12)

        l_tokens, l_actions, l_mask = next(iter(tl))
        assert l_tokens.shape == (2, 128, 8, 12)
