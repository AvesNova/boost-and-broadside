"""Tests for world model batch length alternation."""
import pytest
import torch
from src.train.data_loader import (
    AlternatingBatchLengthDataset,
    collate_variable_length_batches,
    create_world_model_data_loader,
)


def create_dummy_data(num_episodes=10, episode_len=200, num_ships=8, token_dim=12, action_dim=6):
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


class TestAlternatingBatchLengthDataset:
    """Tests for AlternatingBatchLengthDataset."""
    
    def test_dataset_creation(self):
        """Test that dataset can be created."""
        data = create_dummy_data()
        tokens = data["team_0"]["tokens"]
        actions = data["team_0"]["actions"]
        episode_lengths = data["episode_lengths"]
        
        dataset = AlternatingBatchLengthDataset(
            tokens,
            actions,
            episode_lengths,
            context_len=96,
            short_batch_len=32,
            long_batch_len=128,
            long_batch_ratio=0.2,
        )
        
        assert len(dataset) > 0
    
    def test_short_batch_length(self):
        """Test that short batch lengths work correctly."""
        data = create_dummy_data()
        tokens = data["team_0"]["tokens"]
        actions = data["team_0"]["actions"]
        episode_lengths = data["episode_lengths"]
        
        # Force short batches by setting long_batch_ratio=0
        dataset = AlternatingBatchLengthDataset(
            tokens,
            actions,
            episode_lengths,
            context_len=96,
            short_batch_len=32,
            long_batch_len=128,
            long_batch_ratio=0.0,  # Always short
        )
        
        # Sample multiple times to verify length
        for _ in range(10):
            batch_tokens, batch_actions = dataset[0]
            assert batch_tokens.shape[0] == 32, f"Expected 32 timesteps, got {batch_tokens.shape[0]}"
            assert batch_actions.shape[0] == 32
    
    def test_long_batch_length(self):
        """Test that long batch lengths work correctly."""
        data = create_dummy_data()
        tokens = data["team_0"]["tokens"]
        actions = data["team_0"]["actions"]
        episode_lengths = data["episode_lengths"]
        
        # Force long batches by setting long_batch_ratio=1
        dataset = AlternatingBatchLengthDataset(
            tokens,
            actions,
            episode_lengths,
            context_len=96,
            short_batch_len=32,
            long_batch_len=128,
            long_batch_ratio=1.0,  # Always long
        )
        
        # Sample multiple times to verify length
        for _ in range(10):
            batch_tokens, batch_actions = dataset[0]
            assert batch_tokens.shape[0] == 128, f"Expected 128 timesteps, got {batch_tokens.shape[0]}"
            assert batch_actions.shape[0] == 128
    
    def test_alternating_lengths(self):
        """Test that both short and long lengths appear when alternating."""
        data = create_dummy_data()
        tokens = data["team_0"]["tokens"]
        actions = data["team_0"]["actions"]
        episode_lengths = data["episode_lengths"]
        
        dataset = AlternatingBatchLengthDataset(
            tokens,
            actions,
            episode_lengths,
            context_len=96,
            short_batch_len=32,
            long_batch_len=128,
            long_batch_ratio=0.5,  # 50/50 mix
        )
        
        # Sample many times and check we get both lengths
        lengths = []
        for _ in range(100):
            batch_tokens, _ = dataset[0]
            lengths.append(batch_tokens.shape[0])
        
        assert 32 in lengths, "Short batch length (32) never appeared"
        assert 128 in lengths, "Long batch length (128) never appeared"


class TestCollateFunction:
    """Tests for collate_variable_length_batches."""
    
    def test_collate_same_length(self):
        """Test collating batches with same length."""
        batch = [
            (torch.randn(32, 8, 12), torch.randn(32, 8, 6)),
            (torch.randn(32, 8, 12), torch.randn(32, 8, 6)),
            (torch.randn(32, 8, 12), torch.randn(32, 8, 6)),
        ]
        
        tokens, actions = collate_variable_length_batches(batch)
        
        assert tokens.shape == (3, 32, 8, 12)
        assert actions.shape == (3, 32, 8, 6)
    
    def test_collate_different_lengths(self):
        """Test collating batches with different lengths."""
        batch = [
            (torch.randn(32, 8, 12), torch.randn(32, 8, 6)),
            (torch.randn(128, 8, 12), torch.randn(128, 8, 6)),
            (torch.randn(64, 8, 12), torch.randn(64, 8, 6)),
        ]
        
        tokens, actions = collate_variable_length_batches(batch)
        
        # Should pad to max length (128)
        assert tokens.shape == (3, 128, 8, 12)
        assert actions.shape == (3, 128, 8, 6)
        
        # Check padding is zeros
        assert torch.all(tokens[0, 32:] == 0)  # First sample padded from 32 to 128
        assert torch.all(tokens[2, 64:] == 0)  # Third sample padded from 64 to 128


class TestDataLoader:
    """Integration tests for data loader with alternating batch lengths."""
    
    def test_dataloader_with_alternating_lengths(self):
        """Test that dataloader works with alternating batch lengths."""
        data = create_dummy_data()
        
        train_loader, val_loader = create_world_model_data_loader(
            data,
            batch_size=4,
            context_len=96,
            validation_split=0.2,
            num_workers=0,  # Use 0 for testing
            use_alternating_lengths=True,
            short_batch_len=32,
            long_batch_len=128,
            long_batch_ratio=0.5,
        )
        
        # Get a few batches
        for i, (tokens, actions) in enumerate(train_loader):
            if i >= 5:  # Just test first 5 batches
                break
            
            # Check shapes
            assert tokens.ndim == 4  # (B, T, N, F)
            assert actions.ndim == 4  # (B, T, N, A)
            assert tokens.shape[0] <= 4  # Batch size
            assert tokens.shape[2] == 8  # Num ships
            assert tokens.shape[3] == 12  # Token dim
            assert actions.shape[3] == 6  # Action dim
            
            # Length should be either 32 or 128 (or padded to match within batch)
            assert tokens.shape[1] in [32, 128], f"Unexpected length: {tokens.shape[1]}"
    
    def test_both_lengths_appear_in_training(self):
        """Test that both short and long batches appear during training."""
        data = create_dummy_data(num_episodes=50)  # More episodes for better sampling
        
        train_loader, _ = create_world_model_data_loader(
            data,
            batch_size=4,
            context_len=96,
            validation_split=0.1,
            num_workers=0,
            use_alternating_lengths=True,
            short_batch_len=32,
            long_batch_len=128,
            long_batch_ratio=0.3,
        )
        
        # Collect batch lengths
        batch_lengths = []
        for tokens, _ in train_loader:
            batch_lengths.append(tokens.shape[1])
        
        # Both lengths should appear
        assert 32 in batch_lengths, "Short batches (32) never appeared in training"
        assert 128 in batch_lengths, "Long batches (128) never appeared in training"
