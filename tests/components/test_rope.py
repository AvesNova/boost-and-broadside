"""
Unit tests for RoPE (Rotary Position Embedding) implementation.
"""
import torch
import pytest

from src.agents.rope import RotaryPositionEmbedding


def test_rope_initialization():
    """Test that RoPE initializes correctly."""
    rope = RotaryPositionEmbedding(dim=64, max_seq_len=128, base=10000.0)
    assert rope.dim == 64
    assert rope.max_seq_len == 128
    assert rope.base == 10000.0
    assert rope.cos_cached.shape == (128, 32)  # (max_seq_len, dim // 2)
    assert rope.sin_cached.shape == (128, 32)


def test_rope_invalid_dim():
    """Test that RoPE raises error for odd dimensions."""
    with pytest.raises(ValueError, match="must be even"):
        RotaryPositionEmbedding(dim=63)


def test_rope_forward_shape():
    """Test that RoPE preserves tensor shapes."""
    rope = RotaryPositionEmbedding(dim=64, max_seq_len=128)
    
    batch_size = 4
    seq_len = 32
    dim = 64
    
    q = torch.randn(batch_size, seq_len, dim)
    k = torch.randn(batch_size, seq_len, dim)
    position_ids = torch.arange(seq_len)
    
    q_rot, k_rot = rope(q, k, position_ids)
    
    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_rope_with_position_offset():
    """Test RoPE with position offset for KV caching."""
    rope = RotaryPositionEmbedding(dim=64, max_seq_len=128)
    
    batch_size = 4
    seq_len = 16
    dim = 64
    offset = 32
    
    q = torch.randn(batch_size, seq_len, dim)
    k = torch.randn(batch_size, seq_len, dim)
    position_ids = torch.arange(offset, offset + seq_len)
    
    q_rot, k_rot = rope(q, k, position_ids)
    
    assert q_rot.shape == (batch_size, seq_len, dim)
    assert k_rot.shape == (batch_size, seq_len, dim)


def test_rope_rotation_property():
    """Test that RoPE actually rotates the embeddings (not identity)."""
    rope = RotaryPositionEmbedding(dim=64, max_seq_len=128)
    
    batch_size = 2
    seq_len = 8
    dim = 64
    
    q = torch.randn(batch_size, seq_len, dim)
    k = torch.randn(batch_size, seq_len, dim)
    position_ids = torch.arange(seq_len)
    
    q_rot, k_rot = rope(q, k, position_ids)
    
    # Rotated tensors should not be identical to originals
    assert not torch.allclose(q, q_rot)
    assert not torch.allclose(k, k_rot)


def test_rope_cache_extension():
    """Test that RoPE extends cache when needed."""
    rope = RotaryPositionEmbedding(dim=64, max_seq_len=64)
    
    # Request longer sequence than initial cache
    batch_size = 2
    seq_len = 96
    dim = 64
    
    q = torch.randn(batch_size, seq_len, dim)
    k = torch.randn(batch_size, seq_len, dim)
    position_ids = torch.arange(seq_len)
    
    q_rot, k_rot = rope(q, k, position_ids)
    
    # Cache should be extended
    assert rope.cos_cached.shape[0] >= seq_len
    assert rope.sin_cached.shape[0] >= seq_len
    assert q_rot.shape == (batch_size, seq_len, dim)


def test_rope_different_positions():
    """Test that different positions produce different rotations."""
    rope = RotaryPositionEmbedding(dim=64, max_seq_len=128)
    
    batch_size = 1
    seq_len = 1
    dim = 64
    
    q = torch.randn(batch_size, seq_len, dim)
    k = torch.randn(batch_size, seq_len, dim)
    
    # Apply RoPE at position 0
    position_ids_0 = torch.tensor([0])
    q_rot_0, k_rot_0 = rope(q, k, position_ids_0)
    
    # Apply RoPE at position 10
    position_ids_10 = torch.tensor([10])
    q_rot_10, k_rot_10 = rope(q, k, position_ids_10)
    
    # Different positions should produce different rotations
    assert not torch.allclose(q_rot_0, q_rot_10)
    assert not torch.allclose(k_rot_0, k_rot_10)
