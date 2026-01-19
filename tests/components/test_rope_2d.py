"""
Unit tests for Continuous 2D RoPE implementation.
"""

import torch

from agents.rope import Continuous2DRotaryEmbedding


def test_2d_rope_initialization():
    """Test that 2D RoPE initializes correctly."""
    dim = 64
    rope = Continuous2DRotaryEmbedding(dim=dim, base=10000.0)
    assert rope.dim == dim
    assert rope.inv_freq.shape == (dim // 2,)
    # Should be on CPU by default
    assert rope.inv_freq.device.type == "cpu"


def test_2d_rope_forward_shape():
    """Test that 2D RoPE preserves tensor shapes."""
    dim = 64
    rope = Continuous2DRotaryEmbedding(dim=dim)

    batch_size = 2
    n_ships = 4
    n_heads = 4
    head_dim = dim  # 64

    # Input: (Batch, N, Heads, Dim)
    # Actually checking dimensions:
    # In WorldModel, we pass (Batch, N, Heads, HeadDim) ?
    # Let's check usage in world_model.py
    # key = key.view(B*T, N, heads, head_dim).transpose(1, 2) -> (B*T, heads, N, head_dim)
    # Then Before RoPE: query = query.transpose(1, 2) -> (B*T, N, heads, head_dim)

    # So input is (Batch, N, Heads, HeadDim)

    q = torch.randn(batch_size, n_ships, n_heads, dim)
    k = torch.randn(batch_size, n_ships, n_heads, dim)

    # Coords: (Batch, N, 2)
    coords = torch.rand(batch_size, n_ships, 2)

    q_rot, k_rot = rope(q, k, coords)

    assert q_rot.shape == q.shape
    assert k_rot.shape == k.shape


def test_2d_rope_translation_invariance():
    """
    Test that relative attention scores are invariant to translation.

    If we shift all ships by the same amount, their relative positions remain same.
    The dot product (Attention Score) of their rotated embeddings should remain same.
    Arguments:
        q_i . k_j  approx  q_rot_i . k_rot_j
    """
    dim = 64
    rope = Continuous2DRotaryEmbedding(dim=dim)

    # Single batch, 2 ships, 1 head
    batch_size = 1
    n_ships = 2
    n_heads = 1

    q = torch.randn(batch_size, n_ships, n_heads, dim)
    k = torch.randn(batch_size, n_ships, n_heads, dim)

    # Scenario A: Original Positions
    coords_a = torch.tensor(
        [[[0.1, 0.1], [0.2, 0.2]]], dtype=torch.float32
    )  # (1, 2, 2)

    q_rot_a, k_rot_a = rope(q, k, coords_a)

    # Compute dot product between Ship 0 and Ship 1
    # q[0] dot k[1]
    attn_score_a = torch.sum(q_rot_a[0, 0, 0] * k_rot_a[0, 1, 0])

    # Scenario B: Shifted Positions (Translate by +0.5)
    coords_b = coords_a + 0.5

    q_rot_b, k_rot_b = rope(q, k, coords_b)

    # Compute dot product
    attn_score_b = torch.sum(q_rot_b[0, 0, 0] * k_rot_b[0, 1, 0])

    # They should be very close
    # Note: FP error might be slightly higher due to trig functions
    assert torch.allclose(attn_score_a, attn_score_b, atol=1e-4)


def test_2d_rope_rotation():
    """Verify that different positions cause different outputs."""
    dim = 64
    rope = Continuous2DRotaryEmbedding(dim=dim)

    q = torch.randn(1, 1, 1, dim)
    k = torch.randn(1, 1, 1, dim)

    coords_1 = torch.tensor([[[0.0, 0.0]]])
    coords_2 = torch.tensor([[[0.1, 0.1]]])

    q1, _ = rope(q, k, coords_1)
    q2, _ = rope(q, k, coords_2)

    assert not torch.allclose(q1, q2)
