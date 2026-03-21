"""Fix G: pairwise target uses absolute pos at t+1, not velocity deltas."""
import pytest
import torch


def compute_pairwise_target_old(d_pos):
    """Old (buggy): uses delta positions."""
    d_pos_i = d_pos.unsqueeze(3)
    d_pos_j = d_pos.unsqueeze(2)
    return d_pos_j - d_pos_i


def compute_pairwise_target_new(pos_next):
    """New (fixed): uses absolute position at t+1."""
    pos_next_i = pos_next.unsqueeze(3)
    pos_next_j = pos_next.unsqueeze(2)
    return pos_next_j - pos_next_i


def test_pairwise_target_correct_formula():
    """target[b,t,i,j] == pos_next[b,t,j] - pos_next[b,t,i]."""
    B, T, N = 2, 4, 4
    pos_next = torch.randn(B, T, N, 2)
    
    target = compute_pairwise_target_new(pos_next)
    
    # Manually verify a few entries
    for b in range(B):
        for t in range(T):
            for i in range(N):
                for j in range(N):
                    expected = pos_next[b, t, j] - pos_next[b, t, i]
                    assert torch.allclose(target[b, t, i, j], expected), \
                        f"Mismatch at b={b},t={t},i={i},j={j}"


def test_pairwise_target_antidiagonal():
    """Swapping i,j should negate: target[i,j] == -target[j,i]."""
    B, T, N = 2, 3, 4
    pos_next = torch.randn(B, T, N, 2)
    target = compute_pairwise_target_new(pos_next)
    
    # target[b,t,i,j] + target[b,t,j,i] should be zero
    assert torch.allclose(target + target.permute(0, 1, 3, 2, 4), 
                          torch.zeros_like(target), atol=1e-6)


def test_pairwise_diagonal_is_zero():
    """Self-pair (i==i) should be zero displacement."""
    B, T, N = 2, 3, 4
    pos_next = torch.randn(B, T, N, 2)
    target = compute_pairwise_target_new(pos_next)
    
    for i in range(N):
        assert torch.allclose(target[:, :, i, i], torch.zeros(B, T, 2), atol=1e-6), \
            f"Diagonal i={i} should be zero"


def test_old_and_new_differ_when_nonzero_current_pos():
    """Old formula (delta-based) differs from new (absolute) when t>0 positions differ."""
    B, T, N = 2, 4, 4
    # pos_curr is nonzero so d_pos != pos_next
    pos_curr = torch.randn(B, T, N, 2)
    pos_next = torch.randn(B, T, N, 2)
    d_pos = pos_next - pos_curr
    
    old_target = compute_pairwise_target_old(d_pos)
    new_target = compute_pairwise_target_new(pos_next)
    
    assert not torch.allclose(old_target, new_target), \
        "Old and new formulas should differ when pos_curr != 0"
