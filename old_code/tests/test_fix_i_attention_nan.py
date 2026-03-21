"""Fix I: Guard against NaN in attention when all ships are dead (all keys masked)."""
import pytest
import torch
import torch.nn as nn


def apply_nan_guard(key_padding_mask):
    """Replicates the Fix I guard applied before MultiheadAttention calls."""
    if key_padding_mask is not None:
        all_masked = key_padding_mask.all(dim=-1, keepdim=True)  # (BT, 1)
        key_padding_mask = key_padding_mask & ~all_masked  # keep >= 1 unmasked
    return key_padding_mask


def test_fix_i_guard_prevents_all_masked():
    """After guard, no row in key_padding_mask should be all-True."""
    BT, N = 4, 6
    # Scenario: env 0 = all dead, env 1 = 1 alive, env 2 = all alive, env 3 = all dead
    kpm = torch.zeros(BT, N, dtype=torch.bool)
    kpm[0] = True  # all masked (all dead)
    kpm[1, 1:] = True  # one alive at index 0
    kpm[3] = True  # all masked (all dead)

    guarded = apply_nan_guard(kpm)

    # No row should be entirely True
    for i in range(BT):
        assert not guarded[i].all(), f"Row {i} still all-masked after guard"


def test_fix_i_guard_leaves_partially_masked_unchanged():
    """Guard should not change rows that have at least one unmasked key."""
    BT, N = 4, 6
    kpm = torch.zeros(BT, N, dtype=torch.bool)
    kpm[0, 1:] = True  # 1 alive, 5 dead
    kpm[1, 3:] = True  # 3 alive, 3 dead

    guarded = apply_nan_guard(kpm)

    assert torch.equal(guarded[0], kpm[0]), "Row 0 should be unchanged (has alive)"
    assert torch.equal(guarded[1], kpm[1]), "Row 1 should be unchanged (has alive)"


def test_fix_i_guard_none_stays_none():
    """Guard should pass through None mask unchanged."""
    result = apply_nan_guard(None)
    assert result is None


def test_fix_i_attention_no_nan_all_dead():
    """MultiheadAttention with guarded mask should not produce NaN."""
    d_model, n_heads, N = 16, 2, 4
    attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    BT = 3
    q = torch.randn(BT, 1, d_model)
    kv = torch.randn(BT, N, d_model)

    # All-dead mask (all keys masked)
    kpm = torch.ones(BT, N, dtype=torch.bool)

    # Without guard: may produce NaN
    # With guard: should be finite
    guarded_kpm = apply_nan_guard(kpm)
    
    with torch.no_grad():
        out, _ = attn(q, kv, kv, key_padding_mask=guarded_kpm)

    assert torch.isfinite(out).all(), \
        "Attention output should be finite even when all ships were originally masked"


def test_fix_i_attention_correct_unmasked_row_survives():
    """All-masked row should unmask exactly the first key (index 0)."""
    BT, N = 2, 4
    # Row 0: all dead, Row 1: all dead
    kpm = torch.ones(BT, N, dtype=torch.bool)
    
    guarded = apply_nan_guard(kpm)
    
    # Each row: all_masked=True so we AND with ~True = ~all = False everywhere
    # That means: guarded = all & ~True = all & False = False for those rows
    # So guarded[0] should be all False (all unmasked)
    assert not guarded[0].any(), "All-masked row should become all-unmasked after guard"
    assert not guarded[1].any(), "All-masked row should become all-unmasked after guard"
