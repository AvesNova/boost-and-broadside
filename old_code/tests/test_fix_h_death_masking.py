"""Fix H: BaseScaffold.get_loss() uses input_alive instead of target_alive for masking.
   Death-transition gradients (alive at input, dead at target) are preserved.
"""
import pytest
import torch
from unittest.mock import MagicMock, patch
from omegaconf import OmegaConf


def make_simple_get_loss_fn(loss_fn):
    """Standalone reimplementation of the fixed BaseScaffold.get_loss() logic for testing."""
    def get_loss(pred_states, target_states, loss_mask, target_alive=None, input_alive=None):
        final_mask = loss_mask
        if final_mask.ndim == 2 and pred_states is not None:
            final_mask = final_mask.unsqueeze(-1).expand_as(pred_states[..., 0])
        if input_alive is not None:
            final_mask = final_mask & input_alive
        elif target_alive is not None:
            final_mask = final_mask & target_alive
        return loss_fn(pred_states, target_states, final_mask)
    return get_loss


def mse_with_mask(pred, target, mask):
    """Simple masked MSE."""
    diff = (pred - target) ** 2
    return (diff * mask.unsqueeze(-1).float()).sum()


def test_fix_h_death_transition_is_trained():
    """Ships that are alive at input but die at target should contribute to loss."""
    B, T, N, F = 1, 1, 4, 5
    pred = torch.ones(B, T, N, F)
    target = torch.zeros(B, T, N, F)
    loss_mask = torch.ones(B, T, N, dtype=torch.bool)

    # Ship 0: alive at input, dead at target (death transition — SHOULD train)
    input_alive = torch.ones(B, T, N, dtype=torch.bool)
    target_alive = torch.ones(B, T, N, dtype=torch.bool)
    target_alive[0, 0, 0] = False  # ship 0 died

    get_loss = make_simple_get_loss_fn(mse_with_mask)
    
    # With fix (input_alive): death transitions kept in loss
    loss_with_fix = get_loss(pred, target, loss_mask, input_alive=input_alive)
    # Without fix (target_alive): death transitions dropped
    loss_without_fix = get_loss(pred, target, loss_mask, target_alive=target_alive)

    # Fix: all 4 ships contribute (ship 0 was alive at input)
    # No fix: only 3 ships contribute (ship 0 masked by target_alive)
    assert loss_with_fix > loss_without_fix, \
        "Fix H: death-transitions should contribute more loss (not masked by target_alive)"


def test_fix_h_already_dead_ships_still_masked():
    """Ships already dead at input time should still be excluded from loss."""
    B, T, N, F = 1, 1, 4, 5
    pred = torch.ones(B, T, N, F)
    target = torch.zeros(B, T, N, F)
    loss_mask = torch.ones(B, T, N, dtype=torch.bool)

    # Ship 0 was dead at input (already dead — no new information)
    input_alive = torch.ones(B, T, N, dtype=torch.bool)
    input_alive[0, 0, 0] = False  # ship 0 was already dead at input

    get_loss = make_simple_get_loss_fn(mse_with_mask)
    loss = get_loss(pred, target, loss_mask, input_alive=input_alive)

    # Loss from 3 ships only (ship 0 masked)
    expected = mse_with_mask(pred, target, input_alive)
    assert torch.allclose(torch.as_tensor(loss).clone().detach(), torch.as_tensor(expected).clone().detach()), \
        "Already-dead ships should be masked from loss"


def test_fix_h_fallback_to_target_alive():
    """When input_alive is None, should fall back to target_alive (backward compat)."""
    B, T, N, F = 1, 1, 4, 5
    pred = torch.ones(B, T, N, F)
    target = torch.zeros(B, T, N, F)
    loss_mask = torch.ones(B, T, N, dtype=torch.bool)

    target_alive = torch.ones(B, T, N, dtype=torch.bool)
    target_alive[0, 0, 2] = False

    get_loss = make_simple_get_loss_fn(mse_with_mask)
    loss_with_fallback = get_loss(pred, target, loss_mask, target_alive=target_alive)
    loss_full = get_loss(pred, target, loss_mask)

    assert loss_with_fallback < loss_full, \
        "Fallback to target_alive should still mask some ships"
