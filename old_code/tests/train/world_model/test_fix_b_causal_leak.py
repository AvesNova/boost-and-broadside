"""Fix B: Causal leak fix — input_actions is properly offset (prev-action conditioning)."""

import pytest
import torch


def compute_input_actions(actions):
    """Replicates the fixed input_actions logic from trainer._train_step."""
    target_actions = actions[:, :-1]
    zero_action = torch.zeros_like(actions[:, :1])
    input_actions = torch.cat([zero_action, actions[:, :-1]], dim=1)[:, :-1]
    return input_actions, target_actions


def test_input_actions_first_step_is_zero():
    """input_actions[:, 0] must be all zeros (no prev action at t=0)."""
    B, T, N = 2, 8, 4
    actions = torch.randint(0, 3, (B, T, N, 3))
    input_actions, _ = compute_input_actions(actions)

    # First timestep should always be zero
    assert input_actions[:, 0].sum() == 0, (
        "input_actions[:, 0] must be zero (no prev-action before t=0)"
    )


def test_input_actions_is_properly_shifted():
    """input_actions[t] == actions[t-1] for t >= 1 (causal conditioning)."""
    B, T, N = 2, 8, 4
    actions = torch.arange(T).float().view(1, T, 1, 1).expand(B, T, N, 3)
    input_actions, _ = compute_input_actions(actions)

    T_minus_1 = T - 1
    # input_actions has shape (B, T-1, N, 3)
    # input_actions[:, 1:] should equal actions[:, :-2]
    assert input_actions.shape[1] == T_minus_1
    assert torch.allclose(input_actions[:, 1:], actions[:, :-2]), (
        "input_actions[t] should equal actions[t-1] for t>=1"
    )


def test_target_and_input_differ():
    """With the fix, input_actions != target_actions (before fix they were identical)."""
    B, T, N = 2, 8, 4
    actions = torch.randint(1, 7, (B, T, N, 3))  # No zeros in actions
    input_actions, target_actions = compute_input_actions(actions)

    # Because of the zero-prepend + shift, they cannot be equal
    # (input_actions[:, 0] == 0 but target_actions[:, 0] == actions[:, 0] != 0)
    assert not torch.equal(input_actions, target_actions), (
        "After fix, input_actions must differ from target_actions"
    )


def test_seq_length_preserved():
    """Both input_actions and target_actions should have length T-1."""
    B, T, N = 3, 10, 8
    actions = torch.randint(0, 3, (B, T, N, 3))
    input_actions, target_actions = compute_input_actions(actions)
    assert input_actions.shape[1] == T - 1
    assert target_actions.shape[1] == T - 1
