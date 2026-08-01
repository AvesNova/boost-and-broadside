"""Tests for the fixed play-mode preset and keyboard action routing."""

import torch

from boost_and_broadside.modes.interactive import PLAY_ENV_CONFIG, _apply_keyboard_override


def test_play_preset_is_unlimited_one_vs_one_with_four_fields() -> None:
    assert PLAY_ENV_CONFIG.num_ships == 2
    assert PLAY_ENV_CONFIG.num_fields == 4
    assert PLAY_ENV_CONFIG.max_episode_steps is None
    assert not PLAY_ENV_CONFIG.single_team


def test_play_keyboard_controls_team_zero_but_not_null_team_one() -> None:
    action = torch.zeros((1, 2, 3), dtype=torch.int32)
    team_id = torch.tensor([[1, 0]], dtype=torch.int32)
    keyboard = torch.tensor([1, 3, 1], dtype=torch.int32)

    result = _apply_keyboard_override(action, team_id, keyboard, frozenset({0}))

    assert torch.equal(result[0, 0], torch.zeros(3, dtype=torch.int32))
    assert torch.equal(result[0, 1], keyboard)
