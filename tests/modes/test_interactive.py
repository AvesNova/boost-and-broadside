"""Tests for the frontline play preset and single-ship keyboard routing."""

import torch

from boost_and_broadside.modes.interactive import PLAY_ENV_CONFIG, _apply_keyboard_override


def test_play_preset_is_timed_frontline_with_scriptable_fleets() -> None:
    assert PLAY_ENV_CONFIG.num_ships == 8
    assert PLAY_ENV_CONFIG.num_fields == 0
    assert PLAY_ENV_CONFIG.max_episode_steps == 18_000
    assert PLAY_ENV_CONFIG.frontline is not None
    assert not PLAY_ENV_CONFIG.single_team
    assert PLAY_ENV_CONFIG.frontline.zone_radius == 330.0
    assert PLAY_ENV_CONFIG.frontline.capture_seconds == 8.0


def test_play_keyboard_controls_team_zero_but_not_null_team_one() -> None:
    action = torch.zeros((1, 2, 3), dtype=torch.int32)
    team_id = torch.tensor([[1, 0]], dtype=torch.int32)
    keyboard = torch.tensor([1, 3, 1], dtype=torch.int32)

    result = _apply_keyboard_override(action, team_id, keyboard, frozenset({0}), 1)

    assert torch.equal(result[0, 0], torch.zeros(3, dtype=torch.int32))
    assert torch.equal(result[0, 1], keyboard)


def test_keyboard_controls_only_one_selected_ally() -> None:
    action = torch.zeros((1, 4, 3), dtype=torch.int32)
    team_id = torch.tensor([[0, 1, 0, 1]], dtype=torch.int32)
    keyboard = torch.tensor([1, 3, 1], dtype=torch.int32)

    result = _apply_keyboard_override(action, team_id, keyboard, frozenset({0}), 2)

    assert torch.equal(result[0, 0], torch.zeros(3, dtype=torch.int32))
    assert torch.equal(result[0, 2], keyboard)


def test_spectator_selection_leaves_every_ship_scripted() -> None:
    action = torch.tensor(
        [[[1, 2, 1], [2, 4, 0], [0, 1, 1], [1, 0, 0]]], dtype=torch.int32
    )
    team_id = torch.tensor([[0, 1, 0, 1]], dtype=torch.int32)
    keyboard = torch.tensor([2, 6, 1], dtype=torch.int32)

    result = _apply_keyboard_override(action, team_id, keyboard, frozenset({0}), None)

    assert torch.equal(result, action)
