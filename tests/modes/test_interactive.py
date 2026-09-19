"""Tests for the frontline play preset and single-ship keyboard routing."""

import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.env.frontline import (
    FRONTLINE_FIELD_RADIUS_MAX,
    frontline_ship_config,
)
from boost_and_broadside.env.observation import ObsKey, YemongObservation
from boost_and_broadside.modes.interactive import (
    PLAY_ENV_CONFIG,
    _apply_keyboard_override,
    _apply_policy_action_delay,
    _set_observation_previous_action,
)


def test_play_preset_is_timed_frontline_with_scriptable_fleets() -> None:
    assert PLAY_ENV_CONFIG.num_ships == 10
    assert PLAY_ENV_CONFIG.num_fields == 10
    assert PLAY_ENV_CONFIG.max_episode_steps == 9_000
    assert PLAY_ENV_CONFIG.frontline is not None
    assert not PLAY_ENV_CONFIG.single_team
    assert PLAY_ENV_CONFIG.frontline.zone_radius == 330.0
    assert PLAY_ENV_CONFIG.frontline.capture_seconds == 10.0
    assert PLAY_ENV_CONFIG.frontline.respawn_power == 40.0
    assert PLAY_ENV_CONFIG.frontline.shield_recharge_delay == 5.0
    assert PLAY_ENV_CONFIG.frontline.shield_recharge_per_second == 15.0
    assert FRONTLINE_FIELD_RADIUS_MAX == 750.0
    ship_config = frontline_ship_config(ShipConfig())
    assert ship_config.dt == 1.0 / 30.0
    assert ship_config.field_integrator == "two_step"
    assert ship_config.field_integration_substeps == 1


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
    action = torch.tensor([[[1, 2, 1], [2, 4, 0], [0, 1, 1], [1, 0, 0]]], dtype=torch.int32)
    team_id = torch.tensor([[0, 1, 0, 1]], dtype=torch.int32)
    keyboard = torch.tensor([2, 6, 1], dtype=torch.int32)

    result = _apply_keyboard_override(action, team_id, keyboard, frozenset({0}), None)

    assert torch.equal(result, action)


def test_policy_actions_are_delayed_while_scripted_actions_are_immediate() -> None:
    team_id = torch.tensor([[0, 1, 0, 1]], dtype=torch.int32)
    decided = torch.tensor(
        [[[1, 1, 1], [2, 2, 0], [1, 3, 0], [0, 4, 1]]], dtype=torch.int32
    )
    buffered = torch.tensor(
        [[[2, 6, 0], [1, 5, 1], [0, 2, 1], [2, 1, 0]]], dtype=torch.int32
    )

    applied, next_buffer = _apply_policy_action_delay(
        decided, buffered, team_id, frozenset({0})
    )

    assert torch.equal(applied[:, (0, 2)], buffered[:, (0, 2)])
    assert torch.equal(applied[:, (1, 3)], decided[:, (1, 3)])
    assert torch.equal(next_buffer[:, (0, 2)], decided[:, (0, 2)])
    assert torch.equal(next_buffer[:, (1, 3)], torch.zeros((1, 2, 3), dtype=torch.int32))


def test_policy_action_buffer_starts_with_neutral_first_tick() -> None:
    team_id = torch.tensor([[0, 1]], dtype=torch.int32)
    decided = torch.tensor([[[1, 3, 1], [2, 4, 1]]], dtype=torch.int32)

    applied, _ = _apply_policy_action_delay(
        decided, torch.zeros_like(decided), team_id, frozenset({0})
    )

    assert torch.equal(applied[0, 0], torch.zeros(3, dtype=torch.int32))
    assert torch.equal(applied[0, 1], decided[0, 1])


def test_next_observation_exposes_queued_action_to_both_team_views() -> None:
    previous = torch.zeros((1, 3, 3), dtype=torch.float32)
    team1_previous = torch.zeros_like(previous)
    observation = YemongObservation(
        data={ObsKey.PREVIOUS_ACTION: previous},
        team1_data={ObsKey.PREVIOUS_ACTION: team1_previous},
    )
    decided = torch.tensor([[[1, 3, 1], [2, 4, 0]]], dtype=torch.int32)

    _set_observation_previous_action(observation, decided, num_ships=2)

    assert torch.equal(previous[:, :2], decided.float())
    assert torch.equal(team1_previous[:, :2], decided.float())
    assert torch.equal(previous[:, 2], torch.zeros((1, 3)))
