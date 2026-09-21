"""Tests for the frontline play preset and single-ship keyboard routing."""

from dataclasses import replace

import pytest
import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.env.frontline import (
    FRONTLINE_FIELD_RADIUS_MAX,
    frontline_rules_match,
    frontline_scale,
    frontline_ship_config,
)
from boost_and_broadside.env.observation import ObsKey, YemongObservation
from boost_and_broadside.modes.interactive import (
    PLAY_ENV_CONFIG,
    _apply_keyboard_override,
    _apply_policy_action_delay,
    _frontline_interactive_config,
    _interactive_device,
    _interactive_render_config,
    _selected_human_mask,
    _set_observation_previous_action,
)
from boost_and_broadside.ui.renderer import RenderConfig


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


def test_keyboard_controls_the_selected_ship_on_either_team() -> None:
    action = torch.zeros((1, 2, 3), dtype=torch.int32)
    keyboard = torch.tensor([1, 3, 1], dtype=torch.int32)

    result = _apply_keyboard_override(action, keyboard, 1)

    assert torch.equal(result[0, 0], torch.zeros(3, dtype=torch.int32))
    assert torch.equal(result[0, 1], keyboard)


def test_keyboard_controls_only_one_selected_ally() -> None:
    action = torch.zeros((1, 4, 3), dtype=torch.int32)
    keyboard = torch.tensor([1, 3, 1], dtype=torch.int32)

    result = _apply_keyboard_override(action, keyboard, 2)

    assert torch.equal(result[0, 0], torch.zeros(3, dtype=torch.int32))
    assert torch.equal(result[0, 2], keyboard)


def test_spectator_selection_leaves_every_ship_scripted() -> None:
    action = torch.tensor([[[1, 2, 1], [2, 4, 0], [0, 1, 1], [1, 0, 0]]], dtype=torch.int32)
    keyboard = torch.tensor([2, 6, 1], dtype=torch.int32)

    result = _apply_keyboard_override(action, keyboard, None)

    assert torch.equal(result, action)


def test_policy_actions_are_delayed_while_scripted_actions_are_immediate() -> None:
    team_id = torch.tensor([[0, 1, 0, 1]], dtype=torch.int32)
    decided = torch.tensor([[[1, 1, 1], [2, 2, 0], [1, 3, 0], [0, 4, 1]]], dtype=torch.int32)
    buffered = torch.tensor([[[2, 6, 0], [1, 5, 1], [0, 2, 1], [2, 1, 0]]], dtype=torch.int32)

    applied, next_buffer = _apply_policy_action_delay(decided, buffered, team_id, frozenset({0}))

    assert torch.equal(applied[:, (0, 2)], buffered[:, (0, 2)])
    assert torch.equal(applied[:, (1, 3)], decided[:, (1, 3)])
    assert torch.equal(next_buffer[:, (0, 2)], decided[:, (0, 2)])
    assert torch.equal(next_buffer[:, (1, 3)], torch.zeros((1, 2, 3), dtype=torch.int32))


def test_human_policy_override_is_immediate_and_clears_its_policy_buffer_slot() -> None:
    team_id = torch.tensor([[0, 1]], dtype=torch.int32)
    decided = torch.tensor([[[1, 3, 1], [2, 4, 0]]], dtype=torch.int32)
    buffered = torch.tensor([[[2, 6, 0], [1, 5, 1]]], dtype=torch.int32)
    human = _selected_human_mask(team_id, True, 0)

    applied, next_buffer = _apply_policy_action_delay(
        decided, buffered, team_id, frozenset({0}), immediate_mask=human
    )

    assert torch.equal(applied[0, 0], decided[0, 0])
    assert torch.equal(next_buffer[0, 0], torch.zeros(3, dtype=torch.int32))
    # Releasing control returns the policy ship to its neutral queued first tick,
    # never to a stale keyboard action.
    released, _ = _apply_policy_action_delay(decided, next_buffer, team_id, frozenset({0}))
    assert torch.equal(released[0, 0], torch.zeros(3, dtype=torch.int32))


def test_large_interactive_fleet_keeps_the_requested_cuda_device() -> None:
    assert _interactive_device("cuda", PLAY_ENV_CONFIG) == "cpu"
    assert _interactive_device("cuda", replace(PLAY_ENV_CONFIG, num_ships=100)) == "cuda"


def test_play_and_watch_share_frontline_sizing_and_dev_render_configuration() -> None:
    env_config, ship_config = _frontline_interactive_config(
        ships_per_team=50, num_fields=72, ship_config=frontline_ship_config(ShipConfig())
    )
    render_config = _interactive_render_config(RenderConfig(), ship_config, env_config)

    assert env_config.num_ships == 100
    assert env_config.num_fields == 72
    # Rules are the play preset's; geometry is the density-matched 50v50 map.
    assert frontline_rules_match(env_config.frontline, PLAY_ENV_CONFIG.frontline)
    scale = frontline_scale(100)
    baseline = PLAY_ENV_CONFIG.frontline
    assert env_config.frontline.playable_radius == pytest.approx(baseline.playable_radius * scale)
    assert env_config.frontline.zone_radius == pytest.approx(baseline.zone_radius * scale)
    assert env_config.frontline.zone_ring_radius / env_config.frontline.playable_radius == (
        pytest.approx(baseline.zone_ring_radius / baseline.playable_radius)
    )
    assert render_config.fps == 30
    assert render_config.show_unlimited_button
    assert render_config.show_frame_pacing_toggle
    assert render_config.vision_mode.value == "FULL"


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
