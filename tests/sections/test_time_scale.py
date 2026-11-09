"""The tuning panel's time scale: the game k times faster at fixed map lengths, by the
units of every rule setting and card field (`bnb.game.handling.similarity`), energies with
gravity; the values held to whole ticks or Hz snap."""

import time

import pytest
import torch

from bnb.contracts import GameConfig
from bnb.game import api
from bnb.game.handling.similarity import CARD_UNITS, HANDLE_UNITS, RULE_UNITS, scale_handles
from bnb.play.loop import TIME_SCALES, MatchLoop, MatchSetup
from bnb.play.tuning import Tuner, base_rules, time_scaled_card, time_scaled_settings

RULES = api.rules_of(api.PRESETS["frontline"])


def _values(rules):
    return {s.key: api.tuning.value(rules, s.key) for s in api.tuning.settings_for(rules)}


def test_every_setting_handle_and_card_field_has_units():
    assert {s.key for s in api.tuning.SETTINGS} <= set(RULE_UNITS)
    assert set(HANDLE_UNITS) == set(api.HANDLE_NAMES)
    assert set(CARD_UNITS) == set(api.default_card())


def test_settings_scale_by_their_units_and_times_snap():
    values = _values(RULES)
    faster = time_scaled_settings(values, 2.0, RULES.tick_length)
    assert faster["flight.gravity"] == 4.0 * values["flight.gravity"]
    assert faster["rules.spawn_speed"] == 2.0 * values["rules.spawn_speed"]
    assert faster["flight.nose_frequency_floor"] == 2.0 * values["flight.nose_frequency_floor"]
    assert faster["pool.episode_seconds"] == 150.0
    assert faster["mode.capture_seconds"] == 5.0
    assert faster["pool.arena_radius"] == values["pool.arena_radius"]  # lengths fixed
    assert faster["rules.bullet_life"] == pytest.approx(0.2)  # 6 ticks
    slower = time_scaled_settings(values, 1.0 / 3.0, RULES.tick_length)
    assert round(slower["rules.bullet_life"] / RULES.tick_length) == 36
    snapped = time_scaled_settings(values, 0.7, RULES.tick_length)
    ticks = snapped["rules.bullet_life"] / RULES.tick_length
    assert ticks == pytest.approx(round(ticks)) and round(ticks) == round(12 / 0.7)


def test_the_card_scales_by_its_units_with_energies_and_snaps_the_fire_rate():
    types = api.ship_types()
    faster = time_scaled_card(types, RULES, 2.0)
    assert torch.allclose(faster.engine_power, 8.0 * types.engine_power)
    assert torch.allclose(faster.max_energy, 4.0 * types.max_energy)  # the height E/g kept
    assert torch.allclose(faster.spawn_energy, 4.0 * types.spawn_energy)
    assert torch.allclose(faster.bullet_energy_cost, 4.0 * types.bullet_energy_cost)
    assert torch.allclose(faster.max_acceleration, 4.0 * types.max_acceleration)
    assert torch.allclose(faster.coast_exchange_scale, 4.0 * types.coast_exchange_scale)
    assert torch.allclose(faster.muzzle_speed, 2.0 * types.muzzle_speed)
    assert torch.equal(faster.max_lift_coefficient, types.max_lift_coefficient)  # 1/m
    assert faster.fire_rate.tolist() == [30.0, 15.0]  # 20 Hz is not a whole number of ticks
    assert torch.allclose(
        faster.shield_recharge_delay_scale, 0.5 * types.shield_recharge_delay_scale
    )
    slower = time_scaled_card(types, RULES, 0.5)
    assert slower.fire_rate.tolist() == [6.0, 5.0]  # 7.5 and 5 Hz, held to whole Hz at 30 Hz


@pytest.mark.slow
def test_a_time_scaled_game_flies_the_scaled_handles():
    """Measured, the scaled card under the scaled rules flies every handle scaled by its
    units (the tick stays, so within its discretisation)."""
    factor = 2.0
    card = api.ship_types(dtype=torch.float64)
    values = time_scaled_settings(_values(RULES), factor, RULES.tick_length)
    rules = api.tuning.tuned(RULES, values, bounded=False)
    handles = api.measure_handles(card, RULES)
    scaled = api.measure_handles(time_scaled_card(card, RULES, factor), rules)
    expected = scale_handles(handles, 1.0, 1.0 / factor)
    for name in api.FLOWN_HANDLES:
        assert torch.allclose(getattr(scaled, name), getattr(expected, name), rtol=0.03), name
    assert torch.allclose(scaled.ceiling, factor**2 * handles.ceiling)
    assert torch.allclose(scaled.gun_range, handles.gun_range, rtol=1e-4)  # 6 ticks exactly


def test_playback_runs_from_a_quarter_to_eight_times():
    assert TIME_SCALES[0] == 0.25 and TIME_SCALES[1] == 0.5 and TIME_SCALES[-1] == 8.0


@pytest.mark.slow
def test_make_this_speed_normal_scales_the_live_game_and_resets_playback():
    setup = MatchSetup(GameConfig(2, 4, 900.0, mode="frontline"))
    loop = MatchLoop(setup, compiled=False)
    tuner = Tuner(loop, base_rules(setup.config, setup.lineup))

    def frames(done, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            tuner.update()
            if done():
                return True
            time.sleep(0.01)
        return False

    try:
        assert frames(lambda: tuner.targets is not None, 300.0), tuner.message
        cruise = tuner.targets.cruise_speed.clone()
        gravity, store = loop.rules.flight.gravity, loop.types.max_energy.clone()
        tuner.time_scale(4.0)
        assert loop.rules.flight.gravity == 16.0 * gravity
        assert torch.allclose(loop.types.max_energy, 16.0 * store)
        assert "fire rate" in tuner.message  # 15 Hz x 4 is past the tick rate
        assert frames(lambda: tuner.targets is not None, 300.0), tuner.message
        assert torch.allclose(tuner.targets.cruise_speed, 4.0 * cruise, rtol=0.03)
        assert tuner.tuning.rules["flight.gravity"] == 16.0 * gravity
    finally:
        tuner.close()
