"""The real-scale variant (`docs/explanation/game/real-scale.md`): its ship set, its rules
and its presets, and that its cards fly their handles."""

import math

import pytest
import torch

from bnb.contracts import GameConfig
from bnb.game import api, real_scale, ships
from bnb.game.handling.similarity import scale_flight
from bnb.game.rules import FlightConstants

SET = "real-scale"


def test_the_presets_play_the_real_scale_set_on_a_map_three_times_wider():
    for mode in ("deathmatch", "frontline"):
        standard, real = api.PRESETS[mode], api.PRESETS[f"real_scale_{mode}"]
        assert real.ship_set == SET and real.arena_radius == 3.0 * standard.arena_radius
        rules = api.rules_of(real)
        assert rules.pools[0].arena_radius / api.MAP_SCALE_RADIUS == 3.0
        standard_rules = api.rules_of(standard)
        assert rules.pools[0].episode_ticks == 2 * standard_rules.pools[0].episode_ticks
        assert rules.pools[0].respawn_ticks == 2 * standard_rules.pools[0].respawn_ticks
        assert rules.pools[0].mode.capture_seconds == 20.0
        assert rules.pools[0].mode.group_radius == 900.0
        assert rules.field_band_width == 120.0


def test_the_rules_are_todays_in_real_units():
    rules = api.rules_of(api.PRESETS["real_scale_frontline"])
    assert rules.flight.gravity == 9.81
    expected = scale_flight(FlightConstants(), real_scale.LENGTH, real_scale.TIME)
    for name in ("coast_exchange", "climb_exchange", "nose_frequency_floor"):
        assert getattr(rules.flight, name) == pytest.approx(getattr(expected, name), rel=1e-12)
    assert rules.bullet_moves == 18  # 0.6 s
    assert rules.bullet_drag == pytest.approx(8e-4 / 5.82)
    assert rules.vision_range == 3072.0
    stall = torch.sqrt(rules.flight.gravity / api.ship_types(ship_set=SET).max_lift_coefficient)
    assert rules.spawn_speed == pytest.approx(1.3 * float(stall.max()), rel=1e-4)


def test_the_fighter_has_the_p51s_handles_and_a_600_m_gun():
    fighter = ships.load(SET)[0]
    handles = fighter.handles
    assert handles["stall_speed"] == 46.0 and handles["g_limit"] == 8.0
    assert handles["cruise_speed"] == 163.0 and handles["ceiling"] == 125_300.0
    assert handles["sustained_turn_rate"] == pytest.approx(math.radians(18.0))
    assert handles["muzzle_speed"] == 880.0 and handles["hit_radius"] == 5.64
    assert handles["gun_range"] == pytest.approx(600.0, abs=10.0)
    assert "placeholder" in fighter.notes["low_speed_acceleration"]
    assert "P-51D" in fighter.notes["stall_speed"]
    assert set(fighter.notes) == set(api.HANDLE_NAMES)


def test_the_standard_game_is_untouched_by_the_set():
    assert api.rules_of(api.PRESETS["frontline"]).flight == FlightConstants()
    assert api.type_names() == ("Fighter", "Heavy")
    assert GameConfig(1, 1, 1.0).ship_set == "standard"


@pytest.mark.slow
def test_each_real_scale_card_flies_its_handles():
    """Measured (minutes: a 125 kJ store climbs for about an hour of game time), every
    flown handle is within the solver's 0.5% of the ship file's, and the climb's initial
    charge is the P-51's g x 16.3 m/s."""
    rules = api.rules_of(api.PRESETS["real_scale_frontline"])
    card = api.ship_types(dtype=torch.float64, ship_set=SET)
    flown = api.measure_handles(card, rules)
    for index, ship in enumerate(ships.load(SET)):
        for name in api.FLOWN_HANDLES:
            assert float(getattr(flown, name)[index]) == pytest.approx(
                ship.handles[name], rel=5e-3
            ), (ship.name, name)
    charge = real_scale.initial_climb_charge(rules, card, flown.cruise_speed)
    assert float(charge[0]) == pytest.approx(9.81 * 16.3, rel=real_scale.CLIMB_TOLERANCE)
