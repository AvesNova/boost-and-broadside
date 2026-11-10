"""The scripted controllers' lengths follow the card: the deathmatch/wide classification in
gun ranges and the chaser's distances in corner radii and gun ranges reproduce the standard
card's metres exactly, and a real-scale (P-51-like) card plays its own, with the real-scale
parameter sets."""

import dataclasses
import math

import pytest
import torch
from scripted_cases import CHARACTERISTICS, absent, observation, ship

from bnb.contracts.obs import GLOBAL
from bnb.controllers.scripted import (
    STANDARD_CORNER_RADIUS,
    STANDARD_GUN_RANGE,
    Chaser,
    corner_radius,
    gun_range,
    mode_index,
)
from bnb.controllers.scripted_parameters import MODES

# The standard fighter at P-51 scale: lengths x5.82 and g = 9.81 m/s^2 (times x4.00), with
# the real-scale variant's 600 m gun range.
LENGTH = 5.82
TIME = math.sqrt(LENGTH * 27.0 / 9.81)
REAL_SCALE = dataclasses.replace(
    CHARACTERISTICS,
    corner_speed=CHARACTERISTICS.corner_speed * (LENGTH / TIME),
    max_acceleration=CHARACTERISTICS.max_acceleration * (LENGTH / TIME**2),
    gun_range=CHARACTERISTICS.gun_range * (600.0 / STANDARD_GUN_RANGE),
    gravity=torch.full_like(CHARACTERISTICS.gravity, 9.81),
)
# The same lengths at the standard gravity: a card the agent classifies by gun ranges alone.
SCALED_LENGTHS = dataclasses.replace(REAL_SCALE, gravity=CHARACTERISTICS.gravity)


def _deathmatch(arena_radius: float):
    enemy = ship(x=400.0, team_relative=1)
    return observation(
        [ship(), absent(team_relative=0)], [enemy, absent()], arena_radius=arena_radius
    )


def test_the_standard_lengths_are_the_standard_cards():
    assert gun_range(CHARACTERISTICS) == STANDARD_GUN_RANGE
    assert corner_radius(CHARACTERISTICS) == STANDARD_CORNER_RADIUS


def test_deathmatch_or_wide_by_gun_ranges():
    """The standard presets classify as before (2600 m deathmatch, 8200 m wide); a card of
    the standard gravity with a 600 m gun range classifies the standard map 3 times as wide
    (7800 m) as deathmatch, and its wide map (24,600 m) as wide."""
    narrow, wide = MODES.index("deathmatch"), MODES.index("deathmatch_wide")
    for radius, expected in ((2600.0, narrow), (8200.0, wide)):
        assert mode_index(_deathmatch(radius), CHARACTERISTICS).tolist() == [expected]
    for radius, expected in ((7800.0, narrow), (3 * 8200.0, wide)):
        assert mode_index(_deathmatch(radius), SCALED_LENGTHS).tolist() == [expected]


def test_a_real_scale_card_plays_the_real_scale_sets():
    """A card at the real-scale gravity plays real_scale_deathmatch on any deathmatch map and
    real_scale_frontline in Frontline; the real-scale presets' own card is one."""
    from bnb.controllers.scripted import REAL_SCALE_GRAVITY
    from bnb.game import api
    from bnb.game.handling.characteristics import characteristics

    deathmatch = MODES.index("real_scale_deathmatch")
    for radius in (2600.0, 7800.0, 3 * 8200.0):
        assert mode_index(_deathmatch(radius), REAL_SCALE).tolist() == [deathmatch]
    view = _deathmatch(7800.0)
    view.globals[:, GLOBAL["mode_frontline"]] = 1.0
    view.globals[:, GLOBAL["mode_team_deathmatch"]] = 0.0
    assert mode_index(view, REAL_SCALE).tolist() == [MODES.index("real_scale_frontline")]
    assert mode_index(view, CHARACTERISTICS).tolist() == [MODES.index("frontline")]
    game = api.PRESETS["real_scale_deathmatch"]
    card = characteristics(api.ship_types(ship_set=game.ship_set), api.rules_of(game))
    assert float(card.gravity.amax()) < REAL_SCALE_GRAVITY < float(CHARACTERISTICS.gravity[0])


def test_the_chaser_plays_the_race_distances_on_the_standard_card():
    """Boost beyond 600 m, reverse inside 120 m, shoot inside 250 m: the same numbers in
    float32, where the observation's distances are compared."""
    chaser = Chaser(CHARACTERISTICS, 2)
    for distance, metres in (
        (chaser.boost_distance, 600.0),
        (chaser.reverse_distance, 120.0),
        (chaser.shoot_distance, 250.0),
    ):
        assert torch.tensor(distance, dtype=torch.float32).item() == metres


def _chased(chaser: Chaser, distance: float):
    """(power, shoot) of the chaser's first ship with one enemy dead ahead at `distance`."""
    enemy = ship(x=distance, team_relative=1)
    view = observation([ship(), absent(team_relative=0)], [enemy, absent()])
    action = chaser.step(None, view, None)[0]
    return int(action.power[0, 0]), int(action.shoot[0, 0])


def test_the_chaser_distances_follow_the_card():
    """At each distance just inside and outside the standard thresholds, then the same
    distances in the real-scale card's lengths: the same actions."""
    standard, real = Chaser(CHARACTERISTICS, 2), Chaser(REAL_SCALE, 2)
    radius_ratio = corner_radius(REAL_SCALE) / corner_radius(CHARACTERISTICS)
    range_ratio = gun_range(REAL_SCALE) / gun_range(CHARACTERISTICS)
    assert real.boost_distance == pytest.approx(standard.boost_distance * radius_ratio)
    assert real.reverse_distance == pytest.approx(standard.reverse_distance * radius_ratio)
    assert real.shoot_distance == pytest.approx(standard.shoot_distance * range_ratio)
    for metres, ratio in ((601.0, radius_ratio), (599.0, radius_ratio), (119.0, radius_ratio)):
        assert _chased(standard, metres)[0] == _chased(real, metres * ratio)[0]
    for metres in (249.0, 251.0):
        assert _chased(standard, metres)[1] == _chased(real, metres * range_ratio)[1]
    assert [_chased(standard, metres) for metres in (601.0, 300.0, 249.0, 119.0)] == [
        (1, 0),
        (0, 0),
        (0, 1),
        (2, 1),
    ]
