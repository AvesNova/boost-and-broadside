"""A game's arena radius at another fleet size (game v16, `api.arena_radius_for`): the
density rule, then the mode's minimum (team deathmatch 1000 m at map scale 1, 3000 m at
real scale; elimination and frontline none), applied after any arena scale."""

import math

import pytest

from bnb.game import api


def test_real_scale_deathmatch_plays_small_fleets_on_at_least_3000_m():
    game = api.PRESETS["real_scale_deathmatch"]  # 5v5 at 4200 m
    assert api.minimum_arena_radius(game) == pytest.approx(3000.0)
    assert api.arena_radius_for(game, 1) == pytest.approx(3000.0)  # the rule alone: 1878 m
    assert api.arena_radius_for(game, 2) == pytest.approx(3000.0)  # the rule alone: 2656 m
    assert api.arena_radius_for(game, 5) == 4200.0
    assert api.arena_radius_for(game, 10) == round(4200.0 * math.sqrt(2.0), 1)


def test_the_standard_set_floors_at_a_third_and_the_floor_follows_the_scale():
    game = api.PRESETS["deathmatch"]  # 5v5 at 2600 m, map scale 1
    assert api.minimum_arena_radius(game) == 1000.0
    assert api.arena_radius_for(game, 1) == round(2600.0 * math.sqrt(0.2), 1)  # above it
    # the floor holds after the arena scale: a half-size 1v1 map is still 1000 m
    assert api.arena_radius_for(game, 1, arena_scale=0.5) == 1000.0
    assert api.arena_radius_for(game, 5, arena_scale=0.5) == 1300.0


def test_elimination_has_no_floor_yet():
    game = api.PRESETS["real_scale_elimination"]
    assert api.minimum_arena_radius(game) == 0.0
    assert api.arena_radius_for(game, 1) == round(game.arena_radius * math.sqrt(0.2), 1)


def test_frontline_s_floor_keeps_its_floored_zones_inside_the_boundary():
    """Frontline's arena floor (4200 m at real scale) holds a half-size sampled map above
    the radius where the floored zones would reach past the boundary (about 0.52 of the
    real-scale 5v5 arena)."""
    game = api.PRESETS["real_scale_frontline"]
    assert api.minimum_arena_radius(game) == 4200.0
    assert api.arena_radius_for(game, 5, arena_scale=0.5) == 4200.0
    assert api.arena_radius_for(game, 5) == game.arena_radius


def test_the_presets_derive_their_sizes_by_the_one_function():
    assert api.PRESETS["clover"].arena_radius == 4933.2
    for base in ("deathmatch", "real_scale_deathmatch", "frontline", "real_scale_frontline"):
        for n in (10, 20, 50):
            radius = api.PRESETS[f"{base}_{n}v{n}_density"].arena_radius
            assert radius == api.arena_radius_for(api.PRESETS[base], n)
