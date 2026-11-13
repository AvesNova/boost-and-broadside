"""The boundary (game v14): the smooth damage ramp in every mode, elimination's shrinking
boundary (its hold, its laws, its end at the clock), what the observation shows of it, the
live settings, the scripted agent's steer and the model's inputs."""

import dataclasses
import math

import modes_cases as cases
import pytest
import torch

from bnb.contracts import GameConfig
from bnb.contracts.obs import GLOBAL
from bnb.game import api
from bnb.game.layout import boundary_fraction, boundary_time_to_zero, environment
from bnb.game.physics.numerics import smootherstep
from bnb.game.tuning import BY_KEY, live_of, tuned, with_live

# 3v3, 1 km arena (map scale 1000 / 2600), a 30 s clock: the hold is 18 s, the shrink 12 s.
ELIMINATION = GameConfig(3, 3, 1000.0, episode_seconds=30.0, mode="elimination")
DEATHMATCH = GameConfig(3, 3, 1000.0, episode_seconds=30.0, respawn_seconds=1.0)


def _at(rules, state, tick: int):
    """The state at this tick of its episode."""
    match = dataclasses.replace(
        state.match, episode_tick=torch.full_like(state.match.episode_tick, tick)
    )
    return dataclasses.replace(state, match=match)


def _one_ship_at(rules, state, radius: float):
    """Ship 0 alone at this distance from the centre on the x axis, nose outward, the rest
    near the centre; full shields, no fields."""
    N = rules.num_ships
    x = torch.zeros_like(state.ships.x) + 10.0 * torch.arange(N)
    x[:, 0] = radius
    return cases.fields_away(
        cases.edit_ships(
            state,
            x=x,
            y=torch.zeros_like(x),
            heading_cos=torch.ones_like(x),
            heading_sin=torch.zeros_like(x),
            proper_speed=torch.full_like(x, 1e-3),
        )
    )


# ------------------------------------------------------------------ the shrink


def test_the_boundary_holds_then_shrinks_to_zero_by_the_clock_linear_or_smooth():
    rules = api.rules_of(ELIMINATION)
    ticks = rules.pools[0].episode_ticks
    hold = round(18.0 / rules.tick_length)
    elapsed = torch.tensor([0, hold, hold + (ticks - hold) // 4, (ticks + hold) // 2, ticks])
    arena = environment(rules, len(elapsed))
    linear = boundary_fraction(arena, elapsed)
    torch.testing.assert_close(linear, torch.tensor([1.0, 1.0, 0.75, 0.5, 0.0]))
    smooth_rules = tuned(rules, {"mode.shrink_law": 1})
    smooth = boundary_fraction(environment(smooth_rules, len(elapsed)), elapsed)
    s = torch.tensor([0.0, 0.0, 0.25, 0.5, 1.0])
    torch.testing.assert_close(smooth, 1.0 - smootherstep(s))
    left = boundary_time_to_zero(arena, elapsed, rules.tick_length)
    torch.testing.assert_close(left, (ticks - elapsed).float() * rules.tick_length)


def test_other_modes_keep_the_arena_radius_and_show_no_time_to_zero():
    rules = api.rules_of(DEATHMATCH)
    elapsed = torch.tensor([0, 300, 899])
    arena = environment(rules, 3)
    assert boundary_fraction(arena, elapsed).tolist() == [1.0, 1.0, 1.0]
    assert boundary_time_to_zero(arena, elapsed, rules.tick_length).tolist() == [0.0] * 3


def test_a_hold_past_the_clock_shrinks_in_the_last_tick_only():
    rules = tuned(api.rules_of(ELIMINATION), {"mode.shrink_hold_seconds": 600.0})
    ticks = rules.pools[0].episode_ticks
    arena = environment(rules, 3)
    fraction = boundary_fraction(arena, torch.tensor([0, ticks - 1, ticks]))
    assert fraction.tolist() == [1.0, 1.0, 0.0]


# ------------------------------------------------------------------ the damage


@pytest.mark.parametrize("config", [ELIMINATION, DEATHMATCH], ids=["elimination", "deathmatch"])
def test_boundary_damage_is_the_most_times_the_smootherstep_of_the_depth(config):
    """Mid-hold (elimination's boundary still at the arena radius): nothing inside, a
    smootherstep of the depth over the ramp (100 m x the map scale) past it, all of it a
    ramp width out."""
    rules = api.rules_of(config)
    ramp = rules.boundary_ramp_width * config.arena_radius / api.MAP_SCALE_RADIUS
    for depth in (-50.0, 0.0, 0.3 * ramp, 0.8 * ramp, 2.0 * ramp):
        _, state = cases.rules_and_state(config, batch=1)
        state = _at(rules, _one_ship_at(rules, state, config.arena_radius + depth), 30)
        after, events = cases.tick(rules, state)
        moved = float(torch.hypot(after.ships.x[0, 0], after.ships.y[0, 0]))
        z = min(max((moved - config.arena_radius) / ramp, 0.0), 1.0)
        want = rules.boundary_max_damage * z**3 * (z * (6 * z - 15) + 10) * rules.tick_length
        assert float(events.boundary_damage[0, 0]) == pytest.approx(want, abs=1e-6), depth


def test_the_shrunk_boundary_damages_a_ship_well_inside_the_arena():
    """Elimination, halfway through the shrink: the boundary stands at half the arena
    radius, so a ship at 0.6 R_a is 0.1 R_a past it (more than a ramp width: all of it)."""
    rules = api.rules_of(ELIMINATION)
    ticks = rules.pools[0].episode_ticks
    hold = round(18.0 / rules.tick_length)
    halfway = (ticks + hold) // 2 - 1  # the damage reads the boundary after the move
    _, state = cases.rules_and_state(ELIMINATION, batch=1)
    state = _at(rules, _one_ship_at(rules, state, 0.6 * ELIMINATION.arena_radius), halfway)
    _, events = cases.tick(rules, state)
    full = rules.boundary_max_damage * rules.tick_length
    assert float(events.boundary_damage[0, 0]) == pytest.approx(full)
    assert float(events.boundary_damage[0, 1:].abs().sum()) == 0.0  # near the centre: 0
    _, deathmatch_state = cases.rules_and_state(DEATHMATCH, batch=1)
    deathmatch_rules = api.rules_of(DEATHMATCH)
    deathmatch_state = _one_ship_at(deathmatch_rules, deathmatch_state, 600.0)
    _, events = cases.tick(deathmatch_rules, _at(deathmatch_rules, deathmatch_state, halfway))
    assert float(events.boundary_damage.abs().sum()) == 0.0


def test_at_the_clock_no_ship_is_safe_and_the_health_left_decides_the_rest():
    """The clock's last tick: the boundary is at radius 0, so every ship away from the
    centre takes boundary damage; the survivors' health left decides."""
    rules = api.rules_of(ELIMINATION)
    _, state = cases.rules_and_state(ELIMINATION, batch=1)
    state = _at(rules, cases.fields_away(state), rules.pools[0].episode_ticks - 1)
    _, events = cases.tick(rules, state)
    reach = torch.hypot(state.ships.x, state.ships.y)
    hurt = events.boundary_damage[0] > 0
    assert hurt[reach[0] > 100.0].all()
    assert bool(events.done[0])


# ------------------------------------------------------------------ the observation


def test_the_observation_shows_the_boundary_and_keeps_the_arena_radius():
    rules = api.rules_of(ELIMINATION)
    ticks = rules.pools[0].episode_ticks
    hold = round(18.0 / rules.tick_length)
    _, state = cases.rules_and_state(ELIMINATION, batch=1)
    _, obs, _ = api.perceive(rules, _at(rules, state, (ticks + hold) // 2))
    globals_ = obs.globals[0]  # (2, columns): both views alike
    assert globals_[:, GLOBAL["arena_radius"]].tolist() == [1000.0, 1000.0]
    torch.testing.assert_close(globals_[:, GLOBAL["boundary_fraction"]], torch.full((2,), 0.5))
    left = (ticks - (ticks + hold) // 2) * rules.tick_length
    torch.testing.assert_close(globals_[:, GLOBAL["boundary_time_to_zero"]], torch.full((2,), left))


# ------------------------------------------------------------------ the settings


def test_the_shrink_and_the_ramp_are_live_settings_of_the_map_panel():
    from bnb.play.tuning_panel import MAP_GROUPS

    for key in ("mode.shrink_hold_seconds", "mode.shrink_law"):
        assert BY_KEY[key].group in MAP_GROUPS and BY_KEY[key].modes == ("elimination",)
    for key in ("rules.boundary_max_damage", "rules.boundary_ramp_width"):
        assert BY_KEY[key].group in MAP_GROUPS
    rules = tuned(
        api.rules_of(ELIMINATION),
        {"mode.shrink_hold_seconds": 6.0, "rules.boundary_ramp_width": 50.0},
    )
    live = live_of(rules)
    assert float(live["pool0.mode.shrink_hold_seconds"]) == 6.0
    assert live["pool0.mode.shrink_law"].dtype == torch.int32
    rebuilt = with_live(api.tuning.static_of(rules), live)
    arena = environment(rebuilt, 1)
    assert float(arena.shrink_hold_ticks[0]) == pytest.approx(6.0 / rules.tick_length)
    assert float(rebuilt.boundary_ramp_width) == 50.0


# ------------------------------------------------------------------ the agents


def test_the_scripted_agent_steers_by_the_boundary_as_it_stands():
    from bnb.controllers.scripted_combat import context
    from bnb.controllers.scripted_defaults import MODE_DEFAULTS
    from bnb.game.handling.characteristics import characteristics

    rules = api.rules_of(ELIMINATION)
    _, state = cases.rules_and_state(ELIMINATION, batch=1)
    ticks = rules.pools[0].episode_ticks
    hold = round(18.0 / rules.tick_length)
    _, obs, _ = api.perceive(rules, _at(rules, state, (ticks + hold) // 2))
    view = type(obs)(**{name: part[:, 0] for name, part in vars(obs).items()})
    measured = characteristics(cases.TYPES, rules)
    fleet = context(view, measured, MODE_DEFAULTS.sets["deathmatch"])
    assert float(fleet.boundary_radius[0, 0]) == pytest.approx(500.0)
    assert float(fleet.arena_radius[0, 0]) == 1000.0


def test_a_new_build_reads_the_boundary_and_an_older_one_does_not():
    from bnb.models.yemong.inputs import BOUNDARY_INPUTS, MODE_INPUTS, global_inputs
    from bnb.models.yemong.model import YemongConfig
    from bnb.runtime.train import TrainConfig

    assert TrainConfig(GameConfig(5, 10, 2600.0)).yemong.boundary_inputs
    assert not YemongConfig().boundary_inputs
    assert global_inputs(MODE_INPUTS, True, True) == global_inputs(MODE_INPUTS, True) + (
        BOUNDARY_INPUTS
    )
    assert math.isclose(BOUNDARY_INPUTS, 10)
