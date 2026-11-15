"""The boundary (game v14): the smooth damage ramp in every mode, elimination's shrinking
boundary (its hold, its laws, its close a ramp width past the centre before the clock: game
v16), what the observation shows of it, the live settings, Imugi's steer and
the model's inputs."""

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

# 3v3, 1 km arena (map scale 1000 / 2600), a 30 s clock: the hold is 18 s, the shrink 4 s
# (to the close, 8 s before the clock), the close a ramp width (100 / 2600 of the arena
# radius) past the centre.
ELIMINATION = GameConfig(3, 3, 1000.0, episode_seconds=30.0, mode="elimination")
CLOSE = -100.0 / 2600.0  # the boundary's fraction from its close on
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


def _schedule(rules):
    """(hold, close, clock) ticks of the rules' elimination shrink."""
    ticks = rules.pools[0].episode_ticks
    mode = rules.pools[0].mode
    hold = round(mode.shrink_hold_seconds / rules.tick_length)
    close = ticks - round(mode.shrink_end_lead_seconds / rules.tick_length)
    return hold, close, ticks


def test_the_boundary_holds_shrinks_past_zero_and_closes_before_the_clock_linear():
    rules = api.rules_of(ELIMINATION)
    hold, close, ticks = _schedule(rules)
    zero = hold + (close - hold) / (1.0 - CLOSE)  # the linear law's zero crossing
    elapsed = torch.tensor([0, hold, (hold + close) // 2, close, ticks])
    arena = environment(rules, len(elapsed))
    torch.testing.assert_close(arena.shrink_zero_ticks, torch.full((5,), zero))
    fraction = boundary_fraction(arena, elapsed)
    middle = 1.0 - (1.0 - CLOSE) * ((hold + close) // 2 - hold) / (close - hold)
    torch.testing.assert_close(fraction, torch.tensor([1.0, 1.0, middle, CLOSE, CLOSE]))
    pair = torch.tensor([math.floor(zero), math.ceil(zero)])
    at_zero = boundary_fraction(environment(rules, 2), pair)
    assert float(at_zero[0]) > 0.0 > float(at_zero[1])
    left = boundary_time_to_zero(arena, elapsed, rules.tick_length)
    want = (zero - elapsed.float()).clamp_min(0.0) * rules.tick_length
    torch.testing.assert_close(left, want)


def test_the_smooth_law_crosses_zero_where_its_smootherstep_reaches_the_close():
    rules = tuned(api.rules_of(ELIMINATION), {"mode.shrink_law": 1})
    hold, close, ticks = _schedule(rules)
    arena = environment(rules, 1)
    zero = float(arena.shrink_zero_ticks[0])
    s = (zero - hold) / (close - hold)
    assert float(smootherstep(torch.tensor(s))) == pytest.approx(1.0 / (1.0 - CLOSE), abs=1e-6)
    elapsed = torch.tensor([hold, (hold + close) // 2, close, ticks])
    s = ((elapsed.float() - hold) / (close - hold)).clamp(0.0, 1.0)
    fraction = boundary_fraction(environment(rules, 4), elapsed)
    torch.testing.assert_close(fraction, 1.0 - (1.0 - CLOSE) * smootherstep(s))
    pair = torch.tensor([math.floor(zero), math.ceil(zero)])
    at_zero = boundary_fraction(environment(rules, 2), pair)
    assert float(at_zero[0]) > 0.0 > float(at_zero[1])


@pytest.mark.parametrize("power", [2.0, 3.0])
def test_the_power_law_falls_fast_then_slow_and_crosses_zero_in_closed_form(power):
    rules = tuned(api.rules_of(ELIMINATION), {"mode.shrink_law": 2, "mode.shrink_power": power})
    hold, close, ticks = _schedule(rules)
    elapsed = torch.tensor([hold, (3 * hold + close) // 4, (hold + close) // 2, close])
    s = ((elapsed.float() - hold) / (close - hold)).clamp(0.0, 1.0)
    fraction = boundary_fraction(environment(rules, 4), elapsed)
    torch.testing.assert_close(fraction, CLOSE + (1.0 - CLOSE) * (1.0 - s) ** power)
    assert float(fraction[0] - fraction[1]) > float(fraction[2] - fraction[3])  # fast first
    zero = float(environment(rules, 1).shrink_zero_ticks[0])
    pair = torch.tensor([math.ceil(zero) - 1, math.floor(zero) + 1])
    at_zero = boundary_fraction(environment(rules, 2), pair)
    assert float(at_zero[0]) > 0.0 > float(at_zero[1])


def test_the_corner_law_reaches_its_corner_on_time_then_closes_past_zero():
    rules = api.rules_of(ELIMINATION)
    hold, close, ticks = _schedule(rules)
    knee_ticks = (close - hold) // 4
    rules = tuned(
        rules,
        {
            "mode.shrink_law": 3,
            "mode.shrink_knee_seconds": knee_ticks * rules.tick_length,
            "mode.shrink_knee_fraction": 0.3,
        },
    )
    elapsed = torch.tensor([hold, hold + knee_ticks, close])
    fraction = boundary_fraction(environment(rules, 3), elapsed)
    torch.testing.assert_close(fraction, torch.tensor([1.0, 0.3, CLOSE]))
    zero = float(environment(rules, 1).shrink_zero_ticks[0])
    pair = torch.tensor([math.ceil(zero) - 1, math.floor(zero) + 1])
    at_zero = boundary_fraction(environment(rules, 2), pair)
    assert float(at_zero[0]) > 0.0 > float(at_zero[1])


def test_the_speed_cap_holds_the_boundary_back_and_moves_its_zero_crossing():
    rules = tuned(api.rules_of(ELIMINATION), {"mode.shrink_law": 2, "mode.shrink_power": 3.0})
    hold, close, ticks = _schedule(rules)
    free = environment(rules, 1)
    cap = 200.0  # m/s on a 1 km arena: the uncapped law falls faster than this at its start
    capped_rules = tuned(rules, {"mode.shrink_max_speed": cap})
    arena = environment(capped_rules, 1)
    elapsed = torch.arange(hold, ticks + 1)
    radius = 1000.0 * boundary_fraction(environment(capped_rules, len(elapsed)), elapsed)
    speed = -(radius[1:] - radius[:-1]) / rules.tick_length
    assert float(speed.max()) <= cap * (1.0 + 1e-4)
    zero = hold + 1000.0 / (cap * rules.tick_length)
    assert float(free.shrink_zero_ticks[0]) < zero
    left = boundary_time_to_zero(arena, torch.tensor([hold]), rules.tick_length)
    assert float(left[0]) == pytest.approx((zero - hold) * rules.tick_length, rel=1e-5)


def test_other_modes_keep_the_arena_radius_and_show_no_time_to_zero():
    rules = api.rules_of(DEATHMATCH)
    elapsed = torch.tensor([0, 300, 899])
    arena = environment(rules, 3)
    assert boundary_fraction(arena, elapsed).tolist() == [1.0, 1.0, 1.0]
    assert boundary_time_to_zero(arena, elapsed, rules.tick_length).tolist() == [0.0] * 3


def test_a_hold_past_the_close_shrinks_in_its_last_tick_only():
    rules = tuned(api.rules_of(ELIMINATION), {"mode.shrink_hold_seconds": 600.0})
    _, close, ticks = _schedule(rules)
    arena = environment(rules, 3)
    fraction = boundary_fraction(arena, torch.tensor([0, close - 1, close]))
    torch.testing.assert_close(fraction, torch.tensor([1.0, 1.0, CLOSE]))


@pytest.mark.parametrize("ship_set", ["standard", "real-scale"])
def test_the_lead_outlasts_the_largest_shield_at_the_most_damage(ship_set):
    """From the close a ship at the centre takes the boundary's most every tick (and, hurt,
    recharges nothing): the lead must outlast the largest shield of the set, plus the tick
    that kills a broken shield. A guard on the defaults against the ship card."""
    preset = "elimination" if ship_set == "standard" else "real_scale_elimination"
    rules = api.rules_of(api.PRESETS[preset])
    largest = float(api.ship_types(ship_set=ship_set).max_shield.max())
    needed = largest / rules.boundary_max_damage + rules.tick_length
    assert rules.pools[0].mode.shrink_end_lead_seconds >= needed + 1.0  # a second's margin


@pytest.mark.parametrize("preset", ["elimination", "real_scale_elimination"])
def test_ships_parked_at_the_centre_all_die_before_the_clock(preset):
    """Every ship of the set's largest shield, full, parked within 5 m of the centre and
    flying quiet (coasting, the stick straight, no shot: `cases.still`) from a second
    before the zero crossing: all die before the clock ends, on one tick (the two teams'
    last ships alike: a draw)."""
    config = api.PRESETS[preset]
    rules = api.rules_of(config)
    types = api.ship_types(ship_set=config.ship_set)
    largest = int(types.max_shield.argmax())
    start = int(environment(rules, 1).shrink_zero_ticks[0]) - 30
    ticks = rules.pools[0].episode_ticks
    generator = torch.Generator().manual_seed(0)
    state = api.reset(rules, types, api.uniforms(rules, 1, generator=generator))
    state = cases.fields_away(_at(rules, state, start))
    N = rules.num_ships
    angle = torch.arange(N) * (2.0 * math.pi / N)
    parked = dict(
        x=5.0 * torch.cos(angle)[None],
        y=5.0 * torch.sin(angle)[None],
        proper_speed=torch.full((1, N), 1e-3),
    )
    state = cases.edit_ships(
        state,
        ship_type=torch.full((1, N), largest, dtype=torch.int32),
        shield=torch.full((1, N), float(types.max_shield[largest])),
        shield_recharge_wait=torch.zeros((1, N)),
        **parked,
    )
    for tick in range(start, ticks):
        state = cases.edit_ships(state, **parked)
        state, events = api.step(
            rules,
            types,
            state,
            cases.still(1, N),
            api.spread(rules, 1, generator=generator),
            api.uniforms(rules, 1, generator=generator),
        )
        assert not bool(events.died[0].any()) or bool(events.died[0].all()), tick
        if bool(events.done[0]):
            break
    assert bool(events.died[0].all()) and bool(events.done[0])
    assert tick < ticks - 1  # before the clock's last tick
    assert int(events.result[0]) == 0


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
    """Elimination, halfway through the shrink: the boundary stands near half the arena
    radius, so a ship at 0.6 R_a is about 0.1 R_a past it (more than a ramp width: all of
    it)."""
    rules = api.rules_of(ELIMINATION)
    hold, close, _ = _schedule(rules)
    halfway = (hold + close) // 2 - 1  # the damage reads the boundary after the move
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
    """The clock's last tick: the boundary has closed, so every ship takes boundary damage;
    the survivors' health left decides."""
    rules = api.rules_of(ELIMINATION)
    _, state = cases.rules_and_state(ELIMINATION, batch=1)
    state = _at(rules, cases.fields_away(state), rules.pools[0].episode_ticks - 1)
    _, events = cases.tick(rules, state)
    reach = torch.hypot(state.ships.x, state.ships.y)
    hurt = events.boundary_damage[0] > 0
    assert hurt[reach[0] > 100.0].all() and hurt.all()
    assert bool(events.done[0])


# ------------------------------------------------------------------ the observation


def test_the_observation_shows_the_boundary_and_keeps_the_arena_radius():
    rules = api.rules_of(ELIMINATION)
    hold, close, ticks = _schedule(rules)
    middle = (hold + close) // 2
    zero = float(environment(rules, 1).shrink_zero_ticks[0])
    _, state = cases.rules_and_state(ELIMINATION, batch=1)
    _, obs, _ = api.perceive(rules, _at(rules, state, middle))
    globals_ = obs.globals[0]  # (2, columns): both views alike
    assert globals_[:, GLOBAL["arena_radius"]].tolist() == [1000.0, 1000.0]
    fraction = 1.0 - (1.0 - CLOSE) * (middle - hold) / (close - hold)
    torch.testing.assert_close(globals_[:, GLOBAL["boundary_fraction"]], torch.full((2,), fraction))
    left = (zero - middle) * rules.tick_length
    torch.testing.assert_close(globals_[:, GLOBAL["boundary_time_to_zero"]], torch.full((2,), left))
    # past the zero crossing the observation shows the boundary at 0, with no time left
    _, obs, _ = api.perceive(rules, _at(rules, state, close))
    assert obs.globals[0, :, GLOBAL["boundary_fraction"]].tolist() == [0.0, 0.0]
    assert obs.globals[0, :, GLOBAL["boundary_time_to_zero"]].tolist() == [0.0, 0.0]


# ------------------------------------------------------------------ the settings


def test_the_shrink_and_the_ramp_are_live_settings_of_the_map_panel():
    from bnb.play.tuning_panel import MAP_GROUPS

    for key in ("mode.shrink_hold_seconds", "mode.shrink_end_lead_seconds", "mode.shrink_law"):
        assert BY_KEY[key].group in MAP_GROUPS and BY_KEY[key].modes == ("elimination",)
    for key in ("rules.boundary_max_damage", "rules.boundary_ramp_width"):
        assert BY_KEY[key].group in MAP_GROUPS
    rules = tuned(
        api.rules_of(ELIMINATION),
        {
            "mode.shrink_hold_seconds": 6.0,
            "mode.shrink_end_lead_seconds": 4.0,
            "mode.shrink_law": 1,
            "rules.boundary_ramp_width": 50.0,
        },
    )
    live = live_of(rules)
    assert float(live["pool0.mode.shrink_hold_seconds"]) == 6.0
    assert float(live["pool0.mode.shrink_end_lead_seconds"]) == 4.0
    assert live["pool0.mode.shrink_law"].dtype == torch.int32
    rebuilt = with_live(api.tuning.static_of(rules), live)
    arena = environment(rebuilt, 1)
    assert float(arena.shrink_hold_ticks[0]) == pytest.approx(6.0 / rules.tick_length)
    ticks = rules.pools[0].episode_ticks
    assert float(arena.shrink_end_ticks[0]) == pytest.approx(ticks - 4.0 / rules.tick_length)
    # the live rules' zero crossing (tensors) is the static rules' (numbers)
    torch.testing.assert_close(arena.shrink_zero_ticks, environment(rules, 1).shrink_zero_ticks)
    assert float(rebuilt.boundary_ramp_width) == 50.0


# ------------------------------------------------------------------ the agents


def test_imugi_steers_by_the_boundary_as_it_stands():
    from bnb.controllers.imugi_combat import context
    from bnb.controllers.imugi_defaults import MODE_DEFAULTS
    from bnb.game.handling.characteristics import characteristics

    rules = api.rules_of(ELIMINATION)
    _, state = cases.rules_and_state(ELIMINATION, batch=1)
    hold, close, _ = _schedule(rules)
    middle = (hold + close) // 2
    _, obs, _ = api.perceive(rules, _at(rules, state, middle))
    view = type(obs)(**{name: part[:, 0] for name, part in vars(obs).items()})
    measured = characteristics(cases.TYPES, rules)
    fleet = context(view, measured, MODE_DEFAULTS.sets["deathmatch"])
    fraction = 1.0 - (1.0 - CLOSE) * (middle - hold) / (close - hold)
    assert float(fleet.boundary_radius[0, 0]) == pytest.approx(1000.0 * fraction)
    assert float(fleet.arena_radius[0, 0]) == 1000.0


def _view(rules, state, tick):
    _, obs, _ = api.perceive(rules, _at(rules, state, tick))
    return type(obs)(**{name: part[:, 0] for name, part in vars(obs).items()})


def test_imugi_tracks_the_boundary_closing_speed_and_keeps_it_on_a_re_read():
    from bnb.controllers.imugi_memory import fresh_carry, track_boundary

    rules = api.rules_of(ELIMINATION)
    _, state = cases.rules_and_state(ELIMINATION, batch=1)
    hold, close, _ = _schedule(rules)
    middle = (hold + close) // 2
    earlier, now = _view(rules, state, middle - 1), _view(rules, state, middle)
    reset = torch.zeros(1, dtype=torch.bool)
    carry = track_boundary(fresh_carry(1, 3, 3, "cpu"), earlier, reset, rules.tick_length)
    assert float(carry.boundary_speed[0]) == 0.0  # the episode's first look
    carry = track_boundary(carry, now, reset, rules.tick_length)
    want = 1000.0 * (1.0 - CLOSE) / ((close - hold) * rules.tick_length)
    assert float(carry.boundary_speed[0]) == pytest.approx(want, rel=1e-3)
    again = track_boundary(carry, now, reset, rules.tick_length)
    assert float(again.boundary_speed[0]) == float(carry.boundary_speed[0])
    fresh = track_boundary(carry, now, torch.ones(1, dtype=torch.bool), rules.tick_length)
    assert float(fresh.boundary_speed[0]) == 0.0


def test_imugi_widens_its_margin_for_a_closing_edge_and_drops_it_below_the_critical_radius():
    from bnb.controllers import imugi_combat
    from bnb.controllers.imugi_defaults import MODE_DEFAULTS
    from bnb.controllers.imugi_memory import fresh_carry
    from bnb.game.handling.characteristics import characteristics

    rules = api.rules_of(ELIMINATION)
    _, state = cases.rules_and_state(ELIMINATION, batch=1)
    view = _view(rules, state, 0)
    measured = characteristics(cases.TYPES, rules)
    parameters = MODE_DEFAULTS.sets["deathmatch"]

    def weights(speed: float):
        carry = fresh_carry(1, 3, 3, "cpu")
        carry = dataclasses.replace(carry, boundary_speed=torch.tensor([speed]))
        return imugi_combat.context(view, measured, parameters, carry=carry).boundary_weight

    still = weights(0.0)
    closing = weights(30.0)
    assert bool((closing >= still).all())  # a closing edge never steers less
    # the still edge's steer is the earlier rule exactly
    imugi_combat.MOVING_EDGE = False
    try:
        torch.testing.assert_close(weights(30.0), still)
    finally:
        imugi_combat.MOVING_EDGE = True
    # f(b) = sqrt(1 - b^2) + b arccos(-b): 1 at rest, pi when the edge is as fast
    assert (
        imugi_combat.CRITICAL_RADIUS_IN_TURN_RADII[0]
        < imugi_combat.CRITICAL_RADIUS_IN_TURN_RADII[1]
    )


def test_every_build_reads_the_boundary():
    from bnb.models.yemong.inputs import BOUNDARY_INPUTS, MODE_INPUTS, POSITION_CODE, global_inputs

    assert global_inputs(MODE_INPUTS) == POSITION_CODE + 8 + len(MODE_INPUTS) + BOUNDARY_INPUTS
    assert math.isclose(BOUNDARY_INPUTS, 10)
