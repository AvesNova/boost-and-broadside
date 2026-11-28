"""The shrinking zone and the end of a game (author 2026-11-28): the zone's radius against time,
boundary damage against the current radius, the derived longest game, the observation's zone
fields, spawns clear of the zone, the scripted agent's boundary term on the current zone and
full games that always end by elimination before the bound. Each test names the rule it
guards."""

import math

import pytest
import torch

from lean.contract import GLOBAL_COLUMNS
from lean.game import GameConfig, card, core, observe, reset, step, tick
from lean.scripted import chaser_act
from lean.scripted import paths as paths_module
from lean.scripted.opponent import own_components

torch.set_num_threads(1)

CPU = torch.device("cpu")
PLAIN = GameConfig(field_count=0, fog=False)
H = PLAIN.tick_length
# The ship row's distance inside the boundary (core.observe: 4 symlog scalars, then the angle,
# its cos and sin, store, shield fraction, shield, wait, cooldown, then this one).
BOUNDARY_SCALAR = 12


def neutral(state):
    return torch.zeros(state.x.shape + (3,), dtype=torch.long)


def no_noise(state):
    return {"spread": torch.zeros(state.x.shape + (2,))}


def radius_at(config, ticks):
    state = reset(config, 1, torch.tensor([[1, 1]]), torch.Generator().manual_seed(0), CPU)
    return core.zone_at_tick(state.constants, torch.as_tensor(ticks))[2][:, 0]


def test_the_zone_shrinks_linearly_from_the_deal_and_holds_at_minus_the_ramp():
    """Rule: the radius is 4,200 m at the deal, shrinks at 35 m/s (0 at 120 s), reaches minus
    the ramp width (-300 m) at the stage's end (128.6 s) and holds there; the centre is the
    map's centre; a fixed arena (shrink speed 0) never moves."""
    ticks = torch.tensor([0, 30, 900, 3600, 3857, 3858, 10_000])
    radius = radius_at(PLAIN, ticks)
    expected = (4200.0 - 35.0 * ticks.double() * H).clamp_min(-300.0)
    assert torch.allclose(radius.double(), expected, atol=2e-3)
    assert abs(float(radius[3])) < 2e-3  # 0 at 120 s
    assert core.zone_stage_end(PLAIN) == pytest.approx(4500.0 / 35.0)
    assert float(radius[-2]) == float(radius[-1]) == -300.0
    fixed = GameConfig(field_count=0, fog=False, zone_shrink_speed=0.0)
    assert (radius_at(fixed, ticks) == 4200.0).all()
    assert math.isinf(core.zone_stage_end(fixed))


def one_ship_at(distance, tick_count, shield=100.0, config=PLAIN):
    """1v1, ship 0 at `distance` from the centre flying tangentially, ship 1 at the centre."""
    state = reset(config, 1, torch.tensor([[1, 1]]), torch.Generator().manual_seed(0), CPU)
    return core.with_velocity(
        state.replace(
            x=torch.tensor([[distance, 0.0]]),
            y=torch.tensor([[0.0, 0.0]]),
            heading_cos=torch.tensor([[0.0, 1.0]]),
            heading_sin=torch.tensor([[1.0, 0.0]]),
            shield=torch.tensor([[shield, shield]]),
            tick=torch.tensor([tick_count]),
        )
    )


def test_boundary_damage_follows_the_current_radius():
    """Rule: the ramp and the damage are unchanged, measured from the zone's edge at the tick's
    end: none inside it, half way up the ramp a half (smootherstep), full beyond the ramp; at
    the deal the same distance is deep inside."""
    per_tick = card.BOUNDARY_MAX_DAMAGE * H
    at_1000 = round((4200.0 - 1000.0) / (35.0 * H)) - 1  # the step ends where the radius is 1,000
    for offset, share in ((-100.0, 0.0), (150.0, 0.5), (400.0, 1.0)):
        state = one_ship_at(1000.0 + offset, at_1000)
        _, out = step(PLAIN, state, neutral(state), no_noise(state))
        # The ship moves about 5 m tangentially in the tick: its distance changes by < 0.01 m.
        assert float(out.boundary_damage[0, 0]) == pytest.approx(share * per_tick, abs=2e-3)
        early = one_ship_at(1000.0 + offset, 0)
        _, out = step(PLAIN, early, neutral(early), no_noise(early))
        assert float(out.boundary_damage[0, 0]) == 0.0


def test_the_longest_game_is_derived_and_a_full_depth_ship_never_recharges():
    """Rule: the longest game is the tick full damage starts everywhere plus a full shield over
    the net boundary damage (no recharge while hurt, no other healing): 3,858 + 75 + 1 = 3,934
    ticks (131.1 s) by default. A quiet ship (straight, coast, no trigger) at full depth loses
    exactly the full damage per tick, and two such ships with full shields from the tick before
    full depth both die by the bound (a draw), within two ticks of it (the bound is tight)."""
    bound = core.longest_game_ticks(PLAIN)
    full = math.ceil(4500.0 / (35.0 * H) - 1e-9)
    shield = 100.0
    per_tick = card.BOUNDARY_MAX_DAMAGE * H
    assert full == 3858 and bound == full + math.ceil(shield / per_tick - 1e-9) + 1 == 3934
    state = reset(PLAIN, 1, torch.tensor([[1, 1]]), torch.Generator().manual_seed(0), CPU)
    c = state.constants
    assert c["max_shield"] == shield
    # The net rate at full depth: the full damage, positive for every type (a hurt ship never
    # recharges, and the recharge rate alone is below the damage too).
    assert card.BOUNDARY_MAX_DAMAGE > 0.0 and c["shield_rate"] < card.BOUNDARY_MAX_DAMAGE
    state = one_ship_at(200.0, full - 1)
    state = state.replace(x=torch.tensor([[200.0, -200.0]]))
    shields, done_at, result = [], None, None
    for _ in range(200):
        state, out = step(PLAIN, state, neutral(state), no_noise(state))
        shields.append(float(state.shield[0, 0]))
        if bool(out.done[0]):
            done_at, result = int(state.tick[0]), int(out.result[0])
            break
    assert shields[0] == pytest.approx(shield - per_tick, abs=1e-4)
    assert shields[10] == pytest.approx(shield - 11 * per_tick, abs=1e-3)  # no recharge
    assert done_at is not None and bound - 2 <= done_at <= bound and result == 2
    with pytest.raises(ValueError):
        core.longest_game_ticks(GameConfig(zone_shrink_speed=0.0))


def test_the_observation_reads_the_current_and_next_circle_and_the_time_left():
    """Rule: the game token reads the current circle (centre, radius), the next (the stage's
    end: the same centre, minus the ramp) and the time to the stage's end; each ship's boundary
    scalar and the map-edge target are the signed distance to the current edge."""
    column = {name: index for index, name in enumerate(GLOBAL_COLUMNS)}
    for tick_count in (0, 1800, 3600, 5000):
        state = one_ship_at(1500.0, tick_count)
        globals_ = observe(PLAIN, state, 0).globals[0]
        seconds = tick_count * H
        radius = max(4200.0 - 35.0 * seconds, -300.0)
        left = max(4500.0 / 35.0 - seconds, 0.0)
        for name, value in (
            ("zone_x", 0.0),
            ("zone_y", 0.0),
            ("zone_radius", radius / core.ZONE_SCALE),
            ("next_zone_x", 0.0),
            ("next_zone_y", 0.0),
            ("next_zone_radius", -300.0 / core.ZONE_SCALE),
            ("stage_time_left", left / core.ZONE_TIME_SCALE),
            ("own_alive", 1.0),
            ("enemy_alive", 1.0),
        ):
            assert float(globals_[column[name]]) == pytest.approx(value, abs=1e-5), name
        ship = observe(PLAIN, state, 0).ship[0, 0]
        assert float(ship[BOUNDARY_SCALAR]) == pytest.approx(
            min(max((radius - 1500.0) / core.BOUNDARY_CLIP, -1.0), 1.0), abs=1e-5
        )
        edge = core.geometry_record(PLAIN, state)[0, 0, core.GEOMETRY_COLUMNS.index("edge")]
        assert float(edge) == pytest.approx(1500.0 - radius, abs=1e-2)


@pytest.mark.parametrize("spawn", ["mixed", "sides"])
def test_spawns_are_clear_of_the_zone_at_the_deal_and_in_the_first_seconds(spawn):
    """Rule: every ship spawns at least (1 - 0.85) x 4,200 = 630 m inside the zone, and no ship
    flying straight (any spawn speed, the training spawn) takes boundary damage in the first two
    seconds (the edge rule turns spawns near the edge inward; measured 2026-11-28 over 16,384
    ships, "mixed": the first at 2.57 s with the zone, 2.80 s on a fixed arena; "sides": none
    in 5 s)."""
    config = GameConfig(spawn=spawn, field_count=0, fog=False, training_spawn=True)
    envs = 512
    sizes = torch.tensor([[4, 4]]).expand(envs, 2)
    state = reset(config, envs, sizes, torch.Generator().manual_seed(5), CPU)
    clearance = -core.boundary_distance(state.constants, state.tick, state.x, state.y)
    assert float(clearance.min()) >= (1.0 - card.SPAWN_AREA_FRACTION) * config.arena_radius - 1e-2
    generator = torch.Generator().manual_seed(6)
    for _ in range(60):
        state, out = tick(config, state, neutral(state), generator)
        assert float(out.boundary_damage.max()) == 0.0


def test_the_scripted_boundary_term_reads_the_current_zone():
    """Rule (one source of truth): the scripted agent's boundary rate is measured against the
    zone at each look-ahead point's time (core.zone_circle): a path 1,500 m from the centre is
    clear of the edge band at the deal and deep past the edge when the radius is 1,000 m."""
    state = reset(PLAIN, 1, torch.tensor([[1, 1]]), torch.Generator().manual_seed(0), CPU)
    points = len(paths_module.LOOK_AHEAD)
    shape = (1, 2, 1, points)
    plain = ("y", "path_sin", "nose_cos", "nose_sin", "world_speed", "energy", "log_index")
    paths = paths_module.Paths(
        x=torch.full(shape, 1500.0),
        path_cos=torch.ones(shape),
        speed=torch.full(shape, 150.0),
        now=None,
        **{name: torch.zeros(shape) for name in plain},
    )
    early = own_components(state.constants, paths, 2.2, torch.tensor([0]))
    late_tick = round((4200.0 - 1000.0) / (35.0 * H))
    late = own_components(state.constants, paths, 2.2, torch.tensor([late_tick]))
    assert float(early.boundary.max()) == 0.0
    assert float(late.boundary.min()) > 1.0  # past the edge: the rate grows beyond the band


@pytest.mark.parametrize("size", [1, 2])
def test_a_full_game_always_ends_by_elimination_before_the_bound(size):
    """Rule: with no clock every game ends by a wipe (one team or both) by the derived bound:
    chaser games with fields and fog, the default zone, played to their end."""
    config = GameConfig()
    envs = 8
    generator = torch.Generator().manual_seed(size)
    state = reset(config, envs, torch.tensor([[size, size]]).expand(envs, 2), generator, CPU)
    bound = core.longest_game_ticks(config)
    running = torch.ones(envs, dtype=torch.bool)
    length = torch.zeros(envs, dtype=torch.long)
    for _ in range(bound):
        action = chaser_act(state, 0) + chaser_act(state, 1)
        state, out = tick(config, state, action, generator)
        ended = out.done & running
        length = torch.where(ended, state.tick, length)
        assert (out.result[ended] >= 0).all()
        left = torch.stack([(state.alive & (state.team == k)).sum(-1) for k in (0, 1)], -1)
        assert (left.min(-1).values[ended] == 0).all()  # a wipe
        running = running & ~out.done
        if not bool(running.any()):
            break
    assert not bool(running.any()) and int(length.max()) <= bound
