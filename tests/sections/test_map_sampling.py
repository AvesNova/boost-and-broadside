"""Map-size sampling (game v17): each training environment's arena scale drawn at every
reset from a log-normal truncated to its bounds, the episode's radius the pool's times it
at least the mode's minimum, and every map length following; ratings, evaluation and play
keep their games' sizes."""

import dataclasses
import math

import pytest
import torch
from torch.utils import _pytree

from bnb.contracts import Action, ArenaScale, GameConfig, Pool
from bnb.contracts.obs import GLOBAL
from bnb.game import api
from bnb.game.layout import draw_arena_scale, environment, sized
from bnb.runs.serialise import from_json, to_json
from bnb.runtime import lanes
from bnb.runtime.train import TrainConfig, training_pools

SEATS = ("a", "b")
SAMPLED = ArenaScale()  # sigma 0.35, [0.5, 2]
REAL_SCALE = api.ship_types(ship_set="real-scale")


def _rules(name: str, batch: int, scale: ArenaScale | None = SAMPLED):
    game = api.PRESETS[name]
    return api.rules_of(game, (Pool("p", 1.0, SEATS, arena_scale=scale),), batch)


def _truncated_moments(sigma: float, low: float, high: float) -> tuple[float, float]:
    """Mean and standard deviation of ln(scale): Normal(0, sigma) truncated to the bounds'
    logs."""
    a, b = math.log(low) / sigma, math.log(high) / sigma
    pdf = lambda x: math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)  # noqa: E731
    cdf = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))  # noqa: E731
    mass = cdf(b) - cdf(a)
    mean = (pdf(a) - pdf(b)) / mass
    variance = 1 + (a * pdf(a) - b * pdf(b)) / mass - mean**2
    return sigma * mean, sigma * math.sqrt(variance)


def test_the_scale_is_a_truncated_log_normal_from_one_uniform():
    batch = 200_000
    rules = _rules("deathmatch", batch)
    arena = environment(rules, batch)
    uniform = torch.rand(batch, generator=torch.Generator().manual_seed(0))
    scale = draw_arena_scale(arena, uniform)
    assert scale.min() >= 0.5 - 1e-5 and scale.max() <= 2.0 + 1e-5
    assert scale.min() < 0.52 and scale.max() > 1.9  # the bounds are reached
    mean, deviation = _truncated_moments(0.35, 0.5, 2.0)
    logs = scale.double().log()
    assert float(logs.mean()) == pytest.approx(mean, abs=3e-3)
    assert float(logs.std()) == pytest.approx(deviation, rel=1e-2)
    # monotone in the uniform: the median uniform gives scale 1, the ends the bounds
    ends = draw_arena_scale(environment(rules, 3), torch.tensor([0.0, 0.5, 1.0 - 1e-7]))
    torch.testing.assert_close(ends, torch.tensor([0.5, 1.0, 2.0]), atol=1e-4, rtol=0)


def test_a_pool_without_sampling_keeps_its_arena_and_its_draws():
    fixed = _rules("deathmatch", 8, scale=None)
    plain = api.rules_of(api.PRESETS["deathmatch"])
    assert not fixed.samples_arena and fixed.tick_uniform_count == plain.tick_uniform_count
    assert _rules("deathmatch", 8).tick_uniform_count == plain.tick_uniform_count
    arena = environment(fixed, 4)
    torch.testing.assert_close(draw_arena_scale(arena, torch.linspace(0.0, 0.9, 4)), torch.ones(4))
    torch.testing.assert_close(sized(arena, torch.full((4,), 0.5)).arena_radius, arena.arena_radius)


def test_the_radius_keeps_the_modes_floor():
    """Real-scale team deathmatch (4200 m) keeps its 3000 m minimum at any scale; the
    radius is `api.arena_radius_for`'s at the drawn scale."""
    batch = 512
    game = api.PRESETS["real_scale_deathmatch"]
    rules = _rules("real_scale_deathmatch", batch)
    state = api.reset(
        rules, REAL_SCALE, api.uniforms(rules, batch, generator=torch.Generator().manual_seed(1))
    )
    scale = state.match.arena_scale
    radius = sized(environment(rules, batch), scale).arena_radius
    expected = torch.tensor(
        [api.arena_radius_for(game, game.ships_per_team, float(s)) for s in scale]
    )
    torch.testing.assert_close(
        radius, expected, atol=0.06, rtol=0
    )  # arena_radius_for rounds to 0.1 m
    assert radius.min() >= 3000.0 - 1e-3 and (radius == 3000.0).any()


def test_frontline_zones_keep_their_floors_on_sampled_maps():
    batch = 256
    rules = _rules("real_scale_frontline", batch)
    state = api.reset(
        rules, REAL_SCALE, api.uniforms(rules, batch, generator=torch.Generator().manual_seed(2))
    )
    mode_rules = rules.pools[0].mode
    floor = mode_rules.zone_radius_floor * mode_rules.set_map_scale
    radius = sized(environment(rules, batch), state.match.arena_scale).arena_radius
    zone = state.mode.zone_radius[:, 0]
    torch.testing.assert_close(
        zone, torch.clamp(mode_rules.zone_radius * radius / 2600.0, min=floor)
    )
    assert (zone == floor).any() and (zone > floor).any()


def test_a_reset_draws_a_new_arena_and_the_shapes_stay():
    """Short episodes: every reset draws the environment's scale anew, the observation's
    arena radius is the episode's, and no tensor changes shape."""
    batch = 64
    config = dataclasses.replace(api.PRESETS["deathmatch"], episode_seconds=0.2)
    rules = api.rules_of(config, (Pool("p", 1.0, SEATS, arena_scale=SAMPLED),), batch)
    generator = torch.Generator().manual_seed(3)
    types = api.ship_types()
    state = api.reset(rules, types, api.uniforms(rules, batch, generator=generator))
    shapes = [leaf.shape for leaf in _pytree.tree_leaves(state)]
    changed = 0
    for _ in range(14):
        before = state.match.arena_scale
        action = torch.zeros(batch, rules.num_ships, 3, dtype=torch.int32)
        state, events = api.step(
            rules,
            types,
            state,
            Action.unpack(action),
            api.spread(rules, batch, generator=generator),
            api.uniforms(rules, batch, generator=generator),
        )
        state, obs, _ = api.perceive(rules, state)
        after = state.match.arena_scale
        assert torch.equal(after[~events.done], before[~events.done])
        changed += int((after[events.done] != before[events.done]).sum())
        radius = sized(environment(rules, batch), after).arena_radius
        torch.testing.assert_close(obs.globals[:, 0, GLOBAL["arena_radius"]], radius)
        assert [leaf.shape for leaf in _pytree.tree_leaves(state)] == shapes
    assert changed > batch // 2


def test_training_samples_and_the_lanes_and_evaluation_do_not():
    config = TrainConfig(api.PRESETS["deathmatch"])
    assert config.arena_scale == ArenaScale(0.35, 0.5, 2.0)
    pools = training_pools(config)
    assert all(pool.arena_scale == config.arena_scale for pool in pools)
    modes = lanes.mode_games(pools, config.game, lambda pool, game: pool.mode or game.mode)
    every, _, _ = lanes.lane_pools(pools, modes, 64, lanes.LaneConfig(environments=64))
    assert [pool.arena_scale for pool in every[len(pools) :]] == [None] * (len(every) - len(pools))
    off = TrainConfig(api.PRESETS["deathmatch"], arena_scale=None)
    assert all(pool.arena_scale is None for pool in training_pools(off))
    # the setting round-trips through the run's config.json
    read, _ = from_json(TrainConfig, to_json(config))
    assert read.arena_scale == config.arena_scale
    assert from_json(TrainConfig, to_json(off))[0].arena_scale is None


def test_play_and_the_plain_game_keep_their_size():
    rules = api.rules_of(GameConfig(5, 10, 2600.0))
    assert not rules.samples_arena
    state = api.reset(
        rules, api.ship_types(), api.uniforms(rules, 4, generator=torch.Generator().manual_seed(4))
    )
    assert torch.equal(state.match.arena_scale, torch.ones(4))
