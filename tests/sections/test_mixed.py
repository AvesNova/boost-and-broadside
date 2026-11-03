"""Mixed batches (mixed spec 6): several games in one batch equal each game run alone,
absent zones are inert, the seat stays hidden, and the runtime and the CLI preset build a
mixed batch."""

import dataclasses
import math

import modes_cases as cases
import pytest
import torch
from torch.utils import _pytree

from bnb.contracts import GameConfig, Pool
from bnb.contracts.obs import GLOBAL, SHIP, ZONE
from bnb.contracts.pools import environment_multiple, pool_game
from bnb.game import api
from bnb.game.layout import environment, slices
from bnb.learn import rewards
from bnb.models.relational import RelationalConfig

# The code under test draws from torch's default generator (model initialisation, the
# rollout, the scripted opponents): each test runs with it seeded and restored.
pytestmark = pytest.mark.usefixtures("global_generator")


SEATS = ("learner", "learner")
# One pool per mode, with arenas and clocks of their own (modes_cases' lively games), in
# the shared sizes of the base config.
BASE = GameConfig(3, 3, 900.0, episode_seconds=6.0, respawn_seconds=0.5)
GAMES = ("frontline", "team_deathmatch", "grouped", "sandbox")
VECTOR_ALIGNED = 32  # environments per game: a multiple of every CPU's vector loop width


# Each pool's own respawn wait (modes_cases' games all wait 0.5 s).
RESPAWN_S = dict(frontline=0.8, team_deathmatch=0.5, grouped=0.3, sandbox=1.0)


def _pools(share: float = 0.25, names=GAMES) -> tuple[Pool, ...]:
    def pool(name: str) -> Pool:
        game = cases.MODES[name]
        return Pool(
            name,
            share,
            SEATS,
            mode=game.mode,
            arena_radius=game.arena_radius,
            spawn_groups=game.spawn_groups,
            episode_seconds=game.episode_seconds,
            respawn_seconds=RESPAWN_S[name],
        )

    return tuple(pool(name) for name in names)


def _mixed(per_game: int):
    pools = _pools()
    rules = api.rules_of(BASE, pools, per_game * len(pools), **cases.LIVELY_RULES)
    alone = [cases.lively(pool_game(BASE, pool)) for pool in pools]
    return rules, alone


# ------------------------------------------------------------------ configuration


def test_pools_take_the_configs_values_or_the_modes_defaults():
    base = GameConfig(5, 10, 2600.0)
    pools = (
        Pool("frontline", 0.5, SEATS, mode="frontline"),
        Pool("squad", 0.25, SEATS),
        Pool(
            "battle", 0.25, SEATS, arena_radius=8200.0, spawn_groups=(2, 2, 1), episode_seconds=60.0
        ),
    )
    rules = api.rules_of(base, pools, 8)
    frontline, squad, battle = rules.pools
    assert [pool.environments for pool in rules.pools] == [4, 2, 2]
    assert frontline.mode.name == "frontline" and frontline.arena_radius == 2600.0
    assert frontline.respawn_ticks == 240 and squad.respawn_ticks == 150  # the modes' clocks
    assert battle.episode_ticks == 1800 and battle.mode.spawn_groups == (2, 2, 1)
    assert rules.num_zones == 5
    # the uniforms are as wide as the widest mode needs
    widths = [api.rules_of(pool_game(base, pool)).tick_uniform_count for pool in pools]
    assert rules.tick_uniform_count == max(widths)
    arena = environment(rules, 8)
    assert arena.arena_radius.tolist() == [2600.0] * 6 + [8200.0] * 2
    assert arena.respawn_ticks.tolist() == [240] * 4 + [150] * 4
    with pytest.raises(ValueError):
        environment(rules, 16)  # the pools hold 8 environments
    with pytest.raises(ValueError):
        api.rules_of(base, (Pool("bad", 1.0, SEATS, mode="frontline", spawn_groups=(5,)),), 8)


def test_adjacent_pools_of_one_game_are_one_slice_and_one_game_is_the_whole_batch():
    base = GameConfig(5, 10, 2600.0)
    lineup = (Pool("self_play", 0.5, SEATS), Pool("chaser", 0.5, ("learner", "chaser")))
    alone = api.rules_of(base, lineup, 8)
    assert alone == api.rules_of(base)  # one game: the single-game rules, at any size
    halves = tuple(dataclasses.replace(pool, share=0.25) for pool in lineup)
    mixed = api.rules_of(
        base,
        halves
        + tuple(
            dataclasses.replace(pool, name=pool.name + "/wide", arena_radius=8200.0)
            for pool in halves
        ),
        16,
    )
    assert [pool.environments for pool in mixed.pools] == [8, 8]
    assert len(slices(mixed, 16)) == 1  # one mode: the arena is per environment
    assert mixed.num_zones == 5  # a fixed shape, like the field count


def test_the_mixed_preset_crosses_the_lineup_with_the_training_mix():
    from bnb.runtime.train import LINEUP, mixed

    base, pools = mixed()
    assert len(pools) == 3 * len(LINEUP)
    assert math.isclose(sum(pool.share for pool in pools), 1.0)
    assert environment_multiple(pools) == 16 and environment_multiple(LINEUP) == 2
    rules = api.rules_of(base, pools, 1024)
    games = [
        (pool.environments, pool.mode.name, pool.arena_radius, pool.mode.spawn_groups)
        for pool in rules.pools
    ]
    assert games == [
        (512, "frontline", 2600.0, ()),
        (384, "team_deathmatch", 2600.0, ()),
        (128, "team_deathmatch", 8200.0, (2, 2, 1)),
    ]
    assert rules.num_fields == 10 and rules.num_ships == 10


# ------------------------------------------------------------------ equivalence


def _cut(tree, start: int, stop: int):
    return _pytree.tree_map_only(torch.Tensor, lambda x: x[start:stop], tree)


def _same(mixed, alone, where: str, tolerance: float = 0.0) -> None:
    """Every tensor of two pytrees equal: bit for bit, or within a tolerance (relative and
    absolute; integers and flags exactly)."""
    mixed_leaves, _ = _pytree.tree_flatten(mixed)
    alone_leaves, _ = _pytree.tree_flatten(alone)
    for index, (got, want) in enumerate(zip(mixed_leaves, alone_leaves, strict=True)):
        label = f"{where} leaf {index}"
        if tolerance and got.is_floating_point():
            torch.testing.assert_close(got, want, rtol=tolerance, atol=tolerance, msg=label)
        else:
            assert torch.equal(got, want), label


def _against_alone(
    pools: tuple[Pool, ...],
    per_game: int,
    ticks: int,
    seed: int,
    tolerance: float = 0.0,
    resync: bool = False,
) -> dict:
    """Lively random play of a mixed batch of these pools (per_game environments each,
    half of them fogged), against each pool's game run alone on its prefix of the same
    draws. Every tick, every environment's state, events, observation, truth, rewards and
    ledger must agree (bit for bit, or within the tolerance). With resync, each game alone
    steps from the mixed batch's previous state, so a tolerance is one tick's, not
    amplified by play. Returns what happened."""
    B = per_game * len(pools)
    rules = api.rules_of(BASE, pools, B, **cases.LIVELY_RULES)
    alone = [cases.lively(pool_game(BASE, pool)) for pool in pools]
    bounds = [(g * per_game, (g + 1) * per_game) for g in range(len(pools))]
    N, types = rules.num_ships, cases.LIVELY_TYPES
    generator = torch.Generator().manual_seed(seed)
    fogged = torch.rand(B, generator=generator) < 0.5
    uniforms = api.uniforms(rules, B, generator=generator)
    state = api.reset(rules, types, uniforms, fogged)
    parts = [
        api.reset(game, types, uniforms[start:stop, : game.tick_uniform_count], fogged[start:stop])
        for game, (start, stop) in zip(alone, bounds)
    ]
    for part, (start, stop) in zip(parts, bounds):
        _same(_cut(state, start, stop), part, "reset", tolerance)
    team = torch.arange(N) // rules.ships_per_team
    ledger = rewards.blank_ledger(B, N)
    ledgers = [rewards.blank_ledger(per_game, N) for _ in alone]
    config = rewards.RewardConfig()
    seen = dict(done=0, died=0, objective=0.0)
    for _ in range(ticks):
        action = cases.random_actions(B, N, generator)
        spread = api.spread(rules, B, generator=generator)
        uniforms = api.uniforms(rules, B, generator=generator)
        if resync:
            parts = [_cut(state, start, stop) for start, stop in bounds]
            ledgers = [_cut(ledger, start, stop) for start, stop in bounds]
        state, events = api.step(rules, types, state, action, spread, uniforms)
        state, obs, truth = api.perceive(rules, state)
        reward, ledger = rewards.reward_step(config, ledger, events, team)
        for g, (game, (start, stop)) in enumerate(zip(alone, bounds)):
            parts[g], part_events = api.step(
                game,
                types,
                parts[g],
                _cut(action, start, stop),
                spread[start:stop],
                uniforms[start:stop, : game.tick_uniform_count],
            )
            parts[g], part_obs, part_truth = api.perceive(game, parts[g])
            part_reward, ledgers[g] = rewards.reward_step(config, ledgers[g], part_events, team)
            name = pools[g].name
            _same(_cut(state, start, stop), parts[g], f"{name} state", tolerance)
            _same(_cut(events, start, stop), part_events, f"{name} events", tolerance)
            _same(_cut(obs, start, stop), part_obs, f"{name} obs", tolerance)
            _same(_cut(truth, start, stop), part_truth, f"{name} truth", tolerance)
            _same(
                (reward[start:stop], _cut(ledger, start, stop)),
                (part_reward, ledgers[g]),
                f"{name} rewards",
                tolerance,
            )
            if game.pools[0].mode.name != "frontline":
                assert not obs.zones[start:stop].any(), name  # absent: every column 0
        seen["done"] += int(events.done.sum())
        seen["died"] += int(events.died.sum())
        seen["objective"] += float(events.objective_progress.abs().sum())
    return seen


@pytest.mark.slow
def test_a_mixed_batch_equals_each_game_run_alone():
    """Eager on the CPU, bit for bit, every environment, through 300 lively ticks (kills,
    respawns at each pool's own wait, resets at each pool's own clock, moving meters).

    Each game's share is a multiple of the CPU's vector width: torch's eager kernels compute
    the tail of a tensor that is not one with scalar code, whose exp or sin can differ from
    the vectorised result in the last bit, and play then amplifies it."""
    seen = _against_alone(_pools(), VECTOR_ALIGNED, ticks=300, seed=7)
    assert seen["done"] > 64 and seen["died"] > 800 and seen["objective"] > 0, seen


def test_an_unaligned_mixed_batch_equals_each_game_alone_to_float32_rounding():
    """Pools of 5 environments (not whole vector widths): every tick, from the same state,
    within the spec's 1e-6."""
    seen = _against_alone(_pools(), 5, ticks=260, seed=9, tolerance=1e-6, resync=True)
    assert seen["done"] > 4 and seen["died"] > 50, seen


# Two arenas and clocks in each mode's slice (team deathmatch first, so Frontline is not
# the first pool).
ONE_SLICE_EACH = (
    Pool(
        "deathmatch_small",
        0.25,
        SEATS,
        arena_radius=600.0,
        episode_seconds=5.0,
        respawn_seconds=0.3,
    ),
    Pool(
        "deathmatch_large",
        0.25,
        SEATS,
        arena_radius=1500.0,
        episode_seconds=7.0,
        respawn_seconds=1.2,
    ),
    Pool(
        "frontline_small",
        0.25,
        SEATS,
        mode="frontline",
        arena_radius=300.0,
        episode_seconds=6.0,
        respawn_seconds=0.4,
    ),
    Pool(
        "frontline_large",
        0.25,
        SEATS,
        mode="frontline",
        arena_radius=700.0,
        episode_seconds=7.0,
        respawn_seconds=0.9,
    ),
)


def test_one_slice_of_several_arenas_and_clocks_equals_each_game_alone():
    """Adjacent pools of one mode share a slice; each environment still plays its own
    pool's arena, map scale, episode length and respawn wait (bit for bit)."""
    rules = api.rules_of(BASE, ONE_SLICE_EACH, 4 * VECTOR_ALIGNED)
    assert [part.mode.name for part in slices(rules, 4 * VECTOR_ALIGNED)] == [
        "team_deathmatch",
        "frontline",
    ]
    seen = _against_alone(ONE_SLICE_EACH, VECTOR_ALIGNED, ticks=240, seed=3)
    assert seen["done"] > 64 and seen["died"] > 500, seen


def test_spawn_protection_holds_where_frontline_is_not_the_first_pool():
    """A hit on a ship in its own spawn zone deals no damage in Frontline, also behind a
    team deathmatch pool; a team deathmatch ship at the same spot (the absent zones' centre)
    takes the hit."""
    from test_modes import _shot_at, _zone_centre

    from bnb.game.modes import frontline

    pools = (Pool("deathmatch", 0.5, SEATS), Pool("frontline", 0.5, SEATS, mode="frontline"))
    rules = api.rules_of(GameConfig(3, 7, 2600.0), pools, 4)
    state = api.reset(
        rules, cases.TYPES, api.uniforms(rules, 4, generator=torch.Generator().manual_seed(12))
    )
    centre = _zone_centre(state, frontline.TEAM0_SPAWN)  # (0, 0) in team deathmatch
    x, y = state.ships.x.clone(), state.ships.y.clone()
    x[:, 0], y[:, 0] = centre[:, 0], centre[:, 1]
    state = _shot_at(cases.edit_ships(state, x=x, y=y), target=0, shooter=3)
    _, events = cases.tick(rules, state)
    assert (events.damage[:2, 3, 0] > 0).all() and (events.shield_lost[:2, 0] > 0).all()
    assert (events.damage[2:] == 0).all() and (events.shield_lost[2:, 0] == 0).all()


@pytest.mark.slow
def test_a_compiled_mixed_tick_equals_each_game_compiled_alone():
    """Compiled, from mid-game states: one tick and its observation, per environment,
    within float32 reordering (1e-6)."""
    per_game = VECTOR_ALIGNED
    rules, alone = _mixed(per_game)
    B, N = per_game * len(alone), rules.num_ships
    types = cases.LIVELY_TYPES

    def tick(rules, state, action, spread, uniforms):
        state, events = api.step(rules, types, state, action, spread, uniforms)
        state, obs, _ = api.perceive(rules, state)
        return state, events, obs

    compiled = torch.compile(tick, fullgraph=True, dynamic=False)
    generator = torch.Generator().manual_seed(8)
    state = api.reset(rules, types, api.uniforms(rules, B, generator=generator))
    checked = 0
    for t in range(120):
        action = cases.random_actions(B, N, generator)
        spread = api.spread(rules, B, generator=generator)
        uniforms = api.uniforms(rules, B, generator=generator)
        if t % 30 == 29:
            got = compiled(rules, state, action, spread, uniforms)
            for g, game in enumerate(alone):
                start, stop = g * per_game, (g + 1) * per_game
                want = compiled(
                    game,
                    _cut(state, start, stop),
                    _cut(action, start, stop),
                    spread[start:stop],
                    uniforms[start:stop, : game.tick_uniform_count],
                )
                _same(_cut(got, start, stop), want, game.pools[0].mode.name, 1e-6)
                checked += 1
        state, _ = api.step(rules, types, state, action, spread, uniforms)
    assert checked == 4 * len(alone)


def test_absent_zones_stay_absent_and_change_nothing():
    """A team deathmatch environment of a mixed batch carries five zone slots that stay
    zero through play; filling them with garbage changes no ship, event or score."""
    per_game = 4
    rules, _ = _mixed(per_game)
    B, N = per_game * 4, rules.num_ships
    generator = torch.Generator().manual_seed(11)
    types = cases.LIVELY_TYPES
    state = api.reset(rules, types, api.uniforms(rules, B, generator=generator))
    others = slice(per_game, B)  # the modes without zones
    for _ in range(60):
        action = cases.random_actions(B, N, generator)
        spread = api.spread(rules, B, generator=generator)
        uniforms = api.uniforms(rules, B, generator=generator)
        garbage = {
            name: torch.where(
                torch.arange(B)[:, None] >= per_game,
                torch.rand(state.mode.zone_x.shape, generator=generator) * 1000.0,
                getattr(state.mode, name),
            )
            for name in ("zone_x", "zone_y", "zone_radius", "progress")
        }
        dirty = dataclasses.replace(state, mode=dataclasses.replace(state.mode, **garbage))
        state, events = api.step(rules, types, state, action, spread, uniforms)
        noisy, noisy_events = api.step(rules, types, dirty, action, spread, uniforms)
        _same(_cut(state.ships, per_game, B), _cut(noisy.ships, per_game, B), "ships")
        _same(_cut(events, per_game, B), _cut(noisy_events, per_game, B), "events")
        assert torch.equal(state.mode.score[others], noisy.mode.score[others])
        for name in ("zone_x", "zone_y", "zone_radius", "progress", "role", "attack_ledger"):
            assert not getattr(state.mode, name)[others].any(), name


# ------------------------------------------------------------------ the seat


def test_a_mixed_batch_does_not_reveal_the_seat():
    """The seat tests of every mode, in one mixed batch: own ships sit the same way from
    either seat; at a match's start both views read the same globals and zone rows (absent
    ones zero in both), and a Frontline ring's handedness is either way half the time."""
    per_game = 1024
    rules, _ = _mixed(per_game)
    state = api.reset(
        rules,
        cases.LIVELY_TYPES,
        api.uniforms(rules, per_game * 4, generator=torch.Generator().manual_seed(23)),
    )
    _, obs, _ = api.perceive(rules, state)
    S = rules.ships_per_team
    arena_radius = environment(rules, per_game * 4).arena_radius
    assert torch.equal(obs.globals[:, 0], obs.globals[:, 1])
    shown = [
        ZONE[name]
        for name in (
            "radius",
            "role",
            "progress",
            "direction",
            "offensive_distance",
            "defensive_distance",
            "present",
        )
    ]
    assert torch.equal(obs.zones[:, 0][..., shown], obs.zones[:, 1][..., shown])
    for part in slices(rules, per_game * 4):
        own = obs.ships[part.start : part.stop, :, :S]
        dx = own[..., 0, SHIP["x"]] - own[..., SHIP["x"]].mean(-1)
        dy = own[..., 0, SHIP["y"]] - own[..., SHIP["y"]].mean(-1)
        angle = torch.atan2(dy, dx)
        spread = 1.0 - torch.hypot(angle.cos().mean(0), angle.sin().mean(0))
        assert (spread > 0.5).all(), (part.mode.name, spread)
        radius = float(arena_radius[part.start])
        mean = own[..., [SHIP["x"], SHIP["y"]]].mean((0, 2))
        assert (mean.abs() < 0.1 * radius).all(), (part.mode.name, mean)
        reach = torch.hypot(own[..., SHIP["x"]], own[..., SHIP["y"]]).mean((0, 2))
        assert abs(float(reach[0] - reach[1])) < 0.05 * radius, part.mode.name
        zones = obs.zones[part.start : part.stop]
        if part.mode.name == "frontline":
            assert (zones[..., ZONE["present"]] == 1).all()
            angle = torch.atan2(zones[..., ZONE["y"]], zones[..., ZONE["x"]])
            step = torch.remainder(angle[..., 1] - angle[..., 0] + math.pi, 2 * math.pi) - math.pi
            counterclockwise = (step > 0).float().mean(0)
            assert ((counterclockwise > 0.4) & (counterclockwise < 0.6)).all()
        else:
            assert not zones.any(), part.mode.name
        one_hot = obs.globals[part.start : part.stop, :, GLOBAL["mode_frontline"]]
        assert (one_hot == float(part.mode.name == "frontline")).all()


# ------------------------------------------------------------------ the runtime


def test_the_rollout_runs_a_mixed_batch_and_reports_every_pool():
    from bnb.learn.ppo import LearnerConfig
    from bnb.runtime.opponents import OPPONENTS
    from bnb.runtime.train import LEARNER, TrainConfig, Trainer

    pools = tuple(
        dataclasses.replace(pool, seats=(LEARNER, other))
        for pool, other in zip(_pools(), ("chaser", "random", LEARNER, OPPONENTS))
    )
    cfg = TrainConfig(
        game=BASE,
        environments=16,
        length=32,
        model="relational",
        relational=RelationalConfig(hidden=32),
        pools=pools,
        compiled=False,
        learner=LearnerConfig(minibatches=2, micro_batch_rows=30),
    )
    trainer = Trainer(cfg, torch.device("cpu"))
    store, bootstraps = trainer.collect()
    metrics = trainer.pool_metrics(store)
    assert set(metrics) == set(GAMES)
    zones = trainer.rollout.carry["obs"]["zones"]
    assert zones.shape[2] == 5
    assert (zones[:4, ..., ZONE["present"]] == 1).all() and not zones[4:].any()
    globals_ = trainer.rollout.carry["obs"]["globals"]
    assert (
        globals_[..., GLOBAL["arena_radius"]][:, 0].tolist()
        == [300.0] * 4 + [900.0] * 4 + [1200.0] * 4 + [300.0] * 4
    )
    stats = trainer.ppo.update(store, bootstraps)
    assert math.isfinite(stats["loss/policy"]) and math.isfinite(stats["loss/value"])


def test_the_rollout_staggers_each_start_clock_within_its_own_episode():
    from bnb.controllers.scripted import Random
    from bnb.runtime.rollout import Rollout

    pools = (
        Pool("short", 0.5, ("random", "random"), episode_seconds=2.0),
        Pool("long", 0.5, ("random", "random"), episode_seconds=20.0),
    )
    rollout = Rollout(
        BASE,
        api.ship_types(),
        pools,
        {"random": Random(BASE.ships_per_team)},
        "random",
        256,
        8,
        torch.device("cpu"),
        compiled=False,
    )
    tick = rollout.carry["state"].match.episode_tick
    length = environment(rollout.rules, 256).episode_ticks
    assert (tick < length).all() and (tick >= 0).all()
    assert tick[128:].max() >= length[0] and tick[:128].max() > length[0] // 2  # spread out


def test_a_mode_without_zones_passes_its_absent_slots_through():
    """Team deathmatch alone: the tick hands back the absent slots it was given (no copy
    of them), through deaths and resets."""
    rules = cases.lively(cases.MODES["team_deathmatch"])
    state = api.reset(rules, cases.LIVELY_TYPES, api.uniforms(rules, 8))
    zone_leaves = (
        "zone_x",
        "zone_y",
        "zone_radius",
        "role",
        "progress",
        "movement",
        "attack_ledger",
        "defense_ledger",
        "front",
        "handedness",
        "ring_rotation",
    )
    for _ in range(200):
        after, _ = cases.tick(rules, state)
        assert all(getattr(after.mode, name) is getattr(state.mode, name) for name in zone_leaves)
        state = after
    assert not any(getattr(state.mode, name).any() for name in zone_leaves)
