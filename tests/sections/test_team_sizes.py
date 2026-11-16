"""Team-size sampling (arm A2, `bnb.contracts.TeamSizes`): each episode plays two team sizes,
equal or not, and a field count drawn together from the pool's table (within each mode's
smallest team), the rest of each team's slots benched and the other fields absent, at the
batch's one static size. The draw is keyed by (seed, environment, episode) and moves no
other draw; a reset spawns and benches by the next episode's sizes; a benched slot is
neither valued nor rewarded; deathmatch's tickets follow each enemy team's drawn size (the
freeze's 4 N^(4/3) to the nearest 5); a rollout without it, or with every slot drawn, is
the game as before; the training log gets the drawn sizes per mode; BC and RL both train
with it, the Yemong masking the benched ships and absent fields."""

import dataclasses

import pytest
import torch

from bnb.contracts import GameConfig, Pool, TeamSizes
from bnb.contracts.obs import SHIP
from bnb.controllers.scripted import Chaser, Random
from bnb.game import api
from bnb.game.handling.characteristics import characteristics
from bnb.game.modes import phase
from bnb.learn import rewards
from bnb.runs.serialise import from_json, to_json
from bnb.runtime import lanes
from bnb.runtime.rollout import Rollout
from bnb.runtime.train import (
    TrainConfig,
    samples_field_counts,
    samples_team_sizes,
    training_pools,
)

CPU = torch.device("cpu")
TYPES = api.ship_types()


@pytest.fixture(autouse=True)
def _default_generator_untouched():
    with torch.random.fork_rng(devices=[]):
        yield


# ------------------------------------------------------------------ the setting


def _pairs(sizes: TeamSizes, minimum: int = 1) -> dict:
    return sizes.pair_probabilities(5, minimum)


def test_the_default_plays_5v5_in_half_the_games_and_spreads_the_rest_over_the_pairs():
    pairs = _pairs(TeamSizes())
    assert len(pairs) == 15 and pairs[(5, 5)] == pytest.approx(0.5)
    assert all(
        chance == pytest.approx(0.5 / 14) for pair, chance in pairs.items() if pair != (5, 5)
    )
    frontline = _pairs(TeamSizes(), minimum=3)
    assert set(frontline) == {(5, 5), (5, 4), (5, 3), (4, 4), (4, 3), (3, 3)}
    assert frontline[(5, 5)] == pytest.approx(0.5) and frontline[(4, 3)] == pytest.approx(0.1)
    even = _pairs(TeamSizes(uneven=False))
    assert set(even) == {(n, n) for n in range(1, 6)} and even[(1, 1)] == pytest.approx(0.125)


def test_the_other_rules():
    independent = _pairs(TeamSizes(rule="independent_teams"))
    assert independent[(5, 5)] == pytest.approx(0.5)
    team = (1 - 0.5**0.5) / 4  # each smaller size's chance for one team
    assert independent[(5, 1)] == pytest.approx(2 * 0.5**0.5 * team)
    assert independent[(2, 2)] == pytest.approx(team**2)
    near = _pairs(TeamSizes(rule="near_full", decay=0.5))
    assert near[(5, 5)] == pytest.approx(0.5)
    assert near[(5, 4)] == pytest.approx(2 * near[(5, 3)]) == pytest.approx(2 * near[(4, 4)])
    table = _pairs(TeamSizes(rule="table", pairs=((5, 5, 3.0), (5, 3, 1.0), (2, 2, 1.0))), 3)
    assert table == pytest.approx({(5, 5): 0.75, (5, 3): 0.25})  # 2v2 is below Frontline's 3
    for sizes in (TeamSizes(), TeamSizes(rule="independent_teams"), TeamSizes(rule="near_full")):
        for minimum in (1, 3):
            assert sum(_pairs(sizes, minimum).values()) == pytest.approx(1.0)


def test_the_table_splits_uneven_pairs_over_both_sides_and_sets_one_field_per_ship():
    table = TeamSizes().table(5, 10, 3)
    assert sum(row[3] for row in table) == pytest.approx(1.0)
    rows = {row[:2]: row for row in table}
    assert rows[(5, 3)][3] == pytest.approx(rows[(3, 5)][3]) == pytest.approx(0.05)
    assert {pair: row[2] for pair, row in rows.items()} == {
        (5, 5): 10, (5, 4): 10, (4, 5): 10, (5, 3): 8, (3, 5): 8, (4, 4): 8,
        (4, 3): 8, (3, 4): 8, (3, 3): 6,
    }  # fmt: skip
    assert TeamSizes().field_counts(2, 5, 10) == {2: 1.0}  # 1v1
    assert TeamSizes().field_counts(3, 5, 9) == {3: 1.0}  # an odd slot count plays any count
    assert TeamSizes(field_spread=1).field_counts(10, 5, 10) == {8: 0.5, 10: 0.5}
    assert TeamSizes(field_spread=1).field_counts(4, 5, 10) == pytest.approx(
        {2: 1 / 3, 4: 1 / 3, 6: 1 / 3}
    )
    assert {row[2] for row in TeamSizes(fields="game").table(5, 10)} == {-1}


@pytest.mark.parametrize(
    "build",
    [
        lambda: TeamSizes(rule="other"),
        lambda: TeamSizes(fields="other"),
        lambda: TeamSizes(full_share=0.0),
        lambda: TeamSizes(full_share=1.5),
        lambda: TeamSizes(low=0),
        lambda: TeamSizes(field_spread=-1),
        lambda: TeamSizes(pairs=((5, 5, 1.0),)),
        lambda: TeamSizes(rule="table"),
        lambda: TeamSizes(rule="table", pairs=((3, 4, 1.0),)),
        lambda: TeamSizes(rule="table", pairs=((5, 5, 0.0),)),
        lambda: TeamSizes(rule="independent_teams", uneven=False),
        lambda: TeamSizes(rule="table", pairs=((6, 5, 1.0),)).pair_probabilities(5),
        lambda: TeamSizes(low=6).pair_probabilities(5),
        lambda: TeamSizes(rule="table", pairs=((2, 2, 1.0),)).pair_probabilities(5, 3),
    ],
)
def test_a_wrong_setting_is_refused(build):
    with pytest.raises(ValueError):
        build()


def test_the_frozen_tickets_follow_the_drawn_size():
    """Real-scale deathmatch's winning score for N enemy ships: 4 N^(4/3) to the nearest 5
    (the freeze, 2026-11-15), per environment from the active counts."""
    rules = api.rules_of(api.PRESETS["real_scale_deathmatch"])
    mode = rules.modes[0]
    sizes = torch.tensor([1, 2, 3, 4, 5, 10, 20], dtype=torch.int32)
    assert phase.tickets(mode, sizes).tolist() == [5, 10, 15, 25, 35, 85, 215]
    active = torch.tensor([[1, 1], [2, 2], [3, 3], [4, 4], [5, 5]], dtype=torch.int32)
    targets = phase.deathmatch_targets(rules, mode, active)
    assert targets.tolist() == [[5, 5], [10, 10], [15, 15], [25, 25], [35, 35]]
    # uneven: each team's winning score is the tickets of the enemy's ships (team 0 with 3
    # ships needs 35 kills of team 1's 5, team 1 needs 15)
    uneven = torch.tensor([[3, 5], [5, 1]], dtype=torch.int32)
    assert phase.deathmatch_targets(rules, mode, uneven).tolist() == [[35, 15], [5, 35]]


# ------------------------------------------------------------------ the rollout

S = 5
GAME = GameConfig(S, 4, 1500.0, episode_seconds=0.5, respawn_seconds=0.2)
ENVIRONMENTS, LENGTH, SEGMENTS = 12, 30, 3
SIZES = TeamSizes()
MINIMUM = {"team_deathmatch": 1, "frontline": 3}


def _pools(team_sizes: TeamSizes | None):
    return (
        Pool("deathmatch", 0.5, ("learner", "other"), 0.5, team_sizes=team_sizes),
        Pool(
            "frontline",
            0.5,
            ("learner", "other"),
            0.5,
            mode="frontline",
            arena_radius=2600.0,
            team_sizes=team_sizes,
        ),
    )


def _rollout(
    team_sizes=SIZES, learner: str = "chaser", seed: int = 0, game: GameConfig = GAME
) -> Rollout:
    chaser = Chaser(characteristics(TYPES, api.rules_of(game)), S)
    players = {"chaser": chaser, "random": Random(S)}
    return Rollout(
        game,
        api.ship_types(CPU),
        _pools(team_sizes),
        {"learner": players[learner], "other": Random(S)},
        "learner",
        ENVIRONMENTS,
        LENGTH,
        CPU,
        compiled=False,
        seed=seed,
    )


def _watched(rollout: Rollout, segments: int = SEGMENTS):
    """Plays `segments` segments; returns {(environment, episode): [each tick's active
    counts]}, the ticks seen at each episode's start (clock 0) as (active, alive per
    team, benched flags per view), and every tick's (active, alive, benched flags)."""
    act, advance, resolve, perceive = rollout.stages
    by_episode, starts, ticks = {}, [], []

    def watched(carry):
        perceive(carry)
        active = carry["active"].clone()
        alive = carry["state"].ships.alive.reshape(ENVIRONMENTS, 2, S)
        benched = carry["obs"]["ships"][..., SHIP["benched"]] > 0.5  # (B, 2, N) by view row
        ticks.append((active, alive.clone(), benched.clone()))
        for index in range(ENVIRONMENTS):
            key = (index, int(carry["episode"][index]))
            by_episode.setdefault(key, []).append(active[index].clone())
            if int(carry["clock"][index]) == 0:
                starts.append((active[index].clone(), alive[index].clone()))

    rollout.stages = (act, advance, resolve, watched)
    for _ in range(segments):
        rollout.run()
    return by_episode, starts, ticks


def test_each_episode_draws_its_team_sizes_within_its_mode():
    rollout = _rollout()
    by_episode, starts, _ = _watched(rollout)
    pool_mode = ["team_deathmatch"] * (ENVIRONMENTS // 2) + ["frontline"] * (ENVIRONMENTS // 2)
    drawn = {mode: set() for mode in MINIMUM}
    for (index, _), counts in by_episode.items():
        first = counts[0]
        assert all(torch.equal(count, first) for count in counts)  # fixed through the episode
        assert MINIMUM[pool_mode[index]] <= int(first.min()) <= int(first.max()) <= S
        drawn[pool_mode[index]].add(tuple(first.tolist()))
    assert (5, 5) in drawn["team_deathmatch"] and (5, 5) in drawn["frontline"]
    assert any(a != b for a, b in drawn["team_deathmatch"] | drawn["frontline"])  # uneven
    # a new episode spawns exactly its active ships
    assert len(starts) >= ENVIRONMENTS
    for active, alive in starts:
        assert alive.sum(-1).tolist() == active.tolist()


def test_the_draw_follows_the_table():
    """Drawn over many episodes, each mode's sizes and fields match its table (GAME has 4
    field slots: one field per ship to the nearest even count)."""
    rollout = _rollout()
    draws = [rollout._team_sizes(torch.full((ENVIRONMENTS,), e)) for e in range(600)]
    active = torch.stack([one[0] for one in draws])  # (E, B, 2)
    fields = torch.stack([one[1] for one in draws])  # (E, B)
    half = ENVIRONMENTS // 2
    for mode, columns in (("team_deathmatch", slice(0, half)), ("frontline", slice(half, None))):
        table = SIZES.table(S, GAME.num_fields, MINIMUM[mode])
        pairs = active[:, columns].reshape(-1, 2)
        counts = fields[:, columns].reshape(-1)
        for first, second, count, chance in table:
            hit = (pairs[:, 0] == first) & (pairs[:, 1] == second)
            assert float(hit.float().mean()) == pytest.approx(chance, abs=0.025)
            assert bool((counts[hit] == count).all())


def test_a_benched_slot_never_plays_and_is_neither_valued_nor_rewarded():
    rollout = _rollout()
    trajectories = []
    act, advance, resolve, perceive = rollout.stages
    ticks = []

    def watched(carry):
        perceive(carry)
        ticks.append((carry["active"].clone(), carry["state"].ships.alive.clone()))

    rollout.stages = (act, advance, resolve, watched)
    for _ in range(SEGMENTS):
        trajectories.append(rollout.run())
        trajectory = trajectories[-1]
        benched = trajectory.obs.ships[..., :S, SHIP["benched"]] > 0.5  # (L, R, S) own rows
        assert bool(benched.any())
        assert not bool((trajectory.valued & benched).any())
        assert not bool((trajectory.alive[..., :S] & benched).any())
        assert bool((trajectory.reward[benched] == 0.0).all())
    for active, alive in ticks:
        benched = api.benched_of(S, active)
        assert not bool((alive & benched).any())


def test_the_size_is_keyed_by_seed_environment_and_episode():
    """Another learner: the k-th episode of each environment plays the same size; the same
    rollout reproduces; another seed differs."""
    first = _watched(_rollout())[0]
    other = _watched(_rollout(learner="random"))[0]
    common = set(first) & set(other)
    assert len(common) >= ENVIRONMENTS
    assert all(torch.equal(first[key][0], other[key][0]) for key in common)
    again = _watched(_rollout())[0]
    assert all(torch.equal(again[key][0], first[key][0]) for key in first)
    seeded = _watched(_rollout(seed=1))[0]
    assert any(not torch.equal(seeded[key][0], first[key][0]) for key in set(seeded) & set(first))


def test_turning_team_size_sampling_on_moves_no_other_draw():
    sampled, plain = _rollout(), _rollout(None)
    episode = torch.arange(ENVIRONMENTS, dtype=torch.int64) % 3
    assert torch.equal(sampled._episode_uniforms(episode), plain._episode_uniforms(episode))
    assert torch.equal(sampled._seats(episode), plain._seats(episode))
    assert plain._team_sizes(episode) == (None, None) and "active" not in plain.carry
    assert "team_size_sums" not in plain.carry
    first, second = sampled.carry["state"].fields, plain.carry["state"].fields
    assert torch.equal(first.x, second.x) and torch.equal(first.radius, second.radius)


def test_every_slot_drawn_plays_the_game_without_sampling():
    """Sizes drawn from {S} alone bench nothing: the records are the plain rollout's."""
    full, plain = (
        _rollout(TeamSizes(rule="table", pairs=((S, S, 1.0),), fields="game")),
        _rollout(None),
    )
    for _ in range(2):
        first, second = full.run(), plain.run()
        for name in ("reward", "done", "result", "alive", "valued", "action"):
            a, b = getattr(first, name), getattr(second, name)
            leaves = [a] if isinstance(a, torch.Tensor) else list(a.__dict__.values())
            others = [b] if isinstance(b, torch.Tensor) else list(b.__dict__.values())
            for x, y in zip(leaves, others, strict=True):
                assert torch.allclose(x.float(), y.float(), atol=1e-6), name
        assert torch.equal(first.obs.ships, second.obs.ships)


def test_a_resume_keeps_the_drawn_sizes():
    rollout = _rollout()
    rollout.run()
    saved = rollout.saved_carry()
    assert torch.equal(saved["active"], rollout.carry["active"])
    assert "active" not in _rollout(None).saved_carry()
    with pytest.raises(ValueError):
        rollout.restart(torch.ones((ENVIRONMENTS, 2), dtype=torch.int32))


def test_a_new_episode_plays_its_drawn_field_count():
    """At every episode start the present fields are the drawn count's (GAME's 4 slots)."""
    rollout = _rollout()
    act, advance, resolve, perceive = rollout.stages
    seen = []

    def watched(carry):
        perceive(carry)
        starting = carry["clock"] == 0
        if bool(starting.any()):
            count = api.active_field_count(carry["state"])
            fields = rollout._team_sizes(carry["episode"])[1]
            seen.append(bool((count[starting] == fields[starting]).all()))

    rollout.stages = (act, advance, resolve, watched)
    for _ in range(SEGMENTS):
        rollout.run()
    assert seen and all(seen)


def test_the_series_count_the_ended_episodes_sizes():
    """The rollout's team-size sums hold every ended episode's pair and fields; the series
    per mode are their means and shares; the setting's `series` False keeps none."""
    from bnb.contracts.series import Mean
    from bnb.runtime.analytics import Analytics

    rollout = _rollout()
    by_episode, _, _ = _watched(rollout)
    sums = rollout.take_team_sizes()
    ended = int(sums[:, 0].sum())
    assert ended > 0 and int(sums[:, 2:].sum()) == ended
    assert float(rollout.carry["team_size_sums"].abs().sum()) == 0.0  # taken
    environment_pool = torch.tensor([0] * (ENVIRONMENTS // 2) + [1] * (ENVIRONMENTS // 2))
    analytics = Analytics(
        ("deathmatch", "frontline"),
        environment_pool,
        environment_pool,
        0.1,
        "team_deathmatch",
        pool_modes=("team_deathmatch", "frontline"),
    )
    series = {
        name: float(value.total) / float(value.count) if isinstance(value, Mean) else value
        for name, value in analytics.team_size_series(sums, S).items()
    }
    for mode in ("team_deathmatch", "frontline"):
        shares = [
            value
            for name, value in series.items()
            if name.startswith("team_size/share/") and name.endswith(mode)
        ]
        assert sum(shares) == pytest.approx(1.0)
        assert 1 <= series[f"team_size/smaller/{mode}"] <= series[f"team_size/larger/{mode}"] <= S
        assert 0 <= series[f"team_size/fields/{mode}"] <= GAME.num_fields
    assert series["team_size/share/2v1/frontline"] == 0.0  # below Frontline's 3
    quiet = _rollout(TeamSizes(series=False))
    assert quiet.take_team_sizes() is None and "team_size_sums" not in quiet.carry


# ------------------------------------------------------------------ the rewards


def test_the_unforced_share_goes_to_the_present_slots_only():
    team = torch.tensor([0, 0, 0, 1, 1, 1])
    value = torch.tensor([[0.0, 0.0, 0.0, 3.0, 0.0, 0.0]])  # an enemy's unforced loss
    present = torch.tensor([[True, False, False, True, True, True]])
    plain = rewards.enemy_team_share(value, team)
    assert plain[0, :3].tolist() == pytest.approx([1.0, 1.0, 1.0])
    shared = rewards.enemy_team_share(value, team, present)
    assert shared[0].tolist() == pytest.approx([3.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    everyone = rewards.enemy_team_share(value, team, torch.ones_like(present))
    assert torch.equal(everyone, plain)


# ------------------------------------------------------------------ the settings


def test_the_training_setting_and_its_plumbing():
    config = TrainConfig(api.PRESETS["deathmatch"])
    assert config.team_sizes is None and not samples_team_sizes(training_pools(config))
    varied = TrainConfig(api.PRESETS["deathmatch"], team_sizes=TeamSizes())
    pools = training_pools(varied)
    assert all(pool.team_sizes == TeamSizes() for pool in pools)
    assert samples_team_sizes(pools) and samples_field_counts(varied.game, pools)
    ships_only = training_pools(dataclasses.replace(varied, team_sizes=TeamSizes(fields="game")))
    assert not samples_field_counts(varied.game, ships_only)
    # the rating lanes play their games' own sizes
    modes = lanes.mode_games(varied.pools, varied.game, lambda pool, game: pool.mode or game.mode)
    every, _, _ = lanes.lane_pools(pools, modes, 64, lanes.LaneConfig(environments=64))
    assert [pool.team_sizes for pool in every[len(pools) :]] == [None] * (len(every) - len(pools))
    # the setting round-trips through the run's config.json
    weighted = dataclasses.replace(
        varied, team_sizes=TeamSizes(rule="table", pairs=((5, 5, 2.0), (5, 3, 1.0)))
    )
    assert from_json(TrainConfig, to_json(weighted))[0].team_sizes == weighted.team_sizes
    with pytest.raises(ValueError):  # past the game's slots
        TrainConfig(api.PRESETS["deathmatch"], team_sizes=TeamSizes(low=6))
    with pytest.raises(ValueError):  # a model that does not mask benched ships
        TrainConfig(api.PRESETS["deathmatch"], team_sizes=TeamSizes(), model="relational")


def test_the_command_line_setting():
    from bnb import cli

    base = TrainConfig(api.PRESETS["deathmatch"])
    assert cli.with_settings(base, [("team_sizes", {})]).team_sizes == TeamSizes()
    config = cli.with_settings(base, [("team_sizes", {"rule": "near_full", "field_spread": 1})])
    assert config.team_sizes == TeamSizes(rule="near_full", field_spread=1)


@pytest.mark.parametrize("objective", ["bc", "ppo"])
def test_bc_and_rl_train_on_drawn_sizes(objective):
    """A tiny Yemong trainer of each objective plays drawn sizes, masks the benched ships
    and finishes an update with finite losses."""
    from bnb.learn import bc
    from bnb.learn.ppo import LearnerConfig
    from bnb.runtime import cloning
    from bnb.runtime import train as training
    from bnb.runtime.train import Trainer
    from tests.runs.runs_cases import tiny_yemong_config

    learner = bc.config(minibatches=2) if objective == "bc" else LearnerConfig(minibatches=2)
    config = TrainConfig(
        game=GameConfig(3, 2, 1200.0, episode_seconds=1.0, respawn_seconds=0.5),
        pools=cloning.LINEUP if objective == "bc" else training.LINEUP,
        environments=8,
        length=8,
        shards=1,
        lookahead=0,
        lanes=lanes.LaneConfig(environments=0),
        yemong=tiny_yemong_config(),
        compiled=False,
        host_store=False,
        learner=learner,
        objective=objective,
        team_sizes=TeamSizes(),
        lineup_draw=False,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        trainer = Trainer(config, CPU)
        assert trainer.model.mask_benched and trainer.model.mask_absent_fields
        assert "active" in trainer.rollout.carry
        # the update's 8 ticks end only the staggered first games, which the logged series
        # leave out: count them here, for the team-size series to read
        trainer.rollout.carry["staggered"].zero_()
        metrics = trainer.update()
    assert int(trainer.rollout.carry["active"].min()) >= 1
    assert "team_size/larger/team_deathmatch" in metrics
    losses = [
        value for name, value in metrics.items() if "loss" in name and isinstance(value, float)
    ]
    assert losses and all(torch.isfinite(torch.tensor(losses)))
