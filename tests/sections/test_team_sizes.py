"""Team-size sampling (arm A2, `bnb.contracts.TeamSizes`): each episode plays an N-against-N
game drawn from the pool's sizes (within each mode's smallest team), the rest of each
team's slots benched, at the batch's one static size. The draw is keyed by (seed,
environment, episode) and moves no other draw; a reset spawns and benches by the next
episode's size; a benched slot is neither valued nor rewarded; deathmatch's tickets follow
the drawn size (the freeze's 4 N^(4/3) to the nearest 5); a rollout without it, or with
every slot drawn, is the game as before; BC and RL both train with it, the Yemong masking
the benched ships."""

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
from bnb.runtime.train import TrainConfig, samples_team_sizes, training_pools

CPU = torch.device("cpu")
TYPES = api.ship_types()


@pytest.fixture(autouse=True)
def _default_generator_untouched():
    with torch.random.fork_rng(devices=[]):
        yield


# ------------------------------------------------------------------ the setting


def test_the_sizes_are_uniform_within_the_modes_smallest_team():
    assert TeamSizes().probabilities(5) == pytest.approx((0.2,) * 5)
    assert TeamSizes().probabilities(5, minimum=3) == pytest.approx((0, 0, 1 / 3, 1 / 3, 1 / 3))
    assert TeamSizes(2, 4).probabilities(5) == pytest.approx((0, 1 / 3, 1 / 3, 1 / 3, 0))
    weighted = TeamSizes(1, 5, (1.0, 1.0, 1.0, 1.0, 4.0))
    assert weighted.probabilities(5) == pytest.approx((0.125,) * 4 + (0.5,))
    assert weighted.probabilities(5, minimum=3) == pytest.approx((0, 0, 1 / 6, 1 / 6, 4 / 6))


@pytest.mark.parametrize(
    "build",
    [
        lambda: TeamSizes(0),
        lambda: TeamSizes(3, 2),
        lambda: TeamSizes(1, None, (1.0,)),
        lambda: TeamSizes(1, 2, (1.0,)),
        lambda: TeamSizes(1, 2, (0.0, 0.0)),
        lambda: TeamSizes(1, 2, (-1.0, 2.0)),
        lambda: TeamSizes(1, 6).probabilities(5),
        lambda: TeamSizes(1, 2).probabilities(5, minimum=3),
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


# ------------------------------------------------------------------ the rollout

S = 5
GAME = GameConfig(S, 4, 1500.0, episode_seconds=0.5, respawn_seconds=0.2)
ENVIRONMENTS, LENGTH, SEGMENTS = 12, 30, 3
SIZES = TeamSizes(1)
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


def test_each_episode_draws_one_size_for_both_teams_within_its_mode():
    rollout = _rollout()
    by_episode, starts, _ = _watched(rollout)
    pool_mode = ["team_deathmatch"] * (ENVIRONMENTS // 2) + ["frontline"] * (ENVIRONMENTS // 2)
    drawn = {mode: set() for mode in MINIMUM}
    for (index, _), counts in by_episode.items():
        first = counts[0]
        assert all(torch.equal(count, first) for count in counts)  # fixed through the episode
        assert int(first[0]) == int(first[1])  # N against N
        size = int(first[0])
        assert MINIMUM[pool_mode[index]] <= size <= S
        drawn[pool_mode[index]].add(size)
    assert drawn["team_deathmatch"] >= {1, 2, 5} and drawn["frontline"] == {3, 4, 5}
    # a new episode spawns exactly its active ships
    assert len(starts) >= ENVIRONMENTS
    for active, alive in starts:
        assert alive.sum(-1).tolist() == active.tolist()


def test_the_draw_is_uniform():
    rollout = _rollout()
    episode = torch.arange(4000, dtype=torch.int64)
    sizes = torch.cat(
        [rollout._team_sizes(torch.full((ENVIRONMENTS,), e))[:, 0] for e in episode[:400]]
    )
    deathmatch = sizes.reshape(400, ENVIRONMENTS)[:, : ENVIRONMENTS // 2].flatten()
    frontline = sizes.reshape(400, ENVIRONMENTS)[:, ENVIRONMENTS // 2 :].flatten()
    counts = torch.bincount(deathmatch, minlength=S + 1)[1:].float() / deathmatch.numel()
    assert counts.tolist() == pytest.approx([0.2] * 5, abs=0.03)
    counts = torch.bincount(frontline, minlength=S + 1)[1:].float() / frontline.numel()
    assert counts.tolist() == pytest.approx([0, 0, 1 / 3, 1 / 3, 1 / 3], abs=0.03)


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
    assert plain._team_sizes(episode) is None and "active" not in plain.carry
    first, second = sampled.carry["state"].fields, plain.carry["state"].fields
    assert torch.equal(first.x, second.x) and torch.equal(first.radius, second.radius)


def test_every_slot_drawn_plays_the_game_without_sampling():
    """Sizes drawn from {S} alone bench nothing: the records are the plain rollout's."""
    full, plain = _rollout(TeamSizes(S)), _rollout(None)
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
    varied = TrainConfig(api.PRESETS["deathmatch"], team_sizes=TeamSizes(1))
    pools = training_pools(varied)
    assert all(pool.team_sizes == TeamSizes(1) for pool in pools)
    assert samples_team_sizes(pools)
    # the rating lanes play their games' own sizes
    modes = lanes.mode_games(varied.pools, varied.game, lambda pool, game: pool.mode or game.mode)
    every, _, _ = lanes.lane_pools(pools, modes, 64, lanes.LaneConfig(environments=64))
    assert [pool.team_sizes for pool in every[len(pools) :]] == [None] * (len(every) - len(pools))
    # the setting round-trips through the run's config.json
    weighted = dataclasses.replace(varied, team_sizes=TeamSizes(2, 5, (1.0, 1.0, 1.0, 2.0)))
    assert from_json(TrainConfig, to_json(weighted))[0].team_sizes == weighted.team_sizes
    with pytest.raises(ValueError):  # past the game's slots
        TrainConfig(api.PRESETS["deathmatch"], team_sizes=TeamSizes(1, 6))
    with pytest.raises(ValueError):  # a model that does not mask benched ships
        TrainConfig(api.PRESETS["deathmatch"], team_sizes=TeamSizes(1), model="relational")


def test_the_command_line_setting():
    from bnb import cli

    config = cli.with_settings(TrainConfig(api.PRESETS["deathmatch"]), [("team_sizes", {"low": 1})])
    assert config.team_sizes == TeamSizes(1)


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
        team_sizes=TeamSizes(1),
        lineup_draw=False,
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        trainer = Trainer(config, CPU)
        assert trainer.model.mask_benched
        assert "active" in trainer.rollout.carry
        metrics = trainer.update()
    assert int(trainer.rollout.carry["active"][:, 0].min()) >= 1
    losses = [
        value for name, value in metrics.items() if "loss" in name and isinstance(value, float)
    ]
    assert losses and all(torch.isfinite(torch.tensor(losses)))
