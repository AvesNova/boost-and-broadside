"""The rollout's keyed draws (`bnb.runtime.rollout`, `bnb.streams`): an environment's k-th
episode plays the same map, seat and draws in every rollout of one seed, whatever else the
rollouts do differently (their controllers, their episode lengths, map-size sampling), and
a rollout reproduces exactly. Eager on the CPU, with scripted controllers."""

import pytest
import torch

from bnb.contracts import ArenaScale, GameConfig, Pool
from bnb.controllers.scripted import Chaser, Random
from bnb.game import api, layout
from bnb.game.handling.characteristics import characteristics
from bnb.runtime.rollout import Rollout

CPU = torch.device("cpu")
S = 2
SHORT = GameConfig(S, 4, 1500.0, episode_seconds=0.5)  # 15-tick episodes at most
SHORTER = GameConfig(S, 4, 1500.0, episode_seconds=0.4)  # 12
ENVIRONMENTS, LENGTH, SEGMENTS = 12, 30, 4


def _rollout(
    learner: str = "chaser",
    seed: int = 0,
    scale: ArenaScale | None = ArenaScale(),
    game: GameConfig = SHORT,
):
    chaser = Chaser(characteristics(api.ship_types(), api.rules_of(game)), S)
    players = {"chaser": chaser, "random": Random(S)}
    pools = (Pool("p", 1.0, ("learner", "other"), 0.5, arena_scale=scale),)
    return Rollout(
        game,
        api.ship_types(CPU),
        pools,
        {"learner": players[learner], "other": Random(S)},
        "learner",
        ENVIRONMENTS,
        LENGTH,
        CPU,
        compiled=False,
        seed=seed,
    )


def _episodes(rollout: Rollout) -> dict:
    """Every tick's carry (the episode counts, the seats, the actions, the fields' centres and
    the arena scale) and, per environment, each episode's start: (episode index, its first
    tick's seat, its fields' x and y, its arena scale), keyed by (environment, episode)."""
    act, advance, resolve, perceive = rollout.stages
    ticks = []

    def recorded(carry):
        perceive(carry)
        state = carry["state"]
        ticks.append(
            dict(
                episode=carry["episode"].clone(),
                seat=carry["seat"].clone(),
                actions=carry["actions"].clone(),
                x=state.fields.x.clone(),
                y=state.fields.y.clone(),
                scale=state.match.arena_scale.clone(),
            )
        )

    rollout.stages = (act, advance, resolve, recorded)
    starts = {}
    for _ in range(SEGMENTS):
        rollout.run()
    for tick in ticks:
        for environment in range(ENVIRONMENTS):
            key = (environment, int(tick["episode"][environment]))
            starts.setdefault(
                key,
                dict(
                    seat=int(tick["seat"][environment]),
                    x=tick["x"][environment],
                    y=tick["y"][environment],
                    scale=float(tick["scale"][environment]),
                ),
            )
    return dict(ticks=ticks, starts=starts)


@pytest.fixture(scope="module")
def chaser_episodes():
    return _episodes(_rollout())


def test_a_rollout_of_the_same_config_reproduces_exactly(chaser_episodes):
    again = _episodes(_rollout())
    for first, second in zip(chaser_episodes["ticks"], again["ticks"], strict=True):
        for name in first:
            assert torch.equal(first[name], second[name]), name


def test_an_episodes_map_and_seat_are_its_environments_and_index_whoever_plays():
    """Another player and shorter episodes: the episodes start at other ticks and play other
    actions, and the k-th episode of each environment has the same map, seat and arena."""
    chaser, random = _episodes(_rollout("chaser")), _episodes(_rollout("random", game=SHORTER))
    common = set(chaser["starts"]) & set(random["starts"])
    assert len(common) >= 2 * ENVIRONMENTS  # at least the first two episodes everywhere
    assert len(random["starts"]) > len(chaser["starts"])  # the episode counts diverged
    assert any(
        not torch.equal(a["actions"], b["actions"])
        for a, b in zip(chaser["ticks"], random["ticks"], strict=True)
    )
    for key in common:
        one, other = chaser["starts"][key], random["starts"][key]
        assert one["seat"] == other["seat"], key
        assert torch.equal(one["x"], other["x"]) and torch.equal(one["y"], other["y"]), key
        assert one["scale"] == other["scale"], key


def test_turning_map_size_sampling_off_moves_no_other_draw():
    """The arena scale's slot is drawn either way: without sampling the map is the same map
    at scale 1 (each field's centre scales with the arena), and the seats are the same."""
    sampled, plain = _rollout(scale=ArenaScale()), _rollout(scale=None)
    episode = torch.arange(ENVIRONMENTS, dtype=torch.int64) % 3
    assert torch.equal(sampled._episode_uniforms(episode), plain._episode_uniforms(episode))
    assert torch.equal(sampled._seats(episode), plain._seats(episode))
    first, second = sampled.carry["state"], plain.carry["state"]
    assert not torch.equal(first.match.arena_scale, second.match.arena_scale)

    def radius(rollout, state):
        arena = layout.sized(
            layout.environment(rollout.rules, ENVIRONMENTS), state.match.arena_scale
        )
        return arena.arena_radius[:, None]

    for axis in ("x", "y"):
        torch.testing.assert_close(
            getattr(first.fields, axis) / radius(sampled, first),
            getattr(second.fields, axis) / radius(plain, second),
            rtol=1e-5,
            atol=1e-6,
        )


def test_another_seed_draws_other_maps(chaser_episodes):
    other = _rollout(seed=1)
    assert not torch.equal(other.carry["state"].fields.x, _rollout().carry["state"].fields.x)
    assert chaser_episodes["starts"]  # the fixture played


def test_restart_keys_each_batch_apart_and_a_batch_number_replays_its_games():
    rollout = _rollout()
    maps = []
    for batch in (0, 1, 0):
        rollout.restart(batch=batch)
        maps.append(rollout.carry["state"].fields.x.clone())
    assert torch.equal(maps[0], maps[2]) and not torch.equal(maps[0], maps[1])


def test_the_decision_keys_are_the_environments_tick_and_the_rows_role():
    rollout = _rollout()
    ticks = rollout._tick_keys(rollout.carry)
    learner, other = (
        rollout._decision_keys(ticks, "learner"),
        rollout._decision_keys(ticks, "other"),
    )
    assert learner.dtype == torch.int64 and learner.shape == (ENVIRONMENTS,)
    assert bool(((learner >= 0) & (learner < 1 << 32)).all())
    assert not bool((learner == other).any())  # the two roles of one environment differ
    rollout.carry["clock"].add_(1)
    assert not bool(
        (rollout._decision_keys(rollout._tick_keys(rollout.carry), "learner") == learner).any()
    )
