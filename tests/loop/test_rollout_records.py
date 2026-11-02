"""What the rollout stores for the learner's views (learner spec 2): rewards by the seat's
team, seats redrawn only at episode ends, the time since seen carried from tick to tick, and
each environment's own episode ends. Mutation-sweep items 16 to 19.

The rollout runs eagerly on the CPU with scripted controllers (the chaser as the learner).
Its seat draws come from the default generator, as in training; the statistical bound below
states the chance of a false failure.
"""

import torch

from bnb.contracts import GameConfig, Pool
from bnb.contracts.config import ticks_of
from bnb.contracts.obs import SHIP
from bnb.controllers.scripted import Chaser, Random
from bnb.learn import rewards
from bnb.runtime.rollout import Rollout

CPU = torch.device("cpu")
S = 2


def _rollout(config, pools, environments, length, **options):
    return Rollout(
        config,
        pools,
        {"learner": Chaser(S), "other": Random(S)},
        "learner",
        environments,
        length,
        CPU,
        compiled=False,
        **options,
    )


def _recording(rollout, monkeypatch):
    """Record, every tick, the seats the resolve stage starts from, the game's rewards
    (B, N, levels) and its episode ends (B,)."""
    record = dict(seat=[], reward=[], done=[])
    act, advance, resolve, perceive = rollout.stages

    def recorded_resolve(carry):
        record["seat"].append(carry["seat"].clone())
        resolve(carry)

    rollout.stages = (act, advance, recorded_resolve, perceive)
    original = rewards.reward_step

    def recorded_step(config, ledger, events, team):
        reward, ledger = original(config, ledger, events, team)
        record["reward"].append(reward.clone())
        record["done"].append(events.done.clone())
        return reward, ledger

    monkeypatch.setattr(rewards, "reward_step", recorded_step)
    return record


# Half-second team deathmatch episodes (15 ticks): many episodes in a few segments.
SHORT = GameConfig(S, 2, 1500.0, episode_seconds=0.5)


def test_the_stored_reward_is_the_learners_seat_teams(monkeypatch):
    rollout = _rollout(SHORT, (Pool("p", 1.0, ("learner", "other")),), 16, 32)
    record = _recording(rollout, monkeypatch)
    lineup = rollout.lineup
    environment, role = lineup.environments["learner"], lineup.role["learner"]
    seen_seats = set()
    for _ in range(3):
        start = len(record["seat"])
        trajectory = rollout.run()
        for tick in range(trajectory.reward.shape[0]):
            seat = record["seat"][start + tick][environment]
            team = (role + seat) % 2
            game = record["reward"][start + tick].reshape(16, 2, S, -1)
            torch.testing.assert_close(
                trajectory.reward[tick], game[environment, team], rtol=0, atol=0
            )
            seen_seats.update(seat.tolist())
    assert seen_seats == {0, 1}


def test_seats_are_redrawn_at_episode_ends_and_only_there(monkeypatch):
    """16 environments, 20 episodes each. A seat drawn fresh and fair at each of 19 ends
    gives an environment one seat throughout with probability 2^-19 (16 environments:
    3e-5 of a false failure)."""
    rollout = _rollout(SHORT, (Pool("p", 1.0, ("learner", "other")),), 16, 100, staggered=False)
    record = _recording(rollout, monkeypatch)
    for _ in range(3):
        rollout.run()
    seats, done = torch.stack(record["seat"]), torch.stack(record["done"])  # (ticks, B)
    episodes = done.sum(0)
    assert int(episodes.min()) >= 19, episodes
    changed = seats[1:] != seats[:-1]
    assert not bool((changed & ~done[:-1]).any()), "a seat changed mid-episode"
    for environment in range(16):
        assert set(seats[:, environment].tolist()) == {0, 1}, environment


def test_the_time_since_seen_rises_from_tick_to_tick_in_the_stored_views():
    """Fogged 2v2 in a wide arena: hidden enemies' times since seen grow by one tick per
    stored step (the counter is carried between ticks, not restarted)."""
    config = GameConfig(S, 2, 6000.0, mode="sandbox")
    rollout = _rollout(config, (Pool("p", 1.0, ("learner", "other"), 1.0),), 8, 64)
    tick = rollout.rules.tick_length
    trajectory = rollout.run()
    since = trajectory.obs.ships[..., SHIP["time_since_seen"]]  # (L, R, N)
    assert float(since.max()) >= 20 * tick - 1e-6
    # step t + 1 follows a reset where the episode ended at step t
    following = (since[1:] > 0) & (since[:-1] > 0) & ~trajectory.done[:-1, :, None]
    step = (since[1:] - since[:-1])[following]
    assert step.numel() > 100
    torch.testing.assert_close(step, torch.full_like(step, tick), rtol=0, atol=1e-5)


def test_each_stored_done_is_its_own_environments():
    """Three pools with two episode lengths, the learner in two of them (rows 16 to 63):
    every stored end is its environment's clock."""
    config = GameConfig(S, 2, 1500.0, mode="sandbox")
    pools = (
        Pool("long_alone", 0.25, ("other", "other"), episode_seconds=1.0),
        Pool("short", 0.25, ("learner", "other"), episode_seconds=0.5),
        Pool("long", 0.5, ("other", "learner"), episode_seconds=1.0),
    )
    rollout = _rollout(config, pools, 64, 64, staggered=False)
    tick = rollout.rules.tick_length
    lengths = torch.tensor([ticks_of(pool.episode_seconds, tick) for pool in pools])  # by pool
    pool_of = rollout.lineup.pool_of["learner"]
    trajectory = rollout.run()
    steps = torch.arange(trajectory.done.shape[0])[:, None]  # (L, 1)
    want = (steps + 1) % lengths[pool_of][None, :] == 0
    assert torch.equal(trajectory.done, want)
