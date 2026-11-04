"""The update's geometry and the rating lanes (rollout-lanes brief): advantages chained over
an update's shards and its lookahead, the shot labels that read the ticks after a segment,
the lanes' seats, tallies and ratings, and BC's score stop on them."""

import dataclasses
import math
import pathlib
import sys
import types

import learner_cases as lc
import pytest
import torch
import yemong_cases as cases
from scripted_cases import CHARACTERISTICS

from bnb.contracts import GameConfig, Pool
from bnb.controllers.relational import RelationalController
from bnb.game import api
from bnb.league.gauge import Gauge
from bnb.league.lanes import LaneHooks, LaneRatings
from bnb.models.relational import Relational, RelationalBuild, RelationalConfig
from bnb.runtime import cloning
from bnb.runtime.lanes import (
    LANES,
    LIVE,
    SCRIPTED_PLAYER,
    Lane,
    LaneConfig,
    LaneSeat,
    Matchup,
    ladder_chain,
    lane_pools,
    split,
)
from bnb.runtime.rollout import Rollout

sys.path.insert(0, str(pathlib.Path(__file__).parents[1] / "runs"))
from runs_cases import tiny_train_config  # noqa: E402

pytestmark = pytest.mark.usefixtures("global_generator")
MODE = "team_deathmatch"


# ------------------------------------------------------------------ chained advantages


def _collect(net, shards: int, length: int, lookahead: int = 0, environments: int = 4):
    """A seeded Yemong rollout of `shards` segments of `length` ticks and the lookahead,
    and the bootstrap after it."""
    torch.manual_seed(7)
    runner = Rollout(
        cases.SHORT_SQUAD,
        api.ship_types(lc.CPU),
        lc.POOLS,
        {"learner": cases.controller(net, cases.SHORT_SQUAD), "chaser": lc.Chaser(5)},
        "learner",
        environments,
        length,
        lc.CPU,
        compiled=False,
        graph_ticks=8,
        shards=shards,
        lookahead=lookahead,
    )
    for shard in range(shards):
        runner.run(shard)
    if lookahead:
        runner.run_lookahead()
    return runner, runner.bootstrap()


def _by_tick(x: torch.Tensor, shards: int) -> torch.Tensor:
    """The store's views (L, shards R, ...) as one segment of the same rows, (shards L, R)."""
    length, views = x.shape[:2]
    return x.unflatten(1, (shards, views // shards)).transpose(0, 1).flatten(0, 1)


@pytest.mark.slow
@pytest.mark.parametrize("lookahead", [0, 8])
def test_chained_returns_over_shards_equal_one_segments(lookahead):
    """N shards of L ticks (and a lookahead of K) give the advantages, returns and outcome
    targets of one segment of N L (+ K) ticks, its first N L: a shard's end bootstraps only
    where the update ends."""
    net = cases.model()
    shards, length = 3, 8
    split_runner, split_bootstrap = _collect(net, shards, length, lookahead)
    whole_runner, whole_bootstrap = _collect(net, 1, shards * length + lookahead)
    split_prepared = lc.learner(net, split_runner).prepare(split_runner.store, split_bootstrap)
    whole_prepared = lc.learner(net, whole_runner).prepare(whole_runner.store, whole_bootstrap)
    for name in ("advantage", "level_returns", "outcome_targets"):
        chained = _by_tick(split_prepared["derived"][name], shards)
        whole = whole_prepared["derived"][name][: shards * length]
        torch.testing.assert_close(chained, whole, rtol=0, atol=0, msg=name)
    # the split rollout is the same play: its rewards are the whole one's, tick for tick
    rewards = split_runner.store.data["reward"][0]
    torch.testing.assert_close(
        _by_tick(rewards, shards), whole_runner.store.data["reward"][0][: shards * length]
    )


def test_the_per_shard_cut_is_gone():
    """The advantage at a shard's last tick reads the next shard's rewards: a reward put
    into the second shard's first tick moves the first shard's last advantage."""
    from bnb.learn.ppo import Learner, LearnerConfig
    from bnb.learn.rewards import RewardConfig

    rows, ships, levels = 2, 2, 5
    build = RelationalBuild(RelationalConfig(hidden=8, value_bins=11), ships, 2 * ships)
    learner = Learner(
        Relational(build),
        LearnerConfig(),
        api.rules_of(GameConfig(2, 2, 1500.0)),
        api.ship_types(),
        RewardConfig().outcome,
        lc.CPU,
    )

    def chain(bump: float) -> dict:
        reward = torch.zeros((8, rows, ships, levels))
        reward[4, :, :, 0] = bump  # the second shard's first tick (shards of 4)
        return dict(
            reward=reward,
            values=torch.zeros((8, rows, ships, levels)),
            outcome=torch.full((8, rows, 4), 0.25),
            done=torch.zeros((8, rows), dtype=torch.bool),
            result=torch.zeros((8, rows), dtype=torch.int32),
        )

    bootstrap = types.SimpleNamespace(
        values=torch.zeros((rows, ships, levels)), outcome=torch.full((rows, 4), 0.25)
    )
    alive = torch.ones((8, rows, 2 * ships), dtype=torch.bool)
    quiet = learner.chained_returns(chain(0.0), alive, bootstrap)["advantage"]
    bumped = learner.chained_returns(chain(1.0), alive, bootstrap)["advantage"]
    assert torch.equal(quiet[3], torch.zeros_like(quiet[3]))
    assert (bumped[3] > 0).all()  # tick 3, the first shard's last


# ------------------------------------------------------------------ the shots after a segment


@pytest.mark.slow
def test_shots_near_a_segments_end_read_the_ticks_after_it(segment_pair):
    """A segment's shots with the bullet flight after it are those of one longer segment:
    the shots fired in its last ticks are labelled, not dropped."""
    learner, whole, first, following = segment_pair
    length = first.done.shape[0]
    spread = torch.randn((2 * length,) + first.capture.alive.shape[1:] + (2,))
    followed = torch.ones(first.done.shape[1], dtype=torch.bool)
    extended, _ = learner.shots(first, spread[:length], following, followed)
    reference, _ = learner.shots(whole, spread)
    for field in dataclasses.fields(extended):
        torch.testing.assert_close(
            getattr(extended, field.name),
            getattr(reference, field.name)[:length],
            rtol=0,
            atol=0,
            msg=field.name,
        )
    alone, _ = learner.shots(first, spread[:length])
    tail = slice(length - learner.rules.bullet_moves + 1, length)
    assert not alone.valid[tail].any() and extended.valid[tail].any()


@pytest.fixture(scope="module")
def segment_pair(module_global_generator):
    """A learner, one 2L-tick segment, its first L ticks and the bullet flight after them."""
    net = cases.model()
    runner = lc.rollout(net, length=32, runs=1)
    whole = lc.whole(runner)
    length, flight = 16, runner.rules.bullet_moves

    def ticks(tree, part: slice):
        return torch.utils._pytree.tree_map_only(torch.Tensor, lambda x: x[part], tree)

    per_tick = {name: getattr(whole, name) for name in runner.store.per_tick}
    first = dataclasses.replace(whole, **ticks(per_tick, slice(0, length)))
    following = dict(
        capture=ticks(whole.capture, slice(length, length + flight)),
        obs=ticks(whole.obs, slice(length, length + flight)),
        done=whole.done[length : length + flight],
    )
    return lc.learner(net, runner), whole, first, following


@pytest.mark.slow
def test_the_store_follows_each_shard_with_the_next_and_the_last_with_the_lookahead():
    net = cases.model()
    runner, _ = _collect(net, 2, 16, lookahead=16)
    store, flight = runner.store, runner.rules.bullet_moves
    rows = store.shard_views
    views = torch.arange(store.views)
    following, followed = store.following(("done", "capture"), views, flight)
    assert followed.all()
    done = store.data["done"][0]
    torch.testing.assert_close(following["done"][:, :rows], done[:flight, rows:])
    torch.testing.assert_close(
        following["done"][:, rows:], store.lookahead(("done",))["done"][:flight]
    )
    assert store.lookahead_ticks["reward"] == 16 and store.lookahead_ticks["capture"] == flight
    alone, _ = _collect(net, 2, 16)
    _, followed = alone.store.following(("done",), views, flight)
    assert followed[:rows].all() and not followed[rows:].any()


# ------------------------------------------------------------------ the lanes


def test_lanes_split_evenly_over_the_modes_then_by_weight():
    assert split(10, [1.0, 1.0, 1.0]) == [4, 3, 3]
    assert split(13, [1.0, 1.0, 0.05, 0.05]) == [6, 5, 1, 1]
    training = (
        Pool("a/self_play", 0.5, ("learner", "learner")),
        Pool("b/self_play", 0.5, ("learner", "learner"), mode="frontline"),
    )
    config = LaneConfig(
        environments=8,
        matchups=(Matchup(LIVE, SCRIPTED_PLAYER, 3.0), Matchup("rung_0.5", SCRIPTED_PLAYER, 1.0)),
    )
    modes = {"a": training[0], "b": training[1]}
    pools, lanes, total = lane_pools(training, modes, 16, config)
    assert total == 24 and [lane.name for lane in lanes] == [
        "a/lane/live_vs_scripted",
        "a/lane/rung_0.5_vs_scripted",
        "b/lane/live_vs_scripted",
        "b/lane/rung_0.5_vs_scripted",
    ]
    assert [round(pool.share * total) for pool in pools] == [8, 8, 3, 1, 3, 1]
    assert all(pool.seats == (LANES, LANES) for pool in pools[2:])
    assert pools[4].mode == "frontline"  # a lane plays its mode's game
    chain = ladder_chain((0.0, 0.5, 0.9), 0.1)
    assert [m.name for m in chain] == [
        "random_vs_rung_0.5",
        "rung_0.5_vs_rung_0.9",
        "rung_0.9_vs_scripted",
    ]


def _seat(lanes: tuple[Lane, ...], rows_per_lane: int = 2) -> LaneSeat:
    game = GameConfig(2, 2, 1500.0)
    build = RelationalBuild(RelationalConfig(hidden=8, value_bins=11), 2, 4)
    controller = RelationalController(Relational(build))
    pool = torch.arange(len(lanes)).repeat_interleave(2 * rows_per_lane)
    role = torch.tensor(([0] * rows_per_lane + [1] * rows_per_lane) * len(lanes))
    del game
    return LaneSeat(controller, lanes, pool, role, CHARACTERISTICS)


def test_a_lane_seat_plays_each_rows_player_and_tallies_each_game_once():
    lanes = (
        Lane(MODE, Matchup(LIVE, SCRIPTED_PLAYER)),
        Lane(MODE, Matchup("rung_0.5", "random")),
    )
    seat = _seat(lanes)
    assert seat.networks == (LIVE,)
    assert seat.network_rows[0].tolist() == [0, 1]  # live: the first lane's first seat
    assert seat.ladder_rows.tolist() == [2, 3, 4, 5, 6, 7]
    assert seat.ladder_agent.scripted_probability.tolist() == [1.0, 1.0, 0.5, 0.5, 0.0, 0.0]
    # rows 0 and 2 are one game (live won), rows 4 and 6 another (rung_0.5 lost): each once,
    # from the first player's side
    done = torch.tensor([True, False, True, False, True, False, True, False])
    result = torch.tensor([1, 0, -1, 0, -1, 0, 1, 0], dtype=torch.int32)
    tally = seat.tally(None, done, result).reshape(2, 3)
    assert tally.tolist() == [[1, 0, 0], [0, 0, 1]]


def test_the_live_lane_plays_the_learners_weights(tmp_path):
    from bnb.runtime.train import Trainer

    config = tiny_train_config(
        lanes=LaneConfig(environments=4, matchups=(Matchup(LIVE, SCRIPTED_PLAYER),))
    )
    trainer = Trainer(config, lc.CPU)
    with torch.no_grad():
        for parameter in trainer.model.parameters():
            parameter.add_(1.0)
    trainer.collect()
    for name, value in trainer.model.state_dict().items():
        assert torch.equal(trainer.lane_seat.parameters[f"model.{name}"][0], value), name
    record = trainer.lane_record
    (lane,) = record
    assert lane.name == f"{MODE}/lane/live_vs_scripted"
    # every lane game ended once in the tally and once in the environments' episode sums
    sums = trainer.rollout.take_episodes()[16:].sum(0)
    assert sum(record[lane]) == int(sums[0])


def _trainer(lanes: tuple[Lane, ...], seat=None, **config) -> types.SimpleNamespace:
    """What `LaneRatings` reads of a trainer: its lanes and their seat, the update's
    geometry (one 128-tick shard, a 128-tick lookahead) and episodes of `longest` ticks,
    a model and its counters."""
    longest = config.pop("longest", 512)
    model = torch.nn.Linear(2, 2)
    return types.SimpleNamespace(
        lanes=lanes,
        lane_seat=seat,
        config=types.SimpleNamespace(
            lanes=LaneConfig(**config), shards=1, length=128, lookahead=128
        ),
        rollout=types.SimpleNamespace(episode_ticks=torch.tensor([float(longest)]), device=lc.CPU),
        model=model,
        ppo=types.SimpleNamespace(environment_steps=0),
        updates_done=0,
    )


def test_lane_ratings_chain_the_ladder_to_the_scripted_agent():
    """rung_0.9 scores 0.25 against the scripted agent and rung_0.8 0.5 against rung_0.9:
    both near 1000 - 400 log10(3); live, without games, is not rated."""
    top = Lane(MODE, Matchup("rung_0.9", SCRIPTED_PLAYER))
    below = Lane(MODE, Matchup("rung_0.8", "rung_0.9"))
    live = Lane(MODE, Matchup(LIVE, SCRIPTED_PLAYER))
    ratings = LaneRatings(_trainer((top, below, live), window=2))
    ratings.update({top: (250, 0, 750), below: (500, 0, 500), live: (0, 0, 0)})
    expected = 1000.0 - 400.0 * math.log10(3.0)
    rung, error = ratings.rating(MODE, "rung_0.9")
    assert rung == pytest.approx(expected, abs=2.0) and 0 < error < 30
    assert ratings.rating(MODE, "rung_0.8")[0] == pytest.approx(rung, abs=3.0)
    assert ratings.rating(MODE, LIVE) is None
    # a lane whose games do not reach the scripted agent rates nobody
    apart = Lane(MODE, Matchup("random", "rung_0.1"))
    ratings = LaneRatings(_trainer((top, below, live, apart), window=2))
    ratings.update({top: (250, 0, 750), below: (500, 0, 500), live: (0, 0, 0), apart: (1, 2, 3)})
    assert ratings.rating(MODE, "random") is None
    series = ratings.series()
    assert f"rating/live/{MODE}" not in series and f"lanes/rating/{MODE}/rung_0.9" in series


def test_a_moving_players_games_leave_after_the_window_and_the_ladders_stay():
    rung = Lane(MODE, Matchup("rung_0.9", SCRIPTED_PLAYER))
    live = Lane(MODE, Matchup(LIVE, SCRIPTED_PLAYER))
    trainer = _trainer((rung, live), window=2)
    ratings = LaneRatings(trainer)
    ratings.update({rung: (10, 0, 30), live: (40, 0, 0)})
    assert ratings.rating(MODE, LIVE)[0] > 1000.0
    ratings.update({rung: (10, 0, 30), live: (20, 0, 20)})
    ratings.update({rung: (10, 0, 30), live: (0, 0, 40)})  # the sweep has left the window
    assert ratings.rating(MODE, LIVE)[0] < 1000.0
    assert ratings._games(MODE)[("rung_0.9", SCRIPTED_PLAYER)] == [30.0, 0.0, 90.0]
    restored = LaneRatings(trainer, ratings.state(), ratings.tensors())
    assert restored.state() == ratings.state() and restored.series() == ratings.series()


def _expected(n: int, gap: float) -> tuple[float, float, float]:
    """`n` games won at the Bradley-Terry odds of a `gap`-point favourite: (wins, 0, losses)."""
    p = 1.0 / (1.0 + 10.0 ** (-gap / 400.0))
    return (n * p, 0.0, n * (1.0 - p))


def test_a_strong_player_is_rated_far_better_through_a_snapshot_chain():
    """A live policy 1000 points above the scripted agent (2000) plays the same 300 games
    in its window: against the scripted agent (p = 0.997), or against the newest of a chain
    of snapshots 250 points apart (p = 0.81), whose links, stationary, have gathered 3000
    games each over the run. Through the chain its standard error is several times
    smaller; both fits find the rating."""
    slots = ("snapshot_0", "snapshot_1", "snapshot_2")
    alone = Lane(MODE, Matchup(LIVE, SCRIPTED_PLAYER))
    direct = LaneRatings(_trainer((alone,), window=1, snapshot_every=10**6))
    direct.update({alone: _expected(300, 1000.0)})
    links = [Lane(MODE, Matchup(slots[0], SCRIPTED_PLAYER))]
    links += [Lane(MODE, Matchup(high, low)) for low, high in zip(slots, slots[1:], strict=False)]
    top = Lane(MODE, Matchup(LIVE, slots[-1]))
    ratings = LaneRatings(_trainer((*links, top), window=1, snapshot_every=10**6))
    ratings.holders = [f"lane_snapshot_{index}" for index in range(3)]
    ratings.since = [-(10**6)] * 3
    for _ in range(10):  # the links' games over ten updates
        ratings.update({lane: _expected(300, 250.0) for lane in links})
    ratings.update({top: _expected(300, 250.0)})
    through_chain, chain_error = ratings.rating(MODE, LIVE)
    straight, direct_error = direct.rating(MODE, LIVE)
    # each player's prior game against the scripted agent pulls the far end in a little
    assert through_chain == pytest.approx(2000.0, abs=25.0)
    assert abs(straight - 2000.0) < direct_error  # within its own (large) error
    assert chain_error * 4 < direct_error, (chain_error, direct_error)
    assert ratings.rating(MODE, "lane_snapshot_1")[0] == pytest.approx(1500.0, abs=10.0)


def test_snapshot_slots_credit_games_to_the_snapshot_and_settle_after_a_swap():
    """A slot's games go to the snapshot it held; a swap is credited to nobody until every
    game since it began after it; the retired snapshot keeps its games and its rating."""
    lane = Lane(MODE, Matchup("snapshot_0", SCRIPTED_PLAYER))
    trainer = _trainer((lane,), window=1, snapshot_every=2, longest=512)
    ratings = LaneRatings(trainer)  # the starting weights, already settled
    assert ratings.settle == 2 and ratings.holders == ["lane_snapshot_0"]
    trainer.updates_done = 1
    ratings.update({lane: (10, 0, 10)}, 1)
    trainer.updates_done, trainer.ppo.environment_steps = 2, 2048
    ratings.freeze(trainer)  # due: the live weights into the (only) slot
    assert ratings.holders == ["lane_snapshot_2048"]
    for update in (3, 4):  # games that began under the old weights
        ratings.update({lane: (99, 0, 0)}, update)
    assert ratings.rating(MODE, "lane_snapshot_2048") is None
    ratings.update({lane: (5, 0, 15)}, 5)
    assert ratings._games(MODE) == {
        ("lane_snapshot_0", SCRIPTED_PLAYER): [10.0, 0.0, 10.0],
        ("lane_snapshot_2048", SCRIPTED_PLAYER): [5.0, 0.0, 15.0],
    }
    assert ratings.rating(MODE, "lane_snapshot_0")[0] == pytest.approx(1000.0, abs=1.0)


def test_a_slot_swap_copies_weights_in_place():
    """The lanes' shapes are fixed at capture: freezing a snapshot into a slot changes its
    weights, not the tensors the captured graph reads."""
    lanes = (Lane(MODE, Matchup("snapshot_0", SCRIPTED_PLAYER)),)
    seat = _seat(lanes)
    trainer = _trainer(lanes, seat=seat, window=1, snapshot_every=1)
    trainer.model = seat.decision.model
    before = {name: (value.data_ptr(), value.shape) for name, value in seat.parameters.items()}
    LaneRatings(trainer)
    with torch.no_grad():
        for parameter in trainer.model.parameters():
            parameter.add_(1.0)
    trainer.updates_done = 1
    ratings = LaneRatings(trainer)
    ratings.freeze(trainer)
    after = {name: (value.data_ptr(), value.shape) for name, value in seat.parameters.items()}
    assert after == before
    for name, value in trainer.model.state_dict().items():
        assert torch.equal(seat.parameters[f"model.{name}"][0], value), name


def test_the_gauge_takes_the_lanes_ladder_and_live_ratings():
    gauge = Gauge(initial_live=850.0)
    assert gauge.live.rating == 850.0  # a warm start's parent's
    gauge.add("rung_0.9", "ladder", 900.0)
    gauge.add("scripted", "ladder", 1000.0)
    gauge.calibrate({"rung_0.9": (260.0, 20.0), "unknown": (1.0, 1.0)})
    assert gauge.entries["rung_0.9"].rating == 260.0 and gauge.entries["scripted"].rating == 1000.0
    gauge.update({}, live=(700.0, 25.0))
    assert (gauge.live.rating, gauge.live.stderr) == (700.0, 25.0)


def test_a_warm_starts_league_starts_from_its_parents_rating(tmp_path):
    import league_cases

    trainer, league = league_cases.league(tmp_path)
    assert all(gauge.live.rating == 0.0 for gauge in league.gauges.values())
    from bnb.league.league import League

    other = League(trainer, tmp_path / "other", ratings={MODE: 640.0})
    assert other.gauges[MODE].live.rating == 640.0


def test_a_clone_keeps_its_lane_rating_in_its_policy_file(tmp_path):
    from bnb.runs import checkpoint, session
    from bnb.runtime.train import Trainer

    config = tiny_train_config(
        objective="bc",
        pools=cloning.LINEUP,
        learner=dataclasses.replace(tiny_train_config().learner, policy=0.025),
        lanes=LaneConfig(environments=2, matchups=cloning.LANES.matchups),
    )
    trainer = Trainer(config, lc.CPU)
    hooks = LaneHooks(trainer)
    (lane,) = trainer.lanes
    hooks.ratings.update({lane: (30, 0, 10)})
    run = types.SimpleNamespace(policy_path=tmp_path / "policy")
    session.save_clone(run, trainer)
    ratings = checkpoint.read_policy(tmp_path / "policy").ratings
    assert ratings[MODE] == pytest.approx(hooks.ratings.rating(MODE, LIVE)[0])
    assert ratings[MODE] > 1000.0


def test_bcs_stop_reads_the_live_policys_lane_games_against_the_scripted_agent():
    record = {
        Lane("a", Matchup(LIVE, SCRIPTED_PLAYER)): (3, 1, 2),
        Lane("b", Matchup(SCRIPTED_PLAYER, LIVE)): (5, 0, 1),  # from the scripted side
        Lane("b", Matchup("rung_0.5", SCRIPTED_PLAYER)): (9, 9, 9),
    }
    trainer = types.SimpleNamespace(lane_record=record)
    assert cloning.lane_results(trainer) == (4, 1, 7)
