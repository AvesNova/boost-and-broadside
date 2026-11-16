"""The unified evaluator's CPU parts: buckets and benching, the seat's lane compaction, the
planner's allocation, the targets' stopping rules and the decisions' information-fraction
looks (the false-positive rate under the null)."""

import math
import random

import pytest
import torch

from bnb.league.sequential import SequentialDesign, boundaries, operating_characteristics
from bnb.league.unified.cells import Cell, Condition, Outcome, arena_uniform, bucket_of
from bnb.league.unified.engine import BucketEngine, Slot, _split, lane_runs, size_environments
from bnb.league.unified.planner import Planner
from bnb.league.unified.seat import RefillSeat
from bnb.league.unified.targets import (
    DecisionTarget,
    EloTarget,
    InformationLooks,
    WinRateTarget,
)

DM, EL, FL = "real_scale_deathmatch", "real_scale_elimination", "real_scale_frontline"


# ---- buckets and benching


def test_bucket_is_the_smallest_holding_the_larger_team():
    assert bucket_of(Condition(DM, 1, 1)) == 5
    assert bucket_of(Condition(DM, 5, 5)) == 5
    assert bucket_of(Condition(DM, 4, 6)) == 10
    assert bucket_of(Condition(FL, 20, 13)) == 20
    with pytest.raises(ValueError):
        bucket_of(Condition(DM, 21, 1))


def test_condition_keys():
    assert Condition(DM, 5, 5).key == DM
    assert Condition(DM, 2, 3).key == f"{DM}_2v3"
    assert Condition(EL, 5, 5, 0.5).key == f"{EL}_scale0.5"


def _stub_engine(bucket: int = 5) -> BucketEngine:
    engine = BucketEngine.__new__(BucketEngine)
    engine.bucket, engine.lineup_types = bucket, (0, 1)
    engine.random = random.Random(0)
    engine.index = {"a": 0, "b": 1}
    engine.scripted = 0

    class Rollout:
        arena_slot = None

    engine.rollout = Rollout()
    return engine


@pytest.mark.parametrize("seat", [0, 1])
def test_benching_puts_each_players_count_on_its_team(seat):
    engine = _stub_engine()
    slot = Slot(Cell(Condition(EL, 2, 3), "a", "b"), seat)
    seat_out, active, ship_type, arena, first, second = engine._pending_values(slot)
    assert seat_out == seat and (first, second) == (0, 1)
    assert active[seat] == 2 and active[1 - seat] == 3
    assert len(ship_type) == 10 and ship_type[:5] == ship_type[5:]  # both teams fly it
    assert arena == -1.0


def test_arena_uniform_inverts_the_draw():
    sigma, low, high = 0.35, 0.5, 2.0
    for scale in (0.5, 0.8, 1.0, 1.7, 2.0):
        u = arena_uniform(scale, sigma, low, high)
        cdf = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))  # noqa: E731
        first, last = cdf(math.log(low) / sigma), cdf(math.log(high) / sigma)
        p = min(max(first + u * (last - first), 1e-6), 1 - 1e-6)
        drawn = math.exp(sigma * float(torch.special.ndtri(torch.tensor(p, dtype=torch.float64))))
        assert drawn == pytest.approx(scale, rel=1e-3)


def test_split_and_sizing():
    assert _split(10, [1.0, 1.0, 0.0]) == [5, 5, 0]
    assert sum(_split(1000, [0.3, 0.3, 0.4])) == 1000
    # 0.7 of 8 GB less 1 GB fixed at 1 MB an environment: 4600 -> 4544 (a multiple of 64)
    assert size_environments(1e9, 1e6, 8_000_000_000) == 4544
    assert size_environments(1e9, 1e9, 1e9) == 64  # at least one multiple


# ---- the seat's lanes


def _stub_seat(runs: list[list[int]], capacity: list[int], players: int, rows: int):
    seat = RefillSeat.__new__(RefillSeat)
    seat.rows, seat.lanes, seat.capacity = rows, runs, capacity
    lane_of, run_of = [-1] * players, [-1] * players
    for run, members in enumerate(runs):
        for lane, index in enumerate(members):
            lane_of[index], run_of[index] = lane, run
    seat.lane_of, seat.run_of = torch.tensor(lane_of), torch.tensor(run_of)
    seat.overflow = torch.zeros(1, dtype=torch.int64)
    seat._row = torch.arange(rows)
    return seat


def test_lane_tables_compact_each_players_rows():
    # players 0 scripted, 1 and 2 in run 0, 3 in run 1
    seat = _stub_seat([[1, 2], [3]], [3, 2], players=4, rows=8)
    player = torch.tensor([1, 0, 2, 1, 3, 0, 2, 2])
    read, write = seat._table(0, player)
    assert read.shape == (2, 3)
    assert write[0].tolist() == [0, 3, 8]  # player 1's rows, then padding (spare row 8)
    assert write[1].tolist() == [2, 6, 7]
    assert read[0].tolist() == [0, 3, 0]  # padding reads a real row
    read, write = seat._table(1, player)
    assert write.tolist() == [[4, 8]]
    assert int(seat.overflow) == 0
    seat._table(1, torch.full((8,), 3))  # 8 rows of player 3, capacity 2
    assert int(seat.overflow) == 6


def test_lane_runs_cut_players_by_demand():
    # group 0: a champion in half the seats, two light players; group 1: one player
    groups = [None, 0, 0, 0, 1]
    shares = [0.4, 0.5, 0.05, 0.04, 0.01]
    runs = lane_runs(groups, shares, rows=1000)
    assert [(g, members) for g, members, _ in runs] == [(0, [1]), (0, [2, 3]), (1, [4])]
    assert [capacity for *_, capacity in runs] == [750, 75, 32]


# ---- the planner


def _simulate(planner: Planner, strengths: dict, slices, rounds: int, seed: int = 0) -> int:
    rng = random.Random(seed)
    games = 0
    for round_ in range(rounds):
        if round_ % 25 == 0 and planner.done():
            break
        for bucket, preset in slices:
            chosen = planner.choose(bucket, preset)
            if chosen is None:
                continue
            cell, _ = chosen
            gap = strengths[cell.first] - strengths[cell.second]
            score = 1.0 if rng.random() < 1 / (1 + math.exp(-gap)) else 0.0
            planner.finished(cell, Outcome(score, seconds=60.0))
            games += 1
    return games


def test_win_rate_cells_fill_evenly_and_stop_at_their_goal():
    cells = [Cell(Condition(EL, n, n + 1), "a", "b") for n in (1, 2, 3)]
    targets = [WinRateTarget(f"w{n}", cell, stderr=0.08) for n, cell in enumerate(cells)]
    planner = Planner(targets)
    _simulate(planner, {"a": 0.3, "b": 0.0}, [(5, EL)], 2000)
    assert planner.done()
    games = [t.games for t in targets]
    assert max(games) - min(games) <= 12
    for target in targets:
        assert target.stderr() <= 0.08 + 1e-9
        assert target.proposals() == {}


def test_in_flight_games_spread_a_round_over_the_cells():
    cells = [Cell(Condition(EL, 1, 1), "a", f"b{n}") for n in range(4)]
    planner = Planner([WinRateTarget(f"w{n}", c, 0.1) for n, c in enumerate(cells)])
    chosen = [planner.choose(5, EL)[0] for _ in range(8)]
    assert sorted(chosen.count(c) for c in cells) == [2, 2, 2, 2]


def test_choose_keeps_to_bucket_preset_and_fits():
    a = Cell(Condition(EL, 1, 1), "a", "b")
    b = Cell(Condition(DM, 1, 1), "a", "b")
    c = Cell(Condition(EL, 8, 8), "a", "b")
    planner = Planner([WinRateTarget(f"w{i}", x, 0.1) for i, x in enumerate((a, b, c))])
    assert planner.choose(5, EL)[0] == a
    assert planner.choose(10, EL)[0] == c
    assert planner.choose(5, FL) is None
    assert planner.choose(5, DM, fits=lambda cell: False) is None


def test_seats_alternate_per_cell():
    cell = Cell(Condition(EL, 1, 1), "a", "b")
    planner = Planner([WinRateTarget("w", cell, 0.01)])
    assert [planner.choose(5, EL)[1] for _ in range(4)] == [0, 1, 0, 1]


def test_cheaper_cells_are_preferred_at_equal_value():
    quick = Cell(Condition(EL, 1, 1), "a", "b")
    slow = Cell(Condition(FL, 3, 3), "a", "b")
    planner = Planner([WinRateTarget("q", quick, 0.05), WinRateTarget("s", slow, 0.05)])
    first = [planner.choose(5, preset)[0] for preset in (EL, FL)]
    assert first == [quick, slow]  # each slice plays its own
    for _ in range(4):
        planner.finished(quick, Outcome(1.0, seconds=10.0))
        planner.finished(slow, Outcome(1.0, seconds=500.0))
    assert planner.expected_seconds(quick.condition) < planner.expected_seconds(slow.condition)


def test_elo_target_stops_at_its_goal():
    players = ["scripted", "p1", "p2", "p3"]
    strengths = {"scripted": 0.0, "p1": 0.4, "p2": 0.8, "p3": 1.2}
    target = EloTarget("elo", Condition(DM, 5, 5), players, stderr=80.0)
    planner = Planner([target])
    games = _simulate(planner, strengths, [(5, DM)], 20000)
    assert target.done() and games < 20000
    report = target.report()
    for name, error in report["errors"].items():  # rated to the goal, or a settled bound
        assert (error is not None and error <= 80.0 + 1e-6) or report["status"][name] != "rated"
    # A-optimal: no pair is starved and none gets most of the games
    played = target.games[torch.triu_indices(4, 4, 1).unbind()].tolist()
    assert max(played) < 0.6 * sum(played)


def test_elo_contrast_goal_draws_games_to_its_pair():
    players = ["scripted", "p1", "p2"]
    loose = EloTarget("e", Condition(DM, 5, 5), players, stderr=1e6, contrasts=(("p1", "p2", 40),))
    proposals = loose.proposals()
    best = max(proposals, key=proposals.get)
    assert {best.first, best.second} == {"p1", "p2"}


# ---- decisions


def test_decision_value_depends_on_counts_only():
    cell = Cell(Condition(DM, 5, 5), "a", "b")
    design = SequentialDesign(4, 0.05, 0.2, 0.1)
    winning, losing = DecisionTarget("w", cell, design), DecisionTarget("l", cell, design)
    for _ in range(20):
        winning.record(cell, Outcome(1.0), pending=False)
        losing.record(cell, Outcome(0.0), pending=False)
    assert winning.proposals() == losing.proposals()


def test_looks_come_at_information_fractions():
    looks = InformationLooks(SequentialDesign(4, 0.05, 0.2, 0.1))
    n = looks.maximum_games
    assert not looks.due(n // 4 - 1) and looks.due(math.ceil(n / 4))
    looks.look(math.ceil(0.6 * n), 0.0)  # late: passes the 0.25 and 0.5 looks
    assert looks.next_fraction() == 0.75
    assert looks.looks[-1].fraction == pytest.approx(math.ceil(0.6 * n) / n)


def test_irregular_looks_spend_exactly_alpha_under_the_null():
    """Each look schedule the refill may produce (fractions reached in bursts, overshooting
    the planned ones), its boundaries spent at the actual fractions: the exact chance of a
    winner when the difference is 0 is alpha (the schedule depends on the counts alone, so
    over schedules it stays alpha)."""
    design = SequentialDesign(4, 0.05, 0.2, 0.1)
    looks = InformationLooks(design)
    rng = random.Random(1)
    for _ in range(4):
        fractions, reached = [], 0.0
        for planned in (0.25, 0.5, 0.75, 1.0):
            if planned <= reached:
                continue
            reached = planned + rng.uniform(0.0, 0.12)
            fractions.append(min(reached, 1.12))
        found = boundaries(design, fractions, looks.drift, True)
        rate = operating_characteristics(found, 0.0)
        assert rate.a_wins + rate.b_wins == pytest.approx(design.alpha, abs=2e-3)


def _null_decisions(tests: int, seed: int, draw_share: float = 0.2, delta: float = 0.1) -> float:
    """The share of decisions with a winner when A and B are equal, games ending in random
    bursts (the continuous refill's) and scored win, draw or loss."""
    rng = random.Random(seed)
    cell = Cell(Condition(DM, 5, 5), "a", "b")
    design = SequentialDesign(4, 0.05, 0.2, delta)
    winners = 0
    for _ in range(tests):
        target = DecisionTarget("d", cell, design)
        while not target.done() and target.games < 2 * target.test.maximum_games:
            for _ in range(rng.randint(1, 12)):  # a burst of games ending in one block
                u = rng.random()
                score = 0.5 if u < draw_share else (1.0 if u < 0.5 + draw_share / 2 else 0.0)
                target.record(cell, Outcome(score), pending=False)
        winners += target.test.outcome in ("A", "B")
    return winners / tests


def test_null_false_positive_rate_quick():
    assert _null_decisions(150, 0) <= 0.05 + 3 * math.sqrt(0.05 * 0.95 / 150)


@pytest.mark.slow
def test_null_false_positive_rate():
    """At an 850-game maximum (delta 0.05; the first look at about 210 games). Measured
    2026-11-17: 0.045 and 0.048 +- 0.002 over 10,000 tests without and with 20% draws; at a
    213-game maximum (delta 0.1, the first look at 53 games) 0.050 and 0.054 +- 0.0016
    over 20,000, the estimated variance's small-sample excess."""
    tests = 4000
    rate = _null_decisions(tests, 2, delta=0.05)
    assert rate <= 0.05 + 2 * math.sqrt(0.05 * 0.95 / tests)


# ---- records


def test_game_log_resumes_and_exports_to_a_rating_pool(tmp_path):
    from bnb.league.matches import player
    from bnb.league.rating_pool import RatingPool, fit
    from bnb.league.unified.engine import GameLog

    players = [player("imugi"), player("rung_T2")]
    names = [one.name for one in players]
    pool = RatingPool.create(tmp_path / "pool", lineup_types=("Yongma", "Jujak"))
    log = GameLog(tmp_path / "run")
    condition = Condition(DM, 5, 5)
    log.export_to(pool, players, {condition.key})
    cell = Cell(condition, names[0], names[1])
    for score in (1.0, 1.0, 0.5, 0.0):
        log.add(cell, Outcome(score, 3, 1, 120.0), Slot(cell, 0), 5)
    other = Cell(Condition(EL, 2, 3), names[0], names[1])
    log.add(other, Outcome(1.0), Slot(other, 1), 5)
    log.flush()
    replayed = GameLog(tmp_path / "run").replay()
    assert [c for c, _ in replayed] == [cell] * 4 + [other]
    assert replayed[2][1].score == 0.5 and replayed[0][1].first_kills == 3
    counted, _, _, _ = fit(pool, condition.key, ("win_loss", "win_loss"))
    assert counted.games == 4  # the Elo condition only
