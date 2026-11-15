"""Snapshot schedules (`bnb.runtime.snapshots`): a list of steps, a period, and a period
that doubles with the steps; the RL policy files' default (decided 2026-11-14) and the league
members' (decided 2026-11-15)."""

import json

import pytest

from bnb.contracts import GameConfig
from bnb.runs.serialise import from_json, to_json
from bnb.runtime.opponents import LeagueConfig
from bnb.runtime.snapshots import LEAGUE_SNAPSHOTS, MILLION, RL_SNAPSHOTS, SnapshotSchedule
from bnb.runtime.train import TrainConfig


def test_rl_snapshots_are_dense_early_then_every_5m():
    expected = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 25, 30, 35, 40]
    assert RL_SNAPSHOTS.points(40 * MILLION) == [step * MILLION for step in expected]
    assert TrainConfig(game=GameConfig(3, 2, 1200.0)).policy_snapshots == RL_SNAPSHOTS


def test_league_members_are_every_5m_to_50m_then_the_interval_doubles_with_the_steps():
    expected = (
        list(range(5, 51, 5))  # every 5M to 50M
        + [60, 70, 80, 90, 100]  # every 10M to 100M
        + [120, 140, 160, 180, 200]  # every 20M to 200M
        + [240, 280, 320, 360, 400]  # every 40M to 400M
        + [480, 560]  # every 80M to 800M
    )
    assert LEAGUE_SNAPSHOTS.points(600 * MILLION) == [step * MILLION for step in expected]
    assert LEAGUE_SNAPSHOTS.points(4 * MILLION) == []
    # each stage holds five members, ever after
    for stage in range(10):
        start = 50 * MILLION << stage
        stretch = [point for point in LEAGUE_SNAPSHOTS.points(2 * start) if point > start]
        assert stretch == [start + k * (10 * MILLION << stage) for k in range(1, 6)]
    assert LeagueConfig().snapshots == LEAGUE_SNAPSHOTS


def test_the_count_matches_the_points():
    for schedule in (
        RL_SNAPSHOTS,
        LEAGUE_SNAPSHOTS,
        SnapshotSchedule((3, 8), 2, doubling_from=12),
    ):
        small = [0, 1, 2, 3, 7, 8, 12, 13, 48, 49, 50, 99]
        for steps in [*small, *range(0, 3000 * MILLION, 7 * MILLION)]:
            assert schedule.count(steps) == len(schedule.points(steps)), (schedule, steps)


@pytest.mark.parametrize("schedule", [RL_SNAPSHOTS, LEAGUE_SNAPSHOTS])
@pytest.mark.parametrize("step", [600_000, 750_000, 999_999])
def test_an_update_is_due_when_a_point_falls_in_it(schedule, step):
    """Updates of `step` environment steps (under the schedule's smallest gap) from 0 to
    1000M, most crossing a scheduled point off the updates' grid: each point is taken by
    exactly the update whose (before, after] holds it."""
    taken = []
    for before in range(0, 1000 * MILLION, step):
        if schedule.due(before, before + step):
            taken.append(before + step)
    points = schedule.points(before + step)
    assert len(taken) == len(points)
    assert all(after - step < point <= after for point, after in zip(points, taken, strict=True))


def test_a_period_alone_is_the_old_cadence():
    schedule = SnapshotSchedule(every=256)
    for before in range(0, 5000, 96):
        after = before + 96
        assert schedule.due(before, after) == (after // 256 > before // 256)
    assert SnapshotSchedule((7,)).period == 7


@pytest.mark.parametrize(
    "steps, every, doubling_from",
    [
        ((), None, None),
        ((2, 1), None, None),
        ((0, 3), None, None),
        ((1, 2), 0, None),
        ((), 5, 5),  # not a multiple of twice the period
        ((), 5, 15),
        ((20,), 5, 20),  # not after the listed steps
        ((3,), 5, 10),  # off the period's grid after 3
    ],
)
def test_a_schedule_needs_ascending_positive_steps_or_a_period(steps, every, doubling_from):
    with pytest.raises(ValueError, match="snapshot"):
        SnapshotSchedule(steps, every, doubling_from)


def test_a_doubling_schedule_survives_the_checkpoints_json():
    data = to_json(LEAGUE_SNAPSHOTS)
    assert json.loads(json.dumps(data)) == data
    assert from_json(SnapshotSchedule, data) == (LEAGUE_SNAPSHOTS, [])
    # a schedule recorded before the doubling existed reads as one without it
    assert from_json(SnapshotSchedule, dict(steps=[], every=256))[0] == SnapshotSchedule(every=256)
