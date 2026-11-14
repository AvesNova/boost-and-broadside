"""Snapshot schedules (`bnb.runtime.snapshots`): a list of steps then a period; the RL
policy files' and the league members' defaults (decided 2026-11-14)."""

import pytest

from bnb.contracts import GameConfig
from bnb.runtime.opponents import LeagueConfig
from bnb.runtime.snapshots import LEAGUE_SNAPSHOTS, MILLION, RL_SNAPSHOTS, SnapshotSchedule
from bnb.runtime.train import TrainConfig


def test_rl_snapshots_are_dense_early_then_every_5m():
    expected = [1, 2, 3, 4, 6, 8, 10, 12, 16, 20, 25, 30, 35, 40]
    assert RL_SNAPSHOTS.points(40 * MILLION) == [step * MILLION for step in expected]
    assert TrainConfig(game=GameConfig(3, 2, 1200.0)).policy_snapshots == RL_SNAPSHOTS


def test_league_members_follow_the_list_then_keep_its_last_gap():
    listed = [5, 10, 15, 20, 25, 30, 40, 50, 60, 75, 90, 110, 130, 160]
    expected = listed + [190, 220, 250]
    assert LEAGUE_SNAPSHOTS.period == 30 * MILLION
    assert LEAGUE_SNAPSHOTS.points(250 * MILLION) == [step * MILLION for step in expected]
    assert LeagueConfig().snapshots == LEAGUE_SNAPSHOTS


@pytest.mark.parametrize("schedule", [RL_SNAPSHOTS, LEAGUE_SNAPSHOTS])
def test_an_update_is_due_when_a_point_falls_in_it(schedule):
    """Updates of 0.75M steps from 0 to 300M: each scheduled point is taken by exactly the
    update whose (before, after] holds it."""
    step, taken = 750_000, []
    for before in range(0, 300 * MILLION, step):
        if schedule.due(before, before + step):
            taken.append(before + step)
    points = schedule.points(300 * MILLION)
    assert len(taken) == len(points)
    assert all(after - step < point <= after for point, after in zip(points, taken, strict=True))


def test_a_period_alone_is_the_old_cadence():
    schedule = SnapshotSchedule(every=256)
    for before in range(0, 5000, 96):
        after = before + 96
        assert schedule.due(before, after) == (after // 256 > before // 256)
    assert SnapshotSchedule((7,)).period == 7


@pytest.mark.parametrize("steps, every", [((), None), ((2, 1), None), ((0, 3), None), ((1, 2), 0)])
def test_a_schedule_needs_ascending_positive_steps_or_a_period(steps, every):
    with pytest.raises(ValueError, match="snapshot"):
        SnapshotSchedule(steps, every)
