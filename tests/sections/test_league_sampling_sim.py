"""The league-sampling simulation (`bench.league_sampling`) on synthetic inputs: the run's old
ladder fit and its failure, the residency bookkeeping, the slot simulation's limits and the
samplers' weights."""

import math

import pytest
import torch

from bench import league_sampling as sim
from bnb.league.ratings import fit_ratings

LADDER = [("random", "rung", 0.0), ("rung_T2", "rung", 950.0), ("scripted", "scripted", 1000.0)]


def _update(index: int = 0, snapshots: int = 1, live: float = 1500.0) -> sim.Update:
    entries = [*LADDER, ("exponential_average", "average", 1450.0)]
    entries += [(f"snapshot_{k}", "checkpoint", 1400.0 + 50 * k) for k in range(snapshots)]
    rows = {
        mode: [
            sim.Row(name, kind, rating, 50.0, 0.0, 0.0, None, None)
            for name, kind, rating in entries
        ]
        for mode in sim.MODES
    }
    update = sim.Update(index, (index + 1) * 1000, dict.fromkeys(sim.MODES, live), rows, {}, {}, {})
    for mode in sim.MODES:
        weights = sim.weights_at(live, rows[mode])
        for row, weight in zip(rows[mode], weights.tolist()):
            row.share = weight
    return update


def test_the_runs_ladder_fit_runs_away_on_a_near_sweep_where_the_maximum_is_finite():
    # a snapshot that won 28 of 29 games against rungs rated 900-1000 (run 2026-11-06's
    # update-17 credit, frontline): the old fit's first Newton step from 0 lands where
    # every probability is 1 in float64, the Hessian is 0 and it stops there
    opponents = torch.tensor(
        [680.0, 920.0, 930.0, 955.0, 960.0, 962.0, 1000.0], dtype=torch.float64
    )
    won = torch.tensor([1.0, 3.0, 2.0, 1.0, 4.0, 9.0, 8.0], dtype=torch.float64)
    lost = torch.tensor([0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=torch.float64)
    count = opponents.shape[0] + 1
    scored = torch.zeros((count, count), dtype=torch.float64)
    scored[-1, :-1], scored[:-1, -1] = won, lost
    fixed = torch.tensor([True] * (count - 1) + [False])
    ratings = torch.cat([opponents, torch.zeros(1, dtype=torch.float64)])
    runaway = float(sim.fit_ladder_724153e7(scored, fixed, ratings, prior_games=1.0)[-1])
    maximum = sim.snapshot_maximum(won, lost, opponents, float(opponents.max()))
    assert runaway > 10_000
    assert 1300 < maximum < 2000
    # today's one fit, the rungs pinned to the scripted agent by many games, is sane
    wins = torch.zeros((count, count), dtype=torch.float64)
    scripted = count - 2
    for index in range(count - 2):
        p = 1 / (1 + 10 ** ((1000 - float(opponents[index])) / 400))
        wins[index, scripted], wins[scripted, index] = 2000 * p, 2000 * (1 - p)
    wins[-1, :-1], wins[:-1, -1] = won, lost
    fitted, _ = fit_ratings(wins, scripted, 1.0)
    assert abs(float(fitted[-1]) + 1000 - maximum) < 150


def test_residency_splits_the_draining_slot_and_spreads_a_ladder_slot_over_the_rungs():
    ladder = {mode: {"random": 0.25, "rung_T2": 0.25, "scripted": 0.5} for mode in sim.MODES}
    held = ["scripted", "exponential_average", "snapshot_0", "snapshot_0", "random"]
    resident = sim.residency(held, 0.5, 3, ladder)
    for mode in sim.MODES:
        assert sum(resident[mode].values()) == pytest.approx(1.0)
        # a slot and a half of ladder, a quarter of it random
        assert resident[mode]["random"] == pytest.approx(0.25 * 0.25 + 0.125 * 0.25)
        assert resident[mode]["snapshot_0"] == pytest.approx(0.25 + 0.125)


def test_the_slots_follow_the_weights_on_average_and_a_draw_per_row_is_the_weights():
    updates = [_update(index) for index in range(60)]
    weights = [
        {mode: dict(zip(u.names(), sim.shares(u.rows[mode]).tolist())) for mode in sim.MODES}
        for u in updates
    ]
    (instant,) = sim.simulate(updates, weights, seeds=1, instant=True)
    assert instant[-1]["frontline"] == pytest.approx(weights[-1]["frontline"])
    runs = sim.simulate(updates, weights, seeds=300, drain=4)
    mean = sum(run[-1]["frontline"].get("snapshot_0", 0.0) for run in runs) / len(runs)
    assert mean == pytest.approx(weights[-1]["frontline"]["snapshot_0"], abs=0.03)


@pytest.mark.parametrize("name", list(sim.SAMPLERS))
def test_every_sampler_gives_a_distribution(name):
    update = _update(snapshots=3)
    ratings = {mode: {row.name: row.rating for row in update.rows[mode]} for mode in sim.MODES}
    (weights,) = sim.sampler_weights(sim.SAMPLERS[name], [update], [ratings])
    for mode in sim.MODES:
        values = list(weights[mode].values())
        assert all(value >= 0 and math.isfinite(value) for value in values)
        assert sum(values) == pytest.approx(1.0)


def test_nash_over_a_transitive_pool_is_its_strongest_entry():
    update = _update(snapshots=3)
    ratings = {mode: {row.name: row.rating for row in update.rows[mode]} for mode in sim.MODES}
    (weights,) = sim.sampler_weights(sim.nash, [update], [ratings])
    assert max(weights["frontline"], key=weights["frontline"].get) == "snapshot_2"
    assert weights["frontline"]["snapshot_2"] > 0.95
