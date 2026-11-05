"""The pool (league spec 1, 3 and 5): retention, the sampling weights and the averages."""

import math

import pytest
import torch

from bnb.league.pool import (
    RECENT_SNAPSHOTS,
    Entry,
    average_into,
    ladder_entries,
    ladder_name,
    retained,
    sampling_weights,
)
from bnb.runtime.opponents import Rung, ladder_rung, strength_order


def _apply_after_every_snapshot(count: int) -> set[int]:
    kept: set[int] = set()
    for ordinal in range(count):
        kept = retained(kept | {ordinal})
    return kept


def test_retention_keeps_the_newest_eight():
    kept = retained(set(range(8)))
    assert kept == set(range(8))
    assert set(range(92, 100)) <= _apply_after_every_snapshot(100)


def test_retention_keeps_the_snapshot_nearest_each_power_of_two_back():
    for count in (9, 20, 37, 100, 513):
        kept = _apply_after_every_snapshot(count)
        newest = count - 1
        older = sorted(newest - ordinal for ordinal in kept if newest - ordinal >= RECENT_SNAPSHOTS)
        # each older one is the nearest of the kept to some age 2^k, k >= 3
        for age in older:
            assert any(
                min(older, key=lambda a: (abs(a - (1 << k)), -a)) == age for k in range(3, 20)
            ), (count, age, older)
        assert len(kept) <= RECENT_SNAPSHOTS + math.ceil(math.log2(count)) + 1


def test_retention_keeps_the_first_snapshot_as_the_deepest_history():
    assert 0 in _apply_after_every_snapshot(300)


def test_retention_spans_the_history_geometrically():
    kept = _apply_after_every_snapshot(1000)
    ages = sorted(999 - ordinal for ordinal in kept)
    for power in range(3, 10):
        target = 1 << power
        assert min(abs(age - target) for age in ages) <= target / 2, (target, ages)


def test_the_sampling_weight_is_p_times_one_minus_p_plus_the_floor():
    probability = torch.tensor([0.5, 0.9, 0.99, 0.2], dtype=torch.float64)
    weights = sampling_weights(probability, 0.05)
    raw = probability * (1 - probability)
    torch.testing.assert_close(weights, 0.95 * raw / raw.sum() + 0.05 / 4)
    assert float(weights.sum()) == pytest.approx(1.0, abs=1e-15)


def test_a_pool_of_sure_results_samples_evenly():
    weights = sampling_weights(torch.tensor([1.0, 0.0, 1.0], dtype=torch.float64), 0.05)
    assert torch.equal(weights, torch.full((3,), 1.0 / 3.0, dtype=torch.float64))


def test_the_ladder_names_random_rungs_and_scripted():
    rungs = (Rung(math.inf), Rung(2.0, 0.25), Rung(4.0), Rung(1.5), Rung(1.0))
    names = [entry.name for entry in ladder_entries(rungs)]
    assert names == ["random", "rung_T2_p0.25", "rung_T4", "rung_T1.5", "scripted"]
    assert ladder_name(Rung(2.5)) == "rung_T2.5"
    assert list(rungs) == sorted(rungs, key=strength_order)  # weakest first
    for rung in (Rung(math.inf), Rung(8.0), Rung(1.25), Rung(1.0), Rung(2.0, 0.1)):
        assert ladder_rung(ladder_name(rung)) == rung
    for other in (
        "rung_T0.5",
        "rung_T1",
        "rung_Tx",
        "rung_0.5",
        "live",
        "rung_T2_p0",
        "rung_T2_p1.5",
    ):
        assert ladder_rung(other) is None


def test_only_a_ladder_entry_has_a_rung():
    with pytest.raises(ValueError):
        Entry("snapshot_1", "checkpoint", rung=Rung(2.0))
    with pytest.raises(ValueError):
        Entry("rung", "ladder")


def test_the_averages_follow_their_definitions():
    generator = torch.Generator().manual_seed(0)
    history = [torch.randn(3, generator=generator) for _ in range(6)]
    uniform = {"weight": history[0].clone(), "count": torch.tensor(0)}
    exponential = {"weight": history[0].clone(), "count": torch.tensor(0)}
    for count, value in enumerate(history[1:], start=2):
        average_into(
            uniform, exponential, {"weight": value, "count": torch.tensor(count)}, count, 0.95
        )
    torch.testing.assert_close(uniform["weight"], torch.stack(history).mean(0))
    expected = history[0]
    for value in history[1:]:
        expected = 0.95 * expected + 0.05 * value
    torch.testing.assert_close(exponential["weight"], expected)
    assert int(uniform["count"]) == int(exponential["count"]) == 6
