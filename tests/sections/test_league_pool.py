"""The pool (league spec 1, 3 and 5): retention, the sampling weights and the averages."""

import math

import pytest
import torch

from bnb.league.pool import (
    IMUGI_ENTRY,
    RECENT_SNAPSHOTS,
    Entry,
    average_into,
    retained,
    sampling_weights,
)
from bnb.runtime.opponents import LeagueConfig, Rung, ladder_name, ladder_rung


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


def test_the_sampling_weight_is_one_minus_p_squared_with_no_floor():
    """By default (`LeagueConfig()`) an entry weighs (1 - p)^2 over every entry, no floor:
    an entry the live policy always beats is never drawn."""
    probability = torch.tensor([0.5, 0.9, 0.99, 0.2, 1.0], dtype=torch.float64)
    config = LeagueConfig()
    assert (config.sampling, config.floor) == ("hard", 0.0)
    weights = sampling_weights(probability, config.sampling, config.floor)
    raw = (1 - probability) ** 2
    torch.testing.assert_close(weights, raw / raw.sum())
    assert float(weights[-1]) == 0.0
    assert float(weights.sum()) == pytest.approx(1.0, abs=1e-12)


def test_the_variance_sampling_with_a_floor_is_p_times_one_minus_p_plus_the_floor():
    probability = torch.tensor([0.5, 0.9, 0.99, 0.2], dtype=torch.float64)
    weights = sampling_weights(probability, "variance", 0.05)
    raw = probability * (1 - probability)
    torch.testing.assert_close(weights, 0.95 * raw / raw.sum() + 0.05 / 4)
    assert float(weights.sum()) == pytest.approx(1.0, abs=1e-12)
    with pytest.raises(ValueError, match="sampling"):
        sampling_weights(probability, "uniform")
    with pytest.raises(ValueError, match="sampling"):
        LeagueConfig(sampling="uniform")


@pytest.mark.parametrize(
    "sampling, sure", [("hard", [1.0, 1.0, 1.0]), ("variance", [1.0, 0.0, 1.0])]
)
def test_a_pool_of_sure_results_samples_evenly(sampling, sure):
    """Every weight zero (the live policy beats every entry, or under "variance" every
    result is sure): the draw is even."""
    weights = sampling_weights(torch.tensor(sure, dtype=torch.float64), sampling)
    torch.testing.assert_close(weights, torch.full((3,), 1.0 / 3.0, dtype=torch.float64))


def test_the_ladder_names_random_rungs_and_scripted():
    rungs = (Rung(math.inf), Rung(2.0, 0.25), Rung(4.0), Rung(1.5), Rung(1.0))
    names = [ladder_name(rung) for rung in rungs]
    assert names == ["random", "rung_T2_p0.25", "rung_T4", "rung_T1.5", "scripted"]
    assert ladder_name(Rung(2.5)) == "rung_T2.5"
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


def test_the_pools_scripted_entry_is_the_full_strength_imugi():
    assert IMUGI_ENTRY.name == ladder_name(Rung(1.0)) == "scripted"
    assert not IMUGI_ENTRY.network
    for name, kind in (("random", "scripted"), ("rung_T2", "scripted"), ("scripted", "average")):
        with pytest.raises(ValueError):
            Entry(name, kind)
    with pytest.raises(ValueError):
        Entry("rung_T2", "ladder")  # no ladder kind


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
