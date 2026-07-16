"""Tests for persistent count-based Bradley-Terry league ratings."""

import math

import pytest
import torch

from boost_and_broadside.train.rl.rating import DRAW, LOSS, WIN, MatchCounts, solve_bt


def _add_expected_results(
    counts: MatchCounts,
    first_id: str,
    second_id: str,
    first_rating: float,
    second_rating: float,
    games: float,
) -> None:
    probability = 1.0 / (1.0 + 10.0 ** ((second_rating - first_rating) / 400.0))
    counts.add_pair(
        first_id,
        second_id,
        wins=games * probability,
        losses=games * (1.0 - probability),
    )


def test_solve_bt_recovers_known_ratings() -> None:
    expected = {"random": 0.0, "middle": 175.0, "strong": 425.0}
    counts = MatchCounts(expected)
    for first_id, second_id in (("random", "middle"), ("random", "strong"), ("middle", "strong")):
        _add_expected_results(
            counts,
            first_id,
            second_id,
            expected[first_id],
            expected[second_id],
            10_000.0,
        )

    ratings, _ = solve_bt(counts, "random")

    assert ratings == pytest.approx(expected, abs=0.1)


def test_biased_match_frequency_does_not_reverse_checkpoint_order() -> None:
    """Farming one opponent cannot drag its globally fitted rating below random."""
    expected = {"random": 0.0, "checkpoint": 100.0, "live": 200.0}
    counts = MatchCounts(expected)
    _add_expected_results(counts, "checkpoint", "random", 100.0, 0.0, 100.0)
    _add_expected_results(counts, "live", "checkpoint", 200.0, 100.0, 1_000.0)

    ratings, _ = solve_bt(counts, "random")

    assert ratings["checkpoint"] > ratings["random"]


def test_anchor_is_pinned_at_zero_with_nonzero_warm_start() -> None:
    counts = MatchCounts(("random", "live"))
    counts.add_pair("live", "random", wins=8.0, losses=2.0)

    ratings, errors = solve_bt(counts, "random", {"random": 900.0, "live": 1_000.0})

    assert ratings["random"] == 0.0
    assert errors["random"] == 0.0


def test_decay_multiplies_every_count_involving_agent_once() -> None:
    counts = MatchCounts(("live", "avg", "frozen"))
    counts.add_pair("live", "avg", wins=4.0)
    counts.add_pair("frozen", "live", losses=6.0)
    counts.add_pair("avg", "frozen", draws=8.0)

    counts.decay("live", 0.5)

    assert counts.tensor[counts.index("live"), counts.index("avg"), WIN] == 2.0
    assert counts.tensor[counts.index("frozen"), counts.index("live"), LOSS] == 3.0
    assert counts.tensor[counts.index("avg"), counts.index("frozen"), DRAW] == 8.0


def test_warm_start_converges_to_same_solution() -> None:
    counts = MatchCounts(("random", "a", "b"))
    counts.add_pair("a", "random", wins=7.0, losses=3.0)
    counts.add_pair("b", "a", wins=6.0, losses=4.0)
    cold, _ = solve_bt(counts, "random")

    warm, _ = solve_bt(counts, "random", {"random": 0.0, "a": -800.0, "b": 900.0})

    assert warm == pytest.approx(cold, abs=1e-6)


def test_ties_place_players_at_equal_ratings() -> None:
    counts = MatchCounts(("random", "other"))
    counts.add_pair("other", "random", draws=20.0)

    ratings, _ = solve_bt(counts, "random")

    assert ratings["other"] == pytest.approx(0.0, abs=1e-8)


def test_record_scatter_adds_directed_outcomes() -> None:
    counts = MatchCounts(("a", "b"), dtype=torch.float32)
    counts.record(
        torch.tensor([0, 0, 1, 1]),
        torch.tensor([1, 1, 0, 0]),
        torch.tensor([WIN, DRAW, LOSS, WIN]),
    )

    assert counts.tensor[0, 1].tolist() == [1.0, 0.0, 1.0]
    assert counts.tensor[1, 0].tolist() == [1.0, 1.0, 0.0]


def test_json_round_trip_preserves_fractional_counts(tmp_path) -> None:
    counts = MatchCounts(("random", "live"))
    counts.add_pair("live", "random", wins=2.5, losses=1.25, draws=0.5)
    path = tmp_path / "counts.json"
    counts.save_json(path)

    restored = MatchCounts.load_json(path)

    assert restored.agent_ids == counts.agent_ids
    assert torch.equal(restored.tensor, counts.tensor)


def test_unplayed_agent_has_infinite_standard_error() -> None:
    counts = MatchCounts(("random", "played", "new"))
    counts.add_pair("played", "random", wins=1.0, losses=1.0)

    _, errors = solve_bt(counts, "random")

    assert math.isinf(errors["new"])
