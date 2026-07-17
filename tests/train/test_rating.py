"""Tests for persistent count-based Bradley-Terry league ratings."""

import math

import pytest
import torch

from boost_and_broadside.train.rl.rating import (
    DRAW,
    LOSS,
    WIN,
    MatchCounts,
    information_pair_weights,
    solve_bt,
    solve_league,
    solve_probe,
)


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


def test_gameless_agent_keeps_warm_start_rating_exactly() -> None:
    counts = MatchCounts(("live", "random", "scripted"))
    counts.add_pair("live", "random", wins=8.0, losses=2.0)

    ratings, _ = solve_bt(counts, "random", {"scripted": 1_000.0})

    assert ratings["scripted"] == 1_000.0
    assert ratings["random"] == 0.0
    assert ratings["live"] > 0.0


def test_disconnected_component_holds_warm_start() -> None:
    # Run-664 early state: live has only losses vs scripted; the anchor
    # (random) has no games at all. The component's offset is unidentified,
    # so both members must keep their warm start with infinite error.
    counts = MatchCounts(("live", "random", "scripted"))
    counts.add_pair("live", "scripted", losses=500.0)
    warm = {"live": 123.0, "scripted": 1000.0}

    ratings, errors = solve_bt(counts, "random", warm)

    assert ratings["live"] == 123.0
    assert ratings["scripted"] == 1000.0
    assert ratings["random"] == 0.0
    assert errors["live"] == math.inf
    assert errors["scripted"] == math.inf


def test_proportional_prior_caps_saturated_gap_independent_of_volume() -> None:
    def saturated_gap(games: float, prior_frac: float) -> float:
        counts = MatchCounts(("a", "anchor"))
        counts.add_pair("a", "anchor", wins=games)
        ratings, _ = solve_bt(counts, "anchor", prior_frac=prior_frac)
        return ratings["a"]

    # Without the proportional prior the gap grows unboundedly with volume
    # (400*log10(2n+1)); with it, a 100x volume increase buys < 100 points and
    # the gap stays near the 400*log10(2/frac) ~ 800 asymptote.
    unregularized_growth = saturated_gap(10_000.0, 0.0) - saturated_gap(100.0, 0.0)
    capped_growth = saturated_gap(10_000.0, 0.02) - saturated_gap(100.0, 0.02)
    assert unregularized_growth > 700.0
    assert capped_growth < 100.0
    assert saturated_gap(10_000.0, 0.02) < 850.0


def test_select_agents_extracts_ordered_submatrix() -> None:
    counts = MatchCounts(("a", "b", "c"))
    counts.add_pair("a", "b", wins=3.0)
    counts.add_pair("a", "c", wins=7.0)

    sub = counts.select_agents(["a", "c"])

    assert sub.agent_ids == ["a", "c"]
    assert float(sub.tensor[0, 1, WIN]) == 7.0
    assert float(sub.tensor.sum()) == 7.0


def test_solve_probe_recovers_known_rating() -> None:
    counts = MatchCounts(("ckpt_0", "scripted"))
    counts.add_pair("scripted", "ckpt_0", wins=425.0, losses=75.0)  # 85%

    rating, error = solve_probe(counts, {"ckpt_0": 0.0}, "scripted")

    assert rating == pytest.approx(400.0 * math.log10(0.85 / 0.15), abs=15.0)
    assert 0.0 < error < 50.0


def test_solve_probe_without_pool_games_keeps_warm_start() -> None:
    counts = MatchCounts(("ckpt_0", "scripted"))

    rating, error = solve_probe(counts, {"ckpt_0": 0.0}, "scripted", warm_start=1000.0)

    assert rating == 1000.0
    assert error == math.inf


def test_solve_league_probe_games_never_move_pool_ratings() -> None:
    def build(with_stomps: bool) -> MatchCounts:
        counts = MatchCounts(("live", "ckpt_0", "scripted", "random"))
        counts.add_pair("live", "ckpt_0", wins=600.0, losses=400.0)
        counts.add_pair("scripted", "ckpt_0", wins=425.0, losses=75.0)
        if with_stomps:
            counts.add_pair("scripted", "live", wins=5000.0)
        return counts

    solve = lambda counts: solve_league(  # noqa: E731
        counts, "ckpt_0", ["live", "ckpt_0"], ["scripted", "random"], {"random": -800.0}
    )
    baseline, baseline_errors = solve(build(with_stomps=False))
    stomped, stomped_errors = solve(build(with_stomps=True))

    # 5000 scripted stomps move scripted's probe rating but no pool member.
    assert stomped["live"] == pytest.approx(baseline["live"])
    assert stomped["ckpt_0"] == 0.0
    assert stomped["scripted"] > baseline["scripted"]
    # Game-less probe keeps its warm start with infinite error.
    assert stomped["random"] == -800.0
    assert stomped_errors["random"] == math.inf
    # Pool member measured from pool games only: 60% over the anchor.
    assert baseline["live"] == pytest.approx(400.0 * math.log10(0.6 / 0.4), abs=15.0)
    assert math.isfinite(stomped_errors["live"])


def test_information_weights_stay_proportional_with_infinite_se() -> None:
    ratings = {"a": 0.0, "b": 0.0, "c": 0.0}
    errors = {"a": math.inf, "b": 100.0, "c": 100.0}

    weights = information_pair_weights(ratings, errors, [("a", "b"), ("b", "c")])

    # The inf-SE pair is capped, not winner-take-all.
    assert float(weights[0]) > 0.0
    assert float(weights[1]) > 0.0
    assert float(weights[0]) > float(weights[1])
    assert float(weights.sum()) == pytest.approx(1.0)
