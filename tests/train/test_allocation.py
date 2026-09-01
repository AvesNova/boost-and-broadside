"""Tests for c-optimal ladder game allocation (live-elo-plan Phase 4)."""

import numpy as np
import pytest

from boost_and_broadside.train.rl.allocation import (
    DEFAULT_FLOOR_FRACTION,
    allocation_weights,
    current_flow_scores,
)
from boost_and_broadside.train.rl.match_matrix import MatchMatrix

# A rung well above the floor, with the chain that carries its offset down to it.
RATINGS = {
    "random": 0.0,
    "semi_scripted_0p5": 500.0,
    "scripted": 1000.0,
    "ckpt_1": 1400.0,
    "floating": 1600.0,
}


def chain() -> MatchMatrix:
    matrix = MatchMatrix()
    matrix.record("scripted", "random", 900, 20, 80)
    matrix.record("scripted", "semi_scripted_0p5", 800, 150, 50)
    matrix.record("ckpt_1", "scripted", 700, 280, 20)
    matrix.record("floating", "ckpt_1", 600, 380, 20)
    matrix.record("floating", "scripted", 850, 140, 10)
    return matrix


def scores(candidates):
    return current_flow_scores(
        chain(), RATINGS, protagonist="floating", anchor="scripted", candidates=candidates
    )


class TestCurrentFlow:
    def test_a_dead_end_branch_scores_far_below_the_chain(self):
        """No current flows past the anchor, so the weak end falls out itself.

        This is what replaces a hard 'ignore ratings below 1000' threshold,
        which would have cut load-bearing links along with dead ones.
        """
        result = scores(["random", "ckpt_1"])
        assert result[1] > 20.0 * result[0]

    def test_the_saturated_anchor_still_wins_while_the_gap_is_small(self):
        """Local information ranks this edge last; global position ranks it first."""
        result = scores(["scripted", "semi_scripted_0p5"])
        assert result[0] > result[1]

    def test_the_anchor_edge_yields_to_the_chain_as_the_rung_pulls_ahead(self):
        """The rule answers 'when does the anchor saturate' as it happens."""
        share = []
        for elo in (1200.0, 1600.0, 2200.0):
            ratings = {**RATINGS, "floating": elo}
            result = current_flow_scores(
                chain(),
                ratings,
                protagonist="floating",
                anchor="scripted",
                candidates=["scripted", "ckpt_1"],
            )
            share.append(result[0] / result.sum())
        assert share[0] > share[1] > share[2]

    def test_an_unreachable_anchor_gives_no_opinion(self):
        """Early in a run nothing has been played; None means keep the old rule."""
        matrix = MatchMatrix()
        matrix.record("floating", "ckpt_1", 5, 5, 0)
        assert (
            current_flow_scores(
                matrix,
                RATINGS,
                protagonist="floating",
                anchor="scripted",
                candidates=["scripted"],
            )
            is None
        )

    def test_an_unrated_protagonist_gives_no_opinion(self):
        assert (
            current_flow_scores(
                chain(),
                {label: elo for label, elo in RATINGS.items() if label != "floating"},
                protagonist="floating",
                anchor="scripted",
                candidates=["scripted"],
            )
            is None
        )


class TestAllocationWeights:
    def test_it_returns_a_distribution(self):
        weights = allocation_weights(
            chain(),
            RATINGS,
            protagonist="floating",
            anchor="scripted",
            candidates=["random", "semi_scripted_0p5", "scripted", "ckpt_1"],
        )
        assert weights.sum() == pytest.approx(1.0)
        assert np.all(weights > 0.0)

    def test_no_candidate_is_ever_starved(self):
        """A starved edge can disconnect the graph, which is unrecoverable."""
        candidates = ["random", "semi_scripted_0p5", "scripted", "ckpt_1"]
        weights = allocation_weights(
            chain(), RATINGS, protagonist="floating", anchor="scripted",
            candidates=candidates,
        )
        assert weights.min() >= DEFAULT_FLOOR_FRACTION / len(candidates) - 1e-12

    def test_the_floor_does_not_swamp_the_ranking(self):
        candidates = ["random", "ckpt_1"]
        weights = allocation_weights(
            chain(), RATINGS, protagonist="floating", anchor="scripted",
            candidates=candidates,
        )
        assert weights[1] > weights[0]

    def test_a_full_floor_is_uniform(self):
        candidates = ["random", "scripted", "ckpt_1"]
        weights = allocation_weights(
            chain(), RATINGS, protagonist="floating", anchor="scripted",
            candidates=candidates, floor_fraction=0.999,
        )
        assert weights == pytest.approx(np.full(3, 1 / 3), abs=1e-3)

    def test_an_impossible_floor_is_rejected(self):
        with pytest.raises(ValueError, match="floor_fraction"):
            allocation_weights(
                chain(), RATINGS, protagonist="floating", anchor="scripted",
                candidates=["scripted"], floor_fraction=1.0,
            )

    def test_no_candidates_means_no_opinion(self):
        assert (
            allocation_weights(
                chain(), RATINGS, protagonist="floating", anchor="scripted", candidates=[]
            )
            is None
        )

    def test_an_unconnected_graph_means_no_opinion(self):
        assert (
            allocation_weights(
                MatchMatrix(), RATINGS, protagonist="floating", anchor="scripted",
                candidates=["scripted"],
            )
            is None
        )


class TestAgainstBaselines:
    """The measurement that decides whether the rule earns its complexity.

    Deterministic: each rule is given the same budget and spends it in expected
    proportion rather than by sampling, so the comparison is about the rules
    rather than about a seed. The score is the standard error of the quantity
    anyone actually asks for -- the rung's offset from the floor.
    """

    LABELS = ["random", "semi_scripted_0p5", "scripted", "ckpt_1", "floating"]

    @staticmethod
    def seed() -> MatchMatrix:
        matrix = MatchMatrix()
        matrix.record("scripted", "random", 90, 2, 8)
        matrix.record("scripted", "semi_scripted_0p5", 80, 15, 5)
        matrix.record("ckpt_1", "scripted", 70, 28, 2)
        matrix.record("floating", "ckpt_1", 60, 38, 2)
        matrix.record("floating", "scripted", 85, 14, 1)
        return matrix

    def spend(self, rule, batches=40, budget=200.0):
        """Run one rule to completion and return the final floor-offset error."""
        from boost_and_broadside.train.rl.bradley_terry import fisher_information
        from boost_and_broadside.train.rl.elo_diagnostics import effective_resistance

        matrix = self.seed()
        candidates = [label for label in self.LABELS if label != "floating"]
        for _ in range(batches):
            weights = rule(matrix, candidates)
            for label, share in zip(candidates, weights):
                games = budget * share
                if games <= 0.0:
                    continue
                probability = 1.0 / (
                    1.0 + 10.0 ** ((RATINGS[label] - RATINGS["floating"]) / 400.0)
                )
                matrix.record(
                    "floating", label, games * probability, games * (1.0 - probability), 0.0
                )
        ratings = np.array([RATINGS[label] for label in self.LABELS])
        laplacian = fisher_information(matrix.pair_games(self.LABELS), ratings)
        variance = effective_resistance(
            laplacian, self.LABELS.index("floating"), self.LABELS.index("scripted")
        )
        return float(np.sqrt(variance))

    @staticmethod
    def uniform(matrix, candidates):
        return np.full(len(candidates), 1.0 / len(candidates))

    @staticmethod
    def local_information(matrix, candidates):
        """The rule this replaces: p(1-p) with no notion of what is being asked."""
        gap = RATINGS["floating"] - np.array([RATINGS[c] for c in candidates])
        probability = 1.0 / (1.0 + 10.0 ** (-gap / 400.0))
        variance = probability * (1.0 - probability)
        return variance / variance.sum()

    @staticmethod
    def c_optimal(matrix, candidates):
        weights = allocation_weights(
            matrix, RATINGS, protagonist="floating", anchor="scripted",
            candidates=candidates,
        )
        return weights if weights is not None else np.full(
            len(candidates), 1.0 / len(candidates)
        )

    def test_it_beats_both_baselines_on_the_quantity_that_matters(self):
        c_optimal = self.spend(self.c_optimal)
        assert c_optimal < self.spend(self.uniform)
        assert c_optimal < self.spend(self.local_information)

    def test_the_local_rule_is_the_one_it_has_to_beat_by_a_margin(self):
        """Ranking the most valuable game last is the failure being fixed."""
        assert self.spend(self.local_information) > 1.3 * self.spend(self.c_optimal)

    def test_every_rule_improves_with_budget(self):
        """Guards the harness: a rule that ignored its games would tie itself."""
        assert self.spend(self.c_optimal, batches=80) < self.spend(
            self.c_optimal, batches=10
        )
