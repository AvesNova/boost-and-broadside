"""Tests for shared Elo rating math."""

import pytest
import torch

from boost_and_broadside.train.rl.elo_eval import (
    EloEvaluator,
    LadderOpponent,
    expected_score,
    information_weights,
)


@pytest.mark.parametrize(
    ("rating", "opponent", "expected"),
    [(0.0, 0.0, 0.5), (400.0, 0.0, 10.0 / 11.0), (0.0, 400.0, 1.0 / 11.0)],
)
def test_expected_score(rating: float, opponent: float, expected: float) -> None:
    assert expected_score(rating, opponent) == pytest.approx(expected)


def test_information_weights_sum_to_one() -> None:
    weights = information_weights(1000.0, torch.tensor([700.0, 1100.0], dtype=torch.float64))
    assert weights.sum().item() == pytest.approx(1.0)


def test_information_weights_prefer_near_equal_matchups() -> None:
    weights = information_weights(1000.0, torch.tensor([0.0, 1000.0], dtype=torch.float64))
    assert weights[1].item() > weights[0].item()


def test_information_weights_equal_for_symmetric_gaps() -> None:
    weights = information_weights(1000.0, torch.tensor([800.0, 1200.0], dtype=torch.float64))
    assert weights[0].item() == pytest.approx(weights[1].item())


def test_information_weights_uniform_when_all_matchups_saturated() -> None:
    """When every matchup is a foregone conclusion, variance underflows and the
    weights fall back to uniform rather than dividing by ~zero."""
    weights = information_weights(1e7, torch.tensor([0.0, 100.0], dtype=torch.float64))
    assert weights[0].item() == pytest.approx(0.5)


class TestLadderCountFlush:
    """Readback of the floating-vs-anchor tally (live-elo-plan Phase 1).

    Exercised on a bare instance: the method reads only the count tensor and the
    anchor list, and building a real evaluator would drag in an environment and
    a device for what is a pure relabelling.
    """

    @staticmethod
    def evaluator(specs: list[LadderOpponent], rows: list[list[float]]) -> EloEvaluator:
        instance = object.__new__(EloEvaluator)
        instance._anchor_specs = specs
        instance._ladder_counts = torch.tensor(rows, dtype=torch.float64)
        return instance

    @staticmethod
    def anchor(label: str) -> LadderOpponent:
        return LadderOpponent(policy=None, elo=0.0, label=label)

    def test_rows_come_back_keyed_by_anchor_label(self) -> None:
        evaluator = self.evaluator(
            [self.anchor("random"), self.anchor("scripted")],
            [[3.0, 1.0, 2.0], [5.0, 4.0, 0.0]],
        )
        assert evaluator._flush_ladder_counts() == {
            "random": (3, 1, 2),
            "scripted": (5, 4, 0),
        }

    def test_anchors_without_games_are_omitted(self) -> None:
        evaluator = self.evaluator(
            [self.anchor("random"), self.anchor("scripted")],
            [[0.0, 0.0, 0.0], [5.0, 4.0, 0.0]],
        )
        assert set(evaluator._flush_ladder_counts()) == {"scripted"}

    def test_a_shared_label_sums_rather_than_overwrites(self) -> None:
        """Two anchors can briefly carry one label across a promotion."""
        evaluator = self.evaluator(
            [self.anchor("ckpt_10"), self.anchor("ckpt_10")],
            [[3.0, 1.0, 0.0], [2.0, 2.0, 1.0]],
        )
        assert evaluator._flush_ladder_counts() == {"ckpt_10": (5, 3, 1)}

    def test_the_tally_is_consumed_by_the_read(self) -> None:
        """Counts are handed to the accumulator, so leaving them would double."""
        evaluator = self.evaluator([self.anchor("random")], [[3.0, 1.0, 2.0]])
        evaluator._flush_ladder_counts()
        assert evaluator._flush_ladder_counts() == {}

    def test_trailing_rows_beyond_the_anchor_set_are_ignored(self) -> None:
        """The tensor is sized for the largest ladder promotion can reach."""
        evaluator = self.evaluator(
            [self.anchor("random")], [[3.0, 1.0, 0.0], [9.0, 9.0, 9.0]]
        )
        assert evaluator._flush_ladder_counts() == {"random": (3, 1, 0)}
