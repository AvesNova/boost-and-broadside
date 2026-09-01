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


class TestAnchorProbabilityTable:
    """The ladder resolves its mixture against each env's assigned rung.

    Building one action tensor per rung and gathering afterwards computes a
    dozen candidate actions per environment and throws all but one away. Indexing
    the probability by the assignment first is equivalent in distribution and
    costs one blend however long the ladder gets.
    """

    @staticmethod
    def evaluator(specs: list[LadderOpponent]) -> EloEvaluator:
        instance = object.__new__(EloEvaluator)
        instance._anchor_specs = specs
        instance.device = torch.device("cpu")
        return instance

    def test_each_rung_contributes_its_own_probability(self) -> None:
        specs = [
            LadderOpponent(policy=None, elo=0.0, label="random"),
            LadderOpponent(policy=None, elo=500.0, label="semi", p_scripted=0.5),
            LadderOpponent(policy=None, elo=1000.0, label="scripted", p_scripted=1.0),
        ]
        table = self.evaluator(specs)._anchor_p_tensor()
        assert table.tolist() == pytest.approx([0.0, 0.5, 1.0])

    def test_the_random_agent_falls_out_of_the_same_expression(self) -> None:
        """p = 0 makes the blend always pick random, so it needs no branch."""
        table = self.evaluator(
            [LadderOpponent(policy=None, elo=0.0, label="random")]
        )._anchor_p_tensor()
        assert table.tolist() == pytest.approx([0.0])

    def test_a_policy_anchor_takes_a_placeholder(self) -> None:
        """Its entry is never read — its envs are overwritten by the policy."""
        specs = [
            LadderOpponent(policy=None, elo=1000.0, label="scripted", p_scripted=1.0),
            LadderOpponent(policy=object(), elo=1200.0, label="ckpt_1"),
        ]
        assert self.evaluator(specs)._anchor_p_tensor().tolist() == pytest.approx([1.0, 0.0])

    def test_indexing_by_assignment_matches_a_per_rung_blend(self) -> None:
        """Equivalence against the formulation this replaces."""
        torch.manual_seed(0)
        specs = [
            LadderOpponent(policy=None, elo=0.0, label="random"),
            LadderOpponent(policy=None, elo=300.0, label="a", p_scripted=0.3),
            LadderOpponent(policy=None, elo=900.0, label="b", p_scripted=0.9),
        ]
        table = self.evaluator(specs)._anchor_p_tensor()
        envs, ships, draws = 4096, 4, 40
        idx = torch.randint(0, len(specs), (envs,))

        collapsed = 0.0
        for _ in range(draws):
            follow = torch.rand(envs, ships) < table[idx].unsqueeze(1)
            collapsed += follow.float().mean(dim=1)
        per_rung = 0.0
        for _ in range(draws):
            stacked = torch.stack(
                [torch.rand(envs, ships) < float(s.p_scripted or 0.0) for s in specs]
            )
            per_rung += stacked.gather(
                0, idx.view(1, -1, 1).expand(1, envs, ships)
            ).squeeze(0).float().mean(dim=1)

        expected = table[idx] * draws
        assert collapsed.mean() == pytest.approx(expected.mean(), abs=0.4)
        assert collapsed.mean() == pytest.approx(per_rung.mean(), abs=0.4)
