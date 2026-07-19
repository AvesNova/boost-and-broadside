"""Tests for shared ELO rating math."""

import pytest
import torch

from boost_and_broadside.train.rl.elo_eval import expected_score, information_weights


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
