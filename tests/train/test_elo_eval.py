"""Tests for shared ELO rating math."""

import pytest
import torch

from boost_and_broadside.train.rl.elo_eval import expected_score, optimal_eval_ratio


@pytest.mark.parametrize(
    ("rating", "opponent", "expected"),
    [(0.0, 0.0, 0.5), (400.0, 0.0, 10.0 / 11.0), (0.0, 400.0, 1.0 / 11.0)],
)
def test_expected_score(rating: float, opponent: float, expected: float) -> None:
    assert expected_score(rating, opponent) == pytest.approx(expected)


def test_tensor_rating_math_matches_scalar() -> None:
    scalar = optimal_eval_ratio(250.0, 1000.0)
    tensor = optimal_eval_ratio(torch.tensor(250.0), 1000.0)
    assert tensor.item() == pytest.approx(scalar)
