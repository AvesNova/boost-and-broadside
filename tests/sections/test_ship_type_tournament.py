"""The ship-type tournament's tables (bench/ship_type_tournament.py): the leave-one-out
error its models are scored by. (Its features were of the researched real-scale types,
which left the game on 2026-11-13.)"""

import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # `bench`
from bench import ship_type_tournament as tournament  # noqa: E402


def test_the_leave_one_out_error_is_zero_for_an_exact_line_and_the_spread_for_the_mean():
    x = torch.arange(10, dtype=torch.float64)[:, None]
    assert tournament._loo_error(x, 3.0 * x[:, 0] + 2.0) == pytest.approx(0.0, abs=1e-9)
    y = torch.tensor([1.0, -1.0] * 5, dtype=torch.float64)
    # the mean of the other nine is off by 10/9
    assert tournament._loo_error(x[:, :0], y) == pytest.approx(10.0 / 9.0)
