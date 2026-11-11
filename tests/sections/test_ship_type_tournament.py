"""The ship-type tournament's tables (bench/ship_type_tournament.py): its features from the
real-scale ship files, and the leave-one-out error its models are scored by."""

import pathlib
import sys

import pytest
import torch

from bnb.game import real_scale

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))  # `bench`
from bench import ship_type_tournament as tournament  # noqa: E402


def test_the_features_follow_the_ship_files():
    features = tournament.features("real-scale")
    assert list(features)[: len(real_scale.AIRCRAFT)] == [
        one.name for one in real_scale.AIRCRAFT
    ]  # the researched types first; player-made ones may follow
    p51 = features["P-51D"]
    assert p51["weight_of_fire"] == pytest.approx(360.0)
    assert p51["climb_rate"] == pytest.approx(15.2, rel=1e-3)  # its researched climb
    assert p51["shots_to_kill_p51"] == 5 and p51["kill_time_ratio"] == pytest.approx(1.0)


def test_the_leave_one_out_error_is_zero_for_an_exact_line_and_the_spread_for_the_mean():
    x = torch.arange(10, dtype=torch.float64)[:, None]
    assert tournament._loo_error(x, 3.0 * x[:, 0] + 2.0) == pytest.approx(0.0, abs=1e-9)
    y = torch.tensor([1.0, -1.0] * 5, dtype=torch.float64)
    # the mean of the other nine is off by 10/9
    assert tournament._loo_error(x[:, :0], y) == pytest.approx(10.0 / 9.0)
