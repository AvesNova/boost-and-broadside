"""Tests for the pure calculation helpers in modes/ar_report.py.

The plotting/markdown layer has no automated coverage (visual output), but the numeric
helpers extracted from `_generate_report` carry the report's metric contract — toroidal
unwrapping, toroidal center of mass, error masking — and are unit-tested here so the
refactor that split them out of the monolithic function cannot silently change the math.
"""

import numpy as np

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.config.defaults import MODEL_CONFIG, REWARDS
from boost_and_broadside.modes import ar_report
from boost_and_broadside.modes.ar_report import (
    _calc_toroidal_euclidean,
    _clamp_alive_prob,
    _toroidal_center_of_mass,
    _unwrap_1d,
)


def test_canonical_ar_mode_owns_one_4v4_scenario(monkeypatch) -> None:
    captured = {}
    monkeypatch.setattr(ar_report, "run_ar_report_mode", lambda **kwargs: captured.update(kwargs))

    ar_report.run_canonical_ar_report_mode(
        "scripted",
        "random",
        12,
        ShipConfig(),
        REWARDS,
        MODEL_CONFIG,
        "cpu",
    )
    assert captured["env_config"].num_ships == 8
    assert captured["env_config"].max_episode_steps == 12
    assert captured["out_dir"] == "docs/ar_report/4v4"


def test_unwrap_1d_makes_boundary_crossing_continuous():
    # A ship drifting off the right edge (98 -> 2) reappears on the left after wrapping;
    # unwrapping must extend the trajectory past the edge rather than jump back across it.
    W = 100.0
    wrapped = np.array([95.0, 98.0, 2.0, 5.0])
    unwrapped = _unwrap_1d(wrapped, W)
    steps = np.diff(unwrapped)
    assert np.all(np.abs(steps) < W / 2)
    assert np.all(steps > 0)  # motion stays monotonic, no phantom reversal at the seam


def test_toroidal_center_of_mass_anchors_near_the_wrap_seam():
    # Two ships hugging opposite edges (x=1 and x=99) are adjacent on the torus; their CoM
    # belongs at the seam (~0/100), not the naive arithmetic mean of 50.
    W_x = W_y = 100.0
    points = np.array([[1.0, 50.0], [99.0, 50.0]])
    com_x, com_y = _toroidal_center_of_mass(points, W_x, W_y)
    wrapped_x = com_x % W_x
    assert min(wrapped_x, W_x - wrapped_x) < 5.0  # near the seam, far from 50
    assert abs(com_y - 50.0) < 1e-6


def test_toroidal_euclidean_uses_short_way_and_masks_dead_pairs():
    W_x = W_y = 100.0
    pos1 = np.array([[[1.0, 10.0], [50.0, 50.0]]])  # (1 step, 2 ships, 2)
    pos2 = np.array([[[99.0, 10.0], [50.0, 50.0]]])
    alive1 = np.array([[True, True]])
    alive2 = np.array([[True, False]])  # ship 1 dead in method 2
    dist = _calc_toroidal_euclidean(pos1, pos2, W_x, W_y, alive1, alive2)
    assert dist[0, 0] == 2.0  # short way across the seam, not 98
    assert np.isnan(dist[0, 1])  # dead pair masked out


def test_clamp_alive_prob_zeros_from_first_death_onward():
    # A ship that dies at step 2 must read prob 0 for every later step, even if the raw
    # predicted alive-prob "revives" it — deaths are permanent within a rollout.
    alive_prob = np.array([[0.9], [0.8], [0.3], [0.7], [0.6]])
    alive = np.array([[True], [True], [False], [False], [False]])
    clamped = _clamp_alive_prob(alive_prob, alive, plot_N=1)
    assert np.array_equal(clamped[:2, 0], np.array([0.9, 0.8]))
    assert np.all(clamped[2:, 0] == 0.0)
