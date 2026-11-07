"""The state-delta analysis scripts (`bench.state_deltas_measure`, `bench.state_deltas_codes`)
on synthetic inputs: the pair mask, the kinematic baselines on an exact turn, and the codes'
round trips."""

import math

import numpy as np

from bench import state_deltas_codes as codes
from bench import state_deltas_measure as measure
from bnb.contracts.obs import SHIP_COLUMNS

DT = 1.0 / 30.0


def test_valid_pairs_need_life_throughout_and_no_reset():
    alive = np.ones((6, 1, 2), bool)
    alive[3, 0, 1] = False  # ship 1 dead at tick 3 only
    done = np.zeros((6, 1), bool)
    done[1, 0] = True  # the environment reset after tick 1
    one = measure.valid_pairs(alive, done, 1)[:, 0]
    assert one[:, 0].tolist() == [True, False, True, True, True]
    assert one[:, 1].tolist() == [True, False, False, False, True]
    two = measure.valid_pairs(alive, done, 2)[:, 0]
    assert two[:, 0].tolist() == [False, False, True, True]
    assert two[:, 1].tolist() == [False, False, False, False]


def _turning_segment(path, ticks=80, speed=60.0, omega=1.5):
    """One ship flying an exact circle at constant speed and nose rate, nose along the path."""
    rows = np.zeros((ticks, 1, 1, len(SHIP_COLUMNS)), np.float32)
    column = {name: i for i, name in enumerate(SHIP_COLUMNS)}
    t = np.arange(ticks) * DT
    heading = 0.3 + omega * t
    radius = speed / omega
    values = dict(
        x=100.0 + radius * np.sin(heading),
        y=-50.0 - radius * np.cos(heading),
        heading_cos=np.cos(heading),
        heading_sin=np.sin(heading),
        velocity_x=speed * np.cos(heading),
        velocity_y=speed * np.sin(heading),
        nose_rate=np.full(ticks, omega),
        alive=np.ones(ticks),
        shield=np.full(ticks, 100.0),
        energy=np.full(ticks, 5000.0),
    )
    for name, value in values.items():
        rows[:, 0, 0, column[name]] = value
    flags = np.zeros((ticks, 1, 1), bool)
    np.savez(
        path,
        rows=rows,
        power=flags.astype(np.int8),
        stick=flags.astype(np.int8),
        hit=flags,
        boundary=flags,
        zone=flags,
        done=np.zeros((ticks, 1), bool),
    )


def test_arc_and_heading_baselines_are_exact_on_a_constant_turn(tmp_path):
    path = tmp_path / "segment_0.npz"
    _turning_segment(path)
    series, _, situations = measure.segment_series(path, list(SHIP_COLUMNS), DT)
    for k in (1, 8, 32):
        arc = series[f"position/arc/{k}/norm"]
        straight = series[f"position/dead_reckoning/{k}/norm"]
        assert arc.max() < 2e-3 * k  # float32 positions of ~100 m
        assert straight.min() > 10 * arc.max()
        assert np.abs(series[f"heading/dead_reckoning/{k}/angle"]).max() < 1e-5
        rotated = series[f"velocity/rotated/{k}/norm"]
        assert rotated.max() < 1e-3 < series[f"velocity/persistence/{k}/norm"].min()
    assert situations["boost"].sum() == 0


def test_two_hot_decodes_exactly_inside_and_clips_outside():
    centres = codes.symlog_grid(-10.0, 10.0, 21, 0.5)
    assert centres[10] == 0.0 and np.allclose(centres, -centres[::-1])
    z = np.array([-12.0, -3.3, 0.0, 0.01, 7.7, 12.0])
    lower, fraction, clipped = codes.two_hot(centres, z)
    decoded = (1 - fraction) * centres[lower] + fraction * centres[lower + 1]
    assert np.allclose(decoded, np.clip(z, -10, 10))
    assert codes.grid_report(centres, z)["tail"] == 2


def test_hierarchy_resolves_to_its_finest_level():
    z = np.random.default_rng(0).uniform(-1.0, 1.0, 10_000)
    result = codes.hierarchy_1d(z, 2.0, 3, 7)
    assert math.isclose(result["finest"], 2.0 / 3**7)
    assert np.abs(z - result["nearest"]).max() <= 0.5 * result["finest"] + 1e-12
    assert result["H"] <= 7 * math.log(3) + 1e-9


def test_quantile_grid_keeps_an_exact_zero_centre():
    z = np.concatenate([np.zeros(900), np.linspace(-1, 1, 100)])
    grid = codes.quantile_grid(z, 21)
    assert 0.0 in grid and np.all(np.diff(grid) > 0)
