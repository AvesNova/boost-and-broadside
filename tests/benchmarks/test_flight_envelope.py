"""CPU tests for the flight-envelope harness.

The envelope numbers in ``docs/internal/flight-envelope-*.json`` are only as
good as the probe that measured them, so each test checks the probe against a
closed form of the flight model.
"""

import math

import pytest
import torch

from benchmarks.flight_envelope import (
    FlightProbe,
    constant,
    first_tick,
    fly,
    measure_bullets,
    measure_em_grid,
    measure_nose,
)
from boost_and_broadside.config.defaults import SHIP_CONFIG
from boost_and_broadside.constants import PowerActions, TurnActions
from boost_and_broadside.env.frontline import frontline_ship_config

CPU = torch.device("cpu")
LIVE = frontline_ship_config(SHIP_CONFIG)


def _coast_turn(turn: TurnActions, index: float, start_speed: float, seconds: float):
    probe = FlightProbe(LIVE, torch.tensor([index]), CPU)
    probe.set_flight(torch.tensor([start_speed]), torch.tensor([LIVE.max_power]))
    trace = fly(probe, round(seconds / LIVE.dt), constant([(PowerActions.COAST, turn)], CPU))
    return probe, trace


@pytest.mark.parametrize("index", [0.5, 1.0, 2.0])
def test_coast_cruise_is_the_thrust_drag_balance_at_every_index(index):
    """Coast thrust balances quadratic drag at u = sqrt(base_thrust / drag) in proper speed."""
    _, trace = _coast_turn(TurnActions.GO_STRAIGHT, index, 50.0, 90.0)
    expected = math.sqrt(LIVE.base_thrust / LIVE.zero_slip_drag_coeff)
    assert trace.proper_speed[-1, 0].item() == pytest.approx(expected, rel=0.01)


def test_the_probe_holds_the_ship_inside_the_uniform_field():
    probe, _ = _coast_turn(TurnActions.SHARP_LEFT, 0.5, 100.0, 5.0)
    assert probe.state.ship_local_index[0, 0].item() == pytest.approx(0.5, abs=1e-5)


@pytest.mark.parametrize(
    ("turn", "lift"),
    [
        (TurnActions.TURN_LEFT, LIVE.normal_lift_coeff),
        (TurnActions.SHARP_LEFT, LIVE.max_lift_coeff),
    ],
)
def test_sustained_radius_is_close_to_the_inverse_lift_coefficient(turn, lift):
    """Lift alone gives radius 1/lift; the slipped thrust tightens it by a few percent."""
    _, trace = _coast_turn(turn, 1.0, 100.0, 20.0)
    window = round(2.0 / LIVE.dt)
    distance = (trace.position[-window:] - trace.position[-window - 1 : -1]).abs().sum()
    turned = math.radians(abs(trace.heading[-1, 0] - trace.heading[-1 - window, 0]))
    assert (distance.item() / turned) == pytest.approx(1.0 / lift, rel=0.07)


def test_settled_sharp_turn_rate_at_corner_matches_lift_times_speed():
    rows = measure_em_grid(LIVE, CPU, indices=(1.0,), speeds=(LIVE.corner_speed,))
    row = next(r for r in rows if r["power"] == "COAST" and r["turn"] == "SHARP_LEFT")
    expected = -math.degrees(LIVE.max_lift_coeff * LIVE.corner_speed)
    assert row["turn_rate_deg_s"] == pytest.approx(expected, rel=0.05)


def test_world_turn_rate_is_reciprocal_in_the_index_at_fixed_proper_speed():
    rows = measure_em_grid(LIVE, CPU, indices=(1.0, 2.0), speeds=(100.0,))
    rate = {
        r["index"]: r["turn_rate_deg_s"]
        for r in rows
        if r["power"] == "COAST" and r["turn"] == "SHARP_LEFT"
    }
    assert rate[2.0] == pytest.approx(rate[1.0] / 2.0, rel=0.02)


def test_straight_flight_does_not_turn_the_path():
    _, trace = _coast_turn(TurnActions.GO_STRAIGHT, 1.0, 150.0, 2.0)
    assert trace.heading.abs().max().item() == pytest.approx(0.0, abs=1e-4)


def test_bullet_range_matches_exact_quadratic_drag():
    """A forward shot from rest travels ln(1 + c v0 T) / c under quadratic drag."""
    (row,) = measure_bullets(LIVE, CPU, ship_speeds=(0.0,))
    c, v0 = LIVE.bullet_drag_coeff, row["muzzle_speed_world"]
    expected = math.log1p(c * v0 * LIVE.bullet_lifetime) / c
    assert row["range_world_px"] == pytest.approx(expected, rel=0.03)


def test_first_tick_reports_none_for_a_column_that_never_crosses():
    mask = torch.tensor([[False, False], [False, True], [False, True]])  # (T+1, B)
    assert first_tick(mask) == [None, 1]


def test_above_corner_the_settled_turn_is_g_limited():
    rows = measure_em_grid(LIVE, CPU, indices=(1.0,), speeds=(200.0,))
    row = next(r for r in rows if r["power"] == "COAST" and r["turn"] == "SHARP_LEFT")
    allowance = LIVE.base_thrust * math.sin(LIVE.stall_angle)
    assert row["lateral_accel"] == pytest.approx(LIVE.max_lateral_accel, abs=allowance + 1.0)


def test_the_nose_onset_is_reported_in_the_design_window():
    rows = measure_nose(LIVE, CPU, speeds=(60.0,))
    assert all(0.13 <= r["onset_90_seconds"] <= 0.20 for r in rows)
