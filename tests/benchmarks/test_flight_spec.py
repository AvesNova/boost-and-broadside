"""Tests for the flight-spec solver prototype (closed forms only, no simulation)."""

import math
from dataclasses import replace

import numpy as np
import pytest

from benchmarks.flight_spec import PARAMETERS, SPEC_QUANTITIES, evaluate, solve
from boost_and_broadside.config.defaults import SHIP_CONFIG
from boost_and_broadside.env.frontline import frontline_ship_config

BASE = frontline_ship_config(SHIP_CONFIG)
TODAY = evaluate(BASE)
SPEC = {name: TODAY[name] for name in SPEC_QUANTITIES}


def _perturbed(factor: float):
    return replace(BASE, **{p: getattr(BASE, p) * factor for p in PARAMETERS})


def test_today_spec_is_full_rank():
    assert solve(SPEC, BASE).rank == len(PARAMETERS)


@pytest.mark.parametrize("name", PARAMETERS)
def test_round_trip_recovers_every_constant_from_a_perturbed_base(name):
    solved = solve(SPEC, _perturbed(1.7)).config
    assert getattr(solved, name) == pytest.approx(getattr(BASE, name), rel=1e-9)


def test_solved_config_hits_new_targets_in_closed_form():
    spec = {**SPEC, "top_speed": 155.0, "sharp_radius": 50.0}
    del spec["recharge_time"]
    spec["unlimited_top_speed"] = TODAY["unlimited_top_speed"]
    achieved = evaluate(solve(spec, BASE).config)
    assert all(achieved[n] == pytest.approx(v, rel=1e-9) for n, v in spec.items())


def test_a_redundant_target_is_reported_as_one_dependency():
    spec = {**SPEC, "unlimited_top_speed": TODAY["unlimited_top_speed"]}
    assert len(solve(spec, BASE).dependencies) == 1


def test_a_reported_dependency_holds_for_any_constants():
    """Π h^e must be invariant: the model alone fixes it."""
    spec = {**SPEC, "unlimited_top_speed": TODAY["unlimited_top_speed"]}
    (dependency,) = solve(spec, BASE).dependencies
    rng = np.random.default_rng(0)
    other = replace(BASE, **{p: getattr(BASE, p) * rng.uniform(0.5, 2.0) for p in PARAMETERS})
    product = lambda h: sum(e * math.log(h[n]) for n, e in dependency.items())  # noqa: E731
    assert product(evaluate(other)) == pytest.approx(product(TODAY), abs=1e-2)


def test_inconsistent_redundant_targets_leave_residuals():
    spec = {**SPEC, "unlimited_top_speed": 2.0 * TODAY["unlimited_top_speed"]}
    assert solve(spec, BASE).residuals


def test_an_underdetermined_spec_leaves_untouched_constants_at_base():
    solved = solve({"sharp_radius": 50.0}, BASE).config
    changed = {p for p in PARAMETERS if getattr(solved, p) != getattr(BASE, p)}
    assert changed == {"sharp_turn_lift_coeff"}


def test_a_shared_free_direction_is_reported():
    spec = {n: v for n, v in SPEC.items() if n != "recharge_time"}
    assert len(solve(spec, BASE).free_directions) == 1


def test_reverse_thrust_stays_negative_from_a_positive_base():
    base = replace(BASE, reverse_thrust=abs(BASE.reverse_thrust))
    assert solve(SPEC, base).config.reverse_thrust < 0.0
