"""Flight constants from handling targets (closed forms only, no simulation)."""

import math
from dataclasses import replace

import numpy as np
import pytest

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.config.defaults import SHIP_CONFIG, STARTING_HANDLING
from boost_and_broadside.config.handling import (
    PARAMETERS,
    HandlingSpec,
    correct,
    handling,
    infeasibilities,
    ship_config_from_handling,
    solve,
)

BASE = ShipConfig()
TODAY = handling(BASE)

# One full-rank set of sixteen: every constant pinned by exactly one target.
FULL_RANK = (
    "cruise_speed",
    "top_speed",
    "boost_accel",
    "reverse_decel",
    "recharge_time",
    "tank_speed",
    "sharp_radius",
    "normal_radius",
    "sharp_sustained_rate",
    "normal_sustained_rate",
    "corner_speed",
    "onset_time",
    "nose_slew_rate",
    "sharp_gun_offset",
    "fade_speed",
    "stall_speed",
)
SPEC = {name: TODAY[name] for name in FULL_RANK}


def _perturbed(factor: float) -> ShipConfig:
    return replace(BASE, **{p: getattr(BASE, p) * factor for p in PARAMETERS})


def test_the_full_rank_spec_pins_every_constant():
    assert solve(SPEC, BASE).rank == len(PARAMETERS)


@pytest.mark.parametrize("name", PARAMETERS)
def test_round_trip_recovers_every_constant_from_a_perturbed_base(name):
    solved = solve(SPEC, _perturbed(1.3)).config
    assert getattr(solved, name) == pytest.approx(getattr(BASE, name), rel=1e-9)


def test_solved_config_hits_new_targets_in_closed_form():
    spec = {**SPEC, "top_speed": 155.0, "corner_speed": 120.0}
    achieved = handling(solve(spec, BASE).config)
    assert all(achieved[n] == pytest.approx(v, rel=1e-9) for n, v in spec.items())


def test_a_redundant_target_is_reported_as_one_dependency():
    spec = {**SPEC, "max_turn_rate": TODAY["max_turn_rate"]}
    assert len(solve(spec, BASE).dependencies) == 1


def test_a_reported_dependency_holds_for_any_constants():
    """prod(h ** e) must be invariant: the model alone fixes it."""
    spec = {**SPEC, "max_turn_rate": TODAY["max_turn_rate"]}
    (dependency,) = solve(spec, BASE).dependencies
    rng = np.random.default_rng(0)
    other = replace(BASE, **{p: getattr(BASE, p) * rng.uniform(0.8, 1.25) for p in PARAMETERS})
    product = lambda h: sum(e * math.log(h[n]) for n, e in dependency.items())  # noqa: E731
    assert product(handling(other)) == pytest.approx(product(TODAY), abs=1e-2)


def test_inconsistent_redundant_targets_leave_residuals():
    spec = {**SPEC, "max_turn_rate": 2.0 * TODAY["max_turn_rate"]}
    assert solve(spec, BASE).residuals


def test_an_underdetermined_spec_leaves_untouched_constants_at_base():
    solved = solve({"sharp_radius": 50.0}, BASE).config
    changed = {p for p in PARAMETERS if getattr(solved, p) != getattr(BASE, p)}
    assert changed == {"max_lift_coeff"}


def test_reverse_thrust_stays_negative_from_a_positive_base():
    base = replace(BASE, reverse_thrust=abs(BASE.reverse_thrust))
    assert solve(SPEC, base).config.reverse_thrust < 0.0


def test_corner_speed_is_the_config_property():
    assert TODAY["corner_speed"] == pytest.approx(BASE.corner_speed, rel=1e-12)


def test_the_starting_spec_derives_the_live_flight_constants():
    derived = ship_config_from_handling(STARTING_HANDLING, BASE)
    assert all(getattr(derived, p) == getattr(SHIP_CONFIG, p) for p in PARAMETERS)


def test_the_starting_spec_meets_its_targets():
    achieved = handling(SHIP_CONFIG)
    targets = STARTING_HANDLING.targets()
    assert all(achieved[n] == pytest.approx(v, rel=0.01) for n, v in targets.items())


def test_a_corner_above_top_speed_is_rejected():
    spec = replace(STARTING_HANDLING, corner_speed=150.0)
    with pytest.raises(ValueError, match="corner"):
        ship_config_from_handling(spec, BASE)


def test_a_fade_above_the_coasting_sharp_turn_is_rejected():
    spec = replace(STARTING_HANDLING, fade_speed=60.0)
    with pytest.raises(ValueError, match="fade"):
        ship_config_from_handling(spec, BASE)


def test_inconsistent_pins_beyond_tolerance_are_rejected():
    spec = replace(STARTING_HANDLING, unlimited_top_speed=400.0)
    with pytest.raises(ValueError, match="inconsistent"):
        ship_config_from_handling(spec, BASE)


def test_the_live_config_has_no_infeasibilities():
    assert infeasibilities(SHIP_CONFIG) == []


def test_correction_meets_a_biased_forward_model():
    """A model that reads the sharp rate 5% high is corrected to the target."""

    def biased(config: ShipConfig) -> dict[str, float]:
        return {"sharp_sustained_rate": 1.05 * handling(config)["sharp_sustained_rate"]}

    targets = STARTING_HANDLING.targets()
    result = correct(targets, BASE, biased)
    measured = biased(result.solution.config)["sharp_sustained_rate"]
    assert measured == pytest.approx(targets["sharp_sustained_rate"], rel=0.005)


def test_an_empty_spec_returns_the_base():
    assert ship_config_from_handling(HandlingSpec(), BASE) == BASE
