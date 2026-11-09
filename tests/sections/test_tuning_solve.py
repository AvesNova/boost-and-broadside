"""Live tuning of the handles (`api.Estimate`, `solve_handles(start=..., progress=...,
cancel=...)`, `bnb.play.tuning.Tuner`): the instant estimate is the anchor at the anchor
and close to the solve near it, a solve started from the estimate converges in a few
iterations and reports each, and the tuner applies the estimate, then the exact card."""

import time

import pytest
import torch

from bnb.contracts import GameConfig
from bnb.game import api
from bnb.game.handling.manoeuvres import fly
from bnb.game.handling.solver import ERROR_FLOOR
from bnb.play.loop import MatchLoop, MatchSetup, SideSetup
from bnb.play.tuning import Tuner, base_rules

pytestmark = pytest.mark.slow

RULES = api.rules_of(GameConfig(1, 1, 1000.0))


def _double(types: api.ShipTypes) -> api.ShipTypes:
    return api.ShipTypes(**{name: value.double() for name, value in vars(types).items()})


def _flown_error(types: api.ShipTypes, targets: api.Handles) -> torch.Tensor:
    """(types, 8) relative error of each flown handle the card flies against its target."""
    flown = fly(RULES, _double(types))
    errors = []
    for name in api.FLOWN_HANDLES:
        target = getattr(targets, name)
        scale = target.abs().clamp_min(ERROR_FLOOR[name])
        errors.append(((getattr(flown, name)[0] - target) / scale).abs())
    return torch.stack(errors, -1)


@pytest.fixture(scope="module")
def anchor():
    types = api.ship_types()
    return types, api.measure_handles(_double(types), RULES)


def test_the_estimate_is_the_anchor_at_the_anchor(anchor):
    types, handles = anchor
    estimate = api.Estimate(RULES, types, handles)
    for name, value in vars(estimate.card(handles)).items():
        torch.testing.assert_close(value, getattr(types, name), rtol=1e-6, atol=0.0)


@pytest.mark.parametrize(
    "name", ["cruise_speed", "climb_time", "nose_reversal_time", "dive_acceleration"]
)
def test_the_estimate_lands_within_two_percent_of_a_ten_percent_move(anchor, name):
    """Moving one handle by 10% (each way), the estimated card flies every flown handle
    within 2% of its target (measured: within 1.4%)."""
    types, handles = anchor
    estimate = api.Estimate(RULES, types, handles)
    for factor in (0.9, 1.1):
        targets = handles.replace(**{name: getattr(handles, name) * factor})
        error = _flown_error(estimate.card(targets), targets)
        assert error.max() < 0.02, (name, factor, error.max().item())


def test_a_solve_started_from_the_estimate_converges_and_reports(anchor):
    types, handles = anchor
    targets = handles.replace(cruise_speed=handles.cruise_speed * 1.1)
    start = api.Estimate(RULES, types, handles).card(targets)
    reports = []
    solution = api.solve_handles(targets, RULES, start=start, progress=reports.append)
    assert solution.converged.all() and solution.iterations <= 4
    assert [report.iteration for report in reports] == list(range(1, solution.iterations + 1))
    assert reports[-1].worst.max() < 5e-3
    assert reports[0].flown.shape == (2, len(api.FLOWN_HANDLES))


def test_a_cancelled_solve_stops_before_flying(anchor):
    types, handles = anchor
    started = time.perf_counter()
    solution = api.solve_handles(handles, RULES, start=types, cancel=lambda: True, strict=False)
    assert solution.iterations == 0 and not solution.converged.any()
    assert time.perf_counter() - started < 5.0


def test_the_tuner_applies_the_estimate_then_the_exact_card():
    """A live match with the scripted agent: after the handles are measured, a cruise
    change applies within seconds (the estimate) and becomes exact (the solve), with the
    agent's characteristics retuned; a bullet life the slots cannot hold is refused."""
    setup = MatchSetup(
        GameConfig(2, 4, 900.0, mode="frontline"), sides=(SideSetup("scripted_agent"), SideSetup())
    )
    loop = MatchLoop(setup, compiled=False)
    tuner = Tuner(loop, base_rules(setup.config, setup.lineup))

    def frames(done, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            tuner.update()
            loop.advance(1.0 / 30.0)
            if done():
                return True
            time.sleep(0.01)
        return False

    try:
        assert frames(lambda: tuner.targets is not None, 300.0), tuner.message
        traits = loop.sides[0].characteristics
        cruise = float(tuner.targets.cruise_speed[0]) * 1.1
        power = loop.types.engine_power.clone()
        tuner.set_handle("cruise_speed", 0, cruise)
        assert frames(lambda: not torch.equal(loop.types.engine_power, power), 10.0)
        assert frames(lambda: tuner.exact, 600.0), tuner.message
        assert tuner.flying["cruise_speed"][0] == pytest.approx(cruise, rel=5e-3)
        assert frames(lambda: loop.sides[0].characteristics is not traits, 300.0)
        tuner.set_setting("rules.bullet_life", 1.2)
        assert "bullet slots" in tuner.message
        assert loop.rules.bullet_moves == RULES.bullet_moves
    finally:
        tuner.close()
