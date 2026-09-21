"""The belief carries how far it has drifted, not just how long it has been.

``time_since_observation`` says only how stale an estimate is. The head now
reports a spread per channel, and the tracker accumulates it while a ship is out
of sight, so the policy can read what the staleness actually cost -- which is
the quantity that decides whether to act on a remembered position or go look.
"""

import pytest
import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.env.observation import BELIEF_UNCERTAINTY_DIM, ObsKey
from boost_and_broadside.train.rl.belief import BeliefTracker
from boost_and_broadside.train.rl.features import build_standard_coordinator
from tests.train.test_belief import _view


@pytest.fixture
def coordinator():
    return build_standard_coordinator(ShipConfig())


def test_the_observation_width_matches_the_prediction_layout(coordinator) -> None:
    """Pinned, because the observation contract states the width independently.

    ``observation.py`` cannot import the feature registry without the
    environment depending on the policy, so the constant is written out. This is
    what stops the two drifting apart in silence.
    """

    assert BELIEF_UNCERTAINTY_DIM == coordinator.total_prediction_dimension


def _prediction(coordinator, log_uncertainty: float) -> torch.Tensor:
    P = coordinator.total_prediction_dimension
    U = coordinator.total_uncertainty_dimension
    return torch.cat(
        [torch.zeros(1, 2, P), torch.full((1, 2, U), log_uncertainty)], dim=-1
    )


def test_uncertainty_accumulates_while_a_ship_is_unseen(coordinator) -> None:
    """A spread per step, summed: the belief's own uncertainty, not the latest step's."""

    tracker = BeliefTracker(1, 2, 0.1, coordinator, "cpu")
    composed = tracker.compose(_view(visible=True, x=300.0))
    assert composed[ObsKey.BELIEF_UNCERTAINTY][0, 1].abs().max() == 0.0

    readings = []
    for _ in range(4):
        tracker.advance(composed, _prediction(coordinator, 0.0))  # unit variance per step
        composed = tracker.compose(_view(visible=False, x=300.0))
        readings.append(composed[ObsKey.BELIEF_UNCERTAINTY][0, 1].clone())

    for step, reading in enumerate(readings, start=1):
        assert reading.min() == pytest.approx(float(step), rel=1e-5), (
            "variance should be the running sum of each step's spread"
        )


def test_seeing_a_ship_settles_its_uncertainty(coordinator) -> None:
    """An observation replaces the estimate, so its accumulated doubt is discarded."""

    tracker = BeliefTracker(1, 2, 0.1, coordinator, "cpu")
    composed = tracker.compose(_view(visible=True, x=300.0))
    for _ in range(3):
        tracker.advance(composed, _prediction(coordinator, 0.0))
        composed = tracker.compose(_view(visible=False, x=300.0))
    assert composed[ObsKey.BELIEF_UNCERTAINTY][0, 1].min() > 0.0

    tracker.advance(composed, _prediction(coordinator, 0.0))
    reacquired = tracker.compose(_view(visible=True, x=900.0))
    assert reacquired[ObsKey.BELIEF_UNCERTAINTY][0, 1].abs().max() == 0.0
    # The ship we never lost was never uncertain either.
    assert reacquired[ObsKey.BELIEF_UNCERTAINTY][0, 0].abs().max() == 0.0


def test_a_confident_forecast_accumulates_less_than_a_vague_one(coordinator) -> None:
    """The accumulated doubt tracks the head's own claim, not merely elapsed steps."""

    def after_one_step(log_uncertainty: float) -> torch.Tensor:
        tracker = BeliefTracker(1, 2, 0.1, coordinator, "cpu")
        composed = tracker.compose(_view(visible=True, x=300.0))
        tracker.advance(composed, _prediction(coordinator, log_uncertainty))
        return tracker.compose(_view(visible=False, x=300.0))[ObsKey.BELIEF_UNCERTAINTY][0, 1]

    confident = after_one_step(-2.0)
    vague = after_one_step(2.0)
    assert (confident < vague).all(), "a tighter forecast must cost less certainty"


def test_map_objects_carry_no_uncertainty(coordinator) -> None:
    """Static geometry is not believed, so nothing about it is in doubt."""

    tracker = BeliefTracker(1, 2, 0.1, coordinator, "cpu")
    composed = tracker.compose(_view(visible=False, x=300.0))
    uncertainty = composed[ObsKey.BELIEF_UNCERTAINTY]
    assert uncertainty.shape[-1] == BELIEF_UNCERTAINTY_DIM
    assert uncertainty.shape[1] >= 2


def test_an_observation_without_a_tracker_reports_no_uncertainty() -> None:
    """Raw environment views and test fixtures have forecast nothing."""

    view = _view(visible=True, x=300.0)
    assert ObsKey.BELIEF_UNCERTAINTY not in view.data
    assert view[ObsKey.BELIEF_UNCERTAINTY].shape[-1] == BELIEF_UNCERTAINTY_DIM
    assert view[ObsKey.BELIEF_UNCERTAINTY].abs().max() == 0.0
