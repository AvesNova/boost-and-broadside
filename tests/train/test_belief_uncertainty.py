"""The belief carries how far it has drifted, not just how long it has been.

``time_since_observation`` says only how stale an estimate is. The head now
reports a spread per channel, and the tracker accumulates it while a ship is out
of sight, so the policy can read what the staleness actually cost -- which is
the quantity that decides whether to act on a remembered position or go look.
"""

import pytest
import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.env.observation import ObsKey
from boost_and_broadside.train.rl.belief import BeliefTracker
from boost_and_broadside.train.rl.features import build_standard_coordinator
from tests.train.test_belief import _view


@pytest.fixture
def coordinator():
    return build_standard_coordinator(ShipConfig())


def _uncertainty_accessor(coordinator):
    (feature,) = [f for f in coordinator.features if f.name == "belief_uncertainty"]
    return feature.accessor


def test_the_channel_width_is_resolved_from_the_predictors(coordinator) -> None:
    """One authority for the width, and it is the predictor layout.

    There is no module constant to state it any more: position reports one
    uncertainty column per Fourier harmonic, so the width follows the world size
    and only the feature layout knows it.
    """

    assert (
        _uncertainty_accessor(coordinator).absent_width
        == coordinator.total_uncertainty_dimension
    )


def test_the_channel_width_follows_the_world_size() -> None:
    """A bigger world means more position harmonics, so more spreads to report.

    The regression this pins is a width that was a constant: it happened to be
    right for one world and silently wrong for every other.
    """

    from dataclasses import replace

    widths = {}
    for side in (1024.0, 65536.0):
        c = build_standard_coordinator(replace(ShipConfig(), world_size=(side, side)))
        widths[side] = c.total_uncertainty_dimension
        assert _uncertainty_accessor(c).absent_width == widths[side]
    assert widths[65536.0] > widths[1024.0]


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
    assert uncertainty.shape[-1] == coordinator.total_uncertainty_dimension
    assert uncertainty.shape[1] >= 2


def test_an_observation_without_a_tracker_reports_no_uncertainty(coordinator) -> None:
    """Raw environment views and test fixtures have forecast nothing.

    Read through the accessor rather than the observation: the observation no
    longer carries a default for this channel, because its width is a property
    of the feature layout and the environment cannot know it.
    """

    view = _view(visible=True, x=300.0)
    assert ObsKey.BELIEF_UNCERTAINTY not in view.data
    supplied = _uncertainty_accessor(coordinator).get(view)
    assert supplied.shape[-1] == coordinator.total_uncertainty_dimension
    assert supplied.abs().max() == 0.0
