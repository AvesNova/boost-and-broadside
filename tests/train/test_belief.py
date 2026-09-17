"""Contracts for recursive hidden-enemy point estimates."""

import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.env.observation import ObjectType, ObsKey, YemongObservation
from boost_and_broadside.train.rl.belief import BELIEF_TARGET_LIMIT, BeliefTracker
from boost_and_broadside.train.rl.features import build_standard_coordinator


def _view(*, visible: bool, x: float = 300.0) -> YemongObservation:
    b, n = 1, 2
    seen = torch.tensor([[True, visible]])
    data = {
        ObsKey.POS: torch.tensor([[[100.0, 200.0], [x, 400.0]]]),
        ObsKey.VEL: torch.tensor([[[1.0, 2.0], [3.0, 4.0]]]),
        ObsKey.ATT: torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]),
        ObsKey.ANG_VEL: torch.tensor([[[0.1], [0.2]]]),
        ObsKey.HEALTH: torch.tensor([[[80.0], [70.0]]]),
        ObsKey.POWER: torch.tensor([[[60.0], [50.0]]]),
        ObsKey.COOLDOWN: torch.tensor([[[0.2], [0.3]]]),
        ObsKey.TEAM_ID: torch.tensor([[0, 1]], dtype=torch.int32),
        ObsKey.ALIVE: torch.ones((b, n), dtype=torch.bool),
        ObsKey.VISIBLE: seen.clone(),
        ObsKey.BELIEF_VALID: seen.clone(),
        ObsKey.TIME_SINCE_OBSERVATION: torch.zeros((b, n, 1)),
        ObsKey.OBJECT_TYPE: torch.full((b, n), int(ObjectType.SHIP), dtype=torch.int32),
        ObsKey.ZONE_ROLE: torch.full((b, n), 5, dtype=torch.int32),
        ObsKey.PREVIOUS_ACTION: torch.tensor([[[1, 2, 1], [2, 3, 1]]]),
        ObsKey.RADIUS: torch.full((b, n, 1), 16.0),
        ObsKey.LOCAL_LOG_INDEX: torch.tensor([[[0.1], [0.2]]]),
        ObsKey.LOCAL_INDEX_GRADIENT: torch.tensor([[[0.3, 0.4], [0.5, 0.6]]]),
    }
    if not visible:
        for key, value in data.items():
            if key in {ObsKey.VISIBLE, ObsKey.BELIEF_VALID, ObsKey.TIME_SINCE_OBSERVATION}:
                continue
            if value.dim() == 2:
                value[:, 1] = 0
            else:
                value[:, 1] = 0
    return YemongObservation(data=data)


def test_never_seen_enemy_stays_absent() -> None:
    coordinator = build_standard_coordinator(ShipConfig())
    tracker = BeliefTracker(1, 2, 0.1, coordinator, "cpu")

    composed = tracker.compose(_view(visible=False))

    assert not composed[ObsKey.BELIEF_VALID][0, 1]
    assert not composed[ObsKey.VISIBLE][0, 1]
    assert composed[ObsKey.POS][0, 1].equal(torch.zeros(2))
    assert composed[ObsKey.TIME_SINCE_OBSERVATION][0, 1, 0] == 0


def test_seen_then_hidden_uses_recursive_prediction_and_age() -> None:
    coordinator = build_standard_coordinator(ShipConfig())
    tracker = BeliefTracker(1, 2, 0.1, coordinator, "cpu")
    visible = tracker.compose(_view(visible=True, x=300.0))
    tracker.advance(
        visible,
        torch.zeros((1, 2, coordinator.total_prediction_dimension)),
    )

    hidden = tracker.compose(_view(visible=False, x=9999.0))

    assert hidden[ObsKey.BELIEF_VALID][0, 1]
    assert not hidden[ObsKey.VISIBLE][0, 1]
    assert torch.allclose(hidden[ObsKey.POS][0, 1], torch.tensor([300.0, 400.0]))
    assert hidden[ObsKey.TEAM_ID][0, 1] == 1
    assert hidden[ObsKey.TIME_SINCE_OBSERVATION][0, 1, 0] == 0.1
    assert hidden[ObsKey.PREVIOUS_ACTION][0, 1].equal(torch.zeros(3, dtype=torch.long))
    assert hidden[ObsKey.LOCAL_INDEX_GRADIENT][0, 1].equal(torch.zeros(2))

    tracker.advance(
        hidden,
        torch.zeros((1, 2, coordinator.total_prediction_dimension)),
    )
    hidden_again = tracker.compose(_view(visible=False))
    assert hidden_again[ObsKey.TIME_SINCE_OBSERVATION][0, 1, 0] == 0.2


def test_reacquisition_overwrites_prediction_and_reset_forgets() -> None:
    coordinator = build_standard_coordinator(ShipConfig())
    tracker = BeliefTracker(1, 2, 0.1, coordinator, "cpu")
    first = tracker.compose(_view(visible=True, x=300.0))
    tracker.advance(first, torch.zeros((1, 2, coordinator.total_prediction_dimension)))
    tracker.compose(_view(visible=False))

    reacquired = tracker.compose(_view(visible=True, x=777.0))
    assert reacquired[ObsKey.POS][0, 1, 0] == 777.0
    assert reacquired[ObsKey.TIME_SINCE_OBSERVATION][0, 1, 0] == 0.0

    tracker.reset(torch.tensor([True]))
    forgotten = tracker.compose(_view(visible=False))
    assert not forgotten[ObsKey.BELIEF_VALID][0, 1]


def test_a_runaway_forecast_cannot_reach_infinity() -> None:
    """The belief is an unbounded autoregressive rollout, so it needs a floor of
    numerical safety independent of whether the next-state head is well behaved.

    Run 734 died here: a ship hidden long enough accumulated velocity error
    until the stored target overflowed, and the resulting non-finite logits
    asserted inside ``torch.multinomial``. The guard that existed used
    ``nan_to_num``'s defaults, which map ``+inf`` to float32's maximum -- and
    targets are symlog, so that decoded straight back to infinity on the next
    ``compose``. What matters is therefore not merely that the stored target is
    finite, but that it survives the exponential inverse.
    """

    coordinator = build_standard_coordinator(ShipConfig())
    tracker = BeliefTracker(1, 2, 0.1, coordinator, "cpu")
    visible = tracker.compose(_view(visible=True, x=300.0))

    prediction = torch.full((1, 2, coordinator.total_prediction_dimension), float("inf"))
    prediction[0, 0, 0] = float("nan")
    prediction[0, 1, 0] = -float("inf")
    tracker.advance(visible, prediction)

    stored = tracker.predicted_targets
    assert torch.isfinite(stored).all()
    assert stored.abs().max() <= BELIEF_TARGET_LIMIT
    assert tracker.clamp_events > 0

    # The property the old guard lacked: finite after decoding out of symlog.
    for raw in coordinator.decode_targets(stored).values():
        assert torch.isfinite(raw).all()
        # And finite again after the squarings the physics applies downstream.
        assert torch.isfinite(raw.double().square()).all()


def test_the_guard_does_not_bind_on_ordinary_predictions() -> None:
    """A guard that fires in normal operation would be silently reshaping the
    model rather than catching a failure, so ordinary deltas must pass through
    untouched and leave the counter at zero."""

    coordinator = build_standard_coordinator(ShipConfig())
    tracker = BeliefTracker(1, 2, 0.1, coordinator, "cpu")
    visible = tracker.compose(_view(visible=True, x=300.0))

    prediction = torch.full((1, 2, coordinator.total_prediction_dimension), 0.5)
    tracker.advance(visible, prediction)

    assert torch.isfinite(tracker.predicted_targets).all()
    assert tracker.predicted_targets.abs().max() < BELIEF_TARGET_LIMIT
    assert int(tracker.clamp_events) == 0


def test_hidden_frontline_enemy_with_zero_shields_remains_alive():
    coordinator = build_standard_coordinator(ShipConfig())
    tracker = BeliefTracker(1, 2, 0.1, coordinator, "cpu")
    observed = _view(visible=True)
    observed.data[ObsKey.GAME_MODE] = torch.ones(1, 2, 1)
    visible = tracker.compose(observed)
    tracker.advance(visible, torch.zeros(1, 2, coordinator.total_prediction_dimension))
    tracker.predicted_targets[..., coordinator.target_slices()["health"]] = 0
    tracker.predicted_targets[..., coordinator.target_slices()["shield_delay"]] = -1
    hidden = _view(visible=False)
    hidden.data[ObsKey.GAME_MODE] = torch.ones(1, 2, 1)
    composed = tracker.compose(hidden)
    assert composed[ObsKey.HEALTH][0, 1, 0] == 0
    assert composed[ObsKey.ALIVE][0, 1]
    assert composed[ObsKey.BELIEF_VALID][0, 1]
    assert composed[ObsKey.SHIELD_DELAY][0, 1, 0] == 0


def test_imagined_frontline_ship_with_zero_shields_remains_alive():
    from boost_and_broadside.evaluation.next_state import decode_targets_to_observation

    coordinator = build_standard_coordinator(ShipConfig())
    observed = _view(visible=True)
    observed.data[ObsKey.GAME_MODE] = torch.ones(1, 2, 1)
    targets = coordinator.get_target_vector(observed)
    targets[..., coordinator.target_slices()["health"]] = 0
    imagined = decode_targets_to_observation(
        targets, observed, torch.zeros(1, 2, 3, dtype=torch.long), 2, coordinator
    )
    assert imagined[ObsKey.ALIVE].all()
    assert imagined[ObsKey.HEALTH].eq(0).all()
