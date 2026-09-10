"""Contracts for recursive hidden-enemy point estimates."""

import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.env.observation import ObjectType, ObsKey, YemongObservation
from boost_and_broadside.train.rl.belief import BeliefTracker
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
