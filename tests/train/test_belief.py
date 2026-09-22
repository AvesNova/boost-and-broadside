"""Contracts for recursive hidden-enemy point estimates."""

import pytest
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


def _hold(coordinator, composed, num_ships: int = 2) -> torch.Tensor:
    """The scaled prediction that means "no change".

    Not zeros. Position and attitude are predicted *absolutely* now, so a zero
    output claims the origin rather than declining to move; only the remaining
    delta channels read zero as "stand still". ``compute_labels`` of a state
    against itself is exactly that distinction, per predictor, already scaled the
    way ``advance`` expects its argument to be.
    """

    targets = coordinator.get_target_vector(composed)[:, :num_ships]
    return coordinator.compute_labels(targets, targets)


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
    tracker.advance(visible, _hold(coordinator, visible))

    hidden = tracker.compose(_view(visible=False, x=9999.0))

    assert hidden[ObsKey.BELIEF_VALID][0, 1]
    assert not hidden[ObsKey.VISIBLE][0, 1]
    assert torch.allclose(hidden[ObsKey.POS][0, 1], torch.tensor([300.0, 400.0]))
    assert hidden[ObsKey.TEAM_ID][0, 1] == 1
    assert hidden[ObsKey.TIME_SINCE_OBSERVATION][0, 1, 0] == 0.1
    assert hidden[ObsKey.PREVIOUS_ACTION][0, 1].equal(torch.zeros(3, dtype=torch.long))
    assert hidden[ObsKey.LOCAL_INDEX_GRADIENT][0, 1].equal(torch.zeros(2))

    tracker.advance(hidden, _hold(coordinator, hidden))
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


def test_deploy_reveal_makes_belief_valid_constant() -> None:
    """One revealed tick marks every ship valid for the rest of the episode.

    This is the property the attention key mask removal rests on: ``valid`` is
    sticky, so a single opening reveal makes ``BELIEF_VALID`` a constant rather
    than something the trunk has to be told about.
    """

    from boost_and_broadside.config import EnvConfig
    from boost_and_broadside.config.defaults import REWARDS
    from boost_and_broadside.env.wrapper import YemongEnvWrapper

    ship = ShipConfig()
    coordinator = build_standard_coordinator(ship)
    config = EnvConfig(
        num_ships=4,
        max_bullets=2,
        max_episode_steps=64,
        vision_range=100.0,
        spawn_reveal=True,
    )
    wrapper = YemongEnvWrapper(2, ship, config, REWARDS, "cpu")
    obs = wrapper.reset(seed=11)
    tracker = BeliefTracker(2, 4, 0.1, coordinator, "cpu")

    composed = tracker.compose(obs.for_team(0))
    assert composed[ObsKey.BELIEF_VALID][:, :4].all(), "opening tick must reveal every ship"

    prediction = torch.zeros((2, 4, coordinator.total_prediction_dimension))
    for _ in range(8):
        tracker.advance(composed.for_team(0), prediction)
        obs, *_ = wrapper.step(torch.zeros((2, 4, 3), dtype=torch.long))
        composed = tracker.compose(obs.for_team(0))
        # Ships drift apart and out of sight, but validity never lapses.
        assert composed[ObsKey.BELIEF_VALID][:, :4].all()


def test_a_revealed_respawn_corrects_a_stale_belief() -> None:
    """The reveal exists to end a lifecycle discontinuity the tracker cannot see.

    ``BeliefTracker`` advances a hidden ship by the policy's own forecast and is
    never told it died, so without the spawn reveal an unobserved respawn leaves
    the belief tracking a corpse's trajectory: the policy acts on a phantom at
    the old position, and the auxiliary label carries a teleport no head could
    have predicted, on every step until the ship is next seen.
    ``transition_contiguous`` masks only the step the teleport happened on.
    """

    coordinator = build_standard_coordinator(ShipConfig())
    tracker = BeliefTracker(1, 2, 0.1, coordinator, "cpu")

    seen = tracker.compose(_view(visible=True, x=300.0))
    assert seen[ObsKey.POS][0, 1, 0] == pytest.approx(300.0)

    # Out of contact, and meanwhile it dies and respawns far away at x=900.
    for _ in range(3):
        tracker.advance(seen, _hold(coordinator, seen))
        seen = tracker.compose(_view(visible=False, x=900.0))
    stale = seen[ObsKey.POS][0, 1, 0].item()
    assert stale == pytest.approx(300.0), "belief should still be on the old trajectory"
    assert abs(900.0 - stale) > 500.0, "and so the label would carry the whole teleport"

    # The spawn reveal shows it for one decision: the belief snaps to truth.
    tracker.advance(seen, _hold(coordinator, seen))
    revealed = tracker.compose(_view(visible=True, x=900.0))
    assert revealed[ObsKey.POS][0, 1, 0] == pytest.approx(900.0)
    assert revealed[ObsKey.TIME_SINCE_OBSERVATION][0, 1, 0] == 0

    # And it stays corrected once contact is lost again.
    tracker.advance(revealed, _hold(coordinator, revealed))
    after = tracker.compose(_view(visible=False, x=900.0))
    assert after[ObsKey.POS][0, 1, 0] == pytest.approx(900.0)


def test_the_environment_latches_a_spawn_for_the_whole_decision() -> None:
    """Visibility reads a latch, not the one-tick physics flag.

    ``ship_respawned`` is cleared at the start of the next physics tick, so with
    ``action_repeat`` above 1 it would be gone by the time the observation is
    built. The latch is what survives to be observed.
    """

    from boost_and_broadside.config import EnvConfig
    from boost_and_broadside.config.defaults import REWARDS
    from boost_and_broadside.env.wrapper import YemongEnvWrapper

    ship = ShipConfig()
    config = EnvConfig(
        num_ships=4, max_bullets=2, max_episode_steps=64, vision_range=100.0, spawn_reveal=True
    )
    wrapper = YemongEnvWrapper(2, ship, config, REWARDS, "cpu")
    wrapper.reset(seed=5)
    assert wrapper.state.ship_spawned.all(), "a reset is a spawn for every ship"

    wrapper.step(torch.zeros((2, 4, 3), dtype=torch.long))
    assert not wrapper.state.ship_spawned.any(), "and the latch clears once observed"
