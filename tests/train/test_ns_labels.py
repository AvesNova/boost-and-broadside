"""The next-state label steps from the believed state to the true next one.

``BeliefTracker.advance`` applies the head's forecast to the belief, so
``belief[t+1] = belief[t] + pred[t]``. Training on a truth-to-truth delta instead
makes the substitution

    error[t+1] = belief[t] + (true[t+1] - true[t]) - true[t+1] = error[t]

-- the belief error is conserved exactly, every step's noise is retained forever,
and the head is never once shown what "too far" looks like. Run 734 died that
way. Re-basing on the belief makes the label the correction that carries the
believed state onto the true next one, so error is nulled each step to whatever
extent it is inferable.

For a ship the observer can see, the belief *is* truth, so its label is exactly
the Phase-1 truth-to-truth delta the scales were calibrated on: re-basing adds
signal where the drift happens and leaves the rest of the supervision alone.
"""

import math
import types

import pytest
import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.env.observation import ObjectType, ObsKey
from boost_and_broadside.train.rl.physical_belief import (
    PHYSICAL_MEAN_DIM,
    PHYSICAL_MEAN_NAMES,
    POSITION_X,
    VELOCITY_X,
    PhysicalNextState,
    physical_means_from_observation,
)
from boost_and_broadside.train.rl.ppo import PPOTrainer

T, B, N = 2, 1, 2
_SHIP = ShipConfig()
_LOG_SCALE = 2.0 * math.log(_SHIP.field_index_step)


def _obs(
    positions: torch.Tensor,
    velocities: torch.Tensor | None = None,
    indices: torch.Tensor | None = None,
) -> dict:
    """A ``(T+1, B, N, ...)`` observation with the given ship physical state."""

    shape = (T + 1, B, N)
    return {
        ObsKey.POS: positions,
        ObsKey.VEL: torch.full((*shape, 2), 3.0) if velocities is None else velocities,
        ObsKey.ATT: torch.tensor([1.0, 0.0]).expand(*shape, 2).clone(),
        ObsKey.ANG_VEL: torch.full((*shape, 1), 0.1),
        ObsKey.SHIELD_DELAY: torch.zeros((*shape, 1)),
        ObsKey.HEALTH: torch.full((*shape, 1), 80.0),
        ObsKey.POWER: torch.full((*shape, 1), 60.0),
        # Inside the physical bound: the belief recursion clamps a cooldown
        # to the configured firing interval, which is 0.1 s.
        ObsKey.COOLDOWN: torch.full((*shape, 1), 0.05),
        ObsKey.TEAM_ID: torch.zeros(shape, dtype=torch.int32),
        ObsKey.ALIVE: torch.ones(shape, dtype=torch.bool),
        ObsKey.VISIBLE: torch.ones(shape, dtype=torch.bool),
        ObsKey.BELIEF_VALID: torch.ones(shape, dtype=torch.bool),
        ObsKey.TIME_SINCE_OBSERVATION: torch.zeros((*shape, 1)),
        ObsKey.OBJECT_TYPE: torch.full(shape, int(ObjectType.SHIP), dtype=torch.int32),
        ObsKey.ZONE_ROLE: torch.full(shape, 5, dtype=torch.int32),
        ObsKey.PREVIOUS_ACTION: torch.zeros((*shape, 42)),
        ObsKey.RADIUS: torch.full((*shape, 1), 16.0),
        ObsKey.LOCAL_LOG_INDEX: (torch.full((*shape, 1), 0.1) if indices is None else indices),
        ObsKey.LOCAL_INDEX_GRADIENT: torch.full((*shape, 2), 0.3),
    }


def _run() -> tuple:
    """Drive ``_precompute_ns_labels`` over a belief that is stale by 40 px."""

    next_state = PhysicalNextState.from_ship_config(_SHIP)
    truth_pos = torch.zeros((T + 1, B, N, 2))
    truth_pos[..., 0] = torch.tensor([100.0, 110.0, 120.0]).view(T + 1, 1, 1)
    truth_pos[..., 1] = 200.0

    believed_pos = truth_pos.clone()
    believed_pos[:, :, 1, 0] += 40.0  # ship 1's belief lags reality

    truth_vel = torch.full((T + 1, B, N, 2), 3.0)
    believed_vel = truth_vel.clone()
    believed_vel[:, :, 1, 0] += 5.0

    believed_obs = _obs(believed_pos, believed_vel)
    truth_means = physical_means_from_observation(
        _obs(truth_pos, truth_vel), _LOG_SCALE, num_ships=N
    )
    believed_means = physical_means_from_observation(believed_obs, _LOG_SCALE, num_ships=N)

    buf = types.SimpleNamespace(
        num_steps=T,
        num_envs=B,
        num_ships=N,
        obs=believed_obs,
        privileged_means=truth_means,
        ns_labels=None,
    )
    trainer = types.SimpleNamespace(
        cfg=types.SimpleNamespace(next_state_coef=1.0),
        next_state=next_state,
        _index_log_scale=_LOG_SCALE,
        _believed_means=lambda buf, steps: PPOTrainer._believed_means(trainer, buf, steps),
        _precompute_belief_diagnostics=lambda *args: None,
    )
    PPOTrainer._precompute_ns_labels(trainer, buf)
    return next_state, buf, believed_means, truth_means


def test_the_label_steps_from_the_belief_to_the_true_next_state() -> None:
    next_state, buf, believed, truth = _run()
    expected = next_state.labels(believed[:T], truth[1:])
    assert torch.allclose(buf.ns_labels, expected)


def test_a_stale_belief_shows_up_in_the_label() -> None:
    """The visible ship's label is the ordinary delta; the stale one's is bigger.

    Ship 0's belief is truth, so its position label is the 10 px step divided by
    the calibrated 2.5 px scale. Ship 1's belief lags by 40 px, so its label
    carries that correction as well -- which is the signal re-basing exists to
    produce, and the one a truth-to-truth label cannot contain.
    """
    _, buf, _, _ = _run()
    labels = buf.ns_labels
    assert labels[0, 0, 0, POSITION_X].item() == pytest.approx(10.0 / 2.5)
    assert labels[0, 0, 1, POSITION_X].item() == pytest.approx((10.0 - 40.0) / 2.5)
    assert labels[0, 0, 0, VELOCITY_X].item() == pytest.approx(0.0)
    assert labels[0, 0, 1, VELOCITY_X].item() == pytest.approx(-5.0 / 4.0)


def test_applying_the_label_to_the_belief_lands_on_the_truth() -> None:
    """The identity the re-basing exists for: error is nulled, not conserved."""
    next_state, buf, believed, truth = _run()
    prediction = torch.cat([buf.ns_labels[0], torch.zeros(B, N, PHYSICAL_MEAN_DIM + 2)], dim=-1)
    landed = next_state.apply_means(believed[0], prediction)
    assert torch.allclose(landed, truth[1], atol=1e-3)


def test_a_truth_to_truth_label_would_conserve_the_error() -> None:
    """The failure mode, stated as arithmetic rather than as a comment."""
    next_state, _, believed, truth = _run()
    conserving = next_state.labels(truth[:T], truth[1:])
    prediction = torch.cat([conserving[0], torch.zeros(B, N, PHYSICAL_MEAN_DIM + 2)], dim=-1)
    landed = next_state.apply_means(believed[0], prediction)
    error_before = (believed[0, 0, 1, POSITION_X] - truth[0, 0, 1, POSITION_X]).abs()
    error_after = (landed[0, 1, POSITION_X] - truth[1, 0, 1, POSITION_X]).abs()
    assert error_after.item() == pytest.approx(error_before.item(), abs=1e-3)


def test_the_label_scale_diagnostic_reports_the_correction_that_recalibrates_it() -> None:
    """A conditioned label has mean square 1; the suggestion is what restores it.

    The Phase-1 scales are fixed by contract, so this is a diagnostic rather than
    a control input -- but it has to be readable as a scale or it says nothing.
    """

    import tempfile

    from tests.train.test_ppo import _make_trainer

    torch.manual_seed(3)
    with tempfile.TemporaryDirectory() as tmp:
        trainer = _make_trainer(checkpoint_dir=tmp)
        runtime = trainer._initialize_rollout_runtime()
        dones = trainer._collect_rollout(runtime, False)
        trainer._compute_rollout_gae(runtime, dones)
        metrics = trainer._update_epochs(
            all_buffers=[trainer.buffer, *trainer.aux_buffers], record_histograms=False
        )

    calibrated = 0
    for index, name in enumerate(PHYSICAL_MEAN_NAMES):
        mean_sq = metrics[f"next_state_label_sq/{name}"]
        assert math.isfinite(mean_sq) and mean_sq >= 0.0
        if mean_sq == 0.0:
            # No variation in this sample, so no scale to suggest.
            assert f"next_state_label_scale/{name}" not in metrics
            continue
        calibrated += 1
        # suggested = current * sqrt(mean_sq): applying it would drive the
        # label's mean square to 1.
        suggested = metrics[f"next_state_label_scale/{name}"]
        assert suggested / math.sqrt(mean_sq) == pytest.approx(
            trainer.next_state.scales[index], rel=1e-5
        )

    assert calibrated, "no channel produced a usable scale suggestion"
