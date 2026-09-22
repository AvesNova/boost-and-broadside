"""Every shipped channel is predicted absolutely, so its label is the truth.

``BeliefTracker.advance`` applies the head's forecast to the *composed*
observation. While a channel was predicted as a delta that made the label's base
load-bearing: training on a truth-to-truth delta conserved the belief error
exactly instead of correcting it, which is how run 734 died, and re-basing onto
the believed state was the fix.

Nothing in the shipped feature table is a delta any more. An absolute channel
asks the head for the state rather than a step away from a base, so there is no
base to be stale, the label is simply the true next state, and the error cannot
be conserved because it is never carried forward at all.

These therefore pin two different things: that the shipped labels really are the
truth, and -- against a coordinator built for the purpose -- that the re-basing
machinery still works, since it is what any future delta channel would need.
"""

import math
import types

import pytest
import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.env.observation import ObjectType, ObsKey, YemongObservation
from boost_and_broadside.train.rl.features import (
    Accessor,
    AdditivePredictor,
    Feature,
    FeatureCoordinator,
    Identity,
    build_standard_coordinator,
)
from boost_and_broadside.train.rl.ppo import PPOTrainer

T, B, N = 2, 1, 2


def _obs(
    positions: torch.Tensor,
    velocities: torch.Tensor | None = None,
    indices: torch.Tensor | None = None,
) -> dict:
    """A (T+1, B, N, ...) observation with the given ship positions/velocities."""

    shape = (T + 1, B, N)
    return {
        ObsKey.POS: positions,
        ObsKey.VEL: torch.full((*shape, 2), 3.0) if velocities is None else velocities,
        ObsKey.ATT: torch.tensor([1.0, 0.0]).expand(*shape, 2).clone(),
        ObsKey.ANG_VEL: torch.full((*shape, 1), 0.1),
        ObsKey.HEALTH: torch.full((*shape, 1), 80.0),
        ObsKey.POWER: torch.full((*shape, 1), 60.0),
        ObsKey.COOLDOWN: torch.full((*shape, 1), 0.2),
        ObsKey.TEAM_ID: torch.zeros(shape, dtype=torch.int32),
        ObsKey.ALIVE: torch.ones(shape, dtype=torch.bool),
        ObsKey.VISIBLE: torch.ones(shape, dtype=torch.bool),
        ObsKey.BELIEF_VALID: torch.ones(shape, dtype=torch.bool),
        ObsKey.TIME_SINCE_OBSERVATION: torch.zeros((*shape, 1)),
        ObsKey.OBJECT_TYPE: torch.full(shape, int(ObjectType.SHIP), dtype=torch.int32),
        ObsKey.ZONE_ROLE: torch.full(shape, 5, dtype=torch.int32),
        ObsKey.PREVIOUS_ACTION: torch.ones((*shape, 3), dtype=torch.long),
        ObsKey.RADIUS: torch.full((*shape, 1), 16.0),
        ObsKey.LOCAL_LOG_INDEX: (torch.full((*shape, 1), 0.1) if indices is None else indices),
        ObsKey.LOCAL_INDEX_GRADIENT: torch.full((*shape, 2), 0.3),
    }


def _targets(coordinator, obs: dict) -> torch.Tensor:
    flat = YemongObservation(
        data={k: v.reshape((T + 1) * B, N, *v.shape[3:]) for k, v in obs.items()}
    )
    return coordinator.get_target_vector(flat).reshape(T + 1, B, N, -1)


def _run() -> tuple:
    """Drive _precompute_ns_labels over a belief that is stale by 40px."""

    coordinator = build_standard_coordinator(ShipConfig())
    truth_pos = torch.zeros((T + 1, B, N, 2))
    truth_pos[..., 0] = torch.tensor([100.0, 110.0, 120.0]).view(T + 1, 1, 1)
    truth_pos[..., 1] = 200.0

    believed_pos = truth_pos.clone()
    believed_pos[:, :, 1, 0] += 40.0  # ship 1's belief lags reality

    # Stale in a delta channel too, so the re-basing has something to correct:
    # with position predicted absolutely, a position-only lag is invisible to the
    # label by design.
    truth_vel = torch.full((T + 1, B, N, 2), 3.0)
    believed_vel = truth_vel.clone()
    believed_vel[:, :, 1, 0] += 5.0

    truth_index = torch.full((T + 1, B, N, 1), 0.1)
    believed_index = truth_index.clone()
    believed_index[:, :, 1, 0] += 0.3

    believed_obs = _obs(believed_pos, believed_vel, believed_index)
    truth_targets = _targets(coordinator, _obs(truth_pos, truth_vel, truth_index))
    believed_targets = _targets(coordinator, believed_obs)

    buf = types.SimpleNamespace(
        num_steps=T,
        num_envs=B,
        num_ships=N,
        obs=believed_obs,
        privileged_targets=truth_targets,
        ns_labels=None,
    )
    trainer = types.SimpleNamespace(
        cfg=types.SimpleNamespace(next_state_coef=1.0),
        coordinator=coordinator,
        _precompute_belief_diagnostics=lambda *args: None,
    )
    PPOTrainer._precompute_ns_labels(trainer, buf)
    return coordinator, buf, believed_targets, truth_targets


def test_label_steps_from_the_believed_state_to_the_true_next_state() -> None:
    coordinator, buf, believed, truth = _run()
    expected = coordinator.compute_labels(believed[:T], truth[1:])
    assert torch.allclose(buf.ns_labels, expected)


def test_every_shipped_channel_is_absolute_so_its_label_is_the_truth() -> None:
    """The invariant that made the belief base stop mattering.

    Not a restatement of the line above: this says the *shipped table* has no
    delta channel left, so a stale belief cannot reach any label. Add one back
    and this fails, which is the point -- it would also reintroduce the
    error-conservation mode that re-basing exists to prevent.
    """

    coordinator, buf, _, truth = _run()
    assert not [
        spec.name
        for spec in coordinator._predictor_specs
        if isinstance(spec.predictor, AdditivePredictor)
    ]
    # Ship 1's belief is stale by 40 px, 5 px/s and 0.3 of log index; none of it
    # appears, because no label is measured from a base.
    assert torch.equal(buf.ns_labels, truth[1:])


def test_a_delta_channel_would_still_re_base() -> None:
    """The mechanism, kept under test against the table that no longer uses it.

    Re-basing is what any future delta channel would depend on, and run 734 is
    what happens without it. Exercised against a coordinator built for the
    purpose rather than deleted along with its last caller.
    """

    coordinator = FeatureCoordinator(
        [
            Feature(
                name="angular_velocity",
                accessor=Accessor(ObsKey.ANG_VEL),
                input_encoder=Identity(),
                target_encoder=Identity(),
                predictor=AdditivePredictor(),
            )
        ]
    )
    believed = torch.tensor([[[0.5]], [[0.5]]])
    truth = torch.tensor([[[0.2]], [[0.9]]])

    re_based = coordinator.compute_labels(believed[:1], truth[1:])
    conserving = coordinator.compute_labels(truth[:1], truth[1:])
    # The belief is stale by 0.3, so the correction back onto truth differs from
    # the truth-to-truth step by exactly that much.
    assert re_based.item() == pytest.approx(0.4)
    assert conserving.item() == pytest.approx(0.7)
    # And applying the re-based label to the belief lands on the truth, which is
    # the identity the conserving form fails.
    landed = coordinator.apply_all_predictions(believed[0], re_based[0])
    assert landed.item() == pytest.approx(truth[1].item())


def test_applying_the_label_to_the_belief_lands_on_the_truth() -> None:
    """The identity the fix exists for: error is nulled, not conserved."""

    coordinator, buf, believed, truth = _run()
    scale = coordinator.label_scale_vector(buf.ns_labels.device)
    landed = coordinator.apply_all_predictions(
        believed[0].reshape(B * N, -1), (buf.ns_labels[0] / scale).reshape(B * N, -1)
    )
    assert torch.allclose(landed, truth[1].reshape(B * N, -1), atol=1e-4)


def test_label_scale_diagnostic_reports_the_correction_that_recalibrates_it() -> None:
    """A calibrated label has mean square 1; the suggestion is what restores that.

    ``label_scale`` is defined as 1/std(raw label), so the metric exists to be
    read off a chart and written back into the feature. The relation below is
    that definition, and it must hold per prediction dimension or the number is
    not usable as a scale.
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

    current = trainer.coordinator.label_scale_vector(torch.device("cpu"))
    names = trainer.coordinator.get_feature_names()
    assert names, "no predicted features to calibrate"

    calibrated = 0
    for index, name in enumerate(names):
        mean_sq = metrics[f"next_state_label_sq/{name}"]
        assert math.isfinite(mean_sq) and mean_sq >= 0.0
        if mean_sq == 0.0:
            # No variation in this sample, so no scale to suggest.
            assert f"next_state_label_scale/{name}" not in metrics
            continue
        calibrated += 1
        # suggested = current / sqrt(mean_sq): applying it would drive the
        # label's mean square to 1.
        suggested = metrics[f"next_state_label_scale/{name}"]
        assert suggested * math.sqrt(mean_sq) == pytest.approx(current[index].item(), rel=1e-5)

    assert calibrated, "no feature produced a usable scale suggestion"
