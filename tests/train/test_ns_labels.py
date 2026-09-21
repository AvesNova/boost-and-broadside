"""The next-state label is the step from the believed state to the true one.

``BeliefTracker.advance`` applies the head's forecast to the *composed*
observation, so training that head on a truth-to-truth delta conserves the
belief error exactly instead of correcting it. These pin the re-based label.
"""

import types

import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.env.observation import ObjectType, ObsKey, YemongObservation
from boost_and_broadside.train.rl.features import build_standard_coordinator
from boost_and_broadside.train.rl.ppo import PPOTrainer

T, B, N = 2, 1, 2


def _obs(positions: torch.Tensor) -> dict:
    """A (T+1, B, N, ...) observation whose ship positions are ``positions``."""

    shape = (T + 1, B, N)
    return {
        ObsKey.POS: positions,
        ObsKey.VEL: torch.full((*shape, 2), 3.0),
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
        ObsKey.LOCAL_LOG_INDEX: torch.full((*shape, 1), 0.1),
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

    believed_obs = _obs(believed_pos)
    truth_targets = _targets(coordinator, _obs(truth_pos))
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


def test_label_is_not_the_truth_to_truth_delta_when_the_belief_is_stale() -> None:
    """The distinction is the whole point: a stale belief must see a correction."""

    coordinator, buf, _, truth = _run()
    conserving = coordinator.compute_labels(truth[:T], truth[1:])
    # Ship 0's belief is exact, so its label is untouched by the re-basing.
    assert torch.allclose(buf.ns_labels[:, :, 0], conserving[:, :, 0])
    # Ship 1's is stale, so its label now carries the correction back to truth.
    assert not torch.allclose(buf.ns_labels[:, :, 1], conserving[:, :, 1])


def test_applying_the_label_to_the_belief_lands_on_the_truth() -> None:
    """The identity the fix exists for: error is nulled, not conserved."""

    coordinator, buf, believed, truth = _run()
    scale = coordinator.label_scale_vector(buf.ns_labels.device)
    landed = coordinator.apply_all_predictions(
        believed[0].reshape(B * N, -1), (buf.ns_labels[0] / scale).reshape(B * N, -1)
    )
    assert torch.allclose(landed, truth[1].reshape(B * N, -1), atol=1e-4)
