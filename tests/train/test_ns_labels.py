"""The next-state target is the true next state, never a step from the belief.

The head's decoded output *is* the next belief, so it is trained toward the
exact code of where the ship really is one decision later. A stale belief cannot
survive that: whatever the head reads, the target says where the ship is. Run 734
died of the opposite, a truth-to-truth step applied to a belief, which conserves
the belief's error exactly.
"""

import math
import types

import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.env.observation import ObjectType, ObsKey
from boost_and_broadside.train.rl.physical_belief import (
    PHYSICAL_MEAN_DIM,
    POSITION_X,
    physical_means_from_observation,
)
from boost_and_broadside.train.rl.ppo import PPOTrainer
from boost_and_broadside.train.rl.ship_codes import ShipStateCodec

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
        ObsKey.PREVIOUS_ACTION: torch.zeros((*shape, 30)),
        ObsKey.RADIUS: torch.full((*shape, 1), 16.0),
        ObsKey.LOCAL_LOG_INDEX: (torch.full((*shape, 1), 0.1) if indices is None else indices),
        ObsKey.LOCAL_INDEX_GRADIENT: torch.full((*shape, 2), 0.3),
    }


def _run(privileged: bool = True) -> tuple:
    """Drive ``_precompute_ns_labels`` over a belief that is stale by 40 px."""

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
        privileged_means=truth_means if privileged else None,
        ns_labels=None,
    )
    trainer = types.SimpleNamespace(
        cfg=types.SimpleNamespace(next_state_coef=1.0),
        _outcome_global_k=(),
        _index_log_scale=_LOG_SCALE,
        _believed_means=lambda buf, steps: PPOTrainer._believed_means(trainer, buf, steps),
        _precompute_belief_diagnostics=lambda *args: None,
    )
    PPOTrainer._precompute_ns_labels(trainer, buf)
    return buf, believed_means, truth_means


def test_the_target_is_the_true_next_state() -> None:
    buf, _, truth = _run()
    assert buf.ns_labels.shape == (T, B, N, PHYSICAL_MEAN_DIM)
    assert torch.equal(buf.ns_labels, truth[1:])


def test_a_stale_belief_does_not_move_the_target() -> None:
    """Ship 1's belief lags by 40 px; its target is where it really is."""
    buf, believed, _ = _run()
    assert buf.ns_labels[0, 0, 1, POSITION_X].item() == 110.0
    assert believed[0, 0, 1, POSITION_X].item() == 140.0


def test_a_head_that_meets_its_target_lands_the_belief_on_the_truth() -> None:
    """The identity the absolute target exists for: error is nulled, not conserved."""
    buf, _, truth = _run()
    codec = ShipStateCodec.from_ship_config(_SHIP)
    landed = codec.decode(codec.sharp(buf.ns_labels[0]))[..., :PHYSICAL_MEAN_DIM]
    torch.testing.assert_close(landed, truth[1], atol=1e-3, rtol=0.0)


def test_without_privileged_truth_the_target_is_the_next_observation() -> None:
    buf, believed, _ = _run(privileged=False)
    assert torch.equal(buf.ns_labels, believed[1:])
