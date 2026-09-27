"""Physical one-decision deltas used by next-state belief calibration.

This module defines quantities in physical space, before the policy's Fourier,
symlog, or scalar input transforms.  Keeping this definition independent of the
current auxiliary-head representation gives Phase 3 one source of truth for the
calibration labels and their eventual normalization.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from boost_and_broadside.env.state import TensorState

PHYSICAL_DELTA_NAMES = (
    "position_x",
    "position_y",
    "velocity_x",
    "velocity_y",
    "attitude",
    "angular_velocity",
    "shield_delay",
    "health",
    "power",
    "cooldown",
    "local_log_index",
)

# Fixed Phase-1 calibration for the production 5v5 Frontline profile. Continuous
# channels use rounded RMS scales across valid truth-to-truth transitions;
# sparse/reset channels use their physical event magnitude instead of allowing
# the frequency of zeros to determine the scale. Position and velocity each use
# one scale for both axes by contract. See the checked-in calibration artifact
# and Phase-1 handoff for the distributions and selection rationale.
PHYSICAL_DELTA_SCALES = (
    2.5,  # position x (px)
    2.5,  # position y (px)
    4.0,  # velocity x (px/s)
    4.0,  # velocity y (px/s)
    0.1,  # attitude (rad)
    7.853981633974483,  # angular velocity (rad/s): 2.5*pi, p90 nonzero event
    5.0,  # shield delay (s): damage reset value
    10.0,  # health: nominal projectile damage
    0.75,  # power: rounded RMS
    0.1,  # cooldown (s): full firing reset
    0.05,  # natural log-index: representative field-interface change
)


@dataclass(frozen=True)
class PhysicalShipSnapshot:
    """The ship fields needed for a truth-to-truth physical delta."""

    position: torch.Tensor
    velocity: torch.Tensor
    attitude: torch.Tensor
    angular_velocity: torch.Tensor
    shield_delay: torch.Tensor
    health: torch.Tensor
    power: torch.Tensor
    cooldown: torch.Tensor
    local_index: torch.Tensor

    @classmethod
    def from_state(cls, state: TensorState) -> PhysicalShipSnapshot:
        """Snapshot fields that physics rebinds during a decision.

        ``TensorState`` advances by tensor reassignment, so aliases retain the
        pre-step values without cloning the much larger bullet and map state.
        """

        return cls(
            position=state.ship_pos,
            velocity=state.ship_vel,
            attitude=state.ship_attitude,
            angular_velocity=state.ship_ang_vel,
            shield_delay=state.ship_shield_delay,
            health=state.ship_health,
            power=state.ship_power,
            cooldown=state.ship_cooldown,
            local_index=state.ship_local_index,
        )


def physical_ship_deltas(
    current: PhysicalShipSnapshot,
    next_: PhysicalShipSnapshot,
    world_size: tuple[float, float],
) -> torch.Tensor:
    """Return the 11 physical deltas in ``PHYSICAL_DELTA_NAMES`` order.

    Position uses the torus's minimum-image displacement.  Attitude uses the
    signed shortest rotation in ``[-pi, pi]``.  Local log-index means the
    natural logarithm of the physical refractive index, not the observation's
    profile-dependent normalization of that value.
    """

    position_raw = next_.position - current.position
    world_w, world_h = world_size
    position = torch.complex(
        (position_raw.real + world_w / 2.0) % world_w - world_w / 2.0,
        (position_raw.imag + world_h / 2.0) % world_h - world_h / 2.0,
    )
    velocity = next_.velocity - current.velocity
    attitude = torch.angle(next_.attitude * torch.conj(current.attitude))
    return torch.stack(
        (
            position.real,
            position.imag,
            velocity.real,
            velocity.imag,
            attitude,
            next_.angular_velocity - current.angular_velocity,
            next_.shield_delay - current.shield_delay,
            next_.health - current.health,
            next_.power - current.power,
            next_.cooldown - current.cooldown,
            torch.log(next_.local_index) - torch.log(current.local_index),
        ),
        dim=-1,
    )


def normalize_physical_deltas(deltas: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Divide physical deltas by positive fixed scales.

    Ordinary division is intentional: IEEE arithmetic maps an exact zero
    physical delta to exact zero, with no offset or moving statistic involved.
    """

    if deltas.shape[-1] != len(PHYSICAL_DELTA_NAMES):
        raise ValueError(
            f"expected {len(PHYSICAL_DELTA_NAMES)} delta channels, got {deltas.shape[-1]}"
        )
    if scales.shape[-1] != len(PHYSICAL_DELTA_NAMES):
        raise ValueError(f"expected {len(PHYSICAL_DELTA_NAMES)} scales, got {scales.shape[-1]}")
    if not bool(torch.all(torch.isfinite(scales) & (scales > 0))):
        raise ValueError("physical delta scales must be finite and positive")
    return deltas / scales
