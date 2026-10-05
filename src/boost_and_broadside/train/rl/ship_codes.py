"""The ship-state code: what the encoder reads and the next-state head predicts.

Every predicted ship channel has a categorical code (``frontline-redesign-plan.md``
§8.4–8.7). One code vector per ship token, 469 wide, is the concatenation of:

=====================  ======  =====================================
channel                width   groups (one softmax each)
=====================  ======  =====================================
position               81      9 nested levels of 9 colours
velocity               243     3 projection axes of 81 bins
attitude               16      4 nested levels of 4 colours
angular velocity       41      1
shield delay           21      1
health                 21      1
power                  21      1
local log index        21      1
cooldown               4       1
=====================  ======  =====================================

:class:`ShipStateCodec` turns the belief's moments into that vector
(``encode``), truth into its exact code (``sharp``), and a predicted code back
into moments (``decode``). With zero residual the round trip returns the same
moments, which is what lets a hidden ship's belief stand still on its own.

The head's logits are a residual on the input code, ``log(code + eps) + f(h)``;
``eps`` keeps the log finite where the code is zero. :meth:`ShipStateCodec.decode_logits`
removes that floor before decoding, so a zero residual decodes to exactly the
moments it was built from.

All arithmetic here is elementwise. The trainer runs under bf16 autocast, which
casts matrix products down, and the round trip needs float32.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.train.rl.categorical_codes import (
    ATTITUDE_COLOURS,
    ATTITUDE_LEVELS,
    POSITION_COLOURS,
    POSITION_LEVELS,
    CircularCode,
    PositionCode,
    ScalarCode,
    VelocityCode,
)
from boost_and_broadside.train.rl.physical_belief import (
    ANGULAR_VELOCITY,
    ANGULAR_VELOCITY_SIGMA,
    ATTITUDE,
    ATTITUDE_SIGMA,
    BELIEF_MOMENT_DIM,
    COOLDOWN,
    COOLDOWN_BINS,
    COOLDOWN_RESIDUAL,
    PHYSICAL_MEAN_DIM,
    PHYSICAL_UNCERTAINTY_DIM,
    POSITION_SIGMA,
    POSITION_X,
    SCALAR_SIGMAS,
    VELOCITY_COVARIANCE,
    VELOCITY_X,
)

#: Floor under the input code before its log becomes the head's baseline.
BASELINE_EPS = 1e-3
#: Bins of every bounded scalar channel (§8.7).
SCALAR_BINS = 21
#: Bins of angular velocity (§8.6).
ANGULAR_VELOCITY_BINS = 41
#: Shield delay range in seconds: the Frontline recharge delay (§8.7).
SHIELD_DELAY_MAX = 5.0


@dataclass(frozen=True)
class CodeRun:
    """A contiguous run of equally sized softmax groups in the code vector."""

    name: str
    start: int
    groups: int
    size: int

    @property
    def width(self) -> int:
        return self.groups * self.size

    @property
    def stop(self) -> int:
        return self.start + self.width

    def group_names(self) -> tuple[str, ...]:
        if self.groups == 1:
            return (self.name,)
        return tuple(f"{self.name}_{k}" for k in range(self.groups))


def _runs() -> tuple[CodeRun, ...]:
    layout = (
        ("position", POSITION_LEVELS, POSITION_COLOURS**2),
        ("velocity", 3, VelocityCode().bins),
        ("attitude", ATTITUDE_LEVELS, ATTITUDE_COLOURS),
        ("angular_velocity", 1, ANGULAR_VELOCITY_BINS),
        ("shield_delay", 1, SCALAR_BINS),
        ("health", 1, SCALAR_BINS),
        ("power", 1, SCALAR_BINS),
        ("local_log_index", 1, SCALAR_BINS),
        ("cooldown", 1, COOLDOWN_BINS),
    )
    runs = []
    start = 0
    for name, groups, size in layout:
        runs.append(CodeRun(name, start, groups, size))
        start += groups * size
    return tuple(runs)


#: The code layout, in vector order.
CODE_RUNS = _runs()
CODE_RUN = {run.name: run for run in CODE_RUNS}
#: Width of one ship's code vector.
SHIP_CODE_DIM = CODE_RUNS[-1].stop
#: One name per softmax group, in vector order.
CODE_GROUP_NAMES = tuple(name for run in CODE_RUNS for name in run.group_names())
CODE_GROUP_DIM = len(CODE_GROUP_NAMES)
#: The position code's width; the one part of the code non-ship tokens carry.
POSITION_CODE_DIM = CODE_RUN["position"].width


@dataclass(frozen=True)
class ShipStateCodec:
    """Moments ⇄ categorical code for one ship configuration."""

    position: PositionCode
    velocity: VelocityCode
    attitude: CircularCode
    angular_velocity: ScalarCode
    #: Shield delay, health, power and local log index, in ``SCALAR_SIGMAS`` order.
    scalars: tuple[ScalarCode, ScalarCode, ScalarCode, ScalarCode]
    cooldown: ScalarCode

    @classmethod
    def from_ship_config(cls, ship_config: ShipConfig) -> ShipStateCodec:
        # Angular velocity spans the g-limited path rate plus the slip-rate limit
        # at the lowest field index, where both are fastest (§8.6).
        lowest_index = ship_config.field_index_step**-2
        turn_rate = (
            math.sqrt(ship_config.max_lateral_accel * ship_config.max_lift_coeff)
            + ship_config.max_slip_rate
        ) / lowest_index
        index_span = 2.0 * math.log(ship_config.field_index_step)
        return cls(
            position=PositionCode(
                (float(ship_config.world_size[0]), float(ship_config.world_size[1]))
            ),
            velocity=VelocityCode(),
            attitude=CircularCode(2.0 * math.pi, ATTITUDE_COLOURS, ATTITUDE_LEVELS),
            angular_velocity=ScalarCode(-turn_rate, turn_rate, ANGULAR_VELOCITY_BINS),
            scalars=(
                ScalarCode(0.0, SHIELD_DELAY_MAX, SCALAR_BINS),
                ScalarCode(0.0, float(ship_config.max_health), SCALAR_BINS),
                ScalarCode(0.0, float(ship_config.max_power), SCALAR_BINS),
                ScalarCode(-index_span, index_span, SCALAR_BINS),
            ),
            cooldown=ScalarCode(0.0, float(ship_config.firing_cooldown), COOLDOWN_BINS),
        )

    # ------------------------------------------------------------------
    # Moments → code
    # ------------------------------------------------------------------

    def encode_position(self, position: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """``(..., 81)`` position code from a ``(..., 2)`` mean and a sigma."""
        return self.position.encode(position, sigma).flatten(-2)

    def encode(self, means: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        """``(..., 469)`` probabilities from ``(..., 11)`` means and ``(..., 14)`` spreads."""
        parts = [
            self.encode_position(
                means[..., POSITION_X : POSITION_X + 2], uncertainty[..., POSITION_SIGMA]
            ),
            self.encode_state(means, uncertainty),
        ]
        return torch.cat(parts, dim=-1)

    def encode_state(self, means: torch.Tensor, uncertainty: torch.Tensor) -> torch.Tensor:
        """``(..., 388)``: every channel of :meth:`encode` after position."""
        parts = [
            self.velocity.encode(
                means[..., VELOCITY_X : VELOCITY_X + 2], uncertainty[..., VELOCITY_COVARIANCE]
            ).flatten(-2),
            self.attitude.encode(means[..., ATTITUDE], uncertainty[..., ATTITUDE_SIGMA]).flatten(
                -2
            ),
            self.angular_velocity.encode(
                means[..., ANGULAR_VELOCITY], uncertainty[..., ANGULAR_VELOCITY_SIGMA]
            ),
        ]
        for code, (channel, column) in zip(self.scalars, SCALAR_SIGMAS, strict=True):
            parts.append(code.encode(means[..., channel], uncertainty[..., column]))
        cooldown = self.cooldown.sharp(means[..., COOLDOWN]) + uncertainty[..., COOLDOWN_RESIDUAL]
        parts.append(cooldown.clamp_min(0.0))
        return torch.cat(parts, dim=-1)

    def sharp(self, means: torch.Tensor) -> torch.Tensor:
        """``(..., 469)`` exact code of ``(..., 11)`` true means: every spread zero."""
        zeros = torch.zeros(
            (*means.shape[:-1], PHYSICAL_UNCERTAINTY_DIM), device=means.device, dtype=means.dtype
        )
        return self.encode(means, zeros)

    # ------------------------------------------------------------------
    # Code → moments
    # ------------------------------------------------------------------

    def decode(self, probabilities: torch.Tensor) -> torch.Tensor:
        """``(..., 25)`` moments, means then spreads, from ``(..., 469)`` probabilities."""
        run = CODE_RUN["position"]
        position, position_sigma = self.position.decode(
            probabilities[..., run.start : run.stop].unflatten(-1, (run.groups, run.size))
        )
        run = CODE_RUN["velocity"]
        velocity, covariance = self.velocity.decode(
            probabilities[..., run.start : run.stop].unflatten(-1, (run.groups, run.size))
        )
        run = CODE_RUN["attitude"]
        attitude, attitude_sigma = self.attitude.decode(
            probabilities[..., run.start : run.stop].unflatten(-1, (run.groups, run.size))
        )
        attitude = (attitude + math.pi).remainder(2.0 * math.pi) - math.pi
        run = CODE_RUN["angular_velocity"]
        angular_velocity, angular_sigma = self.angular_velocity.decode(
            probabilities[..., run.start : run.stop]
        )
        scalar_means, scalar_sigmas = [], []
        for code, name in zip(
            self.scalars, ("shield_delay", "health", "power", "local_log_index"), strict=True
        ):
            run = CODE_RUN[name]
            mean, sigma = code.decode(probabilities[..., run.start : run.stop])
            scalar_means.append(mean)
            scalar_sigmas.append(sigma)
        run = CODE_RUN["cooldown"]
        cooldown_probabilities = probabilities[..., run.start : run.stop]
        cooldown = (cooldown_probabilities * self.cooldown.centres(probabilities)).sum(-1)
        residual = cooldown_probabilities - self.cooldown.sharp(cooldown)

        means = [None] * PHYSICAL_MEAN_DIM
        means[POSITION_X], means[POSITION_X + 1] = position.unbind(-1)
        means[VELOCITY_X], means[VELOCITY_X + 1] = velocity.unbind(-1)
        means[ATTITUDE] = attitude
        means[ANGULAR_VELOCITY] = angular_velocity
        for (channel, _), mean in zip(SCALAR_SIGMAS, scalar_means, strict=True):
            means[channel] = mean
        means[COOLDOWN] = cooldown
        spreads = [None] * (PHYSICAL_UNCERTAINTY_DIM - COOLDOWN_BINS)
        spreads[POSITION_SIGMA] = position_sigma
        spreads[VELOCITY_COVARIANCE] = covariance.unbind(-1)
        spreads[ATTITUDE_SIGMA] = attitude_sigma
        spreads[ANGULAR_VELOCITY_SIGMA] = angular_sigma
        for (_, column), sigma in zip(SCALAR_SIGMAS, scalar_sigmas, strict=True):
            spreads[column] = sigma
        return torch.cat(
            [torch.stack(means, dim=-1), torch.stack(spreads, dim=-1), residual], dim=-1
        )

    # ------------------------------------------------------------------
    # Logits
    # ------------------------------------------------------------------

    @staticmethod
    def baseline(code: torch.Tensor) -> torch.Tensor:
        """The head's zero-residual logits: the input code, floored and logged."""
        return torch.log(code + BASELINE_EPS)

    @staticmethod
    def log_probabilities(logits: torch.Tensor) -> torch.Tensor:
        """``(..., 469)`` log-softmax within each group."""
        parts = [
            torch.log_softmax(
                logits[..., run.start : run.stop].unflatten(-1, (run.groups, run.size)).float(),
                dim=-1,
            ).flatten(-2)
            for run in CODE_RUNS
        ]
        return torch.cat(parts, dim=-1)

    @staticmethod
    def cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """``(..., 22)`` cross-entropy of each group against a target code."""
        log_probabilities = ShipStateCodec.log_probabilities(logits)
        product = -(target * log_probabilities)
        parts = [
            product[..., run.start : run.stop].unflatten(-1, (run.groups, run.size)).sum(-1)
            for run in CODE_RUNS
        ]
        return torch.cat(parts, dim=-1)

    def decode_logits(self, logits: torch.Tensor) -> torch.Tensor:
        """``(..., 25)`` moments of a head output, its baseline floor removed.

        ``softmax(log(p + eps)) = (p + eps) / (1 + n eps)`` for a group of ``n``,
        so inverting that mixture with uniform recovers ``p`` exactly when the
        residual is zero. A learned output is read the same way, with any
        probability the inversion takes below zero clipped.
        """
        probabilities = self.log_probabilities(logits).exp()
        parts = []
        for run in CODE_RUNS:
            group = probabilities[..., run.start : run.stop].unflatten(-1, (run.groups, run.size))
            group = ((1.0 + run.size * BASELINE_EPS) * group - BASELINE_EPS).clamp_min(0.0)
            group = group / group.sum(-1, keepdim=True).clamp_min(torch.finfo(group.dtype).tiny)
            parts.append(group.flatten(-2))
        return self.decode(torch.cat(parts, dim=-1))


__all__ = [
    "BASELINE_EPS",
    "BELIEF_MOMENT_DIM",
    "CODE_GROUP_DIM",
    "CODE_GROUP_NAMES",
    "CODE_RUN",
    "CODE_RUNS",
    "POSITION_CODE_DIM",
    "SHIP_CODE_DIM",
    "CodeRun",
    "ShipStateCodec",
]
