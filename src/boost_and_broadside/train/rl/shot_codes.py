"""Categorical targets for the counterfactual shot heads.

The environment emits one compact physical example per completed shot
(``env/shot_labels.py``); this module turns it into what the two heads are
trained toward, on the fly in the loss, the way the next-state code is.

Trajectory, at each queried bullet age, in the launch attitude's frame:

=================  =====  ===================================================
channel            width  code
=================  =====  ===================================================
displacement       72     the ship position code (nested bilinear 9-colour
                          levels) on a square torus wide enough that the
                          farthest flight never wraps, 8 levels
travel direction   20     circular, 5 nested levels of 4 colours
world speed        41     two-hot, zero to the fastest possible bullet
local log index    21     two-hot, the ship code's index span
=================  =====  ===================================================

Whole-shot outcome, one softmax of 19:

``[enemy hit, ally hit, enemy near 1..8, ally near 1..8, clear miss]``

Hits are hard one-hots. A miss is placed by its closest pass on the side of the
ship it passed closest to, with an HL-Gauss spread over log-distance bins that
start at the collision radius and step geometrically to six radii. Clear miss is
the next bin on that axis, shared by both sides: a pass just inside it shares
mass with clear miss, and anything at or beyond it is clear miss alone.

The two queried ages are a uniform draw and, for a hit or a near miss, the
encounter itself; a clear miss gets a second uniform draw instead.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.env import shot_labels as labels
from boost_and_broadside.train.rl.categorical_codes import (
    ATTITUDE_COLOURS,
    POSITION_COLOURS,
    CircularCode,
    PositionCode,
    ScalarCode,
)
from boost_and_broadside.train.rl.ship_codes import SCALAR_BINS

#: Queried ages per shot.
SHOT_QUERIES = 2
#: Width of one stored label row: a validity flag, then the environment's example.
SHOT_LABEL_DIM = 1 + labels.EXAMPLE_DIM

DISPLACEMENT_LEVELS = 8
DIRECTION_LEVELS = 5
SPEED_BINS = 41
#: Near-miss bins per side, from one to six collision radii.
NEAR_BINS = 8
NEAR_SPAN_RADII = 6.0
#: HL-Gauss spread, in bins.
NEAR_SIGMA_BINS = 0.75

ENEMY_HIT = 0
ALLY_HIT = 1
ENEMY_NEAR = slice(2, 2 + NEAR_BINS)
ALLY_NEAR = slice(ENEMY_NEAR.stop, ENEMY_NEAR.stop + NEAR_BINS)
CLEAR_MISS = ALLY_NEAR.stop
OUTCOME_CLASSES = CLEAR_MISS + 1

#: Trajectory code groups, in vector order: (name, groups, size).
TRAJECTORY_RUNS = (
    ("displacement", DISPLACEMENT_LEVELS, POSITION_COLOURS**2),
    ("direction", DIRECTION_LEVELS, ATTITUDE_COLOURS),
    ("speed", 1, SPEED_BINS),
    ("local_log_index", 1, SCALAR_BINS),
)
TRAJECTORY_DIM = sum(groups * size for _, groups, size in TRAJECTORY_RUNS)
TRAJECTORY_GROUPS = sum(groups for _, groups, _ in TRAJECTORY_RUNS)
TRAJECTORY_PART_NAMES = tuple(name for name, _, _ in TRAJECTORY_RUNS)


class ShotTargets(NamedTuple):
    """Everything the shot losses need for a ``(...)`` batch of ship-steps."""

    valid: torch.Tensor  # (...) bool
    ages: torch.Tensor  # (..., 2) float, queried bullet ages in ticks
    trajectory: torch.Tensor  # (..., 2, TRAJECTORY_DIM) target probabilities
    outcome: torch.Tensor  # (..., OUTCOME_CLASSES) target probabilities


@dataclass(frozen=True)
class ShotCodec:
    """Physical shot examples to categorical targets, for one ship configuration."""

    displacement: PositionCode
    direction: CircularCode
    speed: ScalarCode
    log_index: ScalarCode
    near: ScalarCode
    max_age: int
    collision_radius: float

    @classmethod
    def from_ship_config(cls, config: ShipConfig) -> ShotCodec:
        # The fastest bullet: the muzzle speed plus the fastest ship, both
        # proper speeds, in the lowest-index medium where world speed peaks.
        lowest_index = config.field_index_step**-2
        top_speed = (config.bullet_speed + config.max_speed) / lowest_index
        reach = top_speed * config.bullet_lifetime
        # A displacement code is a torus: 2.5 reaches keeps the farthest flight
        # in either direction clear of the wrap.
        period = 2.5 * reach
        index_span = 2.0 * math.log(config.field_index_step)
        near_step = math.log(NEAR_SPAN_RADII) / (NEAR_BINS - 1)
        low = math.log(config.collision_radius)
        return cls(
            displacement=PositionCode((period, period), DISPLACEMENT_LEVELS),
            direction=CircularCode(2.0 * math.pi, ATTITUDE_COLOURS, DIRECTION_LEVELS),
            speed=ScalarCode(0.0, top_speed, SPEED_BINS),
            log_index=ScalarCode(-index_span, index_span, SCALAR_BINS),
            # NEAR_BINS near bins, then clear miss as the next one.
            near=ScalarCode(low, low + NEAR_BINS * near_step, NEAR_BINS + 1),
            max_age=labels.shot_label_delay(config),
            collision_radius=config.collision_radius,
        )

    @property
    def clear_distance(self) -> float:
        """The clear-miss bin's centre: at or beyond it a shot is clear miss alone."""
        return math.exp(self.near.high)

    # ------------------------------------------------------------------
    # Targets
    # ------------------------------------------------------------------

    def targets(self, rows: torch.Tensor) -> ShotTargets:
        """Build targets from stored label rows, ``(..., SHOT_LABEL_DIM)``."""
        rows = rows.float()
        valid = rows[..., 0] > 0.5
        example = rows[..., 1:]
        hit = example[..., labels.EX_HIT]
        distance = example[..., labels.EX_CLOSEST_DSQ].clamp_min(0.0).sqrt()
        near = (hit == labels.NO_HIT) & (distance < self.clear_distance)
        encounter = (hit != labels.NO_HIT) | near
        second_age = torch.where(
            encounter, example[..., labels.EX_EVENT_TAU], example[..., labels.EX_TAU_FALLBACK]
        )
        second_snapshot = torch.where(
            encounter.unsqueeze(-1),
            example[..., labels.EX_SNAP_EVENT],
            example[..., labels.EX_SNAP_FALLBACK],
        )
        ages = torch.stack([example[..., labels.EX_TAU1], second_age], dim=-1)
        snapshots = torch.stack([example[..., labels.EX_SNAP1], second_snapshot], dim=-2)
        return ShotTargets(
            valid=valid,
            ages=ages,
            trajectory=self.trajectory_code(snapshots),
            outcome=self.outcome_code(hit, distance, example[..., labels.EX_CLOSEST_SIDE]),
        )

    def trajectory_code(self, snapshots: torch.Tensor) -> torch.Tensor:
        """Exact codes of ``(..., 5)`` snapshots, ``(..., TRAJECTORY_DIM)``."""
        displacement = snapshots[..., labels.SNAP_DX : labels.SNAP_DY + 1]
        vx = snapshots[..., labels.SNAP_VX]
        vy = snapshots[..., labels.SNAP_VY]
        zero = torch.zeros_like(vx)
        log_index = snapshots[..., labels.SNAP_INDEX].clamp_min(1e-6).log()
        parts = [
            self.displacement.encode(displacement, zero).flatten(-2),
            self.direction.encode(torch.atan2(vy, vx), zero).flatten(-2),
            self.speed.sharp(torch.hypot(vx, vy)),
            self.log_index.sharp(log_index),
        ]
        return torch.cat(parts, dim=-1)

    def outcome_code(
        self, hit: torch.Tensor, distance: torch.Tensor, side: torch.Tensor
    ) -> torch.Tensor:
        """``(..., OUTCOME_CLASSES)`` target from hit, closest pass and its side."""
        log_distance = distance.clamp_min(self.collision_radius).log()
        sigma = torch.full_like(log_distance, NEAR_SIGMA_BINS * self.near.spacing)
        spread = self.near.encode(log_distance, sigma)  # (..., NEAR_BINS + 1)
        clear_only = F.one_hot(torch.full_like(hit, NEAR_BINS, dtype=torch.long), NEAR_BINS + 1)
        spread = torch.where(
            (distance >= self.clear_distance).unsqueeze(-1), clear_only.to(spread.dtype), spread
        )
        enemy_side = (side != labels.ALLY).unsqueeze(-1).to(spread.dtype)
        miss = torch.cat(
            [
                torch.zeros_like(spread[..., :2]),
                spread[..., :NEAR_BINS] * enemy_side,
                spread[..., :NEAR_BINS] * (1.0 - enemy_side),
                spread[..., NEAR_BINS:],
            ],
            dim=-1,
        )
        enemy_hit = F.one_hot(torch.full_like(hit, ENEMY_HIT, dtype=torch.long), OUTCOME_CLASSES)
        ally_hit = F.one_hot(torch.full_like(hit, ALLY_HIT, dtype=torch.long), OUTCOME_CLASSES)
        return torch.where(
            (hit == labels.ENEMY).unsqueeze(-1),
            enemy_hit.to(miss.dtype),
            torch.where((hit == labels.ALLY).unsqueeze(-1), ally_hit.to(miss.dtype), miss),
        )

    # ------------------------------------------------------------------
    # Losses
    # ------------------------------------------------------------------

    @staticmethod
    def trajectory_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Per-part cross-entropy, ``(..., 4)``, each summed over its groups."""
        logits = logits.float()
        parts = []
        start = 0
        for _, groups, size in TRAJECTORY_RUNS:
            stop = start + groups * size
            group_logits = logits[..., start:stop].unflatten(-1, (groups, size))
            group_target = target[..., start:stop].unflatten(-1, (groups, size))
            ce = -(group_target * F.log_softmax(group_logits, dim=-1)).sum(-1)
            parts.append(ce.sum(-1))
            start = stop
        return torch.stack(parts, dim=-1)

    @staticmethod
    def outcome_cross_entropy(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """``(...)`` cross-entropy of the outcome softmax."""
        return -(target * F.log_softmax(logits.float(), dim=-1)).sum(-1)


def shot_time_features(ages: torch.Tensor, max_age: int, harmonics: int) -> torch.Tensor:
    """``(..., 1 + 2 * harmonics)``: normalized age and a small Fourier basis."""
    normalized = ages.float() / max_age
    k = torch.arange(1, harmonics + 1, device=ages.device, dtype=normalized.dtype)
    angle = 2.0 * math.pi * normalized.unsqueeze(-1) * k
    return torch.cat([normalized.unsqueeze(-1), angle.sin(), angle.cos()], dim=-1)
