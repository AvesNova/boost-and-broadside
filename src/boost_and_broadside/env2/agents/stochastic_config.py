from __future__ import annotations
from dataclasses import dataclass
from typing import ClassVar, Sequence, Tuple
import numpy as np

# Max distance on a toroidal 1024x1024 world (half-world diagonal)
_MAX_WORLD_DIST: float = float(np.sqrt(512.0**2 + 512.0**2))  # ≈ 724.1

@dataclass
class StochasticAgentConfig:
    """
    Configuration for the Stochastic Scripted Agent's brain.

    Each ramp is defined by two values:
      - A threshold range [low, high] in physical units (distance, speed, angle, ratio).
      - A probability range [prob_lo, prob_hi] that maps to the low and high ends.

    For example, boost_speed_ramp=(20, 40), boost_speed_prob=(1.0, 0.0) means:
      - speed < 20  → probability = 1.0  (definitely boost)
      - speed > 40  → probability = 0.0  (definitely don't boost)

    flat_action_sampling: Whether to sample from the joint 42-dim distribution
                          or 3 independent marginal distributions (3, 7, 2).

    # Power Ramps
    boost_speed_ramp:    Speed range where boost probability transitions.
    boost_speed_prob:    (prob at low speed, prob at high speed).

    close_range_ramp:    Distance range where reverse probability transitions.
    close_range_prob:    (prob at close, prob at far).

    # Turn Ramps
    turn_angle_ramp:       Abs angle range where turn probability transitions.
    turn_angle_prob:       (prob at small angle, prob at large angle).

    sharp_turn_angle_ramp: Abs angle range where sharp-turn probability transitions.
    sharp_turn_angle_prob: (prob at small angle, prob at large angle).

    # Shoot Ramps
    shoot_angle_ramp:    Range of (abs_angle / shoot_threshold) ratio where shoot
                         alignment probability transitions.
    shoot_angle_prob:    (prob at low ratio, prob at high ratio).

    shoot_distance_ramp: Direct distance range where shoot probability transitions.
                         shoot_distance_ramp[1] also acts as the effective max
                         shooting range for the boost metric.
    shoot_distance_prob: (prob at close distance, prob at far distance).
    """
    flat_action_sampling: bool = False

    # Power Ramps
    boost_speed_ramp: Tuple[float, float] = (20.0, 40.0)
    boost_speed_prob: Tuple[float, float] = (1.0, 0.0)

    close_range_ramp: Tuple[float, float] = (30.0, 50.0)
    close_range_prob: Tuple[float, float] = (1.0, 0.0)

    # Turn Ramps
    turn_angle_ramp: Tuple[float, float] = (np.deg2rad(2.0), np.deg2rad(5.0))
    turn_angle_prob: Tuple[float, float] = (0.0, 1.0)

    sharp_turn_angle_ramp: Tuple[float, float] = (np.deg2rad(10.0), np.deg2rad(20.0))
    sharp_turn_angle_prob: Tuple[float, float] = (0.0, 1.0)

    # Shoot Ramps
    # Angle: ratio = abs_angle / shoot_threshold (threshold derived from ShipConfig.collision_radius)
    shoot_angle_ramp: Tuple[float, float] = (0.8, 1.2)
    shoot_angle_prob: Tuple[float, float] = (1.0, 0.0)

    # Distance: direct world-unit distances
    shoot_distance_ramp: Tuple[float, float] = (400.0, 700.0)
    shoot_distance_prob: Tuple[float, float] = (1.0, 0.0)

    # ---------------------------------------------------------------------------
    # Flat-vector interface for hyperparameter search
    # ---------------------------------------------------------------------------
    # One (lo, hi) bound per field. Both elements of a Tuple[float, float] field
    # are scaled using the same bound. All input values are expected in [0, 1].
    PARAM_BOUNDS: ClassVar[list[tuple[float, float]]] = [
        (0.0, 180.0),         # boost_speed_ramp       — speed units
        (0.0, 1.0),           # boost_speed_prob        — probability
        (0.0, _MAX_WORLD_DIST),  # close_range_ramp    — world units
        (0.0, 1.0),           # close_range_prob        — probability
        (0.0, np.pi),         # turn_angle_ramp         — radians
        (0.0, 1.0),           # turn_angle_prob         — probability
        (0.0, np.pi),         # sharp_turn_angle_ramp   — radians
        (0.0, 1.0),           # sharp_turn_angle_prob   — probability
        (0.0, 3.0),           # shoot_angle_ramp        — ratio
        (0.0, 1.0),           # shoot_angle_prob        — probability
        (0.0, _MAX_WORLD_DIST),  # shoot_distance_ramp  — world units
        (0.0, 1.0),           # shoot_distance_prob     — probability
    ]
    # Vector length = 2 * len(PARAM_BOUNDS) = 24

    @classmethod
    def from_vector(cls, v: Sequence[float], flat_action_sampling: bool = False) -> "StochasticAgentConfig":
        """
        Construct a StochasticAgentConfig from a 24-element flat vector.
        All values in v must be in [0, 1]; they are scaled to physical units
        using PARAM_BOUNDS.

        Vector layout (pairs in field order):
          [0,1]   boost_speed_ramp      [2,3]   boost_speed_prob
          [4,5]   close_range_ramp      [6,7]   close_range_prob
          [8,9]   turn_angle_ramp       [10,11] turn_angle_prob
          [12,13] sharp_turn_angle_ramp [14,15] sharp_turn_angle_prob
          [16,17] shoot_angle_ramp      [18,19] shoot_angle_prob
          [20,21] shoot_distance_ramp   [22,23] shoot_distance_prob
        """
        expected = 2 * len(cls.PARAM_BOUNDS)
        if len(v) != expected:
            raise ValueError(f"Expected vector of length {expected}, got {len(v)}")

        def scale(field_idx: int, x: float) -> float:
            lo, hi = cls.PARAM_BOUNDS[field_idx]
            return lo + float(x) * (hi - lo)

        def pair(field_idx: int, vi: int) -> Tuple[float, float]:
            return (scale(field_idx, v[vi]), scale(field_idx, v[vi + 1]))

        return cls(
            flat_action_sampling=flat_action_sampling,
            boost_speed_ramp=pair(0, 0),
            boost_speed_prob=pair(1, 2),
            close_range_ramp=pair(2, 4),
            close_range_prob=pair(3, 6),
            turn_angle_ramp=pair(4, 8),
            turn_angle_prob=pair(5, 10),
            sharp_turn_angle_ramp=pair(6, 12),
            sharp_turn_angle_prob=pair(7, 14),
            shoot_angle_ramp=pair(8, 16),
            shoot_angle_prob=pair(9, 18),
            shoot_distance_ramp=pair(10, 20),
            shoot_distance_prob=pair(11, 22),
        )

    @classmethod
    def default_vector(cls) -> list[float]:
        """
        Returns the 24-element [0, 1]-normalized vector corresponding to the
        default StochasticAgentConfig. Useful as a starting point for search.
        """
        cfg = cls()
        raw = [
            *cfg.boost_speed_ramp,      *cfg.boost_speed_prob,
            *cfg.close_range_ramp,      *cfg.close_range_prob,
            *cfg.turn_angle_ramp,       *cfg.turn_angle_prob,
            *cfg.sharp_turn_angle_ramp, *cfg.sharp_turn_angle_prob,
            *cfg.shoot_angle_ramp,      *cfg.shoot_angle_prob,
            *cfg.shoot_distance_ramp,   *cfg.shoot_distance_prob,
        ]
        # Expand bounds: each field bound applies to both elements of the tuple
        expanded_bounds = [b for b in cls.PARAM_BOUNDS for _ in range(2)]
        return [(r - lo) / (hi - lo) for r, (lo, hi) in zip(raw, expanded_bounds)]
