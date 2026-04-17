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
    prob_offset: float = 0.1

    # Power Ramps
    boost_speed_ramp: Tuple[float, float] = (16, 32)
    boost_speed_prob: Tuple[float, float] = (0.68, 0.04)

    close_range_ramp: Tuple[float, float] = (300, 500)
    close_range_prob: Tuple[float, float] = (0.067, 0.95)

    # Turn Ramps
    turn_angle_ramp: Tuple[float, float] = (0.03, 0.12)
    turn_angle_prob: Tuple[float, float] = (0.068, 0.95)

    sharp_turn_angle_ramp: Tuple[float, float] = (0.24, 0.42)
    sharp_turn_angle_prob: Tuple[float, float] = (0.05, 0.95)

    # Shoot Ramps
    # Angle: ratio = abs_angle / shoot_threshold (threshold derived from ShipConfig.collision_radius)
    shoot_angle_ramp: Tuple[float, float] = (1.0, 1.5)
    shoot_angle_prob: Tuple[float, float] = (0.95, 0.05)

    # Distance: direct world-unit distances
    shoot_distance_ramp: Tuple[float, float] = (200, 500)
    shoot_distance_prob: Tuple[float, float] = (0.95, 0.05)

    # Obstacle avoidance — distances are to obstacle EDGE (center dist - radius)
    obs_avoid_dist_ramp: Tuple[float, float] = (80.0, 250.0)
    obs_avoid_turn_prob: Tuple[float, float] = (0.92, 0.0)
    obs_sharp_dist_ramp: Tuple[float, float] = (30.0, 90.0)
    obs_sharp_turn_prob: Tuple[float, float] = (0.90, 0.0)
    obs_reverse_dist_ramp: Tuple[float, float] = (30.0, 90.0)
    obs_reverse_prob: Tuple[float, float] = (0.75, 0.0)

    # ---------------------------------------------------------------------------
    # Flat-vector interface for hyperparameter search
    # ---------------------------------------------------------------------------
    # One (lo, hi) bound per field. Both elements of a Tuple[float, float] field
    # are scaled using the same bound. All input values are expected in [0, 1].
    PARAM_BOUNDS: ClassVar[list[tuple[float, float]]] = [
        (0.0, 180.0),  # boost_speed_ramp       — speed units
        (0.0, 1.0),  # boost_speed_prob        — probability
        (0.0, _MAX_WORLD_DIST),  # close_range_ramp        — world units
        (0.0, 1.0),  # close_range_prob        — probability
        (0.0, np.pi),  # turn_angle_ramp         — radians
        (0.0, 1.0),  # turn_angle_prob         — probability
        (0.0, np.pi),  # sharp_turn_angle_ramp   — radians
        (0.0, 1.0),  # sharp_turn_angle_prob   — probability
        (0.0, 3.0),  # shoot_angle_ramp        — ratio
        (0.0, 1.0),  # shoot_angle_prob        — probability
        (0.0, _MAX_WORLD_DIST),  # shoot_distance_ramp     — world units
        (0.0, 1.0),  # shoot_distance_prob     — probability
    ]
    # Vector length = 2 * len(PARAM_BOUNDS) = 24

    @classmethod
    def from_vector(
        cls, v: Sequence[float], flat_action_sampling: bool = False
    ) -> "StochasticAgentConfig":
        """
        Construct a StochasticAgentConfig from a 24-element flat vector.
        All values in v must be in [0, 1]; they are scaled to physical units
        using PARAM_BOUNDS.
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
        """Returns the 24-element [0, 1]-normalized vector for the default config."""
        cfg = cls()
        raw = [
            *cfg.boost_speed_ramp,
            *cfg.boost_speed_prob,
            *cfg.close_range_ramp,
            *cfg.close_range_prob,
            *cfg.turn_angle_ramp,
            *cfg.turn_angle_prob,
            *cfg.sharp_turn_angle_ramp,
            *cfg.sharp_turn_angle_prob,
            *cfg.shoot_angle_ramp,
            *cfg.shoot_angle_prob,
            *cfg.shoot_distance_ramp,
            *cfg.shoot_distance_prob,
        ]
        expanded_bounds = [b for b in cls.PARAM_BOUNDS for _ in range(2)]
        return [(r - lo) / (hi - lo) for r, (lo, hi) in zip(raw, expanded_bounds)]
