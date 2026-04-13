"""Core configuration dataclasses: physics, environment, model, and reward shapes.

ShipConfig has defaults — it defines the reference game. Everything else has
no defaults; all values must be set explicitly so nothing is ever silently wrong.
"""

import numpy as np
from dataclasses import dataclass


@dataclass(frozen=True)
class ShipConfig:
    """Physics and game-mechanics constants.

    These define the ship model. Defaults are the reference game values.
    Override only when experimenting with alternate physics.
    """

    # Collision
    collision_radius: float = 10.0

    # Health / Power caps
    max_health: float = 100.0
    max_power: float = 100.0

    # Spawn settings
    random_speed: bool = False
    min_speed: float = 1.0
    max_speed: float = 180.0
    default_speed: float = 100.0

    # Thrust
    base_thrust: float = 8.0
    boost_thrust: float = 80.0
    reverse_thrust: float = -10.0

    # Gravity (attraction between fast-moving ships)
    gravity_factor: float = 0.0  # 5.0
    gravity_eps: float = 10000.0

    # Power regeneration / consumption rates (per second)
    base_power_gain: float = 10.0
    boost_power_gain: float = -40.0
    reverse_power_gain: float = 20.0

    # Drag and lift coefficients
    no_turn_drag_coeff: float = 8e-4
    normal_turn_drag_coeff: float = 1.2e-3
    normal_turn_lift_coeff: float = 15e-3
    sharp_turn_drag_coeff: float = 5.0e-3
    sharp_turn_lift_coeff: float = 27e-3

    # Maneuverability angles (radians)
    normal_turn_angle: float = float(np.deg2rad(5.0))
    sharp_turn_angle: float = float(np.deg2rad(15.0))

    # Bullet parameters
    bullet_speed: float = 500.0
    bullet_energy_cost: float = 3.0
    bullet_damage: float = 10.0
    bullet_min_damage_frac: float = 0.1  # fraction of bullet_damage at a head-on hit
    bullet_lifetime: float = 1.0  # seconds
    bullet_spread: float = 12.0  # pixels/s of noise added to velocity
    firing_cooldown: float = 0.1  # seconds

    # World
    world_size: tuple[float, float] = (1024.0, 1024.0)

    # Simulation timestep
    dt: float = 1.0 / 60.0


@dataclass(frozen=True)
class EnvConfig:
    """Environment sizing. No defaults — all values required."""

    num_ships: int  # total ships per env (both teams combined)
    max_bullets: int  # bullet ring-buffer size per ship
    max_episode_steps: int  # truncation horizon


@dataclass(frozen=True)
class ModelConfig:
    """Policy network architecture. No defaults — all values required."""

    d_model: int  # token embedding dimension
    n_heads: int  # attention heads (must divide d_model)
    n_fourier_freqs: int  # number of Fourier frequencies for position encoding
    n_transformer_blocks: int  # number of pre-norm transformer blocks before the GRU

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}"
            )
        if self.n_fourier_freqs < 1:
            raise ValueError(
                f"n_fourier_freqs must be >= 1, got {self.n_fourier_freqs}"
            )
        if self.n_transformer_blocks < 0:
            raise ValueError(
                f"n_transformer_blocks must be >= 0, got {self.n_transformer_blocks}"
            )


@dataclass(frozen=True)
class RewardConfig:
    """Reward weights and geometry parameters for the 9-component critic.

    No default values — all fields must be set explicitly at the call site.
    Reward group scales (true_reward_scale, important_scale, aux_scale) live in
    TrainingSchedule since they vary over the course of a run.

    Groups (for the group-scale multiplier):
        true_reward  → ally_win, enemy_win
        important    → ally_damage, enemy_damage, ally_death, enemy_death,
                       facing, closing_speed, shoot_quality

    Lambda matrix configuration:
        enemy_neg_lambda_components: enemy ships get lambda=-1 (zero-sum accounting).
        ally_zero_components:        same-team ships get lambda=0 (enemy-perspective only).
        Components in ally_zero_components should also be in enemy_neg_lambda_components.
    """

    # --- Outcome reward weights ---
    ally_damage_weight: float
    enemy_damage_weight: float
    ally_death_weight: float
    enemy_death_weight: float
    ally_win_weight: float
    enemy_win_weight: float

    # --- Shaping reward weights ---
    facing_weight: float
    closing_speed_weight: float
    shoot_quality_weight: float

    # --- Geometry params (required, no defaults) ---
    proximity_radius: float  # used by FacingReward
    shoot_quality_radius: float  # used by ShootQualityReward

    # --- Lambda configuration ---
    enemy_neg_lambda_components: frozenset[str]
    ally_zero_components: frozenset[str]
