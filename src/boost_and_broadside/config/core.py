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
    """Static reward weights and geometry parameters.

    All weights default to 0.0 — only set the components you want active.
    Reward group scales (true_reward_scale, important_scale, aux_scale) live in
    TrainingSchedule since they may vary over the course of a run.

    Groups (for the group-scale multiplier):
        true_reward  → victory
        important    → death, damage, team_damage, team_death
        aux          → all others

    Static geometry params (radii, range bounds) are set once and never change.
    Set-valued fields (enemy_neg_lambda_components, disabled_rewards) are also
    fixed for the run — schedule them if future experiments require it.
    """

    # --- Individual reward weights ---
    victory_weight: float = 0.0
    death_weight: float = 0.0
    damage_weight: float = 0.0
    team_damage_weight: float = 0.0
    team_death_weight: float = 0.0
    facing_weight: float = 0.0
    exposure_weight: float = 0.0
    turn_rate_weight: float = 0.0
    closing_speed_weight: float = 0.0
    proximity_weight: float = 0.0
    positioning_weight: float = 0.0
    power_range_weight: float = 0.0
    speed_range_weight: float = 0.0
    shoot_quality_weight: float = 0.0

    # --- Geometry / threshold params ---
    positioning_radius: float = 400.0
    proximity_radius: float = 400.0
    power_range_lower: float = 0.2  # lower bound as fraction of max_power
    power_range_upper: float = 0.8  # upper bound as fraction of max_power
    speed_range_lower: float = 40.0  # lower speed bound (world units/s)
    speed_range_upper: float = 120.0  # upper speed bound (world units/s)
    shoot_quality_radius: float = 200.0

    # --- Set-valued fields ---
    enemy_neg_lambda_components: frozenset[str] = frozenset(
        {"damage", "death", "team_damage", "team_death", "victory", "exposure"}
    )
    disabled_rewards: frozenset[str] = frozenset()
