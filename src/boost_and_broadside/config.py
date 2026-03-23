"""Frozen dataclass configurations for Boost and Broadside.

All configuration lives here. No magic numbers in code — every constant
that controls behavior must be a field on one of these dataclasses.

Conventions:
  - ShipConfig: physics simulation constants (has defaults — defines the game).
  - EnvConfig, ModelConfig, TrainConfig, RewardConfig: hyperparameters (NO defaults —
    every value must be set explicitly in main.py, so nothing is ever silently wrong).
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
    gravity_factor: float = 0.0 #5.0
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
    bullet_lifetime: float = 1.0       # seconds
    bullet_spread: float = 12.0        # pixels/s of noise added to velocity
    firing_cooldown: float = 0.1       # seconds

    # World
    world_size: tuple[float, float] = (1024.0, 1024.0)

    # Simulation timestep
    dt: float = 1.0 / 60.0


@dataclass(frozen=True)
class EnvConfig:
    """Environment sizing. No defaults — all values required."""
    num_ships: int          # total ships per env (both teams combined)
    max_bullets: int        # bullet ring-buffer size per ship
    max_episode_steps: int  # truncation horizon


@dataclass(frozen=True)
class ModelConfig:
    """Policy network architecture. No defaults — all values required."""
    d_model: int        # token embedding dimension
    n_heads: int        # attention heads (must divide d_model)
    n_fourier_freqs: int  # number of Fourier frequencies for position encoding

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}")
        if self.n_fourier_freqs < 1:
            raise ValueError(f"n_fourier_freqs must be >= 1, got {self.n_fourier_freqs}")


@dataclass(frozen=True)
class RewardConfig:
    """Reward shaping weights. No defaults — all values required."""
    damage_weight: float        # weight on damage-given vs damage-taken reward
    kill_weight: float          # bonus per enemy kill
    death_weight: float         # penalty per ally death
    victory_weight: float       # bonus/penalty for win/loss
    positioning_weight: float   # weight on the positioning formula reward
    positioning_radius: float   # R: influence radius for positioning reward (world units)
    facing_weight: float        # weight on facing-toward-enemy reward
    exposure_weight: float      # weight on being-in-enemy-sights penalty
    proximity_weight: float     # weight on close-to-enemy reward
    proximity_radius: float     # R: falloff radius for proximity reward (world units)
    closing_speed_weight: float  # weight on velocity-toward-enemy (closing speed) reward
    turn_rate_weight: float      # weight on angular-velocity-toward-enemy reward
    power_range_weight: float   # weight on power-in-range reward
    power_range_lo: float       # lower bound as fraction of max_power (e.g. 0.2)
    power_range_hi: float       # upper bound as fraction of max_power (e.g. 0.8)
    speed_range_weight: float   # weight on speed-in-range reward
    speed_range_lo: float       # lower speed bound (world units/s)
    speed_range_hi: float       # upper speed bound (world units/s)
    shoot_quality_weight: float # weight on shoot-quality reward/penalty
    shoot_quality_radius: float # R: effective range for shot quality evaluation (world units)
    scripted_agent_weight: float = 0.0  # weight on scripted-agent log-prob reward (0.0 = disabled)
    disabled_rewards: frozenset[str] = frozenset()  # component names to exclude entirely


@dataclass(frozen=True)
class TrainConfig:
    """PPO training hyperparameters. No defaults — all values required."""
    num_envs: int
    num_steps: int          # rollout length per environment
    num_epochs: int         # PPO update epochs per rollout
    num_minibatches: int    # minibatches per epoch (num_envs must be divisible)
    learning_rate: float
    gamma: float            # discount factor
    gae_lambda: float       # GAE lambda
    clip_coef: float        # PPO clip epsilon
    ent_coef: float         # entropy bonus coefficient
    vf_coef: float          # value loss coefficient
    max_grad_norm: float    # gradient clipping norm
    total_timesteps: int    # total environment steps before stopping
    checkpoint_interval: int = 0           # save every N updates; 0 = disabled
    checkpoint_dir: str = "checkpoints"    # directory to write .pt files
    avg_policy_warmup_steps: int = 0       # global steps before weight averaging begins; 0 = immediate
    avg_policy_opp_fraction: float = 0.5  # fraction of envs that play against avg_policy

    def __post_init__(self) -> None:
        if self.num_envs % self.num_minibatches != 0:
            raise ValueError(
                f"num_envs={self.num_envs} must be divisible by "
                f"num_minibatches={self.num_minibatches}"
            )
