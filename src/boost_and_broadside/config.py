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
    bullet_min_damage_frac: float = 0.1  # fraction of bullet_damage at a head-on hit
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
    d_model: int              # token embedding dimension
    n_heads: int              # attention heads (must divide d_model)
    n_fourier_freqs: int      # number of Fourier frequencies for position encoding
    n_transformer_blocks: int  # number of pre-norm transformer blocks before the GRU

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}")
        if self.n_fourier_freqs < 1:
            raise ValueError(f"n_fourier_freqs must be >= 1, got {self.n_fourier_freqs}")
        if self.n_transformer_blocks < 0:
            raise ValueError(f"n_transformer_blocks must be >= 0, got {self.n_transformer_blocks}")


@dataclass(frozen=True)
class RewardConfig:
    """Reward shaping weights. No defaults — all values required."""
    damage_weight: float        # weight on damage-taken per ship
    death_weight: float         # penalty for dying (own ship only)
    victory_weight: float       # bonus/penalty for win/loss (per-ship, from own team's perspective)
    enemy_neg_lambda_components: frozenset[str]  # component names where enemy ships get lambda=-1
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
class ScaleConfig:
    """One training scale: an environment config paired with a batch size.

    All scales share the same policy, optimizer, and return scaler.
    Gradients are accumulated across scales before each optimizer step.
    scales[0] in TrainConfig is the primary scale and supports scripted /
    avg-model / league opponents; scales[1:] run pure self-play.

    Args:
        env_config: Environment config for this scale — num_ships defines N.
        num_envs:   Parallel environments. Set inversely proportional to N so
                    total ships-per-update stays constant across scales.
    """
    env_config: EnvConfig
    num_envs:   int


@dataclass(frozen=True)
class TrainConfig:
    """PPO training hyperparameters. No defaults — all values required.

    scales[0] is the primary scale (scripted / avg-model / league opponents enabled).
    scales[1:] are auxiliary scales (pure self-play, gradient accumulation).
    """
    scales: tuple[ScaleConfig, ...]  # at least one entry
    num_steps: int          # rollout length per environment
    num_epochs: int         # PPO update epochs per rollout
    num_minibatches: int    # minibatches per epoch (scales[0].num_envs must be divisible)
    learning_rate: float
    gamma: float            # discount factor
    gae_lambda: float       # GAE lambda
    clip_coef: float        # PPO clip epsilon
    ent_coef: float         # entropy bonus coefficient
    vf_coef: float          # value loss coefficient
    max_grad_norm: float    # gradient clipping norm
    total_timesteps: int    # total environment steps before stopping
    return_ema_alpha: float  # EMA decay for per-component return percentile scaler
    return_min_span: float   # minimum p95-p5 span (symlog-space) — guards disabled components
    lr_warmup_steps: int = 0               # linearly ramp LR from 0 over this many global steps; 0 = disabled
    checkpoint_interval: int = 0           # save every N updates; 0 = disabled
    checkpoint_dir: str = "checkpoints"    # directory to write .pt files
    scripted_frac: float = 0.0             # fraction of primary envs using scripted opponent
    avg_model_frac: float = 0.0            # fraction of primary envs using avg-model opponent
    avg_model_min_steps: int = 0           # don't start building avg model until this many global steps
    bc_coef: float = 0.0                   # auxiliary BC loss coefficient (0 = disabled)
    bc_hold_steps: int = 0                 # hold bc_coef at full value for this many global steps
    bc_decay_steps: int = 0               # then linearly decay bc_coef to 0 over this many steps

    # --- Shaping reward schedules ---
    # Each entry: (component_name, hold_steps, decay_steps)
    # hold_steps:  weight stays at its RewardConfig value for this many global steps.
    # decay_steps: then linearly decays to 0 over this many more steps.
    # Components not listed here are held constant forever (no decay).
    shaping_schedules: tuple[tuple[str, int, int], ...] = ()

    # --- League play + ELO ---
    league_frac:              float = 0.0          # fraction of primary envs using a roster-sampled opponent
    league_size:              int   = 20           # max number of checkpoint entries in the roster
    elo_eval_games:           int   = 256          # parallel games per ELO evaluation matchup
    elo_eval_interval:        int   = 10           # run ELO eval every N updates (0 = disabled)
    elo_milestone_gap:        float = 50.0         # add checkpoint to roster every N normalized ELO points gained
    elo_k_factor:             float = 32.0         # ELO K-factor (score sensitivity per match)
    elo_temperature:          float = 200.0        # ELO bandwidth for proximity-weighted sampling
    league_uniform_sampling:  bool  = False        # if True, sample league opponents uniformly
    scripted_roster_min_steps: int  = 300_000_000  # delay adding scripted to roster until this many steps

    def __post_init__(self) -> None:
        if len(self.scales) == 0:
            raise ValueError("scales must contain at least one ScaleConfig")
        primary_envs = self.scales[0].num_envs
        if primary_envs % self.num_minibatches != 0:
            raise ValueError(
                f"scales[0].num_envs={primary_envs} must be divisible by "
                f"num_minibatches={self.num_minibatches}"
            )
        if self.scripted_frac + self.avg_model_frac + self.league_frac >= 1.0:
            raise ValueError(
                f"scripted_frac + avg_model_frac + league_frac must be < 1.0, "
                f"got {self.scripted_frac} + {self.avg_model_frac} + {self.league_frac}"
            )
