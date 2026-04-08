"""Frozen dataclass configurations for Boost and Broadside.

All configuration lives here. No magic numbers in code — every constant
that controls behavior must be a field on one of these dataclasses.

Conventions:
  - ShipConfig: physics simulation constants (has defaults — defines the game).
  - EnvConfig, ModelConfig, TrainConfig: hyperparameters (NO defaults —
    every value must be set explicitly in main.py, so nothing is ever silently wrong).
  - PhaseConfig / TimelineConfig: scheduling — all training scalars that vary over
    time live here. Float fields are linearly interpolated between phases; bool, int,
    and frozenset fields step at phase boundaries. The base phase (step=0) must define
    every field; subsequent phases carry forward any omitted values.
"""

import numpy as np
from dataclasses import dataclass, fields as dc_fields


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
class PhaseConfig:
    """A keyframe on the training timeline.

    All fields except ``step`` default to ``None``; omitted values carry
    forward from the previous resolved phase.  The base phase (step=0) must
    define every field — ``TimelineConfig.__post_init__`` enforces this.

    Float fields are linearly interpolated between adjacent keyframes.
    Bool, int, and frozenset fields use step-function semantics (the value
    of the earlier phase is kept until the later phase begins).
    """

    step: int  # global-step at which this keyframe becomes active

    # --- Optimization ---
    learning_rate: float | None = None
    pg_coef: float | None = None  # 0.0 = pretrain (no policy gradient); 1.0 = RL
    ent_coef: float | None = None
    bc_coef: float | None = None
    vf_coef: float | None = None

    # --- Opponents ---
    scripted_frac: float | None = None   # fraction of primary envs w/ scripted opponent
    avg_model_frac: float | None = None  # fraction of primary envs w/ avg-model opponent
    league_frac: float | None = None     # fraction of primary envs w/ league opponent
    allow_avg_model_updates: bool | None = None
    allow_scripted_in_roster: bool | None = None

    # --- League / Checkpointing ---
    elo_eval_games: int | None = None      # parallel games per ELO evaluation matchup
    elo_eval_interval: int | None = None   # run ELO eval every N updates (0 = disabled)
    checkpoint_interval: int | None = None  # save every N updates (0 = disabled)

    # --- Reward Group Scales ---
    # Effective weight = group_scale * individual_weight.
    # Groups: true_reward → victory; important → death, damage; aux → all others.
    true_reward_scale: float | None = None
    important_scale: float | None = None
    aux_scale: float | None = None

    # --- Reward Individual Weights ---
    victory_weight: float | None = None
    death_weight: float | None = None
    damage_weight: float | None = None
    facing_weight: float | None = None
    exposure_weight: float | None = None
    turn_rate_weight: float | None = None
    closing_speed_weight: float | None = None
    proximity_weight: float | None = None
    positioning_weight: float | None = None
    power_range_weight: float | None = None
    speed_range_weight: float | None = None
    shoot_quality_weight: float | None = None

    # --- Reward Static Params (rarely change, but schedulable) ---
    positioning_radius: float | None = None   # influence radius for positioning (world units)
    proximity_radius: float | None = None     # falloff radius for proximity (world units)
    power_range_lo: float | None = None       # lower bound as fraction of max_power
    power_range_hi: float | None = None       # upper bound as fraction of max_power
    speed_range_lo: float | None = None       # lower speed bound (world units/s)
    speed_range_hi: float | None = None       # upper speed bound (world units/s)
    shoot_quality_radius: float | None = None  # effective range for shot quality (world units)

    # --- Set-valued Fields ---
    enemy_neg_lambda_components: frozenset[str] | None = None  # lambda=-1 for enemy ships
    disabled_rewards: frozenset[str] | None = None             # components to exclude entirely


@dataclass(frozen=True)
class TimelineConfig:
    """Resolves a sequence of PhaseConfig keyframes into a complete training timeline.

    During ``__post_init__`` every ``None`` value in a subsequent phase is
    filled in from the running resolved state, so ``self.phases`` is always a
    tuple of fully-specified ``PhaseConfig`` objects after construction.
    """

    phases: tuple[PhaseConfig, ...]

    def __post_init__(self) -> None:
        if not self.phases:
            raise ValueError("TimelineConfig requires at least one PhaseConfig.")
        first = self.phases[0]
        if first.step != 0:
            raise ValueError(
                f"The first PhaseConfig must have step=0, got step={first.step}."
            )
        current: dict[str, object] = {}
        for f in dc_fields(PhaseConfig):
            val = getattr(first, f.name)
            if val is None:
                raise ValueError(
                    f"The base PhaseConfig (step=0) must define all fields. Missing: '{f.name}'"
                )
            current[f.name] = val
        resolved: list[PhaseConfig] = [PhaseConfig(**current)]
        last_step = 0
        for i, phase in enumerate(self.phases[1:], 1):
            if phase.step <= last_step:
                raise ValueError(
                    f"Phase {i} step ({phase.step}) must be strictly greater than "
                    f"the previous phase step ({last_step})."
                )
            last_step = phase.step
            for f in dc_fields(PhaseConfig):
                val = getattr(phase, f.name)
                if val is not None:
                    current[f.name] = val
            resolved.append(PhaseConfig(**current))
        object.__setattr__(self, "phases", tuple(resolved))


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
    num_envs: int


@dataclass(frozen=True)
class TrainConfig:
    """PPO training hyperparameters. No defaults — all values required.

    scales[0] is the primary scale (scripted / avg-model / league opponents enabled).
    scales[1:] are auxiliary scales (pure self-play, gradient accumulation).

    All scalar values that vary over training — learning rate, loss coefficients,
    opponent fractions, reward weights — live in ``timeline`` rather than here.
    Fields here are static for the entire run.
    """

    scales: tuple[ScaleConfig, ...]  # at least one entry
    timeline: TimelineConfig         # all time-varying training scalars
    num_steps: int                   # rollout length per environment
    num_epochs: int                  # PPO update epochs per rollout
    num_minibatches: int             # minibatches per epoch (scales[0].num_envs must be divisible)
    gamma: float                     # discount factor
    gae_lambda: float                # GAE lambda
    clip_coef: float                 # PPO clip epsilon
    max_grad_norm: float             # gradient clipping norm
    total_timesteps: int             # total environment steps before stopping
    return_ema_alpha: float          # EMA decay for per-component return percentile scaler
    return_min_span: float           # minimum p95-p5 span (symlog-space) — guards disabled components
    checkpoint_dir: str              # directory to write .pt files
    avg_model_min_steps: int         # don't start building avg model until this many global steps

    # --- League play + ELO (static tournament parameters) ---
    league_size: int                    # max number of checkpoint entries in the roster
    elo_milestone_gap: float            # add checkpoint to roster every N normalized ELO points gained
    elo_k_factor: float                 # ELO K-factor (score sensitivity per match)
    elo_temperature: float              # ELO bandwidth for proximity-weighted sampling
    league_uniform_sampling: bool       # if True, sample league opponents uniformly
    scripted_roster_min_steps: int      # delay adding scripted to roster until this many steps

    def __post_init__(self) -> None:
        if len(self.scales) == 0:
            raise ValueError("scales must contain at least one ScaleConfig")
        primary_envs = self.scales[0].num_envs
        if primary_envs % self.num_minibatches != 0:
            raise ValueError(
                f"scales[0].num_envs={primary_envs} must be divisible by "
                f"num_minibatches={self.num_minibatches}"
            )
