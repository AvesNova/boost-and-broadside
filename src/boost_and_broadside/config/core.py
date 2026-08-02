"""Core configuration dataclasses: physics, environment, model, and reward shapes.

ShipConfig has defaults — it defines the reference game. Everything else has
no defaults; all values must be set explicitly so nothing is ever silently wrong.
"""

from dataclasses import dataclass
from enum import IntEnum

import numpy as np


class RefractiveIndexLevel(IntEnum):
    """Absolute refractive-index exponent relative to ambient ``n=1``."""

    VERY_LOW = -2
    LOW = -1
    AMBIENT = 0
    HIGH = 1
    VERY_HIGH = 2


class InterfaceDamageLevel(IntEnum):
    """Interface damage multiplier; independent from refractive index."""

    NONE = 0
    STANDARD = 1
    SEVERE = 2


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
    reverse_thrust: float = -80.0

    # Gravity (attraction between fast-moving ships)
    gravity_factor: float = 0.0  # 5.0
    gravity_eps: float = 10000.0

    # Static refractive fields. ``transition_width`` is the complete interface
    # band, extending half the width to either side of the nominal radius.
    field_index_step: float = float(np.sqrt(2.0))  # levels span n=1/2 through n=2
    field_interface_damage: float = 10.0
    field_radius_min: float = 30.0
    field_radius_max: float = 160.0
    field_transition_width_min: float = 40.0
    field_transition_width_max: float = 40.0
    # Fixed substeps keep the hot path static-shaped and make interface
    # total-variation damage robust at the configured ship speeds.
    field_integrator: str = "midpoint"  # "two_step" or "midpoint"
    field_integration_substeps: int = 2

    # Power exchange: E = ½n²|v|² + power_speed_constant*power is conserved
    # across thrust/reverse (ignoring drag and explicit passive regeneration).
    # Calibrated so boost at cruise speed (~100 px/s) drains ~40 power/s.
    power_speed_constant: float = 200.0
    # Passive power regen added every step regardless of action (like a slow engine recharge).
    passive_power_gain: float = 10.0

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
    bullet_drag_coeff: float = 8e-4  # quadratic drag, integrated exactly like ship drag
    bullet_field_integrator: str = "two_step"  # "two_step" or ship-quality "midpoint"
    bullet_field_integration_substeps: int = 2
    # Bullet damage potential lost per point of interface damage crossed.
    # At 0.1, a 10-damage interface reduces a 10-damage bullet to 9 damage.
    bullet_field_damage_scale: float = 0.1

    # World
    world_size: tuple[float, float] = (1024.0, 1024.0)

    # Simulation timestep
    dt: float = 1.0 / 60.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.field_index_step) or self.field_index_step <= 1.0:
            raise ValueError("field_index_step must be greater than 1")
        if not np.isfinite(self.field_interface_damage) or self.field_interface_damage < 0.0:
            raise ValueError("field_interface_damage must be non-negative")
        if len(self.world_size) != 2 or not all(
            np.isfinite(side) and side > 0.0 for side in self.world_size
        ):
            raise ValueError("world_size must contain two positive finite dimensions")
        if not 0.0 < self.field_radius_min <= self.field_radius_max:
            raise ValueError("field radii must satisfy 0 < min <= max")
        if not 0.0 < self.field_transition_width_min <= self.field_transition_width_max:
            raise ValueError("field transition widths must satisfy 0 < min <= max")
        if self.field_radius_min <= 0.5 * self.field_transition_width_max:
            raise ValueError(
                "field_radius_min must exceed field_transition_width_max/2 so every "
                "field has a non-empty flat core"
            )
        if self.field_integration_substeps < 1:
            raise ValueError("field_integration_substeps must be positive")
        if self.field_integrator not in {"two_step", "midpoint"}:
            raise ValueError("field_integrator must be 'two_step' or 'midpoint'")
        if not np.isfinite(self.bullet_drag_coeff) or self.bullet_drag_coeff < 0.0:
            raise ValueError("bullet_drag_coeff must be non-negative")
        if self.bullet_field_integrator not in {"two_step", "midpoint"}:
            raise ValueError("bullet_field_integrator must be 'two_step' or 'midpoint'")
        if (
            self.bullet_field_integration_substeps < 2
            or self.bullet_field_integration_substeps % 2 != 0
        ):
            raise ValueError("bullet_field_integration_substeps must be a positive even integer")
        if not np.isfinite(self.bullet_field_damage_scale) or self.bullet_field_damage_scale < 0.0:
            raise ValueError("bullet_field_damage_scale must be non-negative")

        # A toroidal circle must stay strictly below the nearest antipode. At or
        # beyond half the shorter world dimension its radial contour is ambiguous.
        safe_limit = 0.5 * min(self.world_size)
        outer_extent = self.field_radius_max + 0.5 * self.field_transition_width_max
        if outer_extent >= safe_limit:
            raise ValueError(
                "field_radius_max + field_transition_width_max/2 must be below "
                f"the toroidal limit {safe_limit:g}, got {outer_extent:g}"
            )


@dataclass(frozen=True)
class EnvConfig:
    """Environment sizing."""

    num_ships: int  # total ships per env (both teams combined)
    max_bullets: int  # bullet ring-buffer size per ship (0 = no bullets, skips all bullet physics)
    max_episode_steps: int | None  # truncation horizon; None disables time-based truncation
    num_fields: int = 0  # static refractive fields per env (0 = ambient-only baseline)
    single_team: bool = False  # all ships share one randomly-chosen team id (no opponents)

    def __post_init__(self) -> None:
        if self.max_episode_steps is not None and self.max_episode_steps < 1:
            raise ValueError(
                f"max_episode_steps must be positive or None, got {self.max_episode_steps}"
            )
        if self.num_fields < 0:
            raise ValueError(f"num_fields must be non-negative, got {self.num_fields}")

    @property
    def num_obstacles(self) -> int:
        """Deprecated read-only alias for pre-field integrations."""

        return self.num_fields


@dataclass(frozen=True)
class ModelConfig:
    """Policy network architecture. No defaults — all values required.

    Fourier frequency counts per feature are set by the FeatureCoordinator
    in train/rl/features.py, not here.
    """

    d_model: int  # token embedding dimension
    n_heads: int  # attention heads (must divide d_model)
    n_transformer_blocks: int  # number of pre-norm transformer blocks before the GRU
    # Recompute each Yemong block's activations during the PPO backward pass instead
    # of storing them (torch.utils.checkpoint). Trades ~one extra forward per block
    # in backward for activation memory that no longer scales with depth — set True
    # to fit deeper networks. Only affects the update-time re-evaluation path.
    grad_checkpoint: bool = False

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}")
        if self.n_transformer_blocks < 0:
            raise ValueError(f"n_transformer_blocks must be >= 0, got {self.n_transformer_blocks}")


@dataclass(frozen=True)
class RewardConfig:
    """Reward weights and geometry parameters for the decomposed critic.

    Core reward weights and geometry must be set explicitly at the call site.
    Optional behavior-shaping rewards default to disabled values.
    Reward group scales (true_reward_scale, global_scale, local_scale) live in
    TrainingSchedule since they vary over the course of a run.

    Global rewards flow through the lambda aggregation matrix at PPO update time,
    so a ship's signal can affect its teammates' and enemies' advantages:
        ally_*   → each ship reports its own outcome; allies see it via lambda=1,
                   enemies see it via lambda=-1 (zero-sum) if listed in
                   enemy_neg_lambda_components.
        enemy_*  → same signal, opposite team's perspective; combined with
                   ally_zero_components to avoid double-counting on the ally side.

    Local rewards are self-only: lambda=0 for every other ship (diagonal lambda
    matrix), so the signal never propagates. Each ship is the sole recipient of
    its own reward.

    Group scales (applied as a multiplier on top of individual weights; the
    authoritative component → group mapping is _GROUP in train/rl/ppo.py):
        true_reward  → win components (ally_win, enemy_win)
        global       → team outcome rewards (ally/enemy damage and death)
        local        → self-only per-ship rewards (shaping, kill credit,
                       per-ship damage/death)
    """

    # --- Global outcome rewards (lambda-aggregated across ships) ---
    ally_damage_weight: float  # damage taken by this ship (negative; enemies zero-sum via lambda)
    enemy_damage_weight: float  # same signal, enemy-team perspective (pair with ally_damage)
    ally_death_weight: float  # -1 on death of this ship
    enemy_death_weight: float  # same signal, enemy-team perspective (pair with ally_death)
    ally_win_weight: float  # +1 when this ship's team wins
    enemy_win_weight: float  # same signal, enemy-team perspective (pair with ally_win)

    # --- Local per-ship rewards (self-only, lambda=0 for all other ships) ---
    facing_weight: float  # pointing nose toward nearest enemy (shaping)
    closing_speed_weight: float  # velocity component toward nearest enemy (shaping)
    shoot_quality_weight: float  # shot quality when firing (shaping)
    kill_shot_weight: float  # proportional share of +1.0 per kill, weighted by step damage
    kill_assist_weight: float  # proportional share of +1.0 per kill, weighted by episode damage
    damage_taken_weight: float  # damage received by this ship this step (negative reward)
    damage_dealt_enemy_weight: float  # damage dealt to enemies this step (positive reward)
    damage_dealt_ally_weight: float  # damage dealt to allies this step (friendly-fire penalty)
    death_weight: float  # -1 on the step this ship dies; fires via just_died, not alive mask

    # --- Geometry params ---
    proximity_radius: float  # falloff radius used by FacingReward
    shoot_quality_radius: float  # engagement radius used by ShootQualityReward

    # --- Lambda configuration ---
    enemy_neg_lambda_components: frozenset[str]  # enemies get lambda=-1 (zero-sum)
    ally_zero_components: frozenset[str]  # allies get lambda=0 (enemy-perspective only)

    # --- Behaviour shaping (local, self-only; 0.0 = disabled) ---
    shooting_penalty_weight: float = 0.0  # negative reward each step this ship fires
    speed_weight: float = 0.0  # penalty when speed < speed_penalty_min
    speed_penalty_min: float = 40.0  # speed threshold below which penalty is applied
