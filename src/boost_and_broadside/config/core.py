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
    field_radius_max: float = 490.0
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
    # Truncation horizon in *physics ticks*, so it is a fixed span of game time
    # regardless of action_repeat.
    max_episode_steps: int | None  # None disables time-based truncation
    num_fields: int = 0  # static refractive fields per env (0 = ambient-only baseline)
    single_team: bool = False  # all ships share one randomly-chosen team id (no opponents)
    # Physics ticks each chosen action is held for. Physics always runs at
    # ShipConfig.dt; this decides only how often the policy gets to change its
    # mind, so collision and projectile integration stay exact.
    #
    # dt is 1/60, and at repeat 1 the policy re-decides every 16.7 ms against a
    # firing cooldown of 6 ticks and a full 360-degree turn of 78-138 ticks —
    # far finer than the plant can respond to, which makes consecutive decisions
    # near-duplicates. Repeat 2 gives 30 Hz: 3 decisions per cooldown and 39-69
    # per full turn, still ample authority, for half the tokens per second of
    # game time.
    #
    # Coarsening is not free — it costs combat effectiveness monotonically — so
    # the chosen value is a measured trade, not a default. See profiles/rl.py.
    # Discounts encode horizons in seconds, so moving this means re-deriving
    # gamma and lambda as g ** (rate_old / rate_new), not reusing the number.
    action_repeat: int = 1
    # Fractional spread of the per-ship spawn draw for health and power: each is
    # sampled uniformly in [(1-spread)*max, max], and cooldown in [0, firing
    # cooldown]. 0 spawns every ship at full resources, which correlates health
    # tightly with elapsed episode time and means damaged states are only ever
    # reached by playing into them.
    spawn_resource_spread: float = 0.0

    def __post_init__(self) -> None:
        if self.max_episode_steps is not None and self.max_episode_steps < 1:
            raise ValueError(
                f"max_episode_steps must be positive or None, got {self.max_episode_steps}"
            )
        if self.num_fields < 0:
            raise ValueError(f"num_fields must be non-negative, got {self.num_fields}")
        if self.action_repeat < 1:
            raise ValueError(f"action_repeat must be positive, got {self.action_repeat}")
        if not 0.0 <= self.spawn_resource_spread < 1.0:
            raise ValueError(
                f"spawn_resource_spread must lie in [0, 1), got {self.spawn_resource_spread}"
            )

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
    n_yemong_blocks: int  # number of Yemong blocks in the trunk
    # Sublayers inside every Yemong block. All blocks share one structure, so the
    # trunk is n_yemong_blocks repetitions of (n_spatial_per_block spatial layers
    # followed by n_temporal_per_block temporal layers). Spatial layers cost roughly
    # a quarter of a temporal layer at these token counts, so raising the spatial
    # count is the cheap way to buy relational depth.
    n_spatial_per_block: int = 1
    n_temporal_per_block: int = 1
    # Per-entity-type first projection in the encoder, with a shared second layer.
    # A field token otherwise spends most of its input width on ship-only channels
    # that are hard zeros for it. The shared output layer is what keeps both token
    # types in one latent space, which the single spatial W_qkv depends on.
    encoder_split: bool = False
    # Spatial sublayers per block that cross-attend to bullets, counted from the
    # first. 0 disables bullet observation entirely. The read must precede at
    # least one further spatial layer for a ship to reason about fire aimed at
    # *another* ship, so it is counted from the front rather than the back.
    n_bullet_cross_per_block: int = 0
    # Bullet encoder hidden width. Deliberately narrow: it runs over N*K entities
    # rather than N+M, so it, not the entity encoder, sets encoder cost.
    bullet_encoder_hidden: int = 64
    # Recompute each Yemong block's activations during the PPO backward pass instead
    # of storing them (torch.utils.checkpoint). Trades ~one extra forward per block
    # in backward for activation memory that no longer scales with depth — set True
    # to fit deeper networks. Only affects the update-time re-evaluation path.
    grad_checkpoint: bool = False

    @property
    def n_hidden_layers(self) -> int:
        """Recurrent state slots in the trunk — one per temporal sublayer."""

        return self.n_yemong_blocks * self.n_temporal_per_block

    @property
    def reads_bullets(self) -> bool:
        """Whether any spatial sublayer cross-attends to bullets."""

        return self.n_bullet_cross_per_block > 0 and self.n_yemong_blocks > 0

    def __post_init__(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError(f"d_model={self.d_model} must be divisible by n_heads={self.n_heads}")
        if self.n_yemong_blocks < 0:
            raise ValueError(f"n_yemong_blocks must be >= 0, got {self.n_yemong_blocks}")
        if self.n_spatial_per_block < 0:
            raise ValueError(f"n_spatial_per_block must be >= 0, got {self.n_spatial_per_block}")
        if self.n_temporal_per_block < 0:
            raise ValueError(f"n_temporal_per_block must be >= 0, got {self.n_temporal_per_block}")
        if not 0 <= self.n_bullet_cross_per_block <= self.n_spatial_per_block:
            raise ValueError(
                "n_bullet_cross_per_block must be between 0 and n_spatial_per_block "
                f"({self.n_spatial_per_block}), got {self.n_bullet_cross_per_block}"
            )


@dataclass(frozen=True)
class RewardConfig:
    """Reward weights and geometry parameters for the decomposed critic.

    Core reward weights and geometry must be set explicitly at the call site.
    Optional behavior-shaping rewards default to disabled values.
    Reward tier scales (outcome, kill/death, damage, shaping) live in
    TrainingSchedule since they vary over the course of a run.

    Weights obey one rule: **every event pays one side exactly what it charges the
    other.** A ship's death costs its team ``death_weight`` and pays the ships that
    caused it ``death_weight`` between them; damage works the same way with
    ``damage_weight``. That fixes every ratio in the system and leaves four numbers.

    Three consequences are worth knowing, because they look like coincidences and
    are not:

    * ``enemy_field_death`` must equal ``kill_shot``. A ship killed by a field was
      shot by nobody on its fatal step, so ``kill_shot`` pays zero there and only
      ``kill_assist`` fires. Balance then needs the shortfall made up exactly, and
      the shortfall is ``kill_shot``. The same argument gives ``enemy_field_damage
      == damage_weight``. These two are the only source-split components with a
      non-zero weight: they exist to supply the offensive side of events that have
      no shooter to attribute to.
    * Killing a teammate costs the team twice. The ally is charged
      ``death_weight`` for dying and the shooter is charged ``death_weight`` for
      causing it, while the enemy is paid nothing — so friendly fire is
      structurally twice as expensive as being killed by an opponent, without a
      special case saying so.
    * The remaining ``ally_*`` and ``enemy_combat_*`` components stay at zero.
      Their events are already fully paid for by the local per-ship components and
      by damage attribution; turning them on would charge the same event twice.

    Equal weight is not equal gradient. The kill side spends its weight across two
    correlated components while the death side spends it on one, so the kill side
    delivers roughly 87% of the death side's gradient magnitude. That is expected
    and is left alone: weights state what an event means, and pressure is allowed
    to follow how coherent each signal actually is.

    Local rewards are self-only: lambda=0 for every other ship (diagonal lambda
    matrix), so the signal never propagates. Global rewards flow through the lambda
    aggregation matrix at PPO update time, so a ship's signal reaches its teammates
    and, for zero-sum components, its enemies.

    Reward tier scales (outcome, kill/death, damage, shaping) live in
    TrainingSchedule since they vary over the course of a run.

    Tier scales (applied as a multiplier on top of the derived weights; the
    authoritative component -> tier mapping is _TIER in train/rl/ppo.py):
        outcome     -> ally_win, enemy_win
        kill_death  -> kills, deaths, friendly kills
        damage      -> damage dealt and taken
        shaping     -> dense geometry rewards
    """

    # --- Event weights ---
    # Every event component derives from these four numbers, because the weights
    # are not free of one another: an event that costs one team should pay the
    # other the same. See ``component_weights`` in env/rewards.py for the algebra.
    win_weight: float  # W: to the winning team, charged to the losing one
    death_weight: float  # U: charged to a dying ship, paid to whoever caused it
    damage_weight: float  # V: charged to a damaged ship, paid to whoever dealt it
    # How U splits between "landed the finishing blow" and "contributed damage".
    # The only ratio the balance rules leave free.
    kill_shot_fraction: float
    # ``kill_payout_ratio`` breaks the balance rule for the kill tier on purpose;
    # it lives with the other defaulted fields below.

    # --- Shaping ---
    # Not events. Facing a target is a state, not something that happens to
    # somebody, so there is no opposing side to charge and no rule to apply.
    # These stay individually weighted.
    facing_weight: float  # pointing nose toward nearest enemy
    closing_speed_weight: float  # velocity component toward nearest enemy

    # --- Geometry params ---
    proximity_radius: float  # falloff radius used by FacingReward
    shoot_quality_radius: float  # engagement radius used by ShootQualityReward

    # --- Lambda configuration ---
    enemy_neg_lambda_components: frozenset[str]  # enemies get lambda=-1 (zero-sum)
    ally_zero_components: frozenset[str]  # allies get lambda=0 (enemy-perspective only)

    # A named, deliberate exception to the balance rule: the kill side is paid
    # ``kill_payout_ratio * U`` while the dying side is still charged ``U``. At
    # 1.0 the rule holds and an event pays exactly what it charges.
    #
    # It is a knob rather than a constant because the balance rule cannot express
    # it at all -- raising ``U`` raises charge and payout together, so no setting
    # of the four event weights reaches a ratio other than 1:1. The reference run
    # sat at 2:1 by accident, carrying ``kill_shot`` and ``kill_assist`` at the
    # same weight as ``combat_death``, and is the strongest policy measured;
    # every balanced run since has been more passive than it. Charging a death
    # and paying the kill equally makes an even trade worth nothing, so a policy
    # that cannot reliably win the trade declines it.
    #
    # Kill tier only. Damage and win were balanced in the reference run too, so
    # the evidence for an asymmetry is specific to this tier, and a second free
    # ratio would not be separable from the first inside one run.
    kill_payout_ratio: float = 1.0

    # --- Behaviour shaping (local, self-only; 0.0 = disabled) ---
    shoot_quality_weight: float = 0.0  # shot quality when firing
    shooting_penalty_weight: float = 0.0  # negative reward each step this ship fires
    speed_weight: float = 0.0  # penalty when speed < speed_penalty_min
    speed_penalty_min: float = 40.0  # speed threshold below which penalty is applied

    def __post_init__(self) -> None:
        if not 0.0 <= self.kill_shot_fraction <= 1.0:
            raise ValueError(
                f"kill_shot_fraction must be in [0, 1], got {self.kill_shot_fraction}"
            )
        if not np.isfinite(self.kill_payout_ratio) or self.kill_payout_ratio < 0.0:
            raise ValueError(
                f"kill_payout_ratio must be finite and non-negative, "
                f"got {self.kill_payout_ratio}"
            )
        for name in ("win_weight", "death_weight", "damage_weight"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")
