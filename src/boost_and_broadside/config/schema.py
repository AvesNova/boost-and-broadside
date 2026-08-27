"""Profile intent and complete resolved training launches.

:class:`ProfileSpec` is what a person edits; :class:`TrainConfig` is what the
trainer reads. The two still exist because a handful of values are genuinely
derived rather than chosen -- discounts normalized to the decision rate, shard
width and count, the compiled schedule -- and it is worth being able to see
which is which. Everything else passes through unchanged.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal

from boost_and_broadside.config.core import EnvConfig, ModelConfig, RewardConfig, ShipConfig
from boost_and_broadside.config.schedule_spec import TrainingScheduleSpec
from boost_and_broadside.config.training import (
    EloEvalConfig,
    FieldMapConfig,
    TrainConfig,
)

type ResolutionSource = Literal[
    "profile",
    "derived",
    "vram-cache",
    "vram-preset",
    "cli",
]

RESOLVED_CONFIG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class LaunchSizingSpec:
    """Legacy machine launch defaults excluded from the profile fingerprint."""

    rollout_tokens: int | None = None
    num_envs: int | None = None
    microbatches_per_minibatch: int | None = None
    microbatch_tokens: int | None = None


@dataclass(frozen=True)
class ProfileSpec:
    """A complete experiment intent, flat.

    This used to be six nested sub-specs -- environment, rollout, discounts,
    objective, optimizer, league -- whose only job was to be copied field by
    field into ``TrainConfig``. The grouping bought nothing: no sub-spec was ever
    passed anywhere on its own, and the nesting made adding a hyperparameter a
    four-file edit. Flat, it is two.

    Discounts are stated per 60 Hz physics tick and raised to ``action_repeat``
    during resolution, so a horizon means the same amount of game time whatever
    the decision rate. ``launch`` stays a sub-object because machine sizing is
    genuinely a different kind of value -- it is excluded from the profile
    fingerprint and may be overridden per launch.

    **Naming is load-bearing.** A field with the same name here and on
    :class:`TrainConfig` is passed through untouched, and ``resolve_profile``
    copies it by name rather than listing it. Anything the resolver *transforms*
    must therefore be named differently on each side -- ``gamma_per_tick`` for
    the stated value against ``gamma`` for the normalized one, ``schedule_spec``
    for declarative intent against ``schedule`` for the compiled closure. Giving
    a transformed value the same name on both sides would silently pass the
    intent through as if it were the result.
    """

    name: str
    ship_config: ShipConfig
    model_config: ModelConfig

    # --- Environment ---
    num_ships: int
    num_fields: int
    max_bullets: int
    max_episode_steps: int | None
    action_repeat: int
    spawn_resource_spread: float
    field_map: FieldMapConfig | None

    # --- Rollout shape ---
    logical_batch_tokens: int
    num_steps: int
    num_minibatches: int

    # --- Objective ---
    paradigm: str
    schedule_spec: TrainingScheduleSpec
    rewards: RewardConfig
    predictive_state_coef: float
    predictive_action_coef: float
    prediction_horizon: int
    predictive_mode: str

    # --- Discounts, per physics tick ---
    gamma_per_tick: float
    gae_lambda_per_tick: float
    component_gammas_per_tick: Mapping[str, float]
    component_lambdas_per_tick: Mapping[str, float]

    # --- Optimizer, scalers, budget ---
    clip_coef: float
    max_grad_norm: float
    total_timesteps: int
    return_ema_alpha: float
    return_min_span: float
    advantage_min_rms: float
    value_huber_delta: float

    # --- League and live evaluation ---
    league_size: int
    league_slots: int
    live_reference_probabilities: tuple[float, ...]
    elo_milestone_gap: float
    elo_temperature: float
    league_uniform_sampling: bool
    elo_eval: EloEvalConfig
    bc_winrate_target: float

    # --- Persistence and logging ---
    checkpoint_dir: str
    histogram_interval: int
    log_interval: int

    # --- Machine sizing, excluded from the profile fingerprint ---
    launch: LaunchSizingSpec = field(default_factory=LaunchSizingSpec)
    single_team: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "component_gammas_per_tick",
            MappingProxyType(dict(self.component_gammas_per_tick)),
        )
        object.__setattr__(
            self,
            "component_lambdas_per_tick",
            MappingProxyType(dict(self.component_lambdas_per_tick)),
        )


@dataclass(frozen=True)
class ResolvedTrainConfig:
    """Complete immutable launch description and its provenance fingerprints."""

    profile_name: str
    ship_config: ShipConfig
    model_config: ModelConfig
    train_config: TrainConfig
    schedule_spec: TrainingScheduleSpec
    value_sources: Mapping[str, ResolutionSource]

    @property
    def env_config(self) -> EnvConfig:
        """Return the primary environment without hiding the complete scale."""

        return self.train_config.scales[0].env_config
