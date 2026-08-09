"""Semantic profile intent and complete resolved training launches.

The existing :class:`TrainConfig` remains the trainer-facing shape.  The types
in this module separate what a profile means from values which can be derived
mechanically or selected for one launch.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
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

PROFILE_SCHEMA_VERSION = 1
RESOLVED_CONFIG_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class EnvironmentSpec:
    """Profile-owned environment intent before construction of ``EnvConfig``."""

    num_ships: int
    num_fields: int
    max_bullets: int
    max_episode_steps: int | None
    action_repeat: int
    spawn_resource_spread: float
    single_team: bool = False


@dataclass(frozen=True)
class RolloutSpec:
    """Semantic logical-batch intent, independent from device memory sizing."""

    logical_batch_tokens: int
    num_steps: int
    num_minibatches: int


@dataclass(frozen=True)
class LaunchSizingSpec:
    """Legacy machine launch defaults excluded from the profile fingerprint."""

    rollout_tokens: int | None = None
    num_envs: int | None = None
    microbatches_per_minibatch: int | None = None
    microbatch_tokens: int | None = None


@dataclass(frozen=True)
class DiscountSpec:
    """Discounts expressed per physics tick, then normalized to decision time."""

    gamma_per_tick: float
    gae_lambda_per_tick: float
    component_gammas_per_tick: Mapping[str, float]
    component_lambdas_per_tick: Mapping[str, float]

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
class ObjectiveSpec:
    """The objective and time-varying schedule a profile intends to optimize."""

    paradigm: str
    schedule: TrainingScheduleSpec
    rewards: RewardConfig
    next_state_coef: float
    windowed_loss_coef: float


@dataclass(frozen=True)
class OptimizerSpec:
    """Static optimizer, scaler, budget, and persistence intent."""

    clip_coef: float
    max_grad_norm: float
    total_timesteps: int
    return_ema_alpha: float
    return_min_span: float
    advantage_min_rms: float
    return_quantile_samples: int | None
    checkpoint_dir: str
    histogram_interval: int
    log_interval: int


@dataclass(frozen=True)
class LeagueSpec:
    """League and live-evaluation intent retained unchanged in S02."""

    league_size: int
    league_slots: int
    reference_ladder: tuple[tuple[float, float], ...]
    random_elo: float
    elo_milestone_gap: float
    elo_temperature: float
    league_uniform_sampling: bool
    elo_eval: EloEvalConfig
    bc_winrate_target: float


@dataclass(frozen=True)
class ProfileSpec:
    """Independent semantic experiment intent, not a launch-ready config."""

    name: str
    ship_config: ShipConfig
    model_config: ModelConfig
    environment: EnvironmentSpec
    rollout: RolloutSpec
    launch_defaults: LaunchSizingSpec
    discounts: DiscountSpec
    objective: ObjectiveSpec
    optimizer: OptimizerSpec
    league: LeagueSpec
    field_map: FieldMapConfig | None


@dataclass(frozen=True)
class ResolvedTrainConfig:
    """Complete immutable launch description and its provenance fingerprints."""

    profile_name: str
    ship_config: ShipConfig
    model_config: ModelConfig
    train_config: TrainConfig
    schedule_spec: TrainingScheduleSpec
    value_sources: Mapping[str, ResolutionSource]
    profile_fingerprint: str
    resolved_config_fingerprint: str

    @property
    def env_config(self) -> EnvConfig:
        """Return the primary environment without hiding the complete scale."""

        return self.train_config.scales[0].env_config
