"""Resolve independent profile intent into validated trainer configurations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from decimal import Decimal
from types import MappingProxyType
from typing import Any

from boost_and_broadside.config.core import EnvConfig
from boost_and_broadside.config.fingerprint import canonical_data, fingerprint
from boost_and_broadside.config.schema import (
    PROFILE_SCHEMA_VERSION,
    RESOLVED_CONFIG_SCHEMA_VERSION,
    ProfileSpec,
    ResolutionSource,
    ResolvedTrainConfig,
)
from boost_and_broadside.config.training import ScaleConfig, TrainConfig


@dataclass(frozen=True)
class LaunchOverrides:
    """Deliberately exposed per-launch sizing overrides.

    These are applied after the profile's legacy launch preset and all
    derivations.  The fully overridden result is then validated.
    """

    num_envs: int | None = None
    microbatch_tokens: int | None = None


def derive_aligned_num_envs(
    *,
    rollout_tokens: int,
    entity_tokens: int,
    num_steps: int,
    num_minibatches: int,
) -> int:
    """Fit a rollout-token target and align environments to minibatches."""

    for name, value in (
        ("rollout_tokens", rollout_tokens),
        ("entity_tokens", entity_tokens),
        ("num_steps", num_steps),
        ("num_minibatches", num_minibatches),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive, got {value}")
    unaligned = rollout_tokens // entity_tokens // num_steps
    aligned = unaligned // num_minibatches * num_minibatches
    if aligned < 1:
        raise ValueError(
            "rollout token target is too small for one minibatch-aligned environment group"
        )
    return aligned


def derive_time_normalized_value(per_tick_value: float, *, action_repeat: int) -> float:
    """Convert one-physics-tick discount or lambda to one policy decision."""

    if not math.isfinite(per_tick_value) or not 0.0 < per_tick_value <= 1.0:
        raise ValueError(f"per_tick_value must be finite and in (0, 1], got {per_tick_value}")
    if action_repeat < 1:
        raise ValueError(f"action_repeat must be positive, got {action_repeat}")
    # Decimal exponentiation preserves the intentionally stated base values
    # exactly when converted back to float (for example 0.975² -> 0.950625),
    # avoiding a one-ULP drift from the S01 resolved configuration.
    return float(Decimal(str(per_tick_value)) ** action_repeat)


def _profile_fingerprint_payload(profile: ProfileSpec) -> dict[str, Any]:
    payload = canonical_data(profile)
    del payload["launch_defaults"]
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profile": payload,
    }


def _resolved_fingerprint_payload(
    profile: ProfileSpec,
    *,
    env_config: EnvConfig,
    train_config: TrainConfig,
) -> dict[str, Any]:
    train_payload = canonical_data(train_config)
    # Runtime closure names are characterization evidence, not schema.  The
    # resolved fingerprint binds the same final schedule through its declarative
    # intent instead.
    train_payload["schedule"] = canonical_data(profile.objective.schedule)
    return {
        "schema_version": RESOLVED_CONFIG_SCHEMA_VERSION,
        "profile_name": profile.name,
        "ship_config": canonical_data(profile.ship_config),
        "model_config": canonical_data(profile.model_config),
        "env_config": canonical_data(env_config),
        "train_config": train_payload,
    }


def _leaf_paths(value: Any, prefix: str = "") -> set[str]:
    if isinstance(value, dict):
        if not value:
            return {prefix}
        paths: set[str] = set()
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            paths.update(_leaf_paths(item, path))
        return paths
    if isinstance(value, list):
        if not value:
            return {prefix}
        paths = set()
        for index, item in enumerate(value):
            path = f"{prefix}.{index}" if prefix else str(index)
            paths.update(_leaf_paths(item, path))
        return paths
    return {prefix}


def _source_map(
    config_payload: dict[str, Any],
    *,
    num_envs_source: ResolutionSource,
    microbatch_source: ResolutionSource,
) -> dict[str, ResolutionSource]:
    sources: dict[str, ResolutionSource] = {
        path: "profile" for path in _leaf_paths(config_payload)
    }
    derived_prefixes = (
        "train_config.gamma",
        "train_config.gae_lambda",
        "train_config.component_gammas",
        "train_config.component_lambdas",
        "train_config.rollouts_per_update",
    )
    for path in sources:
        if path.startswith(derived_prefixes):
            sources[path] = "derived"
        if path.startswith("train_config.scales.0.num_envs"):
            sources[path] = num_envs_source
        if path.startswith("train_config.microbatch_tokens"):
            sources[path] = microbatch_source
    return sources


def _validate_profile(profile: ProfileSpec) -> None:
    if not profile.name or profile.name != profile.name.strip():
        raise ValueError(f"profile name must be non-empty and trimmed, got {profile.name!r}")
    if profile.environment.num_ships < 1:
        raise ValueError("environment.num_ships must be positive")
    if profile.environment.num_fields < 0:
        raise ValueError("environment.num_fields must be non-negative")
    if profile.environment.max_bullets < 0:
        raise ValueError("environment.max_bullets must be non-negative")
    if profile.rollout.logical_batch_tokens < 1:
        raise ValueError("rollout.logical_batch_tokens must be positive")
    if profile.rollout.num_steps < 1:
        raise ValueError("rollout.num_steps must be positive")
    if profile.rollout.num_minibatches < 1:
        raise ValueError("rollout.num_minibatches must be positive")
    launch = profile.launch_defaults
    if (launch.rollout_tokens is None) == (launch.num_envs is None):
        raise ValueError("launch defaults must define exactly one of rollout_tokens or num_envs")
    if launch.rollout_tokens is not None and launch.rollout_tokens < 1:
        raise ValueError(f"rollout_tokens must be positive, got {launch.rollout_tokens}")
    if launch.num_envs is not None and launch.num_envs < 1:
        raise ValueError(f"num_envs must be positive, got {launch.num_envs}")
    if launch.microbatches_per_minibatch is not None and launch.microbatch_tokens is not None:
        raise ValueError(
            "launch defaults cannot define both microbatches_per_minibatch and microbatch_tokens"
        )
    if launch.microbatches_per_minibatch is not None and launch.microbatches_per_minibatch < 1:
        raise ValueError("microbatches_per_minibatch must be positive")
    if launch.microbatch_tokens is not None and launch.microbatch_tokens < 1:
        raise ValueError("microbatch_tokens must be positive")
    optimizer = profile.optimizer
    if optimizer.total_timesteps < 1:
        raise ValueError("optimizer.total_timesteps must be positive")
    for name, value in (
        ("optimizer.clip_coef", optimizer.clip_coef),
        ("optimizer.max_grad_norm", optimizer.max_grad_norm),
        ("optimizer.return_ema_alpha", optimizer.return_ema_alpha),
        ("optimizer.return_min_span", optimizer.return_min_span),
        ("optimizer.advantage_min_rms", optimizer.advantage_min_rms),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite, got {value}")
    if optimizer.clip_coef >= 1.0:
        raise ValueError("optimizer.clip_coef must be below 1")
    if optimizer.return_ema_alpha > 1.0:
        raise ValueError("optimizer.return_ema_alpha must be at most 1")
    if optimizer.return_quantile_samples is not None and optimizer.return_quantile_samples < 1:
        raise ValueError("optimizer.return_quantile_samples must be positive or None")
    if optimizer.histogram_interval < 1 or optimizer.log_interval < 1:
        raise ValueError("optimizer logging intervals must be positive")
    if not optimizer.checkpoint_dir:
        raise ValueError("optimizer.checkpoint_dir must be non-empty")
    league = profile.league
    if league.league_size < 1 or league.league_slots < 1:
        raise ValueError("league size and slots must be positive")
    if not math.isfinite(league.elo_milestone_gap) or league.elo_milestone_gap < 0.0:
        raise ValueError("league.elo_milestone_gap must be finite and non-negative")
    if not math.isfinite(league.elo_temperature) or league.elo_temperature <= 0.0:
        raise ValueError("league.elo_temperature must be positive and finite")
    elo_eval = league.elo_eval
    if elo_eval.envs_per_matchup < 1 or elo_eval.step_interval < 1:
        raise ValueError("league Elo evaluation widths and intervals must be positive")
    if elo_eval.window_size < 1 or elo_eval.min_games_to_freeze < 0:
        raise ValueError("league Elo evaluation window/gate values are invalid")
    for name, value in (
        ("league.random_elo", league.random_elo),
        ("league.elo_eval.k_factor", elo_eval.k_factor),
        ("league.elo_eval.scripted_elo_init", elo_eval.scripted_elo_init),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
    if elo_eval.k_factor <= 0.0:
        raise ValueError("league.elo_eval.k_factor must be positive")


def validate_resolved_config(config: TrainConfig) -> None:
    """Validate cross-field launch constraints after all precedence is applied."""

    scale = config.scales[0]
    if scale.num_envs < 1:
        raise ValueError(f"num_envs must be positive, got {scale.num_envs}")
    if scale.num_envs % config.num_minibatches:
        raise ValueError(
            f"num_envs={scale.num_envs} must be divisible by "
            f"num_minibatches={config.num_minibatches}"
        )
    if config.num_steps < 1:
        raise ValueError(f"num_steps must be positive, got {config.num_steps}")
    if config.microbatch_tokens is not None:
        if config.microbatch_tokens < 1:
            raise ValueError(
                f"microbatch_tokens must be positive or None, got {config.microbatch_tokens}"
            )
        minibatch_tokens = (
            scale.num_envs
            * config.num_steps
            * (scale.env_config.num_ships + scale.env_config.num_fields)
            // config.num_minibatches
        )
        if config.microbatch_tokens > minibatch_tokens:
            raise ValueError(
                f"microbatch_tokens={config.microbatch_tokens} exceeds the resolved "
                f"minibatch size {minibatch_tokens}"
            )
    for name, value in (("gamma", config.gamma), ("gae_lambda", config.gae_lambda)):
        if not math.isfinite(value) or not 0.0 < value <= 1.0:
            raise ValueError(f"{name} must be finite and in (0, 1], got {value}")
    previous_probability = 0.0
    for probability, rating in config.reference_ladder:
        if not previous_probability < probability < 1.0:
            raise ValueError("reference_ladder probabilities must be strictly increasing in (0, 1)")
        if not math.isfinite(rating):
            raise ValueError("reference_ladder ratings must be finite")
        previous_probability = probability


def resolve_profile(
    profile: ProfileSpec,
    overrides: LaunchOverrides | None = None,
) -> ResolvedTrainConfig:
    """Resolve profile → derivations → launch preset → CLI overrides → validation."""

    _validate_profile(profile)
    overrides = overrides or LaunchOverrides()
    environment = profile.environment
    rollout = profile.rollout
    launch = profile.launch_defaults
    entity_tokens = environment.num_ships + environment.num_fields

    if launch.rollout_tokens is not None:
        if launch.rollout_tokens < 1:
            raise ValueError(f"rollout_tokens must be positive, got {launch.rollout_tokens}")
        default_num_envs = derive_aligned_num_envs(
            rollout_tokens=launch.rollout_tokens,
            entity_tokens=entity_tokens,
            num_steps=rollout.num_steps,
            num_minibatches=rollout.num_minibatches,
        )
        if profile.rollout.logical_batch_tokens % launch.rollout_tokens:
            raise ValueError("logical_batch_tokens must be divisible by rollout_tokens")
        rollouts_per_update = profile.rollout.logical_batch_tokens // launch.rollout_tokens
    else:
        assert launch.num_envs is not None
        default_num_envs = launch.num_envs
        actual_rollout_tokens = default_num_envs * entity_tokens * rollout.num_steps
        if profile.rollout.logical_batch_tokens % actual_rollout_tokens:
            raise ValueError(
                "logical_batch_tokens must be divisible by the fixed-environment rollout size"
            )
        rollouts_per_update = profile.rollout.logical_batch_tokens // actual_rollout_tokens

    if launch.microbatches_per_minibatch is not None:
        if launch.rollout_tokens is None:
            raise ValueError("microbatches_per_minibatch requires a rollout token target")
        if launch.microbatches_per_minibatch < 1:
            raise ValueError("microbatches_per_minibatch must be positive")
        default_microbatch_tokens = (
            launch.rollout_tokens
            // rollout.num_minibatches
            // launch.microbatches_per_minibatch
        )
        microbatch_source: ResolutionSource = "vram-preset"
    else:
        default_microbatch_tokens = launch.microbatch_tokens
        microbatch_source = "vram-preset" if launch.microbatch_tokens is not None else "profile"

    num_envs = overrides.num_envs if overrides.num_envs is not None else default_num_envs
    microbatch_tokens = (
        overrides.microbatch_tokens
        if overrides.microbatch_tokens is not None
        else default_microbatch_tokens
    )
    if overrides.num_envs is not None:
        num_envs_source: ResolutionSource = "cli"
    elif launch.rollout_tokens is not None:
        num_envs_source = "derived"
    else:
        num_envs_source = "vram-preset"
    if overrides.microbatch_tokens is not None:
        microbatch_source = "cli"

    env_config = EnvConfig(
        num_ships=environment.num_ships,
        num_fields=environment.num_fields,
        max_bullets=environment.max_bullets,
        max_episode_steps=environment.max_episode_steps,
        single_team=environment.single_team,
        action_repeat=environment.action_repeat,
        spawn_resource_spread=environment.spawn_resource_spread,
    )
    action_repeat = environment.action_repeat
    component_gammas = {
        name: derive_time_normalized_value(value, action_repeat=action_repeat)
        for name, value in profile.discounts.component_gammas_per_tick.items()
    }
    component_lambdas = {
        name: derive_time_normalized_value(value, action_repeat=action_repeat)
        for name, value in profile.discounts.component_lambdas_per_tick.items()
    }
    objective = profile.objective
    optimizer = profile.optimizer
    league = profile.league
    train_config = TrainConfig(
        scales=(ScaleConfig(env_config=env_config, num_envs=num_envs),),
        paradigm=objective.paradigm,
        schedule=objective.schedule.compile(),
        rewards=objective.rewards,
        num_steps=rollout.num_steps,
        rollouts_per_update=rollouts_per_update,
        num_minibatches=rollout.num_minibatches,
        gamma=derive_time_normalized_value(
            profile.discounts.gamma_per_tick,
            action_repeat=action_repeat,
        ),
        gae_lambda=derive_time_normalized_value(
            profile.discounts.gae_lambda_per_tick,
            action_repeat=action_repeat,
        ),
        clip_coef=optimizer.clip_coef,
        max_grad_norm=optimizer.max_grad_norm,
        total_timesteps=optimizer.total_timesteps,
        return_ema_alpha=optimizer.return_ema_alpha,
        return_min_span=optimizer.return_min_span,
        advantage_min_rms=optimizer.advantage_min_rms,
        checkpoint_dir=optimizer.checkpoint_dir,
        league_size=league.league_size,
        league_slots=league.league_slots,
        reference_ladder=league.reference_ladder,
        random_elo=league.random_elo,
        elo_milestone_gap=league.elo_milestone_gap,
        elo_temperature=league.elo_temperature,
        league_uniform_sampling=league.league_uniform_sampling,
        elo_eval=league.elo_eval,
        bc_winrate_target=league.bc_winrate_target,
        histogram_interval=optimizer.histogram_interval,
        return_quantile_samples=optimizer.return_quantile_samples,
        microbatch_tokens=microbatch_tokens,
        next_state_coef=objective.next_state_coef,
        windowed_loss_coef=objective.windowed_loss_coef,
        field_map=profile.field_map,
        log_interval=optimizer.log_interval,
        component_gammas=component_gammas,
        component_lambdas=component_lambdas,
    )
    validate_resolved_config(train_config)

    canonical_train_config = canonical_data(train_config)
    canonical_train_config["schedule"] = canonical_data(profile.objective.schedule)
    config_payload = {
        "ship_config": canonical_data(profile.ship_config),
        "model_config": canonical_data(profile.model_config),
        "train_config": canonical_train_config,
    }
    sources = _source_map(
        config_payload,
        num_envs_source=num_envs_source,
        microbatch_source=microbatch_source,
    )
    profile_digest = fingerprint(_profile_fingerprint_payload(profile))
    resolved_digest = fingerprint(
        _resolved_fingerprint_payload(
            profile,
            env_config=env_config,
            train_config=train_config,
        )
    )
    return ResolvedTrainConfig(
        profile_name=profile.name,
        ship_config=profile.ship_config,
        model_config=profile.model_config,
        train_config=train_config,
        schedule_spec=profile.objective.schedule,
        value_sources=MappingProxyType(sources),
        profile_fingerprint=profile_digest,
        resolved_config_fingerprint=resolved_digest,
    )
