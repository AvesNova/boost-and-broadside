"""Resolve independent profile intent into validated trainer configurations."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from dataclasses import fields as dataclass_fields
from decimal import Decimal
from types import MappingProxyType
from typing import Any

from boost_and_broadside.config.core import EnvConfig, ModelConfig
from boost_and_broadside.config.fingerprint import canonical_data, fingerprint
from boost_and_broadside.config.live_elo import validate_live_reference_probabilities
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

    Each knob carries the source that chose it, because the same three values
    can arrive from a VRAM cache entry, a VRAM preset, or the command line, and
    a launch has to record which one it was.  ``cli`` is the default so that a
    caller which only knows it is applying a user override says so by omission.
    """

    num_envs: int | None = None
    microbatch_tokens: int | None = None
    grad_checkpoint: bool | None = None
    num_envs_source: ResolutionSource = "cli"
    microbatch_tokens_source: ResolutionSource = "cli"
    grad_checkpoint_source: ResolutionSource = "cli"


@dataclass(frozen=True)
class LaunchGeometry:
    """The token arithmetic every launch of one profile has to preserve.

    ``aligned_logical_batch_tokens`` is the profile's nominal logical batch
    after environment alignment.  It is fixed: a launch may redistribute it
    across shard widths (plan tier 2) but may not resize it (tier 3).
    """

    entity_tokens: int
    num_steps: int
    num_minibatches: int
    aligned_logical_batch_tokens: int
    default_num_envs: int
    default_rollouts_per_update: int
    default_microbatch_tokens: int | None
    default_num_envs_source: ResolutionSource
    default_microbatch_source: ResolutionSource

    def rollout_tokens(self, num_envs: int) -> int:
        """Entity tokens resident in one rollout shard at this width."""

        return num_envs * self.entity_tokens * self.num_steps

    def minibatch_tokens(self, num_envs: int) -> int:
        """Entity tokens in one PPO minibatch at this width."""

        return self.rollout_tokens(num_envs) // self.num_minibatches

    def shard_widths(self) -> tuple[tuple[int, int], ...]:
        """Every ``(num_envs, rollouts_per_update)`` preserving the fixed batch.

        Widest shard first.  A width qualifies only when it divides the aligned
        logical batch exactly and stays minibatch-aligned, which is the same
        rule :func:`derive_rollouts_per_update` and
        :func:`validate_resolved_config` enforce one launch at a time.
        """

        total_env_steps = self.aligned_logical_batch_tokens // self.entity_tokens // self.num_steps
        # A width has to be a positive multiple of the minibatch count, which
        # bounds the shard count without scanning the whole batch.
        widths = []
        for shards in range(1, total_env_steps // self.num_minibatches + 1):
            num_envs, remainder = divmod(total_env_steps, shards)
            if remainder or num_envs % self.num_minibatches:
                continue
            widths.append((num_envs, shards))
        return tuple(widths)


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


def derive_rollouts_per_update(
    *,
    aligned_logical_batch_tokens: int,
    num_envs: int,
    entity_tokens: int,
    num_steps: int,
) -> int:
    """Preserve an aligned logical batch with an integer rollout-shard count."""

    for name, value in (
        ("aligned_logical_batch_tokens", aligned_logical_batch_tokens),
        ("num_envs", num_envs),
        ("entity_tokens", entity_tokens),
        ("num_steps", num_steps),
    ):
        if value < 1:
            raise ValueError(f"{name} must be positive, got {value}")
    rollout_tokens = num_envs * entity_tokens * num_steps
    rollouts_per_update, remainder = divmod(aligned_logical_batch_tokens, rollout_tokens)
    if remainder or rollouts_per_update < 1:
        raise ValueError(
            f"num_envs={num_envs} cannot preserve the fixed logical batch of "
            f"{aligned_logical_batch_tokens} entity tokens with an integer shard count"
        )
    return rollouts_per_update


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


# Fields TrainConfig takes from the profile verbatim: the names that appear in
# both schemas. Deriving this rather than listing it is what makes adding a plain
# hyperparameter a two-line change -- one field on each dataclass -- with nothing
# to edit here. A transformed value is kept out by being named differently on the
# two sides, which ``tests/config/test_resolution.py`` pins.
_PASS_THROUGH: tuple[str, ...] = tuple(
    sorted(
        {field.name for field in dataclass_fields(ProfileSpec)}
        & {field.name for field in dataclass_fields(TrainConfig)}
    )
)


def _profile_fingerprint_payload(profile: ProfileSpec) -> dict[str, Any]:
    payload = canonical_data(profile)
    del payload["launch"]
    del payload["model_config"]["grad_checkpoint"]
    return {
        "schema_version": PROFILE_SCHEMA_VERSION,
        "profile": payload,
    }


def _resolved_fingerprint_payload(
    profile: ProfileSpec,
    *,
    model_config: ModelConfig,
    env_config: EnvConfig,
    train_config: TrainConfig,
) -> dict[str, Any]:
    train_payload = canonical_data(train_config)
    # Runtime closure names are characterization evidence, not schema.  The
    # resolved fingerprint binds the same final schedule through its declarative
    # intent instead.
    train_payload["schedule"] = canonical_data(profile.schedule_spec)
    return {
        "schema_version": RESOLVED_CONFIG_SCHEMA_VERSION,
        "profile_name": profile.name,
        "ship_config": canonical_data(profile.ship_config),
        "model_config": canonical_data(model_config),
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
    grad_checkpoint_source: ResolutionSource,
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
        if path.startswith("model_config.grad_checkpoint"):
            sources[path] = grad_checkpoint_source
    return sources


def _validate_profile(profile: ProfileSpec) -> None:
    if not profile.name or profile.name != profile.name.strip():
        raise ValueError(f"profile name must be non-empty and trimmed, got {profile.name!r}")
    if profile.num_ships < 1:
        raise ValueError("profile.num_ships must be positive")
    if profile.num_fields < 0:
        raise ValueError("profile.num_fields must be non-negative")
    if profile.max_bullets < 0:
        raise ValueError("profile.max_bullets must be non-negative")
    if profile.logical_batch_tokens < 1:
        raise ValueError("profile.logical_batch_tokens must be positive")
    if profile.num_steps < 1:
        raise ValueError("profile.num_steps must be positive")
    if profile.num_minibatches < 1:
        raise ValueError("profile.num_minibatches must be positive")
    launch = profile.launch
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
    if profile.total_timesteps < 1:
        raise ValueError("profile.total_timesteps must be positive")
    for name, value in (
        ("profile.clip_coef", profile.clip_coef),
        ("profile.max_grad_norm", profile.max_grad_norm),
        ("profile.return_ema_alpha", profile.return_ema_alpha),
        ("profile.return_min_span", profile.return_min_span),
        ("profile.advantage_min_rms", profile.advantage_min_rms),
    ):
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be positive and finite, got {value}")
    if profile.clip_coef >= 1.0:
        raise ValueError("profile.clip_coef must be below 1")
    if profile.return_ema_alpha > 1.0:
        raise ValueError("profile.return_ema_alpha must be at most 1")
    if profile.return_quantile_samples is not None and profile.return_quantile_samples < 1:
        raise ValueError("profile.return_quantile_samples must be positive or None")
    if profile.histogram_interval < 1 or profile.log_interval < 1:
        raise ValueError("optimizer logging intervals must be positive")
    if not profile.checkpoint_dir:
        raise ValueError("profile.checkpoint_dir must be non-empty")
    if profile.league_size < 1 or profile.league_slots < 1:
        raise ValueError("league size and slots must be positive")
    if not math.isfinite(profile.elo_milestone_gap) or profile.elo_milestone_gap < 0.0:
        raise ValueError("profile.elo_milestone_gap must be finite and non-negative")
    if not math.isfinite(profile.elo_temperature) or profile.elo_temperature <= 0.0:
        raise ValueError("profile.elo_temperature must be positive and finite")
    elo_eval = profile.elo_eval
    if elo_eval.envs_per_matchup < 1 or elo_eval.step_interval < 1:
        raise ValueError("league Elo evaluation widths and intervals must be positive")
    if elo_eval.window_size < 1 or elo_eval.min_games_to_freeze < 0:
        raise ValueError("league Elo evaluation window/gate values are invalid")
    for name, value in (
        ("profile.elo_eval.k_factor", elo_eval.k_factor),
        ("profile.elo_eval.scripted_live_elo", elo_eval.scripted_live_elo),
    ):
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite, got {value}")
    validate_live_reference_probabilities(profile.live_reference_probabilities)
    if elo_eval.k_factor <= 0.0:
        raise ValueError("profile.elo_eval.k_factor must be positive")


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
    validate_live_reference_probabilities(config.live_reference_probabilities)


def launch_geometry(profile: ProfileSpec) -> LaunchGeometry:
    """Derive one profile's fixed token arithmetic and its default launch sizing.

    This is the single named sizing derivation.  ``resolve_profile`` applies it
    to one launch; VRAM resolution uses it to enumerate the shard widths that
    preserve the same logical batch.
    """

    _validate_profile(profile)
    launch = profile.launch
    entity_tokens = profile.num_ships + profile.num_fields

    if launch.rollout_tokens is not None:
        default_num_envs = derive_aligned_num_envs(
            rollout_tokens=launch.rollout_tokens,
            entity_tokens=entity_tokens,
            num_steps=profile.num_steps,
            num_minibatches=profile.num_minibatches,
        )
        if profile.logical_batch_tokens % launch.rollout_tokens:
            raise ValueError("logical_batch_tokens must be divisible by rollout_tokens")
        default_rollouts_per_update = profile.logical_batch_tokens // launch.rollout_tokens
        default_num_envs_source: ResolutionSource = "derived"
    else:
        assert launch.num_envs is not None
        default_num_envs = launch.num_envs
        default_rollout_tokens = default_num_envs * entity_tokens * profile.num_steps
        if profile.logical_batch_tokens % default_rollout_tokens:
            raise ValueError(
                "logical_batch_tokens must be divisible by the fixed-environment rollout size"
            )
        default_rollouts_per_update = profile.logical_batch_tokens // default_rollout_tokens
        default_num_envs_source = "vram-preset"

    if launch.microbatches_per_minibatch is not None:
        if launch.rollout_tokens is None:
            raise ValueError("microbatches_per_minibatch requires a rollout token target")
        default_microbatch_tokens = (
            launch.rollout_tokens
            // profile.num_minibatches
            // launch.microbatches_per_minibatch
        )
        default_microbatch_source: ResolutionSource = "vram-preset"
    else:
        default_microbatch_tokens = launch.microbatch_tokens
        default_microbatch_source = (
            "vram-preset" if launch.microbatch_tokens is not None else "profile"
        )

    return LaunchGeometry(
        entity_tokens=entity_tokens,
        num_steps=profile.num_steps,
        num_minibatches=profile.num_minibatches,
        aligned_logical_batch_tokens=(
            default_num_envs * entity_tokens * profile.num_steps * default_rollouts_per_update
        ),
        default_num_envs=default_num_envs,
        default_rollouts_per_update=default_rollouts_per_update,
        default_microbatch_tokens=default_microbatch_tokens,
        default_num_envs_source=default_num_envs_source,
        default_microbatch_source=default_microbatch_source,
    )


def resolve_profile(
    profile: ProfileSpec,
    overrides: LaunchOverrides | None = None,
) -> ResolvedTrainConfig:
    """Resolve profile → derivations → VRAM cache/preset → CLI overrides → validation."""

    geometry = launch_geometry(profile)
    overrides = overrides or LaunchOverrides()
    entity_tokens = geometry.entity_tokens

    num_envs = (
        overrides.num_envs if overrides.num_envs is not None else geometry.default_num_envs
    )
    rollouts_per_update = derive_rollouts_per_update(
        aligned_logical_batch_tokens=geometry.aligned_logical_batch_tokens,
        num_envs=num_envs,
        entity_tokens=entity_tokens,
        num_steps=profile.num_steps,
    )
    microbatch_tokens = (
        overrides.microbatch_tokens
        if overrides.microbatch_tokens is not None
        else geometry.default_microbatch_tokens
    )
    num_envs_source = (
        overrides.num_envs_source
        if overrides.num_envs is not None
        else geometry.default_num_envs_source
    )
    microbatch_source = (
        overrides.microbatch_tokens_source
        if overrides.microbatch_tokens is not None
        else geometry.default_microbatch_source
    )
    if overrides.grad_checkpoint is None:
        model_config = profile.model_config
        grad_checkpoint_source: ResolutionSource = "profile"
    else:
        model_config = replace(profile.model_config, grad_checkpoint=overrides.grad_checkpoint)
        grad_checkpoint_source = overrides.grad_checkpoint_source

    env_config = EnvConfig(
        num_ships=profile.num_ships,
        num_fields=profile.num_fields,
        max_bullets=profile.max_bullets,
        max_episode_steps=profile.max_episode_steps,
        single_team=profile.single_team,
        action_repeat=profile.action_repeat,
        spawn_resource_spread=profile.spawn_resource_spread,
    )
    action_repeat = profile.action_repeat
    component_gammas = {
        name: derive_time_normalized_value(value, action_repeat=action_repeat)
        for name, value in profile.component_gammas_per_tick.items()
    }
    component_lambdas = {
        name: derive_time_normalized_value(value, action_repeat=action_repeat)
        for name, value in profile.component_lambdas_per_tick.items()
    }
    train_config = TrainConfig(
        # Everything named the same on both sides is intent the resolver does not
        # touch; see ProfileSpec on why that convention is enforceable.
        **{name: getattr(profile, name) for name in _PASS_THROUGH},
        scales=(ScaleConfig(env_config=env_config, num_envs=num_envs),),
        schedule=profile.schedule_spec.compile(),
        gamma=derive_time_normalized_value(profile.gamma_per_tick, action_repeat=action_repeat),
        gae_lambda=derive_time_normalized_value(
            profile.gae_lambda_per_tick, action_repeat=action_repeat
        ),
        component_gammas=component_gammas,
        component_lambdas=component_lambdas,
        rollouts_per_update=rollouts_per_update,
        microbatch_tokens=microbatch_tokens,
    )
    validate_resolved_config(train_config)

    canonical_train_config = canonical_data(train_config)
    canonical_train_config["schedule"] = canonical_data(profile.schedule_spec)
    config_payload = {
        "ship_config": canonical_data(profile.ship_config),
        "model_config": canonical_data(model_config),
        "train_config": canonical_train_config,
    }
    sources = _source_map(
        config_payload,
        num_envs_source=num_envs_source,
        microbatch_source=microbatch_source,
        grad_checkpoint_source=grad_checkpoint_source,
    )
    profile_digest = fingerprint(_profile_fingerprint_payload(profile))
    resolved_digest = fingerprint(
        _resolved_fingerprint_payload(
            profile,
            model_config=model_config,
            env_config=env_config,
            train_config=train_config,
        )
    )
    return ResolvedTrainConfig(
        profile_name=profile.name,
        ship_config=profile.ship_config,
        model_config=model_config,
        train_config=train_config,
        schedule_spec=profile.schedule_spec,
        value_sources=MappingProxyType(sources),
        profile_fingerprint=profile_digest,
        resolved_config_fingerprint=resolved_digest,
    )
