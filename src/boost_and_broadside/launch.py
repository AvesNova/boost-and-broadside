"""Compose one training launch from profile intent, VRAM policy, and execution.

``train`` and ``train --print-config`` must agree exactly about what would run,
so both go through :func:`resolve_training_launch`.  It applies the plan's
precedence in order — profile intent and mechanical derivations, then a VRAM
cache entry or preset, then explicit launch overrides, then validation — and
returns the resolved configuration together with the record of how each launch
value was chosen.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from boost_and_broadside.config.resolve import LaunchOverrides
from boost_and_broadside.config.schema import ResolvedTrainConfig
from boost_and_broadside.config.vram import (
    VramKnobs,
    VramResolution,
    apply_cli_overrides,
    launch_overrides,
    parse_vram_policy,
)
from boost_and_broadside.errors import UserFacingError
from boost_and_broadside.execution import ExecutionSettings, resolve_execution_settings
from boost_and_broadside.profiles import resolve_named_profile

ProfileResolver = Callable[..., ResolvedTrainConfig]


def profile_knobs(resolved: ResolvedTrainConfig) -> VramKnobs:
    """The sizing knobs one resolved configuration carries.

    Applied to the profile resolved at its defaults it is the reference a VRAM
    decision is judged against; applied to the final launch it is what that
    launch actually runs at. The distance between the two is the tier list.
    """

    return VramKnobs(
        num_envs=resolved.train_config.scales[0].num_envs,
        microbatch_tokens=resolved.train_config.microbatch_tokens,
        grad_checkpoint=resolved.model_config.grad_checkpoint,
    )


@dataclass(frozen=True)
class TrainingLaunch:
    """Everything one ``bnb train`` invocation decided before allocating."""

    resolved: ResolvedTrainConfig
    execution: ExecutionSettings
    vram: VramResolution
    # The profile's own derived sizing, before any VRAM decision or override.
    baseline: VramKnobs

    def document(self) -> dict[str, Any]:
        """The launch record stored in checkpoints and printed by --print-config."""

        return {
            **self.execution.document(),
            # Judged against what this launch actually runs at, so a width or
            # microbatch the command line chose is counted like any other.
            "vram": self.vram.document(self.baseline, profile_knobs(self.resolved)),
        }


def resolve_training_launch(
    *,
    profile: str,
    vram: str = "auto",
    device: str = "auto",
    seed: int | None = None,
    compile_mode: str | None = None,
    wandb: bool = True,
    allow_config_drift: bool = False,
    num_envs: int | None = None,
    microbatch_tokens: int | None = None,
    allow_probe: bool = True,
    cache_file: Path | None = None,
    runner: Callable[..., Any] | None = None,
    report: Callable[[str], None] | None = None,
    resolve: ProfileResolver = resolve_named_profile,
) -> TrainingLaunch:
    """Resolve a complete launch without constructing a trainer or environment.

    ``allow_probe`` is false for ``--print-config``: printing what a launch
    would do must not spend an hour measuring the card.
    """

    policy = parse_vram_policy(vram)
    if policy.probes and not allow_probe:
        raise UserFacingError(
            f"--print-config cannot run --vram {policy}: printing a launch has no side "
            "effects. Use --vram auto to print what the cache already resolves."
        )
    if policy.probes and (num_envs is not None or microbatch_tokens is not None):
        # A probe exists to determine these two values and store the result for
        # later launches. Pinning one would cache a measurement of a
        # configuration nobody asked the cache about.
        raise UserFacingError(
            f"--vram {policy} determines --num-envs and --microbatch-tokens; pass one or "
            "the other, not both"
        )

    execution = resolve_execution_settings(
        device=device,
        seed=seed,
        compile_mode=compile_mode,
        wandb=wandb,
        allow_config_drift=allow_config_drift,
    )
    # The semantic fingerprint keys the VRAM cache and does not depend on any
    # launch sizing, so resolving the profile at its defaults is enough to ask
    # the cache a question about it.
    intent = resolve(profile)

    from boost_and_broadside.vram_probe import resolve_vram

    resolution = resolve_vram(
        policy,
        profile_name=profile,
        profile_fingerprint=intent.profile_fingerprint,
        device=execution.device,
        compile_mode=execution.compile_mode,
        cache_file=cache_file,
        runner=runner,
        report=report,
    )
    resolution = apply_cli_overrides(
        resolution, num_envs=num_envs, microbatch_tokens=microbatch_tokens
    )
    overrides: LaunchOverrides = launch_overrides(
        resolution, num_envs=num_envs, microbatch_tokens=microbatch_tokens
    )
    resolved = intent if _is_noop(overrides) else resolve(profile, overrides)
    return TrainingLaunch(
        resolved=resolved,
        execution=execution,
        vram=resolution,
        baseline=profile_knobs(intent),
    )


def _is_noop(overrides: LaunchOverrides) -> bool:
    return (
        overrides.num_envs is None
        and overrides.microbatch_tokens is None
        and overrides.grad_checkpoint is None
    )


__all__ = ["TrainingLaunch", "profile_knobs", "resolve_training_launch"]
