"""Shared validation for CLI execution settings.

This module deliberately depends only on NumPy, Torch and the standard library
so ``train --print-config`` can report the launch it validated without importing
a trainer, environment, renderer, or user-facing mode.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import numpy as np
import torch

from boost_and_broadside.config.diagnostics import (
    GRADIENT_DIAGNOSTICS_OFF,
    GradientDiagnosticsConfig,
)


@dataclass(frozen=True)
class ExecutionSettings:
    """Validated process settings recorded beside a resolved configuration.

    Observability settings live here rather than in the profile: they are chosen
    per launch and change nothing about what the run optimizes.
    """

    device: str
    seed: int
    compile_mode: str | None
    wandb: bool
    allow_config_drift: bool
    gradient_diagnostics: GradientDiagnosticsConfig = GRADIENT_DIAGNOSTICS_OFF

    def document(self) -> dict[str, object]:
        return asdict(self)


def resolve_device(requested: str) -> str:
    """Resolve ``auto`` and reject unavailable or malformed devices."""

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(requested)
    except RuntimeError as error:
        raise ValueError(f"invalid --device value: {requested!r}") from error
    if device.type == "cpu":
        if device.index is not None:
            raise ValueError(f"invalid --device value: {requested!r}; CPU has no device index")
        return str(device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise ValueError(f"--device {requested} was requested, but CUDA is unavailable")
        if device.index is not None and device.index >= torch.cuda.device_count():
            raise ValueError(
                f"--device {requested} was requested, but only {torch.cuda.device_count()} "
                "CUDA device(s) are available"
            )
        return str(device)
    if device.type == "mps":
        if not torch.backends.mps.is_available():
            raise ValueError(f"--device {requested} was requested, but MPS is unavailable")
        if device.index not in (None, 0):
            raise ValueError(f"invalid --device value: {requested!r}; MPS exposes one device")
        return str(device)
    if device.type == "xpu":
        xpu = getattr(torch, "xpu", None)
        if xpu is None or not xpu.is_available():
            raise ValueError(f"--device {requested} was requested, but XPU is unavailable")
        if device.index is not None and device.index >= xpu.device_count():
            raise ValueError(
                f"--device {requested} was requested, but only {xpu.device_count()} "
                "XPU device(s) are available"
            )
        return str(device)
    raise ValueError(
        f"unsupported --device value: {requested!r}; choose auto, cpu, cuda, mps, or xpu"
    )


def resolve_execution_settings(
    *,
    device: str,
    seed: int | None,
    compile_mode: str | None,
    wandb: bool,
    allow_config_drift: bool,
    gradient_diagnostics: GradientDiagnosticsConfig = GRADIENT_DIAGNOSTICS_OFF,
) -> ExecutionSettings:
    """Validate and resolve settings without mutating process RNG state."""

    resolved_seed = torch.initial_seed() if seed is None else seed
    if not -(2**63) <= resolved_seed <= 2**64 - 1:
        raise ValueError(
            f"--seed must be between {-2**63} and {2**64 - 1}, got {resolved_seed}"
        )
    return ExecutionSettings(
        device=resolve_device(device),
        seed=resolved_seed,
        compile_mode=compile_mode,
        wandb=wandb,
        allow_config_drift=allow_config_drift,
        gradient_diagnostics=gradient_diagnostics,
    )


def initialize_execution(settings: ExecutionSettings) -> None:
    """Apply already validated RNG settings to the current process."""

    random.seed(settings.seed)
    # NumPy's global RNG draws the PPO minibatch order. Left unseeded it varies
    # per process, so two runs of the same configuration see their environments
    # grouped differently and cannot be compared as a matched pair.
    np.random.seed(settings.seed % 2**32)
    torch.manual_seed(settings.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.seed)


__all__ = [
    "ExecutionSettings",
    "initialize_execution",
    "resolve_device",
    "resolve_execution_settings",
]
