"""Shared validation for CLI execution settings.

This module deliberately depends only on Torch and the standard library so
``train --print-config`` can report the launch it validated without importing a
trainer, environment, renderer, or user-facing mode.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class ExecutionSettings:
    """Validated process settings recorded beside a resolved configuration."""

    device: str
    seed: int
    compile_mode: str | None
    wandb: bool
    allow_config_drift: bool

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
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError(f"--device {requested} was requested, but CUDA is unavailable")
    return str(device)


def resolve_execution_settings(
    *,
    device: str,
    seed: int | None,
    compile_mode: str | None,
    wandb: bool,
    allow_config_drift: bool,
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
    )


def initialize_execution(settings: ExecutionSettings) -> None:
    """Apply already validated RNG settings to the current process."""

    random.seed(settings.seed)
    torch.manual_seed(settings.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(settings.seed)


__all__ = [
    "ExecutionSettings",
    "initialize_execution",
    "resolve_device",
    "resolve_execution_settings",
]
