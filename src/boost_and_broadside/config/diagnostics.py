"""Launch-time observability settings that do not change what training optimizes.

Gradient diagnostics decompose the PPO update into its named loss terms and
measure how those terms agree with one another. They are an instrument, not a
hyperparameter: nothing here enters the profile fingerprint, and at ``off`` the
trainer takes no diagnostic code path at all.

This module depends only on the standard library so ``execution.py`` can carry
the setting without importing a trainer.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

type GradientDiagnosticsLevel = Literal["off", "top_level", "reward_policy", "reward_full"]

# Ordered cheapest first; the CLI exposes exactly these.
GRADIENT_DIAGNOSTICS_LEVELS: tuple[str, ...] = (
    "off",
    "top_level",
    "reward_policy",
    "reward_full",
)


@dataclass(frozen=True)
class GradientDiagnosticsConfig:
    """How much of the update's gradient structure to measure, and how often.

    Levels:
        off            — no diagnostic autograd, no diagnostic allocation.
        top_level      — one gradient per weighted PPO loss term (policy, value,
                         entropy, behavior cloning, predictive state, predictive
                         action, SIGReg), with norms and pairwise cosines.
        reward_policy  — additionally splits the policy gradient across active
                         reward components. The split is exact: the clipping
                         branch is the one the aggregate PPO objective chose,
                         and the component gradients sum back to it.
        reward_full    — additionally splits the critic gradient across reward
                         components. The expensive level; one extra backward
                         traversal per component per diagnosed micro-batch.

    Args:
        level:       Diagnostic depth (see above).
        interval:    Diagnose every N PPO updates.
        minibatches: Complete optimizer minibatches diagnosed per diagnostic
                     update. Gradients are accumulated over every micro-batch of
                     a minibatch before any norm or cosine is taken, so a
                     measurement always describes a real optimizer step.
    """

    level: GradientDiagnosticsLevel = "off"
    interval: int = 1
    minibatches: int = 1

    def __post_init__(self) -> None:
        if self.level not in GRADIENT_DIAGNOSTICS_LEVELS:
            raise ValueError(
                f"gradient diagnostics level must be one of "
                f"{', '.join(GRADIENT_DIAGNOSTICS_LEVELS)}, got {self.level!r}"
            )
        if self.interval < 1:
            raise ValueError(f"gradient diagnostics interval must be positive, got {self.interval}")
        if self.minibatches < 1:
            raise ValueError(
                f"gradient diagnostics minibatches must be positive, got {self.minibatches}"
            )

    @property
    def enabled(self) -> bool:
        """Whether any diagnostic work runs at all."""

        return self.level != "off"

    @property
    def decomposes_policy_by_reward(self) -> bool:
        """Whether the policy gradient is split across reward components."""

        return self.level in ("reward_policy", "reward_full")

    @property
    def decomposes_value_by_reward(self) -> bool:
        """Whether the critic gradient is split across reward components."""

        return self.level == "reward_full"

    def measures_update(self, update: int) -> bool:
        """Whether PPO update index ``update`` is a diagnostic update."""

        return self.enabled and update % self.interval == 0

    def document(self) -> dict[str, object]:
        """The record stored beside a resolved launch."""

        return asdict(self)


GRADIENT_DIAGNOSTICS_OFF = GradientDiagnosticsConfig()


__all__ = [
    "GRADIENT_DIAGNOSTICS_LEVELS",
    "GRADIENT_DIAGNOSTICS_OFF",
    "GradientDiagnosticsConfig",
    "GradientDiagnosticsLevel",
]
