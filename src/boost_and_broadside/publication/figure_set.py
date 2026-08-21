"""The canonical figure set: every chart a finished run can produce.

This is the single declaration of *what figures exist*. ``bnb figures`` renders
the whole set from one run's own measurements into that run's artifact
directory, and ``docs/publications.toml`` selects which run's rendered figures
are published. Those are different questions -- what a run can show, versus
which run the documents currently illustrate -- and separating them is what
makes a new run's charts a single command rather than a manifest rewrite.

Each entry names the renderer and, per renderer source, the artifact *type* it
reads. The type is enough: a run holds at most one current artifact of each
kind, so the set resolves against any finished run without naming ids.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType


@dataclass(frozen=True)
class FigureSpec:
    """One renderable figure, resolvable against any run's artifacts."""

    name: str
    renderer: str
    # Renderer source name -> artifact type under the run.
    sources: Mapping[str, str]
    description: str

    @property
    def output(self) -> str:
        """Path of this figure inside a rendered figures artifact."""

        return self.name


FIGURES: tuple[FigureSpec, ...] = (
    FigureSpec(
        "elo_curve.png",
        "training-elo-curve-v1",
        MappingProxyType({"calibration": "elo-calibration"}),
        "Post-hoc calibrated Elo over training, against the scripted controller.",
    ),
    FigureSpec(
        "win_rate_vs_scripted.png",
        "training-win-rate-v1",
        MappingProxyType({"wandb_export": "wandb-export"}),
        "Win rate against the scripted controller over training.",
    ),
    FigureSpec(
        "training_health.png",
        "training-health-v1",
        MappingProxyType({"wandb_export": "wandb-export"}),
        "Critic fit, reward, KL, and clip fraction as one optimisation panel.",
    ),
    FigureSpec(
        "next_state_error.png",
        "next-state-error-v1",
        MappingProxyType({"wandb_export": "wandb-export"}),
        "Next-state prediction error per observation dimension.",
    ),
    FigureSpec(
        "crossover_phase.png",
        "crossover-phase-v1",
        MappingProxyType({"crossover": "crossover"}),
        "Where the scripted fleet overtakes the policy, against equal counts.",
    ),
    FigureSpec(
        "crossover_ratio.png",
        "crossover-ratio-v1",
        MappingProxyType({"crossover": "crossover"}),
        "Scripted ships beaten per policy ship.",
    ),
    FigureSpec(
        "crossover.json",
        "crossover-data-v1",
        MappingProxyType({"crossover": "crossover"}),
        "The measured crossover curves as published data.",
    ),
    FigureSpec(
        "elo_scale_scripted_1000.png",
        "elo-scale-v1",
        MappingProxyType({"scale": "elo-scale", "reference": "semi-random-ladder"}),
        "Checkpoint strength across symmetric fleet sizes, scripted-anchored.",
    ),
    FigureSpec(
        "semi_random_connectivity.png",
        "semi-random-connectivity-v1",
        MappingProxyType({"ladder": "semi-random-ladder"}),
        "How informative each step of the random-to-scripted ladder is.",
    ),
    FigureSpec(
        "ar_report_4v4",
        "ar-report-v1",
        MappingProxyType({"ar_report": "ar-report"}),
        "Closed- and open-loop imagined rollouts against ground truth, 4v4.",
    ),
    FigureSpec(
        "noise_calibration",
        "noise-calibration-v1",
        MappingProxyType({"noise": "noise-calibration"}),
        "Next-state prediction error: sigma, autocorrelation, and AR growth.",
    ),
)

FIGURES_BY_NAME: Mapping[str, FigureSpec] = MappingProxyType({f.name: f for f in FIGURES})


def required_artifact_types() -> tuple[str, ...]:
    """Every artifact type the full set reads, in stable order."""

    seen: dict[str, None] = {}
    for figure in FIGURES:
        for artifact_type in figure.sources.values():
            seen.setdefault(artifact_type, None)
    return tuple(seen)


__all__ = ["FIGURES", "FIGURES_BY_NAME", "FigureSpec", "required_artifact_types"]
