"""The README training figures, rendered from stored W&B and calibration evidence.

A run leaves two chart-ready sources behind: the in-training archive ingested
from W&B, and the post-hoc calibration's re-rating in the same export shape.
Reading both through one loader is what makes "in-training versus calibrated" a
question of which artifact an entry selects rather than of which code path ran.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from boost_and_broadside.publication.renderer_api import (
    PublicationError,
    Renderer,
    RenderInputs,
    register,
)
from boost_and_broadside.viz import charts, history
from boost_and_broadside.viz.charts import Line, Panel, Points

# The next-state prediction error logs symmetric spatial channels separately;
# each pair is averaged into one honest curve. Order sets the palette slots.
#
# Each group lists both spellings the trainer has used for the same predictor.
# `afdf406` replaced a hardcoded nine-name tuple with
# `FeatureCoordinator.get_feature_names()`, which renamed every one of them
# positionally and appended a tenth predictor; the pairing below is that
# positional identity, not a guess. A run exported on either side of the rename
# therefore renders the same seven curves. The tenth predictor
# (`local_log_index_0`) is deliberately not charted: no export that predates the
# rename has it, and the landmark policy never learned it (see the S15 handoff).
_NEXT_STATE = [
    ("position", (
        ["next_state/pos_x_dphase", "next_state/pos_y_dphase"],
        ["next_state/position_x_0", "next_state/position_y_0"],
    )),
    ("velocity", (
        ["next_state/vel_dvx_norm", "next_state/vel_dvy_norm"],
        ["next_state/velocity_0", "next_state/velocity_1"],
    )),
    ("angular vel", (["next_state/ang_vel_abs"], ["next_state/angular_velocity_0"])),
    ("attitude", (["next_state/att_dphase"], ["next_state/attitude_0"])),
    ("cooldown", (["next_state/cooldown_dphase"], ["next_state/cooldown_0"])),
    ("health", (["next_state/health_dphase"], ["next_state/health_0"])),
    ("power", (["next_state/power_dphase"], ["next_state/power_0"])),
]


def _required(x: np.ndarray, y: np.ndarray, keys: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Refuse to draw a curve the source has no measurements for.

    A metric rename between the export and this module reads exactly like a
    metric that was never logged: :func:`history.series` drops every row and the
    figure comes out empty but well-formed. A blank canonical figure is worse
    than a failed publish, so an empty series names the keys it looked for and
    stops here.
    """

    if x.size:
        return x, y
    named = ", ".join(keys)
    raise PublicationError(
        f"no finite points for {named} in this export; the figure would be blank. "
        "The stored export either predates or postdates these metric names."
    )


def _line(rows: list[dict], key: str, label: str) -> Line:
    return Line(*_required(*history.series(rows, key), [key]), label)


def _next_state_line(rows: list[dict], label: str, spellings: tuple[list[str], ...]) -> Line:
    """One averaged curve, under whichever spelling this export actually used."""

    for keys in spellings:
        x, y = history.combine_series(rows, keys)
        if x.size:
            return Line(x, y, label=label)
    return Line(*_required(*history.combine_series(rows, spellings[0]),
                           [key for keys in spellings for key in keys]), label=label)


def _wandb_rows(inputs: RenderInputs) -> list[dict]:
    return history.load_history(inputs.artifact("wandb_export").path / "history.jsonl")


def _render_elo_curve(inputs: RenderInputs, out_dir: Path) -> list[Path]:
    calibration = inputs.artifact("calibration")
    rows = history.load_history(calibration.path / "chart_history.jsonl")
    summary = history.load_summary(calibration.path / "chart_summary.json")

    # The scripted controller — the one opponent shared across runs — is the
    # reference line for reading where the policy actually got to.
    scripted = summary.get("calibrated_elo/scripted")
    references = [("scripted controller", float(scripted))] if history._finite(scripted) else None
    # Tournament-rated checkpoints sit on the refit curve as precise points;
    # the last one pins the curve's endpoint.
    steps, ratings, stderr = history.checkpoint_points(summary)
    checkpoints = Points(
        np.asarray(steps),
        np.asarray(ratings),
        np.asarray(stderr),
        "tournament-rated checkpoints (±1 SE)",
    )
    return [
        charts.trend(
            [_line(rows, "calibrated_elo/live", "live policy (refit)")],
            out_dir / "elo_curve.png",
            title="Calibrated Elo over training",
            ylabel="Elo (scripted = 1000)",
            reference_lines=references,
            points=checkpoints,
        )
    ]


def _render_win_rate(inputs: RenderInputs, out_dir: Path) -> list[Path]:
    return [
        charts.trend(
            [_line(_wandb_rows(inputs), "overview/win_rate_vs_scripted", "vs scripted")],
            out_dir / "win_rate_vs_scripted.png",
            title="Win rate vs the scripted controller",
            ylabel="win rate",
            percent=True,
        )
    ]


def _render_health(inputs: RenderInputs, out_dir: Path) -> list[Path]:
    rows = _wandb_rows(inputs)
    return [
        charts.grid(
            [
                Panel(
                    [_line(rows, "overview/explained_variance", "EV")],
                    title="Critic explained variance",
                    ylabel="explained variance",
                ),
                Panel(
                    [_line(rows, "overview/reward_mean", "reward")],
                    title="Mean episode reward",
                    ylabel="reward",
                ),
                Panel(
                    [_line(rows, "overview/kl", "KL")],
                    title="Policy update KL divergence",
                    ylabel="KL divergence",
                ),
                Panel(
                    [_line(rows, "overview/clip_fraction", "clip")],
                    title="PPO clip fraction",
                    ylabel="clip fraction",
                    percent=True,
                ),
            ],
            out_dir / "training_health.png",
        )
    ]


def _render_next_state_error(inputs: RenderInputs, out_dir: Path) -> list[Path]:
    rows = _wandb_rows(inputs)
    return [
        charts.trend(
            [
                _next_state_line(rows, label, spellings)
                for label, spellings in _NEXT_STATE
            ],
            out_dir / "next_state_error.png",
            title="Next-state prediction error by dimension",
            ylabel="prediction error (normalised)",
            log_y=True,
            size=(10.5, 6.2),
        )
    ]


register(
    Renderer(
        name="training-elo-curve-v1",
        description="Post-hoc calibrated Elo over training, with tournament-rated checkpoints.",
        render=_render_elo_curve,
        required_artifacts=("calibration",),
        supported_schemas={"calibration": (2,)},
    )
)
register(
    Renderer(
        name="training-win-rate-v1",
        description="Win rate against the scripted controller over training.",
        render=_render_win_rate,
        required_artifacts=("wandb_export",),
        supported_schemas={"wandb_export": (1,)},
    )
)
register(
    Renderer(
        name="training-health-v1",
        description="Critic fit, reward, KL, and clip fraction as one optimisation panel.",
        render=_render_health,
        required_artifacts=("wandb_export",),
        supported_schemas={"wandb_export": (1,)},
    )
)
register(
    Renderer(
        name="next-state-error-v1",
        description="Next-state prediction error per observation dimension.",
        render=_render_next_state_error,
        required_artifacts=("wandb_export",),
        supported_schemas={"wandb_export": (1,)},
    )
)
