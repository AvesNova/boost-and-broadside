"""Static plots comparing a run's in-training ratings against calibrated ones.

Colours come from a validated categorical palette (blue/green, which clears the
all-pairs colour-vision and contrast gates on a light surface), and no chart
carries more than two categorical series. Every series is direct-labelled as
well as legended, so identity never rests on colour alone.
"""

from functools import partial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display in a training container
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from boost_and_broadside.viz.style import (  # noqa: E402
    AVG,
    CALIBRATED,
    GRID,
    HALF_WIN,
    INK_MUTED,
    INK_SECONDARY,
    SHADE_DARK,
    SHADE_LIGHT,
    SURFACE,
    TRAINING,
)
from boost_and_broadside.viz.style import MAX_PLOT_STDERR as _MAX_PLOT_STDERR  # noqa: E402
from boost_and_broadside.viz.style import (
    draw_reference_lines as _draw_reference_lines,  # noqa: E402
)
from boost_and_broadside.viz.style import label_series_ends as _label_series_ends  # noqa: E402
from boost_and_broadside.viz.style import new_figure as _new_figure  # noqa: E402
from boost_and_broadside.viz.style import style_axes as _style_axes  # noqa: E402

# Human-readable names for the draw conventions, for titles and legends.
_TIE_LABEL = {"half_win": "ties as ½ win", "decisive": "decisive games only"}


def _finite(curve: list[dict], key: str) -> np.ndarray:
    return np.array([point.get(key, float("nan")) for point in curve], dtype=float)


def _measured(values: np.ndarray, stderr: np.ndarray) -> np.ndarray:
    """Mask of points precise enough to plot as a measurement."""
    return np.isfinite(values) & np.isfinite(stderr) & (stderr <= _MAX_PLOT_STDERR)


def _reference_lines(result: dict, key: str = "calibrated_elo") -> list[tuple[str, float]]:
    """Fixed opponents worth drawing as a rule across a curve.

    The scripted agent is the one rating on the ladder with meaning outside this
    run — it is the same opponent in every run — so it doubles as the landmark
    for reading where a policy actually got to.
    """
    lines = []
    for player in result.get("players", []):
        if player["label"] == "scripted" and np.isfinite(player.get(key, float("nan"))):
            lines.append(("scripted agent", float(player[key])))
    return lines


def plot_live_curve(result: dict, path: Path) -> Path:
    """Live rating over training: what the run believed, versus what it was worth.

    The lower panel is the difference between them on a shared x-axis — a second
    panel rather than a second y-scale, so the two measures are never read off
    one set of gridlines.
    """
    curve = result["curve"]
    steps = _finite(curve, "global_step") / 1e6
    training = _finite(curve, "live_training")
    calibrated = _finite(curve, "live_calibrated")
    stderr = _finite(curve, "live_stderr")
    good = _measured(calibrated, stderr)

    figure = _new_figure((11.0, 7.2))
    grid = figure.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.28)
    axes = figure.add_subplot(grid[0])
    lower = figure.add_subplot(grid[1], sharex=axes)

    _style_axes(
        axes,
        f"Live ELO: in-training estimate vs calibrated  —  {result['run']}",
        "Each update's rating refit from its own win/loss record against "
        f"post-hoc measured opponents (reference: {result.get('reference', 'n/a')})",
    )
    _draw_reference_lines(axes, _reference_lines(result))
    axes.fill_between(
        steps[good],
        calibrated[good] - stderr[good],
        calibrated[good] + stderr[good],
        color=CALIBRATED,
        alpha=0.16,
        linewidth=0,
        zorder=2,
    )
    axes.plot(steps, training, color=TRAINING, linewidth=2.0, zorder=3, label="In-training")
    axes.plot(
        steps[good],
        calibrated[good],
        color=CALIBRATED,
        linewidth=2.0,
        zorder=4,
        label="Calibrated (±1 SE)",
    )
    axes.set_ylabel("ELO vs random anchor", color=INK_SECONDARY, fontsize=10)
    axes.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=10, loc="lower right")
    # Direct labels at the right edge, so identity survives without the legend.
    ends = []
    for values, color, name in (
        (training, TRAINING, "in-training"),
        (np.where(good, calibrated, np.nan), CALIBRATED, "calibrated"),
    ):
        finite = np.isfinite(values)
        if finite.any():
            ends.append((steps[finite][-1], values[finite][-1], color, name))
    _label_series_ends(axes, ends)

    axes.tick_params(labelbottom=False)  # x is labelled once, on the lower panel

    offset = result.get("anchor_offset_stderr")
    _style_axes(
        lower,
        subtitle="Gap between the two curves — a near-constant offset means the shape is right "
        + (f"(the scale's zero carries ±{offset:.0f} in common)" if offset else ""),
    )
    drift = np.where(good, calibrated - training, np.nan)
    lower.axhline(0.0, color=INK_MUTED, linewidth=1.0, zorder=2)
    lower.plot(steps, drift, color=INK_SECONDARY, linewidth=1.6, zorder=3)
    lower.fill_between(steps, 0.0, drift, color=INK_SECONDARY, alpha=0.13, linewidth=0, zorder=2)
    lower.set_ylabel("calibrated − in-training", color=INK_SECONDARY, fontsize=9)
    lower.set_xlabel("environment steps (millions)", color=INK_SECONDARY, fontsize=10)

    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_avg_curve(result: dict, path: Path) -> Path:
    """The averaged policy's climb, in-training against calibrated.

    The avg model is rated in a second stage: its only opponent is the live
    policy, which is itself non-stationary, so it can only be placed once the
    live rating for that same update is known. Its errors are correspondingly
    wider than the live curve's, and it starts only once avg accumulation
    switches on.
    """
    curve = result["curve"]
    steps = _finite(curve, "global_step") / 1e6
    training = _finite(curve, "avg_training")
    calibrated = _finite(curve, "avg_calibrated")
    stderr = _finite(curve, "avg_stderr")
    good = _measured(calibrated, stderr)
    if not good.any():
        return path

    figure = _new_figure((11.0, 6.0))
    axes = figure.add_subplot(111)
    _style_axes(
        axes,
        f"Averaged-policy ELO: in-training vs calibrated  —  {result['run']}",
        "Rated through the live policy it plays, so its error is the live curve's plus its own",
    )
    _draw_reference_lines(axes, _reference_lines(result))
    axes.fill_between(
        steps[good],
        calibrated[good] - stderr[good],
        calibrated[good] + stderr[good],
        color=CALIBRATED,
        alpha=0.16,
        linewidth=0,
        zorder=3,
    )
    ends = []
    training_good = np.isfinite(training) & (steps >= steps[good].min())
    axes.plot(
        steps[training_good],
        training[training_good],
        color=TRAINING,
        linewidth=2.0,
        zorder=4,
        label="In-training",
    )
    axes.plot(
        steps[good],
        calibrated[good],
        color=CALIBRATED,
        linewidth=2.0,
        zorder=5,
        label="Calibrated (±1 SE)",
    )
    if training_good.any():
        ends.append(
            (steps[training_good][-1], training[training_good][-1], TRAINING, "in-training")
        )
    ends.append((steps[good][-1], calibrated[good][-1], CALIBRATED, "calibrated"))
    _label_series_ends(axes, ends)
    axes.set_xlabel("environment steps (millions)", color=INK_SECONDARY, fontsize=10)
    axes.set_ylabel("ELO vs random anchor", color=INK_SECONDARY, fontsize=10)
    axes.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=10, loc="lower right")
    axes.margins(x=0.06)

    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_live_and_avg(result: dict, path: Path) -> Path:
    """Both policies' calibrated climbs on one scale.

    The gap between them is the averaging lag: avg trails the live policy by
    however long its window takes to catch up, and closing that gap late in a
    run means the live policy has stopped improving fast enough to outrun it.
    """
    curve = result["curve"]
    steps = _finite(curve, "global_step") / 1e6
    series = (
        (
            _finite(curve, "live_calibrated"),
            _finite(curve, "live_stderr"),
            CALIBRATED,
            "live",
            "Live policy",
        ),
        (
            _finite(curve, "avg_calibrated"),
            _finite(curve, "avg_stderr"),
            AVG,
            "avg",
            "Averaged policy",
        ),
    )
    if not any(_measured(values, error).any() for values, error, *_ in series):
        return path

    figure = _new_figure((11.0, 6.0))
    axes = figure.add_subplot(111)
    _style_axes(
        axes,
        f"Calibrated ELO: live vs averaged policy  —  {result['run']}",
        f"Both refit from their own records ({_TIE_LABEL.get(result.get('tie_mode', ''), '')})",
    )
    _draw_reference_lines(axes, _reference_lines(result))
    ends = []
    for values, error, color, short, label in series:
        good = _measured(values, error)
        if not good.any():
            continue
        axes.fill_between(
            steps[good],
            values[good] - error[good],
            values[good] + error[good],
            color=color,
            alpha=0.15,
            linewidth=0,
            zorder=3,
        )
        axes.plot(steps[good], values[good], color=color, linewidth=2.0, zorder=4, label=label)
        ends.append((steps[good][-1], values[good][-1], color, short))
    _label_series_ends(axes, ends)
    axes.set_xlabel("environment steps (millions)", color=INK_SECONDARY, fontsize=10)
    axes.set_ylabel("ELO vs random anchor", color=INK_SECONDARY, fontsize=10)
    axes.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=10, loc="lower right")
    axes.margins(x=0.06)

    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_tie_conventions(result: dict, path: Path) -> Path:
    """The same records rated under both draw conventions, against the run's own curve.

    The in-training filter scores a draw as half a win, so the half-win refit is
    the like-for-like comparison: whatever still separates it from the run's own
    curve is genuine estimator error rather than a difference of scale.

    The decisive-only curve alongside it is the draw-farming check. A policy that
    survives to the horizon rather than winning earns parity under half-win
    scoring without ever beating anyone; it cannot earn it under decisive-only.
    The two tracking together means draws are being earned honestly.
    """
    curve = result["curve"]
    if not any("live_calibrated_alt" in point for point in curve):
        return path
    primary = result.get("tie_mode", "half_win")
    alt = result.get("tie_mode_alt", "decisive")
    steps = _finite(curve, "global_step") / 1e6
    # The in-training curve has no standard error of its own — it is the filter's
    # running state, defined at every update — so it is drawn unfiltered.
    always = np.isfinite(_finite(curve, "live_training"))
    series = [
        (
            _finite(curve, "live_training"),
            always,
            TRAINING,
            "in-training",
            "In-training (ties = ½ win)",
        ),
    ]
    for mode, value_key, error_key in (
        (primary, "live_calibrated", "live_stderr"),
        (alt, "live_calibrated_alt", "live_stderr_alt"),
    ):
        values = _finite(curve, value_key)
        color = CALIBRATED if mode == "half_win" else HALF_WIN
        series.append(
            (
                values,
                _measured(values, _finite(curve, error_key)),
                color,
                f"calibrated, {_TIE_LABEL.get(mode, mode)}",
                f"Calibrated, {_TIE_LABEL.get(mode, mode)}",
            )
        )

    figure = _new_figure((11.0, 6.4))
    axes = figure.add_subplot(111)
    _style_axes(
        axes,
        f"Calibrated ELO under both draw conventions  —  {result['run']}",
        "Ties as half a win puts the calibrated curve on the same scale the run "
        "itself used; dropping them rescales the anchor",
    )
    _draw_reference_lines(axes, _reference_lines(result))
    ends = []
    for values, good, color, short, label in series:
        axes.plot(steps[good], values[good], color=color, linewidth=2.0, zorder=3, label=label)
        if good.any():
            ends.append((steps[good][-1], values[good][-1], color, short))
    _label_series_ends(axes, ends)
    axes.set_xlabel("environment steps (millions)", color=INK_SECONDARY, fontsize=10)
    axes.set_ylabel("ELO vs random anchor", color=INK_SECONDARY, fontsize=10)
    axes.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=10, loc="lower right")
    axes.margins(x=0.1)

    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_calibrated_only(result: dict, path: Path, tie_mode: str) -> Path:
    """The calibrated curve on its own, with no in-training line to compare to.

    One series, so no legend — the title names it. Useful when the calibrated
    rating is the answer rather than one side of a comparison; the in-training
    curve's offset otherwise compresses the y-range and hides the shape.
    """
    is_primary = tie_mode == result.get("tie_mode", "half_win")
    key = "live_calibrated" if is_primary else "live_calibrated_alt"
    error_key = "live_stderr" if is_primary else "live_stderr_alt"
    curve = result["curve"]
    if not any(key in point for point in curve):
        return path
    steps = _finite(curve, "global_step") / 1e6
    values = _finite(curve, key)
    stderr = _finite(curve, error_key)
    good = _measured(values, stderr)
    if not good.any():
        return path

    figure = _new_figure((11.0, 6.0))
    axes = figure.add_subplot(111)
    _style_axes(
        axes,
        f"Calibrated ELO — {_TIE_LABEL.get(tie_mode, tie_mode)}  —  {result['run']}",
        f"Refit from each update's own record; shaded band is ±1 SE "
        f"(ratings pinned to ±{result['target_stderr']:.0f} against "
        f"{result.get('reference', 'the reference')})",
    )
    reference_key = "calibrated_elo" if is_primary else "calibrated_elo_alt"
    _draw_reference_lines(axes, _reference_lines(result, reference_key))
    color = CALIBRATED if tie_mode == "half_win" else HALF_WIN
    axes.fill_between(
        steps[good],
        values[good] - stderr[good],
        values[good] + stderr[good],
        color=color,
        alpha=0.18,
        linewidth=0,
        zorder=3,
    )
    axes.plot(steps[good], values[good], color=color, linewidth=2.0, zorder=4)
    axes.set_xlabel("environment steps (millions)", color=INK_SECONDARY, fontsize=10)
    axes.set_ylabel("ELO vs random anchor", color=INK_SECONDARY, fontsize=10)
    axes.margins(x=0.02)

    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_checkpoint_ratings(result: dict, path: Path) -> Path:
    """Per-checkpoint before/after — a dumbbell, one hue in two shades."""
    # The random anchor is excluded: it defines the zero, so its "before and
    # after" is 0 to 0 by construction and says nothing about the run.
    players = [
        p for p in result["players"] if p["training_elo"] is not None and p["label"] != "random"
    ]
    players.sort(key=lambda p: p["calibrated_elo"])
    if not players:
        return path
    labels = [p["label"] for p in players]
    training = np.array([p["training_elo"] for p in players])
    calibrated = np.array([p["calibrated_elo"] for p in players])
    stderr = np.array([p["stderr"] for p in players])
    positions = np.arange(len(players))

    figure = _new_figure((10.5, 0.52 * len(players) + 2.6))
    axes = figure.add_subplot(111)
    _style_axes(
        axes,
        f"Ladder checkpoint ratings: in-training vs calibrated  —  {result['run']}",
        "Each rung as the run recorded it, and as a full tournament measures it; "
        f"errors are relative to {result.get('reference', 'the reference')}",
    )
    for position, start, end in zip(positions, training, calibrated):
        axes.plot([start, end], [position, position], color=GRID, linewidth=2.4, zorder=2)
    axes.scatter(
        training,
        positions,
        s=95,
        color=SHADE_LIGHT,
        zorder=3,
        edgecolors=SURFACE,
        linewidths=2,
        label="In-training",
    )
    axes.errorbar(
        calibrated,
        positions,
        xerr=stderr,
        fmt="o",
        markersize=9.5,
        color=SHADE_DARK,
        ecolor=SHADE_DARK,
        elinewidth=1.6,
        capsize=3.5,
        markeredgecolor=SURFACE,
        markeredgewidth=2,
        zorder=4,
        label="Calibrated (±1 SE)",
    )
    # Anchored past the error bar's right cap so the delta never sits on it.
    for position, start, end, error in zip(positions, training, calibrated, stderr):
        axes.annotate(
            f"{end - start:+.0f}",
            (max(start, end + error), position),
            xytext=(12, 0),
            textcoords="offset points",
            color=INK_SECONDARY,
            fontsize=9,
            va="center",
        )
    axes.set_yticks(positions)
    axes.set_yticklabels(labels, color=INK_SECONDARY, fontsize=10)
    axes.set_xlabel("ELO vs random anchor", color=INK_SECONDARY, fontsize=10)
    axes.legend(
        frameon=False,
        labelcolor=INK_SECONDARY,
        fontsize=10,
        loc="lower right",
        bbox_to_anchor=(1.0, -0.02),
    )
    axes.margins(x=0.16, y=0.12)

    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_convergence(result: dict, path: Path) -> Path:
    """How fast the adaptive tournament pinned the ratings down."""
    batches = result["batches"]
    if not batches:
        return path
    games = np.array([b["cumulative_games"] for b in batches], dtype=float)
    worst = np.array([b["max_stderr"] for b in batches], dtype=float)
    mean = np.array([b["mean_stderr"] for b in batches], dtype=float)

    figure = _new_figure((10.0, 5.6))
    axes = figure.add_subplot(111)
    _style_axes(
        axes,
        "Tournament convergence",
        "Standard error after each adaptive batch; games are allocated to whichever "
        "pairings reduce total rating variance most",
    )
    target = result["target_stderr"]
    axes.axhline(target, color=INK_MUTED, linewidth=1.2, linestyle=(0, (5, 4)), zorder=2)
    axes.annotate(
        f"target ±{target:.0f}",
        (games[-1], target),
        xytext=(0, 6),
        textcoords="offset points",
        color=INK_MUTED,
        fontsize=9,
        ha="right",
    )
    axes.plot(
        games,
        worst,
        color=TRAINING,
        linewidth=2.0,
        marker="o",
        markersize=6.5,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        zorder=4,
        label="Worst rating",
    )
    axes.plot(
        games,
        mean,
        color=CALIBRATED,
        linewidth=2.0,
        marker="o",
        markersize=6.5,
        markeredgecolor=SURFACE,
        markeredgewidth=1.5,
        zorder=3,
        label="Mean rating",
    )
    for values, color, name in ((worst, TRAINING, "worst"), (mean, CALIBRATED, "mean")):
        axes.annotate(
            name,
            (games[-1], values[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            color=color,
            fontsize=9.5,
            va="center",
            fontweight="medium",
        )
    axes.set_xlabel("cumulative games played", color=INK_SECONDARY, fontsize=10)
    axes.set_ylabel("standard error (ELO)", color=INK_SECONDARY, fontsize=10)
    axes.set_yscale("log")
    axes.legend(frameon=False, labelcolor=INK_SECONDARY, fontsize=10, loc="upper right")
    axes.margins(x=0.12)

    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    return path


def plot_tie_rates(result: dict, path: Path) -> Path:
    """Draw frequency against the level of the matchup, not its gap.

    This is the evidence for excluding draws from the likelihood: if ties tracked
    the rating gap, the standard tie-aware models would apply. They track the
    absolute level instead, which those models cannot represent.
    """
    rows = result.get("tie_rates") or []
    training_rows = result.get("training_tie_rates") or []
    if not rows and not training_rows:
        return path

    def _series(source: list[dict]) -> tuple[np.ndarray, ...]:
        return (
            np.array([r["mean_rating"] for r in source], dtype=float),
            np.array([r["rating_gap"] for r in source], dtype=float),
            np.array([r["tie_rate"] for r in source], dtype=float) * 100.0,
            np.array([r["games"] for r in source], dtype=float),
        )

    figure = _new_figure((11.5, 5.8))
    grid = figure.add_gridspec(1, 2, wspace=0.2)
    left = figure.add_subplot(grid[0])
    right = figure.add_subplot(grid[1])
    _style_axes(
        left,
        "Draw rate vs matchup level",
        "Draws concentrate at the weak end, whatever the gap",
    )
    _style_axes(right, "Draw rate vs rating gap", "The same points, against difference instead")

    # Tournament pairs cover the trained ladder; the training record is what
    # reaches down to the near-random level, where every draw actually happened.
    for source, color, name in (
        (rows, TRAINING, "post-hoc tournament"),
        (training_rows, CALIBRATED, "during training"),
    ):
        if not source:
            continue
        level, gap, rate, games = _series(source)
        size = np.clip(games / max(games.max(), 1.0) * 150, 18, 170)
        for axis, x in ((left, level), (right, gap)):
            axis.scatter(
                x,
                rate,
                s=size,
                color=color,
                alpha=0.75,
                edgecolors=SURFACE,
                linewidths=1.2,
                zorder=3,
                label=name,
            )
    left.set_xlabel("mean rating of the pair (ELO)", color=INK_SECONDARY, fontsize=10)
    left.set_ylabel("draws (% of games)", color=INK_SECONDARY, fontsize=10)
    right.set_xlabel("rating gap within the pair (ELO)", color=INK_SECONDARY, fontsize=10)
    if rows and training_rows:
        left.legend(
            frameon=False,
            labelcolor=INK_SECONDARY,
            fontsize=9.5,
            loc="upper right",
            markerscale=0.7,
        )

    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    return path


def write_plots(result: dict, run_dir: Path, plot_decisive: bool = False) -> list[Path]:
    """Render the calibration plots into the run directory.

    The secondary draw convention's charts are off by default. Both conventions
    are still fit and written to JSON — they are a diagnostic for draw farming,
    not a result, and rendering them alongside the primary curve invites reading
    two different scales as though they disagreed about the same quantity.
    """
    output = run_dir / "elo_calibration"
    output.mkdir(parents=True, exist_ok=True)
    written = []
    primary = result.get("tie_mode", "half_win")
    renders = [
        ("live_curve.png", plot_live_curve),
        ("avg_curve.png", plot_avg_curve),
        ("live_and_avg.png", plot_live_and_avg),
        (f"calibrated_{primary}.png", partial(plot_calibrated_only, tie_mode=primary)),
        ("checkpoint_ratings.png", plot_checkpoint_ratings),
        ("convergence.png", plot_convergence),
        ("tie_rates.png", plot_tie_rates),
    ]
    if plot_decisive:
        alt = result.get("tie_mode_alt", "decisive")
        renders.extend(
            [
                (f"calibrated_{alt}.png", partial(plot_calibrated_only, tie_mode=alt)),
                ("tie_conventions.png", plot_tie_conventions),
            ]
        )
    for name, render in renders:
        path = render(result, output / name)
        if path.exists():
            written.append(path)
    return written
