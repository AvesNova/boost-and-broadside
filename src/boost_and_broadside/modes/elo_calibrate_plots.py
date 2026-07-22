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

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#83817a"
GRID = "#e3e2de"
# Categorical slots 1-3, in fixed palette order. Slot 3 sits below 3:1 against
# the light surface, so every series it appears with is also direct-labelled.
TRAINING = "#2a78d6"  # blue
CALIBRATED = "#008300"  # green
HALF_WIN = "#e87ba4"  # magenta
# One-hue shades for the before/after dumbbell.
SHADE_LIGHT = "#86b6ef"
SHADE_DARK = "#184f95"


def _style_axes(axes, title: str = "", subtitle: str = "") -> None:
    """Recessive grid and axes; titles in ink, never in a series colour."""
    axes.set_facecolor(SURFACE)
    axes.grid(True, color=GRID, linewidth=0.8, zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(GRID)
    axes.tick_params(colors=INK_SECONDARY, labelsize=9, length=0)
    if title:
        # Pad leaves room for the subtitle, which is offset in points rather than
        # axes fractions so the gap does not shrink as the panel gets shorter.
        axes.set_title(
            title, color=INK, fontsize=13, fontweight="semibold", loc="left",
            pad=30 if subtitle else 12,
        )
    if subtitle:
        axes.annotate(
            subtitle, xy=(0.0, 1.0), xycoords="axes fraction",
            xytext=(0, 7), textcoords="offset points",
            color=INK_MUTED, fontsize=9.5, va="bottom",
        )


# A rating this uncertain is not a measurement — a +/-150 band spans a quarter of
# the whole ladder. These arise early, when the live policy has completed only a
# game or two against an opponent it can actually contest: the fit is real but
# rests on almost nothing, and plotting it puts a wild outlier at the left edge
# that stretches the axis and reads as if the run started far from the anchor.
# Such points stay in the JSON with their error, and are simply not drawn.
_MAX_PLOT_STDERR = 150.0


def _finite(curve: list[dict], key: str) -> np.ndarray:
    return np.array([point.get(key, float("nan")) for point in curve], dtype=float)


def _measured(values: np.ndarray, stderr: np.ndarray) -> np.ndarray:
    """Mask of points precise enough to plot as a measurement."""
    return np.isfinite(values) & np.isfinite(stderr) & (stderr <= _MAX_PLOT_STDERR)


def _label_series_ends(axes, entries: list[tuple[float, float, str, str]]) -> None:
    """Direct-label each series past its right end, nudging apart any that collide.

    Curves that converge — which is the interesting case here, since the whole
    point is comparing scales that should agree — would otherwise stack their
    labels on one another and render both unreadable. Call after all series are
    drawn so the y-limits are settled.
    """
    if not entries:
        return
    low, high = axes.get_ylim()
    minimum = 0.042 * (high - low or 1.0)
    placed: list[float] = []
    for x, y, color, text in sorted(entries, key=lambda entry: entry[1]):
        target = y if not placed else max(y, placed[-1] + minimum)
        placed.append(target)
        axes.annotate(
            text, (x, target), xytext=(7, 0), textcoords="offset points",
            color=color, fontsize=9.5, va="center", fontweight="medium",
        )


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


def _draw_reference_lines(axes, lines: list[tuple[str, float]]) -> None:
    """Draw landmark ratings as recessive dashed rules, labelled in place."""
    for name, value in lines:
        axes.axhline(value, color=INK_MUTED, linewidth=1.2, linestyle=(0, (5, 4)), zorder=2)
        axes.annotate(
            f"{name}  {value:.0f}",
            xy=(0.008, value), xycoords=("axes fraction", "data"),
            xytext=(0, 4), textcoords="offset points",
            color=INK_MUTED, fontsize=9, va="bottom",
        )


def _new_figure(size=(11.0, 6.0)):
    figure = plt.figure(figsize=size, dpi=160, facecolor=SURFACE)
    return figure


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
        color=CALIBRATED, alpha=0.16, linewidth=0, zorder=2,
    )
    axes.plot(steps, training, color=TRAINING, linewidth=2.0, zorder=3, label="In-training")
    axes.plot(steps[good], calibrated[good], color=CALIBRATED, linewidth=2.0, zorder=4,
              label="Calibrated (±1 SE)")
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


def plot_tie_conventions(result: dict, path: Path) -> Path:
    """The same records rated under both draw conventions, against the run's own curve.

    The in-training filter scores a draw as half a win, so the half-win refit is
    the like-for-like comparison: whatever still separates it from the run's own
    curve is genuine estimator error rather than a difference of scale. The
    decisive-only curve is shown alongside to size what the convention is worth.
    """
    curve = result["curve"]
    if not any("live_calibrated_half_win" in point for point in curve):
        return path
    steps = _finite(curve, "global_step") / 1e6
    # The in-training curve has no standard error of its own — it is the filter's
    # running state, defined at every update — so it is drawn unfiltered.
    always = np.isfinite(_finite(curve, "live_training"))
    series = (
        (
            _finite(curve, "live_training"), always,
            TRAINING, "in-training", "In-training (ties = ½ win)",
        ),
        (
            _finite(curve, "live_calibrated_half_win"),
            _measured(
                _finite(curve, "live_calibrated_half_win"),
                _finite(curve, "live_stderr_half_win"),
            ),
            HALF_WIN, "calibrated, ties = ½", "Calibrated, ties as ½ win",
        ),
        (
            _finite(curve, "live_calibrated"),
            _measured(_finite(curve, "live_calibrated"), _finite(curve, "live_stderr")),
            CALIBRATED, "calibrated, decisive", "Calibrated, decisive games only",
        ),
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
    decisive = tie_mode == "decisive"
    key = "live_calibrated" if decisive else "live_calibrated_half_win"
    error_key = "live_stderr" if decisive else "live_stderr_half_win"
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
        f"Calibrated ELO — {'decisive games only' if decisive else 'ties as ½ win'}"
        f"  —  {result['run']}",
        f"Refit from each update's own record; shaded band is ±1 SE "
        f"(ratings pinned to ±{result['target_stderr']:.0f} against "
        f"{result.get('reference', 'the reference')})",
    )
    reference_key = "calibrated_elo" if decisive else "calibrated_elo_half_win"
    _draw_reference_lines(axes, _reference_lines(result, reference_key))
    color = CALIBRATED if decisive else HALF_WIN
    axes.fill_between(
        steps[good], values[good] - stderr[good], values[good] + stderr[good],
        color=color, alpha=0.18, linewidth=0, zorder=3,
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
    axes.scatter(training, positions, s=95, color=SHADE_LIGHT, zorder=3,
                 edgecolors=SURFACE, linewidths=2, label="In-training")
    axes.errorbar(
        calibrated, positions, xerr=stderr, fmt="o", markersize=9.5,
        color=SHADE_DARK, ecolor=SHADE_DARK, elinewidth=1.6, capsize=3.5,
        markeredgecolor=SURFACE, markeredgewidth=2, zorder=4, label="Calibrated (±1 SE)",
    )
    # Anchored past the error bar's right cap so the delta never sits on it.
    for position, start, end, error in zip(positions, training, calibrated, stderr):
        axes.annotate(
            f"{end - start:+.0f}",
            (max(start, end + error), position), xytext=(12, 0), textcoords="offset points",
            color=INK_SECONDARY, fontsize=9, va="center",
        )
    axes.set_yticks(positions)
    axes.set_yticklabels(labels, color=INK_SECONDARY, fontsize=10)
    axes.set_xlabel("ELO vs random anchor", color=INK_SECONDARY, fontsize=10)
    axes.legend(
        frameon=False, labelcolor=INK_SECONDARY, fontsize=10,
        loc="lower right", bbox_to_anchor=(1.0, -0.02),
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
        f"target ±{target:.0f}", (games[-1], target), xytext=(0, 6),
        textcoords="offset points", color=INK_MUTED, fontsize=9, ha="right",
    )
    axes.plot(games, worst, color=TRAINING, linewidth=2.0, marker="o", markersize=6.5,
              markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=4, label="Worst rating")
    axes.plot(games, mean, color=CALIBRATED, linewidth=2.0, marker="o", markersize=6.5,
              markeredgecolor=SURFACE, markeredgewidth=1.5, zorder=3, label="Mean rating")
    for values, color, name in ((worst, TRAINING, "worst"), (mean, CALIBRATED, "mean")):
        axes.annotate(name, (games[-1], values[-1]), xytext=(8, 0), textcoords="offset points",
                      color=color, fontsize=9.5, va="center", fontweight="medium")
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
                x, rate, s=size, color=color, alpha=0.75,
                edgecolors=SURFACE, linewidths=1.2, zorder=3, label=name,
            )
    left.set_xlabel("mean rating of the pair (ELO)", color=INK_SECONDARY, fontsize=10)
    left.set_ylabel("draws (% of games)", color=INK_SECONDARY, fontsize=10)
    right.set_xlabel("rating gap within the pair (ELO)", color=INK_SECONDARY, fontsize=10)
    if rows and training_rows:
        left.legend(
            frameon=False, labelcolor=INK_SECONDARY, fontsize=9.5,
            loc="upper right", markerscale=0.7,
        )

    figure.tight_layout()
    figure.savefig(path, facecolor=SURFACE, bbox_inches="tight")
    plt.close(figure)
    return path


def write_plots(result: dict, run_dir: Path) -> list[Path]:
    """Render every calibration plot into the run directory."""
    output = run_dir / "elo_calibration"
    output.mkdir(parents=True, exist_ok=True)
    written = []
    renders = [
        ("live_curve.png", plot_live_curve),
        ("calibrated_decisive.png", partial(plot_calibrated_only, tie_mode="decisive")),
        ("calibrated_half_win.png", partial(plot_calibrated_only, tie_mode="half_win")),
        ("tie_conventions.png", plot_tie_conventions),
        ("checkpoint_ratings.png", plot_checkpoint_ratings),
        ("convergence.png", plot_convergence),
        ("tie_rates.png", plot_tie_rates),
    ]
    for name, render in renders:
        path = render(result, output / name)
        if path.exists():
            written.append(path)
    return written
