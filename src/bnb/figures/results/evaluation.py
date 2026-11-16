"""Charts from the evaluation queue's records (`box/eval_queue.sh`):

- `elo`: the rating against Imugi (1000) over the chain's steps, one line per mode, from
  `bnb pool curves --out curves.csv` (the 5v5 pool of both stages' scheduled snapshots):
  the exponential average's weights (the averaged agent, the product) solid with its
  standard error, the live weights dotted where the pool holds them, a bound (a rating past
  the pool's reach) an open marker, the hand-off from cloning to RL dashed.
- `suite`: the champion against Imugi on every head-to-head condition of the decision
  suite (`bnb evaluate ... --out DIR`: DIR/batches/*.jsonl), its score (wins and half the
  draws) with a 95% Wilson interval, grouped by mode and team size.
- `averaging`: the averaging sweep (`box/eval_champion.py`'s champion.json): each weight
  average's Elo difference from the incumbent (the training's exponential average), per
  mode and their mean with its standard error, the champion marked.
"""

import csv
import json
import math
import pathlib

from bnb.league.evaluation import read_batches, tallies_of
from bnb.league.evaluation_settings import BY_NAME, HEAD_TO_HEAD

from .. import theme
from ..ladder import BC_SUFFIX, RL_SUFFIX, chain_offsets
from . import common

AVERAGED, LIVE = "exponential_average", "live"
Z95 = 1.959964


def read_curves(path: str | pathlib.Path) -> list[dict]:
    """`bnb pool curves`' rows with numbers parsed (a missing number is None)."""
    rows = []
    with open(path, newline="") as file:
        for row in csv.DictReader(file):
            for key in ("step", "games"):
                row[key] = int(float(row[key])) if row.get(key) else 0
            for key in ("calibrated_elo", "stderr"):
                row[key] = float(row[key]) if row.get(key) not in (None, "") else None
            rows.append(row)
    return rows


def draw_elo(path, out, formats=common.FORMATS, name="elo_over_training"):
    """The Elo chart (see the module); (the files written, the values: per mode the last
    BC snapshot's and the last RL snapshot's averaged rating, with its standard error)."""
    rows = read_curves(path)
    offsets = chain_offsets(rows)
    figure, axes = common.new_figure(1, 1, size=(7.2, 3.6))
    axis = axes[0, 0]
    values: dict[str, float | None] = {}
    lines = []
    handoffs = set()
    for mode in common.MODES:
        mine = [row for row in rows if common.mode_of(row["preset"]) == mode]
        colour = common.mode_colour(mode)
        for kind, style in ((AVERAGED, "-"), (LIVE, ":")):
            points = sorted(
                ((row["step"] + offsets.get(row["run"], 0)) / 1e6, row)
                for row in mine
                if row["kind"] == kind and row["calibrated_elo"] is not None
            )
            if not points:
                continue
            rated = [(x, row) for x, row in points if row["status"] == "rated"]
            xs = [x for x, _ in rated]
            ys = [row["calibrated_elo"] for _, row in rated]
            axis.plot(
                xs,
                ys,
                style,
                color=colour,
                linewidth=1.5 if kind == AVERAGED else 1.0,
                marker="o",
                markersize=2.5,
            )
            if kind == AVERAGED:
                errors = [row["stderr"] or 0.0 for _, row in rated]
                axis.fill_between(
                    xs,
                    [y - e for y, e in zip(ys, errors, strict=True)],
                    [y + e for y, e in zip(ys, errors, strict=True)],
                    color=colour,
                    alpha=0.15,
                    linewidth=0,
                )
                lines.append((mode, xs, ys, colour))
                for x, row in points:
                    if row["status"] != "rated":  # a bound: an open marker
                        axis.plot(
                            [x],
                            [row["calibrated_elo"]],
                            marker="^" if row["status"] == "above" else "v",
                            markerfacecolor="none",
                            markeredgecolor=colour,
                            markersize=5,
                        )
                for suffix, stage in ((BC_SUFFIX, "bc_final"), (RL_SUFFIX, "final")):
                    staged = [row for _, row in points if row["run"].endswith(suffix)]
                    last = max(staged, key=lambda row: row["step"], default=None)
                    rated_last = last if last and last["status"] == "rated" else None
                    values[f"elo_{mode}_{stage}"] = last["calibrated_elo"] if last else None
                    if stage == "final":
                        values[f"elo_{mode}_final_stderr"] = (
                            rated_last["stderr"] if rated_last else None
                        )
        for run, offset in offsets.items():
            if offset and run.endswith(RL_SUFFIX):
                handoffs.add(offset / 1e6)
    for x in handoffs:
        axis.axvline(x, color=theme.MUTED, linestyle="--", linewidth=0.9)
        axis.annotate(
            "cloning | RL",
            (x, 1.0),
            xycoords=("data", "axes fraction"),
            xytext=(0, 2),
            textcoords="offset points",
            ha="center",
            fontsize=7,
            color=theme.MUTED,
        )
    axis.axhline(1000, color=theme.MUTED, linewidth=0.8, linestyle=":")
    axis.set_xlabel("environment steps (M)", fontsize=8)
    axis.set_ylabel("Elo against Imugi (Imugi = 1000)", fontsize=8)
    common.label_ends(axis, lines)
    common.title(figure, "Strength over training, 5 against 5 (the averaged weights; live dotted)")
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    written = common.save(figure, out, name, [str(path)], values, formats)
    return written, values


def wilson(score: float, games: float) -> tuple[float, float, float]:
    """(the rate, its 95% Wilson interval's low and high) of `score` in `games`."""
    if games <= 0:
        return math.nan, math.nan, math.nan
    rate = score / games
    denominator = 1 + Z95 * Z95 / games
    middle = (rate + Z95 * Z95 / (2 * games)) / denominator
    half = Z95 * math.sqrt(rate * (1 - rate) / games + Z95 * Z95 / (4 * games * games))
    return rate, middle - half / denominator, middle + half / denominator


def suite_rows(directory) -> list[dict]:
    """Per head-to-head condition of a suite directory: name, mode, size, score, games and
    the Wilson interval, in the suite's condition order."""
    batches, _, _ = read_batches(pathlib.Path(directory))
    rows = []
    for condition_name, tally in tallies_of(batches).items():
        condition = BY_NAME.get(condition_name)
        if condition is None or condition.kind != HEAD_TO_HEAD:
            continue
        wins, draws, losses = tally.outcomes
        games = wins + draws + losses
        rate, low, high = wilson(wins + 0.5 * draws, games)
        rows.append(
            dict(
                condition=condition_name,
                mode=condition.mode,
                size=condition.ships_per_team,
                score=rate,
                low=low,
                high=high,
                games=games,
                wins=wins,
                draws=draws,
                losses=losses,
            )
        )
    order = list(BY_NAME)
    return sorted(rows, key=lambda row: order.index(row["condition"]))


def draw_suite(directory, out, formats=common.FORMATS, name="suite_conditions"):
    """The suite chart (see the module); (the files written, the values: per condition
    suite_<condition>_score, _low, _high, _games)."""
    rows = suite_rows(directory)
    if not rows:
        raise SystemExit(f"no head-to-head batches in {directory}")
    modes = [mode for mode in common.MODES if any(row["mode"] == mode for row in rows)]
    figure, axes = common.new_figure(1, len(modes), size=(2.6, 2.8), sharey=True)
    values: dict[str, float | None] = {}
    for axis, mode in zip(axes.flat, modes, strict=True):
        mine = sorted((row for row in rows if row["mode"] == mode), key=lambda row: row["size"])
        colour = common.mode_colour(mode)
        for index, row in enumerate(mine):
            axis.plot([index, index], [row["low"], row["high"]], color=colour, linewidth=2.0)
            axis.plot(
                [index],
                [row["score"]],
                "o",
                color=colour,
                markersize=5,
                markeredgecolor=theme.SURFACE,
            )
            axis.annotate(
                f"{row['score']:.0%}",
                (index, row["high"]),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                fontsize=7,
                color=theme.INK,
            )
        axis.set_xticks(range(len(mine)), [f"{r['size']}v{r['size']}" for r in mine], fontsize=8)
        axis.set_xlim(-0.6, len(mine) - 0.4)
        axis.axhline(0.5, color=theme.MUTED, linewidth=0.8, linestyle=":")
        axis.set_title(mode, fontsize=9)
        axis.grid(axis="x", visible=False)
    for row in rows:
        key = f"suite_{row['condition']}"
        values.update(
            {
                f"{key}_score": row["score"],
                f"{key}_low": row["low"],
                f"{key}_high": row["high"],
                f"{key}_games": row["games"],
            }
        )
    axes[0, 0].set_ylim(0, 1.08)
    axes[0, 0].set_ylabel("score against Imugi (wins + half the draws)", fontsize=8)
    common.title(
        figure, "The champion against Imugi on every head-to-head condition (95% intervals)"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.93))
    written = common.save(figure, out, name, [str(directory)], values, formats)
    return written, values


def short_name(name: str) -> str:
    """A pool player's name shortened for a label: its last path part."""
    return name.rstrip("/").split("/")[-1].replace("_", " ")


def draw_averaging(path, out, formats=common.FORMATS, name="averaging_sweep"):
    """The averaging sweep's chart (see the module); (the files written, the values: the
    champion's name, the incumbent, the best gap and its standard error)."""
    found = json.loads(pathlib.Path(path).read_text())
    ranking = found.get("ranking", [])
    figure, axes = common.new_figure(1, 1, size=(6.4, 0.9 + 0.32 * max(len(ranking), 1)))
    axis = axes[0, 0]
    for index, row in enumerate(ranking):
        is_champion = row["name"] == found.get("champion")
        colour = theme.PREDICTED if is_champion else theme.INK
        low, high = row["gap"] - Z95 * row["stderr"], row["gap"] + Z95 * row["stderr"]
        axis.plot([low, high], [index, index], color=colour, linewidth=1.6)
        axis.plot([row["gap"]], [index], "o", color=colour, markersize=5)
        for mode in common.MODES:
            held = next(
                (
                    gap
                    for preset, gap in row.get("presets", {}).items()
                    if common.mode_of(preset) == mode
                ),
                None,
            )
            if held is not None:
                axis.plot(
                    [held[0]],
                    [index],
                    "|",
                    color=common.mode_colour(mode),
                    markersize=8,
                    markeredgewidth=1.6,
                )
    axis.set_yticks(
        range(len(ranking)),
        [
            short_name(row["name"])
            + ("  (champion)" if row["name"] == found.get("champion") else "")
            for row in ranking
        ],
        fontsize=7,
    )
    axis.invert_yaxis()
    axis.axvline(0, color=theme.MUTED, linewidth=0.9, linestyle="--")
    axis.set_xlabel(
        f"Elo difference from the incumbent ({short_name(found.get('incumbent') or '')}), "
        "mean over the modes, 95% interval; ticks per mode",
        fontsize=8,
    )
    from matplotlib.lines import Line2D

    axis.legend(
        [
            Line2D(
                [],
                [],
                marker="|",
                linestyle="none",
                color=common.mode_colour(m),
                markersize=8,
                markeredgewidth=1.6,
            )
            for m in common.MODES
        ],
        list(common.MODES),
        fontsize=7,
        frameon=False,
        loc="lower right",
    )
    common.title(
        figure,
        "The averaging sweep: each weight average against the training's exponential average",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    values = {
        "averaging_best_gap": found.get("gap"),
        "averaging_best_gap_stderr": found.get("gap_stderr"),
        "averaging_candidates": float(len(ranking)),
    }
    written = common.save(figure, out, name, [str(path)], values, formats)
    return written, values


def champion_values(path) -> dict[str, float | str | None]:
    """The champion's values from champion.json: its name, run, steps and weights, read
    from its pool name or policy path, `<run>/policies/step_<steps>/<kind>` or
    `<run>/average/<kind>/step_<steps>`."""
    found = json.loads(pathlib.Path(path).read_text())
    run = steps = kind = None
    for text in (found.get("path") or "", found.get("champion") or ""):
        parts = pathlib.PurePosixPath(text).parts
        for marker in ("policies", "average"):
            if marker not in parts:
                continue
            at = parts.index(marker)
            run = parts[at - 1] if at else run
            rest = parts[at + 1 :]
            numbered = [part for part in rest if part.startswith("step_")]
            if numbered:
                steps = float(numbered[0].removeprefix("step_"))
            names = [part for part in rest if not part.startswith("step_")]
            kind = names[0] if names else kind
        if run is not None:
            break
    return {
        "champion_name": found.get("champion"),
        "champion_run": run,
        "champion_steps": steps,
        "champion_weights": kind,
    }
