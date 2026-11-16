"""The arms forest plot: one row per pairwise arm decision, the arm's score against its
champion with its 95% interval, the zero line, the indifference zone (plus or minus delta)
shaded when known, the decision marked (kept, rejected, pending), and a last row for the
seed-to-seed noise (the same settings at two seeds).

The score is the decision suite's (`bnb.league.evaluation`): the arm's overall expected
score against the champion less one half (win probability, a draw half a win), from the
arm's side.

**Input** (`read_rows`): a JSON file {"title"?, "delta"?, "rows": [...]} or a CSV file of
the rows, one row per decision in the order drawn (top first):

- `name`: the row's label; `key`: its name in the values file (default: `name` slugged);
- `kind`: "arm" (default) or "seed_noise";
- `score`, `stderr`: the arm's score and its standard error; or
- `evaluation`: a `bnb evaluate` directory (relative to the input file), read with
  `bnb.league.evaluation.decide` without recording a look: the last look's score, standard
  error and decision, and the suite's delta; with `arm_side` "A" (default) or "B", the side
  the arm played;
- `decision`: "kept", "rejected" or "pending"; from an evaluation, the arm's side winning is
  kept, the champion's winning or a tie rejected, "continue" pending.
"""

import csv
import json
import math
import pathlib

from .. import theme
from . import common

Z95 = 1.959964
DECISIONS = ("kept", "rejected", "pending")
MARKS = {
    "kept": ("o", theme.SECOND, True),
    "rejected": ("X", theme.TRUE, True),
    "pending": ("o", theme.BASELINE, False),
}


def _from_evaluation(row: dict, base: pathlib.Path) -> dict:
    """A row's score, standard error, decision and delta from its evaluation directory."""
    from bnb.league import evaluation

    directory = pathlib.Path(row["evaluation"])
    directory = directory if directory.is_absolute() else base / directory
    _, config = evaluation.read_settings(directory)
    summary, result, _ = evaluation.decide(directory, record=False)
    side = row.get("arm_side") or "A"
    sign = 1.0 if side == "A" else -1.0
    found = dict(delta=config.delta)
    if summary.overall is not None:
        found["score"] = sign * summary.overall[0]
        found["stderr"] = summary.overall[1]
    if result is not None and not row.get("decision"):
        found["decision"] = (
            "pending"
            if result.decision == "continue"
            else "kept"
            if result.decision == side
            else "rejected"
        )
    return found


def read_rows(path: str | pathlib.Path) -> tuple[list[dict], dict]:
    """The rows of a forest input (JSON or CSV, see the module docstring), each with name,
    key, kind, score, stderr (None while unknown) and decision; and the file's settings
    (title, delta)."""
    path = pathlib.Path(path)
    if path.suffix == ".csv":
        with path.open() as handle:
            raw, settings = list(csv.DictReader(handle)), {}
    else:
        held = json.loads(path.read_text())
        raw, settings = held["rows"], {k: v for k, v in held.items() if k != "rows"}
    rows = []
    for given in raw:
        row = {k: v for k, v in given.items() if v not in (None, "")}
        if "evaluation" in row:
            found = _from_evaluation(row, path.parent)
            settings.setdefault("delta", found.pop("delta"))
            row = {**found, **row}
        kind = row.get("kind", "arm")
        if kind not in ("arm", "seed_noise"):
            raise ValueError(f"row {row.get('name')!r}: kind {kind!r}, arm or seed_noise")
        decision = row.get("decision", "pending" if kind == "arm" else None)
        if kind == "arm" and decision not in DECISIONS:
            raise ValueError(f"row {row.get('name')!r}: decision {decision!r}, one of {DECISIONS}")
        rows.append(
            dict(
                name=row["name"],
                key=row.get("key") or common.slug(row["name"]),
                kind=kind,
                score=None if row.get("score") is None else float(row["score"]),
                stderr=None if row.get("stderr") is None else float(row["stderr"]),
                decision=decision,
            )
        )
    return rows, settings


def draw(
    path: str | pathlib.Path,
    out: str | pathlib.Path,
    formats: tuple[str, ...] = common.FORMATS,
    name: str = "arms_forest",
) -> tuple[list[pathlib.Path], dict]:
    """Draw the forest plot of the input at `path` into `out`; (the files written, the
    values: per row its score and 95% interval)."""
    rows, settings = read_rows(path)
    if not rows:
        raise SystemExit(f"no rows in {path}")
    arms = [row for row in rows if row["kind"] == "arm"]
    noise = [row for row in rows if row["kind"] == "seed_noise"]
    ordered = [*arms, *noise]
    height = 1.5 + 0.38 * len(ordered)
    figure, axes = common.new_figure(1, 1, size=(6.4, height))
    axis = axes[0, 0]
    axis.grid(False, axis="y")
    delta = settings.get("delta")
    if delta:
        axis.axvspan(-delta, delta, color=theme.GRID, alpha=0.7, linewidth=0, zorder=0)
    axis.axvline(0.0, color=theme.MUTED, linewidth=0.9)
    values: dict[str, float] = {}
    positions = list(range(len(ordered)))[::-1]
    for y, row in zip(positions, ordered, strict=True):
        if row["kind"] == "seed_noise":
            marker, colour, filled = ("D", theme.INK, False)
        else:
            marker, colour, filled = MARKS[row["decision"]]
        if row["score"] is None:
            axis.annotate(
                "no estimate yet",
                (0.0, y),
                xytext=(6, 0),
                textcoords="offset points",
                fontsize=7,
                color=theme.MUTED,
                va="center",
            )
            continue
        half = Z95 * (row["stderr"] or 0.0)
        axis.plot(
            [row["score"] - half, row["score"] + half],
            [y, y],
            color=colour,
            linewidth=1.6,
            solid_capstyle="butt",
        )
        axis.scatter(
            [row["score"]],
            [y],
            marker=marker,
            s=42,
            zorder=3,
            facecolors=colour if filled else "white",
            edgecolors=colour,
            linewidths=1.2,
        )
        prefix = "seed_noise" if row["kind"] == "seed_noise" else f"arm_{row['key']}"
        values[f"{prefix}_score"] = row["score"]
        values[f"{prefix}_low"] = row["score"] - half
        values[f"{prefix}_high"] = row["score"] + half
    if noise and arms:
        axis.axhline(len(noise) - 0.5, color=theme.MUTED, linewidth=0.6, linestyle="--")
    labels = [
        row["name"] if row["kind"] == "seed_noise" else f"{row['name']}  ({row['decision']})"
        for row in ordered
    ]
    axis.set_yticks(positions, labels, fontsize=8)
    axis.set_ylim(-0.7, len(ordered) - 0.3)
    widest = max(
        (abs(r["score"]) + Z95 * (r["stderr"] or 0.0) for r in ordered if r["score"] is not None),
        default=0.05,
    )
    limit = max(widest, delta or 0.0) * 1.15
    axis.set_xlim(-limit, limit)
    axis.set_xlabel(
        "arm's score against its champion (win probability - 1/2), 95% interval", fontsize=8
    )
    used = sorted({row["decision"] for row in arms}, key=DECISIONS.index)
    handles = [
        axis.scatter(
            [],
            [],
            marker=MARKS[d][0],
            s=36,
            facecolors=MARKS[d][1] if MARKS[d][2] else "white",
            edgecolors=MARKS[d][1],
            label=d,
        )
        for d in used
    ]
    if noise:
        handles.append(
            axis.scatter(
                [],
                [],
                marker="D",
                s=30,
                facecolors="white",
                edgecolors=theme.INK,
                label="seed to seed",
            )
        )
    if delta:
        import matplotlib.patches as patches

        handles.append(patches.Patch(color=theme.GRID, label=f"indifference zone, +-{delta:g}"))
    columns = 3
    figure.legend(handles=handles, fontsize=7, frameon=False, loc="lower center", ncol=columns)
    legend_rows = -(-len(handles) // columns)
    common.title(figure, settings.get("title", "Arm decisions"))
    figure.tight_layout(rect=(0, (0.12 + 0.22 * legend_rows) / height, 1, 1))
    if not math.isfinite(limit):
        raise ValueError("a non-finite score or standard error")
    written = common.save(figure, out, name, [str(path)], values, formats)
    return written, values
