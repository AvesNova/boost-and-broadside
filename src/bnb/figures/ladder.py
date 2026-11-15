"""The ladder's curves as one figure of small multiples (`bnb pool curves --figure`): one
panel per rated condition, Elo against environment steps, and optionally a row of
break-even panels (the ratio of Imugi's ships to the policy's at even odds, from
`bnb crossover` results), one per mode.

A chain's two runs, `<name>-bc` and `<name>-rl` (`bnb.runs.chain`), are drawn on one axis
of the chain's steps: the RL run's steps follow its BC run's last rated snapshot, marked by
a dashed line. Each run and weights kind is its own series (the averaged agent and the live
weights are never merged), with its standard errors as bars; a bound (a snapshot whose
games give no finite rating) is an open marker at the bound.
"""

import json
import pathlib
import re

from .output import BASELINE, INK, MUTED, PREDICTED, SECOND, TRUE, new_figure, provenance

BC_SUFFIX, RL_SUFFIX = "-bc", "-rl"
COLOURS = (PREDICTED, TRUE, SECOND, BASELINE)  # the series in the order they first appear
MODES = ("frontline", "deathmatch", "elimination")
# a crossover result's directory: <...>/<run>/step_<steps>/artifacts/crossover/<time>/result.json
BREAK_EVEN_PATH = re.compile(r"([^/]+)/step_(\d+)/artifacts/crossover/[^/]+/result\.json$")


def chain_offsets(rows: list[dict]) -> dict[str, int]:
    """Each run's offset on the chain's axis: an RL run `<name>-rl` whose BC run
    `<name>-bc` has rows starts at that run's last rated steps; every other run at 0."""
    last: dict[str, int] = {}
    for row in rows:
        last[row["run"]] = max(last.get(row["run"], 0), int(row["step"]))
    offsets = {}
    for run in last:
        bc = run.removesuffix(RL_SUFFIX) + BC_SUFFIX
        offsets[run] = last[bc] if run.endswith(RL_SUFFIX) and bc in last else 0
    return offsets


def break_even_points(directory: str | pathlib.Path) -> list[dict]:
    """Every `bnb crossover` result below `directory` laid out as
    `<run>/step_<steps>/artifacts/crossover/<time>/result.json` (the dry run's loop), one
    row per mode with a crossing: run, step, mode, ratio and its standard error. The newest
    result of a snapshot stands."""
    found: dict[tuple[str, int], tuple[str, dict]] = {}
    for path in sorted(pathlib.Path(directory).rglob("result.json")):
        matched = BREAK_EVEN_PATH.search(path.as_posix())
        if matched is None:
            continue
        key = (matched.group(1), int(matched.group(2)))
        time = path.parent.name
        if key not in found or time > found[key][0]:
            found[key] = (time, json.loads(path.read_text()))
    points = []
    for (run, step), (_, result) in sorted(found.items()):
        for preset, measured in result.get("presets", {}).items():
            if measured.get("ratio") is None:
                continue
            mode = next((mode for mode in MODES if preset.endswith(mode)), preset)
            points.append(
                dict(
                    run=run,
                    step=step,
                    mode=mode,
                    ratio=float(measured["ratio"]),
                    stderr=float(measured.get("ratio_stderr") or 0.0),
                )
            )
    return points


def _series_label(run: str, kind: str | None) -> str:
    stage = "BC" if run.endswith(BC_SUFFIX) else "RL" if run.endswith(RL_SUFFIX) else run
    return f"{stage} {(kind or '').replace('_', ' ')}".strip()


def draw(
    rows: list[dict],
    conditions: tuple[str, ...],
    out: str | pathlib.Path,
    break_even: list[dict] | None = None,
    columns: int = 5,
) -> pathlib.Path:
    """Write the figure to `out` (a .png path) from `bnb pool curves`' rows
    (`bnb.league.rating_pool.curves`: preset, run, kind, step, calibrated_elo, stderr,
    status) for `conditions` in their order, and break-even's points (`break_even_points`)
    when given. Returns the path."""
    out = pathlib.Path(out)
    offsets = chain_offsets([*rows, *(break_even or [])])
    series: dict[tuple[str, str | None], str] = {}
    for row in sorted(rows, key=lambda row: (not row["run"].endswith(BC_SUFFIX), row["run"])):
        key = (row["run"], row["kind"])
        if key not in series:
            series[key] = COLOURS[len(series) % len(COLOURS)]
    panels = len(conditions)
    grid_rows = -(-panels // columns) + (1 if break_even else 0)
    figure, axes = new_figure(grid_rows, columns, size=(3.2, 2.5), sharex=True)
    handoffs = sorted({offset for offset in offsets.values() if offset > 0})
    for number, condition in enumerate(conditions):
        axis = axes.flat[number]
        axis.set_title(condition.replace("_", " "), fontsize=9)
        axis.axhline(1000.0, color=MUTED, linewidth=0.8, linestyle=":")
        for (run, kind), colour in series.items():
            points = sorted(
                (row for row in rows if (row["run"], row["kind"]) == (run, kind)),
                key=lambda row: int(row["step"]),
            )
            points = [
                row
                for row in points
                if row["preset"] == condition and row.get("calibrated_elo") not in (None, "")
            ]
            if not points:
                continue
            steps = [(int(row["step"]) + offsets[run]) / 1e6 for row in points]
            elo = [float(row["calibrated_elo"]) for row in points]
            errors = [
                float(row["stderr"]) if row.get("stderr") not in (None, "") else 0.0
                for row in points
            ]
            axis.errorbar(
                steps,
                elo,
                yerr=errors,
                color=colour,
                linewidth=1.5,
                marker="o",
                markersize=3,
                elinewidth=0.8,
                capsize=0,
                label=_series_label(run, kind),
            )
            bounds = [
                (s, e)
                for s, e, row in zip(steps, elo, points, strict=True)
                if row.get("status") not in ("rated", None, "")
            ]
            if bounds:
                axis.scatter(
                    *zip(*bounds, strict=True), facecolors="none", edgecolors=colour, s=30, zorder=3
                )
        for handoff in handoffs:
            axis.axvline(handoff / 1e6, color=MUTED, linewidth=0.8, linestyle="--")
        if number % columns == 0:
            axis.set_ylabel("Elo (Imugi 1000)", fontsize=8)
    for number in range(panels, -(-panels // columns) * columns):
        axes.flat[number].set_visible(False)
    if break_even:
        first = -(-panels // columns) * columns
        for index, mode in enumerate(MODES):
            axis = axes.flat[first + index]
            axis.set_title(f"{mode} break-even", fontsize=9)
            axis.axhline(1.0, color=MUTED, linewidth=0.8, linestyle=":")
            for run in sorted({point["run"] for point in break_even}):
                points = sorted(
                    (p for p in break_even if p["run"] == run and p["mode"] == mode),
                    key=lambda p: p["step"],
                )
                if not points:
                    continue
                colour = next((c for (r, _), c in series.items() if r == run), INK)
                axis.errorbar(
                    [(p["step"] + offsets.get(run, 0)) / 1e6 for p in points],
                    [p["ratio"] for p in points],
                    yerr=[p["stderr"] for p in points],
                    color=colour,
                    linewidth=1.5,
                    marker="o",
                    markersize=3,
                    elinewidth=0.8,
                    capsize=0,
                )
            for handoff in handoffs:
                axis.axvline(handoff / 1e6, color=MUTED, linewidth=0.8, linestyle="--")
            if index == 0:
                axis.set_ylabel("Imugi ships per ours", fontsize=8)
        for number in range(first + len(MODES), grid_rows * columns):
            axes.flat[number].set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("environment steps (M)", fontsize=8)
        axis.tick_params(labelbottom=True)
    legend: dict[str, object] = {}
    for axis in axes.flat:
        for handle, label in zip(*axis.get_legend_handles_labels(), strict=True):
            legend.setdefault(label, handle)
    handles, labels = list(legend.values()), list(legend)
    if handles:
        figure.legend(
            handles, labels, loc="upper center", ncol=len(labels), fontsize=8, frameon=False
        )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    out.parent.mkdir(parents=True, exist_ok=True)
    head, dirty = provenance.commit()
    figure.savefig(
        out,
        dpi=130,
        facecolor=figure.get_facecolor(),
        metadata={"Commit": f"{head}{' (dirty)' if dirty else ''}", "Time": provenance.now()},
    )
    import matplotlib.pyplot as plt

    plt.close(figure)
    return out
