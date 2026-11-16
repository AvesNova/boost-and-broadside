"""The break-even ratio over training: Imugi's ships per one of the policy's at even odds
(`bnb crossover`), against the chain's steps, one line per mode with its standard error as
bars, on a log axis (a ratio: 2 and 1/2 equally far from parity).

The same points as the ladder's break-even row (`bnb.figures.ladder.break_even_points`),
drawn as one panel of the modes together for the write-up. One chain only (a run, or its
`-bc` and `-rl` runs); `chain` picks it when the directory holds several.
"""

import pathlib

from ..ladder import BC_SUFFIX, RL_SUFFIX, break_even_points, chain_offsets
from ..output import GRID, MUTED
from . import common


def chain_name(run: str) -> str:
    """The chain a run belongs to: its name without the `-bc` or `-rl` suffix."""
    return run.removesuffix(BC_SUFFIX).removesuffix(RL_SUFFIX)


def draw(
    directory: str | pathlib.Path,
    out: str | pathlib.Path,
    chain: str | None = None,
    formats: tuple[str, ...] = common.FORMATS,
    name: str = "break_even_ratio",
) -> tuple[list[pathlib.Path], dict]:
    """Draw the break-even ratio against steps from the results below `directory` into
    `out`; (the files written, the values: per mode the last snapshot's ratio, its standard
    error and steps)."""
    points = break_even_points(directory)
    chains = sorted({chain_name(point["run"]) for point in points})
    if not points:
        raise SystemExit(f"no crossover results with a crossing below {directory}")
    if chain is None and len(chains) > 1:
        raise SystemExit(f"several chains below {directory}: {chains}; pick one with --chain")
    chain = chain_name(chain) if chain else chains[0]
    points = [point for point in points if chain_name(point["run"]) == chain]
    if not points:
        raise SystemExit(f"no chain {chain!r} below {directory}: {chains}")
    offsets = chain_offsets(points)
    figure, axes = common.new_figure(1, 1, size=(6.4, 3.8))
    axis = axes[0, 0]
    axis.set_yscale("log")
    axis.axhline(1.0, color=MUTED, linewidth=0.8, linestyle=":")
    for handoff in sorted({offset for offset in offsets.values() if offset > 0}):
        axis.axvline(handoff / 1e6, color=MUTED, linewidth=0.8, linestyle="--")
        axis.annotate(
            "BC to RL",
            (handoff / 1e6, 1.0),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
            color=MUTED,
        )
    values: dict[str, float] = {}
    labelled = []
    modes = [mode for mode in common.MODES if any(p["mode"] == mode for p in points)]
    modes += sorted({p["mode"] for p in points} - set(modes))
    for mode in modes:
        mine = sorted(
            (p for p in points if p["mode"] == mode),
            key=lambda p: p["step"] + offsets[p["run"]],
        )
        steps = [(p["step"] + offsets[p["run"]]) / 1e6 for p in mine]
        ratios = [p["ratio"] for p in mine]
        colour = common.mode_colour(mode)
        axis.errorbar(
            steps,
            ratios,
            yerr=[p["stderr"] for p in mine],
            color=colour,
            linewidth=1.6,
            marker="o",
            markersize=3.5,
            elinewidth=0.9,
            capsize=0,
            label=mode,
        )
        labelled.append((mode, steps, ratios, colour))
        key = f"crossover_ratio_{common.slug(mode)}"
        values[key] = mine[-1]["ratio"]
        values[f"{key}_stderr"] = mine[-1]["stderr"]
        values[f"{key}_steps"] = mine[-1]["step"] + offsets[mine[-1]["run"]]
    from matplotlib.ticker import FuncFormatter

    axis.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.yaxis.set_minor_formatter(FuncFormatter(lambda value, _: f"{value:g}"))
    axis.tick_params(axis="y", which="minor", labelsize=7, colors=MUTED)
    axis.grid(True, which="minor", axis="y", color=GRID, linewidth=0.6)
    common.label_ends(axis, labelled)
    axis.set_xlabel("environment steps (M)", fontsize=8)
    axis.set_ylabel("Imugi's ships per one of ours at even odds", fontsize=8)
    common.title(figure, f"Break-even ratio over training ({chain})")
    figure.tight_layout(rect=(0, 0, 1, 1))
    written = common.save(figure, out, name, [str(directory)], values, formats)
    return written, values
