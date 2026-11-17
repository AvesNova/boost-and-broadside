"""Team-size generalization: the policy's win rate against Imugi at each team size, one
line per ratio of Imugi's ships to the policy's (1:1, 1:1.5, 1:2), one panel per mode,
with the team size it trained at marked.

**Input** (`read_tallies`): JSONL files of game tallies, the shape `bnb crossover` records
(`<preset>.batches.jsonl`: each line's `blocks`, `bnb.league.crossover.block_tallies`), or
lines that are tallies themselves: {mode or preset?, policy_count, opponent_count, wins,
draws, losses}. A line without a mode takes its file's (`<preset>.batches.jsonl`); a
directory stands for every `*.batches.jsonl` below it. The x axis is the policy's ship
count; a lineup (p, o) counts for the ratio k when |ln(o / p) - ln k| < `MATCH`, so 1:1.5
has points at the even sizes only. The score counts a draw half a win; its 95% interval is
Wilson's. No game is larger than 20 a side, so 1:2 ends at 10.
"""

import json
import math
import pathlib

from .. import theme
from . import common

MATCH = 0.035
Z95 = 1.959964


def read_tallies(paths: list[str | pathlib.Path]) -> dict[tuple[str, int, int], list[float]]:
    """{(mode, policy_count, opponent_count): [score summed, games]} over the files."""
    files: list[pathlib.Path] = []
    for path in map(pathlib.Path, paths):
        files += sorted(path.rglob("*.batches.jsonl")) if path.is_dir() else [path]
    held: dict[tuple[str, int, int], list[float]] = {}
    for path in files:
        preset = path.name.removesuffix(".batches.jsonl").removesuffix(".jsonl")
        for text in path.read_text().splitlines():
            if not text.strip():
                continue
            line = json.loads(text)
            for tally in line.get("blocks", [line]):
                mode = common.mode_of(
                    tally.get("mode") or tally.get("preset") or line.get("mode") or preset
                )
                key = (mode, int(tally["policy_count"]), int(tally["opponent_count"]))
                sums = held.setdefault(key, [0.0, 0.0])
                sums[0] += tally["wins"] + 0.5 * tally["draws"]
                sums[1] += tally["wins"] + tally["draws"] + tally["losses"]
    return held


def wilson(score: float, games: float) -> tuple[float, float]:
    """The 95% Wilson interval of a mean score over `games`."""
    rate = score / games
    centre = (rate + Z95**2 / (2 * games)) / (1 + Z95**2 / games)
    half = (Z95 / (1 + Z95**2 / games)) * math.sqrt(
        rate * (1 - rate) / games + Z95**2 / (4 * games**2)
    )
    return centre - half, centre + half


def draw(
    paths: list[str | pathlib.Path],
    out: str | pathlib.Path,
    ratios: tuple[float, ...] = (1.0, 1.5, 2.0),
    trained_team_size: int = 5,
    modes: tuple[str, ...] = (),
    formats: tuple[str, ...] = common.FORMATS,
    name: str = "team_size_generalization",
) -> tuple[list[pathlib.Path], dict]:
    """Draw the chart from the tallies in `paths` into `out`; (the files written, the
    values: each point's win rate)."""
    tallies = read_tallies(paths)
    found = [m for m in common.MODES if any(key[0] == m for key in tallies)]
    found += sorted({key[0] for key in tallies} - set(found))
    if modes:
        found = [m for m in found if m in modes]
    if not found:
        raise SystemExit(f"no tallies in {list(map(str, paths))}")
    figure, axes = common.new_figure(1, len(found), size=(4.2, 3.4), sharey=True)
    colours = common.ordered_colours(len(ratios))
    values: dict[str, float] = {}
    games: list[float] = []
    for axis, mode in zip(axes.flat, found, strict=True):
        axis.set_title(mode, fontsize=9)
        axis.axhline(0.5, color=theme.MUTED, linewidth=0.8, linestyle=":")
        axis.axvline(trained_team_size, color=theme.MUTED, linewidth=0.8, linestyle="--")
        axis.annotate(  # at the foot, clear of the lines near 1
            f"trained at {trained_team_size}v{trained_team_size}",
            (trained_team_size, 0.0),
            xycoords=("data", "axes fraction"),
            xytext=(3, 3),
            textcoords="offset points",
            fontsize=7,
            color=theme.MUTED,
            va="bottom",
        )
        labelled = []
        for ratio, colour in zip(ratios, colours, strict=True):
            points = sorted(
                (policy, sums)
                for (m, policy, opponent), sums in tallies.items()
                if m == mode
                and sums[1] > 0
                and abs(math.log(opponent / policy) - math.log(ratio)) < MATCH
            )
            if not points:
                continue
            games += [sums[1] for _, sums in points]
            sizes = [policy for policy, _ in points]
            rates = [score / games for _, (score, games) in points]
            intervals = [wilson(score, games) for _, (score, games) in points]
            label = f"1:{ratio:g}"
            axis.fill_between(
                sizes,
                [i[0] for i in intervals],
                [i[1] for i in intervals],
                color=colour,
                alpha=0.18,
                linewidth=0,
            )
            axis.plot(
                sizes, rates, color=colour, linewidth=1.6, marker="o", markersize=3, label=label
            )
            labelled.append((label, sizes, rates, colour))
            for size, rate in zip(sizes, rates, strict=True):
                values[
                    f"team_size_{common.slug(mode)}_ratio_{common.slug(ratio)}_size_{size}_win_rate"
                ] = rate
        axis.set_ylim(-0.02, 1.02)
        largest = max((key[1] for key in tallies if key[0] == mode), default=20)
        axis.set_xlim(0.5, largest + 0.5)  # the largest team the tallies hold
        axis.set_xticks([size for size in (1, 5, 10, 15, 20) if size <= largest])
        axis.set_xlabel("our ships", fontsize=8)
        common.label_ends(axis, labelled)
    values["team_size_games_min"] = float(min(games, default=0))
    values["team_size_games_max"] = float(max(games, default=0))
    axes.flat[0].set_ylabel("win rate against Imugi", fontsize=8)
    common.title(figure, name)
    figure.tight_layout(rect=(0, 0, 0.98, 1))
    written = common.save(figure, out, name, list(map(str, paths)), values, formats)
    return written, values
