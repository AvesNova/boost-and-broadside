"""The headline fan: the policy's win rate against Imugi over training, one line per
lineup (the policy's `policy_count` ships against N of Imugi's), one panel per mode.

Each snapshot's `bnb crossover` result holds one logistic fit per mode of the policy's score
on the log ratio r = log(policy's ships / Imugi's ships) (`bnb.league.crossover.fit`). A
line's point at a snapshot is that fit at the line's r, its band the 95% interval of the
linear predictor (from the fit's covariance) through the logistic. The games actually played
at a line's ratio (a lineup within `MATCH` in r) are drawn over it as open points. A
snapshot whose fit has no crossing (separated outcomes) has no line point; the panel says how
many there are.

Input: the directory of per-snapshot results, `<run>/step_<n>/artifacts/crossover/<time>/
result.json` (`bnb.figures.ladder.crossover_results`, the dry run's layout); a chain's BC
and RL runs share one axis of its steps (`bnb.figures.ladder.chain_offsets`).
"""

import math
import pathlib

from .. import theme
from ..ladder import chain_offsets, crossover_results
from . import common

MATCH = 0.03  # a played lineup counts as a line's ratio within this in log ratio
Z95 = 1.959964


def win_rate(fit: dict, log_ratio: float) -> tuple[float, float, float] | None:
    """(win rate, 95% low, 95% high) of a crossing fit (intercept, slope, covariance in r)
    at `log_ratio`; None for a fit without a crossing."""
    if fit.get("status") != "crossing" or fit.get("covariance") is None:
        return None
    eta = fit["intercept"] + fit["slope"] * log_ratio
    (aa, as_), (_, ss) = fit["covariance"]
    spread = math.sqrt(max(aa + 2 * log_ratio * as_ + log_ratio * log_ratio * ss, 0.0))

    def logistic(value: float) -> float:
        return 1.0 / (1.0 + math.exp(-value))

    return logistic(eta), logistic(eta - Z95 * spread), logistic(eta + Z95 * spread)


def series(
    results: dict[tuple[str, int], dict],
    policy_count: int,
    opponent_counts: tuple[int, ...],
    offsets: dict[str, int],
) -> dict[str, dict]:
    """Per mode: "lines", per opponent count the snapshots' (steps, rate, low, high) in
    order of steps; "played", per opponent count the (steps, score, games) of the lineups
    played at its ratio; "separated", the snapshots without a crossing. A run's steps are
    offset by `offsets` (`chain_offsets`)."""
    found: dict[str, dict] = {}
    for (run, step), payload in sorted(results.items(), key=lambda i: i[0][1] + offsets[i[0][0]]):
        steps = step + offsets[run]
        for preset, measured in payload.get("presets", {}).items():
            mode = common.mode_of(preset)
            held = found.setdefault(
                mode,
                dict(
                    lines={count: [] for count in opponent_counts},
                    played={count: [] for count in opponent_counts},
                    separated=0,
                ),
            )
            fit = measured.get("fit", {})
            if fit.get("status") != "crossing":
                held["separated"] += 1
            for count in opponent_counts:
                target = math.log(policy_count / count)
                rate = win_rate(fit, target)
                if rate is not None:
                    held["lines"][count].append((steps, *rate))
                for row in measured.get("lineups", []):
                    played = math.log(row["policy_count"] / row["opponent_count"])
                    if abs(played - target) < MATCH and row["games"] > 0:
                        held["played"][count].append((steps, row["score"], row["games"]))
    return {mode: found[mode] for mode in [*common.MODES, *sorted(found)] if mode in found}


def sample_values(results: dict[tuple[str, int], dict]) -> dict[str, float | str]:
    """What the captions quote about the sample: crossover_fixed_count, the ships of the
    side held fixed in every lineup (a string of the counts when they differ);
    crossover_snapshots; crossover_games_min and _max, a snapshot's games in one mode."""
    fixed = sorted(
        {
            measured.get("fixed_count")
            for payload in results.values()
            for measured in payload.get("presets", {}).values()
        }
        - {None}
    )
    games = [
        measured.get("games", 0)
        for payload in results.values()
        for measured in payload.get("presets", {}).values()
    ]
    return dict(
        crossover_fixed_count=float(fixed[0]) if len(fixed) == 1 else ", ".join(map(str, fixed)),
        crossover_snapshots=float(len(results)),
        crossover_games_min=float(min(games, default=0)),
        crossover_games_max=float(max(games, default=0)),
    )


def draw(
    directory: str | pathlib.Path,
    out: str | pathlib.Path,
    policy_count: int = 5,
    opponent_counts: tuple[int, ...] = (5, 6, 8, 10),
    modes: tuple[str, ...] = (),
    formats: tuple[str, ...] = common.FORMATS,
    name: str = "win_rate_fan",
) -> tuple[list[pathlib.Path], dict]:
    """Draw the fan from the results below `directory` into `out`; (the files written, the
    values: per mode and lineup the last snapshot's win rate and its 95% interval)."""
    results = crossover_results(directory)
    offsets = chain_offsets([dict(run=run, step=step) for run, step in results])
    found = series(results, policy_count, opponent_counts, offsets)
    if modes:
        found = {mode: held for mode, held in found.items() if mode in modes}
    if not found:
        raise SystemExit(f"no crossover results below {directory}")
    figure, axes = common.new_figure(1, len(found), size=(4.2, 3.4), sharey=True)
    colours = common.ordered_colours(len(opponent_counts))
    values: dict[str, float | str] = sample_values(results)
    for axis, (mode, held) in zip(axes.flat, found.items(), strict=True):
        axis.set_title(mode, fontsize=9)
        axis.axhline(0.5, color=theme.MUTED, linewidth=0.8, linestyle=":")
        for handoff in sorted({offset for offset in offsets.values() if offset > 0}):
            axis.axvline(handoff / 1e6, color=theme.MUTED, linewidth=0.8, linestyle="--")
        labelled = []
        for count, colour in zip(opponent_counts, colours, strict=True):
            line = held["lines"][count]
            label = f"{policy_count}v{count}"  # the value names' lineup
            shown = f"x{common.slug(count / policy_count).replace('_', '.')}"  # Imugi's ratio
            if line:
                steps = [point[0] / 1e6 for point in line]
                axis.fill_between(
                    steps,
                    [point[2] for point in line],
                    [point[3] for point in line],
                    color=colour,
                    alpha=0.18,
                    linewidth=0,
                )
                axis.plot(
                    steps, [point[1] for point in line], color=colour, linewidth=1.6, label=shown
                )
                labelled.append((shown, steps, [point[1] for point in line], colour))
                last = line[-1]
                key = f"fan_{common.slug(mode)}_{label}"
                values[f"{key}_win_rate"] = last[1]
                values[f"{key}_low"] = last[2]
                values[f"{key}_high"] = last[3]
            played = held["played"][count]
            if played:
                axis.scatter(
                    [point[0] / 1e6 for point in played],
                    [point[1] for point in played],
                    s=14,
                    facecolors="none",
                    edgecolors=colour,
                    linewidths=0.9,
                    zorder=3,
                )
        axis.set_ylim(-0.02, 1.02)
        axis.set_xlabel("environment steps (M)", fontsize=8)
        common.label_ends(axis, labelled)
        if held["separated"]:
            common.note(axis, f"{held['separated']} snapshot(s) without a crossing")
    axes.flat[0].set_ylabel("win rate against Imugi", fontsize=8)
    common.title(figure, name)
    figure.tight_layout(rect=(0, 0, 0.98, 1))
    written = common.save(figure, out, name, [str(directory)], values, formats)
    return written, values
