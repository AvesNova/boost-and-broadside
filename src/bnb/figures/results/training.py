"""Charts from a chain's own training records (`metrics.jsonl`, BC then RL), no evaluation
games needed:

- `behaviour`: the policy's play over training, one line per mode: kills per ship-minute,
  game length, the share of deaths to the boundary, the hit rate (hits per shot), the
  front's moves per minute (frontline) and the mean speed. The games are the training's own
  (cloning's student games, then self-play and the league), so they show how the play
  changed, not how strong it is.
- `throughput`: environment steps per second, collection and learning together, for each
  stage on each GPU, the median over the updates after the first (the first compiles).
  A GPU with no record is an empty, labelled slot.
- `lanes`: the ratings the training's own lanes measure against the scripted anchor
  (Imugi = 1000), per mode: the live policy and its exponential average, each with its
  standard error. Live training ratings, not the evaluation's.

Each source is a run directory (its `metrics.jsonl` and `run.json`); several are a chain,
in order. The steps are the chain's (`counters/chain_environment_steps`).
"""

import json
import math
import pathlib
import statistics

from bnb.runs.sinks import METRICS, read_metrics

from .. import theme
from . import common

CHAIN_STEPS = "counters/chain_environment_steps"
STEPS_PER_SECOND = "train/environment_steps_per_second"
UPDATE_SECONDS = "train/time/update_seconds"
GPUS = {"rtx_5090": "RTX 5090 (32 GB)", "laptop_4070": "RTX 4070 Laptop (8 GB)"}
STAGES = ("bc", "rl")


def read_chain(paths: list[str | pathlib.Path]) -> list[tuple[str, dict, list[dict]]]:
    """[(run name, run.json, metric rows)] in chain order."""
    found = []
    for path in map(pathlib.Path, paths):
        directory = path if path.is_dir() else path.parent
        run = json.loads((directory / "run.json").read_text())
        found.append((run["name"], run, read_metrics(directory / METRICS)))
    return found


def stage_of(name: str) -> str:
    """A run's stage from its name: "bc" or "rl"."""
    return "bc" if name.endswith("-bc") else "rl"


def _handoffs(axis, chain) -> None:
    for _, _, rows in chain[1:]:
        if rows:
            first = rows[0].get(CHAIN_STEPS, 0.0) - rows[0].get("counters/environment_steps", 0.0)
            axis.axvline(first / 1e6, color=theme.MUTED, linestyle="--", linewidth=0.9)


def _series(chain, value) -> list[tuple[float, float]]:
    """[(chain steps in millions, value(row))] over the chain, rows where value is None
    left out."""
    points = []
    for _, _, rows in chain:
        for row in rows:
            got = value(row)
            if got is not None and math.isfinite(got) and CHAIN_STEPS in row:
                points.append((row[CHAIN_STEPS] / 1e6, got))
    return points


def _preset_mean(row: dict, stem: str, preset_mode: str) -> float | None:
    """The episode-count-weighted mean of `episode/<stem>/<preset>/<pool>` over the pools
    of the presets whose mode is `preset_mode` (lanes left out)."""
    total = weight = 0.0
    prefix = f"episode/{stem}/"
    for key, value in row.items():
        parts = key.removeprefix(prefix).split("/")
        if not key.startswith(prefix) or len(parts) != 2 or parts[1] == "lane":
            continue
        if common.mode_of(parts[0]) != preset_mode:
            continue
        count = row.get(f"episode/count/{parts[0]}/{parts[1]}", 0)
        if count:
            total += value * count
            weight += count
    return total / weight if weight else None


def _weighted_event(row: dict, event: str, mode: str) -> float | None:
    """The per-game mean of `episode/event/<event>/<preset>/<pool>`, weighted by the pools'
    game counts over the presets whose mode is `mode` (lanes left out)."""
    total = weight = 0.0
    for key, value in row.items():
        parts = key.split("/")
        if len(parts) != 5 or parts[:3] != ["episode", "event", event]:
            continue
        if common.mode_of(parts[3]) != mode:
            continue
        count = row.get(f"episode/count/{parts[3]}/{parts[4]}", 0)
        if count:
            total += value * count
            weight += count
    return total / weight if weight else None


def _ratio(numerator, denominator):
    if numerator is None or not denominator:
        return None
    return numerator / denominator


BEHAVIOUR = (  # (title, y label, value(row, mode) or None for every mode)
    (
        "kills per ship-minute",
        "kills / ship / minute",
        lambda row, mode: next(
            (
                v
                for k, v in row.items()
                if k.startswith("combat/deaths_per_ship_minute/")
                and common.mode_of(k.split("/")[-1]) == mode
            ),
            None,
        ),
    ),
    ("game length", "seconds", lambda row, mode: _preset_mean(row, "length_seconds", mode)),
    (
        "deaths to the boundary",
        "share of deaths",
        lambda row, mode: _ratio(
            _weighted_event(row, "boundary_deaths", mode),
            (_weighted_event(row, "boundary_deaths", mode) or 0.0)
            + (_weighted_event(row, "combat_deaths", mode) or 0.0),
        ),
    ),
    (
        "hit rate",
        "hits per shot",
        lambda row, mode: _ratio(
            _weighted_event(row, "hits", mode), _weighted_event(row, "shots_fired", mode)
        ),
    ),
)


def draw_behaviour(paths, out, formats=common.FORMATS, name="behaviour_over_training"):
    """The behaviour chart (see the module); (the files written, the values: each mode's
    last BC and last RL value per panel)."""
    chain = read_chain(paths)
    singles = (
        (
            "front moves per minute (frontline)",
            "moves / game-minute",
            lambda row: row.get("frontline/front_moves_per_environment_minute"),
        ),
        ("mean speed", "m/s", lambda row: row.get("physics/speed_metres_per_second")),
    )
    panels = len(BEHAVIOUR) + len(singles)
    figure, axes = common.new_figure(2, 3, size=(3.6, 2.7))
    values: dict[str, float | None] = {}
    for axis, (heading, label, value) in zip(axes.flat, BEHAVIOUR, strict=False):
        lines = []
        for mode in common.MODES:
            points = _series(chain, lambda row, m=mode: value(row, m))
            if not points:
                continue
            xs, ys = zip(*points, strict=True)
            colour = common.mode_colour(mode)
            axis.plot(xs, ys, color=colour, linewidth=1.4)
            lines.append((mode, list(xs), list(ys), colour))
            for run, _, rows in chain:
                last = [value(row, mode) for row in rows]
                last = [v for v in last if v is not None]
                slug = common.slug(heading)
                values[f"behaviour_{mode}_{slug}_{stage_of(run)}_final"] = (
                    statistics.fmean(last[-5:]) if last else None
                )
        axis.set_title(heading, fontsize=9)
        axis.set_ylabel(label, fontsize=8)
        _handoffs(axis, chain)
        common.label_ends(axis, lines)
    for axis, (heading, label, value) in zip(axes.flat[len(BEHAVIOUR) :], singles, strict=False):
        points = _series(chain, value)
        if points:
            xs, ys = zip(*points, strict=True)
            colour = common.mode_colour("frontline") if "frontline" in heading else theme.BASELINE
            axis.plot(xs, ys, color=colour, linewidth=1.4)
        axis.set_title(heading, fontsize=9)
        axis.set_ylabel(label, fontsize=8)
        _handoffs(axis, chain)
    values.update(game_length_values(chain))
    for axis in axes.flat[panels:]:
        axis.set_visible(False)
    for axis in axes[-1]:
        axis.set_xlabel("environment steps (M)", fontsize=8)
    common.title(
        figure,
        "Behaviour over training (the training's own games; BC, then RL from the dashed line)",
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    written = common.save(figure, out, name, [str(p) for p in paths], values, formats)
    return written, values


def game_length_values(chain, updates: int = 5) -> dict[str, float | None]:
    """Per mode, over the chain's last stage's last `updates` updates (its own games, every
    pool but the lanes): the mean game length and the share of games that ran the clock
    out, `game_length_<mode>_training_mean_seconds` and `_training_clock_out_share`."""
    rows = chain[-1][2][-updates:] if chain else []
    values: dict[str, float | None] = {}
    for mode in common.MODES:
        lengths = [_preset_mean(row, "length_seconds", mode) for row in rows]
        clocks = [_preset_mean(row, "end/timeout", mode) for row in rows]
        lengths = [value for value in lengths if value is not None]
        clocks = [value for value in clocks if value is not None]
        values[f"game_length_{mode}_training_mean_seconds"] = (
            statistics.fmean(lengths) if lengths else None
        )
        values[f"game_length_{mode}_training_clock_out_share"] = (
            statistics.fmean(clocks) if clocks else None
        )
    return values


def stage_numbers(chain) -> dict[str, dict[str, float]]:
    """Per stage: steps per second (the median over the updates after the first), hours,
    compile minutes (the first update's time over the median update's) and steps."""
    numbers = {}
    for run_name, run, rows in chain:
        rates = [row[STEPS_PER_SECOND] for row in rows[1:] if STEPS_PER_SECOND in row]
        updates = [row[UPDATE_SECONDS] for row in rows if UPDATE_SECONDS in row]
        steady = statistics.median(updates[1:]) if len(updates) > 1 else None
        numbers[stage_of(run_name)] = dict(
            steps_per_second=statistics.median(rates) if rates else None,
            hours=run.get("hours"),
            compile_minutes=(updates[0] - steady) / 60 if steady is not None else None,
            steps=rows[-1].get("counters/environment_steps") if rows else None,
        )
    return numbers


def draw_throughput(
    paths,
    out,
    gpu="rtx_5090",
    others: dict | None = None,
    formats=common.FORMATS,
    name="throughput",
):
    """Steps per second per stage, one group per GPU: `gpu`'s from the chain in `paths`,
    the others' from `others` ({gpu: {stage: steps per second or None}}; a None is an
    empty slot marked pending). (the files written, the values)."""
    chain = read_chain(paths)
    numbers = stage_numbers(chain)
    table = {gpu: {stage: numbers.get(stage, {}).get("steps_per_second") for stage in STAGES}}
    for other, held in (others or {}).items():
        table[other] = {stage: held.get(stage) for stage in STAGES}
    order = [key for key in GPUS if key in table] + [k for k in table if k not in GPUS]
    figure, axes = common.new_figure(1, 1, size=(5.6, 3.0))
    axis = axes[0, 0]
    height = 0.36
    colours = {"bc": theme.slot(0), "rl": theme.slot(1)}
    top = max((v for row in table.values() for v in row.values() if v), default=1.0)
    for index, key in enumerate(order):
        for offset, stage in enumerate(STAGES):
            y = index + (offset - 0.5) * height * 1.1
            value = table[key][stage]
            if value:
                axis.barh(y, value, height=height, color=colours[stage])
                axis.text(
                    value + top * 0.01, y, f"{value:,.0f}", va="center", fontsize=7, color=theme.INK
                )
            else:
                axis.barh(
                    y,
                    top,
                    height=height,
                    color="none",
                    edgecolor=theme.MUTED,
                    linestyle=(0, (2, 2)),
                    linewidth=0.8,
                )
                axis.text(top * 0.02, y, "pending", va="center", fontsize=7, color=theme.MUTED)
    axis.set_yticks(range(len(order)), [GPUS.get(key, key) for key in order], fontsize=8)
    axis.invert_yaxis()
    axis.set_xlim(0, top * 1.18)
    axis.set_xlabel("environment steps per second (whole updates)", fontsize=8)
    axis.grid(axis="y", visible=False)
    from matplotlib.patches import Patch

    axis.legend(
        [Patch(color=colours[stage]) for stage in STAGES],
        ["behaviour cloning", "reinforcement learning"],
        fontsize=7,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.28),
        ncol=2,
    )
    common.title(figure, "Throughput")
    figure.tight_layout()
    values: dict[str, float | None] = {}
    for key in order:
        for stage in STAGES:
            values[f"throughput_{key}_{stage}_steps_per_second"] = table[key][stage]
    for stage in STAGES:
        held = numbers.get(stage, {})
        values[f"chain_hours_{gpu}_{stage}"] = held.get("hours")
        values[f"chain_steps_{stage}"] = held.get("steps")
    values[f"compile_minutes_{gpu}"] = (
        sum(held["compile_minutes"] for held in numbers.values() if held.get("compile_minutes"))
        or None
    )
    written = common.save(figure, out, name, [str(p) for p in paths], values, formats)
    return written, values


def draw_lanes(paths, out, formats=common.FORMATS, name="training_ratings"):
    """The training lanes' ratings against the scripted anchor per mode (see the module);
    (the files written, the values: each mode's last live and averaged rating)."""
    chain = read_chain(paths)
    figure, axes = common.new_figure(1, 3, size=(3.6, 2.8), sharey=True)
    values: dict[str, float | None] = {}
    for axis, mode in zip(axes.flat, common.MODES, strict=True):
        colour = common.mode_colour(mode)
        for kind, style, label in (("live", "-", "live"), ("average", "--", "exponential average")):

            def rating(row, kind=kind):
                for key, value in row.items():
                    if key.startswith(f"rating/{kind}/") and common.mode_of(key) == mode:
                        return value
                return None

            def error(row, kind=kind):
                for key, value in row.items():
                    if key.startswith(f"rating/{kind}_stderr/") and common.mode_of(key) == mode:
                        return value
                return None

            points = _series(chain, rating)
            errors = dict(_series(chain, error))
            if not points:
                continue
            xs, ys = zip(*points, strict=True)
            spread = [errors.get(x, 0.0) for x in xs]
            axis.plot(xs, ys, color=colour, linestyle=style, linewidth=1.3, label=label)
            if kind == "average":
                axis.fill_between(
                    xs,
                    [y - s for y, s in zip(ys, spread, strict=True)],
                    [y + s for y, s in zip(ys, spread, strict=True)],
                    color=colour,
                    alpha=0.15,
                    linewidth=0,
                )
            values[f"lanes_{mode}_{kind}_final"] = ys[-1]
            values[f"lanes_{mode}_{kind}_final_stderr"] = spread[-1]
        axis.axhline(1000, color=theme.MUTED, linewidth=0.8, linestyle=":")
        axis.set_title(mode, fontsize=9)
        axis.set_xlabel("environment steps (M)", fontsize=8)
        _handoffs(axis, chain)
    axes[0, 0].set_ylabel("Elo (Imugi = 1000)", fontsize=8)
    axes[0, 0].legend(fontsize=7, frameon=False, loc="lower right")
    common.title(figure, "Training lanes' ratings against the scripted anchor (live and averaged)")
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    written = common.save(figure, out, name, [str(p) for p in paths], values, formats)
    return written, values
