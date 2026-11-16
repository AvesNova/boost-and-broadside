"""What `bnb figures results` does: draw one chart from its sources, write it (PNG with its
provenance, and the vector formats), and merge its numbers into the values file."""

import json
import pathlib

from .. import theme
from . import (
    arms,
    break_even,
    common,
    evaluation,
    fan,
    game_length,
    health,
    team_size,
    training,
)

CHARTS = (
    "fan",
    "break-even",
    "arms",
    "team-size",
    "health",
    "game-length",
    "behaviour",
    "throughput",
    "lanes",
    "elo",
    "suite",
    "averaging",
    "maps",
)


def results(options) -> None:
    """`bnb figures results`: see `bnb.cli.FiguresResults`. Draws the chart in each theme
    variant (`bnb.figures.theme.VARIANTS`); the values are the same in both."""
    written, values = [], {}
    for variant in theme.VARIANTS:
        with theme.use(variant):
            drawn, values = _draw(options)
        written += drawn
    for path in written:
        print(f"wrote {path}")
    if options.values:
        common.write_values(values, options.values)
        print(f"{len(values)} values into {options.values}")


def _draw(options) -> tuple[list, dict]:
    """One chart in the active theme variant: (the files written, the values)."""
    sources = list(options.sources)
    if not sources:
        raise SystemExit(f"bnb figures results {options.chart} needs its sources")
    single = sources[0]
    if (
        options.chart in ("fan", "break-even", "arms", "elo", "suite", "averaging", "maps")
        and len(sources) != 1
    ):
        raise SystemExit(f"bnb figures results {options.chart} reads one source, not {sources}")
    formats = tuple(options.formats)
    if options.chart == "fan":
        written, values = fan.draw(
            single,
            options.out,
            options.policy_count,
            tuple(options.opponent_counts),
            tuple(options.modes),
            formats,
        )
    elif options.chart == "break-even":
        written, values = break_even.draw(single, options.out, options.chain, formats)
    elif options.chart == "arms":
        written, values = arms.draw(single, options.out, formats)
    elif options.chart == "team-size":
        written, values = team_size.draw(
            sources,
            options.out,
            tuple(options.ratios),
            options.trained_team_size,
            tuple(options.modes),
            formats,
        )
    elif options.chart == "health":
        written, values = health.draw(sources, options.out, options.samples, formats=formats)
    elif options.chart == "game-length":
        written, values = game_length.draw(sources, options.out, formats)
    elif options.chart == "behaviour":
        written, values = training.draw_behaviour(sources, options.out, formats)
    elif options.chart == "throughput":
        written, values = training.draw_throughput(
            sources, options.out, options.gpu, _other_gpus(options), formats
        )
    elif options.chart == "lanes":
        written, values = training.draw_lanes(sources, options.out, formats)
    elif options.chart == "elo":
        written, values = evaluation.draw_elo(single, options.out, formats)
    elif options.chart == "suite":
        written, values = evaluation.draw_suite(single, options.out, formats)
    elif options.chart == "maps":  # values only: a table on the page, no figure
        written, values = [], evaluation.map_values(single)
    elif options.chart == "averaging":
        written, values = evaluation.draw_averaging(single, options.out, formats)
        values.update(evaluation.champion_values(single))
    else:
        raise SystemExit(f"unknown chart {options.chart!r}; choose from {CHARTS}")
    return written, values


def _other_gpus(options) -> dict:
    """The other GPUs' steps per second from the values file (a null is a pending slot)."""
    path = pathlib.Path(options.values) if options.values else None
    held = json.loads(path.read_text()) if path and path.exists() else {}
    others = {}
    for gpu in training.GPUS:
        if gpu != options.gpu:
            others[gpu] = {
                stage: held.get(f"throughput_{gpu}_{stage}_steps_per_second")
                for stage in training.STAGES
            }
    return others
