"""What `bnb figures results` does: draw one chart from its sources, write it (PNG with its
provenance, and the vector formats), and merge its numbers into the values file."""

from .. import theme
from . import arms, break_even, common, fan, game_length, health, team_size

CHARTS = ("fan", "break-even", "arms", "team-size", "health", "game-length")


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
    if options.chart in ("fan", "break-even", "arms") and len(sources) != 1:
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
    else:
        raise SystemExit(f"unknown chart {options.chart!r}; choose from {CHARTS}")
    return written, values
