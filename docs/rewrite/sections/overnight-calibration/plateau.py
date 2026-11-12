"""plateau.py: BC's plateau stop (`bnb.learn.bc.Plateau`) replayed offline on a run's logged
BC KL, and its parameters fitted to where the run's Elo against the scripted agent
plateaus.

    replay METRICS [--fraction F --window W --from S --smoothing M --hold H]
        the step at which the rule fires on METRICS (a run's metrics.jsonl), the code's
        defaults for any parameter not given
    fit ELO METRICS...
        ELO: a JSON file {run: [[step, elo, stderr], ...]} (the EMA snapshots' ratings);
        each METRICS the run's metrics.jsonl, its run named by its directory. Per run: the
        Elo plateau (the first snapshot after which every later one is within the combined
        error of the last), then every rule of the grid replayed, and per run and for all
        runs together the rules that fire at or after the plateau, latest-first ranked by
        how late they fire (the median lateness, then the worst).

The rule is the training code's own class, fed the same (steps, KL) sequence the run fed it.
"""

import argparse
import dataclasses
import itertools
import json
import math
import pathlib
import statistics

from bnb.learn.bc import Plateau, StopConfig

MILLION = 1_000_000
GRID = dict(
    plateau_fraction=(0.02, 0.03, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3),
    plateau_window_steps=tuple(int(m * MILLION) for m in (2.5, 5, 7.5, 10)),
    plateau_from_steps=tuple(int(m * MILLION) for m in (2.5, 5)),
    plateau_smoothing_steps=tuple(int(m * MILLION) for m in (2.5, 5)),
    plateau_hold_steps=tuple(int(m * MILLION) for m in (0, 2.5, 5, 10)),
)
DEFAULTS = StopConfig()


def series(metrics: pathlib.Path) -> list[tuple[int, float]]:
    """(environment steps, BC KL) per update, in order (a resumed run's repeated updates:
    the later record of a step wins)."""
    rows = {}
    for line in metrics.read_text().splitlines():
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "bc/kl" in row:
            rows[int(row["counters/environment_steps"])] = float(row["bc/kl"])
    return sorted(rows.items())


def fires(kl: list[tuple[int, float]], config: StopConfig) -> int | None:
    """The step at which the rule fires, None if it never does."""
    rule = Plateau(config)
    for steps, value in kl:
        if rule.observe(steps, value):
            return steps
    return None


def elo_plateau(points: list[list[float]]) -> int:
    """The first snapshot step after which every later snapshot (itself included) is
    within the combined standard error of the last snapshot's rating."""
    points = sorted(points)
    last_elo, last_error = points[-1][1], points[-1][2]
    plateau = int(points[-1][0])
    for step, elo, error in reversed(points):
        if abs(elo - last_elo) <= math.hypot(error, last_error):
            plateau = int(step)
        else:
            break
    return plateau


def grid():
    names = list(GRID)
    for values in itertools.product(*(GRID[name] for name in names)):
        yield dataclasses.replace(DEFAULTS, **dict(zip(names, values)))


def describe(config: StopConfig) -> str:
    return (
        f"fraction {config.plateau_fraction:g}, window {config.plateau_window_steps / MILLION:g}M, "
        f"from {config.plateau_from_steps / MILLION:g}M, smoothing "
        f"{config.plateau_smoothing_steps / MILLION:g}M, hold {config.plateau_hold_steps / MILLION:g}M"
    )


def fit(elo_file: pathlib.Path, metrics: list[pathlib.Path], top: int) -> dict:
    ratings = json.loads(elo_file.read_text())
    runs = {path.parent.name: series(path) for path in metrics}
    plateaus = {run: elo_plateau(ratings[run]) for run in runs}
    end = {run: kl[-1][0] for run, kl in runs.items()}
    results = []
    for config in grid():
        steps = {run: fires(kl, config) for run, kl in runs.items()}
        late = {}
        for run, step in steps.items():
            # a rule that never fires stops at the budget: as late as the run's end
            late[run] = (step if step is not None else end[run]) - plateaus[run]
        results.append((config, steps, late))
    out = dict(plateaus=plateaus, defaults=None, per_run={}, together=[])
    default_steps = {run: fires(kl, DEFAULTS) for run, kl in runs.items()}
    out["defaults"] = dict(rule=describe(DEFAULTS), fires=default_steps)

    def ranked(items, members):
        good = [
            (config, steps, late)
            for config, steps, late in items
            if all(late[run] >= 0 and steps[run] is not None for run in members)
        ]
        good.sort(
            key=lambda item: (
                statistics.median(item[2][run] for run in members),
                max(item[2][run] for run in members),
                abs(math.log(item[0].plateau_fraction / DEFAULTS.plateau_fraction)),
            )
        )
        return [
            dict(rule=describe(config), fires={run: steps[run] for run in members})
            for config, steps, _ in good[:top]
        ]

    for run in runs:
        out["per_run"][run] = ranked(results, [run])
    out["together"] = ranked(results, list(runs))
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    replay = sub.add_parser("replay")
    replay.add_argument("metrics", type=pathlib.Path)
    for name, field in (
        ("fraction", "plateau_fraction"),
        ("window", "plateau_window_steps"),
        ("from", "plateau_from_steps"),
        ("smoothing", "plateau_smoothing_steps"),
        ("hold", "plateau_hold_steps"),
    ):
        replay.add_argument(f"--{name}", dest=field, type=float, default=None)
    fitting = sub.add_parser("fit")
    fitting.add_argument("elo", type=pathlib.Path)
    fitting.add_argument("metrics", type=pathlib.Path, nargs="+")
    fitting.add_argument("--top", type=int, default=5)
    options = parser.parse_args()
    if options.command == "replay":
        given = {
            name: (value if name == "plateau_fraction" else int(value))
            for name, value in vars(options).items()
            if name.startswith("plateau_") and value is not None
        }
        config = dataclasses.replace(DEFAULTS, **given)
        step = fires(series(options.metrics), config)
        print(f"{describe(config)}: fires at {'never' if step is None else f'{step / MILLION:.1f}M'}")
    else:
        print(json.dumps(fit(options.elo, options.metrics, options.top), indent=1))


if __name__ == "__main__":
    main()
