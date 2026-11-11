"""BC's plateau rule (`bnb.learn.bc.Plateau`) replayed on the hub's BC runs: the step at
which it fires on the BC KL and, for comparison, on the joint agreement, per run.

    uv run --no-sync python -m bench.bc_plateau --records DIRECTORY
    uv run --no-sync python -m bench.bc_plateau --records DIRECTORY --hold-steps 5000000
    uv run --no-sync python -m bench.bc_plateau --records DIRECTORY --fixture OUT.json

`--records` holds `<run>/metrics.jsonl` per run; a missing one is downloaded from the
public Hugging Face repository (`julianzia/boost-and-broadside-runs`, about 3-8 MB each).
A rewound run logs the rewound updates twice: the later copy, the lineage that went on,
is the one read. The defaults are `StopConfig`'s; `--fraction`, `--window-steps`,
`--from-steps`, `--smoothing-steps` and `--hold-steps` replace them. The agreement
(`bc/agreement/joint`) rises, so the rule reads its negative. `--fixture` writes the
runs' (environment steps, BC KL) pairs as the JSON the plateau test replays
(`tests/sections/golden/bc_plateau_hub_kl.json`). Reads files only: no GPU, no lock.
"""

import argparse
import dataclasses
import json
import pathlib
import urllib.request

from bnb.learn.bc import Plateau, StopConfig

HUB = "https://huggingface.co/julianzia/boost-and-broadside-runs/resolve/main/runs"
# the arms the rule was chosen on (2026-11-11): the null, the champion, the champion's
# parent that stopped at 60M still improving, and the nearest-neighbour MLP
RUNS = (
    "2026-11-09-bc-absolute",
    "2026-11-09-bc-deadreckon-split-hiddenphysics",
    "2026-11-09-bc-deadreckon-split",
    "2026-11-10-bc-nearest",
)
KL, AGREEMENT = "bc/kl", "bc/agreement/joint"


def rows(directory: pathlib.Path, run: str) -> list[dict]:
    """The run's records, one per update in update order (a rewound update's later copy),
    downloading its metrics.jsonl first when `directory` lacks it."""
    path = directory / run / "metrics.jsonl"
    if not path.is_file():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(f"{HUB}/{run}/metrics.jsonl", path)
    by_update = {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        by_update[row["counters/updates"]] = row
    return [by_update[update] for update in sorted(by_update)]


def fires(config: StopConfig, series: list[tuple[int, float]]) -> int | None:
    """The environment steps of the first update at which the plateau fires on `series`
    ((environment steps, value) per update, a value that falls as it improves); None when
    it never does."""
    plateau = Plateau(config)
    for steps, value in series:
        if plateau.observe(steps, value):
            return steps
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--records", type=pathlib.Path, required=True)
    parser.add_argument("--runs", nargs="+", default=RUNS)
    parser.add_argument("--fraction", type=float)
    parser.add_argument("--window-steps", type=int)
    parser.add_argument("--from-steps", type=int)
    parser.add_argument("--smoothing-steps", type=int)
    parser.add_argument("--hold-steps", type=int)
    parser.add_argument("--fixture", type=pathlib.Path)
    arguments = parser.parse_args()
    changes = {
        f"plateau_{name}": getattr(arguments, name)
        for name in ("fraction", "window_steps", "from_steps", "smoothing_steps", "hold_steps")
        if getattr(arguments, name) is not None
    }
    config = dataclasses.replace(StopConfig(), **changes)
    print(
        f"fraction {config.plateau_fraction}, window {config.plateau_window_steps:,}, "
        f"from {config.plateau_from_steps:,}, smoothing {config.plateau_smoothing_steps:,}, "
        f"hold {config.plateau_hold_steps:,} environment steps"
    )
    print(f"{'run':48s} {'end':>7s} {'kl':>7s} {'agreement':>9s}  (millions of steps)")
    fixture = {}
    for run in arguments.runs:
        records = rows(arguments.records, run)
        steps = [row["counters/environment_steps"] for row in records]
        kl = list(zip(steps, (row[KL] for row in records), strict=True))
        agreement = [(step, -row[AGREEMENT]) for step, row in zip(steps, records, strict=True)]
        shown = [fires(config, series) for series in (kl, agreement)]
        cells = ["-" if step is None else f"{step / 1e6:.1f}" for step in shown]
        print(f"{run:48s} {steps[-1] / 1e6:7.1f} {cells[0]:>7s} {cells[1]:>9s}")
        fixture[run] = [list(entry) for entry in kl]
    if arguments.fixture is not None:
        arguments.fixture.write_text(json.dumps(fixture, separators=(",", ":")) + "\n")
        print(f"wrote {arguments.fixture}")


if __name__ == "__main__":
    main()
