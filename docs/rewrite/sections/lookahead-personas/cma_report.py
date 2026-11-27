"""Round 2, step 4: the CMA journal (tune.py's cma.jsonl) as a table, and its candidates as
configs (rows over the defaults) for the fresh verification.

    cma_report.py CMA.jsonl SPACE.json OUTDIR [--top 3]

Writes OUTDIR/cma-mean.json (the search distribution's mean after the last generation: the
robust candidate) and OUTDIR/cma-top{k}.json (the evaluated points with the highest mean Elo
over the last third of the generations; biased upward by selection, so verified afresh), and
prints the generations table (markdown).
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tune  # noqa: E402

from lean.scripted import ScriptedConfig  # noqa: E402


def rows_over_defaults(config: ScriptedConfig) -> dict:
    default = ScriptedConfig().values
    return {k: v for k, v in config.values.items() if abs(v - default[k]) > 1e-12}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("journal")
    parser.add_argument("space")
    parser.add_argument("out", type=Path)
    parser.add_argument("--top", type=int, default=3)
    arguments = parser.parse_args()
    lines = [json.loads(line) for line in Path(arguments.journal).read_text().splitlines()]
    space = tune.space_of(json.loads(Path(arguments.space).read_text()))
    print(
        "| generation | time | mean of the population (1v1 / 4v4 / 16v16 / mean) | best | sigma |"
    )
    print("|---|---|---|---|---|")
    for line in lines:
        elo = np.array([[e[str(s)] for s in tune.SIZES] for e in line["elo"]])
        means = elo.mean(0)
        print(
            f"| {line['generation']} | {line['time']} | {means[0]:.0f} / {means[1]:.0f} / "
            f"{means[2]:.0f} / {means.mean():.0f} | {max(line['mean']):.0f} | "
            f"{line['sigma']:.3f} |"
        )
    arguments.out.mkdir(parents=True, exist_ok=True)
    mean = lines[-1]["cma_mean"]
    (arguments.out / "cma-mean.json").write_text(
        json.dumps(rows_over_defaults(space.config(mean)), indent=1)
    )
    tail = lines[-max(1, len(lines) // 3) :]
    points = [(m, z) for line in tail for m, z in zip(line["mean"], line["solutions"], strict=True)]
    points.sort(key=lambda item: -item[0])
    for k, (m, z) in enumerate(points[: arguments.top], start=1):
        (arguments.out / f"cma-top{k}.json").write_text(
            json.dumps(rows_over_defaults(space.config(z)), indent=1)
        )
        print(f"top{k}: mean Elo {m:.0f} in the search", file=sys.stderr)
    print(f"generations {len(lines)}, last sigma {lines[-1]['sigma']:.3f}", file=sys.stderr)


if __name__ == "__main__":
    main()
