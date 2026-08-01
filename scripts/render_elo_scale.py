"""Render the fleet-scale rating view from saved checkpoint and reference tournaments.

Usage:
    uv run scripts/render_elo_scale.py
"""

import argparse
import json
from pathlib import Path

from boost_and_broadside.modes.elo_scale import combine_reference_ladder
from boost_and_broadside.modes.elo_scale_plots import write_scale_plots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("checkpoints/resilient-resonance-682/elo_scale.json"),
    )
    parser.add_argument(
        "--reference-data",
        type=Path,
        default=Path(
            "checkpoints/resilient-resonance-682/semi_random_tournament.json"
        ),
        help="Semi-random reference tournament joined through random and scripted endpoints.",
    )
    parser.add_argument("--out", type=Path, default=Path("docs/results"))
    args = parser.parse_args()

    result = json.loads(args.data.read_text())
    if args.reference_data.exists():
        reference_result = json.loads(args.reference_data.read_text())
        result = combine_reference_ladder(result, reference_result)
    for path in write_scale_plots(result, args.out):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
