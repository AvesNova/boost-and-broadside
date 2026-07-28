"""Render the three fleet-scale rating views from a saved tournament.

Usage:
    uv run scripts/render_elo_scale.py
"""

import argparse
import json
from pathlib import Path

from boost_and_broadside.modes.elo_scale_plots import write_scale_plots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path("checkpoints/resilient-resonance-682/elo_scale.json"),
    )
    parser.add_argument("--out", type=Path, default=Path("docs/results"))
    args = parser.parse_args()

    result = json.loads(args.data.read_text())
    for path in write_scale_plots(result, args.out):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
