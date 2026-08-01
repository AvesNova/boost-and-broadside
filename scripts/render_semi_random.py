"""Render the reference-ladder diagnostic from its saved tournament.

Usage:
    uv run scripts/render_semi_random.py
"""

import argparse
import json
from pathlib import Path

from boost_and_broadside.modes.semi_random_tournament_plots import write_plots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(
            "checkpoints/resilient-resonance-682/semi_random_tournament.json"
        ),
    )
    parser.add_argument("--out", type=Path, default=Path("docs/results"))
    args = parser.parse_args()

    result = json.loads(args.data.read_text())
    for path in write_plots(result, args.out):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
