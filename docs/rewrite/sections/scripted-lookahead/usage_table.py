"""The 42 actions' usage as a Markdown table from measure.py's JSON files (selfplay or old):
one row per (stick, power), the share of living ship-ticks firing and not, per file.

    PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/scripted-lookahead/usage_table.py \
        FILE [FILE ...]
"""

import json
import sys
from pathlib import Path

from lean.contract import FIRES, POWERS, STICKS
from lean.game import card

POWER_NAMES = ("coast", "dive", "climb")


def stick_name(index: int) -> str:
    level = card.STICK_LEVELS[index]
    if level == 0:
        return "straight"
    return f"{card.LEVEL_NAMES[int(abs(level)) - 1]} {'L' if level > 0 else 'R'}"


def main():
    runs = [(Path(name).stem, json.loads(Path(name).read_text())) for name in sys.argv[1:]]
    header = "| stick | power | " + " | ".join(f"{name} (no fire / fire)" for name, _ in runs)
    print(header + " |")
    print("|---|---|" + "---|" * len(runs))
    for stick in range(STICKS):
        for power in range(POWERS):
            cells = []
            for _, run in runs:
                index = (stick * POWERS + power) * FIRES
                usage = run["usage"]
                cells.append(f"{100 * usage[index]:.2f}% / {100 * usage[index + 1]:.2f}%")
            print(f"| {stick_name(stick)} | {POWER_NAMES[power]} | " + " | ".join(cells) + " |")
    for name, run in runs:
        unused = sum(1 for share in run["usage"] if share == 0.0)
        rare = sum(1 for share in run["usage"] if share < 1e-3)
        print(
            f"\n{name}: entropy {run['entropy']:.3f} nats, kills per env-minute "
            f"{run['kills_per_environment_minute']:.2f}, unused actions {unused}, "
            f"under 0.1% {rare}, "
            f"games {run['ended']} (draws {run['draws']})"
        )


if __name__ == "__main__":
    main()
