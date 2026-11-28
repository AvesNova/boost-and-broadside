"""Item 12: the behaviour tables, core's rules against the new, from behaviour.py's JSON.

    uv run --no-sync python docs/rewrite/sections/rules-12/tables.py DATA > tables.md

DATA holds core/ and new/ (one JSON per agent and size)."""

import json
import sys
from pathlib import Path

ROWS = (
    ("games", "games", "{:.0f}"),
    ("kills_per_minute", "kills per game-minute", "{:.2f}"),
    ("collision_death_share", "deaths by collision", "{:.1%}"),
    ("bullet_death_share", "deaths by bullet", "{:.1%}"),
    ("contacts_per_game", "collisions per game", "{:.2f}"),
    ("mutual_share_of_contacts", "collisions killing both", "{:.2%}"),
    ("mutual_share_of_fatal_contacts", "fatal collisions killing both", "{:.1%}"),
    ("tail_share_of_bullet_hits", "tail-chase share of bullet hits", "{:.1%}"),
    ("mean_hit_range", "mean hit range, m", "{:.0f}"),
    ("bullet_hits_per_ship_minute", "bullet hits per ship-minute", "{:.2f}"),
    ("shots_per_ship_minute", "shots per ship-minute", "{:.0f}"),
    ("mean_length_seconds", "mean game length, s", "{:.1f}"),
    ("wipe_share", "games ending by elimination", "{:.1%}"),
)


def main() -> None:
    data = Path(sys.argv[1])
    for agent in ("centre", "yardstick"):
        print(f"\n### {agent} (mirror), fields and fog on\n")
        header = ["measure"]
        cells = {key: [label] for key, label, _ in ROWS}
        for size in (1, 4, 16):
            for rules in ("core", "new"):
                path = data / rules / f"{agent}_{size}v{size}.json"
                if not path.exists():
                    continue
                result = json.loads(path.read_text())
                header.append(f"{size}v{size} {rules}")
                for key, _, form in ROWS:
                    cells[key].append(form.format(result[key]))
        print("| " + " | ".join(header) + " |")
        print("|" + "---|" * len(header))
        for key, _, _ in ROWS:
            print("| " + " | ".join(cells[key]) + " |")


if __name__ == "__main__":
    main()
