"""Item 13: the behaviour table, item 12's yardstick numbers (new rules, 60 s clock) beside the
shrinking zone's, from the behaviour.py JSON.

    uv run --no-sync python docs/rewrite/sections/zone-13/tables.py DATA > tables.md"""

import json
import sys
from pathlib import Path

# Item 12, yardstick mirror, new rules, fields and fog (docs/rewrite/sections/rules-12/tables.md).
ITEM_12 = {
    1: {"games": 4096, "kills_per_minute": 0.43, "collision_death_share": 0.007,
        "bullet_death_share": 0.991, "tail_share_of_bullet_hits": 0.757, "mean_hit_range": 121,
        "bullet_hits_per_ship_minute": 3.94, "shots_per_ship_minute": 13,
        "mean_length_seconds": 51.2, "wipe_share": 0.370},
    4: {"games": 2048, "kills_per_minute": 5.68, "collision_death_share": 0.031,
        "bullet_death_share": 0.969, "tail_share_of_bullet_hits": 0.534, "mean_hit_range": 157,
        "bullet_hits_per_ship_minute": 15.77, "shots_per_ship_minute": 59,
        "mean_length_seconds": 50.8, "wipe_share": 0.635},
}  # fmt: skip
ROWS = (
    ("games", "games", "{:.0f}"),
    ("mean_length_seconds", "game length mean, s", "{:.1f}"),
    ("median_length_seconds", "game length median, s", "{:.1f}"),
    ("p90_length_seconds", "game length p90, s", "{:.1f}"),
    ("max_length_seconds", "game length max, s (bound 131.1)", "{:.1f}"),
    ("wipe_share", "games ending by elimination", "{:.1%}"),
    ("one_team_wiped_share", "one team wiped (a win)", "{:.1%}"),
    ("both_wiped_share", "both wiped (a draw)", "{:.2%}"),
    ("kills_per_minute", "kills per game-minute", "{:.2f}"),
    ("bullet_death_share", "deaths by bullet", "{:.1%}"),
    ("collision_death_share", "deaths by collision", "{:.1%}"),
    ("boundary_death_share", "deaths by boundary", "{:.1%}"),
    ("tail_share_of_bullet_hits", "tail-chase share of bullet hits", "{:.1%}"),
    ("mean_hit_range", "mean hit range, m", "{:.0f}"),
    ("bullet_hits_per_ship_minute", "bullet hits per ship-minute", "{:.2f}"),
    ("shots_per_ship_minute", "shots per ship-minute", "{:.0f}"),
)


def main() -> None:
    data = Path(sys.argv[1])
    print("### yardstick (mirror), fields and fog on: item 12 (60 s clock) vs the zone\n")
    header = ["measure", "1v1 item 12", "1v1 zone", "4v4 item 12", "4v4 zone"]
    print("| " + " | ".join(header) + " |")
    print("|" + "---|" * len(header))
    results = {s: json.loads((data / f"yardstick_{s}v{s}.json").read_text()) for s in (1, 4)}
    for key, label, form in ROWS:
        cells = [label]
        for size in (1, 4):
            old = ITEM_12[size].get(key)
            cells.append("-" if old is None else form.format(old))
            cells.append(form.format(results[size][key]))
        print("| " + " | ".join(cells) + " |")
    print(
        "\nMap-edge target (living ships every 15th tick, m): quantiles 0 / 0.1% / 1% / 50% / "
        "99% / 99.9% / 1"
    )
    for size in (1, 4):
        q = results[size]["edge_quantiles"]
        print(f"- {size}v{size}: " + " / ".join(f"{value:,.0f}" for value in q.values()))


if __name__ == "__main__":
    main()
