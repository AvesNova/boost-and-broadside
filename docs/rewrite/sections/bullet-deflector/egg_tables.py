"""Markdown tables of egg_play.py's cell records.

PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/bullet-deflector/egg_tables.py DIR
"""

import json
import sys
from pathlib import Path

COLUMNS = (
    ("kills/min", "kills_per_minute", "{:.2f}"),
    ("TTK median s", "time_to_kill_median_s", "{:.2f}"),
    ("TTK p25-p75 s", None, None),
    ("draws", "draw_share", "{:.2f}"),
    ("hit chance/bullet", "hit_chance_per_bullet", "{:.3f}"),
    ("shots/kill", "shots_per_kill", "{:.0f}"),
    ("friendly fire", "friendly_fire_share", "{:.3f}"),
    ("head-on kills", "head_on_kill_share", "{:.2f}"),
    ("fire share", "fire_share", "{:.2f}"),
    ("store", "mean_store", "{:.2f}"),
    ("power coast/dive/climb", None, None),
    ("entropy", "entropy", "{:.2f}"),
    ("stall", "stall_share", "{:.3f}"),
    ("in enemy egg", "inside_enemy_egg_share", "{:.3f}"),
    ("enemy <30 m", "enemy_within_30m_share", "{:.3f}"),
)
TRACE = (
    ("passes", "passes", "{:d}"),
    ("hits/pass", "hits_per_pass_mean", "{:.2f}"),
    ("no-hit passes", "passes_with_no_hit", "{:.2f}"),
    ("hits in control", "hits_in_control_share", "{:.2f}"),
    ("merge turn 90 s", "merge_turn_90_median_s", "{:.2f}"),
    ("hits head-on/side/rear", None, None),
)


def cell(record, key, form, label):
    if label == "TTK p25-p75 s":
        low, high = record.get("time_to_kill_p25_s"), record.get("time_to_kill_p75_s")
        return "-" if low is None else f"{low:.2f}-{high:.2f}"
    if label == "power coast/dive/climb":
        return "/".join(f"{p:.2f}" for p in record["power_use"])
    if label == "hits head-on/side/rear":
        a = record.get("hits_by_aspect")
        return "-" if a is None else "/".join(f"{a[k]:.2f}" for k in ("head_on", "side", "rear"))
    value = record.get(key)
    return "-" if value is None else form.format(value)


def table(records, columns):
    head = "| cell | " + " | ".join(label for label, _, _ in columns) + " |"
    rule = "|---" * (len(columns) + 1) + "|"
    rows = [
        "| "
        + r["cell"]
        + " | "
        + " | ".join(cell(r, k, f, label) for label, k, f in columns)
        + " |"
        for r in records
    ]
    return "\n".join([head, rule] + rows)


def main():
    folder = Path(sys.argv[1])
    records = [json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))]
    print(table(records, COLUMNS))
    traced = [r for r in records if "passes" in r]
    if traced:
        print()
        print(table(traced, TRACE))


if __name__ == "__main__":
    main()
