"""Summary of the god-mode records (godmode.json = seed 0, godmode-seed*.json): the mean over
seeds and the range, per preset, as a Markdown table.

    uv run --no-sync python docs/rewrite/sections/bullet-deflector/summary.py
"""

import glob
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
ROWS = (
    ("god_mode", "damaging_hits_per_minute_per_game", "damaging hits / min / game", 1),
    ("god_mode", "bullet_damage_per_minute_per_game", "bullet damage / min / game", 0),
    ("god_mode", "hits_per_pass_mean", "damaging hits per pass (mean)", 2),
    ("god_mode", "passes_with_no_hit", "passes with no damaging hit", 3),
    ("god_mode", "passes_with_11_or_more", "passes with 11+ hits (a kill's worth)", 3),
    ("god_mode", "hits_in_control_share", "share of hits in control", 3),
    ("god_mode", "control_time_share", "share of ship-time in control", 3),
    ("god_mode", "contacts", "contacts (real bullets that touched)", 0),
    ("god_mode", "frontal_contacts", "contacts within 30 deg of the nose", 0),
    ("god_mode", "frontal_damaging_hits", "damaging hits within 30 deg of the nose", 0),
    ("god_mode", "pass_interval_median_s", "pass interval median, s", 2),
    ("kills_1v1", "kills_per_minute", "kills / min, 1v1", 3),
    ("kills_1v1", "draws", "1v1 draws", 0),
    ("kills_4v4", "kills_per_minute", "kills / min, 4v4 (all 8 ships)", 2),
)


def main():
    files = [os.path.join(HERE, "godmode.json")] + sorted(
        glob.glob(os.path.join(HERE, "godmode-seed*.json"))
    )
    runs = [json.load(open(path)) for path in files]
    presets = [record["god_mode"]["preset"] for record in runs[0]]
    print(f"{len(runs)} seeds x 128 games x 60 s each; mean (min-max over seeds)\n")
    print("| | " + " | ".join(presets) + " |")
    print("|---" * (len(presets) + 1) + "|")
    for block, key, label, digits in ROWS:
        cells = []
        for preset in presets:
            values = [
                r[block][key]
                for run in runs
                for r in [x for x in run if x["god_mode"]["preset"] == preset]
            ]
            mean = statistics.mean(values)
            cells.append(f"{mean:.{digits}f} ({min(values):.{digits}f}-{max(values):.{digits}f})")
        print(f"| {label} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
