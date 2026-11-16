"""The before/after table from data/*.json (measure.py's)."""

import json
import pathlib

DATA = pathlib.Path(__file__).parent / "data"
LABELS = {
    "before": "v22, old Imugi",
    "lethality": "v23, old Imugi",
    "noextend": "v23, sticky, no extension",
    "after2": "v23, sticky + extension",
}
LINEUPS = ("mixed", "energy v turn", "turn v energy")
print(
    "| run | lineup | passes/kill | no-merge kills | engagement s | kills/min/ship | "
    "first hit to kill s | Yongma score | Yongma kill share |"
)
print("|---|---|---|---|---|---|---|---|---|")
for label, name in LABELS.items():
    rows = json.loads((DATA / f"{label}.json").read_text())
    for lineup in LINEUPS:
        r = rows[lineup]["summary"]
        score, share = r["score_a"][0], r["kill_share_a"][0]
        if lineup == "turn v energy":  # side A is Jujak: Yongma's is the rest
            score, share = 1 - score, 1 - share
        yongma = "" if lineup == "mixed" else f"{score:.3f}"
        yongma_share = "" if lineup == "mixed" else f"{share:.3f}"
        print(
            f"| {name} | {lineup} | {r['passes_per_kill']:.2f} | {r['no_merge_kills'][0]:.2f} | "
            f"{r['engagement_length_median']} | {r['kills_per_minute_per_ship']:.2f} | "
            f"{r['first_hit_to_kill_median']} | {yongma} | {yongma_share} |"
        )
