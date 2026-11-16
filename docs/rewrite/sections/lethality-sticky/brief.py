"""One line per lineup of measure.py's data/LABEL.json files: python brief.py LABEL..."""

import json
import pathlib
import sys

DATA = pathlib.Path(__file__).parent / "data"
for label in sys.argv[1:]:
    for lineup, entry in json.loads((DATA / f"{label}.json").read_text()).items():
        r = entry["summary"]
        score, share = r["score_a"][0], r["kill_share_a"][0]
        if lineup.endswith("turn v energy"):
            score, share = 1 - score, 1 - share
        hit = r.get("hit_rate")
        print(
            f"{label:10} {lineup:20} hit {hit if hit is None else round(hit, 4)} "
            f"passes {r['passes_per_kill']:.2f} no-merge {r['no_merge_kills'][0]:.2f} "
            f"kpm/ship {r['kills_per_minute_per_ship']:.3f} Yongma score {score:.3f} "
            f"share {share:.3f}"
        )
