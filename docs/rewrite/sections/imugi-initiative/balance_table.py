"""The balance spot check's table: Yongma's kill and point share (with standard errors),
before and after, from bench.action_mix's JSON (data/bal-{before,after}-<condition>.json)."""

import json
import pathlib
import sys

data = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).parent / "data")
BANDS = {"deathmatch-2v2": ".481-.521", "deathmatch-5v5": ".494-.507",
         "elimination-5v5": ".498-.545", "frontline-5v5": ".491-.513"}  # fmt: skip
print("| condition (512 games) | band | kill share before | after | point share before | after |")
print("|---|---|---|---|---|---|")
for condition, band in BANDS.items():
    cells = []
    for key in ("kill_share_a", "point_share_a"):
        for when in ("before", "after"):
            path = data / f"bal-{when}-{condition}.json"
            if path.exists():
                value, error = json.loads(path.read_text())[key]
                cells.append(f"{value:.3f} ± {error:.3f}")
            else:
                cells.append("-")
    print(f"| {condition} | {band} | " + " | ".join(cells) + " |")
