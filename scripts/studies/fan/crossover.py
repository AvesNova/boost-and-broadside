"""50% crossover opponent count for the final live and averaged 731 checkpoints.

Usage: crossover.py OUT.json fan_731_n8.jsonl [fan_731_n16.jsonl ...]
Linear interpolation of win rate (ties as non-wins) between the bracketing grid points;
null when the win rate never crosses 50% on the grid. Raw points are included.
"""

import json
import sys


def crossing(pts):
    pts = sorted(pts, key=lambda r: r["scripted"])
    for a, b in zip(pts, pts[1:]):
        if a["win_rate"] >= 0.5 > b["win_rate"]:
            f = (a["win_rate"] - 0.5) / (a["win_rate"] - b["win_rate"])
            return a["scripted"] + f * (b["scripted"] - a["scripted"])
    return None


out = {
    "run": "wild-universe-731",
    "method": "linear interpolation of win rate (ties as non-wins) between grid points",
    "sizes": {},
}
for path in sys.argv[2:]:
    rows = [json.loads(line) for line in open(path) if line.strip()]
    N = rows[0]["trained"]
    entry = {}
    for key, name in (
        ("final_live", "/step_000671846400.pt"),
        ("final_averaged", "/avg_step_000671846400.pt"),
    ):
        pts = [r for r in rows if r["ckpt"].endswith(name)]
        if not pts:
            continue
        x = crossing(pts)
        entry[key] = {
            "checkpoint": pts[0]["ckpt"].split("/")[-1],
            "crossover_opponents": x,
            "crossover_ratio": None if x is None else x / N,
            "points": [
                {
                    k: r[k]
                    for k in ("trained", "scripted", "wins", "losses", "ties", "games", "win_rate")
                    if k in r
                }
                | {"ci95": 1.96 * r["se"]}
                for r in sorted(pts, key=lambda r: r["scripted"])
            ],
        }
    out["sizes"][str(N)] = entry
json.dump(out, open(sys.argv[1], "w"), indent=1)
for N, e in out["sizes"].items():
    print(
        N,
        {
            k: (round(v["crossover_opponents"], 2) if v["crossover_opponents"] else None)
            for k, v in e.items()
        },
    )
