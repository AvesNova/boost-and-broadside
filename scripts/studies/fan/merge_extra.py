"""Merge base + extra fan rows per (ckpt, trained, scripted) into fan_731_nN.jsonl.

Sums the counts and recomputes the rates. Usage: merge_extra.py SIZE [SIZE ...], run in the
fan directory. Extra rows of ladder checkpoints are skipped: the ladder stays at its seed-1
games, and only the final live and averaged checkpoints get the seed-2 games.
"""

import json
import math
import sys
from collections import OrderedDict

for n in sys.argv[1:]:
    rows = OrderedDict()
    for src in (f"fan_731_n{n}.base.jsonl", f"fan_731_n{n}.extra.jsonl"):
        try:
            lines = open(src).read().splitlines()
        except FileNotFoundError:
            continue
        for line in lines:
            if not line.strip():
                continue
            r = json.loads(line)
            k = (r["ckpt"], r["trained"], r["scripted"])
            if src.endswith(".extra.jsonl") and "ladder_step" in r["ckpt"]:
                continue  # fan snapshots stay at the base games
            if k not in rows:
                rows[k] = dict(r, steps_total=r["mean_steps"] * r["games"])
                continue
            m = rows[k]
            for f in ("wins", "losses", "ties", "capped", "games", "seconds"):
                m[f] = m.get(f, 0) + r.get(f, 0)
            m["steps_total"] += r["mean_steps"] * r["games"]
            m["max_steps"] = max(m.get("max_steps", 0), r.get("max_steps", 0))
    with open(f"fan_731_n{n}.jsonl", "w") as f:
        for m in rows.values():
            g = m["games"]
            m["win_rate"] = m["wins"] / g
            m["se"] = math.sqrt(m["win_rate"] * (1 - m["win_rate"]) / g)
            m["mean_steps"] = m.pop("steps_total") / g
            f.write(json.dumps(m) + "\n")
    print(n, len(rows), "points, games per point", sorted({m["games"] for m in rows.values()}))
