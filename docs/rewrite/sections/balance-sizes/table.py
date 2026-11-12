"""A sizes.py result as a compact table: per candidate and size, E v T score / kill share,
and the mirrors' mutual head-on kill share and life."""

import json
import pathlib
import sys

d = json.loads((pathlib.Path(__file__).parent / sys.argv[1]).read_text())
for c in d["candidates"]:
    cells = []
    for s, rows in d["sizes"].items():
        mine = {r["lineup"]: r for r in rows if r["candidate"] == c["name"]}
        r = mine["energy v turn"]
        cell = f"{s}v{s} {r['score'][0]:.2f}/{r['kill_score'][0]:.3f}"
        for lineup, tag in (("turn mirror", "JJ"), ("energy mirror", "YY")):
            if lineup in mine:
                m = mine[lineup]
                cell += f" {tag} m{m['kill_share'][0]:.2f} L{m['life_seconds']:.0f}"
        cells.append(cell)
    print(f"{c['name'][:44]:<44} " + " | ".join(cells))
