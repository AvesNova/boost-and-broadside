"""The three-ship tables of a `threeship.py` result.

    python docs/rewrite/sections/ships-r9m2/table3.py threeship/A-result.json

Per lineup and size: side A's score (the first named ship's); for the mirrors and the pairs
at 1v1 the mutual head-on kill share, life and the post-merge choice entropy; for the mixed
lineups each type's kills a minute a ship and its kill/death ratio (both sides pooled).
"""

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "maneuvers-2"))

from table import entropy  # noqa: E402

NAMES = ("Y", "Me", "J", "Mt")


def main() -> None:
    result = json.loads((HERE / sys.argv[1]).read_text())
    sizes = sorted(result["sizes"], key=int)
    lineups = list(dict.fromkeys(r["lineup"] for s in sizes for r in result["sizes"][s]))
    by = {(s, r["lineup"]): r for s in sizes for r in result["sizes"][s]}
    print("| Lineup | " + " | ".join(f"{s}v{s}" for s in sizes) + " | 1v1 mutual, life, H |")
    print("|---|" + "---|" * (len(sizes) + 1))
    for lineup in lineups:
        cells = []
        for s in sizes:
            r = by.get((s, lineup))
            cells.append(f"{r['score'][0]:.2f}" if r else "")
        one = by.get(("1", lineup))
        shape = (
            f"{one['kill_share'][0]:.2f}, {one['life_seconds']:.0f} s, {entropy(one):.2f}"
            if one
            else ""
        )
        print(f"| {lineup} | " + " | ".join(cells) + f" | {shape} |")
    for s in sizes:
        for lineup in lineups:
            r = by.get((s, lineup))
            if not r or not lineup.startswith("mixed"):
                continue
            kills = r["kills_by_row"]  # killer side, killer row, victim row
            parts = []
            for row, name in enumerate(NAMES):
                k = sum(sum(kills[side][row]) for side in (0, 1))
                d = sum(kills[side][killer][row] for side in (0, 1) for killer in range(4))
                if k + d:
                    parts.append(f"{name} K/D {k / max(d, 1):.2f}")
            print(f"\n{s}v{s} {lineup}: A {r['score'][0]:.2f}; " + ", ".join(parts))


if __name__ == "__main__":
    main()
