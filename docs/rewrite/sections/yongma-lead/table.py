"""The yongma-lead table of a `sizes.py` result: per card and doctrine, the self-play energy
v turn (the Yongma's score) by size, the score against r9m2 at 1v1 and 5v5 (the mean of
energy v turn / turn v energy / mixed / mixed reversed, as Elo), and the Yongma mirror at
1v1 (mutual head-on kill share, life, merges a kill).

    python docs/rewrite/sections/yongma-lead/table.py sizes.json
"""

import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
LINEUPS = ("energy v turn", "turn v energy", "mixed", "mixed reversed")


def elo(s: float) -> float:
    return 400 * math.log10(s / (1 - s))


def main() -> None:
    path = HERE / sys.argv[1]
    result = json.loads(path.read_text())
    sizes = sorted(int(s) for s in result["sizes"])
    versus_sizes = sorted({r["size"] for r in result["versus"]})
    versus = {(r["candidate"], r["size"], r["lineup"]): r["score"] for r in result["versus"]}
    selfplay = {
        (r["candidate"], int(s), r["lineup"]): r
        for s, rows in result["sizes"].items()
        for r in rows
    }
    lines = [
        "| Card: doctrine | E v T "
        + " / ".join(f"{s}v{s}" for s in sizes)
        + " | "
        + " | ".join(f"v r9m2 {s}v{s} (Elo)" for s in versus_sizes)
        + " | YY 1v1: mutual, life, merges a kill |",
        "|---" * (len(versus_sizes) + 3) + "|",
    ]
    for c in result["candidates"]:
        name = c["name"]
        tilt = " / ".join(
            f"{selfplay[(name, s, 'energy v turn')]['score'][0]:.3f}"
            if (name, s, "energy v turn") in selfplay
            else ""
            for s in sizes
        )
        cells = []
        for s in versus_sizes:
            scores = [versus.get((name, s, lineup)) for lineup in LINEUPS]
            if None in scores:
                cells.append("")
                continue
            mean = sum(x[0] for x in scores) / 4
            error = math.sqrt(sum(x[1] ** 2 for x in scores)) / 4
            spread = 400 / math.log(10) * error / (mean * (1 - mean))
            cells.append(
                " / ".join(f"{x[0]:.3f}" for x in scores) + f" ({elo(mean):+.0f} ± {spread:.0f})"
            )
        mirror = selfplay.get((name, 1, "energy mirror"))
        yy = ""
        if mirror is not None:
            e = mirror["engagement"]["two turn diameters"]
            yy = (
                f"{mirror['kill_share'][0]:.2f}, {mirror['life_seconds']:.0f} s, "
                f"{e['merges_per_kill_mean']:.2f}"
            )
        lines.append(f"| {name} | {tilt} | " + " | ".join(cells) + f" | {yy} |")
    text = "\n".join(lines) + "\n"
    path.with_suffix(".table.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
