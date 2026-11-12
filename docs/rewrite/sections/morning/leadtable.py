"""Item 3's table from `lead/sizes.json`: per card and doctrine, the score against r9m2 per
size (energy v turn / turn v energy / mixed / mixed reversed, and their mean as Elo) and the
self-play energy v turn (the Yongma's score: the tilt).

    python docs/rewrite/sections/morning/leadtable.py
"""

import json
import math
import pathlib

HERE = pathlib.Path(__file__).parent
LINEUPS = ("energy v turn", "turn v energy", "mixed", "mixed reversed")


def elo(s: float) -> float:
    return 400 * math.log10(s / (1 - s))


def main() -> None:
    result = json.loads((HERE / "lead" / "sizes.json").read_text())
    sizes = sorted({r["size"] for r in result["versus"]})
    versus = {(r["candidate"], r["size"], r["lineup"]): r["score"] for r in result["versus"]}
    selfplay = {
        (r["candidate"], int(s)): r["score"]
        for s, rows in result["sizes"].items()
        for r in rows
        if r["lineup"] == "energy v turn"
    }
    lines = [
        "| Card: doctrine | "
        + " | ".join(f"v r9m2 {s}v{s}: EvT / TvE / mix / rev (mean, Elo)" for s in sizes)
        + " | self-play E v T "
        + " / ".join(f"{s}v{s}" for s in sizes)
        + " |",
        "|---" * (len(sizes) + 2) + "|",
    ]
    for c in result["candidates"]:
        name = c["name"]
        cells = []
        for s in sizes:
            scores = [versus.get((name, s, lineup)) for lineup in LINEUPS]
            if None in scores:
                cells.append("")
                continue
            mean = sum(x[0] for x in scores) / 4
            error = math.sqrt(sum(x[1] ** 2 for x in scores)) / 4
            cells.append(
                " / ".join(f"{x[0]:.3f}" for x in scores)
                + f" ({mean:.3f}, {elo(mean):+.0f} ± {400 / math.log(10) * error / (mean * (1 - mean)):.0f})"
            )
        tilt = " / ".join(
            f"{selfplay[(name, s)][0]:.3f}" if (name, s) in selfplay else "" for s in sizes
        )
        lines.append(f"| {name} | " + " | ".join(cells) + f" | {tilt} |")
    text = "\n".join(lines) + "\n"
    (HERE / "lead" / "table.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
