"""The yongma-lead summary of `sizes.py` results: the Jujak flies plain r9m2 in every game, so
the self-play energy v turn and the energy v turn against r9m2 are the same game (the
variant's Yongma against r9m2's Jujak) and pool; at 1v1 the "mixed" lineup against r9m2 is
the Yongma mirror (the variant's Yongma against r9m2's), and turn v energy and the 1v1
mixed reversed (the Jujak mirror) are plain r9m2 on both sides (controls). Per card and
doctrine: the Yongma's pooled E v T score by size, its change from the same run's r9m2,
the Yongma mirror head-to-head (1v1 mixed), and the strength in crowds (the 5v5 and 10v10
mixed / mixed reversed against r9m2, as Elo).

    python docs/rewrite/sections/yongma-lead/summary.py sizes.json [confirm.json ...]
"""

import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).parent


def elo(s: float) -> float:
    return 400 * math.log10(s / (1 - s))


def pooled(cells: list) -> tuple[float, float] | None:
    """The inverse-variance mean of [score, se] cells."""
    cells = [c for c in cells if c is not None]
    if not cells:
        return None
    weights = [1 / c[1] ** 2 for c in cells]
    mean = sum(w * c[0] for w, c in zip(weights, cells, strict=True)) / sum(weights)
    return mean, math.sqrt(1 / sum(weights))


def main() -> None:
    lines = []
    for argument in sys.argv[1:]:
        result = json.loads((HERE / argument).read_text())
        sizes = sorted(int(s) for s in result["sizes"])
        versus = {(r["candidate"], r["size"], r["lineup"]): r["score"] for r in result["versus"]}
        selfplay = {
            (r["candidate"], int(s), r["lineup"]): r["score"]
            for s, rows in result["sizes"].items()
            for r in rows
        }
        crowd_sizes = sorted({r["size"] for r in result["versus"] if r["size"] > 1})
        lines += [
            f"{argument}:",
            "",
            "| Card: doctrine | Yongma E v T "
            + " / ".join(f"{s}v{s}" for s in sizes)
            + " (change from r9m2) | YY 1v1 v r9m2 | "
            + " | ".join(f"v r9m2 {s}v{s} mixed / reversed (Elo)" for s in crowd_sizes)
            + " |",
            "|---" * (3 + len(crowd_sizes)) + "|",
        ]
        evt = {}
        for c in result["candidates"]:
            name = c["name"]
            for s in sizes:
                evt[(name, s)] = pooled(
                    [
                        selfplay.get((name, s, "energy v turn")),
                        versus.get((name, s, "energy v turn")),
                    ]
                )
        for c in result["candidates"]:
            name = c["name"]
            control = name.split(" | ")[0] + " | r9m2"
            tilt = []
            for s in sizes:
                own, base = evt[(name, s)], evt.get((control, s))
                if own is None:
                    tilt.append("")
                elif name == control or base is None:
                    tilt.append(f"{own[0]:.3f}")
                else:
                    tilt.append(f"{own[0]:.3f} ({own[0] - base[0]:+.3f})")
            yy = versus.get((name, 1, "mixed"))
            crowds = []
            for s in crowd_sizes:
                cells = [versus.get((name, s, lineup)) for lineup in ("mixed", "mixed reversed")]
                both = pooled(cells)
                if both is None:
                    crowds.append("")
                    continue
                spread = 400 / math.log(10) * both[1] / (both[0] * (1 - both[0]))
                crowds.append(
                    " / ".join(f"{x[0]:.3f}" for x in cells if x is not None)
                    + f" ({elo(both[0]):+.0f} ± {spread:.0f})"
                )
            lines.append(
                f"| {name} | {' / '.join(tilt)} | "
                + (f"{yy[0]:.3f} ± {yy[1]:.3f}" if yy else "")
                + " | "
                + " | ".join(crowds)
                + " |"
            )
        lines.append("")
    text = "\n".join(lines)
    (HERE / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
