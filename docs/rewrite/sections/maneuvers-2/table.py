"""The maneuvers-2 tables of a `sizes.py` result.

    python docs/rewrite/sections/maneuvers-2/table.py round1/sizes.json [--versus r9m]

Per candidate: its score against the versus player (1v1 YY / JJ, i.e. mixed / reversed;
2v2 and 5v5 mixed / reversed) and the worst of them; the 5v5 self-play (energy v turn: the
Yongma's score, life, the post-merge choice entropy H in bits over same sense / opposite /
extend / turn v extender, each choice's share and its net outcome: the chooser's kills less
its deaths, a choice); the 1v1 mirrors (JJ, YY: mutual head-on kill share, life, merges a
kill, H).
"""

import json
import math
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
CHOICES = ("same sense", "opposite sense", "extend", "turn v extender")


def pooled(row: dict | None) -> list[list[float]] | None:
    """Per choice [count, kills, killed] pooled over the choosers' kinds."""
    if row is None or not row.get("post_merge_choices"):
        return None
    out = [[0.0, 0.0, 0.0] for _ in CHOICES]
    for choices in row["post_merge_choices"].values():
        for i, c in enumerate(CHOICES):
            n = choices[c]["count"]
            out[i][0] += n
            out[i][1] += n * choices[c]["outcome"][0]
            out[i][2] += n * choices[c]["outcome"][1]
    return out


def entropy(row: dict | None) -> float:
    counts = pooled(row)
    if counts is None:
        return math.nan
    total = sum(c[0] for c in counts)
    if total <= 0:
        return math.nan
    return -sum(c[0] / total * math.log2(c[0] / total) for c in counts if c[0] > 0)


def choices(row: dict | None) -> str:
    """Shares s/o/e/t and net outcomes (kills - deaths a choice)."""
    counts = pooled(row)
    if counts is None:
        return ""
    total = sum(c[0] for c in counts) or 1.0
    shares = "/".join(f"{c[0] / total:.2f}" for c in counts)
    nets = "/".join(f"{(c[1] - c[2]) / c[0]:+.2f}" if c[0] > 0 else "-" for c in counts)
    return f"{shares}; net {nets}"


def mirror(r: dict | None) -> str:
    if r is None:
        return ""
    e = r["engagement"]["two turn diameters"]
    return (
        f"{r['kill_share'][0]:.2f}, {r['life_seconds']:.0f} s, "
        f"{e['merges_per_kill_mean']:.1f}, H {entropy(r):.2f}"
    )


def main() -> None:
    path = HERE / sys.argv[1]
    result = json.loads(path.read_text())
    versus = sys.argv[sys.argv.index("--versus") + 1] if "--versus" in sys.argv else None
    sizes = result["sizes"]
    lines = [
        "| Candidate | v 1v1 YY / JJ | 2v2 | 5v5 | worst | 5v5 E v T, life, H | "
        "5v5 choices s/o/e/t; net | JJ 1v1 mutual, life, merges, H | YY 1v1 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in result["candidates"]:
        name = c["name"]

        def row(size, lineup, b=None):
            if b is None:
                rows = sizes.get(str(size), [])
                return next(
                    (r for r in rows if r["candidate"] == name and r["lineup"] == lineup), None
                )
            return next(
                (
                    r
                    for r in result["versus"]
                    if r["candidate"] == name
                    and r["lineup"] == lineup
                    and r["size"] == size
                    and (versus is None or r["b"] == versus)
                ),
                None,
            )

        cells, scores = [], []
        for size in (1, 2, 5):
            pair = [row(size, lu, b=True) for lu in ("mixed", "mixed reversed")]
            got = [p["score"][0] for p in pair if p]
            scores += got
            cells.append(" / ".join(f"{s:.3f}" for s in got))
        worst = f"{min(scores):.3f}" if scores else ""
        five = row(5, "energy v turn")
        evt = (
            f"{five['score'][0]:.2f}, {five['life_seconds']:.0f} s, H {entropy(five):.2f}"
            if five
            else ""
        )
        lines.append(
            f"| {name} | {' | '.join(cells)} | {worst} | {evt} | {choices(five)} | "
            f"{mirror(row(1, 'turn mirror'))} | {mirror(row(1, 'energy mirror'))} |"
        )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
