"""The maneuvers tables of a `sizes.py` result.

    python docs/rewrite/sections/maneuvers/mtable.py round1/sizes.json [--card D2]

Per candidate, the mirrors in self-play at 1v1 (JJ, YY): the mutual head-on kill share,
life, merges a kill (mean / median / p90), one / two circle / scissors, side-by-side share
of the two-circle fights, the post-merge choices (same sense / opposite / extend); energy v
turn at 1v1; and the candidate against r9: 1v1 JJ, YY (and 5v5 mixed / reversed when
played), with the damage a game taken by the candidate / by r9 in the 1v1 JJ.
"""

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent


def mirror(r: dict | None) -> str:
    if r is None:
        return "|"
    e = r["engagement"]["two turn diameters"]
    med, p90 = e["merges_per_kill_median"], e["merges_per_kill_p90"]
    kinds = "/".join(f"{v:.2f}" for v in e["one_two_scissors"])
    side = r.get("two_circles", {}).get("side by side", {}).get("share", float("nan"))
    choices = r.get("post_merge_choices", {})
    chosen = " ".join(
        "/".join(f"{c[k]['share']:.2f}" for k in ("same sense", "opposite sense", "extend"))
        for c in choices.values()
    )
    return (
        f"{r['kill_share'][0]:.2f}, {r['life_seconds']:.0f} s, "
        f"{e['merges_per_kill_mean']:.1f}/{med}/{p90} | {kinds}, sbs {side:.2f}, ch {chosen}"
    )


def main() -> None:
    result = json.loads((HERE / sys.argv[1]).read_text())
    card = sys.argv[sys.argv.index("--card") + 1] if "--card" in sys.argv else None
    one = result["sizes"].get("1", [])
    lines = [
        "| Candidate | JJ mutual, life, merges m/med/p90 | JJ 1-c/2-c/sc, side by side, "
        "choices s/o/e | YY mutual, life, merges | YY shape | E v T 1v1 | v r9 JJ, YY 1v1 "
        "| JJ dmg own/r9 | v r9 5v5 m / r |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in result["candidates"]:
        name = c["name"]
        if card and not name.startswith(card + " |"):
            continue

        def row(lineup, rows=one):
            return next((r for r in rows if r["candidate"] == name and r["lineup"] == lineup), None)

        versus1 = [r for r in result["versus"] if r.get("size") == 1]
        versus5 = [r for r in result["versus"] if r.get("size") == 5]
        evt = row("energy v turn")
        jj, yy = row("mixed reversed", versus1), row("mixed", versus1)
        damage = ""
        if jj and "damage_by_side" in jj:
            damage = f"{jj['damage_by_side'][0][0]:.0f}/{jj['damage_by_side'][1][0]:.0f}"
        v5 = " / ".join(
            f"{r['score'][0]:.3f}" for lu in ("mixed", "mixed reversed") if (r := row(lu, versus5))
        )
        lines.append(
            f"| {name} | {mirror(row('turn mirror'))} | {mirror(row('energy mirror'))} | "
            f"{evt['score'][0]:.2f} | "
            if evt
            else f"| {name} | | | | | | "
        )
        lines[-1] += (
            " / ".join(f"{r['score'][0]:.3f}" for r in (jj, yy) if r) + f" | {damage} | {v5} |"
        )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
