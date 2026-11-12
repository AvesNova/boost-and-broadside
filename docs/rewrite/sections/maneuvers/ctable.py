"""The confirm tables of a `sizes.py` result run at sizes 1 2 5 with versus at 1 2 5.

    python docs/rewrite/sections/maneuvers/ctable.py confirm/sizes.json

Table 1 (strength and balance): against r9 at 1v1 (JJ / YY), 2v2 (mixed / reversed), 5v5
(mixed / reversed); energy v turn 1v1 / 2v2 / 5v5. Table 2 (the duels, self-play 1v1): per
mirror the mutual head-on kill share, life, merges a kill (mean / median / p90), one /
two-circle / scissors, side-by-side share, post-merge choices (same / opposite / extend).
Table 3 (the crowd, self-play energy v turn 5v5): life, kills a minute a ship, the post-merge
choices by type (same / opposite / extend).
"""

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent


def score(r):
    return f"{r['score'][0]:.3f}" if r else "-"


def choices(r):
    out = []
    for kind, c in r.get("post_merge_choices", {}).items():
        out.append(
            kind[0].upper()
            + " "
            + "/".join(f"{c[k]['share']:.2f}" for k in ("same sense", "opposite sense", "extend"))
        )
    return ", ".join(out)


def duel(r):
    if r is None:
        return "-"
    e = r["engagement"]["two turn diameters"]
    kinds = "/".join(f"{v:.2f}" for v in e["one_two_scissors"])
    side = r.get("two_circles", {}).get("side by side", {}).get("share", float("nan"))
    return (
        f"{r['kill_share'][0]:.2f}, {r['life_seconds']:.0f} s, {e['merges_per_kill_mean']:.1f}/"
        f"{e['merges_per_kill_median']}/{e['merges_per_kill_p90']}; {kinds}; sbs {side:.2f}; "
        f"{choices(r)}"
    )


def main() -> None:
    result = json.loads((HERE / sys.argv[1]).read_text())

    def find(name, size, lineup, versus):
        rows = (
            [r for r in result["versus"] if r.get("size") == size]
            if versus
            else result["sizes"].get(str(size), [])
        )
        return next((r for r in rows if r["candidate"] == name and r["lineup"] == lineup), None)

    one = [
        "| Candidate | v r9 1v1 JJ / YY | v r9 2v2 m / r | v r9 5v5 m / r | E v T 1 / 2 / 5 |",
        "|---|---|---|---|---|",
    ]
    two = ["| Candidate | JJ self-play | YY self-play |", "|---|---|---|"]
    three = [
        "| Candidate | E v T 5v5 life, kills/min/ship | choices by type (same/opp/extend) |",
        "|---|---|---|",
    ]
    for c in result["candidates"]:
        n = c["name"]

        def versus(size, first, second):
            return f"{score(find(n, size, first, True))} / {score(find(n, size, second, True))}"

        evt = " / ".join(
            score(find(n, s, "energy v turn", False))[:4]
            if find(n, s, "energy v turn", False)
            else "-"
            for s in (1, 2, 5)
        )
        duels = versus(1, "mixed reversed", "mixed")
        pairs = versus(2, "mixed", "mixed reversed")
        crowd = versus(5, "mixed", "mixed reversed")
        one.append(f"| {n} | {duels} | {pairs} | {crowd} | {evt} |")
        jj, yy = find(n, 1, "turn mirror", False), find(n, 1, "energy mirror", False)
        two.append(f"| {n} | {duel(jj)} | {duel(yy)} |")
        r5 = find(n, 5, "energy v turn", False)
        if r5:
            pace = f"{r5['life_seconds']:.0f} s, {r5['kills_per_minute_per_ship']:.2f}"
            three.append(f"| {n} | {pace} | {choices(r5)} |")
    print("\n".join(one) + "\n\n" + "\n".join(two) + "\n\n" + "\n".join(three))


if __name__ == "__main__":
    main()
