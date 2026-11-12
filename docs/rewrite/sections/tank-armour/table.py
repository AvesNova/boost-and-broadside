"""The tank-armour tables of a `sizes.py` result.

    uv run --no-sync --active python docs/rewrite/sections/tank-armour/table.py round1/sizes.json
    ... table.py round1/sizes.json --fights   # the two-circle and post-merge choice table

Main table, per candidate: the mirrors' 1v1 mutual head-on kill share, life and merges a kill
(mean / median / first-pass share / p90, two turn diameters); energy v turn by size; the
score against the versus player at 5v5 (mixed / reversed); life and kills a minute a ship
over every lineup played (min-max); the largest trade fraction; the mixed 1v1 kill mix.
"""

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
CHOICES = ("same sense", "opposite sense", "extend", "turn v extender")


def merges(row: dict) -> str:
    e = row["engagement"]["two turn diameters"]
    median = e["merges_per_kill_median"]
    p90 = e["merges_per_kill_p90"]
    return (
        f"{e['merges_per_kill_mean']:.1f} / {median if median is None else round(median)} / "
        f"{e['first_pass_kill_share']:.2f} / {p90 if p90 is None else round(p90)}"
    )


def load(path: str) -> dict:
    return json.loads((HERE / path).read_text())


def finder(result: dict, name: str):
    def row(size: str, lineup: str):
        return next(
            (
                r
                for r in result["sizes"].get(size, [])
                if r["candidate"] == name and r["lineup"] == lineup
            ),
            None,
        )

    return row


def main_table(result: dict) -> str:
    sizes = sorted(result["sizes"], key=int)
    lines = [
        "| Candidate | JJ 1v1 mutual, life; merges m / med / first / p90 | YY 1v1 mutual, life; "
        "merges | E v T " + " / ".join(f"{s}v{s}" for s in sizes) + " | v r9 5v5 | Life s "
        "| Kills/min/ship | Trades max | Mixed 1v1 kills m / o / f / t |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for candidate in result["candidates"]:
        name = candidate["name"]
        row = finder(result, name)

        def mirror(lineup: str) -> str:
            r = row("1", lineup)
            if r is None:
                return ""
            return f"{r['kill_share'][0]:.2f}, {r['life_seconds']:.0f} s; {merges(r)}"

        evt = [row(s, "energy v turn") for s in sizes]
        everything = [r for r in evt if r] + [
            r for lineup in ("turn mirror", "energy mirror") if (r := row("1", lineup))
        ]
        if not everything:
            continue
        lives = [r["life_seconds"] for r in everything]
        paces = [r["kills_per_minute_per_ship"] for r in everything]
        trades = max(r["trade_fraction"] for r in everything)
        versus = " / ".join(
            f"{r['score'][0]:.3f}"
            for lineup in ("mixed", "mixed reversed")
            for r in result["versus"]
            if r["candidate"] == name and r["lineup"] == lineup
        )
        one = evt[0]
        mix = " / ".join(f"{v:.2f}" for v in one["kill_share"]) if one else ""
        lines.append(
            f"| {name.replace(' | ', ': ')} | {mirror('turn mirror')} | "
            f"{mirror('energy mirror')} | "
            + " / ".join(f"{r['score'][0]:.2f}" if r else "" for r in evt)
            + f" | {versus} | {min(lives):.0f}-{max(lives):.0f} | "
            f"{min(paces):.2f}-{max(paces):.2f} | {trades:.2f} | {mix} |"
        )
    return "\n".join(lines)


def fights_table(result: dict) -> str:
    """Per candidate and 1v1 lineup: two-circle fights overlapping / side by side (share;
    the share ending in a kill), and each kind's post-merge choices (share; kills / killed)."""
    lines = [
        "| Candidate | Lineup | Two-circle: overlapping, side by side (share; kill) "
        "| Chooser | Same sense | Opposite sense | Extend | Turn v extender |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for candidate in result["candidates"]:
        name = candidate["name"]
        row = finder(result, name)
        for lineup in ("turn mirror", "energy mirror", "energy v turn"):
            r = row("1", lineup)
            if r is None or "two_circles" not in r:
                continue
            circles = r["two_circles"]
            two = ", ".join(
                f"{circles[k]['share']:.2f}; {circles[k]['kill']:.2f}"
                for k in ("overlapping", "side by side")
            )
            for kind, choices in r["post_merge_choices"].items():
                cells = []
                for c in CHOICES:
                    v = choices[c]
                    cells.append(f"{v['share']:.2f}; {v['outcome'][0]:.2f} / {v['outcome'][1]:.2f}")
                lines.append(
                    f"| {name.replace(' | ', ': ')} | {lineup} | {two} | {kind} | "
                    + " | ".join(cells)
                    + " |"
                )
                two = ""
    return "\n".join(lines)


def main() -> None:
    result = load(sys.argv[1])
    if "--fights" in sys.argv:
        print(fights_table(result))
    else:
        print(main_table(result))


if __name__ == "__main__":
    main()
