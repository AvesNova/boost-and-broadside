"""The r9m2-maps tables of a `sizes.py` result: per map, r9 then r9m2.

    python docs/rewrite/sections/r9m2-maps/table.py ../maneuvers-2/r9m2-maps/sizes.json

Columns: A v B (ship A's score, the Yongma slot) at each size; the A v A and B v B 1v1
mirrors (mutual head-on kill share, life, merges a kill mean / median / first-pass / p90, H
the post-merge choice entropy in bits over same sense / opposite / extend / turn v extender);
the pace (life a ship and kills a minute a ship at 1v1 / 5v5 / 10v10 A v B; trades at 5v5);
the 5v5 A v B choice entropy and choice shares; and, when the run had --versus players,
each candidate's score against them per size and lineup.
"""

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE.parent / "maneuvers-2"))

from table import choices, entropy  # noqa: E402


def mirror(r: dict | None) -> str:
    if r is None:
        return ""
    e = r["engagement"]["two turn diameters"]
    return (
        f"{r['kill_share'][0]:.2f}, {r['life_seconds']:.0f} s; {e['merges_per_kill_mean']:.1f} / "
        f"{e['merges_per_kill_median']:.0f} / {e['first_pass_kill_share']:.2f} / "
        f"{e['merges_per_kill_p90']:.0f}; H {entropy(r):.2f}"
    )


def main() -> None:
    result = json.loads((HERE / sys.argv[1]).read_text())
    sizes = sorted(result["sizes"], key=int)
    lines = [
        "| Map, doctrine | A v B "
        + " / ".join(f"{s}v{s}" for s in sizes)
        + " | AA 1v1: mutual, life; merges m / med / first / p90; H | BB 1v1 | "
        "Life s 1v1 / 5v5 / 10v10 | Kills/min/ship 1v1 / 5v5 / 10v10 | Trades 5v5 | "
        "5v5 H; choices s/o/e/t; net |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for c in result["candidates"]:
        name = c["name"]
        shown = name.replace(" | ", ", ")

        def row(size, lineup):
            return next(
                (
                    r
                    for r in result["sizes"].get(str(size), [])
                    if r["candidate"] == name and r["lineup"] == lineup
                ),
                None,
            )

        evt = {s: row(s, "energy v turn") for s in sizes}
        balance = " / ".join(f"{r['score'][0]:.2f}" if (r := evt[s]) else "-" for s in sizes)
        pace_sizes = [s for s in ("1", "5", "10") if s in evt and evt[s]]
        life = " / ".join(f"{evt[s]['life_seconds']:.0f}" for s in pace_sizes)
        kpm = " / ".join(f"{evt[s]['kills_per_minute_per_ship']:.2f}" for s in pace_sizes)
        five = evt.get("5")
        trades = f"{five['trade_fraction']:.2f}" if five else ""
        h5 = f"{entropy(five):.2f}; {choices(five)}" if five else ""
        lines.append(
            f"| {shown} | {balance} | {mirror(row(1, 'energy mirror'))} | "
            f"{mirror(row(1, 'turn mirror'))} | {life} | {kpm} | {trades} | {h5} |"
        )
    print("\n".join(lines))
    if result.get("versus"):
        print()
        print(strength(result["versus"]))


def strength(rows: list[dict]) -> str:
    """Each candidate's score against each --versus player per size: every lineup (side A the
    candidate) and their mean, the SE of the mean from the lineups' SEs."""
    lineups = list(dict.fromkeys(r["lineup"] for r in rows))
    lines = [
        "| Candidate | v | size | " + " | ".join(lineups) + " | mean |",
        "|---|---|---|" + "---|" * (len(lineups) + 1),
    ]
    keys = list(dict.fromkeys((r["candidate"], r["b"], r["size"]) for r in rows))
    for name, b, size in keys:
        key = (name, b, size)
        by = {r["lineup"]: r for r in rows if (r["candidate"], r["b"], r["size"]) == key}
        scores = [by[lineup]["score"] for lineup in lineups if lineup in by]
        mean = sum(s[0] for s in scores) / len(scores)
        se = sum(s[1] ** 2 for s in scores) ** 0.5 / len(scores)
        cells = " | ".join(f"{by[x]['score'][0]:.2f}" if x in by else "-" for x in lineups)
        shown = name.replace(" | ", ", ")
        lines.append(f"| {shown} | {b} | {size}v{size} | {cells} | {mean:.3f} ± {se:.3f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
