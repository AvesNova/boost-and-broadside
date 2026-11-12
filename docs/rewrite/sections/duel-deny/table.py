"""The duel-deny table of a `sizes.py` result: per candidate, the mirrors' 1v1 mutual head-on
kill share, life and merges a kill; energy v turn by size; the score against the versus
player at 5v5 (mixed / reversed).

    uv run --no-sync --active python docs/rewrite/sections/duel-deny/table.py search/sizes.json
"""

import json
import pathlib
import sys

HERE = pathlib.Path(__file__).parent


def merges(row: dict) -> float:
    return row["engagement"]["two turn diameters"]["merges_per_kill_mean"]


def main() -> None:
    result = json.loads((HERE / sys.argv[1]).read_text())
    sizes = sorted(result["sizes"], key=int)
    lines = [
        "| Candidate | JJ 1v1 mutual, life, merges | YY 1v1 mutual, life, merges | E v T "
        + " / ".join(f"{s}v{s}" for s in sizes)
        + " | v r9 (5v5 mixed / reversed) |",
        "|---|---|---|---|---|",
    ]
    for candidate in result["candidates"]:
        name = candidate["name"]

        def row(size: str, lineup: str):
            return next(
                (
                    r
                    for r in result["sizes"].get(size, [])
                    if r["candidate"] == name and r["lineup"] == lineup
                ),
                None,
            )

        def mirror(lineup: str) -> str:
            r = row("1", lineup)
            if r is None:
                return ""
            return f"{r['kill_share'][0]:.2f}, {r['life_seconds']:.0f} s, {merges(r):.1f}"

        evt = " / ".join(
            f"{r['score'][0]:.2f}" if (r := row(s, "energy v turn")) else "" for s in sizes
        )
        versus = " / ".join(
            f"{r['score'][0]:.3f}"
            for lineup in ("mixed", "mixed reversed")
            for r in result["versus"]
            if r["candidate"] == name and r["lineup"] == lineup
        )
        lines.append(
            f"| {name} | {mirror('turn mirror')} | {mirror('energy mirror')} | {evt} | {versus} |"
        )
    print("\n".join(lines))


if __name__ == "__main__":
    main()
