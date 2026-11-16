"""The finals as Markdown: per balance option, each condition's Bonghwang v Jujak, Bonghwang v
Yongma and the option's own Yongma v Jujak (its reference entry's "C v J"), ± one SE.

    .venv/bin/python docs/rewrite/sections/third-ship/finals_table.py data/finals1-*.jsonl
"""

import json
import pathlib
import re
import sys

CONDITION = re.compile(r"finals\d+-(.+?)(-box|-3070)?\.jsonl")
NAMES = {
    "dm-1v1": "deathmatch 1v1",
    "dm2": "deathmatch 2v2",
    "dm-5v5": "deathmatch 5v5",
    "dm10": "deathmatch 10v10",
    "el5": "elimination 5v5",
    "fl5": "Frontline 5v5",
}


def cell(job: dict) -> str:
    score, se = job["score"]
    return f"{score:.3f} ± {se:.3f}"


def main() -> None:
    table: dict[str, dict[str, dict[str, str]]] = {}
    for path in map(pathlib.Path, sys.argv[1:]):
        condition = NAMES.get(CONDITION.match(path.name).group(1), path.name)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            option, name = row["name"].split(":", 1)
            slot = table.setdefault(option, {}).setdefault(condition, {})
            if name == "ref":
                slot["Y v J"] = cell(row["jobs"]["C v J"])
            else:
                slot.setdefault("card", name)
                if name == slot["card"]:
                    slot["C v J"] = cell(row["jobs"]["C v J"])
                    slot["C v Y"] = cell(row["jobs"]["C v Y"])
                    slot["games"] = str(row["jobs"]["C v J"]["games"])
    order = list(NAMES.values())
    for option, conditions in table.items():
        cards = {c.get("card") for c in conditions.values()}
        print(f"\n**{option}** (Bonghwang {', '.join(sorted(x for x in cards if x))})\n")
        print("| condition (games) | Bonghwang v Jujak | Bonghwang v Yongma | Yongma v Jujak |")
        print("|---|---|---|---|")
        for condition in sorted(conditions, key=lambda c: order.index(c) if c in order else 99):
            c = conditions[condition]
            print(
                f"| {condition} ({c.get('games', '?')}) | {c.get('C v J', '')} | "
                f"{c.get('C v Y', '')} | {c.get('Y v J', '')} |"
            )


if __name__ == "__main__":
    main()
