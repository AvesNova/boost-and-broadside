"""The finals as Markdown: per balance option and Bonghwang card, each condition's Bonghwang v
Jujak and Bonghwang v Yongma, and the option's own Yongma v Jujak (its reference entry's
"C v J"), ± one SE.

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
    "dm5": "deathmatch 5v5",
    "dm10": "deathmatch 10v10",
    "el5": "elimination 5v5",
    "el10": "elimination 10v10",
    "fl5": "Frontline 5v5",
    "fl10": "Frontline 10v10",
}


def cell(job: dict) -> str:
    score, se = job["score"]
    return f"{score:.3f} ± {se:.3f}"


def main() -> None:
    cards: dict[tuple[str, str], dict[str, dict]] = {}
    references: dict[tuple[str, str], str] = {}
    for path in map(pathlib.Path, sys.argv[1:]):
        condition = NAMES.get(CONDITION.match(path.name).group(1), path.name)
        for line in path.read_text().splitlines():
            row = json.loads(line)
            option, name = row["name"].split(":", 1)
            if name == "ref":
                references[(option, condition)] = cell(row["jobs"]["C v J"])
            else:
                cards.setdefault((option, name), {})[condition] = row["jobs"]
    order = list(dict.fromkeys(NAMES.values()))
    for (option, name), conditions in cards.items():
        print(f"\n**{option}: {name}**\n")
        print("| condition (games) | Bonghwang v Jujak | Bonghwang v Yongma | Yongma v Jujak |")
        print("|---|---|---|---|")
        for condition in sorted(conditions, key=lambda c: order.index(c) if c in order else 99):
            jobs = conditions[condition]
            print(
                f"| {condition} ({jobs['C v J']['games']}) | {cell(jobs['C v J'])} | "
                f"{cell(jobs['C v Y'])} | {references.get((option, condition), '')} |"
            )


if __name__ == "__main__":
    main()
