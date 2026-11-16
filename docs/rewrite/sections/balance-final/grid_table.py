"""One row per candidate over several sweep records (data/*.jsonl): the Yongma team's score in
each file, the worst distance from 50%, and Jujak's sustained turn over Yongma's (the
archetypes' distinctness; 1.29 before). Sorted by the turn ratio, most distinct first, among
the candidates within `--band` of 50% in every file.

    uv run --no-sync python docs/rewrite/sections/balance-final/grid_table.py \
        data/grid1-deathmatch-2v2.jsonl data/grid1-deathmatch-5v5.jsonl ... [--band 5]
"""

import argparse
import json
import pathlib
import tomllib

HERE = pathlib.Path(__file__).parent
BEFORE = {
    key: tomllib.loads((HERE / "before" / f"{key}.toml").read_text())["handles"]
    for key in ("yongma", "jujak")
}


def turn_ratio(candidate: dict) -> float:
    handles = candidate.get("handles", {})
    jujak = BEFORE["jujak"]["sustained_turn_rate"] * handles.get("turn_handles", {}).get(
        "sustained_turn_rate", 1.0
    )
    yongma = BEFORE["yongma"]["sustained_turn_rate"] * handles.get("energy_handles", {}).get(
        "sustained_turn_rate", 1.0
    )
    return jujak / yongma


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="+")
    parser.add_argument("--band", type=float, default=100.0)
    args = parser.parse_args()
    scores: dict[str, dict[str, float]] = {}
    candidates = {}
    labels = []
    for path in args.files:
        label = pathlib.Path(path).stem
        labels.append(label)
        for line in open(path):
            record = json.loads(line)
            scores.setdefault(record["name"], {})[label] = (
                100 * record["jobs"]["energy v turn"]["score"][0]
            )
            candidates[record["name"]] = record["candidate"]
    rows = []
    for name, by_file in scores.items():
        if len(by_file) < len(labels):
            continue
        worst = max(abs(v - 50) for v in by_file.values())
        rows.append((turn_ratio(candidates[name]), worst, name, by_file))
    print("| candidate | J/Y turn | worst from 50 | " + " | ".join(labels) + " |")
    print("|---|---|---|" + "---|" * len(labels))
    for ratio, worst, name, by_file in sorted(rows, key=lambda r: (-(r[1] <= args.band), -r[0])):
        cells = " | ".join(f"{by_file[label]:.1f}" for label in labels)
        mark = "" if worst <= args.band else " (out)"
        print(f"| {name}{mark} | {ratio:.3f} | {worst:.1f} | {cells} |")


if __name__ == "__main__":
    main()
