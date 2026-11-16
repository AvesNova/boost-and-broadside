"""Kill-share profile of each probe (candidates/probe1.json) over the modes and sizes: the
Yongma team's kill share minus the centre card's (E, the mean of "E" and "E-again"), in
kill points, per file; and the score. Reads data/<probe>-<mode>-<size>.jsonl files.

    uv run --no-sync python docs/rewrite/sections/balance-final/probe_table.py data/probe1-*.jsonl
"""

import json
import pathlib
import sys


def main() -> None:
    files = sorted(sys.argv[1:])
    table: dict[str, dict[str, tuple[float, float, float]]] = {}
    labels = []
    for path in files:
        label = pathlib.Path(path).stem.split("-", 1)[1]
        labels.append(label)
        for line in open(path):
            record = json.loads(line)
            job = record["jobs"]["energy v turn"]
            table.setdefault(record["name"].split("-", 1)[1], {})[label] = (
                100 * job["kill_score"][0],
                100 * job["kill_score"][1],
                100 * job["score"][0],
            )
    centre = {
        label: sum(table[n][label][0] for n in ("E", "E-again") if label in table.get(n, {}))
        / max(1, sum(label in table.get(n, {}) for n in ("E", "E-again")))
        for label in labels
    }
    print("kill share minus E's (kill points); score in brackets")
    print("| probe | " + " | ".join(labels) + " |")
    print("|---|" + "---|" * len(labels))
    print("| E (kill share) | " + " | ".join(f"{centre[l]:.1f}" for l in labels) + " |")
    for name, row in table.items():
        cells = [
            f"{row[l][0] - centre[l]:+.1f} ({row[l][2]:.0f})" if l in row else "-" for l in labels
        ]
        print(f"| {name} | " + " | ".join(cells) + " |")
    print("SE of one kill share (points): " + ", ".join(
        f"{l} {sum(r[l][1] for r in table.values() if l in r) / sum(l in r for r in table.values()):.2f}"
        for l in labels
    ))  # fmt: skip


if __name__ == "__main__":
    main()
