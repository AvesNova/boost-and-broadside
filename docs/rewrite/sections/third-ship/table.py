"""A results file's candidates as a table: C v J, C v Y, the mixed (doctrine v the base) and
the mirror's pace, each score ± its standard error.

    .venv/bin/python docs/rewrite/sections/third-ship/table.py data/X.jsonl [--sort mixed]
"""

import argparse
import json
import pathlib


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("files", type=pathlib.Path, nargs="+")
    parser.add_argument("--sort", default=None)
    args = parser.parse_args()
    rows = []
    for path in args.files:
        for line in path.read_text().splitlines():
            r = json.loads(line)
            j = r["jobs"]
            row = {"name": r["name"]}
            for label in ("Y v J", "C v J", "C v Y"):
                if label in j:
                    row[label] = tuple(j[label]["score"])
            if "mixed" in j:
                a, b = j["mixed"]["score"], j["mixed reversed"]["score"]
                row["mixed"] = ((a[0] + b[0]) / 2, ((a[1] ** 2 + b[1] ** 2) ** 0.5) / 2)
            if "C mirror" in j:
                row["kpm"] = j["C mirror"]["kills_per_minute_mean"]
            rows.append(row)
    if args.sort:
        rows.sort(key=lambda r: -r.get(args.sort, (-1, 0))[0])
    for r in rows:
        cells = [
            f"{k} {v[0]:.3f}±{v[1]:.3f}" if isinstance(v, tuple) else f"{k} {v:.2f}"
            for k, v in r.items()
            if k != "name"
        ]
        print(r["name"], " | ".join(cells))


if __name__ == "__main__":
    main()
