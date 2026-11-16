"""The deathmatch clock study (`clock_5090.sh`): the Yongma team's kill share had the clock
stopped after each minute of the same 600 s games, per size, and its standard error.

    cd docs/rewrite/sections/balance-eval && python clock_table.py data/box5090/clock-*.jsonl
"""

import json
import pathlib
import sys


def main() -> None:
    rows = []
    for name in sys.argv[1:]:
        for line in pathlib.Path(name).read_text().splitlines():
            record = json.loads(line)
            rows.append((record["condition"], record["jobs"]["energy v turn"]))
    minutes = len(rows[0][1]["kill_share_by_minute"])
    header = " | ".join(f"{m + 1} min" for m in range(minutes))
    print(f"| condition | games | {header} |")
    print("|---" * (minutes + 2) + "|")
    for condition, job in rows:
        cells = " | ".join(
            f"{100 * share:.1f} ± {100 * error:.1f}" for share, error in job["kill_share_by_minute"]
        )
        print(f"| {condition} | {job['games']} | {cells} |")


if __name__ == "__main__":
    main()
