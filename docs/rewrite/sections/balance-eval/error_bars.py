"""Are the per-game standard errors right? The Jujak and Yongma mirrors at deathmatch 5v5,
4 seeds x 512 games each (`round2_run.sh`): per metric, the standard error each seed's record
gives against (1) the spread of the 4 seeds' values and (2) the mirror's known truth, 50%
(seats alternate on the same card), as the mean z^2 over all 8 seed-mirror values (1 when the
errors are right; its own 95% range for 8 values is about 0.27-1.94).

    python docs/rewrite/sections/balance-eval/error_bars.py docs/rewrite/sections/balance-eval/data
"""

import json
import math
import pathlib
import statistics
import sys

METRICS = ("kill_share_a", "point_share_a", "kill_score", "score")


def main() -> None:
    folder = pathlib.Path(sys.argv[1])
    rows: dict[str, dict[str, list[tuple[float, float]]]] = {}
    for path in sorted(folder.glob("mirrors-deathmatch-5v5-seed*.jsonl")):
        for line in path.read_text().splitlines():
            record = json.loads(line)
            job = record["jobs"]["energy v turn"]
            for metric in METRICS:
                rows.setdefault(metric, {}).setdefault(record["name"], []).append(
                    tuple(job[metric])
                )
    print("| metric | mirror | seeds: value ± se | mean se | sd between seeds / mean se |")
    print("|---|---|---|---|---|")
    for metric, by_mirror in rows.items():
        squares = []
        for mirror, values in by_mirror.items():
            means = [v for v, _ in values]
            errors = [e for _, e in values]
            spread = statistics.stdev(means) if len(means) > 1 else float("nan")
            mean_error = statistics.mean(errors)
            squares += [((v - 0.5) / e) ** 2 for v, e in values if e > 0]
            listed = ", ".join(f"{100 * v:.2f} ± {100 * e:.2f}" for v, e in values)
            print(
                f"| {metric} | {mirror} | {listed} | {100 * mean_error:.2f} "
                f"| {spread / mean_error:.2f} |"
            )
        mean_square = statistics.mean(squares)
        print(
            f"| {metric} | both: mean z^2 against 50% over {len(squares)} | "
            f"{mean_square:.2f} (error scale {math.sqrt(mean_square):.2f}) | | |"
        )


if __name__ == "__main__":
    main()
