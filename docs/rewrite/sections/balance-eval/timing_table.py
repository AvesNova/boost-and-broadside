"""The timing records (`bench.balance_sweep timing`, data/timing.jsonl) as a Markdown table.

cd docs/rewrite/sections/balance-eval && python timing_table.py data/timing.jsonl
"""

import json
import pathlib
import sys


def main() -> None:
    rows = [json.loads(line) for line in pathlib.Path(sys.argv[1]).read_text().splitlines()]
    print(
        "| condition | clock s | match s (mean) | batch s | games/s | game-s/s | s a card "
        "(256 games) | GPU util | busiest threads | kill share Y v J | compile s |"
    )
    print("|---" * 11 + "|")
    for r in rows:
        threads = " / ".join(f"{100 * t:.0f}%" for t in r["busiest_threads"][:2])
        print(
            f"| {r['condition']} | {r['clock_seconds']:.0f} | {r['match_seconds']:.0f} "
            f"| {r['batch_seconds']:.1f} | {r['games_per_second']:.1f} "
            f"| {r['game_seconds_per_second'] / 1e3:.1f}k | {r['seconds_per_card_256_games']:.2f} "
            f"| {r['gpu_utilisation_mean']:.0f}% ({r['gpu_samples']}) | {threads} "
            f"| {100 * r['kill_share_a'][0]:.1f} ± {100 * r['kill_share_a'][1]:.1f} "
            f"| {r['compile_seconds']} |"
        )


if __name__ == "__main__":
    main()
