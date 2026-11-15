"""Markdown tables of sweep.py's records (data/*.jsonl): per candidate, the Yongma team's
score against the Jujak team (wins + half the draws, SE), its kill share, kills a minute,
and the kill aspects (mutual head-on / offset head-on / flank / tail) in the energy-turn games
and in the mixed lineups (both "mixed" jobs pooled).

    uv run --no-sync python docs/rewrite/sections/balance-final/table.py data/FILE.jsonl ...
"""

import json
import sys


def pct(value: float) -> str:
    return f"{100 * value:.1f}"


def main() -> None:
    print(
        "| file | candidate | Yongma team (%) | kill share (%) | kills/min | aspects E v T (%) "
        "| aspects mixed (%) | kills/min mixed |"
    )
    print("|---|---|---|---|---|---|---|---|")
    for path in sys.argv[1:]:
        for line in open(path):
            record = json.loads(line)
            jobs = record["jobs"]
            energy = jobs["energy v turn"]
            mixed = [jobs["mixed"], jobs["mixed reversed"]]
            aspects = " / ".join(pct(x) for x in energy["kill_share"])
            mixed_aspects = " / ".join(
                pct((a + b) / 2) for a, b in zip(*(m["kill_share"] for m in mixed), strict=True)
            )
            mixed_rate = sum(m["kills_per_minute_mean"] for m in mixed) / 2
            print(
                f"| {path.split('/')[-1].removesuffix('.jsonl')} | {record['name']} "
                f"| {pct(energy['score'][0])} ± {pct(energy['score'][1])} "
                f"| {pct(energy['kill_score'][0])} | {energy['kills_per_minute_mean']:.2f} "
                f"| {aspects} | {mixed_aspects} | {mixed_rate:.2f} |"
            )


if __name__ == "__main__":
    main()
