"""curves.py CHAMPION_METRICS ARM_METRICS [--rl]: the two runs' training curves at matched
steps, as a Markdown table: each ~10M window's mean (the updates whose environment steps fall
in it) of BC KL and joint agreement (BC), or outcome Brier skill 5-15 s, outcome explained
variance and PPO KL (RL), and environment steps per second (steady updates: the first update
of each segment, which compiles, is left out). A resumed run's metrics.jsonl repeats the
updates after its last checkpoint: the last row of each update counts."""

import json
import statistics
import sys

BC_KEYS = ["bc/kl", "bc/agreement/joint", "outcome/brier_skill/5_15_seconds"]
RL_KEYS = ["outcome/brier_skill/5_15_seconds", "critic/explained_variance/outcome", "policy/kl"]
WINDOW = 10_485_760


def rows(path: str) -> list[dict]:
    by_update = {}
    for line in open(path):
        if line.strip():
            row = json.loads(line)
            by_update[row["counters/updates"]] = row
    return [by_update[u] for u in sorted(by_update)]


def windows(data: list[dict], keys: list[str]) -> dict[int, dict]:
    out = {}
    for row in data:
        w = (row["counters/environment_steps"] - 1) // WINDOW
        out.setdefault(w, []).append(row)
    table = {}
    for w, members in out.items():
        cells = {k: statistics.fmean(r[k] for r in members if k in r) for k in keys
                 if any(k in r for r in members)}
        steady = [r["train/environment_steps_per_second"] for r in members
                  if r["train/time/update_seconds"] < 1.5 * statistics.median(
                      m["train/time/update_seconds"] for m in data)]
        cells["sps"] = statistics.fmean(steady) if steady else float("nan")
        table[w] = cells
    return table


def main() -> None:
    champion, arm = rows(sys.argv[1]), rows(sys.argv[2])
    keys = RL_KEYS if "--rl" in sys.argv else BC_KEYS
    keys = [k for k in keys if any(k in r for r in champion)]
    a, b = windows(champion, keys), windows(arm, keys)
    names = keys + ["sps"]
    print("| steps (M) | " + " | ".join(f"{k} champion / one head" for k in names) + " |")
    print("|---|" + "---|" * len(names))
    for w in sorted(a.keys() & b.keys()):
        cells = [f"{a[w].get(k, float('nan')):.4g} / {b[w].get(k, float('nan')):.4g}" for k in names]
        print(f"| {w * WINDOW / 1e6:.0f}-{(w + 1) * WINDOW / 1e6:.0f} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
