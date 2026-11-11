"""status.py [RUNS_DIR]: one line per stage of the one-head pipeline: the newest update's
environment steps, its seconds per update (median of the last 5), and the stage's ETA as a
KST clock time at that pace (finished stages say so). Reads only the runs' metrics.jsonl."""

import datetime
import json
import pathlib
import statistics
import sys

KST = datetime.timezone(datetime.timedelta(hours=9))
STAGES = [("2026-11-12-bc-onehead", 120586240), ("2026-11-12-rl-onehead", 180355072)]


def main() -> None:
    runs = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "runs")
    now = datetime.datetime.now(KST)
    for run, final in STAGES:
        path = runs / run / "metrics.jsonl"
        if not path.exists():
            print(f"{run}: not started")
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            print(f"{run}: no update yet")
            continue
        last = rows[-1]
        steps = last["counters/environment_steps"]
        seconds = statistics.median(row["train/time/update_seconds"] for row in rows[-5:])
        per_update = steps / last["counters/updates"]
        left = max(0, round((final - steps) / per_update))
        age = now.timestamp() - path.stat().st_mtime
        eta = now + datetime.timedelta(seconds=left * seconds)
        print(
            f"{run}: {steps / 1e6:.1f}M of {final / 1e6:.1f}M, {seconds:.0f} s/update "
            f"({per_update / seconds:,.0f} SPS), last row {age / 60:.0f} min ago, "
            + ("done" if left == 0 else f"ETA {eta:%a %H:%M} KST")
        )


if __name__ == "__main__":
    main()
