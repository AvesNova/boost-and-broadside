"""Throughput and rating costs from the 5090 run records and the laptop live-Elo logs.

Usage: uv run --no-sync python docs/rewrite/sections/arm-costs/extract.py RUNS LIVE_ELO
RUNS is the box-5090-backup runs directory, LIVE_ELO the live-elo directory. Prints a
Markdown table per source; the output as of 2026-11-15 is in numbers.md beside this file.
"""

import datetime
import json
import re
import statistics
import sys
from pathlib import Path

UPDATE = re.compile(r"^update\s+(\d+)\s+(?:steps\s+([\d,]+)\s+)?rollout\s+(\d+) ms\s+learn\s+(\d+) ms")
BATCH = re.compile(r"^(\S+): batch (\d+), ([\d.]+) games.*\(the batch took (\d+) s\)")
RATING = re.compile(r"^(\d\d):(\d\d) rating (step_\d+)")
STEPS_PER_UPDATE = 1_048_576  # 2048 environments x 512 (bc-v20.log prints it per update)


def gpu(probe_total: int) -> str:
    gigabytes = probe_total / 2**30
    return "RTX 5090 (32 GB)" if gigabytes > 30 else f"laptop RTX 4070 ({gigabytes:.0f} GB)"


def wall_seconds(segment: dict) -> float | None:
    if not segment.get("end_time"):
        return None
    start = datetime.datetime.fromisoformat(segment["start_time"])
    end = datetime.datetime.fromisoformat(segment["end_time"])
    return (end - start).total_seconds()


def training(runs: Path) -> None:
    print("| run | GPU | updates logged | first update s | steady rollout s | steady learn s "
          "| steady SPS | segment 1 steps | segment 1 wall h | SPS incl. compile |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for name in ["bc-v19", "bc-v20", "bc-v21", "rl-v20", "rl-dm", "rl-kl", "rl-mix-kl", "rl-v20-seed1"]:
        record = json.loads((runs / name / "run.json").read_text())
        rows = [UPDATE.match(line) for line in (runs / f"{name}.log").read_text().splitlines()]
        rows = [(int(m[1]), int(m[3]) / 1e3, int(m[4]) / 1e3) for m in rows if m]
        first = rows[0][1] + rows[0][2]
        steady = [r for r in rows if r[0] >= 3]  # update 2 still compiles a little on some runs
        rollout = statistics.fmean(r[1] for r in steady)
        learn = statistics.fmean(r[2] for r in steady)
        segment = record["segments"][0]
        wall = wall_seconds(segment)
        end_step = segment.get("end_step")
        effective = f"{end_step / wall:,.0f}" if wall and end_step else "running"
        print(f"| {name} | {gpu(record['probe']['total'])} | {len(rows)} | {first:.0f} "
              f"| {rollout:.2f} | {learn:.2f} | {STEPS_PER_UPDATE / (rollout + learn):,.0f} "
              f"| {end_step or '-'} | {wall / 3600:.2f} | {effective} |" if wall else
              f"| {name} | {gpu(record['probe']['total'])} | {len(rows)} | {first:.0f} "
              f"| {rollout:.2f} | {learn:.2f} | {STEPS_PER_UPDATE / (rollout + learn):,.0f} | - | - | running |")


def ladder(live: Path) -> None:
    print()
    print("| pool | checkpoints | games per checkpoint (3 modes) | batch seconds per checkpoint "
          "| wall minutes per checkpoint (median) |")
    print("|---|---|---|---|---|")
    for log in sorted(live.glob("*/live_elo.log")):
        checkpoints: dict[str, dict] = {}
        times = []
        current = None
        for line in log.read_text().splitlines():
            if m := RATING.match(line):
                current = m[3]
                checkpoints.setdefault(current, {"games": 0.0, "seconds": 0.0})
                times.append(int(m[1]) * 60 + int(m[2]))
            elif (m := BATCH.match(line)) and current:
                # the games count is cumulative per mode within a checkpoint: keep each mode's last
                checkpoints[current].setdefault("mode_games", {})[m[1]] = float(m[3])
                checkpoints[current]["seconds"] += int(m[4])
        rated = [c for c in checkpoints.values() if c.get("mode_games")]
        if not rated:
            continue
        games = statistics.fmean(sum(c["mode_games"].values()) for c in rated)
        seconds = statistics.fmean(c["seconds"] for c in rated)
        gaps = [(b - a) % 1440 for a, b in zip(times, times[1:])]
        gap = f"{statistics.median(gaps):.0f}" if gaps else "-"
        print(f"| {log.parent.name} | {len(rated)} | {games:.0f} | {seconds:.0f} | {gap} |")


if __name__ == "__main__":
    training(Path(sys.argv[1]))
    ladder(Path(sys.argv[2]))
