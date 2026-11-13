"""summarise.py MACHINE LABEL RAW.json [GPU-FILE] [skip=2]: one record line (JSON) from a
steady.py output, to append to records/records.jsonl.

Steady state: the updates after the first `skip` (warm-up: the first compiles, the first
cadence). The steady SPS is weighted as a run has it, one cadence update in ten
(`metrics.due`): (9 x the plain updates' mean seconds + the cadence updates' mean) / 10.
"""

import json
import statistics
import sys


def main() -> None:
    machine, label, raw, *rest = sys.argv[1:]
    options = dict(a.split("=", 1) for a in rest if "=" in a)
    gpu_file = next((a for a in rest if "=" not in a), None)
    skip = int(options.get("skip", 2))
    r = json.load(open(raw))
    updates = r["updates"]
    steady = updates[skip:]
    plain = [u for u in steady if u["cadence_seconds"] is None]
    cadence = [u for u in steady if u["cadence_seconds"] is not None]
    steps = r["steps_per_update"]

    def mean(rows, key):
        return statistics.fmean(u[key] for u in rows) if rows else None

    plain_seconds = mean(plain, "seconds")
    cadence_seconds = mean(cadence, "seconds")
    weighted = (9 * plain_seconds + cadence_seconds) / 10 if plain and cadence else plain_seconds
    compiles: dict = {}
    for e in r["compiles"]:
        c = compiles.setdefault(e["label"], [0.0, 0, 0])
        c[0] += e["seconds"]
        c[1] += e.get("ops", 0)
        c[2] += 1
    record = dict(
        machine=machine,
        label=label,
        gpu=r["gpu"],
        environments=r["environments"],
        lane_environments=r["lane_environments"],
        micro_batch_rows=r["micro_batch_rows"],
        micro_batch_note=r["micro_batch_note"],
        options=r["options"],
        steps_per_update=steps,
        build_seconds=round(r["build_seconds"], 1),
        first_update_end_since_start=round(updates[0]["end_since_start"], 1),
        update_seconds=[round(u["seconds"], 2) for u in updates],
        steady_updates=len(steady),
        steady_plain_updates=len(plain),
        steady_steps_per_second_plain=steps / plain_seconds if plain else None,
        steady_steps_per_second_one_cadence_in_ten=steps / weighted if weighted else None,
        steady_rollout_ms=round(1000 * mean(plain, "rollout_seconds"), 1) if plain else None,
        steady_learn_ms=round(1000 * mean(plain, "learner_seconds"), 1) if plain else None,
        cadence_update_seconds=round(cadence_seconds, 2) if cadence else None,
        cadence_seconds=round(mean(cadence, "cadence_seconds"), 2) if cadence else None,
        head_sums_seconds_per_cadence_update=round(mean(cadence, "head_sums_seconds"), 2)
        if cadence
        else None,
        head_sums_calls_per_cadence_update=mean(cadence, "head_sums_calls") if cadence else None,
        first_update_head_sums=[updates[0]["head_sums_seconds"], updates[0]["head_sums_calls"]],
        peak_allocated_gb=round(r["peak_allocated_gb"], 3),
        peak_reserved_gb=round(r["peak_reserved_gb"], 3),
        compile_seconds=round(r["compile_seconds"], 1),
        compiles={k: [round(v[0], 1), v[1], v[2]] for k, v in compiles.items()},
        gpu_state=open(gpu_file).read().splitlines() if gpu_file else None,
        torch=r["torch"],
    )
    print(json.dumps(record))


if __name__ == "__main__":
    main()
