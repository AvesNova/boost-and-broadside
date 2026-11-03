"""The skeleton benchmark: 3 sealed random sizes x 1 ship budget, plus CPU realtime at one.

    uv run --no-sync python -m bench.run [--seed N]

Each cell runs in a fresh process. Results are appended to bench/results.jsonl with the
commit. Sized to finish in about five minutes on the dev laptop (compilation dominates).
"""

import argparse
import json
import os
import subprocess
import sys
import time

from bench import sizes
from bench.lock import HELD_VARIABLE, bench_lock

HERE = os.path.dirname(os.path.abspath(__file__))
REALTIME_SIZE = dict(ships_per_team=5, num_fields=10, arena_radius=2600.0)


def cells(seed: int) -> list[dict]:
    """The skeleton cells of a seed: three drawn training sizes, then the realtime cell. The
    cell keys (such as `envs`) are bench/results.jsonl's format."""
    drawn = [
        dict(kind="train", size=size, envs=sizes.environments_for(size))
        for size in sizes.draw(seed, 3)
    ]
    return drawn + [dict(kind="realtime", size=REALTIME_SIZE)]


def run_cell(cell: dict) -> dict:
    """One cell in a fresh process (the lock passed through): its result, or its timeout or
    crash."""
    variables = dict(os.environ, **{HELD_VARIABLE: "1"})
    if cell["kind"] == "realtime":
        variables["CUDA_VISIBLE_DEVICES"] = ""
    started = time.time()
    try:
        process = subprocess.run(
            [sys.executable, "-m", "bench.cell", json.dumps(cell)],
            env=variables,
            cwd=os.path.dirname(HERE),
            capture_output=True,
            text=True,
            timeout=900,
        )
    except subprocess.TimeoutExpired:
        return dict(cell=cell, status="timeout")
    for line in process.stdout.splitlines():
        if line.startswith("RESULT "):
            return dict(json.loads(line[7:]), status="ok", wall_s=time.time() - started)
    return dict(cell=cell, status="crash", error=(process.stderr or process.stdout)[-3000:])


def main() -> None:
    """Run every cell of the seed under the lock, appending each result."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=sizes.SKELETON_SEED)
    args = parser.parse_args()
    commit = subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=HERE
    ).stdout.strip()
    with bench_lock(), open(os.path.join(HERE, "results.jsonl"), "a") as results:
        for cell in cells(args.seed):
            result = run_cell(cell)
            result.update(commit=commit, seed=args.seed, stamp=time.strftime("%Y-%m-%d %H:%M:%S"))
            results.write(json.dumps(result) + "\n")
            results.flush()
            size = cell["size"]
            label = (
                f"{cell['kind']:8} {2 * size['ships_per_team']:3d} ships {size['num_fields']:2d} "
                f"fields R {size['arena_radius']:6.0f}"
            )
            if result["status"] != "ok":
                print(f"{label}: {result['status']} {result.get('error', '')[-500:]}", flush=True)
            elif cell["kind"] == "train":
                rollout = result["rollout_env_ticks_per_s"]
                print(
                    f"{label} x{cell['envs']:5d} envs: rollout {rollout:,.0f}"
                    f" env-ticks/s, train {result['train_env_ticks_per_s']:,.0f}, learn "
                    f"{result['learn_ms']:.0f} ms, VRAM {result['peak_vram_gb']:.2f} GB, "
                    f"compile {result['compile_s']:.0f} s",
                    flush=True,
                )
            else:
                print(
                    f"{label}: p50 {result['p50_ms']:.2f} ms, p99 {result['p99_ms']:.2f} ms, "
                    f"compile {result['compile_s']:.0f} s",
                    flush=True,
                )


if __name__ == "__main__":
    main()
