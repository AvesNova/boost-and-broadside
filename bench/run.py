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
from bench.lock import HELD_ENV, bench_lock

HERE = os.path.dirname(os.path.abspath(__file__))
REALTIME_SIZE = dict(ships_per_team=5, num_fields=10, arena_radius=2600.0)


def cells(seed: int) -> list[dict]:
    out = [dict(kind="train", size=s, envs=sizes.envs_for(s)) for s in sizes.draw(seed, 3)]
    return out + [dict(kind="realtime", size=REALTIME_SIZE)]


def run_cell(cell: dict) -> dict:
    env = dict(os.environ, **{HELD_ENV: "1"})
    if cell["kind"] == "realtime":
        env["CUDA_VISIBLE_DEVICES"] = ""
    t0 = time.time()
    try:
        p = subprocess.run([sys.executable, "-m", "bench.cell", json.dumps(cell)], env=env,
                           cwd=os.path.dirname(HERE), capture_output=True, text=True, timeout=900)
    except subprocess.TimeoutExpired:
        return dict(cell=cell, status="timeout")
    for line in p.stdout.splitlines():
        if line.startswith("RESULT "):
            return dict(json.loads(line[7:]), status="ok", wall_s=time.time() - t0)
    return dict(cell=cell, status="crash", error=(p.stderr or p.stdout)[-3000:])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=sizes.SKELETON_SEED)
    args = ap.parse_args()
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True,
                            text=True, cwd=HERE).stdout.strip()
    with bench_lock(), open(os.path.join(HERE, "results.jsonl"), "a") as out:
        for cell in cells(args.seed):
            r = run_cell(cell)
            r.update(commit=commit, seed=args.seed, stamp=time.strftime("%Y-%m-%d %H:%M:%S"))
            out.write(json.dumps(r) + "\n")
            out.flush()
            s = cell["size"]
            label = f"{cell['kind']:8} {2 * s['ships_per_team']:3d} ships {s['num_fields']:2d} " \
                    f"fields R {s['arena_radius']:6.0f}"
            if r["status"] != "ok":
                print(f"{label}: {r['status']} {r.get('error', '')[-500:]}", flush=True)
            elif cell["kind"] == "train":
                print(f"{label} x{cell['envs']:5d} envs: rollout {r['rollout_env_ticks_per_s']:,.0f}"
                      f" env-ticks/s, train {r['train_env_ticks_per_s']:,.0f}, learn "
                      f"{r['learn_ms']:.0f} ms, VRAM {r['peak_vram_gb']:.2f} GB, compile "
                      f"{r['compile_s']:.0f} s", flush=True)
            else:
                print(f"{label}: p50 {r['p50_ms']:.2f} ms, p99 {r['p99_ms']:.2f} ms, compile "
                      f"{r['compile_s']:.0f} s", flush=True)


if __name__ == "__main__":
    main()
