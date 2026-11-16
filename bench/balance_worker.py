"""A persistent balance worker (the balance-final worker's form, on `bench.balance_sweep`):
one process per condition compiles the game loop once, then plays every job file dropped into
its queue directory, so a new batch of candidates starts in seconds instead of after a compile.

    PYTHONPATH=src:. python -m bench.balance_worker --condition deathmatch-5v5 \
        --queue queue/deathmatch-5v5 [--variants 32] [--lock /tmp/gpu.lock]

A job is a JSON file in the queue directory, taken in name order:

    {"candidates": "path.json" or [...],   # sweep candidates (`bench.balance_sweep`), or
     "pairs": "path.json" or [...],        # card pairs: any card A against any card B
     "out": "out/name.jsonl", "games": 512, "mixed_games": 0, "seed": 0}

Each entry's line is appended to `out` as its chunk finishes (an entry already there is
skipped, so a restarted worker resumes a job). A finished job file moves to done/, a failed
one to failed/ with the error in a .txt beside it; the worker keeps going. Everything a job
can change is data on the compiled loop (the card table through `Loop.set_table`, doctrines
and lineups through the layout, the spawn seed), so no job recompiles: the warm-up plays a
job-like batch (`balance_sweep.warm`) before any lock is taken. The variants per card table
and the environments are fixed per worker. With `--lock`, each job holds the GPU lock (flock)
while it plays, after its cards' characteristics are measured on the CPU (`prewarm`).
"""

import argparse
import contextlib
import dataclasses
import fcntl
import json
import pathlib
import time
import traceback

from bench.balance_sweep import (
    BALANCE,
    CONDITIONS,
    as_pair,
    balance_loop,
    mode_parameters,
    prewarm,
    sweep,
    warm,
)


@contextlib.contextmanager
def held(lock: pathlib.Path | None):
    """The GPU lock (an exclusive flock on the file) for the duration, or nothing."""
    if lock is None:
        yield
        return
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def entries_of(job: dict) -> list[dict]:
    """A job's entries: its candidates or pairs, inline or from a file."""
    entries = job["candidates"] if "candidates" in job else job["pairs"]
    if isinstance(entries, str):
        entries = json.loads(pathlib.Path(entries).read_text())
    return entries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=CONDITIONS, required=True)
    parser.add_argument("--queue", type=pathlib.Path, required=True)
    parser.add_argument("--environments", type=int, default=4096)
    parser.add_argument("--variants", type=int, default=32)
    parser.add_argument("--seconds", type=float, default=None, help="a check's short clock")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poll", type=float, default=5.0, help="seconds between queue looks")
    parser.add_argument("--eager", action="store_true", help="no torch.compile (a CPU check)")
    parser.add_argument("--prewarm-processes", type=int, default=12)
    parser.add_argument("--once", action="store_true", help="stop when the queue is empty")
    parser.add_argument("--lock", type=pathlib.Path, default=None, help="a GPU lock (flock)")
    args = parser.parse_args()
    condition = CONDITIONS[args.condition]
    parameters = mode_parameters(condition)
    for sub in ("done", "failed"):
        (args.queue / sub).mkdir(parents=True, exist_ok=True)
    settings = dataclasses.replace(BALANCE, seconds=args.seconds)
    begun = time.time()
    loop = balance_loop(
        condition, args.environments, args.variants, settings, args.device, not args.eager
    )
    warm(loop, parameters, args.variants)  # compile outside the lock, the job's own graph
    print(
        f"worker {args.condition}: ready in {time.time() - begun:.0f} s, watching {args.queue}",
        flush=True,
    )
    while True:
        jobs = sorted(args.queue.glob("*.json"))
        if not jobs:
            if args.once:
                return
            time.sleep(args.poll)
            continue
        path = jobs[0]
        try:
            job = json.loads(path.read_text())
            entries = entries_of(job)
            out = pathlib.Path(job["out"])
            out.parent.mkdir(parents=True, exist_ok=True)
            done = set()
            if out.exists():
                done = {json.loads(line)["name"] for line in out.read_text().splitlines()}
            entries = [entry for entry in entries if entry["name"] not in done]
            print(f"job {path.name}: {len(entries)} entries -> {out}", flush=True)
            if entries:
                started = time.time()
                pairs = [as_pair(entry) for entry in entries]
                prewarm(loop, pairs, args.variants, args.prewarm_processes)
                print(
                    f"job {path.name}: cards measured in {time.time() - started:.0f} s", flush=True
                )
                with held(args.lock):
                    sweep(
                        loop, entries, out, int(job.get("games", 512)),
                        int(job.get("mixed_games", 0)), int(job.get("seed", 0)), args.variants,
                        args.condition, parameters,
                        log=lambda line: print(line, flush=True),
                    )  # fmt: skip
            path.rename(args.queue / "done" / path.name)
            print(f"job {path.name}: done", flush=True)
        except Exception:  # a bad job must not stop the worker
            (args.queue / "failed" / f"{path.name}.txt").write_text(traceback.format_exc())
            if path.exists():
                path.rename(args.queue / "failed" / path.name)
            print(f"job {path.name}: failed\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
