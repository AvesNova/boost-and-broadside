"""A persistent sweep worker: one process per mode and team size compiles the game loop once,
then plays every job file dropped into its queue directory, so a new batch of candidates
starts in seconds instead of after a 10-20 minute compile (the rented box's slow CPU).

    PYTHONPATH=src:. python docs/rewrite/sections/balance-final/worker.py \
        --mode deathmatch --ships 5 --queue /workspace/queue/deathmatch-5 [--variants 8]

A job is a JSON file in the queue directory, taken in name order:

    {"candidates": "path/to/candidates.json" or [...entries as in sweep.py...],
     "out": "/workspace/out/name.jsonl", "games": 512, "mixed_games": 128, "seed": 0}

Each candidate's line is appended to `out` as its chunk finishes (a candidate already there is
skipped, so a restarted worker resumes a job). A finished job file moves to done/, a failed
one to failed/ with the error in a .txt beside it; the worker keeps going. Everything a job
can change is data on the compiled loop (cards through `Loop.set_types`, doctrines and the
lineups through the layout), so no job recompiles. The card variants per batch and the
environments are fixed per worker (`--variants`, `--environments`); a job's chunk follows
them. The worker compiles before it takes any lock (one tick of a filler game). With
`--lock`, each job holds the GPU lock (flock) while it plays and releases it between jobs, so
other agents' flock-wrapped jobs interleave. Size a job to finish within about 45 minutes.
Progress lines go to stdout.

Any bench on `bench.game_loop.Loop` can use the same pattern: build the loop once, then call
the per-batch function on each job (here `sweep.sweep`).
"""

import argparse
import contextlib
import fcntl
import json
import pathlib
import sys
import time
import traceback

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))

from sweep import MODE_DEFAULTS, Loop, api, before_card, pack, sweep  # noqa: E402

from bench.game_loop import FILLER, Group  # noqa: E402


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("deathmatch", "elimination", "frontline"))
    parser.add_argument("--ships", type=int, default=5)
    parser.add_argument("--queue", type=pathlib.Path, required=True)
    parser.add_argument("--environments", type=int, default=4096)
    parser.add_argument("--variants", type=int, default=8)
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--poll", type=float, default=5.0, help="seconds between queue looks")
    parser.add_argument("--eager", action="store_true", help="no torch.compile (a CPU check)")
    parser.add_argument("--once", action="store_true", help="stop when the queue is empty")
    parser.add_argument(
        "--lock", type=pathlib.Path, default=None,
        help="a GPU lock file (flock) held for each job, not while idle (the box: /tmp/gpu.lock)",
    )  # fmt: skip
    args = parser.parse_args()
    preset = f"real_scale_{args.mode}"
    seconds = args.seconds or (300.0 if args.mode == "deathmatch" else None)
    seconds = seconds or api.PRESETS[preset].episode_seconds
    parameters = tuple(pack(MODE_DEFAULTS.sets[preset]))
    for sub in ("done", "failed"):
        (args.queue / sub).mkdir(parents=True, exist_ok=True)
    loop = Loop(
        args.environments,
        args.variants,
        seconds,
        ships_per_team=args.ships,
        mode=args.mode,
        fields=True,
        fixed_map=True,
        device=args.device,
        compiled=not args.eager,
    )
    loop.base = before_card(loop.base)
    # compile now, outside the GPU lock (torch.compile runs on the first tick): one tick of a
    # filler game (the third-ship agent's fix, tune/third-ship 8ad9de1c)
    loop._tick(*loop.start([Group(0, "mixed", FILLER, FILLER, loop.B)]))
    print(f"worker {args.mode} {args.ships}v{args.ships}: ready, watching {args.queue}", flush=True)
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
            entries = job["candidates"]
            if isinstance(entries, str):
                entries = json.loads(pathlib.Path(entries).read_text())
            out = pathlib.Path(job["out"])
            out.parent.mkdir(parents=True, exist_ok=True)
            done = set()
            if out.exists():
                done = {json.loads(line)["name"] for line in out.read_text().splitlines()}
            entries = [entry for entry in entries if entry["name"] not in done]
            games = int(job.get("games", 512))
            mixed = int(job.get("mixed_games", games)) // 2
            print(f"job {path.name}: {len(entries)} candidates -> {out}", flush=True)
            if entries:
                with held(args.lock):
                    sweep(
                        loop, entries, out, games, mixed, int(job.get("seed", 0)),
                        args.variants, args.mode, args.ships, seconds, parameters,
                    )  # fmt: skip
            path.rename(args.queue / "done" / path.name)
            print(f"job {path.name}: done", flush=True)
        except Exception:  # a bad job must not stop the worker
            (args.queue / "failed" / f"{path.name}.txt").write_text(traceback.format_exc())
            if path.exists():  # a job file moved away by hand while it played
                path.rename(args.queue / "failed" / path.name)
            print(f"job {path.name}: failed\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
