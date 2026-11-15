"""The final freeze check's measurements (game v22, Imugi on both sides, fog on, fields on,
each preset's own map, the default search: explore with the wide spacing). Writes data/.

    PYTHONPATH=src:. uv run --no-sync python docs/rewrite/sections/freeze-final/measure.py \
        [--only LABEL ...]

1. Every condition (`CONDITIONS`) runs `bench.mode_dynamics` on the GPU (compiled) into
   data/modes.jsonl, a JSON line per batch of BATCH first games; after each batch the
   condition's measures (`freeze_report.measures`) are checked against their SE targets
   (`freeze_report.done`) and the run stops there, or at BOUND games. A condition already
   done in the file is skipped, so a stopped run resumes.
2. The head-on passes before each kill and the kills by aspect (`bench.game_loop`, its
   merge definition) at 5v5 in each mode, the stored doctrines and each mode's Imugi
   parameters, mixed lineups, fields on, the preset's map: data/passes.json.

Progress lines go to stdout with a finish time (the unit's journal).
"""

import argparse
import datetime
import json
import pathlib
import subprocess
import sys
import time

import freeze_report

HERE = pathlib.Path(__file__).parent
ROOT = HERE.parents[3]
DATA = HERE / "data"
MODES_FILE = DATA / "modes.jsonl"
BATCH, BOUND = 256, 2048
SEED = 22
MIXED = "--drawn-lineups"
ENERGY_TURN = "yongma|jujak;jujak|yongma"
# (label, mode, ships per team, lineup: MIXED or a fixed one)
CONDITIONS = [
    *[(f"dm-{n}v{n}", "deathmatch", n, MIXED) for n in (1, 2, 5, 10)],
    *[(f"elim-{n}v{n}", "elimination", n, MIXED) for n in (1, 2, 5, 10)],
    *[(f"front-{n}v{n}", "frontline", n, MIXED) for n in (3, 5, 10)],
    *[(f"dm-et-{n}v{n}", "deathmatch", n, ENERGY_TURN) for n in (1, 2, 5, 10)],
    *[(f"elim-et-{n}v{n}", "elimination", n, ENERGY_TURN) for n in (1, 2, 5, 10)],
    *[(f"front-et-{n}v{n}", "frontline", n, ENERGY_TURN) for n in (3, 5, 10)],
]


def log(line: str) -> None:
    print(f"{datetime.datetime.now():%H:%M:%S} {line}", flush=True)


def at(seconds: float) -> str:
    return (datetime.datetime.now() + datetime.timedelta(seconds=seconds)).strftime("%H:%M")


def status(label: str, mode: str, lineup: str) -> tuple[bool, dict]:
    games, entries = freeze_report.load(MODES_FILE, label)
    result = freeze_report.measures(mode, games, entries)
    finished = bool(games) and (
        freeze_report.done(mode, result, lineup != MIXED) or len(games) >= BOUND
    )
    return finished, result


def run_condition(label: str, mode: str, ships: int, lineup: str) -> dict:
    """Batches of one condition until its targets are met or BOUND games."""
    finished, result = status(label, mode, lineup)
    if finished:
        log(f"{label}: already done ({result['games']} games)")
        return result
    command = [
        sys.executable, "-m", "bench.mode_dynamics", mode,
        "--games", str(BOUND), "--batch", str(BATCH), "--ships", str(ships),
        "--fields", "--seed", str(SEED), "--label", label, "--out", str(MODES_FILE),
    ]  # fmt: skip
    command += [MIXED] if lineup == MIXED else ["--lineup", lineup]
    process = subprocess.Popen(command, cwd=ROOT)
    seen = result.get("games", 0)
    while process.poll() is None:
        time.sleep(5)
        finished, result = status(label, mode, lineup)
        if result.get("games", 0) != seen:
            seen = result["games"]
            log(f"{label}: {seen} games")
            if finished:
                process.terminate()
                process.wait()
                break
    if process.returncode not in (0, None, -15):
        raise SystemExit(f"{label}: mode_dynamics failed ({process.returncode})")
    return status(label, mode, lineup)[1]


def passes() -> None:
    """Head-on passes a kill and kills by aspect (bench.game_loop) at 5v5, each mode."""
    out_path = DATA / "passes.json"
    out = json.loads(out_path.read_text()) if out_path.exists() else {}
    from bench.game_loop import Job, Loop, Player, job_summary, run
    from bnb.controllers.imugi_defaults import MODE_DEFAULTS
    from bnb.controllers.imugi_doctrine import stored
    from bnb.controllers.imugi_parameters import pack

    doctrines = stored()["real-scale"]
    for mode, seconds in (("deathmatch", 300.0), ("elimination", 248.0), ("frontline", 600.0)):
        if mode in out:
            continue
        loop = Loop(512, 1, seconds, mode=mode, fields=True, fixed_map=True)
        player = Player(
            "imugi",
            doctrines["yongma"],
            doctrines["jujak"],
            parameters=tuple(pack(MODE_DEFAULTS.sets[f"real_scale_{mode}"])),
        )
        jobs = [Job(0, lineup, player, player, 512) for lineup in ("mixed", "mixed reversed")]
        rows = [job_summary(j, o, seconds) for j, o in zip(jobs, run(loop, jobs, log), strict=True)]
        out[mode] = rows
        out_path.write_text(json.dumps(out, indent=1))
        log(
            f"passes {mode}: "
            + json.dumps([[r["lineup"], r["passes_per_kill"], r["kill_share"]] for r in rows])
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", nargs="*", default=None)
    parser.add_argument("--skip-passes", action="store_true")
    args = parser.parse_args()
    DATA.mkdir(exist_ok=True)
    chosen = [c for c in CONDITIONS if not args.only or c[0] in args.only]
    started, timed = time.time(), 0
    for number, (label, mode, ships, lineup) in enumerate(chosen):
        began = time.time()
        result = run_condition(label, mode, ships, lineup)
        took = time.time() - began
        timed += 1 if took > 30 else 0
        left = len(chosen) - number - 1
        per = (time.time() - started) / max(timed, 1)
        log(
            f"{label} done: {result['games']} games; {left} left, done about {at(per * left)} "
            f"(worst {at(2 * per * left)})"
        )
    if not args.skip_passes:
        passes()
    log("all done")


if __name__ == "__main__":
    main()
