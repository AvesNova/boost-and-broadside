"""The laptop's share of the balance map: the conditions the search does not play, for the
search's promising cards (a pre-screen before the 8-condition confirmation), on the laptop's
GPU through laptop_job.sh (the laptop GPU lock, a memory-capped unit).

    PYTHONPATH=$PWD/src:$PWD uv run --no-sync python docs/rewrite/sections/balance-map/screen.py \
        --remote box5090:/workspace/out/map --local LOCAL [--hours 12] [--floor 40]

Each round: rsync the run's candidates.jsonl and evaluations.jsonl from the box into
LOCAL/run/, choose every card whose searched objective is at least --floor on 512 or more
games and that LOCAL/screen/ has not played, then one `bench.balance_sweep pairwise` job per
unscreened condition (deathmatch 1v1 and 10v10, elimination 10v10, Frontline 5v5), --games a
card, each card's team against the reference's. Records: LOCAL/screen/<condition>.jsonl (a
line a card, written as each chunk ends: a stopped round resumes). Bounded by --hours.
"""

import argparse
import json
import pathlib
import subprocess
import time

from bench import balance_map

OTHER = tuple(c for c in balance_map.EVERY if c not in balance_map.SEARCHED)
WORKTREE = pathlib.Path(__file__).resolve().parents[4]
LAPTOP_JOB = "/home/vizia/avesnova/laptop_job.sh"


def screened(local: pathlib.Path) -> dict[str, set[str]]:
    out = {}
    for condition in OTHER:
        path = local / "screen" / f"{condition}.jsonl"
        out[condition] = (
            {json.loads(line)["name"] for line in path.read_text().splitlines() if line.strip()}
            if path.exists()
            else set()
        )
    return out


def play(condition: str, pairs: pathlib.Path, out: pathlib.Path, games: int, seed: int) -> None:
    """One `bench.balance_sweep pairwise` job on the laptop GPU (laptop_job.sh)."""
    unit = f"bnb-map-{condition}-{int(time.time())}"
    subprocess.run(
        [
            LAPTOP_JOB,
            unit,
            str(WORKTREE),
            str(WORKTREE / ".venv/bin/python"),
            "-m",
            "bench.balance_sweep",
            "pairwise",
            "--condition",
            condition,
            "--pairs",
            str(pairs),
            "--out",
            str(out),
            "--games",
            str(games),
            "--seed",
            str(seed),
            "--environments",
            "4096",
            "--variants",
            "32",
        ],
        check=False,
    )


def confirm(args) -> None:
    """Pair entries (`balance_map.pair_entry`) played once on each of --conditions at
    --games (seed 2): LOCAL/<tag>/<condition>.jsonl."""
    folder = args.local / args.tag
    folder.mkdir(parents=True, exist_ok=True)
    for condition in args.conditions:
        print(f"{time.strftime('%H:%M:%S')} {args.tag} {condition}", flush=True)
        play(condition, args.entries, folder / f"{condition}.jsonl", args.games, 2)
    print(f"{args.tag} done", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--remote", required=True)
    parser.add_argument("--local", type=pathlib.Path, required=True)
    parser.add_argument("--hours", type=float, default=12.0)
    parser.add_argument("--floor", type=float, default=40.0)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--wait", type=float, default=300.0, help="seconds between empty rounds")
    parser.add_argument(
        "--entries",
        type=pathlib.Path,
        default=None,
        help="a confirmation instead: these pair entries, once, on --conditions",
    )
    parser.add_argument("--conditions", nargs="*", default=list(OTHER))
    parser.add_argument("--tag", default="confirm")
    args = parser.parse_args()
    if args.entries is not None:
        return confirm(args)
    deadline = time.time() + args.hours * 3600.0
    run = args.local / "run"
    run.mkdir(parents=True, exist_ok=True)
    (args.local / "screen").mkdir(parents=True, exist_ok=True)
    while time.time() < deadline:
        for name in ("candidates.jsonl", "evaluations.jsonl"):
            subprocess.run(["rsync", "-q", f"{args.remote}/{name}", str(run / name)], check=False)
        records = balance_map.Records(run)
        done = screened(args.local)
        chosen = []
        for candidate in records.candidates:
            if "card" not in candidate:
                continue
            score = records.score(candidate["id"])
            if (
                score["objective"] is not None
                and score["objective"] >= args.floor
                and (score["games"] >= 512)
            ):
                chosen.append(candidate)
        played = False
        for condition in OTHER:
            todo = [c for c in chosen if c["id"] not in done[condition]]
            if not todo:
                continue
            pairs = args.local / "screen" / f"pairs-{condition}.json"
            pairs.write_text(json.dumps([balance_map.pair_entry(c["id"], c["card"]) for c in todo]))
            print(f"{time.strftime('%H:%M:%S')} {condition}: {len(todo)} cards", flush=True)
            play(condition, pairs, args.local / "screen" / f"{condition}.jsonl", args.games, 1)
            played = True
        if not played:
            time.sleep(args.wait)
    print("screen done", flush=True)


if __name__ == "__main__":
    main()
