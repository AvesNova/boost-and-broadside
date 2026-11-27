"""Round 2, step 1: game durations and end reasons per team size (fields + fog, the game's
defaults otherwise) under a long cap, so the wipe time of nearly every game is seen.

    clock.py STORE play --size 4 --cap 9000 --games 512 --minutes 12
    clock.py STORE report OUT.json

Pairings: the yardstick mirror and the yardstick against six round-1 personas (spread 1;
the best and worst of round 1 at 1v1 and 4v4, and two near the median). A game whose length
is under the cap ended by a wipe (one team, or both: a draw); at the cap it would be decided by
the clock rule. The agent does not read the clock, so a game under a clock T is this game cut
at T: the share of games wiped before T is read off these lengths for every T. Games go into
the SQLite store per runner call (resumable); `--minutes` bounds one invocation.
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import skill  # noqa: E402  (round 1's agents: the same 64 personas)

from lean import game  # noqa: E402
from lean.eval import Evaluator, Store  # noqa: E402

PERSONAS = ("persona14", "persona21", "persona27", "persona00", "persona53", "persona44")
TICKS_PER_SECOND = 30
CAPS = {1: 18000, 2: 9000, 4: 9000, 8: 9000, 16: 9000}


def chosen_agents():
    listed = {label: agent for label, _, agent in skill.agents()}
    return listed["yardstick"], [(label, listed[label]) for label in PERSONAS]


def config_of(cap: int):
    return dataclasses.replace(game.GameConfig(), clock_ticks=cap)


def play(store_path, size, cap, games, minutes, step, envs):
    started = time.perf_counter()
    store = Store(store_path)
    evaluator = Evaluator(store, config_of(cap), envs=envs, minimum_envs=envs, seed=2711 + size)
    yard, personas = chosen_agents()
    pairings = [(yard, yard)] + [(yard, agent) for _, agent in personas]
    sizes = (size, size)
    target = 0
    while True:
        have = [evaluator.counts(a, b, sizes)["games"] for a, b in pairings]
        print(
            f"{time.strftime('%H:%M:%S')} {size}v{size} cap {cap}: games {have} "
            f"(target {games}), {time.perf_counter() - started:.0f} s",
            file=sys.stderr,
            flush=True,
        )
        if min(have) >= games:
            print(f"{size}v{size}: done", file=sys.stderr, flush=True)
            return
        if (time.perf_counter() - started) / 60.0 > minutes:
            print(f"{size}v{size}: time bound", file=sys.stderr, flush=True)
            return
        target = min(games, min(have) + step)
        evaluator.ensure([(a, b, sizes, target) for a, b in pairings])


def report(store_path, out):
    store = Store(store_path)
    yard, personas = chosen_agents()
    labels = {yard.id: "yardstick"} | {agent.id: label for label, agent in personas}
    result = {}
    for size, cap in CAPS.items():
        evaluator = Evaluator(store, config_of(cap))
        conditions, _ = evaluator.conditions((size, size))
        rows = store.connection.execute(
            "SELECT agent_a, agent_b, result, length FROM games WHERE conditions = ?",
            (conditions,),
        ).fetchall()
        if not rows:
            continue
        lengths = np.array([row[3] for row in rows])
        wiped = lengths < cap
        entry = {"cap": cap, "games": len(rows), "groups": {}}

        def summary(selected):
            values = lengths[selected]
            wipe = values < cap
            quantiles = {
                f"p{q}": float(np.percentile(values, q)) for q in (10, 25, 50, 75, 80, 90, 95, 99)
            }
            ended = {
                str(seconds): float(np.mean(values < seconds * TICKS_PER_SECOND))
                for seconds in (30, 60, 90, 120, 150, 180, 240, 300, 450, 600)
            }
            return {
                "games": int(selected.sum()),
                "at_cap": float(np.mean(~wipe)),
                "draw_wipes": float(
                    np.mean([row[2] == 2 for row, s in zip(rows, selected, strict=True) if s])
                ),
                "mean": float(values.mean()),
                "quantiles": quantiles,
                "wiped_before_seconds": ended,
            }

        everything = np.ones(len(rows), dtype=bool)
        entry["all"] = summary(everything)
        mirror = np.array([row[0] == row[1] for row in rows])
        entry["groups"]["mirror"] = summary(mirror)
        for label, agent in personas:
            selected = np.array([agent.id in (row[0], row[1]) for row in rows])
            if selected.any():
                entry["groups"][label] = summary(selected)
        entry["wiped_share"] = float(wiped.mean())
        result[f"{size}v{size}"] = entry
    Path(out).write_text(json.dumps(result, indent=1))
    for key, entry in result.items():
        quantiles = entry["all"]["quantiles"]
        print(
            f"{key}: {entry['games']} games, at cap {entry['all']['at_cap']:.3f}, "
            f"p50 {quantiles['p50'] / 30:.0f} s p80 {quantiles['p80'] / 30:.0f} s "
            f"p90 {quantiles['p90'] / 30:.0f} s p95 {quantiles['p95'] / 30:.0f} s; "
            f"wiped before 60 s {entry['all']['wiped_before_seconds']['60']:.3f}",
            file=sys.stderr,
        )
        for label, group in entry["groups"].items():
            print(
                f"   {label}: p50 {group['quantiles']['p50'] / 30:.0f} s "
                f"p90 {group['quantiles']['p90'] / 30:.0f} s at cap {group['at_cap']:.3f}",
                file=sys.stderr,
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("store")
    parser.add_argument("command", choices=("play", "report"))
    parser.add_argument("out", nargs="?")
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--games", type=int, default=512)
    parser.add_argument("--step", type=int, default=256)
    parser.add_argument("--minutes", type=float, default=12.0)
    parser.add_argument("--envs", type=int, default=2048)
    arguments = parser.parse_args()
    if arguments.command == "play":
        play(
            arguments.store,
            arguments.size,
            CAPS[arguments.size],
            arguments.games,
            arguments.minutes,
            arguments.step,
            arguments.envs,
        )
    else:
        report(arguments.store, arguments.out)


if __name__ == "__main__":
    main()
