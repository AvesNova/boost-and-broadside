"""Round 2, step 2: skill and style of the look-ahead personas against the yardstick at 1v1,
4v4 and 16v16, all with fields and fog (the game's defaults), each size at its study clock
(step 1, `CLOCKS`).

    skill2.py STORE play --size 16 --target-error 0.02 --minutes 12
    skill2.py STORE report OUT.json

Agents: round 1's (skill.agents): the yardstick, each sampled row alone at its spread-1 box
edges, the 64 population personas (spread 1, the same unit points). Every agent plays only the
yardstick, one agent per team (all its ships alike), sides balanced, games added adaptively
(lean.eval Evaluator.adaptive) until every pairing's score standard error is within the
target. Elo: lean.eval's Bradley-Terry fit, the yardstick at 1000. Every runner call's games
go into the SQLite store at once (resumable); `--minutes` bounds an invocation.
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import skill  # noqa: E402

from lean import game  # noqa: E402
from lean.eval import Evaluator, Store  # noqa: E402
from lean.scripted.parameters import SAMPLED  # noqa: E402

# The study clock per team size (ticks at 30 Hz; step 1, progress-2.md).
CLOCKS = {1: 12000, 2: 2700, 4: 2700, 8: 2400, 16: 2400}  # 2v2, 8v8: provisional
SIZES = (1, 4, 16)


def config_of(size: int):
    return dataclasses.replace(game.GameConfig(), clock_ticks=CLOCKS[size])


def play(store_path, size, target, minutes, round_games, envs):
    started = time.perf_counter()
    store = Store(store_path)
    evaluator = Evaluator(store, config_of(size), envs=envs, minimum_envs=envs, seed=1128 + size)
    listed = skill.agents()
    yard = listed[0][2]
    sizes = (size, size)
    pairings = [(yard, agent, sizes) for _, _, agent in listed[1:]]
    while True:
        rows = evaluator.report(pairings)
        open_count = sum(row["error"] > target for row in rows)
        games = sum(row["games"] for row in rows)
        print(
            f"{time.strftime('%H:%M:%S')} {size}v{size}: {games} games, {open_count}/{len(rows)} "
            f"pairings above {target}, worst {max(row['error'] for row in rows):.4f}, "
            f"{time.perf_counter() - started:.0f} s",
            file=sys.stderr,
            flush=True,
        )
        if open_count == 0:
            print(f"{size}v{size}: done", file=sys.stderr, flush=True)
            return
        if (time.perf_counter() - started) / 60.0 > minutes:
            print(f"{size}v{size}: time bound", file=sys.stderr, flush=True)
            return
        evaluator.adaptive(
            pairings, target_error=target, budget=round_games, round_games=round_games
        )


def report(store_path, out):
    store = Store(store_path)
    listed = skill.agents()
    result = {"agents": [], "setups": {}, "clocks": CLOCKS}
    for label, unit, agent in listed:
        result["agents"].append(
            {
                "label": label,
                "id": agent.id,
                "unit": None if unit is None else unit.tolist(),
                "values": {name: agent.config().values[name] for name in SAMPLED},
            }
        )
    for size in SIZES:
        evaluator = Evaluator(store, config_of(size))
        sizes = (size, size)
        yard = listed[0][2]
        pairings = [(yard, agent, sizes) for _, _, agent in listed[1:]]
        rows = evaluator.report(pairings)
        elo = {
            identity: (rating, error)
            for identity, rating, error in evaluator.elo(
                sizes, agents=[agent for _, _, agent in listed]
            )
        }
        conditions, mirrored = evaluator.conditions(sizes)
        styles = {
            row["agent"]: row
            for row in store.styles(conditions, [a.id for _, _, a in listed], mirrored)
        }
        lengths = dict(
            store.connection.execute(
                "SELECT agent_b, AVG(length) FROM games WHERE conditions = ? GROUP BY agent_b",
                (conditions,),
            ).fetchall()
        )
        at_clock = dict(
            store.connection.execute(
                "SELECT agent_b, AVG(length >= ?) FROM games WHERE conditions = ? "
                "GROUP BY agent_b",
                (CLOCKS[size], conditions),
            ).fetchall()
        )
        entries = {}
        for (label, _, agent), row in zip(listed, [None] + rows, strict=True):
            entry = {
                "elo": elo.get(agent.id, (None, None))[0],
                "elo_error": elo.get(agent.id, (None, None))[1],
                "style": styles.get(agent.id),
                "mean_length": lengths.get(agent.id),
                "at_clock": at_clock.get(agent.id),
            }
            if row is not None:
                entry.update(
                    {
                        key: row[key]
                        for key in ("games", "wins", "draws", "losses", "score", "error")
                    }
                )
            entries[label] = entry
        result["setups"][f"{size}v{size}"] = entries
    Path(out).write_text(json.dumps(result, indent=1))
    print(f"wrote {out}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("store")
    parser.add_argument("command", choices=("play", "report"))
    parser.add_argument("out", nargs="?")
    parser.add_argument("--size", type=int, choices=SIZES, default=1)
    parser.add_argument("--target-error", type=float, default=0.015)
    parser.add_argument("--minutes", type=float, default=12.0)
    parser.add_argument("--round-games", type=int, default=16384)
    parser.add_argument("--envs", type=int, default=4096)
    arguments = parser.parse_args()
    if arguments.command == "play":
        play(
            arguments.store,
            arguments.size,
            arguments.target_error,
            arguments.minutes,
            arguments.round_games,
            arguments.envs,
        )
    else:
        report(arguments.store, arguments.out)


if __name__ == "__main__":
    main()
