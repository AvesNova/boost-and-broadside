"""Skill and style of look-ahead personas against the yardstick (lean.eval).

    skill.py STORE play --setup 1v1 --target-error 0.02 --minutes 8
    skill.py STORE report OUT.json

Agents: the yardstick (1000 Elo), each sampled row alone at its spread-1 box edges (u = 0 and
1, the rest at the defaults: `oat`), and the 64 population personas of divergence.py (spread
1, the same unit points: `population`). Every agent plays the yardstick only, sides balanced,
games added adaptively (lean.eval Evaluator.adaptive: each round's games go to the pairings
with the largest score standard error) until every pairing is within the target error. Elo is
the Bradley-Terry fit (lean.eval.rating) of those games. Setups as divergence.py: "1v1" fog off
no fields, "4v4" fields and fog; the game's defaults otherwise (clock 1,800 ticks, standard
spawn). Every runner call's games go into the SQLite store at once (resumable); `--minutes`
bounds an invocation. Style: the store's `styles` view (lean.eval.style) per agent.
"""

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import divergence  # noqa: E402

from lean import game  # noqa: E402
from lean.eval import Evaluator, ScriptedAgent, Store  # noqa: E402
from lean.scripted import ScriptedConfig  # noqa: E402
from lean.scripted.parameters import SAMPLED  # noqa: E402


def agents() -> list[tuple[str, np.ndarray | None, ScriptedAgent]]:
    """(label, unit point or None, agent) for every agent, yardstick first."""
    out = [("yardstick", None, ScriptedAgent.of())]
    for column, name in enumerate(SAMPLED):
        for level in (0.0, 1.0):
            unit = divergence.CENTRE.copy()
            unit[column] = level
            out.append(
                (
                    f"{name}@{'low' if level == 0 else 'high'}",
                    unit,
                    ScriptedAgent.of(ScriptedConfig(divergence.unit_rows(unit))),
                )
            )
    rng = np.random.default_rng(20261127)
    units = rng.random((divergence.POPULATION, divergence.DIMENSIONS))  # = divergence.population
    for index, unit in enumerate(units):
        out.append(
            (
                f"persona{index:02d}",
                unit,
                ScriptedAgent.of(ScriptedConfig(divergence.unit_rows(unit))),
            )
        )
    return out


def setup_config(setup: str):
    return dataclasses.replace(game.GameConfig(), **divergence.SETUPS[setup]["changes"])


def sizes_of(setup: str) -> tuple[int, int]:
    size = divergence.SETUPS[setup]["size"]
    return (size, size)


def play(
    store_path: str, setup: str, target: float, minutes: float, round_games: int, envs: int
) -> None:
    started = time.perf_counter()

    def progress(games, busy, seconds):
        print(
            f"  {time.strftime('%H:%M:%S')} {games} games started, {busy} playing, {seconds:.0f} s",
            file=sys.stderr,
            flush=True,
        )

    store = Store(store_path)
    evaluator = Evaluator(
        store, setup_config(setup), envs=envs, minimum_envs=envs, seed=1127, progress=progress
    )
    listed = agents()
    yard = listed[0][2]
    sizes = sizes_of(setup)
    pairings = [(yard, agent, sizes) for _, _, agent in listed[1:]]
    while True:
        rows = evaluator.report(pairings)
        open_count = sum(row["error"] > target for row in rows)
        games = sum(row["games"] for row in rows)
        print(
            f"{time.strftime('%H:%M:%S')} {setup}: {games} games, "
            f"{open_count}/{len(rows)} pairings "
            f"above {target}, worst {max(row['error'] for row in rows):.4f}, "
            f"{time.perf_counter() - started:.0f} s",
            file=sys.stderr,
            flush=True,
        )
        if open_count == 0:
            print(f"{setup}: done", file=sys.stderr, flush=True)
            return
        if (time.perf_counter() - started) / 60.0 > minutes:
            print(f"{setup}: time bound", file=sys.stderr, flush=True)
            return
        evaluator.adaptive(
            pairings, target_error=target, budget=round_games, round_games=round_games
        )


def report(store_path: str, out: str) -> None:
    store = Store(store_path)
    listed = agents()
    result = {"agents": [], "setups": {}}
    for label, unit, agent in listed:
        result["agents"].append(
            {
                "label": label,
                "id": agent.id,
                "unit": None if unit is None else unit.tolist(),
                "values": {name: agent.config().values[name] for name in SAMPLED},
            }
        )
    for setup in divergence.SETUPS:
        evaluator = Evaluator(store, setup_config(setup))
        sizes = sizes_of(setup)
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
        entries = {}
        for (label, _, agent), row in zip(listed, [None] + rows, strict=True):
            entry = {
                "elo": elo.get(agent.id, (None, None))[0],
                "elo_error": elo.get(agent.id, (None, None))[1],
                "style": styles.get(agent.id),
            }
            if row is not None:
                entry.update(
                    {
                        key: row[key]
                        for key in ("games", "wins", "draws", "losses", "score", "error")
                    }
                )
            entries[label] = entry
        result["setups"][setup] = entries
    Path(out).write_text(json.dumps(result, indent=1))
    print(f"wrote {out}", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("store")
    parser.add_argument("command", choices=("play", "report"))
    parser.add_argument("out", nargs="?")
    parser.add_argument("--setup", choices=tuple(divergence.SETUPS), default="1v1")
    parser.add_argument("--target-error", type=float, default=0.02)
    parser.add_argument("--minutes", type=float, default=8.0)
    parser.add_argument("--round-games", type=int, default=8192)
    parser.add_argument("--envs", type=int, default=4096)
    arguments = parser.parse_args()
    if arguments.command == "play":
        play(
            arguments.store,
            arguments.setup,
            arguments.target_error,
            arguments.minutes,
            arguments.round_games,
            arguments.envs,
        )
    else:
        report(arguments.store, arguments.out)


if __name__ == "__main__":
    main()
