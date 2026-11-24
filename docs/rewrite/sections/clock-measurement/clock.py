"""Scripted wipe times with the clock effectively off (docs/rewrite/clock-measurement.md).

BFM-CMA2 (today's default) against itself and against the screening panel (panel.json, by
row name onto today's rows), standard conditions except the clock (CAP ticks). Every game
goes into one lean.eval store; a rerun plays only what is missing.

    clock.py STORE play      # the games (each condition written to the store as it ends)
    clock.py STORE report    # percentiles per condition, as JSON on stdout
"""

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripted-screening"))
from screen import panel  # noqa: E402

from lean import game  # noqa: E402
from lean.eval import Evaluator, ScriptedAgent, Store  # noqa: E402

CAP = 30000
# (name, arena radius, sizes, mirror games, games per panel member)
CONDITIONS = [
    ("4v4 4,200 m (main)", 4200.0, (4, 4), 1024, 128),
    ("3v3 4,200 m", 4200.0, (3, 3), 256, 32),
    ("5v5 4,200 m", 4200.0, (5, 5), 256, 32),
    ("4v4 3,000 m", 3000.0, (4, 4), 256, 32),
    ("4v4 6,000 m", 6000.0, (4, 4), 256, 32),
]
QUANTILES = (50, 80, 90, 95)


def config(radius: float):
    return dataclasses.replace(game.GameConfig(), arena_radius=radius, clock_ticks=CAP)


def requests(sizes, mirror, per_member):
    default = ScriptedAgent.of()
    out = [(default, default, sizes, mirror)]
    out += [(default, member, sizes, per_member) for member in panel() if member != default]
    return out


def play(store: Store) -> None:
    for number, (name, radius, sizes, mirror, per_member) in enumerate(CONDITIONS):
        started = time.perf_counter()
        Evaluator(store, config(radius), envs=4096, seed=1000 + 17 * number).ensure(
            requests(sizes, mirror, per_member)
        )
        print(f"{time.strftime('%H:%M:%S')} {name}: done in {time.perf_counter() - started:.0f} s; "
              f"store {store.total_games()}", flush=True)


def lengths(store: Store, radius, sizes, agents_b) -> np.ndarray:
    conditions = store.add_conditions(config(radius), sizes)
    default = ScriptedAgent.of().id
    marks = ",".join("?" * len(agents_b))
    rows = store.connection.execute(
        f"SELECT length FROM games WHERE conditions = ? AND agent_a = ? AND agent_b IN ({marks})",
        (conditions, default, *agents_b),
    ).fetchall()
    return np.array([row[0] for row in rows], dtype=np.int64)


def summary(values: np.ndarray, tick_length: float) -> dict:
    out = {"games": int(values.size), "at_cap": float(np.mean(values >= CAP))}
    for q in QUANTILES:
        ticks = float(np.percentile(values, q))
        out[f"p{q}"] = {"ticks": ticks, "seconds": ticks * tick_length}
    return out


def report(store: Store) -> dict:
    tick = game.GameConfig().tick_length
    default = ScriptedAgent.of()
    members = [member.id for member in panel() if member != default]
    out = {}
    for name, radius, sizes, _, _ in CONDITIONS:
        out[name] = {
            "pooled": summary(lengths(store, radius, sizes, [default.id, *members]), tick),
            "mirror": summary(lengths(store, radius, sizes, [default.id]), tick),
            "panel": summary(lengths(store, radius, sizes, members), tick),
        }
    return out


if __name__ == "__main__":
    store = Store(sys.argv[1])
    if sys.argv[2] == "play":
        play(store)
    else:
        print(json.dumps(report(store), indent=1))
