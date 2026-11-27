"""Round 2, redirected study: skill and style of personas around crowd-known (the screen's
configuration, candidates/crowd-known.json) at 1v1, 4v4, 16v16 (fields, fog, study clocks).

    around.py STORE play --size 16 --target-error 0.03 --minutes 10
    around.py STORE report OUT.json

Agents: the centre; each sampled row alone at u = -0.5 and u = 1.5 of its spread-1 box around
the centre (half a box past either edge: the working range); 48 personas, uniform on the
spread-1 box around the centre (seed 20261128). The centre's crowd rows stay fixed. Every
agent plays the centre only, sides balanced, adaptively to the target score SE; Elo is the
Bradley-Terry fit with the centre at 1000. Resumable (the SQLite store); `--minutes` bounds.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import skill2  # noqa: E402
import tune  # noqa: E402

from lean.eval import Evaluator, ScriptedAgent, Store  # noqa: E402
from lean.scripted import ScriptedConfig  # noqa: E402
from lean.scripted.parameters import SAMPLED  # noqa: E402

HERE = Path(__file__).resolve().parent
CENTRE = HERE / "candidates" / "crowd-known.json"
PERSONAS = 48
EDGES = (-0.5, 1.5)


def centre() -> ScriptedConfig:
    return ScriptedConfig().replace(**json.loads(CENTRE.read_text()))


def at_unit(unit) -> ScriptedConfig:
    base = centre()
    values = {
        name: tune.row_value(name, float(u), base.values[name])
        for name, u in zip(SAMPLED, unit, strict=True)
    }
    return base.replace(**values)


def agents():
    """(label, unit point or None, agent), the centre first."""
    out = [("centre", None, ScriptedAgent.of(centre()))]
    for column, name in enumerate(SAMPLED):
        for level in EDGES:
            unit = np.full(len(SAMPLED), 0.5)
            unit[column] = level
            out.append((f"{name}@{level:+.1f}", unit, ScriptedAgent.of(at_unit(unit))))
    units = np.random.default_rng(20261128).random((PERSONAS, len(SAMPLED)))
    for index, unit in enumerate(units):
        out.append((f"persona{index:02d}", unit, ScriptedAgent.of(at_unit(unit))))
    return out


def play(store_path, size, target, minutes, round_games, envs):
    started = time.perf_counter()
    store = Store(store_path)
    evaluator = Evaluator(
        store, skill2.config_of(size), envs=envs, minimum_envs=envs, seed=1300 + size
    )
    listed = agents()
    middle = listed[0][2]
    pairings = [(middle, agent, (size, size)) for _, _, agent in listed[1:]]
    while True:
        rows = evaluator.report(pairings)
        open_count = sum(row["error"] > target for row in rows)
        print(
            f"{time.strftime('%H:%M:%S')} {size}v{size}: {sum(r['games'] for r in rows)} games, "
            f"{open_count}/{len(rows)} above {target}, {time.perf_counter() - started:.0f} s",
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
    listed = agents()
    result = {"agents": [], "setups": {}}
    for label, unit, agent in listed:
        result["agents"].append(
            {
                "label": label,
                "unit": None if unit is None else unit.tolist(),
                "values": {name: agent.config().values[name] for name in SAMPLED},
            }
        )
    for size in skill2.SIZES:
        evaluator = Evaluator(store, skill2.config_of(size))
        sizes = (size, size)
        middle = listed[0][2]
        rows = evaluator.report([(middle, agent, sizes) for _, _, agent in listed[1:]])
        elo = {i: (r, e) for i, r, e in evaluator.elo(sizes, anchor=middle)}
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
                entry.update({k: row[k] for k in ("games", "wins", "draws", "losses", "error")})
            entries[label] = entry
        result["setups"][f"{size}v{size}"] = entries
    Path(out).write_text(json.dumps(result, indent=1))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("store")
    parser.add_argument("command", choices=("play", "report"))
    parser.add_argument("out", nargs="?")
    parser.add_argument("--size", type=int, choices=skill2.SIZES, default=1)
    parser.add_argument("--target-error", type=float, default=0.03)
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument("--round-games", type=int, default=8192)
    parser.add_argument("--envs", type=int, default=4096)
    a = parser.parse_args()
    if a.command == "play":
        play(a.store, a.size, a.target_error, a.minutes, a.round_games, a.envs)
    else:
        report(a.store, a.out)


if __name__ == "__main__":
    main()
