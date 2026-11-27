"""Round 2 (revised brief): games placed where the skill surface is most uncertain near its
top (surface.py), or a ladder of named configs.

    active.py pick TOP.json SURFACE.json ROUND OUT_DIR [--count 12]   # CPU: write the points
    active.py play OUT_DIR --games 128 --minutes 10                   # GPU: play them
    active.py ladder NAME=JSON ... --target-error 0.04 --minutes 10   # every pair, adaptive
    active.py robin NAME=JSON ... --games 1024 --size 1               # every pair, fixed N

pick: from the top region's samples (within the band of the predicted peak), the `count`
points with the largest predicted SD, greedily spread (each next point at least 0.15 box units
from those chosen). play: each point against the two references (crowd-known and the CMA
generation-16 mean, both in many games of the pool) at 1v1, 4v4 and 16v16, `games` games per
pairing, into DATA/active/active.sqlite (resumable). ladder / robin: DATA/ladder/ladder.sqlite.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import skill2  # noqa: E402
import surface  # noqa: E402

from lean.eval import Evaluator, ScriptedAgent, Store  # noqa: E402
from lean.scripted import ScriptedConfig  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = surface.DATA
REFERENCES = ("crowd-known.json", "cma-mean-g16.json")


def config_of(rows: dict) -> ScriptedConfig:
    return ScriptedConfig().replace(**rows)


def pick(top_path, surface_path, round_number, out_dir, count):
    top = json.loads(Path(top_path).read_text())
    space = surface.Space2(surface.Surface(surface_path))
    samples = np.array(top["region_samples"])
    _, sd, _ = space.g(samples)
    order = np.argsort(-sd)
    chosen = []
    for index in order:
        if all(np.linalg.norm(samples[index] - samples[j]) > 0.15 for j in chosen):
            chosen.append(index)
        if len(chosen) == count:
            break
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for k, index in enumerate(chosen):
        name = f"active{round_number}-{k:02d}"
        (out / f"{name}.json").write_text(json.dumps(space.config(samples[index]), indent=1))
        print(f"{name}: predicted sd {sd[index]:.0f}", file=sys.stderr)


def play(out_dir, games, minutes, envs):
    started = time.perf_counter()
    store = Store(DATA / "active" / "active.sqlite")
    references = [
        ScriptedAgent.of(config_of(json.loads((HERE / "candidates" / r).read_text())))
        for r in REFERENCES
    ]
    points = [
        ScriptedAgent.of(config_of(json.loads(p.read_text())))
        for p in sorted(Path(out_dir).glob("active*.json"))
    ]
    for size in skill2.SIZES:
        if (time.perf_counter() - started) / 60.0 > minutes:
            print("time bound", file=sys.stderr, flush=True)
            sys.exit(3)
        evaluator = Evaluator(
            store, skill2.config_of(size), envs=envs, minimum_envs=envs, seed=5000 + size
        )
        evaluator.ensure([(r, p, (size, size), games) for r in references for p in points])
        print(f"{time.strftime('%H:%M:%S')} active {size}v{size} done", file=sys.stderr, flush=True)
    print("active done", file=sys.stderr, flush=True)


def named(items):
    return {
        item.split("=", 1)[0]: ScriptedAgent.of(
            config_of(json.loads(Path(item.split("=", 1)[1]).read_text()))
        )
        for item in items
    }


def ladder(items, sizes, target, minutes, envs):
    started = time.perf_counter()
    store = Store(DATA / "ladder" / "ladder.sqlite")
    agents = named(items)
    names = list(agents)
    for size in sizes:
        evaluator = Evaluator(
            store, skill2.config_of(size), envs=envs, minimum_envs=envs, seed=6000 + size
        )
        pairings = [
            (agents[a], agents[b], (size, size))
            for i, a in enumerate(names)
            for b in names[i + 1 :]
        ]
        while True:
            rows = evaluator.adaptive(pairings, target_error=target, budget=8192, round_games=8192)
            worst = max(row["error"] for row in rows)
            print(
                f"{time.strftime('%H:%M:%S')} ladder {size}v{size}: worst {worst:.4f}, "
                f"{sum(r['games'] for r in rows)} games",
                file=sys.stderr,
                flush=True,
            )
            if worst <= target:
                break
            if (time.perf_counter() - started) / 60.0 > minutes:
                print("time bound", file=sys.stderr, flush=True)
                sys.exit(3)
    print("ladder done", file=sys.stderr, flush=True)


def robin(items, size, games, envs):
    store = Store(DATA / "ladder" / "robin.sqlite")
    agents = named(items)
    names = list(agents)
    evaluator = Evaluator(store, skill2.config_of(size), envs=envs, minimum_envs=envs, seed=7000)
    evaluator.ensure(
        [
            (agents[a], agents[b], (size, size), games)
            for i, a in enumerate(names)
            for b in names[i + 1 :]
        ]
    )
    print("robin done", file=sys.stderr, flush=True)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("command", choices=("pick", "play", "ladder", "robin"))
    parser.add_argument("items", nargs="*")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--minutes", type=float, default=10.0)
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--sizes", type=int, nargs="+", default=[1, 4, 16])
    parser.add_argument("--size", type=int, default=1)
    parser.add_argument("--target-error", type=float, default=0.04)
    a = parser.parse_args()
    if a.command == "pick":
        pick(a.items[0], a.items[1], a.items[2], a.items[3], a.count)
    elif a.command == "play":
        play(a.items[0], a.games, a.minutes, a.envs)
    elif a.command == "ladder":
        ladder(a.items, a.sizes, a.target_error, a.minutes, a.envs)
    else:
        robin(a.items, a.size, a.games, a.envs)


if __name__ == "__main__":
    main()
