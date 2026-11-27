"""Round 2 (revised brief): the validation ladder and the 1v1 round robin (CPU only).

    ladder_report.py OUT.json > tables-2-ladder.md

Per store and size: Bradley-Terry Elo (lean.eval.rating, the old yardstick at 1000) with its
SE, games, each agent's style (lean.eval.style), and the cycles: triples A > B > C > A with each
pairwise score at least 2 standard errors from 1/2. The ladder (adaptive, every pair to score
SE 0.05) against the round robin (1,024 games per pair, 1v1): rank agreement (Kendall tau),
each common agent's rating difference in units of the combined SE (relative to the yardstick in
both), and games per pair.
"""

import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyse2  # noqa: E402
import skill2  # noqa: E402

from lean.eval import Evaluator, ScriptedAgent, Store  # noqa: E402
from lean.eval.rating import bradley_terry, standard_error  # noqa: E402
from lean.scripted import ScriptedConfig  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = Path("/home/vizia/avesnova/boost-and-broadside/data/lookahead-personas-2/ladder")
LADDER = (
    "robust-centre surface-peak cma-top1 cma-top2 cma-mean-g16 yardstick crowd-known "
    "r1-persona14 r1-persona27 r1-persona00 league1 league2 league3 league4 league5 league6"
).split()
ROBIN = {
    "g16": "cma-mean-g16",
    "crowd-known": "crowd-known",
    "yardstick": "yardstick",
    "persona14": "r1-persona14",
    "persona27": "r1-persona27",
    "cma-top1": "cma-top1",
}


def agent(name):
    rows = json.loads((HERE / "candidates" / f"{name}.json").read_text())
    return ScriptedAgent.of(ScriptedConfig().replace(**rows))


def rate(store_path, names, size):
    store = Store(store_path)
    evaluator = Evaluator(store, skill2.config_of(size))
    agents = {n: agent(f) for n, f in names.items()}
    index = {a.id: n for n, a in agents.items()}
    totals = store.pairing_totals(*evaluator.conditions((size, size)))
    totals = [t for t in totals if t[0] in index and t[1] in index]
    if not totals:
        return None
    order = list(names)
    number = {n: i for i, n in enumerate(order)}
    results = [(number[index[a]], number[index[b]], g, s) for a, b, g, s in totals]
    ratings, errors = bradley_terry(results, len(order), anchor=number["yardstick"])
    scores = {}
    for a, b, g, s in totals:
        scores[(index[a], index[b])] = (g, s)
        scores[(index[b], index[a])] = (g, g - s)
    conditions, mirrored = evaluator.conditions((size, size))
    styles = {
        index[row["agent"]]: row
        for row in store.styles(conditions, list(index), mirrored)
        if row["agent"] in index
    }
    cycles = []
    for a, b, c in itertools.permutations(order, 3):
        if a != min(a, b, c):
            continue

        def beats(x, y):
            g, s = scores.get((x, y), (0, 0))
            return g > 0 and s / g - 0.5 > 2 * standard_error(s, g)

        if beats(a, b) and beats(b, c) and beats(c, a):
            cycles.append([a, b, c])
    games = sum(t[2] for t in totals)
    return {
        "elo": dict(zip(order, ratings, strict=True)),
        "error": dict(zip(order, errors, strict=True)),
        "games": games,
        "pairs": len(totals),
        "styles": styles,
        "cycles": cycles,
        "scores": {f"{a}|{b}": v for (a, b), v in scores.items()},
    }


def kendall(a, b):
    pairs = list(itertools.combinations(range(len(a)), 2))
    concordant = sum(np.sign(a[i] - a[j]) == np.sign(b[i] - b[j]) for i, j in pairs)
    return (2 * concordant - len(pairs)) / len(pairs)


def main():
    out = {}
    ladder_names = {n: n for n in LADDER}
    for size in (1, 4, 16):
        r = rate(DATA / "ladder.sqlite", ladder_names, size)
        if r is None:
            continue
        out[f"ladder {size}v{size}"] = r
        print(f"## Ladder {size}v{size}: {r['games']} games over {r['pairs']} pairs\n")
        order = sorted(LADDER, key=lambda n: -r["elo"][n])
        rows = []
        for n in order:
            s = r["styles"].get(n) or {}
            rows.append(
                [n, f"{r['elo'][n]:.0f} +- {r['error'][n]:.0f}"]
                + [
                    f"{s.get(k, math.nan):.2f}"
                    for k in ("store", "speed", "climb", "dive", "turning", "shots")
                ]
                + [f"{s['hits'] / max(s['minutes'], 1e-9):.1f}" if s else "-"]
            )
        print(
            analyse2.table(
                [
                    "agent",
                    "Elo (yardstick 1000)",
                    "store",
                    "speed",
                    "climb",
                    "dive",
                    "turning",
                    "shots/min",
                    "hits/min",
                ],
                rows,
            )
        )
        print(f"\nCycles (every edge 2 SE): {r['cycles'] or 'none'}\n")
    robin = rate(DATA / "robin.sqlite", ROBIN, 1)
    if robin is not None:
        out["robin 1v1"] = robin
        print(f"## Round robin 1v1: {robin['games']} games over {robin['pairs']} pairs\n")
        ladder = out.get("ladder 1v1")
        rows = []
        common = list(ROBIN)
        for n in sorted(common, key=lambda n: -robin["elo"][n]):
            cells = [n, f"{robin['elo'][n]:.0f} +- {robin['error'][n]:.0f}"]
            if ladder:
                lname = ROBIN[n]
                d = ladder["elo"][lname] - robin["elo"][n]
                se = math.hypot(ladder["error"][lname], robin["error"][n])
                cells += [
                    f"{ladder['elo'][lname]:.0f} +- {ladder['error'][lname]:.0f}",
                    f"{d / se:+.1f}" if se else "-",
                ]
            rows.append(cells)
        print(analyse2.table(["agent", "round robin Elo", "ladder Elo", "difference / SE"], rows))
        print(f"\nCycles in the round robin (every edge 2 SE): {robin['cycles'] or 'none'}\n")
        if ladder:
            a = np.array([robin["elo"][n] for n in common])
            b = np.array([ladder["elo"][ROBIN[n]] for n in common])
            tau = kendall(a, b)
            out["agreement"] = {"kendall_tau": tau}
            print(f"Kendall tau (ladder against round robin, 6 agents): {tau:+.2f}.\n")
    (HERE / "ladder-2.json").write_text(json.dumps(out, indent=1, default=float))


if __name__ == "__main__":
    main()
