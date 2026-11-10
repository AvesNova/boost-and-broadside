"""crossover_adaptive.py: the crossover search the author asked for at 20:50 KST (deathmatch only;
the champion and the nearest model against the scripted agent), adaptive and in one process:

- The network keeps n ships; the scripted side has b (any b >= 1). The first guess is the previous
  fleet size's ratio (b = n x ratio), so the engine is built once, at a game size just above the
  largest b the search can need (n x ratio x 1.4), and compiled once.
- A batch is 8 blocks, each its own b at no extra cost (a per-game bench count). Batch 1 spreads
  the 8 levels over +-25% of the guess; a logistic fit of the score (a draw half a win) against b,
  with half a win and half a loss of prior per level so all-win or all-loss levels stay finite,
  gives the crossing b* (score 1/2) and its standard error (delta method); each later batch packs
  its 8 levels within +-2 standard errors of b* (at least +-1 ship), and the fit is redone on
  every game so far. `--batches` batches in all.
- If b* runs off the built size, the case says so (rebuild larger).

Usage (the repo's venv): crossover_adaptive.py AGENT_PATH PRESET N RATIO OUT.json [--batches K]
[--block-environments E]"""

import argparse
import dataclasses
import json
import math
import pathlib
import time

import torch

from bnb.cli import _players
from bnb.game.api import PRESETS
from bnb.league.crossover import summary
from bnb.league.matches import Arena, ArenaConfig, network_groups
from bnb.league.tournament import identity

BLOCKS = 8


def fit(levels: dict[int, list[float]]) -> tuple[float, float, float, float]:
    """Logistic fit of score against b: logit p = a + s b, from {b: [score sum, games]} with half a
    win and half a loss of prior per level. (crossing b*, its standard error, a, s)."""
    xs = torch.tensor([float(b) for b in levels], dtype=torch.float64)
    wins = torch.tensor([v[0] + 0.5 for v in levels.values()], dtype=torch.float64)
    games = torch.tensor([v[1] + 1.0 for v in levels.values()], dtype=torch.float64)
    centre = float(xs.mean())
    x = xs - centre
    a, s = torch.tensor(0.0, dtype=torch.float64), torch.tensor(-0.5, dtype=torch.float64)
    for _ in range(100):
        p = torch.sigmoid(a + s * x)
        w = games * p * (1 - p)
        grad = torch.stack([(wins - games * p).sum(), ((wins - games * p) * x).sum()])
        info = torch.stack([torch.stack([w.sum(), (w * x).sum()]),
                            torch.stack([(w * x).sum(), (w * x * x).sum()])])
        step = torch.linalg.solve(info, grad)
        a, s = a + step[0], s + step[1]
        if step.abs().max() < 1e-10:
            break
    p = torch.sigmoid(a + s * x)
    w = games * p * (1 - p)
    info = torch.stack([torch.stack([w.sum(), (w * x).sum()]),
                        torch.stack([(w * x).sum(), (w * x * x).sum()])])
    cov = torch.linalg.inv(info)
    crossing = centre - float(a / s)
    gradient = torch.stack([-1 / s, a / s**2])  # d(-a/s)/d(a, s)
    se = float(torch.sqrt(gradient @ cov @ gradient))
    return crossing, se, float(a - s * centre), float(s)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("agent")
    parser.add_argument("preset")
    parser.add_argument("n", type=int)
    parser.add_argument("ratio", type=float)
    parser.add_argument("out")
    parser.add_argument("--batches", type=int, default=3)
    parser.add_argument("--block-environments", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    started = time.time()
    n, guess = args.n, args.n * args.ratio
    size = max(n + 2, math.ceil(guess * 1.4))
    game = dataclasses.replace(PRESETS[args.preset], ships_per_team=size)
    first, second = _players([args.agent, "scripted"], pathlib.Path("runs"))
    players = [first, second]
    groups = network_groups(players)
    layout = ArenaConfig(blocks=BLOCKS, block_environments=args.block_environments)
    lanes = tuple(BLOCKS * groups.count(g)
                  for g in range(1 + max((g for g in groups if g is not None), default=-1)))
    torch.manual_seed(args.seed)
    arena = Arena(players, game, torch.device("cuda"), layout, lanes, benching=True)
    built = time.time()
    levels: dict[int, list[float]] = {}
    counts: dict[int, torch.Tensor] = {}
    batches = []
    centre, spread = guess, 0.25 * guess
    for number in range(args.batches):
        span = torch.linspace(centre - spread, centre + spread, BLOCKS)
        bs = [min(size, max(1, round(float(b)))) for b in span]
        results = arena.play([(0, 1)] * BLOCKS, [(n, b) for b in bs])
        for b, result in zip(bs, results, strict=True):
            counts[b] = counts.get(b, torch.zeros(3, dtype=torch.int64)) + result
            won, tied, lost = (int(v) for v in result)
            entry = levels.setdefault(b, [0.0, 0.0])
            entry[0] += won + tied / 2
            entry[1] += won + tied + lost
        crossing, se, a, s = fit(levels)
        batches.append(dict(levels=bs, crossing=crossing, se=se, seconds=time.time() - started))
        print(f"batch {number + 1}: levels {bs} -> crossing {crossing:.2f} +- {se:.2f}", flush=True)
        centre = min(max(crossing, 1.0), float(size))
        spread = max(2 * se, 1.0)
    rows = [dict(scripted=b, **summary(counts[b])) for b in sorted(counts)]
    result = dict(
        agent=args.agent, identity=identity(first), preset=args.preset, own=n,
        prior_ratio=args.ratio, built_size=size, block_environments=args.block_environments,
        seed=args.seed, rows=rows, batches=batches,
        crossing=dict(enemy=crossing, se=se, low=crossing - 1.96 * se, high=crossing + 1.96 * se,
                      ratio=crossing / n, ratio_low=(crossing - 1.96 * se) / n,
                      ratio_high=(crossing + 1.96 * se) / n, slope=s,
                      off_the_built_size=not 1 <= crossing <= size),
        seconds=dict(build=built - started, total=time.time() - started),
    )
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(result, indent=1))
    print(f"{n} network ships break even against {crossing:.2f} scripted "
          f"(x{crossing / n:.2f}, 95% {result['crossing']['ratio_low']:.2f}-"
          f"{result['crossing']['ratio_high']:.2f}); {time.time() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
