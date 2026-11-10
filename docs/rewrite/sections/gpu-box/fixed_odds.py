"""fixed_odds.py: win rates at fixed odds (the author, 21:00 KST, for a paper): the network with n
ships against the scripted agent with n x 1, 1.25, 1.5 and 2 ships (rounded), deathmatch in the
training arena. One engine built at 2n (one compile); each batch's 8 blocks are 2 per ratio;
`--batches` batches (128 games per ratio at the default layouts). Per ratio: wins, draws, losses,
the score (a draw half a win) and its 95% Wilson interval, and the win rate alone with its own.

Usage (the repo's venv): fixed_odds.py AGENT_PATH PRESET N OUT.json [--batches K]
[--block-environments E] [--seed S]"""

import argparse
import dataclasses
import json
import pathlib
import time

import torch

from bnb.cli import _players
from bnb.game.api import PRESETS
from bnb.league.crossover import summary, wilson_interval
from bnb.league.matches import Arena, ArenaConfig, network_groups
from bnb.league.tournament import identity

RATIOS = (1.0, 1.25, 1.5, 2.0)
BLOCKS = 8


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("agent")
    parser.add_argument("preset")
    parser.add_argument("n", type=int)
    parser.add_argument("out")
    parser.add_argument("--batches", type=int, default=1)
    parser.add_argument("--block-environments", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    started = time.time()
    n = args.n
    enemies = [round(n * r) for r in RATIOS]
    size = max(enemies)
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
    per_block = [b for b in enemies for _ in range(BLOCKS // len(RATIOS))]
    counts = {b: torch.zeros(3, dtype=torch.int64) for b in enemies}
    for number in range(args.batches):
        results = arena.play([(0, 1)] * BLOCKS, [(n, b) for b in per_block])
        for b, result in zip(per_block, results, strict=True):
            counts[b] += result
        print(f"batch {number + 1}/{args.batches}: {time.time() - started:.0f} s", flush=True)
    rows = []
    for ratio, b in zip(RATIOS, enemies, strict=True):
        row = dict(ratio=ratio, scripted=b, **summary(counts[b]))
        row["win_rate"] = row["wins"] / row["games"]
        row["win_low"], row["win_high"] = wilson_interval(row["win_rate"], row["games"])
        rows.append(row)
    result = dict(agent=args.agent, identity=identity(first), preset=args.preset, own=n,
                  built_size=size, block_environments=args.block_environments,
                  batches=args.batches, seed=args.seed, rows=rows,
                  seconds=dict(build=built - started, total=time.time() - started))
    pathlib.Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(args.out).write_text(json.dumps(result, indent=1))
    print("  ".join(f"{r['scripted']}/{n}: win {r['win_rate']:.2f} score {r['score']:.2f} "
                    f"[{r['score_low']:.2f},{r['score_high']:.2f}]" for r in rows)
          + f"  ({time.time() - started:.0f} s, build {built - started:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
