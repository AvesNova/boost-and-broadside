"""Skill re-check of the trim (docs/rewrite/scripted-reach.md "Trim (2026-11-25)").

    trim_recheck.py STORE [--games 64] [--restore NAME]

1. Panel skill: the trimmed default (today's lean.scripted) against the screening's panel of 8
   (its members on today's rows, on today's code), the 18 conditions, `games` per cell; the
   same games and skill as recheck.py (screen.play with the label "recheck", recheck.skill).
   The untrimmed BFM-CMA2's panel skill is recheck.json's (the same protocol on core before
   the trim; lean/ unchanged between).
2. Head to head: the trimmed default against the untrimmed BFM-CMA2 (the frozen copy in
   `untrimmed/`) in the same games, `games` per condition (64 x 18 = 1,152), seats alternating.
   Each team decides with its own code and memory; the uniforms are shared.

Writes trim_recheck.json beside this file.
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "scripted-screening"))
import recheck  # noqa: E402
import screen  # noqa: E402
import untrimmed.opponent as old  # noqa: E402
from untrimmed.parameters import ScriptedConfig as OldConfig  # noqa: E402

from lean import game, scripted  # noqa: E402
from lean.eval import ScriptedAgent, Store  # noqa: E402
from lean.scripted import ScriptedConfig  # noqa: E402

OUTPUT = HERE / "trim_recheck.json"
SLOTS = 10
SEED = 20261125


def log(text: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {text}", file=sys.stderr, flush=True)


@torch.no_grad()
def head_to_head_condition(game_config, sizes, games: int, seed: int) -> tuple[float, float, int]:
    """(mean score of the trimmed agent, its unbiased game variance / n, games) at one
    condition: games of `sizes`, the trimmed agent team 0 in even games, team 1 in odd."""
    here = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=here).manual_seed(seed)
    team_sizes = torch.tensor([sizes], device=here).expand(games, 2)
    state = game.deal(game_config, games, SLOTS, team_sizes, generator, here)
    memory_new = scripted.fresh_memory(games, SLOTS, generator, here)
    memory_old = old.fresh_memory(games, SLOTS, generator, here)
    trimmed_team = torch.arange(games, device=here) % 2
    trimmed_rows = (state.team == trimmed_team[:, None])[..., None]
    new_config, old_config = ScriptedConfig(), OldConfig()
    result = torch.full((games,), -1, dtype=torch.long, device=here)
    over = torch.zeros(games, dtype=torch.bool, device=here)
    tick = 0
    while not bool(over.all()) and tick < game_config.clock_ticks + 2:
        uniforms = scripted.draw_uniforms(games, SLOTS, generator, here)
        action_new, memory_new, _ = scripted.decide(
            new_config, state, None, None, uniforms, memory_new
        )
        action_old, memory_old, _ = old.decide(old_config, state, None, None, uniforms, memory_old)
        action = torch.where(trimmed_rows, action_new, action_old)
        noise = game.draw_noise(game_config, games, SLOTS, generator, here)
        state, out = game.step(game_config, state, action, noise)
        result = torch.where(out.done & ~over, out.result, result)
        over = over | out.done
        tick += 1
    won = (result == trimmed_team).float()
    draw = (result == 2).float()
    score = won + 0.5 * draw
    n = games
    mean = float(score.mean())
    variance = max(float((won + 0.25 * draw).mean()) - mean * mean, 0.0) / (n - 1)
    return mean, variance, n


def head_to_head(games: int) -> dict:
    scores, variance, total, cells = [], 0.0, 0, []
    for number, game_config in enumerate(screen.game_configs()):
        for index, sizes in enumerate(screen.SIZES):
            mean, var, n = head_to_head_condition(
                game_config, sizes, games, SEED + 10 * number + index
            )
            scores.append(mean)
            variance += var
            total += n
            cells.append(
                {
                    "radius": game_config.arena_radius,
                    "spawn": game_config.spawn,
                    "sizes": list(sizes),
                    "score": mean,
                }
            )
            log(f"  h2h {game_config.arena_radius:.0f} {game_config.spawn} {sizes}: {mean:.3f}")
    return {
        "score": float(np.mean(scores)),
        "error": math.sqrt(variance) / len(scores),
        "games": total,
        "cells": cells,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("store")
    parser.add_argument("--games", type=int, default=64)
    arguments = parser.parse_args()
    store = Store(arguments.store)
    trimmed = ScriptedAgent.of()
    started = time.perf_counter()
    screen.play(store, [trimmed], arguments.games, "recheck")
    value, error, per_condition, played = recheck.skill(store, trimmed)
    before = json.loads((HERE.parent / "scripted-screening" / "recheck.json").read_text())
    untrimmed = before["BFM-CMA2"]
    difference = value - untrimmed["skill"]
    difference_error = math.sqrt(error**2 + untrimmed["error"] ** 2)
    log(
        f"panel skill: trimmed {value:.4f} +- {error:.4f}, untrimmed {untrimmed['skill']:.4f} "
        f"+- {untrimmed['error']:.4f}; difference {difference:+.4f} +- {difference_error:.4f}"
    )
    match = head_to_head(arguments.games)
    log(f"head to head, trimmed v untrimmed: {match['score']:.4f} +- {match['error']:.4f}")
    out = {
        "games_per_cell": arguments.games,
        "trimmed_id": trimmed.id,
        "trimmed": {
            "skill": value,
            "error": error,
            "games": played,
            "by_condition": per_condition,
            "marginals": recheck.marginals(per_condition),
        },
        "untrimmed": {"skill": untrimmed["skill"], "error": untrimmed["error"]},
        "difference": {"value": difference, "error": difference_error},
        "head_to_head": match,
        "seconds": time.perf_counter() - started,
    }
    OUTPUT.write_text(json.dumps(out, indent=1) + "\n")
    log(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
