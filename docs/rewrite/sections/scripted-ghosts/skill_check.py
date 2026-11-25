"""Panel skill of the default scripted agent on the code `lean` is imported from
(docs/rewrite/sections/scripted-ghosts/notes.md).

    skill_check.py STORE OUT [--games 64] [--head-to-head BEFORE_DIR]

The recheck protocol of scripted-reach.md "Trim": the default (ScriptedAgent.of()) against the
screening's panel of 8 (its members on this code's rows), the 18 conditions, `games` per cell
(screen.play with the label "recheck", recheck.skill), into a fresh store. With
--head-to-head, also the default against a frozen copy of the earlier agent (BEFORE_DIR: a
package with opponent.py, parameters.py, situation.py), 64 games a condition, seats
alternating, each team with its own code and memory (as trim_recheck.py). Writes OUT (json).
"""

import argparse
import importlib
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

import lean

SCREENING = Path(lean.__file__).resolve().parents[1] / "docs/rewrite/sections/scripted-screening"
sys.path.insert(0, str(SCREENING))
import recheck  # noqa: E402
import screen  # noqa: E402

from lean import game, scripted  # noqa: E402
from lean.eval import ScriptedAgent, Store  # noqa: E402
from lean.scripted import ScriptedConfig  # noqa: E402

SLOTS = 10
SEED = 20261125


def log(text: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} {text}", file=sys.stderr, flush=True)


@torch.no_grad()
def head_to_head_condition(old, game_config, sizes, games: int, seed: int):
    here = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=here).manual_seed(seed)
    team_sizes = torch.tensor([sizes], device=here).expand(games, 2)
    state = game.deal(game_config, games, SLOTS, team_sizes, generator, here)
    memory_new = scripted.fresh_memory(games, SLOTS, generator, here)
    memory_old = old.opponent.fresh_memory(games, SLOTS, generator, here)
    new_team = torch.arange(games, device=here) % 2
    new_rows = (state.team == new_team[:, None])[..., None]
    new_config, old_config = ScriptedConfig(), old.parameters.ScriptedConfig()
    result = torch.full((games,), -1, dtype=torch.long, device=here)
    over = torch.zeros(games, dtype=torch.bool, device=here)
    tick = 0
    while not bool(over.all()) and tick < game_config.clock_ticks + 2:
        uniforms = scripted.draw_uniforms(games, SLOTS, generator, here)
        action_new, memory_new, _ = scripted.decide(
            new_config, state, None, None, uniforms, memory_new
        )
        action_old, memory_old, _ = old.opponent.decide(
            old_config, state, None, None, uniforms, memory_old
        )
        action = torch.where(new_rows, action_new, action_old)
        noise = game.draw_noise(game_config, games, SLOTS, generator, here)
        state, out = game.step(game_config, state, action, noise)
        result = torch.where(out.done & ~over, out.result, result)
        over = over | out.done
        tick += 1
    won = (result == new_team).float()
    draw = (result == 2).float()
    score = won + 0.5 * draw
    mean = float(score.mean())
    variance = max(float((won + 0.25 * draw).mean()) - mean * mean, 0.0) / (games - 1)
    return mean, variance, games


def head_to_head(before: Path, games: int) -> dict:
    sys.path.insert(0, str(before.parent))
    old = type("Old", (), {})()
    old.opponent = importlib.import_module(f"{before.name}.opponent")
    old.parameters = importlib.import_module(f"{before.name}.parameters")
    scores, variance, total, cells = [], 0.0, 0, []
    for number, game_config in enumerate(screen.game_configs()):
        for index, sizes in enumerate(screen.SIZES):
            mean, var, n = head_to_head_condition(
                old, game_config, sizes, games, SEED + 10 * number + index
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
    parser.add_argument("out", type=Path)
    parser.add_argument("--games", type=int, default=64)
    parser.add_argument("--head-to-head", type=Path, default=None)
    arguments = parser.parse_args()
    log(f"lean from {lean.__file__}")
    store = Store(arguments.store)
    agent = ScriptedAgent.of()
    started = time.perf_counter()
    screen.play(store, [agent], arguments.games, "recheck")
    value, error, per_condition, played = recheck.skill(store, agent)
    log(f"panel skill {value:.4f} +- {error:.4f} ({played} games)")
    out = {
        "lean": str(Path(lean.__file__).resolve().parent),
        "games_per_cell": arguments.games,
        "agent_id": agent.id,
        "skill": value,
        "error": error,
        "games": played,
        "by_condition": per_condition,
        "marginals": recheck.marginals(per_condition),
    }
    arguments.out.write_text(json.dumps(out, indent=1) + "\n")
    if arguments.head_to_head is not None:
        out["head_to_head"] = head_to_head(arguments.head_to_head.resolve(), arguments.games)
        log(f"head to head: {out['head_to_head']['score']:.4f} +- {out['head_to_head']['error']:.4f}")
    out["seconds"] = time.perf_counter() - started
    arguments.out.write_text(json.dumps(out, indent=1) + "\n")
    log(f"wrote {arguments.out}")


if __name__ == "__main__":
    main()
