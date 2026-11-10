"""A size sweep two ways: an engine per fleet size against one engine at the largest size
with benching (the ships past each team's active count sit out, `Arena(benching=True)`).

For each of the `_arena` fleet presets of `base` (one map radius at every size, so a benched
game is the smaller preset's game), and for the largest size's engine built with benching,
the first segment's time (compile and capture), then the steady tick: a captured block's
replay (best of `repeats`) divided by its ticks, and a whole segment as `play` runs it.
The engines are built one after another in one process (the compile cache shared, as a
ratings run's would be). Prints one JSON line.

Usage: `python bench/fleet_sizes.py players=a,b [base=frontline] [sizes=10,20,40]
[block_environments=32] [repeats=10]`
"""

import json
import sys
import time

import torch

from bnb.cli import compile_limits
from bnb.game.api import PRESETS
from bnb.league.matches import Arena, ArenaConfig, player


def _prime(arena: Arena, matchups: list, active: list | None) -> None:
    """`Arena.play`'s set-up of a batch, without playing it."""
    arena.seat.clear()
    for block, (first, second) in enumerate(matchups):
        arena._load(2 * block, first)
        arena._load(2 * block + 1, second)
    if arena.rollout.ship_type is not None:
        arena.rollout.ship_type.copy_(arena._ship_types(matchups))
    arena.rollout.restart(None if active is None else arena._active(active))
    arena.rollout.carry["seat"].copy_(arena.seats)


def _measure(arena: Arena, matchups: list, active: list | None, repeats: int) -> dict:
    """The first segment (compiles and captures on a new engine) and the steady tick."""
    rollout = arena.rollout
    _prime(arena, matchups, active)
    torch.cuda.synchronize()
    started = time.perf_counter()
    rollout.run()
    torch.cuda.synchronize()
    first = time.perf_counter() - started
    best = float("inf")
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        rollout.graph.replay()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end))
    started = time.perf_counter()
    for _ in range(3):
        rollout.run()
    torch.cuda.synchronize()
    segment = (time.perf_counter() - started) / 3
    return dict(
        first_segment_s=first,
        tick_ms=best / rollout.block,
        segment_tick_ms=segment * 1e3 / rollout.length,
    )


def main() -> None:
    """Build each engine, measure; print one JSON line."""
    options = dict(a.split("=", 1) for a in sys.argv[1:])
    compile_limits()
    names = options["players"].split(",")
    base = options.get("base", "frontline")
    sizes = [int(size) for size in options.get("sizes", "10,20,40").split(",")]
    repeats = int(options.get("repeats", 10))
    device = torch.device("cuda")
    config = ArenaConfig(block_environments=int(options.get("block_environments", 32)))
    matchups = [(i % 2, 1 - i % 2) for i in range(config.blocks)]
    players = [player(name) for name in names]
    result = dict(base=base, players=names, games=config.games, own={}, benched={})
    for size in sizes:
        game = PRESETS[f"{base}_{size}v{size}_arena"]
        started = time.perf_counter()
        arena = Arena(players, game, device, config)
        build = time.perf_counter() - started
        result["own"][size] = dict(build_s=build, **_measure(arena, matchups, None, repeats))
        print(f"own {size}: {result['own'][size]}", file=sys.stderr, flush=True)
        del arena
        torch.cuda.empty_cache()
    largest = PRESETS[f"{base}_{max(sizes)}v{max(sizes)}_arena"]
    started = time.perf_counter()
    arena = Arena(players, largest, device, config, benching=True)
    result["benched"]["build_s"] = time.perf_counter() - started
    for size in sizes:
        active = [(size, size)] * config.blocks
        result["benched"][size] = _measure(arena, matchups, active, repeats)
        print(f"benched {size}: {result['benched'][size]}", file=sys.stderr, flush=True)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
