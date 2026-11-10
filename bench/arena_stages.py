"""Where a steady match-engine batch spends its time: after one batch (compiled, captured),
each compiled stage timed on its own outside the graph (CUDA events, best of `repeats`),
then a replayed block, a segment's host work (the store writes and the reads `play` makes)
and the act stage's parts. Prints one JSON line.

Usage: python bench/arena_stages.py [players=a,b,c] [preset=frontline] [repeats=20]
"""

import json
import sys
import time

import torch
from torch.utils import _pytree

from bnb.game.api import PRESETS
from bnb.league.matches import SEAT, Arena, ArenaConfig, player


def _timed(function, repeats: int) -> float:
    """The best of `repeats` runs, in ms (device time between events)."""
    best = float("inf")
    for _ in range(repeats):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        function()
        end.record()
        torch.cuda.synchronize()
        best = min(best, start.elapsed_time(end))
    return best


def main() -> None:
    """Build the engine, play a batch, time its parts; print one JSON line."""
    options = dict(a.split("=", 1) for a in sys.argv[1:])
    names = (
        options.get("players", "").split(",")
        if options.get("players")
        else [
            "scripted",
            "/home/vizia/avesnova/runs-v2/2026-11-08-bc-arm2d-persistence/policy-exponential",
            "/home/vizia/avesnova/runs-v2/2026-11-08-bc-arm2d-persistence/policy-live",
        ]
    )
    repeats = int(options.get("repeats", 20))
    game = PRESETS[options.get("preset", "frontline")]
    device = torch.device("cuda")
    config = ArenaConfig()
    arena = Arena([player(name) for name in names], game, device, config)
    pairs = [(1, 0), (2, 0), (1, 2), (0, 1), (0, 2), (2, 1), (1, 0), (2, 0)][: config.blocks]
    started = time.perf_counter()
    arena.play(pairs)  # compiles and captures
    first = time.perf_counter() - started
    rollout = arena.rollout
    saved = _pytree.tree_map_only(torch.Tensor, lambda x: x.clone(), rollout.carry)
    result = dict(first_batch_s=first, rows=rollout.environments, block=rollout.block)
    stages = dict(zip(("act", "advance", "resolve", "perceive"), rollout.stages, strict=True))
    for name, stage in stages.items():
        result[f"{name}_ms"] = _timed(lambda stage=stage: stage(rollout.carry), repeats)
    result["tick_sum_ms"] = sum(result[f"{name}_ms"] for name in stages)
    result["graph_block_ms"] = _timed(rollout.graph.replay, repeats)
    result["graph_tick_ms"] = result["graph_block_ms"] / rollout.block

    # one segment as `run` plays it: replays, the store writes, the host reads
    def segment():
        rollout.run()
        rollout.take_tally(SEAT)
        arena.seat.finished(rollout.carry["controllers"][SEAT])

    torch.cuda.synchronize()
    started = time.perf_counter()
    for _ in range(5):
        segment()
    torch.cuda.synchronize()
    result["segment_ms"] = (time.perf_counter() - started) / 5 * 1e3
    result["segment_ticks"] = rollout.length
    result["segment_graph_only_ms"] = result["graph_tick_ms"] * rollout.length
    # the act stage's parts: the scripted pass and the network groups
    seat, carry = arena.seat, rollout.carry
    view = rollout._view(carry, SEAT, rollout.reads + ("globals",))
    from bnb.contracts.obs import GLOBAL

    reset = view.globals[..., GLOBAL["reset"]] > 0.5
    scripted = torch.compile(seat.agent.step, fullgraph=True, dynamic=False)
    network = torch.compile(seat._network, fullgraph=True, dynamic=False)
    held = carry["controllers"][SEAT]
    with torch.no_grad():
        result["scripted_pass_ms"] = _timed(lambda: scripted(held.scripted, view, reset), repeats)
        blank = torch.zeros((seat.rows, seat.ships_per_team, 3), dtype=torch.int32, device=device)
        for group in range(len(seat.decisions)):
            result[f"network_group_{group}_ms"] = _timed(
                lambda group=group: network(group, blank, held.model.get(str(group)), view, reset),
                repeats,
            )
    for target, source in zip(
        _pytree.tree_leaves(rollout.carry), _pytree.tree_leaves(saved), strict=True
    ):
        if isinstance(target, torch.Tensor):
            target.copy_(source)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
