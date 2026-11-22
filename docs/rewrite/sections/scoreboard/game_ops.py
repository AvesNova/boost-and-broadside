"""Scoreboard, game side: compute operations per call (regions.py rule: every aten op that
reaches the dispatcher, views and metadata apart) and CPU time, uncompiled, one thread.

    OMP_NUM_THREADS=1 PYTHONPATH=.:src uv run --no-sync python docs/rewrite/sections/scoreboard/game_ops.py

Operation counts do not depend on the batch size; times are at E = 256, 5v5, CPU, one thread.
"""

import dataclasses
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "simplification"))
import regions as R  # noqa: E402

from lean.game import core  # noqa: E402
from lean.scripted.opponent import ScriptedConfig, act, draw_uniforms, personas  # noqa: E402

torch.set_num_threads(1)
E, PER_TEAM = 256, 5


def measure(label, fn, repeats=30):
    for _ in range(3):
        fn()
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        times.append(time.perf_counter() - start)
    R.reset()
    with R.Counter():
        fn()
    compute = sum(c for c, _ in R.EXCLUSIVE.values())
    views = sum(v for _, v in R.EXCLUSIVE.values())
    print(f"| {label} | {compute} | {views} | {statistics.median(times) * 1e3:.2f} |", flush=True)


def game(config, label):
    generator = torch.Generator().manual_seed(0)
    sizes = torch.tensor([[PER_TEAM, PER_TEAM]]).expand(E, 2)
    state = core.reset(config, E, sizes, generator, torch.device("cpu"))
    n = state.x.shape[1]
    action = torch.stack(
        [torch.randint(0, 7, (E, n), generator=generator), torch.randint(0, 3, (E, n), generator=generator),
         torch.randint(0, 2, (E, n), generator=generator)], -1,
    )
    noise = core.draw_noise(config, E, n, generator, torch.device("cpu"))
    for _ in range(30):  # some bullets in flight
        state, _ = core.step(config, state, action, noise)
    done = torch.zeros(E, dtype=torch.bool)
    done[0] = True
    with torch.no_grad():
        measure(f"step, {label}", lambda: core.step(config, state, action, noise))
        measure(f"observe (one team), {label}", lambda: core.observe(config, state, 0))
        measure(f"reset_done, {label}", lambda: core.reset_done(config, state, done, sizes, generator))
    return state


if __name__ == "__main__":
    print("| call | compute ops | view ops | CPU ms per call (E = 256, 5v5, one thread) |")
    print("|---|---|---|---|")
    v2 = core.GameConfig()
    game(v2, "v2 config (no fields)")
    game(dataclasses.replace(v2, fields=((800.0, 0.0, 600.0, 160.0, 2.0),) * 10), "v2 config, 10 fixed fields")
    state = game(core.GameConfig.core(), "core config (core flight and damage, ghosts, 10 drawn fields)")
    scripted = ScriptedConfig()
    persona = personas(scripted, E, state.x.shape[1], torch.Generator().manual_seed(1), torch.device("cpu"), 0.0)
    uniforms = draw_uniforms(E, state.x.shape[1], torch.Generator().manual_seed(2), "cpu")
    with torch.no_grad():
        measure("scripted act (one team), core config state", lambda: act(scripted, state, 0, persona, uniforms))
