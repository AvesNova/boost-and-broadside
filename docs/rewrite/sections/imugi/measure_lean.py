"""lean/scripted/opponent.py: compute ops per `act` call (regions.py rule: views excluded)
and CPU time per call, one thread, E environments, one team per call.

usage: OMP_NUM_THREADS=1 PYTHONPATH=.:src uv run --no-sync python measure_lean.py [E] [warm_ticks]
"""

import statistics
import sys
import time

import torch

torch.set_num_threads(1)
sys.path.insert(0, "docs/rewrite/sections/simplification")
import regions as R  # noqa: E402

from lean import game  # noqa: E402
from lean.scripted import ScriptedConfig, act, chaser_act, draw_uniforms, fresh_memory, personas  # noqa: E402

E = int(sys.argv[1]) if len(sys.argv) > 1 else 256
WARM = int(sys.argv[2]) if len(sys.argv) > 2 else 450
SLOTS = 10
g = torch.Generator().manual_seed(11)
game_config = game.GameConfig()
config = ScriptedConfig(arena_radius=game_config.arena_radius)
team_sizes = torch.full((E, 2), 5, dtype=torch.long)
state = game.reset(game_config, E, team_sizes, g, "cpu")
yardstick = personas(config, E, SLOTS, g, "cpu", 0.0)
population = personas(config, E, SLOTS, g, "cpu", 1.0)
t0 = time.time()
with torch.no_grad():
    for _ in range(WARM):
        u = draw_uniforms(E, SLOTS, g, "cpu")
        action = act(config, state, 0, yardstick, u) + act(config, state, 1, yardstick, u)
        state, out = game.step(
            game_config, state, action, game.draw_noise(game_config, E, SLOTS, g, "cpu")
        )
print(f"warm {WARM} ticks in {time.time() - t0:.0f} s; alive {state.alive.float().mean():.2f}")


def measure(label, fn, repeats=60):
    with torch.no_grad():
        for _ in range(5):
            fn()
        times = []
        for _ in range(repeats):
            t = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t)
        R.reset()
        with R.Counter():
            fn()
    compute = sum(c for c, _ in R.EXCLUSIVE.values())
    views = sum(v for _, v in R.EXCLUSIVE.values())
    print(
        f"{label:<55} {compute:>4} compute ops (+{views} views)  median {statistics.median(times) * 1e3:.3f} ms  min {min(times) * 1e3:.3f} ms"
    )


uniforms = draw_uniforms(E, SLOTS, g, "cpu")
memory = fresh_memory(E, SLOTS, "cpu")
measure(
    "act, spread 0, no memory (the yardstick as tested)",
    lambda: act(config, state, 0, yardstick, uniforms),
)
measure("act, spread 0, with memory", lambda: act(config, state, 0, yardstick, uniforms, memory))
measure(
    "act, spread 1 persona, with memory",
    lambda: act(config, state, 0, population, uniforms, memory),
)
measure("draw_uniforms (the one ordinary draw)", lambda: draw_uniforms(E, SLOTS, g, "cpu"))
measure("chaser_act", lambda: chaser_act(config, state, 0))
measure(
    "both teams: two act calls, no memory",
    lambda: (
        act(config, state, 0, yardstick, uniforms),
        act(config, state, 1, yardstick, uniforms),
    ),
)
