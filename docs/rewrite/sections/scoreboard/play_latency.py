"""Gate G2: batch-size-1 latency of one play tick on the CPU: two observations, two policy steps
(one per team) and one game step, core config, new model, one thread, uncompiled or compiled.

    PYTHONPATH=.:src python play_latency.py eager|compiled PER_TEAM [TICKS]

Prints p50, p99 and the worst tick in ms over TICKS ticks after 50 warm-up ticks.
"""

import os
import statistics
import sys
import time

import torch

from lean import model
from lean.game import core

torch.set_num_threads(1)
torch.fx.experimental._config.use_duck_shape = False
VARIANT, PER_TEAM = sys.argv[1], int(sys.argv[2])
TICKS = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
DEVICE = torch.device("cpu")

config = core.GameConfig(clock_ticks=10**9)
generator = torch.Generator().manual_seed(0)
sizes = torch.tensor([[PER_TEAM, PER_TEAM]])
state = core.reset(config, 1, sizes, generator, DEVICE)
policy = model.Policy(model.ModelConfig()).eval()
carries = [policy.initial_carry(1, state.x.shape[1], DEVICE) for _ in range(2)]
step_game, step_policy = core.step, policy.step
if VARIANT == "compiled":
    step_game = torch.compile(core.step, dynamic=True)
    step_policy = torch.compile(policy.step, dynamic=True)


def tick():
    global state
    actions = []
    for team in (0, 1):
        observation = core.observe(config, state, team)
        logits, _, carries[team] = step_policy(observation, carries[team])
        actions.append(logits.argmax(-1))
    joint = torch.where(state.team == 0, actions[0], actions[1])
    action = torch.stack([joint // 6, (joint // 2) % 3, joint % 2], -1)
    noise = core.draw_noise(config, 1, state.x.shape[1], generator, DEVICE)
    state, _ = step_game(config, state, action, noise)


with torch.no_grad():
    start = time.perf_counter()
    tick()
    first = time.perf_counter() - start
    for _ in range(50):
        tick()
    times = []
    for _ in range(TICKS):
        start = time.perf_counter()
        tick()
        times.append((time.perf_counter() - start) * 1e3)
times.sort()
load = os.getloadavg()[0]
print(
    f"{VARIANT} {PER_TEAM}v{PER_TEAM}: first tick {first:.1f} s, p50 {statistics.median(times):.2f} ms, "
    f"p99 {times[int(0.99 * len(times))]:.2f} ms, worst {times[-1]:.2f} ms, load {load:.1f}",
    flush=True,
)
