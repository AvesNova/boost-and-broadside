"""Scoreboard, GPU: uncompiled against compiled time, first-call (compile) time, and what a new
team size costs, for the game step (core config) and the policy step. One measurement per
fresh process with an empty inductor cache (pass a fresh TORCHINDUCTOR_CACHE_DIR).

    PYTHONPATH=.:src python gpu_compile.py game|policy eager|compiled|dynamic [E]

Prints one line per team size: size, first call s, steady ms (median of 30 after 5 warm-up).
"""

import statistics
import sys
import time

import torch

from lean.game import core

UNIT, VARIANT = sys.argv[1], sys.argv[2]
E = int(sys.argv[3]) if len(sys.argv) > 3 else 2048
DEVICE = torch.device("cuda")
SIZES = tuple(int(x) for x in __import__("os").environ.get("SIZES", "5,10,20,50").split(","))
torch.manual_seed(0)
if __import__("os").environ.get("NO_DUCK"):
    torch.fx.experimental._config.use_duck_shape = False


def wrap(fn):
    if VARIANT == "eager":
        return fn
    return torch.compile(fn, dynamic=True if VARIANT == "dynamic" else None)


def timed(run):
    torch.cuda.synchronize()
    start = time.perf_counter()
    run()
    torch.cuda.synchronize()
    first = time.perf_counter() - start
    for _ in range(5):
        run()
    times = []
    for _ in range(30):
        torch.cuda.synchronize()
        start = time.perf_counter()
        run()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    return first, statistics.median(times) * 1e3


def game_runner(per_team):
    config = core.GameConfig.core()
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    sizes = torch.tensor([[per_team, per_team]], device=DEVICE).expand(E, 2)
    state = core.reset(config, E, sizes, generator, DEVICE)
    n = state.x.shape[1]
    action = torch.stack(
        [torch.randint(0, 7, (E, n), device=DEVICE, generator=generator),
         torch.randint(0, 3, (E, n), device=DEVICE, generator=generator),
         torch.randint(0, 2, (E, n), device=DEVICE, generator=generator)], -1,
    )
    noise = core.draw_noise(config, E, n, generator, DEVICE)
    step = STEP_GAME

    def run():
        with torch.no_grad():
            step(config, state, action, noise)

    return run


def policy_runner(per_team):
    from lean import model

    config = core.GameConfig.core()
    generator = torch.Generator(device=DEVICE).manual_seed(0)
    sizes = torch.tensor([[per_team, per_team]], device=DEVICE).expand(E, 2)
    state = core.reset(config, E, sizes, generator, DEVICE)
    observation = core.observe(config, state, 0)
    policy = POLICY
    carry = policy.initial_carry(E, state.x.shape[1], DEVICE)
    step = STEP_POLICY

    def run():
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            step(observation, carry)

    return run


if UNIT == "game":
    STEP_GAME = wrap(core.step)
    runner = game_runner
else:
    from lean import model

    POLICY = model.Policy(model.ModelConfig()).to(DEVICE)
    STEP_POLICY = wrap(POLICY.step)
    runner = policy_runner

for per_team in SIZES:
    first, steady = timed(runner(per_team))
    print(f"{UNIT} {VARIANT} E={E} {per_team}v{per_team}: first call {first:.2f} s, steady {steady:.2f} ms", flush=True)
