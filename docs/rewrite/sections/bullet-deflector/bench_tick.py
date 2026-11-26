"""The deflector's cost: the game step's time per tick, off / symmetric / frontal.

    PYTHONPATH=. uv run --no-sync python \
        docs/rewrite/sections/bullet-deflector/bench_tick.py OUT.json
        [--envs 4096] [--size 4] [--ticks 300] [--cpu-envs 256]

GameConfig(deflector=...) otherwise the game as trained (fields, fog); random actions (drawn
outside the timed step). GPU: torch.compile(core.tick's step, dynamic=True, as
lean.contract.compile), warmed, then `--ticks` ticks timed with synchronisation, median of 3
blocks. CPU: eager, `--cpu-envs` environments, 4 threads. Also the eager operation count of one
step (torch dispatch calls, CPU). Records the power profile (powerprofilesctl).
"""

import argparse
import json
import statistics
import subprocess
import time

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from lean.contract import FIRES, POWERS, STICKS
from lean.game import core


class Count(TorchDispatchMode):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.calls += 1
        return func(*args, **(kwargs or {}))


def actions(envs, slots, device, generator):
    return torch.stack(
        [
            torch.randint(0, n, (envs, slots), device=device, generator=generator)
            for n in (STICKS, POWERS, FIRES)
        ],
        -1,
    )


def timed(step, config, envs, size, ticks, device, blocks=3):
    generator = torch.Generator(device=device).manual_seed(0)
    sizes = torch.tensor([[size, size]], device=device).expand(envs, 2)
    state = core.reset(config, envs, sizes, generator, device)
    slots = state.x.shape[1]
    for _ in range(20):  # warm (and compile)
        noise = core.draw_noise(config, envs, slots, generator, device)
        state, _ = step(config, state, actions(envs, slots, device, generator), noise)
    times = []
    for _ in range(blocks):
        drawn = [
            (
                actions(envs, slots, device, generator),
                core.draw_noise(config, envs, slots, generator, device),
            )
            for _ in range(ticks)
        ]
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for action, noise in drawn:
            state, _ = step(config, state, action, noise)
        if device.type == "cuda":
            torch.cuda.synchronize()
        times.append((time.perf_counter() - start) / ticks)
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--ticks", type=int, default=300)
    parser.add_argument("--cpu-envs", type=int, default=256)
    args = parser.parse_args()
    torch.set_num_threads(4)
    try:
        profile = subprocess.run(
            ["powerprofilesctl", "get"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        profile = "unknown"
    record = dict(power_profile=profile, envs=args.envs, size=args.size, presets={})
    cpu = torch.device("cpu")
    for preset in ("off", "symmetric", "frontal"):
        config = core.GameConfig(deflector=preset)
        entry = {}
        generator = torch.Generator().manual_seed(0)
        sizes = torch.tensor([[args.size, args.size]]).expand(args.cpu_envs, 2)
        state = core.reset(config, args.cpu_envs, sizes, generator, cpu)
        slots = state.x.shape[1]
        noise = core.draw_noise(config, args.cpu_envs, slots, generator, cpu)
        with Count() as count:
            core.step(config, state, actions(args.cpu_envs, slots, cpu, generator), noise)
        entry["eager_dispatch_calls"] = count.calls
        entry["cpu_eager_ms"] = 1e3 * timed(core.step, config, args.cpu_envs, args.size, 50, cpu)
        if torch.cuda.is_available():
            torch.compiler.reset()
            compiled = torch.compile(core.step, dynamic=True)
            device = torch.device("cuda")
            started = time.perf_counter()
            entry["gpu_compiled_ms"] = 1e3 * timed(
                compiled, config, args.envs, args.size, args.ticks, device
            )
            entry["gpu_wall_with_compile_s"] = round(time.perf_counter() - started, 1)
            entry["gpu"] = torch.cuda.get_device_name()
        record["presets"][preset] = entry
        print(preset, json.dumps(entry), flush=True)
    with open(args.out, "w") as handle:
        json.dump(record, handle, indent=1)


if __name__ == "__main__":
    main()
