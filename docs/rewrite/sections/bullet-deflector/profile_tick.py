"""Where the deflector's GPU time goes: torch.profiler over the compiled step (off and frontal),
the top CUDA kernels by total time.

    PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/bullet-deflector/profile_tick.py
        [--envs 4096] [--size 4]
"""

import argparse

import torch
from bench_tick import actions  # noqa: E402  (beside this file)
from torch.profiler import ProfilerActivity, profile

from lean.game import core


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--size", type=int, default=4)
    args = parser.parse_args()
    device = torch.device("cuda")
    for preset in ("off", "frontal"):
        torch.compiler.reset()
        config = core.GameConfig(deflector=preset)
        step = torch.compile(core.step, dynamic=True)
        generator = torch.Generator(device=device).manual_seed(0)
        sizes = torch.tensor([[args.size, args.size]], device=device).expand(args.envs, 2)
        state = core.reset(config, args.envs, sizes, generator, device)
        slots = state.x.shape[1]
        drawn = [
            (
                actions(args.envs, slots, device, generator),
                core.draw_noise(config, args.envs, slots, generator, device),
            )
            for _ in range(60)
        ]
        for action, noise in drawn[:20]:
            state, _ = step(config, state, action, noise)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for action, noise in drawn[20:]:
                state, _ = step(config, state, action, noise)
            torch.cuda.synchronize()
        print(f"==== {preset}: 40 ticks")
        print(
            prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=14, max_name_column_width=70
            )
        )


if __name__ == "__main__":
    main()
