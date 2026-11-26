"""Where the compiled look-ahead call's GPU time goes (round 2, step 1): per stage (the paths of
both views, the whole call), the wall time per call, the GPU kernel time per call, the number of
kernels launched per call, and each kernel's share (torch.profiler). 2,048 environments of 4v4,
fields and fog, after 30 scripted ticks.

    PYTHONPATH=. uv run --no-sync python \
        docs/rewrite/sections/scripted-lookahead/profile_kernels.py [--fields 10] [--top 25]
"""

import argparse
import time
from collections import defaultdict

import torch

from lean import game, scripted
from lean.contract import compile_dynamic
from lean.scripted import opponent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=2048)
    parser.add_argument("--fields", type=int, default=10)
    parser.add_argument("--top", type=int, default=25)
    parser.add_argument("--stage", choices=("paths", "call", "both"), default="both")
    parser.add_argument("--plain", action="store_true", help="the plain PyTorch pair stage")
    arguments = parser.parse_args()
    if arguments.plain:
        from lean.scripted import kernels

        kernels.USE_KERNEL = False
    device = torch.device("cuda")
    envs = arguments.envs
    config = game.GameConfig(field_count=arguments.fields)
    generator = torch.Generator(device).manual_seed(0)
    state = game.reset(config, envs, torch.full((envs, 2), 4, device=device), generator, device)
    slots = state.x.shape[1]
    agent = scripted.ScriptedConfig()
    with torch.no_grad():
        for _ in range(30):
            action = scripted.act(
                agent, state, None, None, scripted.draw_uniforms(envs, slots, generator, device)
            )
            state, _ = game.step(
                config, state, action, game.draw_noise(config, envs, slots, generator, device)
            )
        uniforms = scripted.draw_uniforms(envs, slots, generator, device)
        c = state.constants
        enemies = opponent.dead_reckon(
            game.core.enemy_view(state), state.ticks_since_seen.float() * c["tick_length"]
        )
        engaged = torch.ones_like(state.x)

        def read(name, dims=0):
            return agent.values[name]

        stages = {
            "paths": lambda: (
                opponent._predict_views(c, state, enemies, agent, read, engaged)[0].stacked
            ),
            "call": lambda: scripted.decide(agent, state, None, None, uniforms),
        }
        chosen = ("paths", "call") if arguments.stage == "both" else (arguments.stage,)
        for name in chosen:
            call = compile_dynamic(stages[name])
            call()
            torch.cuda.synchronize()
            repeats = 20
            start = time.perf_counter()
            for _ in range(repeats):
                call()
            torch.cuda.synchronize()
            wall = 1000 * (time.perf_counter() - start) / repeats
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CUDA,
                    torch.profiler.ProfilerActivity.CPU,
                ]
            ) as profiler:
                for _ in range(repeats):
                    call()
                torch.cuda.synchronize()
            kernels = defaultdict(lambda: [0, 0.0])
            for event in profiler.events():
                if event.device_type == torch.autograd.DeviceType.CUDA:
                    entry = kernels[event.name]
                    entry[0] += 1
                    entry[1] += event.device_time
            total = sum(entry[1] for entry in kernels.values()) / repeats / 1000
            launches = sum(entry[0] for entry in kernels.values()) / repeats
            print(
                f"== {name}: wall {wall:.2f} ms per call, GPU kernel time {total:.2f} ms, "
                f"{launches:.0f} kernels per call",
                flush=True,
            )
            ranked = sorted(kernels.items(), key=lambda item: -item[1][1])
            for kernel, (count, micro) in ranked[: arguments.top]:
                per_call = micro / repeats / 1000
                print(
                    f"  {per_call:7.3f} ms  {100 * per_call / total:5.1f}%  "
                    f"x{count / repeats:4.0f}  {kernel[:110]}",
                    flush=True,
                )


if __name__ == "__main__":
    main()
