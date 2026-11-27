"""Where the look-ahead agent's GPU time goes at 2,048 environments of 4v4 (fields, fog): the
paths and the whole call eager (synchronised), and with --compiled the compiled call profiled
with torch.profiler, its kernels by CUDA time.

    PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/scripted-lookahead/profile_call.py
"""

import argparse
import time

import torch

from lean import game, scripted
from lean.contract import compile_dynamic
from lean.scripted import paths


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=2048)
    parser.add_argument("--compiled", action="store_true")
    arguments = parser.parse_args()
    device = torch.device("cuda")
    envs = arguments.envs
    config = game.GameConfig()
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

        def stage(name, function, repeats=5):
            function()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(repeats):
                function()
            torch.cuda.synchronize()
            print(
                f"{name}: {1000 * (time.perf_counter() - start) / repeats:.1f} ms (eager)",
                flush=True,
            )

        stage("predict (one view)", lambda: paths.predict(state.constants, state.fields, state))
        stage("whole call", lambda: scripted.decide(agent, state, None, None, uniforms))
        if not arguments.compiled:
            return
        call = compile_dynamic(lambda s, u: scripted.decide(agent, s, None, None, u))
        call(state, uniforms)
        torch.cuda.synchronize()
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
            for _ in range(3):
                call(state, uniforms)
            torch.cuda.synchronize()
        print(
            profiler.key_averages().table(
                sort_by="cuda_time_total", row_limit=25, max_name_column_width=90
            )
        )


if __name__ == "__main__":
    main()
