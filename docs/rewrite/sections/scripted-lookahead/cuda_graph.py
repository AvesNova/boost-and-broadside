"""The scripted call with and without a CUDA graph (round 2, a measurement only; lean compiles
with dynamic shapes and no graphs): the call compiled as lean does (`compile_dynamic`), that
compiled call captured by hand in a torch.cuda.CUDAGraph at the fixed batch (static input
tensors copied in), and torch.compile(mode="reduce-overhead") at a fixed shape. 2,048
environments of 4v4, fields and fog, after 30 scripted ticks. ms per call over 50 calls.

    PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/scripted-lookahead/cuda_graph.py
"""

import argparse
import dataclasses
import time

import torch

from lean import game, scripted
from lean.contract import compile_dynamic, map_fields


def timed(function, repeats: int = 50) -> float:
    for _ in range(5):
        function()
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        function()
    torch.cuda.synchronize()
    return 1000.0 * (time.perf_counter() - start) / repeats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=2048)
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

        def call(state, uniforms):
            return scripted.act(agent, state, None, None, uniforms)

        compiled = compile_dynamic(call)
        begun = time.perf_counter()
        compiled(state, uniforms)
        torch.cuda.synchronize()
        print(f"compile (dynamic): first call {time.perf_counter() - begun:.1f} s", flush=True)
        print(f"compiled, no graph: {timed(lambda: compiled(state, uniforms)):.2f} ms", flush=True)

        # Hand capture: static copies of every input tensor, the compiled call captured once.
        static_state = map_fields(state, lambda tensor: tensor.clone())
        static_state = dataclasses.replace(
            static_state, constants=map_fields(state.constants, lambda tensor: tensor.clone())
        )
        static_uniforms = uniforms.clone()
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                compiled(static_state, static_uniforms)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(graph):
                static_action = compiled(static_state, static_uniforms)

            def replay():
                static_uniforms.copy_(uniforms)  # a new draw per call (the state copies too)
                graph.replay()
                return static_action

            print(f"compiled, hand-captured graph: {timed(replay):.2f} ms", flush=True)
            same = torch.equal(replay(), compiled(state, uniforms))
            print(f"  graph output equals the call's: {same}", flush=True)
        except Exception as error:  # noqa: BLE001 (report the constraint)
            print(f"hand capture failed: {type(error).__name__}: {error}", flush=True)

        static = torch.compile(call, dynamic=False)
        begun = time.perf_counter()
        static(state, uniforms)
        torch.cuda.synchronize()
        first = time.perf_counter() - begun
        milliseconds = timed(lambda: static(state, uniforms))
        print(
            f"static shapes, no graph: {milliseconds:.2f} ms, first call {first:.1f} s", flush=True
        )
        try:
            overhead = torch.compile(call, mode="reduce-overhead", dynamic=False)
            begun = time.perf_counter()
            overhead(state, uniforms)
            torch.cuda.synchronize()
            first = time.perf_counter() - begun
            milliseconds = timed(lambda: overhead(state, uniforms).clone())
            print(
                f"reduce-overhead (static shapes): {milliseconds:.2f} ms, first call {first:.1f} s",
                flush=True,
            )
            same = torch.equal(overhead(state, uniforms).clone(), compiled(state, uniforms))
            print(f"  reduce-overhead output equals the dynamic call's: {same}", flush=True)
        except Exception as error:  # noqa: BLE001
            print(f"reduce-overhead failed: {type(error).__name__}: {error}", flush=True)


if __name__ == "__main__":
    main()
