"""The pair kernel alone on its real inputs (round 2, step 1): its inputs captured from one
compiled-shape call at 2,048 environments of 4v4 (fields, fog, after 30 scripted ticks), then
timed by triton.testing.do_bench, with and without the fields (the cover loop's share), and its
registers and spills from the compiled kernel.

    PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/scripted-lookahead/bench_kernel.py
"""

import argparse

import torch
import triton.testing

from lean import game, scripted
from lean.scripted import kernels


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
    captured = {}
    original = kernels.pair_scores

    def capture(*args):
        captured["args"] = args
        return original(*args)

    with torch.no_grad():
        for _ in range(30):
            action = scripted.act(
                agent, state, None, None, scripted.draw_uniforms(envs, slots, generator, device)
            )
            state, _ = game.step(
                config, state, action, game.draw_noise(config, envs, slots, generator, device)
            )
        kernels.pair_scores = capture
        from lean.scripted import opponent

        opponent.kernels.pair_scores = capture
        scripted.decide(
            agent, state, None, None, scripted.draw_uniforms(envs, slots, generator, device)
        )
        kernels.pair_scores = original
        args = captured["args"]
        paths, constants, parameters, times, rows, now, fields, count, slots_, views = args
        print(
            f"pairs {rows.shape[0]}, path tensor {tuple(paths.shape)} (points, envs, "
            f"views x slots, fields, candidates), game fields {fields.shape[-1]}",
            flush=True,
        )
        for label, field_tensor in (
            ("fields 10", fields),
            ("fields 0 (no cover loop)", fields[..., :0]),
        ):
            run = lambda: original(  # noqa: E731
                paths,
                constants,
                parameters,
                times,
                rows,
                now,
                field_tensor.contiguous(),
                count,
                slots_,
                views,
            )
            milliseconds = triton.testing.do_bench(run, warmup=50, rep=400)
            print(f"{label}: {milliseconds:.3f} ms", flush=True)
        for key, compiled in kernels._pairs.device_caches[0][0].items():
            print(
                f"compiled variant: registers {compiled.n_regs}, spills {compiled.n_spills}, "
                f"shared {compiled.metadata.shared}",
                flush=True,
            )


if __name__ == "__main__":
    main()
