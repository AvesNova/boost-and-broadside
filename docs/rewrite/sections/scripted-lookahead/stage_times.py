"""The look-ahead call's stages compiled one by one (lean.contract.compile_dynamic) at 2,048
environments of 4v4 (fields, fog): the paths, the pair kernel's inputs and the kernel, the
and the call without the team-mates' term; milliseconds per call.

    PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/scripted-lookahead/stage_times.py
"""

import time

import torch

from lean import game, scripted
from lean.contract import compile_dynamic
from lean.scripted import opponent


def main():
    device = torch.device("cuda")
    envs = 2048
    config = game.GameConfig()
    generator = torch.Generator(device).manual_seed(0)
    state = game.reset(config, envs, torch.full((envs, 2), 4, device=device), generator, device)
    slots = state.x.shape[1]
    agent = scripted.ScriptedConfig()
    with torch.no_grad():
        for _ in range(30):
            uniforms = scripted.draw_uniforms(envs, slots, generator, device)
            action = scripted.act(agent, state, None, None, uniforms)
            noise = game.draw_noise(config, envs, slots, generator, device)
            state, _ = game.step(config, state, action, noise)
        uniforms = scripted.draw_uniforms(envs, slots, generator, device)

        def timed(label, function):
            compiled = compile_dynamic(function)
            compiled()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(10):
                compiled()
            torch.cuda.synchronize()
            print(f"{label}: {100 * (time.perf_counter() - start):.2f} ms", flush=True)

        c = state.constants
        enemies = opponent.dead_reckon(
            game.core.enemy_view(state), state.ticks_since_seen.float() * c["tick_length"]
        )
        timed("paths (both views)", lambda: opponent._predict_both(c, state.fields, state, enemies))
        timed("whole call", lambda: scripted.decide(agent, state, None, None, uniforms))
        mates = opponent._mate_contacts
        opponent._mate_contacts = lambda *a, **k: 0.0
        timed("without mates", lambda: scripted.decide(agent, state, None, None, uniforms))
        opponent._mate_contacts = mates


if __name__ == "__main__":
    main()
