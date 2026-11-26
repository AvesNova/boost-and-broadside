"""GPU time of the candidate paths (both views, one pass, compiled) at 2,048 environments of
4v4, with the default fields and without: the cost of evaluating the medium per segment.

    PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/scripted-lookahead/paths_cost.py
"""

import time

import torch

from lean import game
from lean.contract import compile_dynamic
from lean.scripted import opponent


def main():
    device = torch.device("cuda")
    envs = 2048
    for fields in (10, 0):
        config = game.GameConfig(field_count=fields)
        generator = torch.Generator(device).manual_seed(0)
        state = game.reset(config, envs, torch.full((envs, 2), 4, device=device), generator, device)
        c = state.constants
        enemies = opponent.dead_reckon(
            game.core.enemy_view(state), state.ticks_since_seen.float() * c["tick_length"]
        )
        call = compile_dynamic(lambda: opponent._predict_views(c, state, enemies, 2.2))
        with torch.no_grad():
            call()
            torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(20):
                call()
            torch.cuda.synchronize()
        print(f"fields {fields}: {50 * (time.perf_counter() - start):.2f} ms", flush=True)


if __name__ == "__main__":
    main()
