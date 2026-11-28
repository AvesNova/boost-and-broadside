"""The scripted step time of the tree on PYTHONPATH: `act` for every ship of a dealt 4v4 game
(fields and fog on), compiled static (lean.contract.compile_graph, as training),
CUDA-synchronised, median of repeats after warm-up (scripted-delay/speed.py with one module).
Run once from core's tree and once from this branch's.

    PYTHONPATH=TREE uv run --no-sync python speed.py [--envs 4096] [--repeats 50]
"""

import argparse
import json
import statistics
import time

import torch

from lean import game
from lean.contract import compile_graph
from lean.scripted import ScriptedConfig, draw_uniforms, opponent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--label", default="")
    args = parser.parse_args()
    device = torch.device("cuda")
    generator = torch.Generator(device).manual_seed(1)
    config = game.GameConfig()
    sizes = torch.full((args.envs, 2), args.size, dtype=torch.long, device=device)
    state = game.reset(config, args.envs, sizes, generator, device)
    with torch.no_grad():
        for _ in range(30):
            uniforms = draw_uniforms(args.envs, state.x.shape[1], generator, device)
            action, _ = opponent.decide(ScriptedConfig(), state, None, None, uniforms)
            state, _ = game.tick(config, state, action, generator)
    uniforms = draw_uniforms(args.envs, state.x.shape[1], generator, device)
    function = compile_graph(lambda s, u: opponent.act(ScriptedConfig(), s, None, None, u), "static")
    with torch.no_grad():
        started = time.perf_counter()
        function(state, uniforms)
        torch.cuda.synchronize()
        first = time.perf_counter() - started
        for _ in range(10):
            function(state, uniforms)
        times = []
        for _ in range(args.repeats):
            torch.cuda.synchronize()
            t = time.perf_counter()
            function(state, uniforms)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t)
    print(json.dumps({"label": args.label, "envs": args.envs, "size": args.size,
                      "median_ms": 1000 * statistics.median(times), "first_call_s": first}))


if __name__ == "__main__":
    main()
