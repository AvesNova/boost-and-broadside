"""The scripted step time before (old_opponent.py) and after the pending-tick fix: `decide` for
every ship of a dealt 4v4 game (fields and fog on), compiled static (lean.contract.compile_graph, as
training), CUDA-synchronised, median of repeats after warm-up.

    speed.py [--envs 4096] [--repeats 50]
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import old_opponent  # noqa: E402

from lean import game  # noqa: E402
from lean.contract import compile_graph  # noqa: E402
from lean.scripted import ScriptedConfig, draw_uniforms, opponent  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=4096)
    parser.add_argument("--size", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=50)
    args = parser.parse_args()
    device = torch.device("cuda")
    generator = torch.Generator(device).manual_seed(1)
    config = game.GameConfig()
    sizes = torch.full((args.envs, 2), args.size, dtype=torch.long, device=device)
    state = game.reset(config, args.envs, sizes, generator, device)
    # A few ticks in, so the pending actions are not all neutral.
    for _ in range(30):
        uniforms = draw_uniforms(args.envs, state.x.shape[1], generator, device)
        action, _ = opponent.decide(ScriptedConfig(), state, None, None, uniforms)
        state, _ = game.tick(config, state, action, generator)
    uniforms = draw_uniforms(args.envs, state.x.shape[1], generator, device)
    result = {"envs": args.envs, "size": args.size}
    for name, module in (("old", old_opponent), ("new", opponent), ("old again", old_opponent)):
        function = compile_graph(
            lambda s, u, m=module: m.act(ScriptedConfig(), s, None, None, u), "static"
        )
        started = time.perf_counter()
        with torch.no_grad():
            function(state, uniforms)
            torch.cuda.synchronize()
            compile_seconds = time.perf_counter() - started
            for _ in range(10):
                function(state, uniforms)
            times = []
            for _ in range(args.repeats):
                torch.cuda.synchronize()
                t = time.perf_counter()
                function(state, uniforms)
                torch.cuda.synchronize()
                times.append(time.perf_counter() - t)
        result[name] = {
            "median_ms": 1000 * statistics.median(times),
            "first_call_s": compile_seconds,
        }
        print(name, result[name], file=sys.stderr, flush=True)
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
