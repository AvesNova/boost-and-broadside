"""The cost of making fresh deals, apart from taking them: lean.learn.rollout's
draw_next_episodes (once per segment, eagerly: E deals, personas, learner team, position
offset) and lean.game.deal alone, on the GPU, median of `--repeats` timed calls with
synchronisation; then a torch.profiler table of one deal (CPU and CUDA time per op).

    PYTHONPATH=. uv run --no-sync python docs/rewrite/sections/tick-audit/bench_deal.py \
        OUT.json [--envs 2048] [--repeats 10]
"""

import argparse
import json
import os
import statistics
import subprocess
import time

import torch
from torch.profiler import ProfilerActivity, profile

from lean import game, model, scripted
from lean.learn import League, LearnConfig, Opponent, start
from lean.learn.rollout import draw_next_episodes, draw_team_sizes


def timed(function, repeats):
    function()
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        began = time.perf_counter()
        function()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - began)
    return 1e3 * statistics.median(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--envs", type=int, default=LearnConfig().environments)
    parser.add_argument("--repeats", type=int, default=10)
    arguments = parser.parse_args()
    device = torch.device("cuda")
    config, game_config = LearnConfig(), game.GameConfig()
    envs = arguments.envs
    generator = torch.Generator(device=device).manual_seed(0)
    policy = model.Policy(model.ModelConfig()).to(device)
    opponent = Opponent(scripted.ScriptedConfig(), League(config, device))
    state, _ = start(config, game_config, policy, opponent, envs, generator)
    slots = state.exists.shape[1]
    sizes = draw_team_sizes(config, envs, generator, device)
    record = {
        "power_profile": subprocess.run(
            ["powerprofilesctl", "get"], capture_output=True, text=True
        ).stdout.strip(),
        "load_average": os.getloadavg(),
        "envs": envs,
        "slots": slots,
        "segment_ticks": config.segment_ticks,
        "draw_next_episodes_ms": timed(
            lambda: draw_next_episodes(
                config, game_config, policy, opponent, envs, slots, generator, device
            ),
            arguments.repeats,
        ),
        "deal_ms": timed(
            lambda: game.deal(game_config, envs, slots, sizes, generator, device), arguments.repeats
        ),
    }
    record["per_tick_ms"] = record["draw_next_episodes_ms"] / config.segment_ticks
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        game.deal(game_config, envs, slots, sizes, generator, device)
        torch.cuda.synchronize()
    table = prof.key_averages().table(sort_by="cpu_time_total", row_limit=25)
    record["load_average_end"] = os.getloadavg()
    print(json.dumps(record), flush=True)
    print(table, flush=True)
    with open(arguments.out, "w") as handle:
        json.dump({**record, "profile": table}, handle, indent=1)


if __name__ == "__main__":
    main()
