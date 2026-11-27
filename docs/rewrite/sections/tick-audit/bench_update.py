"""The update's time through lean.learn.compiled's graphs, static or dynamic: one update's
segments collected (compiled, the same shapes), then `--warm` updates and `--timed` updates
on that store, each timed with synchronisation; ms per update, per Adam step and per
micro-batch (the loss's forward and backward with the optimiser step shared out).

    PYTHONPATH=<tree> uv run --no-sync python \
        docs/rewrite/sections/tick-audit/bench_update.py OUT.jsonl --shapes static|dynamic

The default LearnConfig (RL update: teacher weight 0.5) and GameConfig, the scripted agent
the only league member. Appends one JSON row: power profile, load average, the times.
"""

import argparse
import dataclasses
import json
import os
import subprocess
import time

import torch

from lean import game, model, scripted
from lean.learn import (
    League,
    LearnConfig,
    Opponent,
    RolloutStore,
    collect,
    make_optimiser,
    start,
    update,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--shapes", default="static")
    parser.add_argument("--envs", type=int, default=LearnConfig().environments)
    parser.add_argument("--warm", type=int, default=1)
    parser.add_argument("--timed", type=int, default=2)
    arguments = parser.parse_args()
    device = torch.device("cuda")
    config = dataclasses.replace(LearnConfig(), compile_shapes=arguments.shapes)
    game_config = game.GameConfig()
    torch.manual_seed(0)
    generator = torch.Generator(device=device).manual_seed(0)
    policy = model.Policy(model.ModelConfig()).to(device)
    opponent = Opponent(scripted.ScriptedConfig(), League(config, device))
    state, carry = start(config, game_config, policy, opponent, arguments.envs, generator)
    store = RolloutStore(config.segments_per_update, config.stream_rollout)
    for segment in range(config.segments_per_update):
        rollout = collect(
            config,
            game_config,
            policy,
            opponent,
            state,
            carry,
            config.segment_ticks,
            generator,
            compiled=True,
            store=store,
            segment=segment,
        )
        state, carry = rollout.state, rollout.carry
    optimiser = make_optimiser(config, policy)
    seconds, steps = [], []
    load_before = None
    for index in range(arguments.warm + arguments.timed):
        if index == arguments.warm:
            load_before = os.getloadavg()
        torch.cuda.synchronize()
        began = time.perf_counter()
        metrics = update(
            config,
            policy,
            optimiser,
            store,
            step_index=10**6,
            generator=generator,
            compiled=True,
            teacher_weight=0.5,
        )
        torch.cuda.synchronize()
        if index >= arguments.warm:
            seconds.append(time.perf_counter() - began)
            steps.append(int(metrics["adam_steps"]))
    views = store.views
    micro_batches = config.epochs * -(-views // min(config.micro_batch_views, views))
    row = {
        "shapes": arguments.shapes,
        "envs": arguments.envs,
        "power_profile": subprocess.run(
            ["powerprofilesctl", "get"], capture_output=True, text=True
        ).stdout.strip(),
        "load_average": load_before,
        "load_average_end": os.getloadavg(),
        "update_ms": [round(1e3 * s, 1) for s in seconds],
        "adam_steps": steps,
        "ms_per_adam_step": [round(1e3 * s / n, 2) for s, n in zip(seconds, steps)],
        "micro_batches_about": micro_batches,
        "ms_per_micro_batch": [round(1e3 * s / micro_batches, 2) for s in seconds],
        "gpu": torch.cuda.get_device_name(),
    }
    print(json.dumps(row), flush=True)
    with open(arguments.out, "a") as handle:
        handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
