"""Steady environment ticks/s of one layout, for the tree on PYTHONPATH (old or routed runner).

    PYTHONPATH=TREE uv run --no-sync python docs/rewrite/sections/eval-scale/tick_compare.py
        [--envs 2688] [--layout net-scripted|net-net|scripted] [--seconds 40]

An untrained network (seed 0) in seat A against the yardstick (net-scripted), against a
second untrained network (net-net), or the yardstick against itself; 4v4, the core game.
A warm-up call first (compiles), then one call cut off after `--seconds`: E x the ticks
before the cutoff over that time.
"""

import argparse
import inspect
import json
import subprocess
import time

import torch

from lean import game, model, scripted
from lean.eval import Queue, play
from lean.scripted.parameters import vector


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--envs", type=int, default=2688)
    parser.add_argument("--layout", default="net-scripted")
    parser.add_argument("--seconds", type=float, default=40.0)
    arguments = parser.parse_args()
    routed = "policies" in inspect.signature(play).parameters
    device = "cuda"
    table = torch.tensor([vector(scripted.ScriptedConfig())], device=device)
    policies = []
    for seed in range(2):
        torch.manual_seed(seed)
        policies.append(model.Policy(model.ModelConfig()).to(device).eval())
    count = {"net-scripted": 1, "net-net": 2, "scripted": 0}[arguments.layout]

    def queue_of(games):
        zeros = torch.zeros(games, dtype=torch.long)
        columns = [zeros, zeros, torch.arange(games) % 2, zeros + 4, zeros + 4]
        if routed:
            nets = [zeros if count >= 1 else zeros - 1, zeros + 1 if count == 2 else zeros - 1]
            columns += nets
        return Queue(*columns)

    def run(games, envs, seconds=None, progress=None):
        if routed:
            return play(
                game.GameConfig(), queue_of(games), table, policies[:count], envs=envs,
                compiled=True, seconds=seconds, progress=progress, minimum_envs=envs,
            )  # fmt: skip
        seats = (policies + [None, None])[:2] if count == 2 else [policies[0], None]
        if count == 0:
            seats = [None, None]
        return play(
            game.GameConfig(), queue_of(games), table, *seats, envs=envs, compiled=True,
            seconds=seconds, progress=progress, minimum_envs=envs,
        )  # fmt: skip

    started = time.perf_counter()
    run(256, 256)
    warm = time.perf_counter() - started
    marks = []
    run(64 * arguments.envs, arguments.envs, arguments.seconds, lambda g, b, s: marks.append(s))
    before = [seconds for seconds in marks if seconds <= arguments.seconds]
    profile = subprocess.run(["powerprofilesctl", "get"], capture_output=True, text=True).stdout
    print(
        json.dumps(
            {
                "routed": routed,
                "layout": arguments.layout,
                "envs": arguments.envs,
                "profile": profile.strip(),
                "warm_up_s": round(warm, 1),
                "env_ticks_per_s": round(len(before) * 32 * arguments.envs / before[-1]),
            }
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
