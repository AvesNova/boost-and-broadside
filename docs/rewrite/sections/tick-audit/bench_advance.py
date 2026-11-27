"""The training tick's time: collect's segments through lean.learn.compiled's graphs, the
whole segment per tick and the two halves (act, advance) by CUDA events around each call.

    PYTHONPATH=<tree> uv run --no-sync python \
        docs/rewrite/sections/tick-audit/bench_advance.py OUT.jsonl [--label NAME]
        [--envs 2048] [--warm 2] [--timed 4] [--shapes static] [--deflector egg]

The default LearnConfig and GameConfig (4v4 to 4v4 team sizes as the config draws them), the
scripted agent the only league member (no network routing), the rollout streamed into a
RolloutStore as in training. Works on trees before compile_shapes existed (they compile
dynamic). Appends one JSON row: power profile, load average, ms per tick.
"""

import argparse
import dataclasses
import json
import os
import subprocess
import time

import torch

import lean
from lean import game, model, scripted
from lean.learn import League, LearnConfig, Opponent, RolloutStore, collect, start


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--label", default="")
    parser.add_argument("--envs", type=int, default=LearnConfig().environments)
    parser.add_argument("--warm", type=int, default=2)
    parser.add_argument("--timed", type=int, default=4)
    parser.add_argument("--shapes", default="static")
    parser.add_argument("--deflector", default=None)
    arguments = parser.parse_args()
    device = torch.device("cuda")
    config = LearnConfig()
    if any(field.name == "compile_shapes" for field in dataclasses.fields(config)):
        config = dataclasses.replace(config, compile_shapes=arguments.shapes)
        shapes = arguments.shapes
    else:
        shapes = "dynamic (old tree)"
    game_config = game.GameConfig()
    if arguments.deflector is not None:
        game_config = dataclasses.replace(game_config, deflector=arguments.deflector)
    envs, ticks = arguments.envs, config.segment_ticks
    torch.manual_seed(0)
    generator = torch.Generator(device=device).manual_seed(0)
    policy = model.Policy(model.ModelConfig()).to(device)
    league = League(config, device)
    opponent = Opponent(scripted.ScriptedConfig(), league)
    state, carry = start(config, game_config, policy, opponent, envs, generator)
    store = RolloutStore(config.segments_per_update, config.stream_rollout)

    from lean.learn import compiled as compiled_module

    graphs = (
        compiled_module.graphs(shapes)
        if shapes in ("static", "dynamic")
        else (compiled_module.graphs())
    )
    events = {"act": [], "advance": []}
    recording = [False]
    originals = {name: getattr(graphs, name) for name in events}

    def timed(name):
        def call(*args, **keywords):
            if not recording[0]:
                return originals[name](*args, **keywords)
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            result = originals[name](*args, **keywords)
            end.record()
            events[name].append((begin, end))
            return result

        return call

    for name in events:
        setattr(graphs, name, timed(name))

    def segment(number):
        nonlocal state, carry
        torch.cuda.synchronize()
        began = time.perf_counter()
        result = collect(
            config,
            game_config,
            policy,
            opponent,
            state,
            carry,
            ticks,
            generator,
            compiled=True,
            store=store,
            segment=number % config.segments_per_update,
        )
        state, carry = result.state, result.carry
        torch.cuda.synchronize()
        return time.perf_counter() - began

    started = time.perf_counter()
    for number in range(arguments.warm):
        segment(number)
    warm_s = time.perf_counter() - started
    load_before = os.getloadavg()
    recording[0] = True
    seconds = [segment(arguments.warm + number) for number in range(arguments.timed)]
    torch.cuda.synchronize()
    per_call = {
        name: sum(b.elapsed_time(e) for b, e in pairs) / len(pairs)
        for name, pairs in events.items()
    }
    try:
        profile = subprocess.run(
            ["powerprofilesctl", "get"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        profile = "unknown"
    row = {
        "label": arguments.label,
        "lean": lean.__file__,
        "shapes": shapes,
        "deflector": game_config.deflector,
        "envs": envs,
        "ticks": ticks,
        "power_profile": profile,
        "load_average": load_before,
        "load_average_end": os.getloadavg(),
        "warm_s": round(warm_s, 1),
        "segment_ms_per_tick": [round(1e3 * s / ticks, 4) for s in seconds],
        "act_ms": round(per_call["act"], 4),
        "advance_ms": round(per_call["advance"], 4),
        "gpu": torch.cuda.get_device_name(),
    }
    print(json.dumps(row), flush=True)
    with open(arguments.out, "a") as handle:
        handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
