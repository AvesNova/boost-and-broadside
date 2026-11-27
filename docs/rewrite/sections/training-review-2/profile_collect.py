"""Profile a steady compiled rollout segment (queue item 4) at the recipe's size (2,048
environments, 4v4, the training game), the trainer's stage set-up (lean.train.new_stage: its
league, BC or RL with the EMA playing), streamed into a store.

    PYTHONPATH=. python docs/rewrite/sections/training-review-2/profile_collect.py bc|rl [MODE] [cold] [--no-profile] [--envs N]

MODE (default "default"): the torch.compile mode of act and advance (the compiler grid).
One compiling segment of 128 ticks, one timed steady segment of 128 ticks, then
torch.profiler over a segment of 16 ticks. Prints one JSON line (appended to
DATA/profile_collect.jsonl).
"""

import sys

sys.path.insert(0, "docs/rewrite/sections/training-review-2")
import common  # noqa: E402

STAGE = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else "default"
if MODE == "reduce-overhead":
    import os as _os

    _os.environ["TORCH_LOGS"] = "perf_hints"  # why a graph skips CUDA graphs
PROFILE = MODE == "default" and "--no-profile" not in sys.argv
TEMPERATURE = "cold" if "cold" in sys.argv else "warm"
if TEMPERATURE == "cold":
    import shutil

    shutil.rmtree(f"{common.DATA}/cache/collect-{STAGE}-{MODE}" + ("-small" if "--envs" in sys.argv else ""), ignore_errors=True)
common.cache_dir(f"collect-{STAGE}-{MODE}" + ("-small" if "--envs" in sys.argv else ""))

import os  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from lean import game, model  # noqa: E402
from lean.learn import LearnConfig, Opponent, collect  # noqa: E402
from lean.learn.store import RolloutStore  # noqa: E402
from lean.train import new_stage  # noqa: E402


def main() -> None:
    started = time.perf_counter()
    record = {"job": "profile_collect", "stage": STAGE, "mode": MODE, "cache": TEMPERATURE, **common.machine()}
    if MODE != "default":
        common.patch_graphs({"act": MODE, "advance": MODE})
    device = torch.device("cuda")
    torch.manual_seed(0)
    generator = torch.Generator(device=device).manual_seed(0)
    config = LearnConfig()
    envs = int(sys.argv[sys.argv.index("--envs") + 1]) if "--envs" in sys.argv else config.environments
    record["envs"] = envs
    game_config = game.GameConfig(training_spawn=True)
    policy = model.Policy(model.ModelConfig.for_arena(game_config.arena_radius)).to(device)
    stage = new_stage(STAGE, 1, config, game_config, policy, envs, generator)
    from lean import scripted

    opponent = Opponent(scripted.ScriptedConfig(), stage.league)
    store = RolloutStore(1)
    state, carry = stage.state, stage.carry

    def run(ticks, target):
        nonlocal state, carry
        rollout = collect(
            config, game_config, policy, opponent, state, carry, ticks, generator,
            compiled=True, store=target,
        )
        state, carry = rollout.state, rollout.carry

    began = time.perf_counter()
    run(128, store)
    torch.cuda.synchronize()
    record["first_segment_s"] = round(time.perf_counter() - began, 1)
    began = time.perf_counter()
    run(128, store)
    torch.cuda.synchronize()
    seconds = time.perf_counter() - began
    record["steady_segment_s"] = round(seconds, 3)
    record["steady_ms_per_tick"] = round(1000 * seconds / 128, 2)
    record["steps_per_second_collect"] = round(envs * 128 / seconds)
    if PROFILE:
        small = RolloutStore(1)
        run(16, small)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            began = time.perf_counter()
            run(16, small)
            record["profiled_host_s_before_sync"] = round(time.perf_counter() - began, 4)
            torch.cuda.synchronize()
            record["profiled_s"] = round(time.perf_counter() - began, 4)
        path = os.path.join(common.DATA, f"trace_collect_{STAGE}_{envs}.json")
        prof.export_chrome_trace(path)
        trace = common.read_trace(path)
        regions = {}
        for name, duration, _ in trace.pop("compiled_regions"):
            entry = regions.setdefault(name, [0, 0.0])
            entry[0] += 1
            entry[1] += duration
        trace["regions_host_ms"] = {k: (v[0], round(v[1] / v[0], 3)) for k, v in regions.items()}
        record.update(trace)
    record["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    record["process_s"] = round(time.perf_counter() - started, 1)
    common.emit(record, os.path.join(common.DATA, "profile_collect.jsonl"))


if __name__ == "__main__":
    main()
