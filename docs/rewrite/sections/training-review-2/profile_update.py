"""Profile a steady compiled update (queue item 4): the recipe's micro-batch (64 views x 128
ticks, 4v4, 10 fields, bf16, geometry heads K = 8), one epoch of one minibatch of three
micro-batches (as heads-8c/cost.py), after one compiling and two steady updates.

    PYTHONPATH=. python docs/rewrite/sections/training-review-2/profile_update.py [ENVS] [off]

Prints one JSON line (and appends it to DATA/profile_update.jsonl): steady update ms, kernel
families, GPU busy share, host runtime calls, memcpy kinds.
"""

import sys

sys.path.insert(0, "docs/rewrite/sections/training-review-2")
import common  # noqa: E402

common.cache_dir("profile")

import os  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from lean import game, model, scripted  # noqa: E402
from lean.learn import LearnConfig, Opponent, collect, start  # noqa: E402
from lean.learn.store import RolloutStore  # noqa: E402
from lean.learn.training import make_optimiser, update  # noqa: E402


def main() -> None:
    envs = int(sys.argv[1]) if len(sys.argv) > 1 else 128
    heads = "off" not in sys.argv
    record = {"job": "profile_update", "envs": envs, "geometry_heads": heads, **common.machine()}
    device = torch.device("cuda")
    torch.manual_seed(0)
    generator = torch.Generator(device=device).manual_seed(0)
    import dataclasses

    config = dataclasses.replace(LearnConfig(), epochs=1, minibatches=1)
    game_config = game.GameConfig(training_spawn=True)
    model_config = model.ModelConfig.for_arena(game_config.arena_radius, geometry_heads=heads)
    policy = model.Policy(model_config).to(device)
    opponent = Opponent(scripted.ScriptedConfig(), None)
    state, carry = start(config, game_config, policy, opponent, envs, generator)
    store = RolloutStore(1)
    collect(config, game_config, policy, opponent, state, carry, 128, generator, store=store)
    optimiser = make_optimiser(config, policy)
    began = time.perf_counter()
    update(config, policy, optimiser, store, generator=generator, compiled=True)
    torch.cuda.synchronize()
    record["first_update_s"] = round(time.perf_counter() - began, 1)
    times = []
    for _ in range(3):
        torch.cuda.synchronize()
        began = time.perf_counter()
        update(config, policy, optimiser, store, generator=generator, compiled=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - began)
    micro = -(-store.views // config.micro_batch_views)
    record["micro_batches"] = micro
    record["steady_update_ms"] = round(1000 * statistics.median(times), 1)
    record["per_micro_batch_ms"] = round(1000 * statistics.median(times) / micro, 1)
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        began = time.perf_counter()
        update(config, policy, optimiser, store, generator=generator, compiled=True)
        host = time.perf_counter() - began
        torch.cuda.synchronize()
    record["profiled_update_host_s_before_sync"] = round(host, 4)
    path = os.path.join(common.DATA, f"trace_update{'' if heads else '_off'}.json")
    prof.export_chrome_trace(path)
    record.update(common.read_trace(path))
    record["peak_allocated_gib"] = round(torch.cuda.max_memory_allocated() / 2**30, 3)
    common.emit(record, os.path.join(common.DATA, "profile_update.jsonl"))


if __name__ == "__main__":
    main()
