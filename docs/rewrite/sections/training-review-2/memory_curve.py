"""Micro-batch size x activation memory x speed (queue item 4): the compiled update (as
heads-8c/cost.py: one epoch, one minibatch) at micro-batch B views, with

- plain: the trainer's graph;
- checkpoint: every residual sublayer (2 temporal, 4 spatial) under non-reentrant
  torch.utils.checkpoint (activation checkpointing per sublayer: only its input is kept,
  the rest recomputed in the backward);
- budget<x>: the partitioner's activation memory budget (torch._functorch.config
  .activation_memory_budget = x: inductor picks which activations to recompute, cheap
  element-wise ones first, to keep at most x of the saved bytes).

    PYTHONPATH=. python docs/rewrite/sections/training-review-2/memory_curve.py MODE B [ENVS]

ENVS (default 256, 384 views). Prints one JSON line (appended to DATA/memory_curve.jsonl):
per micro-batch ms, ms per 64 views (the recipe's unit), peak allocated, the first update's
seconds (cold compile in this variant's own cache).
"""

import sys

sys.path.insert(0, "docs/rewrite/sections/training-review-2")
import common  # noqa: E402

MODE, B = sys.argv[1], int(sys.argv[2])
ENVS = int(sys.argv[3]) if len(sys.argv) > 3 else 256
cache = common.cache_dir(f"memory-{MODE}-{B}")

import dataclasses  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

from lean import game, model, scripted  # noqa: E402
from lean.learn import LearnConfig, Opponent, collect, start  # noqa: E402
from lean.learn.store import RolloutStore  # noqa: E402
from lean.learn.training import make_optimiser, update  # noqa: E402


def checkpointed(module) -> None:
    forward = module.forward

    def wrapped(*arguments):
        return checkpoint(forward, *arguments, use_reentrant=False)

    module.forward = wrapped


def main() -> None:
    record = {"job": "memory_curve", "mode": MODE, "micro_batch_views": B, "envs": ENVS}
    record.update(common.machine())
    if MODE.startswith("budget"):
        torch._functorch.config.activation_memory_budget = float(MODE[len("budget") :])
    device = torch.device("cuda")
    torch.manual_seed(0)
    generator = torch.Generator(device=device).manual_seed(0)
    config = dataclasses.replace(LearnConfig(), epochs=1, minibatches=1, micro_batch_views=B)
    game_config = game.GameConfig(training_spawn=True)
    policy = model.Policy(model.ModelConfig.for_arena(game_config.arena_radius)).to(device)
    if MODE == "checkpoint":
        for block in policy.blocks:
            checkpointed(block.temporal)
            for spatial in block.spatial:
                checkpointed(spatial)
    opponent = Opponent(scripted.ScriptedConfig(), None)
    state, carry = start(config, game_config, policy, opponent, ENVS, generator)
    store = RolloutStore(1)
    collect(config, game_config, policy, opponent, state, carry, 128, generator, store=store)
    optimiser = make_optimiser(config, policy)
    try:
        began = time.perf_counter()
        update(config, policy, optimiser, store, generator=generator, compiled=True)
        torch.cuda.synchronize()
        record["first_update_s"] = round(time.perf_counter() - began, 1)
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(4):
            torch.cuda.synchronize()
            began = time.perf_counter()
            metrics = update(config, policy, optimiser, store, generator=generator, compiled=True)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - began)
        micro = -(-store.views // B)
        steady = statistics.median(times)
        record.update(
            micro_batches=micro,
            steady_update_ms=round(1000 * steady, 1),
            per_micro_batch_ms=round(1000 * steady / micro, 2),
            ms_per_64_views=round(1000 * steady / micro * 64 / B, 2),
            peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3),
            peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 3),
            loss=metrics["loss"],
        )
    except torch.OutOfMemoryError as error:
        record["error"] = "OOM: " + str(error).split("\n")[0][:200]
    common.emit(record, os.path.join(common.DATA, "memory_curve.jsonl"))
    shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    main()
