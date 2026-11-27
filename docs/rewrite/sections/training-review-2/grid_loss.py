"""Compiler-mode grid, the update's graph (queue item 4): minibatch_loss (forward and backward)
under torch.compile mode MODE (static shapes), the update as heads-8c/cost.py (128
environments, one epoch, one minibatch of three micro-batches of 64 views x 128 ticks).

    PYTHONPATH=. python docs/rewrite/sections/training-review-2/grid_loss.py MODE cold|warm

The cache is this mode's own (DATA/cache/loss-MODE): "cold" empties it first, "warm" reuses
what the cold run left. Prints one JSON line (appended to DATA/grid.jsonl): the first update's
seconds (compile included), the steady update and per micro-batch, peak memory, the loss.
"""

import sys

sys.path.insert(0, "docs/rewrite/sections/training-review-2")
import common  # noqa: E402

MODE, TEMPERATURE = sys.argv[1], sys.argv[2]
if MODE == "reduce-overhead":
    import os as _os

    _os.environ["TORCH_LOGS"] = "perf_hints"  # why a graph skips CUDA graphs
import os  # noqa: E402
import shutil  # noqa: E402

if TEMPERATURE == "cold":
    shutil.rmtree(os.path.join(common.DATA, "cache", f"loss-{MODE}"), ignore_errors=True)
common.cache_dir(f"loss-{MODE}")

import dataclasses  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402

from lean import game, model, scripted  # noqa: E402
from lean.learn import LearnConfig, Opponent, collect, start  # noqa: E402
from lean.learn.store import RolloutStore  # noqa: E402
from lean.learn.training import make_optimiser, update  # noqa: E402


def main() -> None:
    started = time.perf_counter()
    record = {"job": "grid_loss", "graph": "minibatch_loss", "mode": MODE, "cache": TEMPERATURE}
    record.update(common.machine())
    common.patch_graphs({"minibatch_loss": MODE})
    device = torch.device("cuda")
    torch.manual_seed(0)
    generator = torch.Generator(device=device).manual_seed(0)
    config = dataclasses.replace(LearnConfig(), epochs=1, minibatches=1)
    game_config = game.GameConfig(training_spawn=True)
    policy = model.Policy(model.ModelConfig.for_arena(game_config.arena_radius)).to(device)
    opponent = Opponent(scripted.ScriptedConfig(), None)
    state, carry = start(config, game_config, policy, opponent, 128, generator)
    store = RolloutStore(1)
    collect(config, game_config, policy, opponent, state, carry, 128, generator, store=store)
    optimiser = make_optimiser(config, policy)
    began = time.perf_counter()
    metrics = update(config, policy, optimiser, store, generator=generator, compiled=True)
    torch.cuda.synchronize()
    record["first_update_s"] = round(time.perf_counter() - began, 1)
    record["first_loss"] = metrics["loss"]
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(6):
        torch.cuda.synchronize()
        began = time.perf_counter()
        update(config, policy, optimiser, store, generator=generator, compiled=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - began)
    micro = -(-store.views // config.micro_batch_views)
    steady = statistics.median(times)
    record.update(
        steady_update_ms=round(1000 * steady, 1),
        steady_all=[round(1000 * t, 1) for t in times],
        per_micro_batch_ms=round(1000 * steady / micro, 2),
        peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3),
        process_s=round(time.perf_counter() - started, 1),
    )
    common.emit(record, os.path.join(common.DATA, "grid.jsonl"))


if __name__ == "__main__":
    main()
