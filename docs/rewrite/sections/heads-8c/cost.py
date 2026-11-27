"""The geometry heads' cost (queue item 8c): the compiled minibatch loss (forward and backward)
at the recipe's micro-batch (64 views x 128 ticks, 4v4: 8 slots, 10 fields), per K.

    PYTHONPATH=. python docs/rewrite/sections/heads-8c/cost.py VARIANT [UPDATES]

VARIANT: off (no geometry heads), all (every pair: K 28 ship pairs, 80 ship-field pairs), or
a number K (K ship pairs and K ship-field pairs per view and tick). One fresh process per
variant with an empty inductor cache (cold compile). Collects one 128-tick segment of 128
environments eagerly (192 views: three micro-batches, one minibatch), then runs `update`
compiled (1 epoch): the first update compiles; then UPDATES (default 6) timed updates. Prints
one JSON line: geometry logits per view and tick, the first update's seconds, the steady
update's median milliseconds and per micro-batch, peak allocated memory in the steady
updates, the parameter count.
"""

import os
import tempfile
import time

PROCESS_START = time.perf_counter()
os.environ["TORCHINDUCTOR_CACHE_DIR"] = tempfile.mkdtemp(prefix="bnb-8c-cost-")
os.environ["TRITON_CACHE_DIR"] = os.path.join(os.environ["TORCHINDUCTOR_CACHE_DIR"], "triton")

import dataclasses  # noqa: E402
import json  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import sys  # noqa: E402

import torch  # noqa: E402

from lean import game, model, scripted  # noqa: E402
from lean.learn import LearnConfig, Opponent, collect, start  # noqa: E402
from lean.learn.store import RolloutStore  # noqa: E402
from lean.learn.training import make_optimiser, update  # noqa: E402


def main() -> None:
    variant = sys.argv[1]
    updates = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    device = torch.device("cuda")
    torch.manual_seed(0)
    generator = torch.Generator(device=device).manual_seed(0)
    envs, ticks = 128, 128
    pairs, field_pairs = 28, 80
    samples = {"all": (pairs, field_pairs), "off": (pairs, field_pairs)}.get(
        variant, (int(variant), int(variant))
    )
    config = dataclasses.replace(
        LearnConfig(),
        epochs=1,
        minibatches=1,
        pair_samples=samples[0],
        field_pair_samples=samples[1],
    )
    game_config = game.GameConfig(training_spawn=True)
    model_config = model.ModelConfig.for_arena(
        game_config.arena_radius, geometry_heads=variant != "off"
    )
    policy = model.Policy(model_config).to(device)
    opponent = Opponent(scripted.ScriptedConfig(), None)
    state, carry = start(config, game_config, policy, opponent, envs, generator)
    store = RolloutStore(1)
    began = time.perf_counter()
    collect(config, game_config, policy, opponent, state, carry, ticks, generator, store=store)
    torch.cuda.synchronize()
    collected = time.perf_counter() - began
    optimiser = make_optimiser(config, policy)
    torch.cuda.synchronize()
    began = time.perf_counter()
    update(config, policy, optimiser, store, generator=generator, compiled=True)
    torch.cuda.synchronize()
    first = time.perf_counter() - began
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(updates):
        torch.cuda.synchronize()
        began = time.perf_counter()
        update(config, policy, optimiser, store, generator=generator, compiled=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - began)
    steady = statistics.median(times)
    micro = -(-store.views // config.micro_batch_views)
    columns = model_config.pair_columns
    if variant == "off":
        logits = 0
    else:
        slots = state.exists.shape[1]
        one_order = sum(column.bins for column in columns if not column.both_orders)
        both = sum(column.bins for column in columns if column.both_orders)
        logits = (
            samples[0] * (one_order + 2 * both)
            + samples[1] * model_config.field_column.bins
            + slots * model_config.edge_column.bins
        )
    print(
        json.dumps(
            {
                "variant": variant,
                "ship_pairs": samples[0] if variant != "off" else 0,
                "field_pairs": samples[1] if variant != "off" else 0,
                "geometry_logits_per_view_tick": logits,
                "parameters": sum(p.numel() for p in policy.parameters()),
                "collect_s": round(collected, 1),
                "first_update_s": round(first, 1),
                "steady_update_ms": round(1000 * steady, 1),
                "steady_update_ms_all": [round(1000 * t, 1) for t in times],
                "micro_batches": micro,
                "per_micro_batch_ms": round(1000 * steady / micro, 1),
                "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 3),
                "process_s": round(time.perf_counter() - PROCESS_START, 1),
            }
        ),
        flush=True,
    )
    shutil.rmtree(os.environ["TORCHINDUCTOR_CACHE_DIR"], ignore_errors=True)


if __name__ == "__main__":
    main()
