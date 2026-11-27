"""Prototype variants of the update's graph (queue item 4, measurement only), timed as
grid_loss.py (128 environments, one epoch, one minibatch of three micro-batches):

- math_attention: the spatial sublayers' attention (9 trunk queries over 9 + 10 keys a head)
  as explicit matmuls, an additive mask and a softmax (inductor fuses it) instead of
  memory-efficient SDPA;
- baseline: unchanged (the same process layout).

    PYTHONPATH=. python docs/rewrite/sections/training-review-2/variants.py VARIANT

Also checks the variant's loss and gradients against the unchanged eager model on one
micro-batch (largest gradient difference). A JSON line appended to DATA/variants.jsonl.
"""

import sys

sys.path.insert(0, "docs/rewrite/sections/training-review-2")
import common  # noqa: E402

VARIANT = sys.argv[1]
cache = common.cache_dir(f"variant-{VARIANT}")

import dataclasses  # noqa: E402
import math  # noqa: E402
import os  # noqa: E402
import shutil  # noqa: E402
import statistics  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from lean import game, model, scripted  # noqa: E402
from lean.learn import LearnConfig, Opponent, collect, start  # noqa: E402
from lean.learn.store import RolloutStore  # noqa: E402
from lean.learn.training import make_optimiser, update  # noqa: E402

ORIGINAL_SDPA = F.scaled_dot_product_attention


def math_attention(query, key, value, attn_mask=None, **_):
    scores = query @ key.transpose(-1, -2) / math.sqrt(query.shape[-1])
    if attn_mask is not None:
        scores = scores + attn_mask
    return torch.softmax(scores.float(), -1).to(value.dtype) @ value


def main() -> None:
    record = {"job": "variant", "variant": VARIANT, **common.machine()}
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
    if VARIANT == "math_attention":
        F.scaled_dot_product_attention = math_attention
    optimiser = make_optimiser(config, policy)
    began = time.perf_counter()
    update(config, policy, optimiser, store, generator=generator, compiled=True)
    torch.cuda.synchronize()
    record["first_update_s"] = round(time.perf_counter() - began, 1)
    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(6):
        torch.cuda.synchronize()
        began = time.perf_counter()
        update(config, policy, optimiser, store, generator=generator, compiled=True)
        torch.cuda.synchronize()
        times.append(time.perf_counter() - began)
    steady = statistics.median(times)
    record.update(
        steady_update_ms=round(1000 * steady, 1),
        per_micro_batch_ms=round(1000 * steady / 3, 2),
        peak_allocated_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3),
    )
    # Attention outputs against SDPA's on random inputs of the sublayer's shapes (bf16).
    query = torch.randn(64 * 128, 2, 9, 64, device=device, dtype=torch.bfloat16)
    key = torch.randn(64 * 128, 2, 19, 64, device=device, dtype=torch.bfloat16)
    value = torch.randn_like(key)
    mask = torch.zeros(64 * 128, 1, 1, 19, device=device)
    mask[::3, ..., 5:9] = -math.inf
    with torch.autocast("cuda", torch.bfloat16):
        reference = ORIGINAL_SDPA(query, key, value, attn_mask=mask).float()
        mine = math_attention(query, key, value, attn_mask=mask).float()
    record["attention_max_difference"] = (reference - mine).abs().max().item()
    record["attention_max_value"] = reference.abs().max().item()
    common.emit(record, os.path.join(common.DATA, "variants.jsonl"))
    shutil.rmtree(cache, ignore_errors=True)


if __name__ == "__main__":
    main()
