"""Yemong's speed under the blind race's protocol: training is `unroll` forward and backward
over 128 steps (V = 16 and 64 views); the rollout is one `step` for 2048 views.

Synthetic segments (`bench.yemong_synthetic`), the loss every head's mean summed (the shot
trajectory head at 4 ages per ship-step), a reset mid-segment, a nonzero carry, the
benchmark lock held, the median of 20 CUDA-synchronised repetitions after warm-up. bf16 is
the model's `bfloat16_matmuls`; fp32 is strict (no TF32). `--plain` turns every kernel off;
`--kernels` turns on only the named ones. Each row records the commit, the kernels'
switches and the load average:

    uv run --no-sync python -m bench.yemong_speed [--only training|rollout] [--views 16]
        [--modes compiled cudagraphs] [--precisions bfloat16] [--plain | --kernels scan]
        [--label text]
"""

import argparse
import dataclasses
import json
import os
import pathlib
import statistics
import subprocess
import time

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")  # as the training entry point

import torch  # noqa: E402
from torch.utils import _pytree  # noqa: E402

from bench.lock import bench_lock  # noqa: E402
from bench.yemong_synthetic import random_segment, time_slice  # noqa: E402
from bnb.contracts.config import GameConfig  # noqa: E402
from bnb.game import api  # noqa: E402
from bnb.models.yemong import kernels  # noqa: E402
from bnb.models.yemong.model import Yemong, YemongConfig, yemong_build  # noqa: E402

SHIPS, FIELDS, ZONES, STEPS, SHOT_AGES, OWN = 10, 10, 5, 128, 4, 5
HEADS = (
    "policy_logits",
    "enemy_action_logits",
    "next_state_logits",
    "value_logits",
    "outcome_logits",
    "density_log_rates",
    "shot_outcome_logits",
)


def make_model(precision):
    """A fresh Yemong (seed 0) on the GPU for the synthetic segments' game, with bf16 matmuls when
    precision is bfloat16."""
    torch.manual_seed(0)
    config = YemongConfig(bfloat16_matmuls=precision == "bfloat16")
    rules = api.rules_of(GameConfig(5, 10, 2600.0))  # 5v5, 10 fields, the segments' arena
    return Yemong(yemong_build(config, rules, api.ship_types())).cuda()


def nonzero_carry(model, views):
    """The model's initial carry for views views with a random normal state and convolution buffer,
    so no cell times an all-zero recurrence."""
    carry = model.initial_carry(views, OWN, SHIPS, device="cuda")
    return dataclasses.replace(
        carry,
        state=torch.randn_like(carry.state),
        convolution_buffer=torch.randn_like(carry.convolution_buffer),
    )


def timed(function, repetitions, warmup):
    """Call function warmup times, then time repetitions CUDA-synchronised calls: (the median in ms,
    every time in ms)."""
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    times = []
    for _ in range(repetitions):
        start = time.perf_counter()
        function()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e3)
    return statistics.median(times), times


@torch.no_grad()
def record_replay(model, carry, obs, reset):
    """The replays the model's step records over the segment obs (L, R, ...) from carry, stacked to
    (L, R, ...)."""
    replays = []
    for t in range(reset.shape[0]):
        keys = torch.arange(reset.shape[1]) + t * reset.shape[1]
        decision = model.step(carry, time_slice(obs, t), reset[t], keys)
        carry = decision.carry
        replays.append(decision.replay)
    leaves, spec = zip(*(_pytree.tree_flatten(r) for r in replays))
    return _pytree.tree_unflatten([torch.stack(group) for group in zip(*leaves)], spec[0])


def training_cell(model, views, mode, repetitions, warmup):
    """Time one training iteration (unroll forward and backward) over a synthetic segment of STEPS
    steps and views views in mode (eager, compiled or cudagraphs): the median ms, view ticks per s
    and peak memory (GiB)."""
    obs, reset = random_segment(
        STEPS, views, SHIPS, FIELDS, ZONES, device="cuda", seed=views, reset_steps=(STEPS // 2,)
    )
    start = nonzero_carry(model, views)
    replay = record_replay(model, start, obs, reset)
    ages = torch.randint(1, 13, (STEPS, views, SHIPS, SHOT_AGES), device="cuda")

    def loss_of(carry, obs, replay, reset, ages):
        out = model.unroll(carry, obs, replay, reset)
        total = sum(getattr(out, key).float().mean() for key in HEADS)
        return total + model.shot_trajectory(out.shot_latents, ages).float().mean()

    compute = {
        "eager": loss_of,
        "compiled": torch.compile(loss_of, fullgraph=True),
        "cudagraphs": torch.compile(loss_of, fullgraph=True, mode="reduce-overhead"),
    }[mode]

    def iteration():
        model.zero_grad(set_to_none=True)
        compute(start, obs, replay, reset, ages).backward()

    torch.cuda.reset_peak_memory_stats()
    median, times = timed(iteration, repetitions, warmup)
    return {
        "ms": median,
        "view_ticks_per_s": views * STEPS / median * 1e3,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30,
    }


def rollout_cell(model, views, mode, repetitions, warmup):
    """Time one step for views views in mode (eager, compiled or cudagraphs): the median ms, view
    steps per s and peak memory (GiB)."""
    obs, _ = random_segment(2, views, SHIPS, FIELDS, ZONES, device="cuda", seed=1)
    single = time_slice(obs, 1)
    reset = torch.zeros(views, dtype=torch.bool, device="cuda")
    reset[: views // 64] = True
    step = {
        "eager": model.step,
        "compiled": torch.compile(model.step, fullgraph=True),
        "cudagraphs": torch.compile(model.step, fullgraph=True, mode="reduce-overhead"),
    }[mode]
    state = {"carry": nonzero_carry(model, views)}

    @torch.no_grad()
    def iteration():
        carry = step(state["carry"], single, reset).carry
        state["carry"] = _pytree.tree_map(torch.clone, carry) if mode == "cudagraphs" else carry

    torch.cuda.reset_peak_memory_stats()
    median, times = timed(iteration, repetitions, warmup)
    return {
        "ms": median,
        "view_steps_per_s": views / median * 1e3,
        "peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30,
    }


def _commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"],
        capture_output=True,
        text=True,
        cwd=pathlib.Path(__file__).parent,
    ).stdout.strip()


def main():
    """Run the selected cells and append each row as a JSON line to --out."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("training", "rollout"))
    parser.add_argument("--views", type=int)
    parser.add_argument("--modes", nargs="+", default=("eager", "compiled", "cudagraphs"))
    parser.add_argument("--precisions", nargs="+", default=("float32", "bfloat16"))
    parser.add_argument("--plain", action="store_true", help="every kernel switched off")
    parser.add_argument("--kernels", nargs="+", help="only these kernels switched on")
    parser.add_argument("--label", default="")
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument(
        "--out", default=str(pathlib.Path(__file__).parent / "yemong_results.jsonl")
    )
    arguments = parser.parse_args()
    torch.set_float32_matmul_precision("highest")
    if arguments.plain or arguments.kernels:
        kernels.switch_all(False)
    for name in arguments.kernels or ():
        setattr(kernels.switches, name, True)
    commit = _commit()
    cells = [("training", v) for v in (16, 64)] + [("rollout", 2048)]
    with bench_lock():
        for kind, views in cells:
            if arguments.only and kind != arguments.only:
                continue
            if arguments.views and views != arguments.views:
                continue
            for precision in arguments.precisions:
                for mode in arguments.modes:
                    model = make_model(precision)
                    cell = training_cell if kind == "training" else rollout_cell
                    try:
                        result = cell(model, views, mode, arguments.repetitions, 5)
                    except torch.OutOfMemoryError:
                        result = {"ms": None, "error": "out of memory"}
                    finally:
                        del model
                        torch._dynamo.reset()
                        torch.cuda.empty_cache()
                    row = {
                        "commit": commit,
                        "label": arguments.label,
                        "kind": kind,
                        "views": views,
                        "precision": precision,
                        "mode": mode,
                        "kernels": dataclasses.asdict(kernels.switches),
                        "load_average": os.getloadavg()[0],
                        **result,
                    }
                    print(json.dumps(row), flush=True)
                    with open(arguments.out, "a") as handle:
                        handle.write(json.dumps(row) + "\n")


if __name__ == "__main__":
    main()
