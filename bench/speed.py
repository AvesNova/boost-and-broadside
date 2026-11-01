"""Yemong speed: training (unroll forward + backward) and rollouts (step forward).

Every timing holds the benchmark lock, excludes warm-up (and compilation), synchronises CUDA
and reports the median of at least 20 repetitions. Results go to bench/results.jsonl, one
JSON line per cell with the commit.

    uv run --no-sync python -m bench.speed                    # every cell
    uv run --no-sync python -m bench.speed --only training --views 64 --precision bfloat16

Training: `unroll` over V views x L = 128 steps, 5v5 (N = 10, M = 10, Z = 5), the default
configuration, from a nonzero carry with one reset mid-segment (step 64, every view); the
loss is the mean of every head's raw output, summed (the shot trajectory head queried at 4
ages per ship-step: a ship firing at the gun's limit has 12 / 3 = 4 bullets in flight).
Rollouts: `step` for V = 2048 views under no_grad, the carry fed back each step.
"""

import argparse
import contextlib
import datetime
import json
import os
import statistics
import subprocess
import time

import torch

from bench.lock import bench_lock
from bnb.models.swift.model import Replay, Yemong, YemongConfig
from bnb.models.swift.synthetic import random_segment, time_slice

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results.jsonl")
SHIPS, FIELDS, ZONES, STEPS = 10, 10, 5, 128
SHOT_AGES = 4
HEAD_KEYS = ("action_logits", "enemy_action_logits", "next_state_logits", "value_logits",
             "outcome_logits", "density_log_rates", "shot_outcome_logits",
             "shot_trajectory_logits")


def commit() -> str:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=HERE,
                                      text=True).strip()
        dirty = subprocess.call(["git", "diff", "--quiet", "HEAD", "--", "src", "bench"],
                                cwd=HERE)
        return sha + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def autocast(precision):
    if precision == "bfloat16":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def timed(function, repetitions: int, warmup: int):
    """Median and all times (ms) of function(), CUDA-synchronised, warm-up excluded."""
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


def make_model():
    torch.manual_seed(0)
    model = Yemong(YemongConfig()).cuda()
    with torch.no_grad():
        model.next_state_head.last.weight.normal_(0, 0.02)
    return model


def nonzero_carry(model, views):
    carry = model.initial_carry(views, SHIPS, device="cuda")
    return carry._replace(state=torch.randn_like(carry.state),
                          convolution_buffer=torch.randn_like(carry.convolution_buffer))


@torch.no_grad()
def record_replay(model, carry, obs, reset):
    """The replay a rollout would have stored (eager steps)."""
    leaves = []
    for t in range(reset.shape[0]):
        carry, out = model.step(carry, time_slice(obs, t), reset[t])
        leaves.append(out["replay"])
    return Replay(*(torch.stack([r[i] for r in leaves]) for i in range(6)))


def training_cell(model, views, precision, mode, repetitions, warmup):
    obs, reset = random_segment(STEPS, views, SHIPS, FIELDS, ZONES, device="cuda", seed=views,
                                reset_steps=(STEPS // 2,))
    start = nonzero_carry(model, views)
    replay = record_replay(model, start, obs, reset)
    ages = torch.randint(1, 13, (STEPS, views, SHIPS, SHOT_AGES), device="cuda")

    def loss_of(carry, obs, replay, reset, ages):
        with autocast(precision):
            out = model.unroll(carry, obs, replay, reset, ages)
            return sum(out[key].float().mean() for key in HEAD_KEYS)

    if mode == "eager":
        compute = loss_of
    elif mode == "compiled":
        compute = torch.compile(loss_of, fullgraph=True)
    else:  # compiled with CUDA graphs
        compute = torch.compile(loss_of, fullgraph=True, mode="reduce-overhead")

    def iteration():
        model.zero_grad(set_to_none=True)
        loss = compute(start, obs, replay, reset, ages)
        loss.backward()

    torch.cuda.reset_peak_memory_stats()
    median, times = timed(iteration, repetitions, warmup)
    return {"ms": median, "view_ticks_per_s": views * STEPS / median * 1e3,
            "peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30, "times_ms": times}


def rollout_cell(model, views, precision, mode, repetitions, warmup):
    obs, reset = random_segment(2, views, SHIPS, FIELDS, ZONES, device="cuda", seed=1)
    single = time_slice(obs, 1)
    reset = torch.zeros(views, dtype=torch.bool, device="cuda")
    reset[: views // 64] = True  # a few environments reset this tick

    @torch.no_grad()
    def step(carry, obs, reset):
        with autocast(precision):
            return model.step(carry, obs, reset)

    carry = nonzero_carry(model, views)
    if mode == "eager":
        compute = step
    elif mode == "compiled":
        compute = torch.compile(step, fullgraph=True)
    else:
        compute = torch.compile(step, fullgraph=True, mode="reduce-overhead")
    state = {"carry": carry}

    def iteration():
        carry, out = compute(state["carry"], single, reset)
        if mode == "cudagraphs":  # outputs live in the graph's pool: keep the carry outside
            carry = type(carry)(*(leaf.clone() for leaf in carry))
        state["carry"] = carry

    torch.cuda.reset_peak_memory_stats()
    median, times = timed(iteration, repetitions, warmup)
    return {"ms": median, "view_steps_per_s": views / median * 1e3,
            "peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30, "times_ms": times}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("training", "rollout"), default=None)
    parser.add_argument("--views", type=int, nargs="*", default=None)
    parser.add_argument("--precision", nargs="*", default=["float32", "bfloat16"])
    parser.add_argument("--mode", nargs="*", default=["eager", "compiled", "cudagraphs"])
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--label", default="")
    parser.add_argument("--no-write", action="store_true")
    arguments = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cells = []
    if arguments.only in (None, "training"):
        cells += [("training", v) for v in (arguments.views or [16, 64])]
    if arguments.only in (None, "rollout"):
        cells += [("rollout", v) for v in (arguments.views or [2048])]
    sha = commit()
    with bench_lock():
        model = make_model()
        for kind, views in cells:
            for precision in arguments.precision:
                for mode in arguments.mode:
                    torch._dynamo.reset()
                    function = training_cell if kind == "training" else rollout_cell
                    result = function(model, views, precision, mode, arguments.repetitions,
                                      arguments.warmup)
                    record = {"kind": kind, "views": views, "steps": STEPS if kind ==
                              "training" else 1, "precision": precision, "mode": mode,
                              "commit": sha, "label": arguments.label,
                              "time": datetime.datetime.now().isoformat(timespec="seconds"),
                              "gpu": torch.cuda.get_device_name()} | result
                    rate = record.get("view_ticks_per_s", record.get("view_steps_per_s"))
                    print(f"{kind:8s} V={views:5d} {precision:8s} {mode:10s} "
                          f"{record['ms']:9.3f} ms  {rate:12.0f} /s  "
                          f"{record['peak_memory_gb']:.2f} GB", flush=True)
                    if not arguments.no_write:
                        with open(RESULTS, "a") as handle:
                            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
