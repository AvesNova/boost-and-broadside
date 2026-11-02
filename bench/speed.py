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
from bnb.models.swift import network
from bnb.models.swift.model import Replay, Yemong, YemongConfig
from bnb.models.swift.runtime import CapturedStep
from bnb.models.swift.synthetic import random_segment, time_slice

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results.jsonl")
SHIPS, FIELDS, ZONES, STEPS = 10, 10, 5, 128
SHOT_AGES = 4
COMPILE_OPTIONS = {"autotune": False}
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
        compute = torch.compile(loss_of, fullgraph=True, mode="max-autotune-no-cudagraphs"
                                if COMPILE_OPTIONS["autotune"] else None)
    else:  # compiled with CUDA graphs
        compute = torch.compile(loss_of, fullgraph=True, mode="max-autotune"
                                if COMPILE_OPTIONS["autotune"] else "reduce-overhead")

    def iteration():
        model.zero_grad(set_to_none=True)
        loss = compute(start, obs, replay, reset, ages)
        loss.backward()

    torch.cuda.reset_peak_memory_stats()
    median, times = timed(iteration, repetitions, warmup)
    return {"ms": median, "view_ticks_per_s": views * STEPS / median * 1e3,
            "peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30, "times_ms": times}


def rollout_cell(model, views, precision, mode, repetitions, warmup):
    """eager: `step` with the carry fed back. compiled: `CapturedStep` without the graph (the
    compiled step writes the carry in place). cudagraphs: `CapturedStep` replayed as one CUDA
    graph. The observation is already in the static buffer, as the environment writes it."""
    obs, _ = random_segment(2, views, SHIPS, FIELDS, ZONES, device="cuda", seed=1)
    single = time_slice(obs, 1)
    reset = torch.zeros(views, dtype=torch.bool, device="cuda")
    reset[: views // 64] = True  # a few environments reset this tick

    if mode == "eager":
        state = {"carry": nonzero_carry(model, views)}

        @torch.no_grad()
        def iteration():
            with autocast(precision):
                state["carry"], _ = model.step(state["carry"], single, reset)
    else:
        captured = CapturedStep(model, views, single, precision, compile=True,
                                graph=mode == "cudagraphs", autotune=COMPILE_OPTIONS["autotune"])
        captured.reset.copy_(reset)
        iteration = captured.replay

    torch.cuda.reset_peak_memory_stats()
    median, times = timed(iteration, repetitions, warmup)
    return {"ms": median, "view_steps_per_s": views / median * 1e3,
            "peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30, "times_ms": times}


RECOMPUTE_LADDER = ("none", "feedforward", "sublayers")


def run_with_retries(function, model, *arguments, attempts: int = 4):
    """A cell. Training that runs out of memory retries recomputing more in the backward: the
    MLPs ("feedforward"), then every trunk sublayer ("sublayers"), recorded as `recompute`;
    any cell that still does (the GPU is shared) waits and retries, then is recorded as an
    error."""
    for attempt in range(attempts):
        recompute = (RECOMPUTE_LADDER[min(attempt, 2)] if function is training_cell
                     else "none")
        model.trunk.recompute = recompute
        try:
            return function(model, *arguments) | {"recompute": recompute}
        except torch.OutOfMemoryError as error:
            message = str(error).split(".")[0]
            print(f"[bench] out of memory (attempt {attempt + 1}, recompute {recompute}): "
                  f"{message}", flush=True)
        finally:
            model.trunk.recompute = "none"
            model.zero_grad(set_to_none=True)
            torch._dynamo.reset()
            torch.cuda.empty_cache()
        if not (function is training_cell and attempt < 2):
            time.sleep(30)
    return {"ms": None, "error": "out of memory"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=("training", "rollout"), default=None)
    parser.add_argument("--views", type=int, nargs="*", default=None)
    parser.add_argument("--precision", nargs="*", default=["float32", "bfloat16"])
    parser.add_argument("--mode", nargs="*", default=["eager", "compiled", "cudagraphs"])
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--label", default="")
    parser.add_argument("--attention", default=None, choices=("sdpa", "matmul", "broadcast"))
    parser.add_argument("--autotune", action="store_true",
                        help="compile with max-autotune (Triton GEMM templates, fused epilogues)")
    parser.add_argument("--no-write", action="store_true")
    arguments = parser.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cells = []
    if arguments.only in (None, "training"):
        cells += [("training", v) for v in (arguments.views or [16, 64])]
    if arguments.only in (None, "rollout"):
        cells += [("rollout", v) for v in (arguments.views or [2048])]
    if arguments.attention:
        network.ATTENTION["kernel"] = arguments.attention
    COMPILE_OPTIONS["autotune"] = arguments.autotune
    sha = commit()
    with bench_lock():
        model = make_model()
        for kind, views in cells:
            for precision in arguments.precision:
                for mode in arguments.mode:
                    torch._dynamo.reset()
                    # tfloat32 (optional): float32 storage, TF32 tensor-core matmuls
                    torch.backends.cuda.matmul.allow_tf32 = precision == "tfloat32"
                    torch.backends.cudnn.allow_tf32 = precision == "tfloat32"
                    function = training_cell if kind == "training" else rollout_cell
                    result = run_with_retries(function, model, views, precision, mode,
                                              arguments.repetitions, arguments.warmup)
                    record = {"kind": kind, "views": views, "steps": STEPS if kind ==
                              "training" else 1, "precision": precision, "mode": mode,
                              "commit": sha, "label": arguments.label,
                              "attention": network.ATTENTION["kernel"],
                              "autotune": arguments.autotune,
                              "time": datetime.datetime.now().isoformat(timespec="seconds"),
                              "gpu": torch.cuda.get_device_name()} | result
                    rate = record.get("view_ticks_per_s") or record.get("view_steps_per_s") or 0
                    if "error" in record:
                        print(f"{kind:8s} V={views:5d} {precision:8s} {mode:10s} "
                              f"{record['error']}", flush=True)
                        continue
                    print(f"{kind:8s} V={views:5d} {precision:8s} {mode:10s} "
                          f"{record['ms']:9.3f} ms  {rate:12.0f} /s  "
                          f"{record['peak_memory_gb']:.2f} GB"
                          + (f"  (recompute {record['recompute']})"
                             if record["recompute"] != "none" else ""), flush=True)
                    if not arguments.no_write:
                        with open(RESULTS, "a") as handle:
                            handle.write(json.dumps(record) + "\n")


if __name__ == "__main__":
    main()
