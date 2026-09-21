"""Bounded 5v5 vector-environment screen for training performance candidates.

This isolates the real RL wrapper's reset and observation/reward step path. An
optional ``module:callable`` can replace only ``TensorEnv.tick``; its signature
is ``candidate(env, actions)``. It must preserve the TensorEnv tick contract.
This is not a PPO throughput benchmark: no policy, rollout buffer, or optimizer
is included in the reported environment measurements.

Example (use an external timeout for CUDA runs)::

    timeout 10m uv run python benchmarks/rl_batch_screen.py --device cuda \
        --batches 1,32,128,256 --candidate my_candidate:tick --out /tmp/rl.jsonl
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import statistics
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import torch


def load_candidate(spec: str | None) -> Callable | None:
    """Load an explicit candidate function from ``module:attribute``."""
    if spec is None:
        return None
    module_name, separator, attr_name = spec.partition(":")
    if not separator or not module_name or not attr_name:
        raise ValueError("candidate must use module:callable syntax")
    candidate = getattr(importlib.import_module(module_name), attr_name)
    if not callable(candidate):
        raise TypeError(f"candidate {spec!r} is not callable")
    return candidate


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _measure(fn: Callable[[], object], *, device: torch.device, warmup: int, steps: int) -> dict:
    for _ in range(warmup):
        fn()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples: list[float] = []
    _sync(device)
    started = time.perf_counter()
    for _ in range(steps):
        sample_start = time.perf_counter()
        fn()
        _sync(device)
        samples.append((time.perf_counter() - sample_start) * 1000)
    elapsed = time.perf_counter() - started
    result = {
        "iterations": steps,
        "mean_ms": statistics.mean(samples),
        "median_ms": statistics.median(samples),
        "p95_ms": sorted(samples)[min(len(samples) - 1, int(0.95 * len(samples)))],
        "samples_ms": samples,
        "window_seconds": elapsed,
    }
    if device.type == "cuda":
        result["peak_allocated_bytes"] = torch.cuda.max_memory_allocated(device)
        result["peak_reserved_bytes"] = torch.cuda.max_memory_reserved(device)
    else:
        result["peak_allocated_bytes"] = None
        result["peak_reserved_bytes"] = None
    return result


def _revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_batch(
    args: argparse.Namespace,
    batch: int,
    arm: str,
    candidate: Callable | None,
    perception_compile_mode: str | None,
    pair: int,
    order: int,
) -> dict:
    from realtime_latency import scenario_config

    from boost_and_broadside.config.defaults import REWARDS
    from boost_and_broadside.env.wrapper import YemongEnvWrapper
    from boost_and_broadside.profiles import PROFILES

    profile = PROFILES["rl"]
    device = torch.device(args.device)
    wrapper = YemongEnvWrapper(
        batch,
        profile.ship_config,
        scenario_config(10, 10, 1.0),
        REWARDS,
        device,
        include_bullets=False,
        perceive_bullets=False,
        perception_compile_mode=perception_compile_mode,
    )
    if candidate is not None:
        original_tick = wrapper.env.tick

        def candidate_tick(actions, *, unlimited_resources=False):
            if unlimited_resources:
                return original_tick(actions, unlimited_resources=True)
            return candidate(wrapper.env, actions)

        wrapper.env.tick = candidate_tick

    torch.manual_seed(args.seed)
    _sync(device)
    startup_started = time.perf_counter()
    wrapper.reset(seed=args.seed)
    _sync(device)
    startup_reset_seconds = time.perf_counter() - startup_started
    actions = torch.zeros((batch, 10, 3), dtype=torch.int32, device=device)
    actions[..., 1] = 1

    def step():
        wrapper.step(actions, auto_reset=False)

    reset_mask = torch.ones(batch, dtype=torch.bool, device=device)

    # Reset includes randomized state construction plus the training wrapper's
    # episode/perception bookkeeping and initial observation generation.
    reset = _measure(
        lambda: wrapper.reset(seed=args.seed), device=device, warmup=args.warmup, steps=args.steps
    )

    # Measure reset_envs at steady state as it is invoked after completed games.
    # Rebuild wrapper-visible state through reset() after each physics reset so
    # the next step still begins from a valid training state.
    def reset_envs():
        wrapper.env.reset_envs(reset_mask)
        wrapper._reset_perception(reset_mask)
        wrapper._refresh_field_obs(reset_mask)

    partial_reset = _measure(reset_envs, device=device, warmup=args.warmup, steps=args.steps)
    _sync(device)
    wrapper.reset(seed=args.seed)
    step_metrics = _measure(step, device=device, warmup=args.warmup, steps=args.steps)
    transitions = batch * args.steps
    step_metrics["environment_decisions_per_second"] = transitions / step_metrics["window_seconds"]
    step_metrics["ship_decisions_per_second"] = transitions * 10 / step_metrics["window_seconds"]
    result = {
        "type": "result",
        "arm": arm,
        "pair": pair,
        "order": order,
        "batch": batch,
        "seed": args.seed,
        "device": str(device),
        "ships": 10,
        "fields": 10,
        "include_bullets": False,
        "perception_compile_mode": perception_compile_mode,
        "startup_reset_seconds": startup_reset_seconds,
        "phases": {
            "full_wrapper_reset": reset,
            "completed_episode_reset": partial_reset,
            "wrapper_step": step_metrics,
        },
    }
    wrapper.close() if hasattr(wrapper, "close") else None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--batches",
        default="1,32,128,256",
        help="Comma-separated widths; defaults include requested 1, 32, 128 and 256.",
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--candidate", help="Optional candidate tick callable as module:callable.")
    parser.add_argument(
        "--perception-compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        help="Candidate-only pure observation compilation mode.",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=1,
        help="A/B pairs per batch; arm order alternates between pairs.",
    )
    parser.add_argument("--out", type=Path, required=True, help="Append-only JSONL output path.")
    args = parser.parse_args()
    try:
        batches = [int(part) for part in args.batches.split(",")]
        candidate = load_candidate(args.candidate)
    except (ValueError, TypeError, ImportError, AttributeError) as exc:
        parser.error(str(exc))
    if not batches or min(batches) < 1 or args.steps < 1 or args.warmup < 0 or args.pairs < 1:
        parser.error("batches/steps/pairs must be positive and warmup must be non-negative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    torch.set_num_threads(1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    metadata = {
        "type": "metadata",
        "schema_version": 1,
        "benchmark": "5v5_vector_environment_screen",
        "revision": _revision(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "platform": platform.platform(),
        "device": args.device,
        "device_name": (
            torch.cuda.get_device_name(0) if args.device == "cuda" else platform.processor()
        ),
        "seed": args.seed,
        "args": vars(args) | {"out": str(args.out)},
        "candidate": {
            "tick": args.candidate,
            "perception_compile_mode": args.perception_compile_mode,
        },
        "comparison_semantics": (
            "reference and optional candidate run sequentially with identical seeds/shapes"
        ),
        "training_pipeline_included": False,
        "ppo_rollout_or_update_included": False,
    }
    with args.out.open("w", encoding="utf-8") as output:
        output.write(json.dumps(metadata, default=str) + "\n")
        output.flush()
        has_candidate = candidate is not None or args.perception_compile_mode is not None
        arms = [
            ("reference", None, None),
            ("candidate", candidate, args.perception_compile_mode),
        ]
        for batch in batches:
            for pair in range(args.pairs):
                ordered_arms = arms if pair % 2 == 0 else list(reversed(arms))
                if not has_candidate:
                    ordered_arms = ordered_arms[:1]
                for order, (arm, fn, perception_mode) in enumerate(ordered_arms):
                    record = run_batch(args, batch, arm, fn, perception_mode, pair, order)
                    output.write(json.dumps(record) + "\n")
                    output.flush()
                    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
