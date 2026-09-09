"""Benchmark pure environment throughput across representative field counts.

This intentionally excludes policy inference so the refractive-physics cost is
not mixed with attention. The report includes the separate token count and
quadratic attention-pair factor that a policy would see.

Example:
    uv run python benchmarks/field_throughput.py --device cuda --num-envs 4096
"""

from __future__ import annotations

import argparse
import time
from dataclasses import fields

import torch

from boost_and_broadside.config import EnvConfig, ShipConfig
from boost_and_broadside.env.env import TensorEnv


def _state_bytes(env: TensorEnv) -> int:
    return sum(
        tensor.numel() * tensor.element_size()
        for field in fields(env.state)
        if isinstance((tensor := getattr(env.state, field.name)), torch.Tensor)
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(
    num_fields: int,
    num_envs: int,
    num_ships: int,
    warmup_steps: int,
    timed_steps: int,
    device: torch.device,
) -> dict[str, float]:
    ship_config = ShipConfig()
    env_config = EnvConfig(
        num_ships=num_ships,
        num_fields=num_fields,
        max_bullets=0,
        max_episode_steps=timed_steps + warmup_steps + 1,
    )
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        allocated_before = torch.cuda.memory_allocated(device)
    else:
        allocated_before = 0

    env = TensorEnv(num_envs, ship_config, env_config, device)
    env.reset(seed=17)
    actions = torch.zeros((num_envs, num_ships, 3), dtype=torch.long, device=device)

    reset_mask = torch.ones(num_envs, dtype=torch.bool, device=device)
    _sync(device)
    reset_start = time.perf_counter()
    env.reset_envs(reset_mask)
    _sync(device)
    reset_elapsed = time.perf_counter() - reset_start

    for _ in range(warmup_steps):
        env.step(actions)
    _sync(device)

    start = time.perf_counter()
    for _ in range(timed_steps):
        env.step(actions)
    _sync(device)
    elapsed = time.perf_counter() - start

    peak_delta = (
        torch.cuda.max_memory_allocated(device) - allocated_before if device.type == "cuda" else 0
    )
    tokens = num_ships + num_fields
    return {
        "fields": float(num_fields),
        "env_sps": num_envs * timed_steps / elapsed,
        "tick_sps": timed_steps / elapsed,
        "state_mib": _state_bytes(env) / 2**20,
        "reset_us_per_env": reset_elapsed * 1e6 / num_envs,
        "peak_delta_mib": peak_delta / 2**20,
        "tokens": float(tokens),
        "attention_factor": (tokens / num_ships) ** 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-envs", type=int, default=4096)
    parser.add_argument("--num-ships", type=int, default=8)
    parser.add_argument("--warmup-steps", type=int, default=30)
    parser.add_argument("--timed-steps", type=int, default=300)
    parser.add_argument(
        "--field-counts",
        type=int,
        nargs="+",
        default=(0, 1, 2, 4, 20),
        help="Field counts to benchmark; 20 is the provisional Frontline preset.",
    )
    args = parser.parse_args()
    device = torch.device(args.device)

    results = [
        benchmark(
            count,
            args.num_envs,
            args.num_ships,
            args.warmup_steps,
            args.timed_steps,
            device,
        )
        for count in args.field_counts
    ]
    baseline = results[0]["env_sps"]

    device_label = (
        f"{device} ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else str(device)
    )
    print(
        f"device={device_label} envs={args.num_envs} ships={args.num_ships} "
        f"warmup={args.warmup_steps} timed={args.timed_steps}"
    )
    print(
        "fields  env-steps/s  relative  state MiB  reset us/env  "
        "peak MiB  tokens  attention-pairs"
    )
    for row in results:
        relative = row["env_sps"] / baseline
        print(
            f"{int(row['fields']):>6}  {row['env_sps']:>11,.0f}  {relative:>7.3f}x  "
            f"{row['state_mib']:>9.2f}  {row['reset_us_per_env']:>12.3f}  "
            f"{row['peak_delta_mib']:>8.2f}  {int(row['tokens']):>6}  "
            f"{row['attention_factor']:>15.3f}x"
        )
    print(
        "Policy inference is excluded. attention-pairs is the theoretical "
        "(N+M)^2/N^2 token-attention multiplier."
    )


if __name__ == "__main__":
    main()
