"""Benchmark pure environment throughput for zero, one, two, and four fields.

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

from boost_and_broadside.config import EnvConfig, FieldMapConfig, ShipConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.field_cache import FieldMapCache


def _tensor_bytes(obj: object) -> int:
    return sum(
        value.numel() * value.element_size()
        for value in vars(obj).values()
        if isinstance(value, torch.Tensor)
    )


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
    field_map = None
    if num_fields:
        field_map = FieldMapCache.generate(
            ship_config,
            env_config,
            FieldMapConfig(cache_size=64, max_generation_attempts=256),
            device,
            seed=2026 + num_fields,
        )

    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        allocated_before = torch.cuda.memory_allocated(device)
    else:
        allocated_before = 0

    env = TensorEnv(num_envs, ship_config, env_config, device, field_map)
    env.reset(seed=17)
    actions = torch.zeros((num_envs, num_ships, 3), dtype=torch.long, device=device)

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
    cache_bytes = _tensor_bytes(field_map) if field_map is not None else 0
    tokens = num_ships + num_fields
    return {
        "fields": float(num_fields),
        "env_sps": num_envs * timed_steps / elapsed,
        "tick_sps": timed_steps / elapsed,
        "state_mib": _state_bytes(env) / 2**20,
        "cache_mib": cache_bytes / 2**20,
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
        for count in (0, 1, 2, 4)
    ]
    baseline = results[0]["env_sps"]

    device_label = (
        f"{device} ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else str(device)
    )
    print(
        f"device={device_label} envs={args.num_envs} ships={args.num_ships} "
        f"warmup={args.warmup_steps} timed={args.timed_steps}"
    )
    print("fields  env-steps/s  relative  state MiB  cache MiB  peak MiB  tokens  attention-pairs")
    for row in results:
        relative = row["env_sps"] / baseline
        print(
            f"{int(row['fields']):>6}  {row['env_sps']:>11,.0f}  {relative:>7.3f}x  "
            f"{row['state_mib']:>9.2f}  {row['cache_mib']:>9.2f}  "
            f"{row['peak_delta_mib']:>8.2f}  {int(row['tokens']):>6}  "
            f"{row['attention_factor']:>15.3f}x"
        )
    print(
        "Policy inference is excluded. attention-pairs is the theoretical "
        "(N+M)^2/N^2 token-attention multiplier."
    )


if __name__ == "__main__":
    main()
