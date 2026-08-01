"""Benchmark GPU bullet storage and physics independently from policy inference.

The sustained-fire workload removes combat/field damage and firing energy cost so
ships remain alive and keep filling the fixed bullet pool for the entire trial.
Collision detection and bullet deactivation still run normally.

Example production-shape baseline:
    uv run python benchmarks/bullet_throughput.py \
        --device cuda --num-envs 3904 --num-ships 8 --num-fields 4 \
        --max-bullets 20 --workload saturated
"""

from __future__ import annotations

import argparse
import statistics
import time
from dataclasses import fields, replace

import torch

from boost_and_broadside.config import EnvConfig, FieldMapConfig, ShipConfig
from boost_and_broadside.constants import ShootActions
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


def _actions(
    workload: str,
    num_envs: int,
    num_ships: int,
    device: torch.device,
) -> torch.Tensor:
    actions = torch.zeros((num_envs, num_ships, 3), dtype=torch.long, device=device)
    if workload == "saturated":
        actions[..., 2] = int(ShootActions.SHOOT)
    return actions


@torch.inference_mode()
def benchmark(args: argparse.Namespace) -> dict[str, float]:
    device = torch.device(args.device)
    ship_config = replace(
        ShipConfig(bullet_energy_cost=2.0, bullet_min_damage_frac=1.0),
        bullet_damage=0.0,
        bullet_energy_cost=0.0,
        bullet_drag_coeff=args.bullet_drag_coeff,
        bullet_field_integrator=args.bullet_field_integrator,
        bullet_field_integration_substeps=args.bullet_field_substeps,
        bullet_field_damage_scale=args.bullet_field_damage_scale,
        field_interface_damage=0.0,
    )
    env_config = EnvConfig(
        num_ships=args.num_ships,
        num_fields=args.num_fields,
        max_bullets=args.max_bullets,
        max_episode_steps=None,
    )
    field_map = None
    if args.num_fields:
        field_map = FieldMapCache.generate(
            ship_config,
            env_config,
            FieldMapConfig(cache_size=args.map_cache_size, max_generation_attempts=256),
            device,
            seed=args.seed,
        )

    collision_compile_mode = args.compile_mode if args.compile_mode != "none" else None
    env = TensorEnv(
        args.num_envs,
        ship_config,
        env_config,
        device,
        field_map,
        collision_compile_mode,
    )
    actions = _actions(args.workload, args.num_envs, args.num_ships, device)
    elapsed_samples: list[float] = []
    occupancy_samples: list[float] = []
    peak_samples: list[float] = []

    for repeat in range(args.repeats):
        env.reset(seed=args.seed + repeat)
        for _ in range(args.warmup_steps):
            env.step(actions)
        _sync(device)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        started = time.perf_counter()
        for _ in range(args.timed_steps):
            env.step(actions)
        _sync(device)
        elapsed_samples.append(time.perf_counter() - started)
        occupancy_samples.append(env.state.bullet_active.float().mean().item())
        if device.type == "cuda":
            peak_samples.append(torch.cuda.max_memory_allocated(device) / 2**20)

    elapsed = statistics.median(elapsed_samples)
    env_steps = args.num_envs * args.timed_steps
    env_sps = env_steps / elapsed
    state_mib = _state_bytes(env) / 2**20
    cache_mib = _tensor_bytes(field_map) / 2**20 if field_map is not None else 0.0
    return {
        "env_sps": env_sps,
        "ship_tokens_per_sec": env_sps * args.num_ships,
        "tick_ms": elapsed / args.timed_steps * 1_000.0,
        "active_fraction": statistics.median(occupancy_samples),
        "active_per_ship": statistics.median(occupancy_samples) * args.max_bullets,
        "state_mib": state_mib,
        "cache_mib": cache_mib,
        "peak_mib": statistics.median(peak_samples) if peak_samples else state_mib + cache_mib,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-envs", type=int, default=3904)
    parser.add_argument("--num-ships", type=int, default=8)
    parser.add_argument("--num-fields", type=int, default=4)
    parser.add_argument("--max-bullets", type=int, default=20)
    parser.add_argument("--bullet-drag-coeff", type=float, default=8e-4)
    parser.add_argument(
        "--bullet-field-integrator",
        choices=("two_step", "midpoint"),
        default="two_step",
    )
    parser.add_argument("--bullet-field-substeps", type=int, default=2)
    parser.add_argument("--bullet-field-damage-scale", type=float, default=0.1)
    parser.add_argument("--workload", choices=("idle", "saturated"), default="saturated")
    parser.add_argument(
        "--compile-mode",
        choices=("none", "default", "max-autotune"),
        default="none",
    )
    parser.add_argument("--map-cache-size", type=int, default=64)
    parser.add_argument("--warmup-steps", type=int, default=60)
    parser.add_argument("--timed-steps", type=int, default=240)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    result = benchmark(args)
    device = torch.device(args.device)
    device_label = (
        f"{device} ({torch.cuda.get_device_name(device)})" if device.type == "cuda" else str(device)
    )
    print(
        f"device={device_label} envs={args.num_envs} ships={args.num_ships} "
        f"fields={args.num_fields} slots={args.max_bullets} workload={args.workload} "
        f"drag={args.bullet_drag_coeff:g} integrator={args.bullet_field_integrator} "
        f"substeps={args.bullet_field_substeps} "
        f"damage_scale={args.bullet_field_damage_scale:g} compile={args.compile_mode} "
        f"warmup={args.warmup_steps} timed={args.timed_steps} repeats={args.repeats}"
    )
    print(
        f"env_steps/s={result['env_sps']:,.0f} "
        f"ship_tokens/s={result['ship_tokens_per_sec']:,.0f} "
        f"tick_ms={result['tick_ms']:.3f} "
        f"active/ship={result['active_per_ship']:.2f} "
        f"active={result['active_fraction']:.1%} "
        f"state={result['state_mib']:.2f}MiB "
        f"cache={result['cache_mib']:.2f}MiB "
        f"peak={result['peak_mib']:.2f}MiB"
    )


if __name__ == "__main__":
    main()
