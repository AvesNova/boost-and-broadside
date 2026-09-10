"""Compare full-attention map tokens with K/V-only map memory.

This isolates model inference and sequence forward/backward cost across map
counts. It does not claim learning quality; Gate 5 training curves are a
separate measurement.

Example:
    .venv/bin/python benchmarks/map_memory_suite.py --device cuda --output result.json
"""

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

import torch

from boost_and_broadside.config.defaults import MODEL_CONFIG
from boost_and_broadside.config.resolve import resolve_profile
from boost_and_broadside.env.wrapper import YemongEnvWrapper
from boost_and_broadside.models.yemong.policy import YemongPolicy
from boost_and_broadside.profiles import PROFILES
from boost_and_broadside.train.rl.features import build_standard_coordinator


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _observation(num_envs: int, num_fields: int, device: torch.device):
    resolved = resolve_profile(PROFILES["rl"])
    scale = resolved.train_config.scales[0]
    env_config = replace(scale.env_config, num_fields=num_fields)
    wrapper = YemongEnvWrapper(
        num_envs=num_envs,
        ship_config=resolved.ship_config,
        env_config=env_config,
        rewards=resolved.train_config.rewards,
        device=device,
        include_bullets=False,
    )
    return wrapper.reset(), resolved.ship_config, env_config


def _policy(mode: str, ship_config, num_ships: int, device: torch.device) -> YemongPolicy:
    config = replace(
        MODEL_CONFIG,
        map_read_mode=mode,
        map_memory_dim=64,
        n_bullet_cross_per_block=0,
    )
    return YemongPolicy(
        config,
        build_standard_coordinator(ship_config),
        num_value_components=12,
        num_ships=num_ships,
        team_pma_k=(0, 1),
    ).to(device)


def _inference_profile(
    policy: YemongPolicy,
    obs,
    *,
    iterations: int,
    warmup: int,
    device: torch.device,
) -> dict[str, float]:
    hidden = policy.initial_hidden(obs.pos.shape[0], policy.num_recurrent_tokens, device)

    def step() -> None:
        nonlocal hidden
        with (
            torch.inference_mode(),
            torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"),
        ):
            _, _, _, _, hidden = policy.get_action_and_value(obs, hidden)

    for _ in range(warmup):
        step()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
    else:
        baseline = 0
    started = time.perf_counter()
    for _ in range(iterations):
        step()
    _sync(device)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) - baseline if device.type == "cuda" else 0
    return {
        "milliseconds_per_batch": elapsed * 1000.0 / iterations,
        "environments_per_second": obs.pos.shape[0] * iterations / elapsed,
        "incremental_peak_mib": peak / 2**20,
    }


def _sequence_profile(
    policy: YemongPolicy,
    obs,
    *,
    steps: int,
    iterations: int,
    warmup: int,
    device: torch.device,
) -> dict[str, float]:
    sequence = type(obs)(
        data={key: value.unsqueeze(0).expand(steps, *value.shape) for key, value in obs.items()}
    )
    batch, ships = obs.pos.shape[0], policy.num_recurrent_tokens
    actions = torch.zeros((steps, batch, ships, 3), dtype=torch.long, device=device)
    hidden = policy.initial_hidden(batch, ships, device)
    alive = sequence["belief_valid"]

    def step() -> None:
        policy.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            outputs = policy.evaluate_actions(sequence, actions, hidden, alive)
            loss = outputs[0].float().mean() + outputs[2].float().mean() + outputs[5].float().mean()
        loss.backward()

    for _ in range(warmup):
        step()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        baseline = torch.cuda.memory_allocated(device)
    else:
        baseline = 0
    started = time.perf_counter()
    for _ in range(iterations):
        step()
    _sync(device)
    elapsed = time.perf_counter() - started
    peak = torch.cuda.max_memory_allocated(device) - baseline if device.type == "cuda" else 0
    tokens = steps * batch * ships
    return {
        "milliseconds_per_forward_backward": elapsed * 1000.0 / iterations,
        "ship_tokens_per_second": tokens * iterations / elapsed,
        "incremental_peak_mib": peak / 2**20,
    }


def run_suite(
    *,
    device: torch.device,
    map_counts: tuple[int, ...],
    inference_envs: int,
    update_envs: int,
    update_steps: int,
    inference_iterations: int,
    update_iterations: int,
) -> dict[str, object]:
    torch.manual_seed(20260910)
    results: dict[str, object] = {}
    for fields in map_counts:
        obs, ship_config, env_config = _observation(inference_envs, fields, device)
        update_obs = obs.slice_envs(slice(0, update_envs))
        modes = {}
        for mode in ("full_attention", "kv_memory"):
            policy = _policy(mode, ship_config, env_config.num_ships, device)
            modes[mode] = {
                "parameters": sum(parameter.numel() for parameter in policy.parameters()),
                "inference": _inference_profile(
                    policy,
                    obs,
                    iterations=inference_iterations,
                    warmup=3,
                    device=device,
                ),
                "sequence_forward_backward": _sequence_profile(
                    policy,
                    update_obs,
                    steps=update_steps,
                    iterations=update_iterations,
                    warmup=1,
                    device=device,
                ),
            }
            del policy
            if device.type == "cuda":
                torch.cuda.empty_cache()
        results[str(fields)] = {
            "fields": fields,
            "map_objects": fields + 6,
            "modes": modes,
        }
    return {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "inference_envs": inference_envs,
        "update_envs": update_envs,
        "update_steps": update_steps,
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--map-counts", type=int, nargs="+", default=(0, 10, 50, 100))
    parser.add_argument("--inference-envs", type=int, default=256)
    parser.add_argument("--update-envs", type=int, default=32)
    parser.add_argument("--update-steps", type=int, default=32)
    parser.add_argument("--inference-iterations", type=int, default=30)
    parser.add_argument("--update-iterations", type=int, default=3)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run_suite(
        device=torch.device(args.device),
        map_counts=tuple(args.map_counts),
        inference_envs=args.inference_envs,
        update_envs=args.update_envs,
        update_steps=args.update_steps,
        inference_iterations=args.inference_iterations,
        update_iterations=args.update_iterations,
    )
    text = json.dumps(result, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
