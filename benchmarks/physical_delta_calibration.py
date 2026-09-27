"""Calibrate fixed physical next-state delta scales on scripted Frontline play.

The runner samples authoritative truth immediately before and after one policy
decision, excludes every ship whose destination is a respawn, and reports both
all-sample and conditional-nonzero distributions.  The latter is necessary for
sparse damage, shield-delay, cooldown, and field-interface events.

Example (the checked-in Phase-1 artifact):
    .venv/bin/python benchmarks/physical_delta_calibration.py \
        --device cuda --envs 64 --steps 3000 --seeds 271828,314159,161803 \
        --out docs/internal/perception-belief-phase1-calibration.json
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config.core import EnvConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.perception import team_visibility_from_state
from boost_and_broadside.profiles import PROFILES
from boost_and_broadside.runtime.actions import PendingActionState, advance_autonomous_decision
from boost_and_broadside.train.rl.physical_deltas import (
    PHYSICAL_DELTA_NAMES,
    PhysicalShipSnapshot,
    physical_ship_deltas,
)

ABS_QUANTILES = (0.5, 0.9, 0.95, 0.99, 0.999, 0.9999)


def _git_head() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def summarize_channel(values: torch.Tensor) -> dict[str, object]:
    """Robust signed and absolute statistics, including sparse nonzero events."""

    values = values.float()
    absolute = values.abs()
    nonzero = absolute > 0
    quantiles = torch.tensor(ABS_QUANTILES)

    def distribution(sample: torch.Tensor) -> dict[str, float] | None:
        if sample.numel() == 0:
            return None
        q = torch.quantile(sample, quantiles)
        result = {f"p{100 * level:g}": value.item() for level, value in zip(ABS_QUANTILES, q)}
        result["max"] = sample.max().item()
        result["mean"] = sample.mean().item()
        return result

    return {
        "count": values.numel(),
        "mean_signed": values.mean().item(),
        "std_signed": values.std(correction=0).item(),
        "zero_fraction": (~nonzero).float().mean().item(),
        "positive_fraction": (values > 0).float().mean().item(),
        "negative_fraction": (values < 0).float().mean().item(),
        "absolute": distribution(absolute),
        "absolute_nonzero": distribution(absolute[nonzero]),
    }


@torch.inference_mode()
def collect(
    *,
    device: torch.device,
    envs: int,
    steps: int,
    seeds: tuple[int, ...],
) -> tuple[torch.Tensor, dict[str, int | float]]:
    """Collect valid truth-to-truth one-decision deltas across independent seeds."""

    profile = PROFILES["rl"]
    env_config = EnvConfig(
        num_ships=profile.num_ships,
        num_fields=profile.num_fields,
        max_bullets=profile.max_bullets,
        max_episode_steps=profile.max_episode_steps,
        action_repeat=profile.action_repeat,
        spawn_resource_spread=profile.spawn_resource_spread,
        frontline=profile.frontline,
        vision_range=profile.vision_range,
        zones_occlude=profile.zones_occlude,
        spawn_reveal=profile.spawn_reveal,
    )
    chunks: list[torch.Tensor] = []
    candidate_count = 0
    respawn_excluded = 0
    terminal_envs = 0
    respawn_decisions = 0

    for seed in seeds:
        torch.manual_seed(seed)
        env = TensorEnv(envs, profile.ship_config, env_config, device)
        env.reset(seed=seed)
        scripted = StochasticScriptedAgent(profile.ship_config, StochasticAgentConfig())
        actions = PendingActionState.allocate(envs, profile.num_ships, device)

        for _ in range(steps):
            current = PhysicalShipSnapshot.from_state(env.state)
            sight = team_visibility_from_state(
                env.state, profile.ship_config, env_config, perceive_bullets=False
            )
            selected = scripted.get_actions(env.state, sight.ship)
            done, truncated, _ = advance_autonomous_decision(env, actions, selected)
            next_ = PhysicalShipSnapshot.from_state(env.state)
            delta = physical_ship_deltas(current, next_, profile.ship_config.world_size)

            destination_respawn = env.state.ship_respawned
            valid = ~destination_respawn
            candidate_count += valid.numel()
            respawn_excluded += destination_respawn.sum().item()
            respawn_decisions += destination_respawn.any(dim=1).sum().item()
            chunks.append(delta[valid].cpu())

            finished = done | truncated
            terminal_envs += finished.sum().item()
            if bool(finished.any()):
                env.reset_envs(finished)
                actions.reset(finished)

    values = torch.cat(chunks)
    return values, {
        "candidate_ship_transitions": candidate_count,
        "included_ship_transitions": values.shape[0],
        "destination_respawn_excluded": respawn_excluded,
        "destination_respawn_excluded_fraction": respawn_excluded / candidate_count,
        "env_decisions_with_respawn": respawn_decisions,
        "terminal_envs_reset": terminal_envs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--envs", type=int, default=64)
    parser.add_argument("--steps", type=int, default=3000, help="decisions per seed")
    parser.add_argument("--seeds", default="271828,314159,161803")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    device = torch.device(args.device)
    if args.envs < 1 or args.steps < 1:
        parser.error("--envs and --steps must be positive")
    if device.type == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but torch.cuda.is_available() is false")
    seeds = tuple(int(value) for value in args.seeds.split(",") if value)
    if not seeds:
        parser.error("--seeds must contain at least one integer")

    profile = PROFILES["rl"]
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    values, counts = collect(device=device, envs=args.envs, steps=args.steps, seeds=seeds)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - started

    artifact = {
        "schema": "boost-and-broadside-physical-delta-calibration-v1",
        "git_commit": _git_head(),
        "collection": {
            "profile": "rl",
            "controller": "StochasticScriptedAgent vs itself",
            "device": str(device),
            "seeds": list(seeds),
            "parallel_envs": args.envs,
            "decisions_per_seed": args.steps,
            "decision_hz": 1.0 / (profile.ship_config.dt * profile.action_repeat),
            "num_ships": profile.num_ships,
            "num_fields": profile.num_fields,
            "vision_range": profile.vision_range,
            "spawn_reveal": profile.spawn_reveal,
            "ship_config": asdict(profile.ship_config),
            "frontline_config": asdict(profile.frontline),
            "elapsed_seconds": elapsed,
            "ship_transitions_per_second": counts["candidate_ship_transitions"] / elapsed,
            **counts,
        },
        "runtime": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "peak_allocated_bytes": (
                torch.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
            "peak_reserved_bytes": (
                torch.cuda.max_memory_reserved(device) if device.type == "cuda" else None
            ),
        },
        "quantity_definition": {
            "position": "minimum-image displacement on the 65536 px torus",
            "attitude": "signed shortest angular displacement in [-pi, pi]",
            "local_log_index": "delta of natural log(physical refractive index)",
            "mask": "exclude every transition whose destination ship_respawned is true",
        },
        "channels": {
            name: summarize_channel(values[:, index])
            for index, name in enumerate(PHYSICAL_DELTA_NAMES)
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(artifact, indent=2) + "\n")
    print(
        json.dumps(
            {
                "out": str(args.out),
                "collection": artifact["collection"],
                "runtime": artifact["runtime"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
