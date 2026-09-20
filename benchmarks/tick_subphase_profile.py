"""Attribute the 50v50 CUDA physics tick without changing its semantics."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch
from realtime_latency import PROFILE, scenario_config

from boost_and_broadside.config.defaults import REWARDS
from boost_and_broadside.env import env as env_module
from boost_and_broadside.env import frontline as frontline_module
from boost_and_broadside.env import physics as physics_module
from boost_and_broadside.env.wrapper import YemongEnvWrapper


def _summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)

    def at(percentile: float) -> float:
        return ordered[int((len(ordered) - 1) * percentile)]

    return {
        "count": len(ordered),
        "p50_ms": statistics.median(ordered),
        "p95_ms": at(0.95),
        "p99_ms": at(0.99),
        "max_ms": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    device = torch.device("cuda")
    config = scenario_config(100, 72, 7000.0 / 2600.0)
    wrapper = YemongEnvWrapper(
        1,
        PROFILE.ship_config,
        config,
        REWARDS,
        device,
        include_bullets=False,
        perceive_bullets=True,
    )
    wrapper.reset(seed=args.seed)
    action = torch.zeros((1, 100, 3), dtype=torch.int32, device=device)
    action[..., 0] = 1
    action[..., 1] = 2
    action[..., 2] = 1

    for _ in range(args.warmup):
        wrapper.env.tick(action)
    torch.cuda.synchronize()

    samples: dict[str, list[float]] = {}
    originals = {}

    def wrap(module, name: str, label: str):
        original = getattr(module, name)
        originals[(module, name)] = original

        def timed(*call_args, **call_kwargs):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            start.record()
            result = original(*call_args, **call_kwargs)
            end.record()
            torch.cuda.synchronize()
            samples.setdefault(label, []).append(start.elapsed_time(end))
            return result

        setattr(module, name, timed)

    boundaries = (
        (env_module, "clear_previous_life_attribution", "tick.clear_attribution"),
        (env_module, "update_ships", "tick.update_ships"),
        (env_module, "advance_bullets", "tick.advance_bullets"),
        (env_module, "resolve_collisions", "tick.resolve_collisions"),
        (env_module, "apply_frontline_tick", "tick.apply_frontline"),
        (physics_module, "_update_kinematics_in_fields", "ships.kinematics_fields"),
        (physics_module, "_handle_shooting", "ships.shooting"),
        (physics_module, "_transport_bullets_through_fields", "bullets.field_transport"),
        (physics_module, "_apply_combat_damage", "collision.combat_damage"),
        (physics_module, "_check_game_over", "collision.game_over"),
        (frontline_module, "zone_membership", "frontline.zone_membership"),
        (frontline_module, "_advance_capture_state", "frontline.capture"),
        (frontline_module, "_apply_frontline_hazards", "frontline.hazards"),
        (frontline_module, "place_ships_at_spawns", "frontline.respawn"),
    )
    for module, name, label in boundaries:
        wrap(module, name, label)
    total: list[float] = []
    try:
        for _ in range(args.samples):
            begin = time.perf_counter()
            wrapper.env.tick(action)
            torch.cuda.synchronize()
            total.append(1000.0 * (time.perf_counter() - begin))
    finally:
        for (module, name), original in originals.items():
            setattr(module, name, original)

    result = {
        "scope": "50v50 TensorEnv.tick synchronized subphase attribution",
        "seed": args.seed,
        "warmup": args.warmup,
        "samples": args.samples,
        "active_bullets_final": int(wrapper.state.bullet_active.sum().item()),
        "summaries": {"tick_wall": _summary(total)}
        | {name: _summary(values) for name, values in samples.items()},
        "raw_ms": {"tick_wall": total} | samples,
        "limitation": "Synchronization around every subphase perturbs normal frame latency.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["summaries"], indent=2))


if __name__ == "__main__":
    main()
