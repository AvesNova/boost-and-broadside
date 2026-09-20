"""Manual CUDA-graph experiment for one fixed-shape 50v50 ``TensorEnv.tick``.

Run under an external timeout, for example::

    timeout 10m uv run --no-sync python benchmarks/cuda_graph_tick.py --out graph.json

This is intentionally isolated from production. It captures a tick followed by
copy-back into the pre-capture TensorState storages, because tick rebinds many
state attributes. The graph therefore advances those same storages on replay.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import fields, replace
from pathlib import Path

import torch

from boost_and_broadside.config import EnvConfig
from boost_and_broadside.env.cuda_graph import CapturedTick
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.state import TensorState
from boost_and_broadside.profiles import PROFILES


def make_env(seed: int) -> TensorEnv:
    profile = PROFILES["rl"]
    scale = 7000.0 / 2600.0
    frontline = replace(
        profile.frontline,
        zone_radius=profile.frontline.zone_radius * scale,
        zone_ring_radius=profile.frontline.zone_ring_radius * scale,
        playable_radius=profile.frontline.playable_radius * scale,
    )
    env = TensorEnv(
        1,
        profile.ship_config,
        EnvConfig(
            num_ships=100,
            num_fields=72,
            max_bullets=profile.max_bullets,
            max_episode_steps=profile.max_episode_steps,
            action_repeat=1,
            frontline=frontline,
            vision_range=profile.vision_range * scale,
            zones_occlude=profile.zones_occlude,
        ),
        "cuda",
    )
    env.reset(seed=seed)
    return env


class RNGParityUnsupported(RuntimeError):
    """CUDA graph cannot be compared to eager with unconstrained bullet spread."""


def assert_state_equal(left: TensorState, right: TensorState) -> None:
    for field in fields(TensorState):
        if not torch.equal(getattr(left, field.name), getattr(right, field.name)):
            raise AssertionError(f"state mismatch: {field.name}")


def check_rng_support(env: TensorEnv, action: torch.Tensor) -> None:
    """Reject the unproven random branch rather than silently changing spread."""

    shoots = bool((action[..., 2] != 0).any())
    if shoots and env.ship_config.bullet_spread != 0.0:
        raise RNGParityUnsupported(
            "tick may call torch.randn_like for bullet spread; CUDA graph RNG replay "
            "has not been aligned to eager's generator stream. Use no-shoot actions "
            "for dispatch measurement or implement generator-offset parity first."
        )


def parity(
    seed: int,
    ticks: int,
    actions: tuple[torch.Tensor, ...],
    *,
    experimental_rng: bool = False,
    reset_at: int | None = None,
) -> None:
    """Fixed-action eager/graph parity, aligning the shared default generator."""

    eager, graphed = make_env(seed), make_env(seed)
    if not experimental_rng:
        check_rng_support(eager, actions[0])
    before_capture = graphed.state.clone()
    before_capture_rng = torch.cuda.get_rng_state()
    captured = CapturedTick(graphed, actions[0])
    assert_state_equal(before_capture, graphed.state)
    if not torch.equal(before_capture_rng, torch.cuda.get_rng_state()):
        raise AssertionError("CUDA graph construction changed the default RNG state")
    for index in range(ticks):
        if reset_at is not None and index == reset_at:
            mask = torch.ones(1, dtype=torch.bool, device="cuda")
            before_rng = torch.cuda.get_rng_state()
            eager.reset_envs(mask)
            torch.cuda.synchronize()
            after_eager_rng = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(before_rng)
            graphed.reset_envs(mask)
            torch.cuda.synchronize()
            after_graph_rng = torch.cuda.get_rng_state()
            if not torch.equal(after_eager_rng, after_graph_rng):
                raise AssertionError("CUDA RNG state mismatch after reset")
            torch.cuda.set_rng_state(after_eager_rng)
            assert_state_equal(eager.state, graphed.state)
            captured.load_state(graphed.state)
        action = actions[index % len(actions)]
        # Eager and graph execution share one process-global default CUDA
        # generator. Replay from the exact pre-eager state, then carry the
        # eager post-state forward so both paths consume the same stream.
        before_rng = torch.cuda.get_rng_state()
        eager_done, eager_truncated = eager.tick(action)
        torch.cuda.synchronize()
        after_eager_rng = torch.cuda.get_rng_state()
        torch.cuda.set_rng_state(before_rng)
        graph_done, graph_truncated = captured.replay(action)
        torch.cuda.synchronize()
        after_graph_rng = torch.cuda.get_rng_state()
        if not torch.equal(after_eager_rng, after_graph_rng):
            raise AssertionError("CUDA RNG state mismatch after graph replay")
        torch.cuda.set_rng_state(after_eager_rng)
        assert torch.equal(eager_done, graph_done)
        assert torch.equal(eager_truncated, graph_truncated)
        assert_state_equal(eager.state, graphed.state)


def summary(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
        "p99_ms": ordered[int(0.99 * (len(ordered) - 1))],
        "max_ms": ordered[-1],
    }


def save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--parity-ticks", type=int, default=5)
    parser.add_argument("--shoot", action="store_true")
    parser.add_argument("--random-actions", action="store_true")
    parser.add_argument("--reset-at", type=int)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")
    action = torch.zeros((1, 100, 3), dtype=torch.int32, device="cuda")
    if args.shoot:
        action[..., 2] = 1
    if args.random_actions:
        generator = torch.Generator().manual_seed(args.seed + 1)
        actions = tuple(
            torch.stack(
                (
                    torch.randint(3, (1, 100), generator=generator),
                    torch.randint(5, (1, 100), generator=generator),
                    torch.randint(2, (1, 100), generator=generator),
                ),
                dim=-1,
            ).to(dtype=torch.int32, device="cuda")
            for _ in range(max(args.parity_ticks, args.warmup + args.samples))
        )
    else:
        actions = (action,)
    startup = time.perf_counter()
    parity(
        args.seed,
        args.parity_ticks,
        actions,
        experimental_rng=args.shoot or args.random_actions,
        reset_at=args.reset_at,
    )
    graph_env = make_env(args.seed)
    capture_start = time.perf_counter()
    captured = CapturedTick(graph_env, action)
    torch.cuda.synchronize()
    result = {
        "scope": "fixed-action TensorEnv.tick only; no policy, renderer, or display",
        "rng": (
            "default CUDA generator state aligned against eager on every parity tick"
            if args.shoot or args.random_actions
            else "no-shoot action"
        ),
        "startup_and_parity_ms": 1000.0 * (capture_start - startup),
        "parity_passed": True,
        "parity_ticks": args.parity_ticks,
        "reset_at": args.reset_at,
        "action_trace": "seeded variable legal actions" if args.random_actions else "fixed",
        "capture_ms": 1000.0 * (time.perf_counter() - capture_start),
        "samples_ms": [],
    }
    save(args.out, result)
    for _ in range(args.warmup):
        captured.replay(actions[_ % len(actions)])
    torch.cuda.synchronize()
    for index in range(args.samples):
        begin = time.perf_counter()
        captured.replay(actions[(args.warmup + index) % len(actions)])
        torch.cuda.synchronize()
        result["samples_ms"].append(1000.0 * (time.perf_counter() - begin))
    result["summary"] = summary(result["samples_ms"])
    result["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    save(args.out, result)


if __name__ == "__main__":
    main()
