"""Bounded performance audit and isolated LOS prototype; no production changes.

Run with --mode profile for component attribution, --mode los for a parity-gated
functional compilation experiment, or --mode realtime to A/B the prototype in
the existing two-policy harness. JSON includes raw samples and local provenance.
"""

from __future__ import annotations

import argparse
import copy
import cProfile
import io
import json
import platform
import pstats
import statistics
import subprocess
import sys
import time
from pathlib import Path

import realtime_latency as realtime
import torch

from boost_and_broadside.constants import EPS
from boost_and_broadside.env import perception
from boost_and_broadside.env.wrapper import YemongEnvWrapper


def load_warp(args):
    if args.warp_path:
        sys.path.append(str(args.warp_path))
    from performance_audit_warp import warp_los

    return warp_los


def real_los(ox, oy, tx, ty, cx, cy, radius, width, height):
    """Pure real-valued equivalent of the dense shortest-path disk LOS test.

    Squared distances avoid complex intermediates and square roots. This is an
    experimental numerical reformulation, requiring boolean parity at boundaries.
    """
    sx = (tx[:, None, :] - ox[:, :, None] + width / 2) % width - width / 2
    sy = (ty[:, None, :] - oy[:, :, None] + height / 2) % height - height / 2
    dx = (cx[:, None, None, :] - ox[:, :, None, None] + width / 2) % width - width / 2
    dy = (cy[:, None, None, :] - oy[:, :, None, None] + height / 2) % height - height / 2
    sx, sy = sx.unsqueeze(-1), sy.unsqueeze(-1)
    projection = ((dx * sx + dy * sy) / (sx * sx + sy * sy).clamp(min=EPS)).clamp(0, 1)
    near_x, near_y = dx - projection * sx, dy - projection * sy
    r2 = radius[:, None, None, :].square()
    both_inside = (dx * dx + dy * dy < r2) & ((dx - sx).square() + (dy - sy).square() < r2)
    return ~((near_x.square() + near_y.square() < r2) & ~both_inside).any(dim=-1)


def adapter(kernel):
    def call(observer, target, core_pos, core_radius, world_size):
        return kernel(
            observer.real,
            observer.imag,
            target.real,
            target.imag,
            core_pos.real,
            core_pos.imag,
            core_radius,
            *world_size,
        )

    return call


def samples(fn, steps, device="cpu", *, window=False):
    """Synchronize each latency sample, or only the complete throughput window."""
    result = []

    def sync():
        if str(device).startswith("cuda"):
            torch.cuda.synchronize()

    sync()
    window_start = time.perf_counter()
    for _ in range(steps):
        start = time.perf_counter()
        fn()
        if not window:
            sync()
        result.append((time.perf_counter() - start) * 1000)
    sync()
    elapsed_ms = (time.perf_counter() - window_start) * 1000
    return {
        "median_ms": statistics.median(result),
        "mean_ms": statistics.mean(result),
        "p99_ms": sorted(result)[int(0.99 * (len(result) - 1))],
        "samples_ms": result,
        "window_mean_ms": elapsed_ms / steps,
        "sample_semantics": "host_enqueue" if window else "completion_latency",
    }


def make_wrapper(name, batch=1, bullets=True, device="cpu"):
    ships, fields, scale = realtime.SCENARIOS[name]
    wrapper = YemongEnvWrapper(
        batch,
        realtime.PROFILE.ship_config,
        realtime.scenario_config(ships, fields, scale),
        realtime.REWARDS,
        device,
        include_bullets=False,
        perceive_bullets=bullets,
    )
    wrapper.reset()
    return wrapper


def profile(args):
    if args.device != "cpu":
        raise ValueError(
            "cProfile component mode is CPU-only; use GPU pipeline profiling separately"
        )
    wrapper = make_wrapper(args.scenario, args.batch)
    action = torch.zeros((args.batch, wrapper.state.max_ships, 3), dtype=torch.int32)
    action[..., 1] = 1
    action[..., 2] = 1
    for _ in range(10):
        wrapper.step(action, auto_reset=False)
    renderer = realtime._headless_renderer(realtime.PROFILE.ship_config, 900)
    calls = {
        "physics": lambda: wrapper.env.tick(action),
        "observation_bullets": wrapper._get_obs,
        "wrapper_step": lambda: wrapper.step(action, auto_reset=False),
        "render": lambda: renderer.draw_frame(wrapper.state, visibility=wrapper.last_visibility),
    }
    result = {"components": {}}
    for label, fn in calls.items():
        result["components"][label] = samples(fn, args.steps)
        prof = cProfile.Profile()
        prof.runcall(lambda: [fn() for _ in range(min(args.steps, 5))])
        stream = io.StringIO()
        pstats.Stats(prof, stream=stream).strip_dirs().sort_stats("cumulative").print_stats(25)
        result[label + "_profile"] = stream.getvalue()
    wrapper.perceive_bullets = False
    result["components"]["observation_no_bullets"] = samples(wrapper._get_obs, args.steps)
    renderer.close()
    result["active_bullets"] = wrapper.state.bullet_active.sum().item()
    return result


def render_ablation(args):
    """Price fog on one fixed state; omitting fog is a lower bound, not a fix."""
    if args.device != "cpu" or args.batch != 1:
        raise ValueError("render ablation requires CPU and B=1")
    wrapper = make_wrapper(args.scenario)
    action = torch.zeros((1, wrapper.state.max_ships, 3), dtype=torch.int32)
    action[..., 2] = 1
    for _ in range(10):
        wrapper.env.tick(action)
    wrapper._get_obs()
    renderer = realtime._headless_renderer(realtime.PROFILE.ship_config, args.window_size)
    original = renderer._draw_fog_overlay
    original_projection = renderer._unwrapped_world_to_screen
    # Prime the initial camera fit, then cache only this fixed frame's transform.
    renderer.draw_frame(wrapper.state, visibility=wrapper.last_visibility)
    center, scale = renderer.camera.center, renderer.camera.scale
    view_w, view_h = renderer.camera.viewport_size

    def cached_projection(position):
        delta = position - center
        return round(view_w / 2 + delta.real * scale), round(view_h / 2 + delta.imag * scale)

    import pygame

    reference_pixels = pygame.image.tobytes(
        renderer.draw_frame(wrapper.state, visibility=wrapper.last_visibility), "RGB"
    )
    renderer._unwrapped_world_to_screen = cached_projection
    cached_pixels = pygame.image.tobytes(
        renderer.draw_frame(wrapper.state, visibility=wrapper.last_visibility), "RGB"
    )
    assert reference_pixels == cached_pixels, "Cached projection changed the rendered image"
    renderer._unwrapped_world_to_screen = original_projection
    result = []
    try:
        for pair in range(args.pairs):
            order = [
                ("full", original, original_projection),
                ("zero_cost_fog_bound", lambda *args: None, original_projection),
                ("cached_projection_static", original, cached_projection),
            ]
            if pair % 2:
                order.reverse()
            for label, fn, projection in order:
                renderer._draw_fog_overlay = fn
                renderer._unwrapped_world_to_screen = projection

                def draw():
                    renderer.draw_frame(wrapper.state, visibility=wrapper.last_visibility)

                for _ in range(3):
                    draw()
                result.append({"pair": pair, "arm": label, **samples(draw, args.steps)})
    finally:
        renderer._draw_fog_overlay = original
        renderer._unwrapped_world_to_screen = original_projection
        renderer.close()
    return result


def tick_compile(args):
    """Re-test whole mutable tick compilation on the installed Torch version.

    Any failed field comparison rejects all speed claims for this candidate.
    No automatic reset or reward/observation wrapper is inside this boundary.
    """
    wrapper = make_wrapper(args.scenario, args.batch, device=args.device)
    reference = wrapper.env
    candidate = copy.deepcopy(reference)
    compiled = torch.compile(candidate.tick, dynamic=False)
    action = torch.zeros(
        (args.batch, wrapper.state.max_ships, 3), dtype=torch.int32, device=args.device
    )
    action[..., 1] = 1
    action[..., 2] = 1
    result = {"parity_passed": False, "checked_ticks": 0, "mismatches": []}
    start = time.perf_counter()
    for tick in range(20):
        torch.manual_seed(99 + tick)
        expected = reference.tick(action)
        torch.manual_seed(99 + tick)
        actual = compiled(action)
        realtime._sync(torch.device(args.device))
        if tick == 0:
            result["first_pair_seconds"] = time.perf_counter() - start
        for name, left in vars(reference.state).items():
            right = getattr(candidate.state, name)
            try:
                floating = left.is_floating_point() or left.is_complex()
                torch.testing.assert_close(
                    left, right, rtol=1e-6 if floating else 0, atol=1e-4 if floating else 0
                )
            except AssertionError as exc:
                result["mismatches"].append({"tick": tick, "field": name, "error": str(exc)})
        for left, right in zip(expected, actual, strict=True):
            if not torch.equal(left, right):
                result["mismatches"].append({"tick": tick, "field": "terminal_flags"})
        result["checked_ticks"] += 1
        if result["mismatches"]:
            return result
    result["parity_passed"] = True
    result["arms"] = []
    realtime.warm_device(torch.device(args.device))
    for pair in range(args.pairs):
        order = [("reference", reference.tick), ("compiled_tick", compiled)]
        if pair % 2:
            order.reverse()
        for label, fn in order:
            result["arms"].append(
                {
                    "pair": pair,
                    "arm": label,
                    **samples(lambda: fn(action), args.steps, args.device),
                }
            )
    return result


def los(args):
    wrapper = make_wrapper(args.scenario, args.batch, device=args.device)
    state = wrapper.state
    cores, radii = perception._occluder_cores(state, wrapper.env_config)
    world = wrapper.ship_config.world_size
    eager = perception._line_of_sight_clear
    real = adapter(real_los)
    compiled = adapter(torch.compile(real_los, fullgraph=True, dynamic=False))
    warp = load_warp(args) if args.warp_path else None
    result = {"parity_cases": 0, "shapes": []}
    # Broad random coverage plus degenerate, tangent, and seam-crossing probes.
    probes = [
        (
            torch.tensor([[0j, 1 + 0j, 16383 + 0j]], dtype=torch.complex64),
            torch.tensor([[0j, 2 + 0j, 1 + 0j, 2 + 1j]], dtype=torch.complex64),
            torch.tensor([[1 + 0j, 1 + 1j]], dtype=torch.complex64),
            torch.tensor([[1.0, 1.0]]),
        ),
    ]
    for seed in range(10):
        gen = torch.Generator().manual_seed(seed)
        probes.append(
            (
                torch.complex(
                    torch.rand(2, 10, generator=gen) * world[0],
                    torch.rand(2, 10, generator=gen) * world[1],
                ),
                torch.complex(
                    torch.rand(2, 100, generator=gen) * world[0],
                    torch.rand(2, 100, generator=gen) * world[1],
                ),
                torch.complex(
                    torch.rand(2, 15, generator=gen) * world[0],
                    torch.rand(2, 15, generator=gen) * world[1],
                ),
                torch.rand(2, 15, generator=gen) * 1000,
            )
        )
    for observer, target, centers, radius in probes:
        observer, target, centers, radius = (
            tensor.to(args.device) for tensor in (observer, target, centers, radius)
        )
        expected = eager(observer, target, centers, radius, world)
        torch.testing.assert_close(real(observer, target, centers, radius, world), expected)
        torch.testing.assert_close(compiled(observer, target, centers, radius, world), expected)
        if warp is not None:
            torch.testing.assert_close(warp(observer, target, centers, radius, world), expected)
        result["parity_cases"] += expected.numel()
    for label, target in (
        ("ships", state.ship_pos),
        ("bullet_capacity", state.bullet_pos.flatten(1)),
    ):
        inputs = (state.ship_pos, target, cores, radii, world)
        expected = eager(*inputs)
        start = time.perf_counter()
        actual = compiled(*inputs)
        realtime._sync(torch.device(args.device))
        cold = time.perf_counter() - start
        torch.testing.assert_close(actual, expected)
        # Changed values, same storage shape: rejects accidental constant capture.
        changed = (state.ship_pos + 31j, target + 17, cores, radii, world)
        torch.testing.assert_close(compiled(*changed), eager(*changed))
        if warp is not None:
            torch.testing.assert_close(warp(*inputs), expected)
            torch.testing.assert_close(warp(*changed), eager(*changed))
        result["parity_cases"] += 2 * expected.numel()
        row = {
            "target": label,
            "shape": [args.batch, state.max_ships, target.shape[1], cores.shape[1]],
            "first_call_seconds": cold,
            "arms": [],
        }
        realtime.warm_device(torch.device(args.device))
        for pair in range(args.pairs):
            order = [("eager_complex", eager), ("eager_real", real), ("compiled_real", compiled)]
            if warp is not None:
                order.append(("warp", warp))
            if pair % 2:
                order.reverse()
            for name, fn in order:
                for _ in range(3):
                    fn(*inputs)
                row["arms"].append(
                    {
                        "pair": pair,
                        "name": name,
                        **samples(lambda: fn(*inputs), args.steps, args.device),
                    }
                )
        result["shapes"].append(row)
    return result


def realtime_ab(args):
    original = perception._line_of_sight_clear
    fused = (
        load_warp(args)
        if args.backend == "warp"
        else adapter(torch.compile(real_los, fullgraph=True, dynamic=False))
    )
    variant_label = "warp_los" if args.backend == "warp" else "compiled_real_los"
    result = []
    try:
        for pair in range(args.pairs):
            order = [("reference", original), (variant_label, fused)]
            if pair % 2:
                order.reverse()
            for label, fn in order:
                perception._line_of_sight_clear = fn
                row = realtime.measure(
                    args.scenario,
                    torch.device(args.device),
                    steps=args.steps,
                    warmup=10,
                    compile_mode=None if args.policy_compile == "none" else args.policy_compile,
                    seed=99,
                    stage="baseline",
                    perceive_bullets=True,
                    policy_sides=2,
                    window_size=args.window_size or None,
                    execution="sequential",
                    enqueue_order="env-first",
                )
                result.append({"pair": pair, "arm": label, **row})
                print(label, row["median_ms"], flush=True)
    finally:
        perception._line_of_sight_clear = original
    return result


def throughput(args):
    """Environment-only batch scaling, with training's reset and perception path."""
    if args.backend == "warp":
        perception._line_of_sight_clear = load_warp(args)
    wrapper = make_wrapper(args.scenario, args.batch, bullets=False, device=args.device)
    action = torch.zeros(
        (args.batch, wrapper.state.max_ships, 3), dtype=torch.int32, device=args.device
    )
    action[..., 2] = 1
    for _ in range(10):
        wrapper.step(action)
    realtime.warm_device(torch.device(args.device))
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    result = samples(lambda: wrapper.step(action), args.steps, args.device, window=True)
    result["env_transitions_per_second"] = 1000 * args.batch / result["window_mean_ms"]
    result["ship_decisions_per_second"] = (
        result["env_transitions_per_second"] * wrapper.state.max_ships
    )
    if args.device == "cuda":
        result["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
        result["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("profile", "los", "realtime", "throughput", "render", "tick-compile"),
        default="profile",
    )
    parser.add_argument("--scenario", choices=realtime.SCENARIOS, default="5v5")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--policy-compile", default="none")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--window-size", type=int, default=900)
    parser.add_argument("--warp-path", type=Path)
    parser.add_argument(
        "--backend",
        choices=("compiled", "warp"),
        default="compiled",
        help=("Realtime: compiled real LOS or Warp. Throughput: reference Torch or Warp LOS."),
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if min(args.steps, args.pairs, args.batch) < 1:
        parser.error("steps, pairs, and batch must be positive")
    torch.set_num_threads(1)
    torch.manual_seed(99)
    metadata = {
        "torch": torch.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if args.device == "cuda" else "CPU",
        "threads": torch.get_num_threads(),
        "seed": 99,
        "revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "working_tree": subprocess.check_output(["git", "status", "--short"], text=True),
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
    }
    with torch.inference_mode():
        result = {
            "profile": profile,
            "los": los,
            "realtime": realtime_ab,
            "throughput": throughput,
            "render": render_ablation,
            "tick-compile": tick_compile,
        }[args.mode](args)
    metadata["dynamo_counters"] = {
        str(group): {str(key): value for key, value in counts.items()}
        for group, counts in torch._dynamo.utils.counters.items()
    }
    if args.mode == "throughput":
        metadata["los_backend"] = "warp" if args.backend == "warp" else "eager_reference"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"metadata": metadata, "result": result}, indent=2) + "\n")
    print(f"Wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
