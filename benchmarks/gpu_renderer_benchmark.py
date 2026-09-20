"""One-arm reference/GPU renderer benchmark for a deterministic Frontline scene.

Example (run under an external timeout):
    timeout 180s uv run --no-sync python benchmarks/gpu_renderer_benchmark.py \\
        --arm gpu --device cuda --samples 300 --warmup 30 --out gpu.json

The reference timing calls ``GameRenderer._draw_frame`` and excludes event
polling and display flip. GPU ``render`` includes command submission and flip.
This is an isolated renderer benchmark: no policies are loaded or evaluated.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import torch

SEED = 20260920
WINDOW_SIZE = 900
SHIP_COUNT = 100
FIELD_COUNT = 72


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _memory(device: torch.device) -> dict[str, int | None]:
    import resource

    result: dict[str, int | None] = {
        "process_max_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * (1 if sys.platform == "darwin" else 1024),
        "cuda_allocated_bytes": None,
        "cuda_reserved_bytes": None,
    }
    if device.type == "cuda" and torch.cuda.is_available():
        result["cuda_allocated_bytes"] = torch.cuda.memory_allocated(device)
        result["cuda_reserved_bytes"] = torch.cuda.memory_reserved(device)
    return result


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {
            "count": 0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "p99_ms": 0.0,
            "max_ms": 0.0,
            "miss_16_7ms": 0,
            "miss_33_3ms": 0,
        }
    ordered = sorted(values)

    def percentile(p: float) -> float:
        return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * p))]

    return {
        "count": len(values),
        "p50_ms": percentile(0.50),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "max_ms": ordered[-1],
        "miss_16_7ms": sum(value > 16.7 for value in values),
        "miss_33_3ms": sum(value > 33.3 for value in values),
    }


def _save(path: Path, result: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _make_env(device: torch.device):
    from boost_and_broadside.config.core import EnvConfig
    from boost_and_broadside.config.defaults import REWARDS
    from boost_and_broadside.env.wrapper import YemongEnvWrapper
    from boost_and_broadside.profiles import PROFILES

    profile = PROFILES["rl"]
    scale = 7000.0 / 2600.0
    frontline = replace(
        profile.frontline,
        zone_radius=profile.frontline.zone_radius * scale,
        zone_ring_radius=profile.frontline.zone_ring_radius * scale,
        playable_radius=profile.frontline.playable_radius * scale,
    )
    config = EnvConfig(
        num_ships=SHIP_COUNT,
        num_fields=FIELD_COUNT,
        max_bullets=profile.max_bullets,
        max_episode_steps=profile.max_episode_steps,
        action_repeat=profile.action_repeat,
        frontline=frontline,
        vision_range=profile.vision_range * scale,
        zones_occlude=profile.zones_occlude,
    )
    wrapper = YemongEnvWrapper(
        num_envs=1,
        ship_config=profile.ship_config,
        env_config=config,
        rewards=REWARDS,
        device=device,
        include_bullets=False,
        perceive_bullets=True,
    )
    wrapper.reset(seed=SEED)
    return wrapper, profile.ship_config, config


def run(args: argparse.Namespace) -> dict:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was selected but PyTorch reports CUDA unavailable")
    torch.manual_seed(SEED)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
    wrapper, ship_config, env_config = _make_env(device)
    state = wrapper.env.state
    actions = torch.zeros((1, SHIP_COUNT, 3), dtype=torch.int32, device=device)
    output = Path(args.out)

    renderer = None
    gpu_renderer = None
    snapshot = None
    gl_info: dict[str, str] = {}
    startup_start = time.perf_counter()
    if args.arm == "reference":
        os.environ.setdefault("HEADLESS", "1")
        from boost_and_broadside.ui.renderer import GameRenderer, RenderConfig, VisionMode

        renderer = GameRenderer(
            ship_config,
            RenderConfig(
                window_size=WINDOW_SIZE,
                fps=60,
                show_ui=False,
                vision_mode=VisionMode.TEAM_0,
                zone_occlusion=env_config.zones_occlude,
            ),
        )
    else:
        if args.moderngl_path:
            sys.path.insert(0, str(Path(args.moderngl_path).expanduser().resolve()))
        from boost_and_broadside.ui.gpu_renderer import FrontlineGPURenderer
        from boost_and_broadside.ui.gpu_snapshot import (
            make_packed_render_snapshot,
            make_render_snapshot,
        )

        snapshot_builder = (
            make_packed_render_snapshot if args.arm == "gpu-packed" else make_render_snapshot
        )
        snapshot = snapshot_builder(
            state,
            world_size=ship_config.world_size,
            visibility=wrapper.last_visibility,
            zones_occlude=env_config.zones_occlude,
        )
        gpu_renderer = FrontlineGPURenderer(snapshot, (WINDOW_SIZE, WINDOW_SIZE))
        gpu_renderer.perspective = "team0"
        context_info = gpu_renderer._ctx.info
        gl_info = {
            "vendor": str(context_info.get("GL_VENDOR", "unknown")),
            "renderer": str(context_info.get("GL_RENDERER", "unknown")),
            "version": str(context_info.get("GL_VERSION", "unknown")),
        }
    startup_seconds = time.perf_counter() - startup_start

    action_trace: list[dict[str, int]] = []
    timings: dict[str, list[float]] = {
        "reference_draw_frame_ms": [],
        "snapshot_extraction_ms": [],
        "gpu_render_including_flip_ms": [],
        "snapshot_plus_gpu_render_ms": [],
    }
    result: dict = {
        "schema_version": 1,
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "command": [sys.executable, *sys.argv],
        "arm": args.arm,
        "renderer_only": True,
        "policies_loaded_or_evaluated": False,
        "revision": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "device": str(device),
            "cuda_runtime": torch.version.cuda,
            "cuda_device": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "gl": gl_info,
        },
        "configuration": {
            "ships": SHIP_COUNT,
            "fields": FIELD_COUNT,
            "projectile_perception": True,
            "team_fog": True,
            "perspective": "team0",
            "window": [WINDOW_SIZE, WINDOW_SIZE],
            "seed": SEED,
            "device": str(device),
            "ship_config": asdict(ship_config),
            "env_config": asdict(env_config),
        },
        "action_trace": action_trace,
        "warmup_frames": args.warmup,
        "sample_frames": args.samples,
        "startup_and_shader_compile_seconds": startup_seconds,
        "timings": timings,
        "memory_after_startup": _memory(device),
        "flip_note": (
            "Excluded: direct GameRenderer._draw_frame call; no event polling or display flip."
            if args.arm == "reference"
            else "Included: FrontlineGPURenderer.render calls pygame.display.flip()."
        ),
    }
    _save(output, result)

    try:
        total_frames = args.warmup + args.samples
        for frame in range(total_frames):
            actions.fill_(0)
            wrapper.step_interactive(actions, auto_reset=True)
            action_trace.append({"frame": frame, "action_code": 0})
            if args.arm == "reference":
                started = time.perf_counter()
                renderer._draw_frame(state, visibility=wrapper.last_visibility)
                elapsed = (time.perf_counter() - started) * 1000
                if frame >= args.warmup:
                    timings["reference_draw_frame_ms"].append(elapsed)
            else:
                extraction_start = time.perf_counter()
                if args.arm == "gpu-packed":
                    previous = snapshot
                    snapshot = make_packed_render_snapshot(
                        state,
                        world_size=ship_config.world_size,
                        visibility=wrapper.last_visibility,
                        zones_occlude=env_config.zones_occlude,
                    )
                else:
                    snapshot = make_render_snapshot(
                        state,
                        world_size=ship_config.world_size,
                        previous=snapshot,
                        visibility=wrapper.last_visibility,
                        zones_occlude=env_config.zones_occlude,
                    )
                extraction_ms = (time.perf_counter() - extraction_start) * 1000
                render_start = time.perf_counter()
                if args.arm == "gpu-packed":
                    gpu_renderer.render_packed(snapshot, previous)
                else:
                    gpu_renderer.render(snapshot)
                render_ms = (time.perf_counter() - render_start) * 1000
                if frame >= args.warmup:
                    timings["snapshot_extraction_ms"].append(extraction_ms)
                    timings["gpu_render_including_flip_ms"].append(render_ms)
                    timings["snapshot_plus_gpu_render_ms"].append(extraction_ms + render_ms)
            if frame == args.warmup - 1 or frame == total_frames - 1:
                result["completed_frames"] = frame + 1
                result["timing_summaries"] = {
                    name: _summary(samples) for name, samples in timings.items()
                }
                result["memory"] = _memory(device)
                _save(output, result)
        result["status"] = "complete"
    finally:
        if gpu_renderer is not None:
            gpu_renderer.close()
        elif renderer is not None:
            import pygame

            pygame.quit()
    result["completed_utc"] = datetime.now(UTC).isoformat()
    _save(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("reference", "gpu", "gpu-packed"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--moderngl-path", help="Optional directory containing ModernGL modules")
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--samples", type=int, default=300)
    parser.add_argument("--out", default="renderer-benchmark.json")
    args = parser.parse_args()
    if args.warmup < 0 or args.samples < 1:
        parser.error("--warmup must be non-negative and --samples must be positive")
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
