"""Alternating complete-stack A/B for the 50v50 deadline finalist."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
from realtime_latency import measure


def _summary(values: list[float]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[int(0.95 * (len(ordered) - 1))],
        "p99_ms": ordered[int(0.99 * (len(ordered) - 1))],
        "max_ms": ordered[-1],
        "deadline_misses": sum(value > 1000.0 / 30.0 for value in ordered),
    }


def _save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=15)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--seed", type=int, default=5150)
    parser.add_argument("--renderer", choices=("none", "packed"), default="none")
    parser.add_argument("--moderngl-path", default="/tmp/bnb-deadline-moderngl")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        parser.error("CUDA is required")

    payload = {
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "command": [sys.executable, *sys.argv],
        "scope": "50v50 Frontline simulation, two distinct random policies, and optional render",
        "configuration": {
            "pairs": args.pairs,
            "warmup": args.warmup,
            "samples_per_arm": args.samples,
            "seed": args.seed,
            "renderer": args.renderer,
            "resolution": [900, 900] if args.renderer == "packed" else None,
            "policy_compile": "default",
            "projectile_perception": True,
            "fog": args.renderer == "packed",
        },
        "runs": [],
    }
    _save(args.out, payload)
    for pair in range(args.pairs):
        order = ("reference", "candidate") if pair % 2 == 0 else ("candidate", "reference")
        pair_rows = {}
        for arm in order:
            candidate = arm == "candidate"
            row = measure(
                "50v50",
                torch.device("cuda"),
                steps=args.samples,
                warmup=args.warmup,
                compile_mode="default",
                seed=args.seed + pair,
                stage="baseline",
                perceive_bullets=True,
                policy_sides=2,
                window_size=900 if args.renderer == "packed" else None,
                execution="sequential",
                enqueue_order="env-first",
                environment_step="step_interactive",
                renderer_backend="gpu-packed" if args.renderer == "packed" else None,
                moderngl_path=args.moderngl_path if args.renderer == "packed" else None,
                belief_compile_mode="default" if candidate else None,
                perception_compile_mode="default" if candidate else None,
                cuda_graph_tick=candidate,
            )
            row.update({"pair": pair, "arm": arm, "order": list(order)})
            payload["runs"].append(row)
            pair_rows[arm] = row
            _save(args.out, payload)
        payload.setdefault("pair_checks", []).append(
            {
                "pair": pair,
                "action_trace_equal": pair_rows["reference"]["action_trace"]
                == pair_rows["candidate"]["action_trace"],
            }
        )
        _save(args.out, payload)

    payload["aggregate"] = {}
    for arm in ("reference", "candidate"):
        rows = [row for row in payload["runs"] if row["arm"] == arm]
        samples = [sample for row in rows for sample in row["raw_samples_ms"]]
        payload["aggregate"][arm] = {
            **_summary(samples),
            "pair_p50_ms": [row["median_ms"] for row in rows],
            "phase_ms": {
                name: statistics.mean(row["phase_ms"][name] for row in rows)
                for name in ("policy", "env_step", "render")
            },
            "first_call_frame_ms": [row["first_call_frame_ms"] for row in rows],
            "peak_allocated_mib": max(row["peak_allocated_mib"] for row in rows),
        }
    payload["status"] = "complete"
    payload["completed_utc"] = datetime.now(UTC).isoformat()
    _save(args.out, payload)


if __name__ == "__main__":
    main()
