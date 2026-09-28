"""Phase-3 exit gate: hidden-state forecast against persistence and dead reckoning.

A learned next-state head is only worth its cost if a hidden ship's belief tracks
better than the two things available without one:

* **persistence** -- the belief standing still, which is what a zero mean delta is;
* **dead reckoning** -- the belief carried forward one decision on its own believed
  velocity, which is the strongest estimate a policy can form with no model at all.

The comparison is the production diagnostic, read out of the metrics the trainer
already logs (``belief/<bucket>/position_px`` against ``persist_position_px`` and
``reckon_position_px``), so this measures the shipped path rather than a
reimplementation of it.

    uv run --no-sync python docs/internal/perception-belief-phase3-baselines-probe.py \
        --device cuda --updates 12 --envs 256 \
        --out docs/internal/perception-belief-phase3-baselines.json

The head starts from random weights, so early updates are expected to lose to dead
reckoning by a wide margin: what the artifact establishes is that the comparison is
wired, calibrated in physical units, and moving in the right direction, not that a
few minutes of training beats a linear extrapolator.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import subprocess
import tempfile
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.profiles import resolve_named_profile
from boost_and_broadside.train.rl.ppo import PPOTrainer

#: The series compared, and the baseline each is measured against.
_COMPARISONS = (
    ("position_px", "persist_position_px", "reckon_position_px"),
    ("velocity_px_s", "persist_velocity_px_s", None),
)
_BUCKETS = (
    "visible",
    "hidden",
    "hidden_age_0_0.1s",
    "hidden_age_0.1_0.5s",
    "hidden_age_0.5_1s",
    "hidden_age_1_2s",
    "hidden_age_2_5s",
    "hidden_age_5_10s",
    "hidden_age_10_30s",
    "hidden_age_30_inf_s",
)


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
    except Exception:
        return "unknown"


def _trainer(args: argparse.Namespace, checkpoint_dir: Path) -> PPOTrainer:
    """The production ``rl`` profile, narrowed to a probe-sized batch.

    Uncompiled on purpose: this measures forecast quality, and Inductor startup
    on a laptop GPU costs more than the whole probe.
    """

    profile = resolve_named_profile("rl")
    scale = profile.train_config.scales[0]
    train_config = dataclasses.replace(
        profile.train_config,
        scales=(dataclasses.replace(scale, num_envs=args.envs),),
        num_steps=args.steps,
        rollouts_per_update=1,
        total_timesteps=args.envs * args.steps * args.updates,
        checkpoint_dir=str(checkpoint_dir),
    )
    return PPOTrainer(
        train_config=train_config,
        model_config=profile.model_config,
        ship_config=profile.ship_config,
        device=args.device,
        use_wandb=False,
        scripted_agent=StochasticScriptedAgent(profile.ship_config, StochasticAgentConfig()),
        compile_mode=None,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    output = Path(args.out)
    # Outside the repository: the trainer writes a milestone snapshot whatever
    # the interval, and this probe's artifact is the JSON, not the weights.
    checkpoint_dir = Path(tempfile.mkdtemp(prefix="phase3-baselines-"))
    started = time.perf_counter()
    result: dict[str, Any] = {
        "schema_version": 1,
        "what": (
            "Phase-3 hidden-state forecast against persistence and dead-reckoning "
            "baselines, read from the production belief diagnostics."
        ),
        "generated_utc": datetime.now(UTC).isoformat(),
        "commit": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "runtime": {
            "device": args.device,
            "torch": torch.__version__,
            "platform": platform.platform(),
            "gpu": (
                torch.cuda.get_device_name(0)
                if args.device == "cuda" and torch.cuda.is_available()
                else None
            ),
        },
        "configuration": {
            "profile": "rl",
            "compile_mode": None,
            "num_envs": args.envs,
            "num_steps": args.steps,
            "updates": args.updates,
        },
        "status": "running",
        "updates": [],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=1))

    trainer = _trainer(args, checkpoint_dir)
    captured: list[dict[str, float]] = []

    def capture(metrics: dict, step: int) -> None:
        keep = {
            key: float(value)
            for key, value in metrics.items()
            if key.startswith("belief/") or key.startswith("loss/next_state")
        }
        keep["global_step"] = float(step)
        captured.append(keep)

    trainer._enqueue_log = capture
    trainer.train()

    for record in captured:
        entry: dict[str, Any] = {"global_step": record.get("global_step")}
        entry["loss_next_state"] = record.get("loss/next_state")
        for bucket in _BUCKETS:
            cells: dict[str, Any] = {}
            for series, persist, reckon in _COMPARISONS:
                model = record.get(f"belief/{bucket}/{series}")
                if model is None:
                    continue
                cell: dict[str, Any] = {"model": model}
                cell["persistence"] = record.get(f"belief/{bucket}/{persist}")
                if reckon is not None:
                    cell["dead_reckoning"] = record.get(f"belief/{bucket}/{reckon}")
                best = min(
                    value
                    for value in (cell["persistence"], cell.get("dead_reckoning"))
                    if value is not None
                )
                cell["model_over_best_baseline"] = (
                    None if best == 0.0 else model / best
                )
                cells[series] = cell
            if cells:
                entry[bucket] = cells
        result["updates"].append(entry)

    result["wall_seconds"] = time.perf_counter() - started
    result["status"] = "complete"
    output.write_text(json.dumps(result, indent=1))
    print(f"wrote {output}")
    for entry in result["updates"]:
        hidden = entry.get("hidden", {}).get("position_px")
        if hidden is None:
            continue
        print(
            f"  step {entry['global_step']:>10,.0f}  hidden position_px "
            f"model {hidden['model']:.2f}  persist {hidden['persistence']:.2f}  "
            f"reckon {hidden['dead_reckoning']:.2f}  "
            f"ratio {hidden['model_over_best_baseline']:.2f}"
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--envs", type=int, default=256)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--updates", type=int, default=12)
    parser.add_argument(
        "--out", default="docs/internal/perception-belief-phase3-baselines.json"
    )
    run(parser.parse_args())


if __name__ == "__main__":
    main()
