"""CUDA subphase profiler for the lean 50v50 interactive frame.

This is a short diagnostic run, not a latency benchmark: every phase is
explicitly synchronized so its CUDA-event time can be attributed. Host wall
time is recorded alongside it because team-view selection is mostly Python.
No renderer or policy compilation is required.

Example:
    timeout 180s uv run --no-sync python benchmarks/deadline_subphase_profile.py \\
        --compile none --seed 271828 --warmup 8 --samples 24 \\
        --out artifacts/deadline-experiment/analysis/subphases.json
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from realtime_latency import SCENARIOS, STAGES, scenario_config

from boost_and_broadside.config.defaults import REWARDS
from boost_and_broadside.env.wrapper import YemongEnvWrapper
from boost_and_broadside.evaluation.match import merge_team_actions
from boost_and_broadside.profiles import PROFILES
from boost_and_broadside.runtime.actions import PendingActionState
from boost_and_broadside.train.rl.belief import BeliefTracker, legal_policy_view
from boost_and_broadside.train.rl.policy_io import build_policy, compile_policy

SEED_DEFAULT = 271828
SCENARIO = "50v50"


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "p50_ms": 0.0, "p95_ms": 0.0, "p99_ms": 0.0, "max_ms": 0.0}
    ordered = sorted(values)

    def percentile(p: float) -> float:
        return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * p))]

    return {
        "count": len(values),
        "p50_ms": statistics.median(values),
        "p95_ms": percentile(0.95),
        "p99_ms": percentile(0.99),
        "max_ms": ordered[-1],
    }


def _profile_call(fn: Callable[[], Any], device: torch.device) -> tuple[Any, float, float]:
    """Return result, CUDA-event ms, and synchronized host wall ms."""
    torch.cuda.synchronize(device)
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter()
    start_event.record()
    result = fn()
    end_event.record()
    torch.cuda.synchronize(device)
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    return result, start_event.elapsed_time(end_event), wall_ms


def _make_wrapper(device: torch.device, seed: int):
    profile = PROFILES["rl"]
    ships, fields, scale = SCENARIOS[SCENARIO]
    env_config = scenario_config(ships, fields, scale)
    wrapper = YemongEnvWrapper(
        num_envs=1,
        ship_config=profile.ship_config,
        env_config=env_config,
        rewards=REWARDS,
        device=device,
        include_bullets=False,
        perceive_bullets=True,
    )
    wrapper.reset(seed=seed)
    return wrapper, profile, env_config, ships


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this profiler")
    device = torch.device("cuda")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    output = Path(args.out)

    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "initializing",
        "started_utc": datetime.now(UTC).isoformat(),
        "revision": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": torch.cuda.get_device_name(device),
        },
        "configuration": {
            "scenario": SCENARIO,
            "ships": SCENARIOS[SCENARIO][0],
            "fields": SCENARIOS[SCENARIO][1],
            "resolution": [900, 900],
            "projectile_perception": True,
            "team_fog": True,
            "two_distinct_random_policies": True,
            "renderer": None,
            "environment_step": "YemongEnvWrapper.step_interactive(auto_reset=False)",
            "action_repeat": PROFILES["rl"].action_repeat,
            "seed": args.seed,
            "policy_compile": args.compile,
            "torch_threads": torch.get_num_threads(),
        },
        "warmup_frames": args.warmup,
        "sample_frames": args.samples,
        "startup_seconds": None,
        "frame_timings": [],
        "phase_samples": {},
    }
    _save(output, result)

    startup = time.perf_counter()
    wrapper, profile, env_config, ships = _make_wrapper(device, args.seed)
    ship_config = profile.ship_config
    model_config = replace(profile.model_config, **STAGES["baseline"])
    compile_mode = None if args.compile == "none" else args.compile
    sides: list[dict[str, Any]] = []
    for _team in range(2):
        policy = build_policy(
            model_config,
            ship_config,
            num_value_components=12,
            num_ships=ships,
            team_pma_k=(0, 1),
        ).to(device)
        policy.eval()
        policy.requires_grad_(False)
        sides.append(
            {
                "policy": compile_policy(policy, compile_mode),
                "belief": BeliefTracker(
                    1,
                    ships,
                    ship_config.dt * env_config.action_repeat,
                    ship_config,
                    device,
                ),
                "hidden": policy.initial_hidden(1, policy.recurrent_token_count(ships), device),
            }
        )
    action_state = PendingActionState.allocate(1, ships, device)
    team1_mask = torch.ones(1, dtype=torch.bool, device=device)
    result["startup_seconds"] = time.perf_counter() - startup
    result["configuration"]["ship_config"] = asdict(ship_config)
    result["configuration"]["env_config"] = asdict(env_config)
    result["status"] = "warming_up"
    _save(output, result)

    tick_original = wrapper.env.tick
    phase_collector: dict[str, list[float]] | None = None

    def timed_method(label: str, fn: Callable[[], Any]) -> Any:
        if phase_collector is None:
            return fn()
        value, cuda_ms, wall_ms = _profile_call(fn, device)
        phase_collector.setdefault(f"{label}.cuda_ms", []).append(cuda_ms)
        phase_collector.setdefault(f"{label}.wall_ms", []).append(wall_ms)
        return value

    def profiled_tick(actions: torch.Tensor, **kwargs: Any):
        return timed_method("environment.tensor_env_tick", lambda: tick_original(actions, **kwargs))

    wrapper.env.tick = profiled_tick

    def frame(profile_frame: bool) -> None:
        nonlocal phase_collector
        phase_collector = {} if profile_frame else None
        actions_by_team = []
        predictions = []
        for team, side in enumerate(sides):
            team_label = f"team{team}"

            # One build per side, from the authoritative state and that side's own
            # belief. There is no shared observation to take a perspective on, so
            # the old ``for_team``/``flip_team``/``compose`` trio is one phase.
            def _compose(team=team, side=side):
                composed = legal_policy_view(
                    side["belief"],
                    wrapper.env.state,
                    ship_config,
                    wrapper.last_visibility,
                    team,
                    num_ships=ships,
                    pending_action=action_state.pending,
                )
                return composed.flip_team(ships, mask=team1_mask) if team == 1 else composed

            if profile_frame:
                view, cuda_ms, wall_ms = _profile_call(_compose, device)
                phase_collector.setdefault(f"{team_label}.legal_view.cuda_ms", []).append(cuda_ms)
                phase_collector.setdefault(f"{team_label}.legal_view.wall_ms", []).append(wall_ms)
                action_result, cuda_ms, wall_ms = _profile_call(
                    lambda: side["policy"].get_action_and_value(
                        view, side["hidden"], return_enemy_action=True
                    ),
                    device,
                )
                action, _logprob, _value, prediction, enemy_logits, side["hidden"] = action_result
                phase_collector.setdefault(f"{team_label}.get_action_and_value.cuda_ms", []).append(
                    cuda_ms
                )
                phase_collector.setdefault(f"{team_label}.get_action_and_value.wall_ms", []).append(
                    wall_ms
                )
                _result, cuda_ms, wall_ms = _profile_call(
                    lambda: side["belief"].advance(prediction, enemy_logits), device
                )
                phase_collector.setdefault(f"{team_label}.belief_advance.cuda_ms", []).append(
                    cuda_ms
                )
                phase_collector.setdefault(f"{team_label}.belief_advance.wall_ms", []).append(
                    wall_ms
                )
            else:
                view = _compose()
                action, _logprob, _value, prediction, enemy_logits, side["hidden"] = side[
                    "policy"
                ].get_action_and_value(view, side["hidden"], return_enemy_action=True)
                side["belief"].advance(prediction, enemy_logits)
            actions_by_team.append(action)
            predictions.append(prediction)

        if profile_frame:
            decided, cuda_ms, wall_ms = _profile_call(
                lambda: merge_team_actions(
                    actions_by_team[0], actions_by_team[1], wrapper.env.state.ship_team_id
                ).int(),
                device,
            )
            phase_collector.setdefault("action.merge.cuda_ms", []).append(cuda_ms)
            phase_collector.setdefault("action.merge.wall_ms", []).append(wall_ms)
        else:
            decided = merge_team_actions(
                actions_by_team[0], actions_by_team[1], wrapper.env.state.ship_team_id
            ).int()

        _, dones, truncated, info = wrapper.step_interactive(
            action_state.applied_action(), auto_reset=False, observe=False
        )
        finished = dones | truncated
        action_state.commit(
            decided,
            info["actuator_contiguous"],
            finished,
        )

        if bool(finished.any()):
            wrapper.reset()
            action_state.reset(finished)
            for side in sides:
                side["belief"].reset(finished)
                side["hidden"] = side["policy"].reset_hidden_for_envs(
                    side["hidden"], finished, side["policy"].recurrent_token_count(ships)
                )

        if profile_frame:
            for name, values in phase_collector.items():
                result["phase_samples"].setdefault(name, []).append(sum(values))
            result["frame_timings"].append(
                {
                    "frame": len(result["frame_timings"]),
                    "action_code_trace_note": (
                        "two independently sampled policy outputs; full tensors omitted"
                    ),
                }
            )
            phase_collector = None

    try:
        for _ in range(args.warmup):
            frame(False)
        result["status"] = "profiling"
        _save(output, result)
        for _ in range(args.samples):
            frame(True)
            result["completed_samples"] = len(result["frame_timings"])
            result["summaries"] = {
                name: _summary(values) for name, values in result["phase_samples"].items()
            }
            _save(output, result)
        result["status"] = "complete"
    finally:
        wrapper.env.tick = tick_original
    result["completed_utc"] = datetime.now(UTC).isoformat()
    _save(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compile", choices=("none", "default", "reduce-overhead"), default="none")
    parser.add_argument("--seed", type=int, default=SEED_DEFAULT)
    parser.add_argument("--warmup", type=int, default=8)
    parser.add_argument("--samples", type=int, default=24)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.warmup < 0 or args.samples < 1:
        parser.error("--warmup must be non-negative and --samples must be positive")
    print(json.dumps(run(args), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
