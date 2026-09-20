"""Compare eager and compiled belief trackers on one fixed 50v50 trace.

This verifies torch.compile(default, dynamic=False) parity for both independent
team views. It is a correctness harness, not a speed benchmark. Run it in a
separate process because the first compiled invocation can take a while:

    timeout 20m uv run --no-sync python benchmarks/belief_compile_parity.py \\
        --device cuda --seed 271828 --steps 24 \\
        --out artifacts/deadline-experiment/analysis/belief-parity.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from realtime_latency import SCENARIOS, scenario_config

from boost_and_broadside.config.defaults import REWARDS
from boost_and_broadside.constants import NUM_POWER_ACTIONS, NUM_SHOOT_ACTIONS, NUM_TURN_ACTIONS
from boost_and_broadside.env.observation import YemongObservation
from boost_and_broadside.env.wrapper import YemongEnvWrapper
from boost_and_broadside.profiles import PROFILES
from boost_and_broadside.train.rl.belief import BeliefTracker
from boost_and_broadside.train.rl.features import build_standard_coordinator

ATOL = 2e-6
RTOL = 2e-6
STATE_FIELDS = ("valid", "age_steps", "predicted_targets", "team_id", "radius", "clamp_events")


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", *args], capture_output=True, text=True, check=True, timeout=5
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _save(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")
    temporary.replace(path)


def _clone_observation(observation: YemongObservation) -> YemongObservation:
    return YemongObservation(
        data={key: value.detach().clone() for key, value in observation.data.items()},
        bullets=(
            None
            if observation.bullets is None
            else {key: value.detach().clone() for key, value in observation.bullets.items()}
        ),
        team1_data=(
            None
            if observation.team1_data is None
            else {key: value.detach().clone() for key, value in observation.team1_data.items()}
        ),
        team1_bullets=(
            None
            if observation.team1_bullets is None
            else {key: value.detach().clone() for key, value in observation.team1_bullets.items()}
        ),
    )


def _compare_tensor(label: str, left: torch.Tensor, right: torch.Tensor, errors: list[str]) -> None:
    if left.shape != right.shape or left.dtype != right.dtype:
        errors.append(
            f"{label}: metadata differs (left={tuple(left.shape)}/{left.dtype}, "
            f"right={tuple(right.shape)}/{right.dtype})"
        )
        return
    if left.dtype.is_floating_point or left.dtype.is_complex:
        if not torch.allclose(left, right, atol=ATOL, rtol=RTOL, equal_nan=True):
            difference = (left - right).abs()
            errors.append(
                f"{label}: float mismatch; max_abs={float(difference.max().item())}, "
                f"atol={ATOL}, rtol={RTOL}"
            )
    elif not torch.equal(left, right):
        errors.append(f"{label}: exact discrete mismatch")


def _compare_mapping(
    label: str,
    left: dict,
    right: dict,
    errors: list[str],
) -> None:
    if left.keys() != right.keys():
        errors.append(f"{label}: keys differ")
        return
    for key in left:
        _compare_tensor(f"{label}.{key}", left[key], right[key], errors)


def _compare_observation(
    label: str,
    left: YemongObservation,
    right: YemongObservation,
    errors: list[str],
) -> None:
    _compare_mapping(f"{label}.data", left.data, right.data, errors)
    if (left.bullets is None) != (right.bullets is None):
        errors.append(f"{label}.bullets: optional-presence differs")
    elif left.bullets is not None and right.bullets is not None:
        _compare_mapping(f"{label}.bullets", left.bullets, right.bullets, errors)


def _compare_tracker(
    label: str,
    eager: BeliefTracker,
    compiled: BeliefTracker,
    errors: list[str],
) -> None:
    for name in STATE_FIELDS:
        _compare_tensor(
            f"{label}.tracker.{name}", getattr(eager, name), getattr(compiled, name), errors
        )


def _tensor_digest(tensors: list[torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for tensor in tensors:
        digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def _observation_tensors(observation: YemongObservation) -> list[torch.Tensor]:
    tensors = list(observation.data.values())
    if observation.bullets is not None:
        tensors.extend(observation.bullets.values())
    return tensors


def _build_trace(device: torch.device, seed: int, steps: int):
    profile = PROFILES["rl"]
    ships, fields, scale = SCENARIOS["50v50"]
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
    observation = wrapper.reset(seed=seed)
    coordinator = build_standard_coordinator(
        profile.ship_config, local_presence=profile.model_config.local_presence
    )
    action_generator = torch.Generator(device="cpu").manual_seed(seed + 1)
    prediction_generator = torch.Generator(device="cpu").manual_seed(seed + 2)
    action_trace = torch.stack(
        (
            torch.randint(NUM_POWER_ACTIONS, (steps, 1, ships), generator=action_generator),
            torch.randint(NUM_TURN_ACTIONS, (steps, 1, ships), generator=action_generator),
            torch.randint(NUM_SHOOT_ACTIONS, (steps, 1, ships), generator=action_generator),
        ),
        dim=-1,
    ).to(dtype=torch.int32, device=device)
    prediction_trace = (
        torch.randn(
            (steps, 2, 1, ships, coordinator.total_prediction_dimension),
            generator=prediction_generator,
        )
        .mul_(0.05)
        .to(device)
    )
    views: list[tuple[YemongObservation, YemongObservation]] = []
    team1_mask = torch.ones(1, dtype=torch.bool, device=device)
    for step in range(steps):
        team0 = observation.for_team(0)
        team1 = observation.for_team(1).flip_team(ships, mask=team1_mask)
        views.append((_clone_observation(team0), _clone_observation(team1)))
        observation, dones, truncated, _info = wrapper.step_interactive(
            action_trace[step], auto_reset=False
        )
        if bool((dones | truncated).any()):
            observation = wrapper.reset()
    trace_tensors = [action_trace, prediction_trace]
    for team_views in views:
        for view in team_views:
            trace_tensors.extend(_observation_tensors(view))
    trace_digest = _tensor_digest(trace_tensors)
    return wrapper, coordinator, env_config, views, prediction_trace, trace_digest


@torch.inference_mode()
def run(args: argparse.Namespace) -> dict[str, Any]:
    global ATOL, RTOL
    ATOL, RTOL = args.atol, args.rtol
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    output = Path(args.out)
    started = time.perf_counter()
    result: dict[str, Any] = {
        "schema_version": 1,
        "status": "building_trace",
        "started_utc": datetime.now(UTC).isoformat(),
        "revision": _git("rev-parse", "HEAD"),
        "dirty": bool(_git("status", "--porcelain")),
        "software": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "configuration": {
            "scenario": "50v50",
            "ships": SCENARIOS["50v50"][0],
            "fields": SCENARIOS["50v50"][1],
            "projectile_perception": True,
            "team_fog": True,
            "team_views": ["team0", "team1 flipped to ego-team0"],
            "steps": args.steps,
            "seed": args.seed,
            "prediction_trace": "seeded CPU torch.randn * 0.05, shared by eager/compiled",
            "action_trace": "seeded legal power/turn/shoot randint actions on CPU, shared",
            "compiler": "torch.compile(mode='default', dynamic=False)",
            "float_tolerance": {"atol": ATOL, "rtol": RTOL},
            "integer_boolean_discrete_comparison": "exact torch.equal",
        },
        "args": vars(args),
        "startup_and_trace_build_seconds": None,
        "compiled_first_call_seconds": {},
        "trace_digest_sha256": None,
        "steps_compared": 0,
        "errors": [],
        "retained_observation_ownership": {"checked": False, "passed": None},
    }
    _save(output, result)

    wrapper, coordinator, env_config, views, predictions, trace_digest = _build_trace(
        device, args.seed, args.steps
    )
    profile = PROFILES["rl"]
    trackers = {
        f"team{team}": {
            "eager": BeliefTracker(
                1,
                SCENARIOS["50v50"][0],
                profile.ship_config.dt * env_config.action_repeat,
                coordinator,
                device,
            ),
            "compiled": BeliefTracker(
                1,
                SCENARIOS["50v50"][0],
                profile.ship_config.dt * env_config.action_repeat,
                coordinator,
                device,
            ),
        }
        for team in range(2)
    }
    compiled_methods = {
        name: (
            torch.compile(pair["compiled"].compose, mode="default", dynamic=False),
            torch.compile(pair["compiled"].advance, mode="default", dynamic=False),
        )
        for name, pair in trackers.items()
    }
    result["startup_and_trace_build_seconds"] = time.perf_counter() - started
    result["trace_digest_sha256"] = trace_digest
    result["status"] = "comparing"
    _save(output, result)

    retained: dict[str, list[tuple[YemongObservation, YemongObservation]]] = {
        name: {"eager": [], "compiled": []} for name in trackers
    }
    first_call_seconds: dict[str, float] = {}
    errors: list[str] = result["errors"]

    try:
        for step, team_views in enumerate(views):
            for team, view in enumerate(team_views):
                name = f"team{team}"
                pair = trackers[name]
                prediction = predictions[step, team]
                eager_start = time.perf_counter()
                eager_composed = pair["eager"].compose(view)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                eager_compose_seconds = time.perf_counter() - eager_start
                compiled_start = time.perf_counter()
                compiled_composed = compiled_methods[name][0](view)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                compiled_compose_seconds = time.perf_counter() - compiled_start
                if step == 0:
                    first_call_seconds[f"{name}.eager_compose"] = eager_compose_seconds
                    first_call_seconds[f"{name}.compiled_compose_including_compile"] = (
                        compiled_compose_seconds
                    )
                _compare_observation(
                    f"step{step}.{name}.compose", eager_composed, compiled_composed, errors
                )
                _compare_tracker(
                    f"step{step}.{name}.after_compose", pair["eager"], pair["compiled"], errors
                )
                retained[name]["eager"].append((eager_composed, _clone_observation(eager_composed)))
                retained[name]["compiled"].append(
                    (compiled_composed, _clone_observation(compiled_composed))
                )
                eager_advance_start = time.perf_counter()
                pair["eager"].advance(eager_composed, prediction)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                eager_advance_seconds = time.perf_counter() - eager_advance_start
                compiled_advance_start = time.perf_counter()
                compiled_methods[name][1](compiled_composed, prediction)
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                compiled_advance_seconds = time.perf_counter() - compiled_advance_start
                if step == 0:
                    first_call_seconds[f"{name}.eager_advance"] = eager_advance_seconds
                    first_call_seconds[f"{name}.compiled_advance_including_compile"] = (
                        compiled_advance_seconds
                    )
                _compare_tracker(
                    f"step{step}.{name}.after_advance", pair["eager"], pair["compiled"], errors
                )
            result["steps_compared"] = step + 1
            result["compiled_first_call_seconds"] = first_call_seconds
            result["error_count"] = len(errors)
            _save(output, result)

        retained_errors: list[str] = []
        for name, arms in retained.items():
            for arm, observations in arms.items():
                for index, (reference, expected) in enumerate(observations):
                    _compare_observation(
                        f"retained.{name}.{arm}.step{index}", reference, expected, retained_errors
                    )
        errors.extend(retained_errors)
        result["retained_observation_ownership"] = {
            "checked": True,
            "passed": not retained_errors,
            "retained_outputs_per_team_and_arm": args.steps,
        }
        result["clamp_events"] = {
            name: {arm: int(pair[arm].clamp_events.item()) for arm in ("eager", "compiled")}
            for name, pair in trackers.items()
        }
        result["tracker_state_final"] = {
            name: {
                arm: {
                    "valid_count": int(pair[arm].valid.sum().item()),
                    "max_age_steps": int(pair[arm].age_steps.max().item()),
                    "predicted_targets_sha256": _tensor_digest([pair[arm].predicted_targets]),
                }
                for arm in ("eager", "compiled")
            }
            for name, pair in trackers.items()
        }
        result["errors"] = errors
        result["error_count"] = len(errors)
        result["parity_passed"] = not errors
        result["status"] = "complete" if not errors else "parity_failed"
    except Exception as error:
        result["status"] = "error"
        result["error"] = f"{type(error).__name__}: {error}"
        _save(output, result)
        raise
    finally:
        result["completed_utc"] = datetime.now(UTC).isoformat()
        _save(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=271828)
    parser.add_argument("--steps", type=int, default=24)
    parser.add_argument("--atol", type=float, default=ATOL)
    parser.add_argument("--rtol", type=float, default=RTOL)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.steps < 2:
        parser.error("--steps must be at least 2 for retained-output ownership checks")
    result = run(args)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result.get("parity_passed", False):
        sys.exit(2)


if __name__ == "__main__":
    main()
