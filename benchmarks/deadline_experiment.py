"""A/B deadline experiment for the shared perceived-observation builder.

Run the harness in a fresh process with an external timeout (for example,
``timeout 20m uv run --no-sync python benchmarks/deadline_experiment.py ...``).
CUDA compilation and display initialization can stall independently of Python
sampling, so the harness intentionally does not install its own process alarm.

Modes:
  parity      fixed-action trace parity, including retained observation storage
  lean-parity compare standard and interactive step paths on an identical trace
  observation isolated repeated observation construction
  rendered    full 50v50 random-policy wrapper step plus 900x900 offscreen draw
  lean-rendered paired standard/interactive wrapper steps plus offscreen draw
  compiled-perception compare eager and compiled pure perception in lean frames
  gpu-compiled-perception paired eager/compiled perception with the GPU renderer
  compiled-parity parity-check eager-unbuffered vs compiled-unbuffered perception

The reference reconstructs the prior implementation: compute visibility once,
then call ``observation_from_state`` independently for teams 0 and 1.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import statistics
import subprocess
import time
from dataclasses import fields
from pathlib import Path

import torch
from realtime_latency import LARGE_SCALE, PROFILE, scenario_config
from realtime_latency import measure as realtime_measure

from boost_and_broadside.config.defaults import REWARDS
from boost_and_broadside.constants import NUM_POWER_ACTIONS, NUM_SHOOT_ACTIONS, NUM_TURN_ACTIONS
from boost_and_broadside.env import wrapper as wrapper_module
from boost_and_broadside.env.observation import (
    BulletObsKey,
    ObsKey,
    YemongObservation,
    _mask_hidden_ships,
    bullet_observation_from_state,
    compile_perception,
    observation_from_state,
)
from boost_and_broadside.env.perception import team_visibility_from_state
from boost_and_broadside.env.state import TensorState
from boost_and_broadside.env.wrapper import YemongEnvWrapper

DEADLINE_MS = 1000.0 / round(1.0 / (PROFILE.ship_config.dt * PROFILE.action_repeat))
FLOAT_ATOL = 2e-6
FLOAT_RTOL = 2e-6


def reference_builder(
    state, ship_config, env_config, buffers=None, include_bullets=False, perceive_bullets=None
):
    """Former independent team-view construction, using one visibility pass."""
    if perceive_bullets is None:
        perceive_bullets = include_bullets
    if include_bullets and not perceive_bullets:
        raise ValueError("bullet observations cannot be built without bullet perception")
    visibility = team_visibility_from_state(state, ship_config, env_config, perceive_bullets)
    views = [
        observation_from_state(
            state,
            ship_config,
            buffers,
            include_bullets=include_bullets,
            ship_visibility=visibility.ship[:, team],
            bullet_visibility=(None if visibility.bullet is None else visibility.bullet[:, team]),
            perspective_team=team,
        )
        for team in (0, 1)
    ]
    return YemongObservation(
        data=views[0].data,
        bullets=views[0].bullets,
        team1_data=views[1].data,
        team1_bullets=views[1].bullets,
    ), visibility


def _builder(arm):
    return reference_builder if arm == "reference" else candidate_observation_from_state


def _compiled_perception_adapter(compiled_builder, first_call_ms=None, device=None):
    """Adapt the pure compiled builder to the wrapper's buffered-call signature."""

    def build(
        state,
        ship_config,
        env_config,
        _buffers=None,
        include_bullets=False,
        perceive_bullets=None,
    ):
        if first_call_ms is None or first_call_ms[0] is not None:
            return compiled_builder(
                state, ship_config, env_config, None, include_bullets, perceive_bullets
            )
        return _timed_call(
            compiled_builder,
            state,
            ship_config,
            env_config,
            include_bullets,
            perceive_bullets,
            first_call_ms,
            time.perf_counter(),
            device,
        )

    return build


def _timed_call(
    builder,
    state,
    ship_config,
    env_config,
    include_bullets,
    perceive_bullets,
    timing,
    begin,
    device,
):
    result = builder(state, ship_config, env_config, None, include_bullets, perceive_bullets)
    if device is not None:
        _sync(device)
    if timing[0] is None:
        timing[0] = 1000.0 * (time.perf_counter() - begin)
    return result


def candidate_observation_from_state(
    state, ship_config, env_config, buffers=None, include_bullets=False, perceive_bullets=None
):
    """Isolated snapshot of the shared-observation candidate for reproducibility."""
    if perceive_bullets is None:
        perceive_bullets = include_bullets
    if include_bullets and not perceive_bullets:
        raise ValueError("bullet observations cannot be built without bullet perception")
    visibility = team_visibility_from_state(state, ship_config, env_config, perceive_bullets)
    common = observation_from_state(state, ship_config, buffers, include_bullets=False)
    common_bullets = (
        bullet_observation_from_state(state, ship_config)
        if include_bullets and state.max_bullets > 0
        else None
    )
    views = [
        _candidate_view_from_common(
            common,
            common_bullets,
            state,
            visibility.ship[:, team],
            None if visibility.bullet is None else visibility.bullet[:, team],
            team,
        )
        for team in (0, 1)
    ]
    return YemongObservation(
        data=views[0].data,
        bullets=views[0].bullets,
        team1_data=views[1].data,
        team1_bullets=views[1].bullets,
    ), visibility


def _candidate_view_from_common(
    common, common_bullets, state, ship_visibility, bullet_visibility, team
):
    data = dict(common.data)
    num_ships = state.max_ships
    object_alive = data[ObsKey.ALIVE][:, num_ships:]
    data[ObsKey.VISIBLE] = torch.cat([ship_visibility, object_alive], dim=1)
    own_ship = (state.ship_team_id == team).unsqueeze(-1)
    previous_action = data[ObsKey.PREVIOUS_ACTION]
    data[ObsKey.PREVIOUS_ACTION] = torch.cat(
        [
            torch.where(
                own_ship,
                previous_action[:, :num_ships],
                torch.zeros_like(previous_action[:, :num_ships]),
            ),
            previous_action[:, num_ships:],
        ],
        dim=1,
    )
    bullets = None
    if common_bullets is not None:
        assert bullet_visibility is not None
        active = common_bullets[BulletObsKey.ACTIVE]
        visible = active & bullet_visibility.reshape(active.shape)
        bullets = {}
        for key, value in common_bullets.items():
            if key in (BulletObsKey.ACTIVE, BulletObsKey.VISIBLE):
                bullets[key] = visible
                continue
            mask = visible.unsqueeze(-1) if value.dim() > visible.dim() else visible
            bullets[key] = torch.where(mask, value, torch.zeros_like(value))
    return _mask_hidden_ships(
        YemongObservation(data=data, bullets=bullets), ship_visibility, num_ships
    )


def _sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _tensor_equal(a, b, path, errors):
    if a is None or b is None:
        if a is not b:
            errors.append(f"{path}: None mismatch")
        return
    if isinstance(a, torch.Tensor):
        if a.shape != b.shape or a.dtype != b.dtype:
            errors.append(f"{path}: metadata mismatch {a.shape}/{a.dtype} vs {b.shape}/{b.dtype}")
        elif a.is_floating_point() or a.is_complex():
            if not torch.allclose(a, b, atol=FLOAT_ATOL, rtol=FLOAT_RTOL, equal_nan=True):
                errors.append(f"{path}: float mismatch; max_abs={float((a - b).abs().max())}")
        elif not torch.equal(a, b):
            errors.append(f"{path}: value mismatch")
        return
    if isinstance(a, dict):
        if a.keys() != b.keys():
            errors.append(f"{path}: key mismatch")
            return
        for key in a:
            _tensor_equal(a[key], b[key], f"{path}.{key}", errors)
        return
    if isinstance(a, (tuple, list)):
        if type(a) is not type(b) or len(a) != len(b):
            errors.append(f"{path}: sequence mismatch")
            return
        for index, (left, right) in enumerate(zip(a, b, strict=True)):
            _tensor_equal(left, right, f"{path}[{index}]", errors)
        return
    if a != b:
        errors.append(f"{path}: value mismatch {a!r} vs {b!r}")


def _compare_obs(a, b, errors, prefix):
    _tensor_equal(a.data, b.data, prefix + ".team0", errors)
    _tensor_equal(a.bullets, b.bullets, prefix + ".team0_bullets", errors)
    _tensor_equal(a.team1_data, b.team1_data, prefix + ".team1", errors)
    _tensor_equal(a.team1_bullets, b.team1_bullets, prefix + ".team1_bullets", errors)


def _wrapper(device, seed):
    config = scenario_config(100, 72, LARGE_SCALE)
    wrapper = YemongEnvWrapper(
        1,
        PROFILE.ship_config,
        config,
        REWARDS,
        device,
        include_bullets=False,
        perceive_bullets=True,
    )
    wrapper.reset(seed=seed)
    return wrapper


def _action_trace(seed, steps, device):
    gen = torch.Generator(device="cpu").manual_seed(seed + 7001)
    raw = torch.stack(
        (
            torch.randint(NUM_POWER_ACTIONS, (steps, 100), generator=gen),
            torch.randint(NUM_TURN_ACTIONS, (steps, 100), generator=gen),
            torch.randint(NUM_SHOOT_ACTIONS, (steps, 100), generator=gen),
        ),
        dim=-1,
    )
    return raw.to(device=device, dtype=torch.int32)


def run_parity(device, seed, steps):
    left, right = _wrapper(device, seed), _wrapper(device, seed)
    # Establish independently cloned but identical complete state snapshots.
    right.env.state = left.env.state.clone()
    actions = _action_trace(seed, steps, device)
    trace = actions.cpu().tolist()
    errors, checked = [], 0
    retained = None
    for index in range(steps):
        outputs = []
        old_builder = wrapper_module.perceived_observation_from_state
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        post_rng_states = None
        try:
            for arm, wrapper in (("reference", left), ("candidate", right)):
                torch.set_rng_state(cpu_rng_state)
                if cuda_rng_state is not None:
                    torch.cuda.set_rng_state(cuda_rng_state, device)
                wrapper_module.perceived_observation_from_state = _builder(arm)
                result = wrapper.step(actions[index : index + 1], auto_reset=False)
                outputs.append((result, wrapper.last_visibility))
                if arm == "reference":
                    post_rng_states = (
                        torch.get_rng_state(),
                        torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
                    )
        finally:
            wrapper_module.perceived_observation_from_state = old_builder
            if post_rng_states is not None:
                torch.set_rng_state(post_rng_states[0])
                if post_rng_states[1] is not None:
                    torch.cuda.set_rng_state(post_rng_states[1], device)
        _compare_obs(outputs[0][0][0], outputs[1][0][0], errors, f"obs[{index}]")
        _tensor_equal(
            outputs[0][1].__dict__, outputs[1][1].__dict__, f"visibility[{index}]", errors
        )
        for result_key, result_index in (("reward", 1), ("done", 2), ("truncated", 3)):
            _tensor_equal(
                outputs[0][0][result_index],
                outputs[1][0][result_index],
                f"{result_key}[{index}]",
                errors,
            )
        _tensor_equal(outputs[0][0][4], outputs[1][0][4], f"events[{index}]", errors)
        if retained is not None:
            old_index, old_obs, old_data, old_team1 = retained
            for key, snapshot in old_data.items():
                _tensor_equal(snapshot, old_obs.data[key], f"retained[{old_index}].{key}", errors)
            for key, snapshot in old_team1.items():
                _tensor_equal(
                    snapshot, old_obs.team1_data[key], f"retained_team1[{old_index}].{key}", errors
                )
        current_obs = outputs[1][0][0]
        retained = (
            index,
            current_obs,
            {k: v.clone() for k, v in current_obs.data.items()},
            {k: v.clone() for k, v in current_obs.team1_data.items()},
        )
        for f in fields(TensorState):
            _tensor_equal(
                getattr(left.env.state, f.name),
                getattr(right.env.state, f.name),
                f"state[{index}].{f.name}",
                errors,
            )
        checked += 1
        if errors:
            break
    if retained is not None and not errors:
        old_index, old_obs, old_data, old_team1 = retained
        for key, snapshot in old_data.items():
            _tensor_equal(snapshot, old_obs.data[key], f"retained[{old_index}].{key}", errors)
        for key, snapshot in old_team1.items():
            _tensor_equal(
                snapshot, old_obs.team1_data[key], f"retained_team1[{old_index}].{key}", errors
            )
    return {
        "steps_checked": checked,
        "parity": not errors,
        "errors": errors[:30],
        "float_tolerance": {"atol": FLOAT_ATOL, "rtol": FLOAT_RTOL},
        "action_trace": trace,
    }


def run_lean_parity(device, seed, steps):
    standard, interactive = _wrapper(device, seed), _wrapper(device, seed)
    interactive.env.state = standard.env.state.clone()
    actions = _action_trace(seed, steps, device)
    trace = actions.cpu().tolist()
    errors, checked = [], 0
    retained = {"standard": None, "interactive": None}
    rewards = {"shape": None, "lean_omitted": False, "compared": False}

    def check_retained(arm):
        value = retained[arm]
        if value is None:
            return
        old_index, old_obs, snapshot = value
        for channel, tensors in snapshot.items():
            current = getattr(old_obs, channel)
            if current is None:
                errors.append(f"retained_{arm}[{old_index}].{channel}: unexpectedly None")
                continue
            for key, saved in tensors.items():
                _tensor_equal(
                    saved,
                    current[key],
                    f"retained_{arm}[{old_index}].{channel}.{key}",
                    errors,
                )

    for index in range(steps):
        cpu_rng_state = torch.get_rng_state()
        cuda_rng_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, device)
        standard_result = standard.step(actions[index : index + 1], auto_reset=False)
        standard_post_rng = (
            torch.get_rng_state(),
            torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        )
        torch.set_rng_state(cpu_rng_state)
        if cuda_rng_state is not None:
            torch.cuda.set_rng_state(cuda_rng_state, device)
        interactive_result = interactive.step_interactive(
            actions[index : index + 1], auto_reset=False
        )
        torch.set_rng_state(standard_post_rng[0])
        if standard_post_rng[1] is not None:
            torch.cuda.set_rng_state(standard_post_rng[1], device)

        if len(standard_result) != 5:
            errors.append(
                f"standard[{index}]: expected 5 return values, got {len(standard_result)}"
            )
        if len(interactive_result) != 4:
            errors.append(
                f"interactive[{index}]: expected 4 return values, got {len(interactive_result)}"
            )
        reward = standard_result[1]
        expected_reward_shape = (1, 100, len(standard._active_names))
        rewards["shape"] = list(reward.shape) if isinstance(reward, torch.Tensor) else None
        if not isinstance(reward, torch.Tensor) or tuple(reward.shape) != expected_reward_shape:
            errors.append(
                f"standard[{index}].reward: expected shape {expected_reward_shape}, "
                f"got {getattr(reward, 'shape', None)}"
            )
        rewards["lean_omitted"] = len(interactive_result) == 4
        rewards["compared"] = False

        _compare_obs(standard_result[0], interactive_result[0], errors, f"obs[{index}]")
        _tensor_equal(standard_result[2], interactive_result[1], f"dones[{index}]", errors)
        _tensor_equal(standard_result[3], interactive_result[2], f"truncated[{index}]", errors)
        _tensor_equal(standard_result[4], interactive_result[3], f"info[{index}]", errors)
        _tensor_equal(
            standard.last_visibility.__dict__,
            interactive.last_visibility.__dict__,
            f"visibility[{index}]",
            errors,
        )
        for f in fields(TensorState):
            _tensor_equal(
                getattr(standard.env.state, f.name),
                getattr(interactive.env.state, f.name),
                f"state[{index}].{f.name}",
                errors,
            )
        for arm, obs in (
            ("standard", standard_result[0]),
            ("interactive", interactive_result[0]),
        ):
            check_retained(arm)
            retained[arm] = (
                index,
                obs,
                {
                    channel: (
                        {key: tensor.clone() for key, tensor in value.items()} if value else {}
                    )
                    for channel in ("data", "bullets", "team1_data", "team1_bullets")
                    if (value := getattr(obs, channel)) is not None
                },
            )
        checked += 1
        if errors:
            break

    if not errors:
        check_retained("standard")
        check_retained("interactive")
    return {
        "steps_checked": checked,
        "parity": not errors,
        "errors": errors[:30],
        "float_tolerance": {"atol": FLOAT_ATOL, "rtol": FLOAT_RTOL},
        "action_trace": trace,
        "rewards": rewards,
        "auto_reset": False,
        "terminal_reset_coverage": "covered by CPU tests",
    }


def run_compiled_perception_parity(device, seed, steps):
    wrapper = _wrapper(device, seed)
    actions = _action_trace(seed, steps, device)
    trace = actions.cpu().tolist()
    compiled_builder = compile_perception("default")
    errors, checked = [], 0
    retained = None
    eager_args = (
        wrapper.ship_config,
        wrapper.env_config,
        None,
        wrapper.include_bullets,
        wrapper.perceive_bullets,
    )
    for index in range(steps):
        state = wrapper.env.state
        eager_obs, eager_visibility = wrapper_module.perceived_observation_from_state(
            state, *eager_args
        )
        compiled_obs, compiled_visibility = compiled_builder(state, *eager_args)
        _compare_obs(eager_obs, compiled_obs, errors, f"obs[{index}]")
        _tensor_equal(
            eager_visibility.__dict__,
            compiled_visibility.__dict__,
            f"visibility[{index}]",
            errors,
        )
        if retained is not None:
            old_index, old_eager, old_compiled, eager_snapshot, compiled_snapshot = retained
            for channel, snapshots, obs in (
                ("eager", eager_snapshot, old_eager),
                ("compiled", compiled_snapshot, old_compiled),
            ):
                for field, values in snapshots.items():
                    current = getattr(obs, field)
                    for key, value in values.items():
                        _tensor_equal(
                            value,
                            current[key],
                            f"retained[{old_index}].{channel}.{field}.{key}",
                            errors,
                        )

        def snapshot(obs):
            return {
                field: {key: tensor.clone() for key, tensor in value.items()}
                for field in ("data", "bullets", "team1_data", "team1_bullets")
                if (value := getattr(obs, field)) is not None
            }

        retained = (
            index,
            eager_obs,
            compiled_obs,
            snapshot(eager_obs),
            snapshot(compiled_obs),
        )
        checked += 1
        if errors:
            break
        wrapper.env.tick(actions[index : index + 1])
    if retained is not None and not errors:
        old_index, old_eager, old_compiled, eager_snapshot, compiled_snapshot = retained
        for channel, snapshots, obs in (
            ("eager", eager_snapshot, old_eager),
            ("compiled", compiled_snapshot, old_compiled),
        ):
            for field, values in snapshots.items():
                current = getattr(obs, field)
                for key, value in values.items():
                    _tensor_equal(
                        value,
                        current[key],
                        f"retained[{old_index}].{channel}.{field}.{key}",
                        errors,
                    )
    return {
        "steps_checked": checked,
        "parity": not errors,
        "errors": errors[:30],
        "float_tolerance": {"atol": FLOAT_ATOL, "rtol": FLOAT_RTOL},
        "action_trace": trace,
        "builders": ["eager-unbuffered", "compiled-unbuffered(default)"],
    }


def _percentile(samples, p):
    ordered = sorted(samples)
    return ordered[min(len(ordered) - 1, int(p * (len(ordered) - 1)))]


def _stats(samples):
    return {
        "n": len(samples),
        "p50_ms": statistics.median(samples),
        "p95_ms": _percentile(samples, 0.95),
        "p99_ms": _percentile(samples, 0.99),
        "max_ms": max(samples),
        "deadline_ms": DEADLINE_MS,
        "deadline_misses": sum(x > DEADLINE_MS for x in samples),
    }


def _memory(device):
    result = {"process_max_rss_kib": _rss_kib()}
    if device.type == "cuda":
        result["cuda_peak_allocated_mib"] = torch.cuda.max_memory_allocated(device) / 2**20
        result["cuda_peak_reserved_mib"] = torch.cuda.max_memory_reserved(device) / 2**20
    return result


def _rss_kib():
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        return None


def measure_observation(arm, device, seed, warmup, samples):
    startup_begin = time.perf_counter()
    wrapper = _wrapper(device, seed)
    startup_ms = (time.perf_counter() - startup_begin) * 1000
    fn = _builder(arm)
    state = wrapper.env.state

    def call():
        return fn(
            state,
            wrapper.ship_config,
            wrapper.env_config,
            wrapper._obs_buffers,
            wrapper.include_bullets,
            wrapper.perceive_bullets,
        )

    begin = time.perf_counter()
    call()
    _sync(device)
    first_ms = (time.perf_counter() - begin) * 1000
    for _ in range(warmup):
        call()
    _sync(device)
    timings = []
    for _ in range(samples):
        begin = time.perf_counter()
        call()
        _sync(device)
        timings.append((time.perf_counter() - begin) * 1000)
    return {
        "startup_ms": startup_ms,
        "first_call_ms": first_ms,
        **_stats(timings),
        "memory": _memory(device),
        "phases_ms": {"perceived_observation": statistics.median(timings)},
    }


def measure_rendered(
    arm,
    device,
    seed,
    warmup,
    samples,
    policy_compile="none",
    observation_comparison=True,
    gpu_rendered=False,
    moderngl_path=None,
):
    original = wrapper_module.perceived_observation_from_state
    lean = arm == "lean" or gpu_rendered
    if observation_comparison:
        wrapper_module.perceived_observation_from_state = _builder(arm)
    startup_begin = time.perf_counter()
    try:
        row = realtime_measure(
            "50v50",
            device,
            steps=samples,
            warmup=warmup,
            compile_mode=None if policy_compile == "none" else policy_compile,
            seed=seed,
            stage="baseline",
            perceive_bullets=True,
            policy_sides=2,
            window_size=900,
            execution="sequential",
            enqueue_order="env-first",
            environment_step="step_interactive" if lean else "step",
            renderer_backend=("legacy" if arm == "legacy" else "gpu") if gpu_rendered else None,
            moderngl_path=moderngl_path,
        )
    finally:
        wrapper_module.perceived_observation_from_state = original
    row["arm_total_wall_ms"] = (time.perf_counter() - startup_begin) * 1000
    row["compile_mode"] = None if policy_compile == "none" else policy_compile
    row["environment_step"] = "step_interactive" if lean else "step"
    row["policy_label"] = "two distinct randomly initialized policies"
    row["render"] = {
        "size": [900, 900],
        "display_flip": gpu_rendered,
        "fps_sleep": False,
        "fog": True,
        "projectile_perception": True,
        "policy_bullet_tokens": False,
    }
    row["deadline_ms"] = row["frame_budget_ms"]
    return row


def measure_compiled_perception(
    arm,
    device,
    seed,
    warmup,
    samples,
    policy_compile="none",
    *,
    gpu_rendered=False,
    moderngl_path=None,
):
    original = wrapper_module.perceived_observation_from_state
    first_builder_call_ms = [None]
    setup_start = time.perf_counter()
    if arm == "candidate":
        wrapper_module.perceived_observation_from_state = _compiled_perception_adapter(
            compile_perception("default"), first_builder_call_ms, device
        )
    perception_setup_ms = 1000.0 * (time.perf_counter() - setup_start)
    wall_start = time.perf_counter()
    try:
        row = realtime_measure(
            "50v50",
            device,
            steps=samples,
            warmup=warmup,
            compile_mode=None if policy_compile == "none" else policy_compile,
            seed=seed,
            stage="baseline",
            perceive_bullets=True,
            policy_sides=2,
            window_size=900,
            execution="sequential",
            enqueue_order="env-first",
            environment_step="step_interactive",
            renderer_backend="gpu" if gpu_rendered else None,
            moderngl_path=moderngl_path,
        )
    finally:
        wrapper_module.perceived_observation_from_state = original
    row["arm_total_wall_ms"] = 1000.0 * (time.perf_counter() - wall_start)
    row["perception_compile_mode"] = "default" if arm == "candidate" else None
    row["perception_adapter"] = (
        "pure compiled adapter (buffers ignored)"
        if arm == "candidate"
        else "production buffered eager"
    )
    row["perception_callable_setup_ms"] = perception_setup_ms
    row["perception_first_builder_call_ms"] = first_builder_call_ms[0]
    row["compile_mode"] = None if policy_compile == "none" else policy_compile
    row["environment_step"] = "step_interactive"
    row["policy_label"] = "two distinct randomly initialized policies"
    row["render"] = {
        "backend": "gpu" if gpu_rendered else "legacy",
        "size": [900, 900],
        "display_flip": gpu_rendered,
        "fps_sleep": False,
        "fog": True,
        "perspective": "team0" if gpu_rendered else None,
        "projectile_perception": True,
        "policy_bullet_tokens": False,
    }
    row["deadline_ms"] = row["frame_budget_ms"]
    return row


def revision_state():
    def git(*args):
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()

    return {
        "revision": git("rev-parse", "HEAD"),
        "branch": git("branch", "--show-current"),
        "dirty": bool(git("status", "--porcelain")),
    }


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--mode",
        choices=(
            "parity",
            "lean-parity",
            "compiled-parity",
            "observation",
            "rendered",
            "lean-rendered",
            "compiled-perception",
            "gpu-rendered",
            "gpu-compiled-perception",
        ),
        required=True,
    )
    p.add_argument(
        "--device", choices=("cpu", "cuda"), default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument("--seed", type=int, default=271828)
    p.add_argument("--steps", type=int, default=40, help="parity fixed-action trace length")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--samples", type=int, default=200)
    p.add_argument("--pairs", type=int, default=3, help="alternating-order A/B pair count")
    p.add_argument(
        "--policy-compile",
        choices=("none", "default", "reduce-overhead"),
        default="none",
        help="policy torch.compile mode for rendered modes; none is the eager baseline",
    )
    p.add_argument("--out", type=Path, required=True, help="incrementally updated JSON result file")
    p.add_argument(
        "--moderngl-path",
        help="isolated ModernGL directory for gpu-rendered and gpu-compiled-perception modes",
    )
    args = p.parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        p.error("CUDA requested but unavailable")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    torch.set_num_threads(1)
    metadata = {
        "mode": args.mode,
        "revision": revision_state(),
        "args": vars(args) | {"out": str(args.out)},
        "software": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "platform": platform.platform(),
        },
        "device": {
            "requested": str(device),
            "name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else platform.processor(),
        },
        "seed": args.seed,
        "compile": "eager builder; policy compile selected by --policy-compile",
        "cache": "process-local empty at startup",
        "map": {"ships": 100, "teams": "50v50", "fields": 72, "render_px": 900},
        "scope": {
            "policy_bullet_tokens": False,
            "projectile_perception": True,
            "fog": True,
            "rendered": args.mode
            in {
                "rendered",
                "lean-rendered",
                "compiled-perception",
                "gpu-rendered",
                "gpu-compiled-perception",
            },
            "display_flip": args.mode in {"gpu-rendered", "gpu-compiled-perception"},
            "policy_label": "two distinct randomly initialized policies"
            if args.mode
            in {
                "rendered",
                "lean-rendered",
                "compiled-perception",
                "gpu-rendered",
                "gpu-compiled-perception",
            }
            else None,
        },
        "warmup": args.warmup,
        "samples": args.samples,
        "arms": {},
        "pairs": args.pairs,
        "policy_compile_mode": (
            args.policy_compile
            if args.mode
            in {
                "rendered",
                "lean-rendered",
                "compiled-perception",
                "gpu-rendered",
                "gpu-compiled-perception",
            }
            else None
        ),
        "perception_compile_mode": (
            "default" if args.mode in {"compiled-perception", "gpu-compiled-perception"} else None
        ),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        args.out.write_text(json.dumps(metadata, indent=2, default=str))

    persist()
    if args.mode in {"parity", "lean-parity", "compiled-parity"}:
        parity_runner = {
            "parity": run_parity,
            "lean-parity": run_lean_parity,
            "compiled-parity": run_compiled_perception_parity,
        }[args.mode]
        metadata["result"] = parity_runner(device, args.seed, args.steps)
        persist()
        return
    # Paired alternating order reduces clock/order bias. Save after every arm.
    run_arm = measure_observation if args.mode == "observation" else measure_rendered
    arms = (
        ("legacy", "gpu")
        if args.mode == "gpu-rendered"
        else (("reference", "lean") if args.mode == "lean-rendered" else ("reference", "candidate"))
    )
    for pair in range(args.pairs):
        order = arms if pair % 2 == 0 else tuple(reversed(arms))
        for arm in order:
            if arm not in metadata["arms"]:
                metadata["arms"][arm] = []
            if args.mode == "observation":
                row = run_arm(arm, device, args.seed + pair, args.warmup, args.samples)
            elif args.mode in {"compiled-perception", "gpu-compiled-perception"}:
                row = measure_compiled_perception(
                    arm,
                    device,
                    args.seed + pair,
                    args.warmup,
                    args.samples,
                    args.policy_compile,
                    gpu_rendered=args.mode == "gpu-compiled-perception",
                    moderngl_path=args.moderngl_path,
                )
            else:
                row = run_arm(
                    arm,
                    device,
                    args.seed + pair,
                    args.warmup,
                    args.samples,
                    args.policy_compile,
                    args.mode == "rendered",
                    args.mode == "gpu-rendered",
                    args.moderngl_path,
                )
            row["pair"] = pair
            row["seed"] = args.seed + pair
            metadata["arms"][arm].append(row)
            persist()
    for arm, rows in metadata["arms"].items():
        if args.mode in {
            "rendered",
            "lean-rendered",
            "compiled-perception",
            "gpu-rendered",
            "gpu-compiled-perception",
        }:
            samples_by_pair = [row["raw_samples_ms"] for row in rows]
            all_samples = [value for pair_samples in samples_by_pair for value in pair_samples]
            p50 = statistics.median(all_samples)
            p95 = _percentile(all_samples, 0.95)
            p99 = _percentile(all_samples, 0.99)
            max_ms = max(all_samples)
            misses = sum(row["deadline_misses"] for row in rows)
        else:
            p50 = statistics.median(r["p50_ms"] for r in rows)
            p95 = _percentile([r["p95_ms"] for r in rows], 0.95)
            p99 = max(r["p99_ms"] for r in rows)
            max_ms = max(r["max_ms"] for r in rows)
            misses = sum(r["deadline_misses"] for r in rows)
        metadata.setdefault("summary", {})[arm] = {
            "p50_ms": p50,
            "p95_ms": p95,
            "p99_ms": p99,
            "max_ms": max_ms,
            "deadline_misses": misses,
        }
    persist()


if __name__ == "__main__":
    main()
