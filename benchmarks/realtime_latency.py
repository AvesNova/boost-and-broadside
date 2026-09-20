"""Can `bnb play` / `bnb watch` hold real time? Single-environment decision latency.

Interactive modes run one environment with a policy on each side, and Frontline
decides at 30 Hz (`dt = 1/30`, `action_repeat = 1`), so a decision has **33.3 ms**
to finish. Everything else this repository measures is throughput at a batch
width of hundreds or thousands, which says nothing about latency at a batch of
one -- at `B=1` the pipeline is dispatch-bound, and a device that wins on
throughput can lose on latency.

Four configurations, which is the whole question:

    5v5   on GPU        5v5   on CPU
    50v50 on GPU        50v50 on CPU

The loop replicates the buffered NN timing used by PPO and interactive watch:

1. perception and observation assembly for both team views;
2. each side's ego view, belief composition, policy forward, belief advance;
3. physics consumes the preceding buffered NN action while the policy produces
   the next one. ``--execution streams`` puts those two branches on separate
   CUDA streams from one host thread; ``threads`` also dispatches them from two
   persistent host threads. ``sequential`` preserves the same one-tick delay.

Two distinct policies, one per side, because that is the general case -- watch
mode short-circuits the second forward only when both sides are literally the
same agent object. The trajectory-imagination overlay is not included: it is
currently disabled (`N_IMAGINE_STEPS = 0`), and each step it is given would add
two more forward passes per frame.

Policies are eager, not compiled. Interactive sessions pay compilation on the
first frames and a batch of one is where `torch.compile` has least to win; the
`--compile` flag measures the alternative.

Usage:
    uv run --no-sync python benchmarks/realtime_latency.py --out realtime.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import torch

from boost_and_broadside.config.core import EnvConfig, entity_token_count
from boost_and_broadside.config.defaults import REWARDS
from boost_and_broadside.env.observation import ObsKey
from boost_and_broadside.env.wrapper import YemongEnvWrapper
from boost_and_broadside.evaluation.match import merge_team_actions
from boost_and_broadside.profiles import PROFILES
from boost_and_broadside.train.rl.belief import BeliefTracker
from boost_and_broadside.train.rl.policy_io import build_policy, compile_policy


def _headless_renderer(ship_config, window_size: int, *, display_flip: bool, zone_occlusion: bool):
    """Build an offscreen ``GameRenderer``.

    Imported here rather than at module scope because ``ui.renderer`` selects
    its video driver when pygame is imported, so ``HEADLESS`` has to be set
    first -- and a module-level assignment between two import blocks is the one
    thing the import linter will not allow. ``play_throughput.py`` sets the same
    variable for the same reason.
    """

    if not display_flip:
        os.environ.setdefault("HEADLESS", "1")
    else:
        # GPU/legacy paired runs alternate renderers in one process.  Never
        # leave SDL's dummy driver set before opening a real OpenGL window.
        os.environ.pop("HEADLESS", None)
        os.environ.pop("SDL_VIDEODRIVER", None)
    from boost_and_broadside.ui.renderer import GameRenderer, RenderConfig, VisionMode

    return GameRenderer(
        ship_config,
        RenderConfig(
            window_size=window_size,
            fps=DECISION_HZ,
            show_ui=True,
            vision_mode=VisionMode.TEAM_0,
            zone_occlusion=zone_occlusion,
        ),
    )


PROFILE = PROFILES["rl"]

#: Linear scale for the 50v50 map, matching ``frontline_inference_scaling``.
LARGE_SCALE = 7000.0 / 2600.0

#: One decision per physics tick at the Frontline timestep.
DECISION_HZ = round(1.0 / (PROFILE.ship_config.dt * PROFILE.action_repeat))
FRAME_BUDGET_MS = 1000.0 / DECISION_HZ

SCENARIOS = {
    # name: (ships, fields, map scale)
    "5v5": (10, 10, 1.0),
    "50v50": (100, 72, LARGE_SCALE),
}


def scenario_config(ships: int, fields: int, scale: float) -> EnvConfig:
    frontline = replace(
        PROFILE.frontline,
        zone_radius=PROFILE.frontline.zone_radius * scale,
        zone_ring_radius=PROFILE.frontline.zone_ring_radius * scale,
        playable_radius=PROFILE.frontline.playable_radius * scale,
    )
    return EnvConfig(
        num_ships=ships,
        num_fields=fields,
        max_bullets=PROFILE.max_bullets,
        max_episode_steps=PROFILE.max_episode_steps,
        action_repeat=PROFILE.action_repeat,
        frontline=frontline,
        vision_range=PROFILE.vision_range * scale,
        zones_occlude=PROFILE.zones_occlude,
    )


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def warm_device(device: torch.device, seconds: float = 4.0) -> None:
    """Drive the GPU to a steady clock before timing anything.

    At a batch of one this loop is dispatch-bound, and a dispatch-bound
    measurement is a measurement of clock state. An idle laptop GPU sits near
    855 MHz of a 3105 MHz maximum and only boosts under sustained load, so a
    scenario measured second in a process inherits the clocks the first one
    earned: 50v50 measured 18.96 ms per frame directly after 5v5 and 58.51 ms
    in a process of its own, which is a 3x difference in nothing but warm-up.

    A fixed burn before timing puts every configuration in the same state. It
    does not make the numbers optimistic -- a real interactive session is
    bursty and may well sit at the lower clock -- it makes them comparable.
    """

    if device.type != "cuda":
        return
    burn = torch.randn(2048, 2048, device=device)
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        burn = torch.mm(burn, burn).clamp(-1.0, 1.0)
    torch.cuda.synchronize()


@torch.inference_mode()
def measure(
    name: str,
    device: torch.device,
    *,
    steps: int,
    warmup: int,
    compile_mode: str | None,
    seed: int,
    stage: str,
    perceive_bullets: bool,
    policy_sides: int,
    window_size: int | None,
    execution: str,
    enqueue_order: str,
    environment_step: str = "step",
    renderer_backend: str | None = None,
    moderngl_path: str | None = None,
    belief_compile_mode: str | None = None,
    perception_compile_mode: str | None = None,
    collision_compile_mode: str | None = None,
    cuda_graph_tick: bool = False,
) -> dict:
    """Time one interactive frame's worth of work, repeatedly."""

    ships, fields, scale = SCENARIOS[name]
    if environment_step not in {"step", "step_interactive"}:
        raise ValueError("environment_step must be 'step' or 'step_interactive'")
    ship_config = PROFILE.ship_config
    env_config = scenario_config(ships, fields, scale)
    torch.manual_seed(seed)

    # The wrapper, not a bare TensorEnv: it owns the reusable ObservationBuffers
    # the interactive loop depends on, and rebuilding those per frame would be
    # measuring an allocation the real mode does not make. `perceive_bullets` is
    # on because the renderer draws projectiles from the perception masks
    # whether or not the policies read bullet tokens.
    wrapper = YemongEnvWrapper(
        num_envs=1,
        ship_config=ship_config,
        env_config=env_config,
        rewards=REWARDS,
        device=device,
        collision_compile_mode=collision_compile_mode,
        include_bullets=False,
        perceive_bullets=perceive_bullets,
        interactive_cuda_graph=cuda_graph_tick,
        interactive_perception_compile_mode=perception_compile_mode,
    )
    observation = wrapper.reset()
    action_buffer = torch.zeros((1, ships, 3), dtype=torch.int32, device=device)
    if cuda_graph_tick and environment_step != "step_interactive":
        raise ValueError("CUDA graph tick requires step_interactive")

    model_config = replace(PROFILE.model_config, **STAGES[stage])
    sides = []
    policy_start = time.perf_counter()
    for _ in range(policy_sides):
        policy = build_policy(
            model_config,
            ship_config,
            num_value_components=12,
            num_ships=ships,
            team_pma_k=(0, 1),
        ).to(device)
        policy.eval()
        policy.requires_grad_(False)
        belief = BeliefTracker(
            1, ships, ship_config.dt * env_config.action_repeat, policy.coordinator, device
        )
        sides.append(
            {
                "policy": compile_policy(policy, compile_mode),
                "belief": belief,
                "belief_compose": (
                    torch.compile(belief.compose, mode=belief_compile_mode, dynamic=False)
                    if belief_compile_mode is not None
                    else belief.compose
                ),
                "belief_advance": (
                    torch.compile(belief.advance, mode=belief_compile_mode, dynamic=False)
                    if belief_compile_mode is not None
                    else belief.advance
                ),
                "hidden": policy.initial_hidden(1, ships, device),
            }
        )
    policy_startup_compile_ms = 1000.0 * (time.perf_counter() - policy_start)

    # Offscreen, via the same ``draw_frame`` capture and the smoke tests use, so
    # the drawing work is real while the display flip and the frame-rate sleep --
    # which are the parts that would *hide* an overrun rather than cause one --
    # are not counted. ``play_throughput.py`` measures the same way.
    renderer = None
    gpu_renderer = None
    previous_snapshot = None
    gl_info: dict[str, str] = {}
    if renderer_backend is not None and name != "50v50":
        raise ValueError("GPU renderer benchmark mode is defined only for 50v50")
    if renderer_backend == "legacy":
        renderer = _headless_renderer(
            ship_config,
            window_size or 900,
            display_flip=True,
            zone_occlusion=env_config.zones_occlude,
        )
    elif renderer_backend in {"gpu", "gpu-packed"}:
        if moderngl_path:
            import sys

            sys.path.insert(0, moderngl_path)
        from boost_and_broadside.ui.gpu_renderer import FrontlineGPURenderer
        from boost_and_broadside.ui.gpu_snapshot import (
            make_packed_render_snapshot,
            make_render_snapshot,
        )

        snapshot_builder = (
            make_packed_render_snapshot
            if renderer_backend == "gpu-packed"
            else make_render_snapshot
        )
        previous_snapshot = snapshot_builder(
            wrapper.state,
            world_size=ship_config.world_size,
            visibility=wrapper.last_visibility,
            zones_occlude=env_config.zones_occlude,
        )
        gpu_renderer = FrontlineGPURenderer(
            previous_snapshot, (window_size or 900, window_size or 900)
        )
        gpu_renderer.perspective = "team0"
        info = gpu_renderer._ctx.info
        gl_info = {
            key: str(info.get(key, "unknown")) for key in ("GL_VENDOR", "GL_RENDERER", "GL_VERSION")
        }
    elif window_size:
        renderer = _headless_renderer(
            ship_config,
            window_size,
            display_flip=False,
            zone_occlusion=env_config.zones_occlude,
        )

    as_team1 = torch.ones(1, dtype=torch.bool, device=device)
    phases = {"policy": 0.0, "env_step": 0.0, "render": 0.0}
    concurrent = execution in {"streams", "threads"}
    threaded = execution == "threads"
    if concurrent and device.type != "cuda":
        raise ValueError("streams execution requires CUDA")
    env_stream = torch.cuda.Stream(device=device) if concurrent else None
    net_stream = torch.cuda.Stream(device=device) if concurrent else None
    executor = ThreadPoolExecutor(max_workers=2) if threaded else None
    # Episode resets are counted, not hidden: a reset rebuilds the field layout
    # and re-initialises the map, which costs far more than a step. A scenario
    # whose matches end constantly would otherwise look slow for a reason that
    # has nothing to do with the per-frame work being measured.
    resets = {"count": 0}

    def policy_actions(source_observation):
        actions = []
        for team, side in enumerate(sides):
            view = source_observation.for_team(team)
            if team == 1:
                # An ego_pass policy only ever learned to act as team 0, so
                # playing team 1 means seeing mirrored team IDs -- exactly what
                # `agent_view` does in the interactive loop.
                view = view.flip_team(ships, mask=as_team1)
            view = side["belief_compose"](view)
            action, _, _, prediction, side["hidden"] = side["policy"].get_action_and_value(
                view, side["hidden"]
            )
            side["belief_advance"](view, prediction)
            actions.append(action)
        # One policy side means the other team is a keyboard or scripted
        # controller, whose action costs nothing measurable next to a forward
        # pass. `bnb watch --team0 null --team1 <ckpt>` is exactly this case.
        opponent = actions[1] if len(actions) > 1 else torch.zeros_like(actions[0])
        return merge_team_actions(actions[0], opponent, wrapper.state.ship_team_id).int()

    def expose_pending_action() -> None:
        observation.data[ObsKey.PREVIOUS_ACTION][:, :ships].copy_(action_buffer)
        if observation.team1_data is not None:
            observation.team1_data[ObsKey.PREVIOUS_ACTION][:, :ships].copy_(action_buffer)

    def frame(record: bool) -> None:
        nonlocal action_buffer, observation, previous_snapshot
        policy_start = policy_end = env_start = env_end = None
        cpu_start = time.perf_counter()

        if device.type == "cuda" and record:
            policy_start = torch.cuda.Event(enable_timing=True)
            policy_end = torch.cuda.Event(enable_timing=True)
            env_start = torch.cuda.Event(enable_timing=True)
            env_end = torch.cuda.Event(enable_timing=True)

        if concurrent:
            # Every returned observation channel owns its final cat/clone
            # storage. Reusable ObservationBuffers sit underneath that final
            # assembly, so building the next observation cannot overwrite this
            # one while the policy reads it.
            policy_observation = observation
            current = torch.cuda.current_stream(device)
            env_stream.wait_stream(current)
            net_stream.wait_stream(current)

            def launch_environment():
                with (
                    torch.inference_mode(),
                    torch.cuda.device(device),
                    torch.cuda.stream(env_stream),
                ):
                    if env_start is not None:
                        env_start.record(env_stream)
                    result = getattr(wrapper, environment_step)(action_buffer, auto_reset=False)
                    if env_end is not None:
                        env_end.record(env_stream)
                return result

            def launch_policy():
                with (
                    torch.inference_mode(),
                    torch.cuda.device(device),
                    torch.cuda.stream(net_stream),
                ):
                    if policy_start is not None:
                        policy_start.record(net_stream)
                    next_action_out = policy_actions(policy_observation)
                    if policy_end is not None:
                        policy_end.record(net_stream)
                return next_action_out

            launchers = (
                (launch_environment, launch_policy)
                if enqueue_order == "env-first"
                else (launch_policy, launch_environment)
            )
            if threaded:
                futures = {launch: executor.submit(launch) for launch in launchers}
                env_result = futures[launch_environment].result()
                next_action = futures[launch_policy].result()
            else:
                outputs = {launch: launch() for launch in launchers}
                env_result = outputs[launch_environment]
                next_action = outputs[launch_policy]
            current.wait_stream(env_stream)
            current.wait_stream(net_stream)
            if environment_step == "step_interactive":
                observation, dones, truncated, _info = env_result
            else:
                observation, _, dones, truncated, _info = env_result
        else:
            if policy_start is not None:
                policy_start.record()
            next_action = policy_actions(observation)
            if policy_end is not None:
                policy_end.record()
            policy_mark = time.perf_counter()
            if env_start is not None:
                env_start.record()
            env_result = getattr(wrapper, environment_step)(action_buffer, auto_reset=False)
            if environment_step == "step_interactive":
                observation, dones, truncated, _info = env_result
            else:
                observation, _, dones, truncated, _info = env_result
            if env_end is not None:
                env_end.record()
            if record and device.type == "cpu":
                phases["policy"] += policy_mark - cpu_start
                phases["env_step"] += time.perf_counter() - policy_mark

        action_buffer = next_action.detach()
        expose_pending_action()

        if record and device.type == "cuda":
            torch.cuda.synchronize(device)
            phases["policy"] += policy_start.elapsed_time(policy_end) / 1000.0
            phases["env_step"] += env_start.elapsed_time(env_end) / 1000.0

        if renderer is not None:
            render_start = time.perf_counter()
            if renderer_backend == "legacy":
                renderer.render(wrapper.state, visibility=wrapper.last_visibility)
            else:
                renderer.draw_frame(wrapper.state, visibility=wrapper.last_visibility)
            if record:
                phases["render"] += time.perf_counter() - render_start
        elif gpu_renderer is not None:
            from boost_and_broadside.ui.gpu_snapshot import (
                make_packed_render_snapshot,
                make_render_snapshot,
            )

            render_start = time.perf_counter()
            if renderer_backend == "gpu-packed":
                current_snapshot = make_packed_render_snapshot(
                    wrapper.state,
                    world_size=ship_config.world_size,
                    visibility=wrapper.last_visibility,
                    zones_occlude=env_config.zones_occlude,
                )
                gpu_renderer.render_packed(current_snapshot, previous_snapshot)
                previous_snapshot = current_snapshot
            else:
                previous_snapshot = make_render_snapshot(
                    wrapper.state,
                    world_size=ship_config.world_size,
                    previous=previous_snapshot,
                    visibility=wrapper.last_visibility,
                    zones_occlude=env_config.zones_occlude,
                )
                gpu_renderer.render(previous_snapshot)
            if record:
                phases["render"] += time.perf_counter() - render_start

        finished = dones | truncated
        if bool(finished.any()):
            resets["count"] += 1
            observation = wrapper.reset()
            action_buffer.zero_()
            for side in sides:
                side["belief"].reset(finished)
                side["hidden"] = side["policy"].reset_hidden_for_envs(
                    side["hidden"], finished, ships
                )
        return next_action

    warm_device(device)
    first_call_start = time.perf_counter()
    frame(False)
    _sync(device)
    first_call_frame_ms = 1000.0 * (time.perf_counter() - first_call_start)
    for _ in range(warmup):
        frame(False)
    _sync(device)

    samples: list[float] = []
    action_trace: list[list] = []
    for _ in range(steps):
        begin = time.perf_counter()
        sampled_action = frame(False)
        _sync(device)
        samples.append(1000.0 * (time.perf_counter() - begin))
        action_trace.append(sampled_action.cpu().tolist())

    for _ in range(max(steps // 4, 5)):
        frame(True)
    breakdown_frames = max(steps // 4, 5)

    if renderer is not None:
        renderer.close()
    if gpu_renderer is not None:
        gpu_renderer.close()
    if executor is not None:
        executor.shutdown()

    raw_samples = samples.copy()
    samples.sort()
    median = statistics.median(samples)
    measured_hz = 1000.0 * len(samples) / sum(samples)
    row = {
        "scenario": name,
        "stage": stage,
        "device": device.type,
        "num_ships": ships,
        "num_fields": fields,
        "entity_tokens": entity_token_count(ships, fields, env_config.frontline),
        "compile_mode": compile_mode,
        "execution": execution,
        "environment_step": environment_step,
        "enqueue_order": enqueue_order if concurrent else None,
        "decision_hz": DECISION_HZ,
        "frame_budget_ms": FRAME_BUDGET_MS,
        "median_ms": median,
        "p95_ms": samples[int(0.95 * (len(samples) - 1))],
        "p90_ms": samples[int(0.9 * (len(samples) - 1))],
        "p99_ms": samples[int(0.99 * (len(samples) - 1))],
        "max_ms": samples[-1],
        "raw_samples_ms": raw_samples,
        "action_trace": action_trace,
        "first_call_frame_ms": first_call_frame_ms,
        "policy_startup_compile_ms": policy_startup_compile_ms,
        "policy_startup_compile_total_ms": policy_startup_compile_ms + first_call_frame_ms,
        "over_budget_fraction": sum(sample > FRAME_BUDGET_MS for sample in samples) / len(samples),
        "deadline_misses": sum(sample > FRAME_BUDGET_MS for sample in samples),
        "realtime_headroom": FRAME_BUDGET_MS / median,
        "measured_hz": measured_hz,
        "perceive_bullets": perceive_bullets,
        "policy_sides": policy_sides,
        "belief_compile_mode": belief_compile_mode,
        "perception_compile_mode": perception_compile_mode,
        "collision_compile_mode": collision_compile_mode,
        "cuda_graph_tick": cuda_graph_tick,
        "window_size": window_size,
        "renderer_backend": renderer_backend or ("headless-legacy" if window_size else None),
        "render_scope": {
            "display_flip": renderer_backend in {"legacy", "gpu", "gpu-packed"},
            "gpu_snapshot_in_render_phase": renderer_backend in {"gpu", "gpu-packed"},
            "team_fog": "team0" if renderer_backend in {"gpu", "gpu-packed"} else None,
            "gl": gl_info or None,
        },
        "torch_threads": torch.get_num_threads(),
        "resets": resets["count"],
        "phase_ms": {key: 1000.0 * value / breakdown_frames for key, value in phases.items()},
    }
    if device.type == "cuda":
        row["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    return row


STAGES: dict[str, dict] = {
    "baseline": {},
    "BC-1": {"n_spatial_heads": 2},
    "BC-2": {"n_spatial_heads": 2, "spatial_rope": True},
    "BC-3": {"n_spatial_heads": 2, "spatial_rope": True, "local_presence": True},
    "BC-4": {
        "n_spatial_heads": 2,
        "spatial_rope": True,
        "local_presence": True,
        "relational_bias": True,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--seed", type=int, default=99)
    parser.add_argument("--stage", default="baseline", choices=sorted(STAGES))
    parser.add_argument("--compile", dest="compile_mode", default="none")
    parser.add_argument("--environment-step", choices=("step", "step_interactive"), default="step")
    parser.add_argument("--device", action="append", choices=("cuda", "cpu"))
    parser.add_argument("--scenario", action="append", choices=sorted(SCENARIOS))
    parser.add_argument(
        "--execution",
        action="append",
        choices=("sequential", "streams", "threads"),
        help="CUDA defaults to sequential, one-host-thread streams, and two host threads",
    )
    parser.add_argument(
        "--enqueue-order",
        choices=("env-first", "policy-first"),
        default="env-first",
        help="host dispatch order for the two CUDA streams",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=0,
        help="offscreen render at this window size each frame; 0 skips rendering",
    )
    parser.add_argument(
        "--sides",
        type=int,
        default=2,
        choices=(1, 2),
        help="teams driven by a policy; 1 is watch-against-keyboard, 2 is policy vs policy",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help=(
            "torch CPU thread count, default 1. Measured: one environment's "
            "tensors are a few hundred elements, where splitting across cores "
            "costs more in barriers than it saves -- 5v5 runs 55.9 ms at one "
            "thread, 62.9 at two, 86.6 at four and 115.9 at torch's default of "
            "ten. benchmarks/play_throughput.py pins 1 for the same reason."
        ),
    )
    parser.add_argument(
        "--no-bullet-perception",
        action="store_true",
        help="ablation: skip projectile line-of-sight, which only the renderer needs",
    )
    parser.add_argument("--out", type=Path)
    parser.add_argument("--belief-compile", choices=("default", "reduce-overhead"))
    parser.add_argument("--perception-compile", choices=("default", "reduce-overhead"))
    parser.add_argument("--collision-compile", choices=("default", "reduce-overhead"))
    parser.add_argument("--cuda-graph-tick", action="store_true")
    parser.add_argument(
        "--moderngl-path",
        help="directory containing an isolated ModernGL install for --renderer-backend gpu callers",
    )
    args = parser.parse_args()

    compile_mode = None if args.compile_mode in ("none", "None") else args.compile_mode
    if args.threads is not None:
        torch.set_num_threads(args.threads)
    devices = args.device or (["cuda", "cpu"] if torch.cuda.is_available() else ["cpu"])
    scenarios = args.scenario or ["5v5", "50v50"]

    print(
        f"Frontline decides at {DECISION_HZ} Hz -> {FRAME_BUDGET_MS:.1f} ms per frame. "
        f"stage={args.stage} compile={compile_mode} threads={torch.get_num_threads()} "
        f"policy_sides={args.sides}"
    )
    header = (
        f"{'scenario':<8}{'device':<7}{'tokens':>7}{'median':>9}{'p99':>8}"
        f"{'max':>8}{'throughput':>11}{'budget':>9}"
    )
    print(header)
    print("-" * len(header))

    rows = []
    for name in scenarios:
        for device_name in devices:
            executions = args.execution or (
                ["sequential", "streams", "threads"] if device_name == "cuda" else ["sequential"]
            )
            for execution in executions:
                if execution in {"streams", "threads"} and device_name != "cuda":
                    continue
                row = measure(
                    name,
                    torch.device(device_name),
                    steps=args.steps,
                    warmup=args.warmup,
                    compile_mode=compile_mode,
                    seed=args.seed,
                    stage=args.stage,
                    perceive_bullets=not args.no_bullet_perception,
                    policy_sides=args.sides,
                    window_size=args.window_size or None,
                    execution=execution,
                    enqueue_order=args.enqueue_order,
                    environment_step=args.environment_step,
                    moderngl_path=args.moderngl_path,
                    belief_compile_mode=args.belief_compile,
                    perception_compile_mode=args.perception_compile,
                    collision_compile_mode=args.collision_compile,
                    cuda_graph_tick=args.cuda_graph_tick,
                )
                rows.append(row)
                verdict = "OK" if row["realtime_headroom"] >= 1.0 else "MISS"
                print(
                    f"{row['scenario']:<8}{row['device']:<7}{row['entity_tokens']:>7}"
                    f"{row['median_ms']:>8.2f}m{row['p99_ms']:>7.2f}{row['max_ms']:>8.2f}"
                    f"{row['measured_hz']:>10.0f}Hz{row['realtime_headroom']:>7.2f}x {verdict}"
                    f"  {row['execution']}"
                )
                phases = row["phase_ms"]
                print(
                    f"{'':<15}phases: {row['policy_sides']} policy pass(es) "
                    f"{phases['policy']:.2f} ms  env {phases['env_step']:.2f} ms  "
                    f"render {phases['render']:.2f} ms  resets {row['resets']}"
                )

    if rows and rows[0]["window_size"] is None:
        print("\nrendering excluded: these are a lower bound on a real frame")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rows, indent=2))
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
