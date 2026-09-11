"""Kernel-level profile of a bounded window of the real RL rollout and update.

The phase profile says *where* wall clock goes; this says whether that time is
GPU work or CPU launch overhead, and which operators dominate each. It runs the
real trainer, warms up, then profiles a fixed number of rollout steps and one
optimizer minibatch so the traces stay small enough to summarize.
"""

from __future__ import annotations

import argparse
import time

import torch


def summarize(events, wall: float, label: str, unit_count: int, unit: str, rows: int) -> None:
    """Print the CPU/GPU split and the top operators for one profiled window."""
    cuda_total = sum(event.self_device_time_total for event in events) / 1e6
    cpu_total = sum(event.self_cpu_time_total for event in events) / 1e6
    calls = sum(event.count for event in events)
    print(f"\n=== {label} ===")
    print(f"wall (profiled, profiler overhead included) : {wall:.3f} s")
    print(f"summed self CUDA time                       : {cuda_total:.3f} s")
    print(f"summed self CPU time                        : {cpu_total:.3f} s")
    print(f"GPU busy fraction of profiled wall          : {100 * cuda_total / wall:.1f}%")
    print(f"total profiler events                       : {calls:,}")
    print(f"events per {unit:<32s}: {calls / unit_count:,.0f}")
    print("\n--- top by self CUDA time ---")
    print(events.table(sort_by="self_device_time_total", row_limit=rows))
    print("\n--- top by self CPU time ---")
    print(events.table(sort_by="self_cpu_time_total", row_limit=rows))


class _StopAfterMinibatches(Exception):
    """Sentinel raised to end a profiled update after a bounded number of steps."""


def profile_update(trainer, runtime, args) -> None:
    """Collect one logical batch, then profile a bounded run of optimizer minibatches.

    Profiling a whole epoch buffers millions of events and exhausts host RAM, so
    the run is cut short with a sentinel after ``--update-minibatches`` optimizer
    steps. Everything up to that point is the real ``_update_epochs`` body.
    """
    import time as _time

    buffers = trainer._collect_host_rollouts(runtime, False)

    limit = args.update_minibatches
    original_step = trainer.optim.step
    counter = {"n": 0}

    def counted_step(*step_args, **step_kwargs):
        original_step(*step_args, **step_kwargs)
        counter["n"] += 1
        if counter["n"] >= limit:
            raise _StopAfterMinibatches

    def run_bounded() -> None:
        counter["n"] = 0
        try:
            trainer._update_epochs(all_buffers=buffers, precomputed=True, update=1)
        except _StopAfterMinibatches:
            pass

    trainer.optim.step = counted_step
    run_bounded()  # warm
    torch.cuda.synchronize()

    unprofiled = _time.perf_counter()
    run_bounded()
    torch.cuda.synchronize()
    unprofiled = _time.perf_counter() - unprofiled

    start = _time.perf_counter()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as prof:
        run_bounded()
        torch.cuda.synchronize()
    wall = _time.perf_counter() - start
    trainer.optim.step = original_step

    micro = trainer.cfg.rollouts_per_update
    print(f"\noptimizer minibatches profiled: {limit}  (shard chunks each: {micro})")
    print(
        f"unprofiled wall for the same work: {unprofiled:.3f} s "
        f"({unprofiled / limit * 1e3:.1f} ms per optimizer minibatch)"
    )
    summarize(
        prof.key_averages(),
        wall,
        f"PPO update window ({limit} optimizer minibatches)",
        limit,
        "optimizer minibatch",
        args.rows,
    )
    trainer.shutdown()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-steps", type=int, default=24)
    parser.add_argument("--profile-steps", type=int, default=16)
    parser.add_argument("--rows", type=int, default=30)
    parser.add_argument(
        "--phase",
        choices=("rollout", "update", "eval", "primary"),
        default="rollout",
        help=(
            "rollout profiles whole collection steps; primary and eval profile the two "
            "halves of one separately; update profiles optimizer minibatches"
        ),
    )
    parser.add_argument(
        "--update-minibatches",
        type=int,
        default=4,
        help="optimizer minibatches to profile in --phase update (whole epochs exhaust host RAM)",
    )
    parser.add_argument("--compile", dest="compile_mode", default="default")
    args = parser.parse_args()

    from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
    from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
    from boost_and_broadside.launch import resolve_training_launch
    from boost_and_broadside.train.rl.ppo import PPOTrainer

    launch = resolve_training_launch(
        profile="rl",
        vram="auto",
        device="cuda",
        seed=1234,
        compile_mode=None if args.compile_mode == "none" else args.compile_mode,
        wandb=False,
        allow_probe=False,
        report=print,
    )
    resolved = launch.resolved
    torch.manual_seed(1234)
    trainer = PPOTrainer(
        train_config=resolved.train_config,
        model_config=resolved.model_config,
        ship_config=resolved.ship_config,
        device="cuda",
        use_wandb=False,
        scripted_agent=StochasticScriptedAgent(resolved.ship_config, StochasticAgentConfig()),
        compile_mode=launch.execution.compile_mode,
        gradient_diagnostics=launch.execution.gradient_diagnostics,
    )
    runtime = trainer._initialize_rollout_runtime()

    if args.phase == "update":
        profile_update(trainer, runtime, args)
        return

    trainer.buffer.reset()
    trainer.buffer.store_initial_hidden(runtime.hidden)
    slots = trainer._prepare_league_slots(runtime.num_recurrent)

    def primary_only(index: int) -> None:
        primary = trainer._collect_primary_step(
            obs=runtime.obs,
            beliefs=runtime.beliefs,
            hidden=runtime.hidden,
            hidden_t1=runtime.hidden_t1,
            action_buffer=runtime.action_buffer,
            num_envs=runtime.num_envs,
            num_ships=runtime.num_ships,
            num_recurrent=runtime.num_recurrent,
            slots=slots,
            env_stream=runtime.env_stream,
            net_stream=runtime.net_stream,
        )
        (
            runtime.obs,
            runtime.hidden,
            runtime.hidden_t1,
            runtime.action_buffer,
            _terminated,
        ) = primary
        runtime.elo_eval.step(index, False)

    def eval_only(index: int) -> None:
        runtime.elo_eval.step(index, False)

    def rollout_step(index: int) -> None:
        primary_only(index)
        eval_only(index)

    # The training environment and the evaluator share no state within a step --
    # separate TensorEnv instances, separate recurrent state, and the policy
    # weights they both read do not change during a rollout. So either half can
    # be advanced on its own, which is what makes profiling them separately
    # meaningful rather than an artefact.
    phases = {"rollout": rollout_step, "primary": primary_only, "eval": eval_only}
    step = phases[args.phase]

    for index in range(args.warmup_steps):
        step(index)
    torch.cuda.synchronize()

    # Unprofiled wall for the same work, so the profiler's own overhead is visible.
    plain_start = time.perf_counter()
    for index in range(args.warmup_steps, args.warmup_steps + args.profile_steps):
        step(index)
    torch.cuda.synchronize()
    plain = time.perf_counter() - plain_start

    base = args.warmup_steps + args.profile_steps
    wall_start = time.perf_counter()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as prof:
        for index in range(base, base + args.profile_steps):
            step(index)
        torch.cuda.synchronize()
    wall = time.perf_counter() - wall_start

    events = prof.key_averages()
    cuda_total = sum(e.self_device_time_total for e in events) / 1e6
    cpu_total = sum(e.self_cpu_time_total for e in events) / 1e6
    launches = sum(e.count for e in events)
    print(f"\n=== {args.phase} window: {args.profile_steps} steps ===")
    print(
        f"unprofiled wall for the same work           : {plain:.3f} s "
        f"({plain / args.profile_steps * 1e3:.1f} ms per step)"
    )
    print(f"profiled wall (profiler overhead included)  : {wall:.3f} s")
    print(
        f"summed self CUDA time                       : {cuda_total:.3f} s "
        f"({cuda_total / args.profile_steps * 1e3:.1f} ms per step)"
    )
    print(f"summed self CPU time                        : {cpu_total:.3f} s")
    print(f"GPU busy fraction of UNPROFILED wall        : {100 * cuda_total / plain:.1f}%")
    print(f"total profiler events                       : {launches:,}")
    print(f"events per step                             : {launches / args.profile_steps:,.0f}")
    print("\n--- top by self CUDA time ---")
    print(events.table(sort_by="self_device_time_total", row_limit=args.rows))
    print("\n--- top by self CPU time ---")
    print(events.table(sort_by="self_cpu_time_total", row_limit=args.rows))
    trainer.shutdown()


if __name__ == "__main__":
    main()
