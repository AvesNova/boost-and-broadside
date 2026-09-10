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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--warmup-steps", type=int, default=24)
    parser.add_argument("--profile-steps", type=int, default=16)
    parser.add_argument("--rows", type=int, default=30)
    parser.add_argument("--compile", dest="compile_mode", default="reduce-overhead")
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
    trainer.buffer.reset()
    trainer.buffer.store_initial_hidden(runtime.hidden)
    slots = trainer._prepare_league_slots(runtime.num_recurrent)

    def rollout_step(index: int) -> None:
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

    for index in range(args.warmup_steps):
        rollout_step(index)
    torch.cuda.synchronize()

    start = time.perf_counter()
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
    ) as prof:
        for index in range(args.warmup_steps, args.warmup_steps + args.profile_steps):
            rollout_step(index)
        torch.cuda.synchronize()
    wall = time.perf_counter() - start

    events = prof.key_averages()
    cuda_total = sum(e.self_device_time_total for e in events) / 1e6
    cpu_total = sum(e.self_cpu_time_total for e in events) / 1e6
    launches = sum(e.count for e in events)
    print(f"\n=== rollout window: {args.profile_steps} steps ===")
    print(f"wall (profiled, profiler overhead included) : {wall:.3f} s")
    print(f"summed self CUDA time                       : {cuda_total:.3f} s")
    print(f"summed self CPU time                        : {cpu_total:.3f} s")
    print(f"GPU busy fraction of profiled wall          : {100 * cuda_total / wall:.1f}%")
    print(f"total profiler events                       : {launches:,}")
    print(f"events per rollout step                     : {launches / args.profile_steps:,.0f}")
    print("\n--- top by self CUDA time ---")
    print(events.table(sort_by="self_device_time_total", row_limit=args.rows))
    print("\n--- top by self CPU time ---")
    print(events.table(sort_by="self_cpu_time_total", row_limit=args.rows))
    print("\n--- top by call count ---")
    print(events.table(sort_by="count", row_limit=args.rows))
    trainer.shutdown()


if __name__ == "__main__":
    main()
