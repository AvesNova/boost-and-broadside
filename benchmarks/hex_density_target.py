"""Cost of the hex density target against one real rollout step.

Builds the production ``rl`` trainer (960 envs, 5v5 Frontline), times whole
rollout steps -- the primary collection step plus the evaluator's, as
``rl_kernel_profile.py`` defines one -- then times the density target alone on
the same environment state. The gate is target time under 5% of step time.

    uv run --no-sync python benchmarks/hex_density_target.py
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

GATE_FRACTION = 0.05


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _time_calls(step, count: int, device: torch.device) -> float:
    """Mean wall seconds per call over ``count`` calls, synchronized at both ends."""
    _synchronize(device)
    start = time.perf_counter()
    for index in range(count):
        step(index)
    _synchronize(device)
    return (time.perf_counter() - start) / count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compile", dest="compile_mode", default="default")
    parser.add_argument("--warmup-steps", type=int, default=8)
    parser.add_argument("--steps", type=int, default=16)
    parser.add_argument("--target-calls", type=int, default=200)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
    from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
    from boost_and_broadside.launch import resolve_training_launch
    from boost_and_broadside.train.rl.hex_density import HexDensityTarget
    from boost_and_broadside.train.rl.ppo import PPOTrainer

    compile_mode = None if args.compile_mode == "none" else args.compile_mode
    launch = resolve_training_launch(
        profile="rl",
        vram="auto",
        device=args.device,
        seed=1234,
        compile_mode=compile_mode,
        wandb=False,
        allow_probe=False,
        report=print,
    )
    resolved = launch.resolved
    device = torch.device(args.device)
    torch.manual_seed(1234)
    trainer = PPOTrainer(
        train_config=resolved.train_config,
        model_config=resolved.model_config,
        ship_config=resolved.ship_config,
        device=args.device,
        use_wandb=False,
        scripted_agent=StochasticScriptedAgent(resolved.ship_config, StochasticAgentConfig()),
        compile_mode=launch.execution.compile_mode,
        gradient_diagnostics=launch.execution.gradient_diagnostics,
    )
    runtime = trainer._initialize_rollout_runtime()
    trainer.buffer.reset()
    trainer.buffer.store_initial_hidden(runtime.hidden)
    slots = trainer._prepare_league_slots()
    trainer._compose_league_views(slots, runtime.action_state, runtime.num_ships, only_new=True)

    def rollout_step(index: int) -> None:
        (
            runtime.obs,
            runtime.hidden,
            runtime.hidden_t1,
            runtime.action_state,
            _terminated,
        ) = trainer._collect_primary_step(
            obs=runtime.obs,
            beliefs=runtime.beliefs,
            hidden=runtime.hidden,
            hidden_t1=runtime.hidden_t1,
            action_state=runtime.action_state,
            num_envs=runtime.num_envs,
            num_ships=runtime.num_ships,
            num_recurrent=runtime.num_recurrent,
            slots=slots,
            env_stream=runtime.env_stream,
            net_stream=runtime.net_stream,
        )
        trainer._collect_aux_steps(
            runtime.aux_obs,
            runtime.aux_beliefs,
            runtime.aux_hiddens,
            runtime.aux_hidden_t1s,
            runtime.aux_action_states,
            runtime.aux_last_dones,
        )
        runtime.elo_eval.step(index, False)

    for index in range(args.warmup_steps):
        rollout_step(index)
    step_seconds = _time_calls(
        lambda index: rollout_step(args.warmup_steps + index), args.steps, device
    )

    env = trainer.wrapper.env
    state = env.state
    target = HexDensityTarget(resolved.ship_config.world_size, device, compile_mode)
    # The live policy is always team 0.
    observer = torch.zeros(state.num_envs, dtype=torch.int32, device=device)
    for _ in range(3):
        target(state, observer)
    target_seconds = _time_calls(lambda _: target(state, observer), args.target_calls, device)

    fraction = target_seconds / step_seconds
    result = {
        "device": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "torch": torch.__version__,
        "compile_mode": args.compile_mode,
        "num_envs": state.num_envs,
        "num_ships": state.max_ships,
        "num_cells": target.num_cells,
        "rollout_step_ms": step_seconds * 1e3,
        "target_ms": target_seconds * 1e3,
        "fraction_of_step": fraction,
        "gate_fraction": GATE_FRACTION,
        "passes": fraction < GATE_FRACTION,
    }
    print(json.dumps(result, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(result, indent=2) + "\n")
    trainer.shutdown()


if __name__ == "__main__":
    main()
