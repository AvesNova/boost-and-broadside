"""Benchmark the wall-clock cost of each gradient diagnostic level.

Times the PPO update phase alone — one rollout is collected and then replayed
through ``_update_epochs`` at every level, so rollout and environment cost stay
out of the comparison and what is reported is the overhead on the work the
diagnostic actually touches.

The measured overhead is a fraction of the *update* phase, not of a training
step: a run also spends time collecting rollouts and evaluating Elo, so the
fraction of total wall clock is smaller than the numbers here.

Example:
    uv run python benchmarks/gradient_diagnostics.py --device cuda --repeats 3
"""

from __future__ import annotations

import argparse
import time

import torch

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config.diagnostics import (
    GRADIENT_DIAGNOSTICS_LEVELS,
    GradientDiagnosticsConfig,
)
from boost_and_broadside.config.resolve import LaunchOverrides
from boost_and_broadside.profiles import resolve_named_profile
from boost_and_broadside.train.rl.ppo import PPOTrainer


def _build_trainer(
    profile: str,
    device: str,
    level: str,
    num_envs: int | None,
    microbatch_tokens: int | None,
) -> PPOTrainer:
    """A trainer for ``profile`` whose diagnostics run at ``level`` every update.

    The width overrides are the ordinary launch knobs, so a card that cannot
    hold the profile's default rollout can still measure the relative cost of
    the levels against each other.
    """
    overrides = LaunchOverrides(num_envs=num_envs, microbatch_tokens=microbatch_tokens)
    resolved = resolve_named_profile(profile, overrides)
    return PPOTrainer(
        train_config=resolved.train_config,
        model_config=resolved.model_config,
        ship_config=resolved.ship_config,
        device=device,
        use_wandb=False,
        scripted_agent=StochasticScriptedAgent(resolved.ship_config, StochasticAgentConfig()),
        compile_mode=None,
        gradient_diagnostics=GradientDiagnosticsConfig(level=level),
    )


def _time_update_phase(trainer: PPOTrainer, repeats: int) -> tuple[float, float]:
    """Mean seconds per update phase and peak allocated bytes over ``repeats``.

    One rollout is collected and then reused for every timed update, so the
    comparison is over identical data.
    """
    runtime = trainer._initialize_rollout_runtime()
    terminated = trainer._collect_rollout(runtime, False)
    trainer._compute_rollout_gae(runtime, terminated)
    buffers = [trainer.buffer, *trainer.aux_buffers]

    cuda = trainer.device.type == "cuda"
    if cuda:
        torch.cuda.reset_peak_memory_stats(trainer.device)
    trainer._update_epochs(all_buffers=buffers, update=1)  # warm up allocators/kernels

    elapsed = 0.0
    for index in range(repeats):
        if cuda:
            torch.cuda.synchronize(trainer.device)
        started = time.perf_counter()
        trainer._update_epochs(all_buffers=buffers, update=index + 1)
        if cuda:
            torch.cuda.synchronize(trainer.device)
        elapsed += time.perf_counter() - started
    peak = float(torch.cuda.max_memory_allocated(trainer.device)) if cuda else 0.0
    return elapsed / repeats, peak


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="rl", help="Registered training profile.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--repeats", type=int, default=3, help="Timed update phases per level.")
    parser.add_argument("--num-envs", type=int, default=None, help="Override the rollout width.")
    parser.add_argument(
        "--microbatch-tokens", type=int, default=None, help="Override entity tokens per backward."
    )
    args = parser.parse_args()

    print(
        f"profile={args.profile}  device={args.device}  repeats={args.repeats}  "
        f"num_envs={args.num_envs}  microbatch_tokens={args.microbatch_tokens}"
    )
    baseline: float | None = None
    rows: list[tuple[str, float, float, int, float]] = []
    for level in GRADIENT_DIAGNOSTICS_LEVELS:
        trainer = _build_trainer(
            args.profile, args.device, level, args.num_envs, args.microbatch_tokens
        )
        seconds, peak = _time_update_phase(trainer, args.repeats)
        terms = len(trainer._active_names)
        baseline = seconds if baseline is None else baseline
        rows.append((level, seconds, peak, terms, seconds / baseline))
        trainer.shutdown()

    print(f"\n{'level':<16}{'update s':>12}{'vs off':>10}{'peak MiB':>12}")
    for level, seconds, peak, _terms, ratio in rows:
        print(f"{level:<16}{seconds:>12.3f}{ratio:>9.2f}x{peak / 2**20:>12.1f}")
    print(
        "\nOverhead is on the update phase only; a training step also collects a "
        "rollout and runs Elo evaluation."
    )


if __name__ == "__main__":
    main()
