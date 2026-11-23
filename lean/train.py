"""Train the lean core: the BC stage, then RL from the stage's averaged weights.

    python -m lean.train                       # BC (LearnConfig.bc_updates updates), then RL
    python -m lean.train --stage bc --out DIR  # BC only: DIR/bc.pt
    python -m lean.train --stage rl --start DIR/bc.pt --out DIR  # RL from a BC checkpoint
    python -m lean.train --smoke [--stage ...] # a few tiny updates on the CPU

The BC stage clones the scripted teacher's full distribution on the student's own games
(half self-play, half against the scripted agent) and trains every head and the critics;
its averaged weights (LearnConfig.handoff, the EMA by default) start RL, with a fresh
optimiser and warm-up. RL is PPO with the teacher term shrinking every update until it
switches off (lean.learn.training.TeacherTerm), and the league (snapshots added every
LearnConfig.snapshot_every updates). Both stages keep an EMA and a uniform average of the
weights (lean.learn.weights), saved with the live weights in DIR/<stage>.pt at the stage's
end. Evaluation (the crossover) is its own command: python -m lean.eval.

`--seed` fixes the initial weights, the scenario draws and the data order (compiled, the
rollout's draws come from torch's default generator, seeded by `--seed` too). `--compile`
(the default) compiles the rollout tick and the update's loss. An update collects
LearnConfig.segments_per_update segments of segment_ticks ticks over `--envs` environments
into one RolloutStore (streamed to pinned host memory on a GPU), then learns from them.
"""

import argparse
import dataclasses
import time
from pathlib import Path

import torch

from lean import game, model, scripted
from lean.learn import (
    League,
    LearnConfig,
    Opponent,
    RolloutStore,
    TeacherTerm,
    WeightAverages,
    abandon,
    collect,
    load_checkpoint,
    make_optimiser,
    policy_weights,
    save_checkpoint,
    start,
    update,
)

# Shown on each update's line, in this order (the rest of the metrics are not printed).
SHOWN = (
    "loss", "surrogate", "bc", "bc_kl", "teacher_weight", "value", "next_state", "hit", "hit_labelled", "entropy",
    "entropy_fraction", "approximate_kl", "clip_fraction", "gradient_norm", "skipped_steps", "advantage_std",
    "learning_rate",
)


def run_stage(stage: str, updates: int, config: LearnConfig, game_config, policy, envs: int, generator,
              compiled: bool, out: Path | None) -> WeightAverages:
    """`updates` updates of the BC stage ("bc") or of RL ("rl") from the policy's current
    weights; a fresh optimiser, league, game and averages. Saves out/<stage>.pt when `out`."""
    device = next(policy.parameters()).device
    ticks = config.segment_ticks
    optimiser = make_optimiser(config, policy)
    averages = WeightAverages(policy, config)
    league = League(config, device)
    opponent = Opponent(scripted.ScriptedConfig(), league)
    state, carry = start(config, game_config, policy, opponent, envs, generator)
    store = RolloutStore(config.segments_per_update, config.stream_rollout)
    teacher = TeacherTerm()
    steps = adam_steps = 0
    for index in range(updates):
        began = time.perf_counter()
        for segment in range(config.segments_per_update):
            rollout = collect(
                config, game_config, policy, opponent, state, carry, ticks, generator, compiled=compiled, store=store,
                segment=segment,
            )
            state, carry = rollout.state, rollout.carry
        del rollout
        update_steps = envs * ticks * config.segments_per_update
        steps += update_steps
        win_rate = league.scripted_win_rate()
        if stage == "bc":
            options = dict(cloning=True)
        else:
            options = dict(teacher_weight=teacher.weight(config, win_rate))
        metrics = update(
            config, policy, optimiser, store, step_index=adam_steps, generator=generator, compiled=compiled, **options
        )
        adam_steps += metrics["adam_steps"]
        averages.update(policy, adam_steps)
        if stage == "rl":
            replaced = league.maybe_add(policy, index + 1)
            if replaced is not None:
                state, carry = abandon(config, game_config, policy, opponent, state, carry, replaced, generator)
        rate = update_steps / (time.perf_counter() - began)
        shown = " ".join(f"{name}={metrics[name]:.4g}" for name in SHOWN)
        shares = " ".join(f"{value:.2f}" for name, value in metrics.items() if name.startswith("advantage_share_"))
        rates = " ".join(f"{value:.2f}" for value in league.win_rates()[league.present].tolist())
        memory = ""
        if device.type == "cuda":
            memory = f" peak_allocated_gib {torch.cuda.max_memory_allocated(device) / 2**30:.2f}"
        print(
            f"{stage} update {index} steps {steps} scripted_win_rate {win_rate:.3f} league_win_rates [{rates}] "
            f"steps_per_second {rate:.0f}{memory} {shown} advantage_shares [{shares}]",
            flush=True,
        )
        warnings = [name.removeprefix("range_warning_") for name, value in metrics.items() if name.startswith("range_warning_") and value > 0]
        if warnings:
            print(f"{stage} update {index} WARNING value support: {' '.join(warnings)}", flush=True)
        if metrics["kl_warning"] > 0:
            epochs = " ".join(f"{value:.4f}" for name, value in metrics.items() if name.startswith("approximate_kl_epoch_"))
            print(f"{stage} update {index} WARNING approximate KL above {config.kl_warning} (epochs: {epochs})", flush=True)
        if metrics["skipped_steps"] > 0:
            print(f"{stage} update {index} WARNING {metrics['skipped_steps']:.0f} steps skipped (gradient norm not finite)", flush=True)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            out / f"{stage}.pt", policy, averages, config, stage=stage, updates=updates, steps=steps,
            adam_steps=adam_steps, teacher=dataclasses.asdict(teacher),
        )
        print(f"{stage}: saved {out / f'{stage}.pt'}", flush=True)
    return averages


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage", choices=("all", "bc", "rl"), default="all")
    parser.add_argument("--smoke", action="store_true", help="a few tiny updates on the CPU")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--envs", type=int, default=None, help="default: LearnConfig.environments")
    parser.add_argument("--bc-updates", type=int, default=None, help="default: LearnConfig.bc_updates")
    parser.add_argument("--updates", type=int, default=1000, help="RL updates")
    parser.add_argument("--start", type=Path, default=None, help="RL: start from this BC checkpoint's hand-off weights")
    parser.add_argument("--out", type=Path, default=None, help="directory for the stages' checkpoints")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--threads", type=int, default=0, help="CPU threads for torch; 0 keeps the default")
    parser.add_argument(
        "--compile", action=argparse.BooleanOptionalAction, default=True,
        help="compile the rollout tick and the update's loss (lean.learn.compiled); --no-compile runs eagerly",
    )
    arguments = parser.parse_args()

    config = LearnConfig()
    envs, updates = arguments.envs or config.environments, arguments.updates
    if arguments.smoke:
        config = LearnConfig(
            team_size_minimum=2, team_size_maximum=2, minibatches=2, epochs=1, snapshot_every=1, segments_per_update=2,
            segment_ticks=8, micro_batch_views=4, warmup_steps=2, bc_updates=2,
        )
        envs, updates = arguments.envs or 4, 2
    bc_updates = config.bc_updates if arguments.bc_updates is None else arguments.bc_updates

    device = torch.device(arguments.device)
    # The smoke run stays eager: on the CPU a compile costs more than the run.
    compiled = arguments.compile and not arguments.smoke
    if arguments.threads > 0:
        torch.set_num_threads(arguments.threads)
    torch.manual_seed(arguments.seed)
    generator = torch.Generator(device=device).manual_seed(arguments.seed)
    # Training deals use the training spawn; evaluation (lean.eval) plays the standard one.
    game_config = game.GameConfig(training_spawn=True)
    policy = model.Policy(model.ModelConfig()).to(device)

    if arguments.stage in ("all", "bc"):
        averages = run_stage("bc", bc_updates, config, game_config, policy, envs, generator, compiled, arguments.out)
        policy.load_state_dict(averages.weights(config.handoff, policy))
    elif arguments.start is not None:
        checkpoint = load_checkpoint(arguments.start, device)
        policy = model.Policy(checkpoint["model_config"]).to(device)
        policy.load_state_dict(policy_weights(checkpoint, config.handoff))
    if arguments.stage in ("all", "rl"):
        run_stage("rl", updates, config, game_config, policy, envs, generator, compiled, arguments.out)


if __name__ == "__main__":
    main()
