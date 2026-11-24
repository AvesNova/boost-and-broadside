"""Train the lean core: the BC stage, then RL from the stage's averaged weights.

    python -m lean.train                       # BC (LearnConfig.bc_updates updates), then RL
    python -m lean.train --stage bc --out DIR  # BC only: DIR/bc.pt
    python -m lean.train --stage rl --start DIR/bc.pt --out DIR  # RL from a BC checkpoint
    python -m lean.train --resume DIR          # continue from DIR's latest checkpoint
    python -m lean.train --smoke [--stage ...] # a few tiny updates on the CPU

The BC stage clones the scripted teacher's full distribution on the student's own games
(half self-play, half against the scripted agent) and trains every head and the critics;
its averaged weights (LearnConfig.handoff, the EMA by default) start RL, with a fresh
optimiser and warm-up. RL is PPO with the teacher term shrinking every update until it
switches off (lean.learn.training.TeacherTerm), against the league (lean.learn.pool.League:
the scripted agent, the EMA and uniform averages, and snapshots on v2's schedule). Both
stages keep an EMA and a uniform average of the weights (lean.learn.weights), saved with the
live weights in DIR/<stage>.pt at the stage's end. Evaluation (the crossover) is its own
command: python -m lean.eval.

Checkpoints: with `--out`, every LearnConfig.checkpoint_every environment steps of a stage
DIR/checkpoints/<stage>_<steps>.pt holds everything a resume needs (none is deleted; each
league snapshot's weights are written once to DIR/snapshots/<steps>.pt, and a checkpoint names
them): the
live weights, both averages, the optimiser's state, the league, the teacher term, the step
counters, the games in progress (game state and carry) and the random generators' states.
`--resume` (a checkpoint, or a directory: its latest) continues the run with its own
settings and plan; `--updates` may extend RL. A resumed run gives the same updates as one
that never stopped (on the CPU eagerly, to rounding; tests/test_train.py).

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
from lean.learn.pool import SNAPSHOTS, SnapshotSchedule

# Shown on each update's line, in this order (the rest of the metrics are not printed).
SHOWN = (
    "loss",
    "surrogate",
    "bc",
    "bc_kl",
    "teacher_weight",
    "value",
    "next_state",
    "hit",
    "hit_labelled",
    "entropy",
    "entropy_fraction",
    "approximate_kl",
    "clip_fraction",
    "gradient_norm",
    "skipped_steps",
    "advantage_std",
    "learning_rate",
)
CHECKPOINTS = "checkpoints"


@dataclasses.dataclass
class Stage:
    """A stage in progress: everything a resume needs besides the policy and the generators."""

    name: str
    updates: int  # planned
    index: int  # updates done
    steps: int  # environment steps done
    adam_steps: int
    optimiser: torch.optim.Optimizer
    averages: WeightAverages
    league: League
    teacher: TeacherTerm
    state: object
    carry: object


def random_states(generator, device) -> dict:
    states = {"generator": generator.get_state(), "torch": torch.get_rng_state()}
    if device.type == "cuda":
        states["cuda"] = torch.cuda.get_rng_state(device)
    return states


def set_random_states(states: dict, generator, device) -> None:
    generator.set_state(states["generator"])
    torch.set_rng_state(states["torch"])
    if "cuda" in states:
        torch.cuda.set_rng_state(states["cuda"], device)


def save_run(path: Path, stage: Stage, policy, config: LearnConfig, generator, plan: dict) -> None:
    """A checkpoint a run can resume from (the module's docstring)."""
    device = next(policy.parameters()).device
    path.parent.mkdir(parents=True, exist_ok=True)
    save_checkpoint(
        path,
        policy,
        stage.averages,
        config,
        stage=stage.name,
        updates=stage.index,
        steps=stage.steps,
        adam_steps=stage.adam_steps,
        teacher=dataclasses.asdict(stage.teacher),
        planned=stage.updates,
        optimiser=stage.optimiser.state_dict(),
        league=stage.league.state(),
        game_state=stage.state,
        # The next segment's deals are drawn afresh when it starts (collect), so they are not kept.
        carry=dataclasses.replace(stage.carry, next_episodes=None),
        random=random_states(generator, device),
        plan=plan,
    )


def new_stage(
    name: str, updates: int, config: LearnConfig, game_config, policy, envs: int, generator
) -> Stage:
    """A fresh optimiser, league, game and averages from the policy's current weights."""
    device = next(policy.parameters()).device
    league = League(config, device)
    state, carry = start(
        config, game_config, policy, Opponent(scripted.ScriptedConfig(), league), envs, generator
    )
    return Stage(
        name,
        updates,
        0,
        0,
        0,
        make_optimiser(config, policy),
        WeightAverages(policy, config),
        league,
        TeacherTerm(),
        state,
        carry,
    )


def restored_stage(
    checkpoint: dict,
    config: LearnConfig,
    policy,
    generator,
    updates: int,
    source: Path,
    out: Path | None,
) -> Stage:
    """The stage a checkpoint holds, the policy loaded with its live weights; the league's
    snapshots are read from the `source` directory (copied into out/snapshots if missing)."""
    device = next(policy.parameters()).device
    policy.load_state_dict(checkpoint["live"])
    optimiser = make_optimiser(config, policy)
    optimiser.load_state_dict(checkpoint["optimiser"])
    averages = WeightAverages(policy, config)
    averages.load({key: value for key, value in checkpoint["averages"].items()})
    league = League(config, device)
    if out is not None:
        league.snapshot_directory = out / SNAPSHOTS
    league.load(checkpoint["league"], policy, averages, source)
    set_random_states(checkpoint["random"], generator, device)
    return Stage(
        checkpoint["stage"],
        updates,
        checkpoint["updates"],
        checkpoint["steps"],
        checkpoint["adam_steps"],
        optimiser,
        averages,
        league,
        TeacherTerm(**checkpoint["teacher"]),
        checkpoint["game_state"],
        checkpoint["carry"],
    )


def run_stage(
    stage: Stage,
    config: LearnConfig,
    game_config,
    policy,
    generator,
    compiled: bool,
    out: Path | None,
    plan: dict,
) -> WeightAverages:
    """The stage's remaining updates; saves out/checkpoints/<stage>_<steps>.pt every
    config.checkpoint_every environment steps and out/<stage>.pt at its end when `out`."""
    device = next(policy.parameters()).device
    ticks = config.segment_ticks
    opponent = Opponent(scripted.ScriptedConfig(), stage.league)
    league = stage.league
    store = RolloutStore(config.segments_per_update, config.stream_rollout)
    checkpoints = SnapshotSchedule((config.checkpoint_every,))
    if out is not None:
        league.snapshot_directory = out / SNAPSHOTS
    envs = stage.state.exists.shape[0]
    while stage.index < stage.updates:
        index = stage.index
        began = time.perf_counter()
        for segment in range(config.segments_per_update):
            rollout = collect(
                config,
                game_config,
                policy,
                opponent,
                stage.state,
                stage.carry,
                ticks,
                generator,
                compiled=compiled,
                store=store,
                segment=segment,
            )
            stage.state, stage.carry = rollout.state, rollout.carry
        del rollout
        update_steps = envs * ticks * config.segments_per_update
        before = stage.steps
        stage.steps += update_steps
        win_rate = league.scripted_win_rate()
        if stage.name == "bc":
            options = dict(cloning=True)
        else:
            options = dict(teacher_weight=stage.teacher.weight(config, win_rate))
        metrics = update(
            config,
            policy,
            stage.optimiser,
            store,
            step_index=stage.adam_steps,
            generator=generator,
            compiled=compiled,
            **options,
        )
        stage.adam_steps += metrics["adam_steps"]
        stage.averages.update(policy, stage.adam_steps)
        if stage.name == "rl":
            member, lost = league.after_update(
                policy, stage.averages, before, stage.steps, stage.carry.member, stage.carry.slot
            )
            stage.carry = dataclasses.replace(stage.carry, member=member)
            if lost is not None:
                stage.state, stage.carry = abandon(
                    config, game_config, policy, opponent, stage.state, stage.carry, lost, generator
                )
                print(
                    f"{stage.name} update {index} WARNING no league slot free: slot {lost}'s games "
                    "re-dealt",
                    flush=True,
                )
        stage.index += 1
        rate = update_steps / (time.perf_counter() - began)
        shown = " ".join(f"{name}={metrics[name]:.4g}" for name in SHOWN)
        shares = " ".join(
            f"{value:.2f}" for name, value in metrics.items() if name.startswith("advantage_share_")
        )
        rates = " ".join(f"{value:.2f}" for value in league.win_rates()[league.present].tolist())
        memory = ""
        if device.type == "cuda":
            memory = f" peak_allocated_gib {torch.cuda.max_memory_allocated(device) / 2**30:.2f}"
        print(
            f"{stage.name} update {index} steps {stage.steps} scripted_win_rate {win_rate:.3f} "
            f"league_members {league.present_count()} snapshots {league.snapshots} retiring "
            f"{len(league.retiring)} "
            f"league_win_rates [{rates}] steps_per_second {rate:.0f}{memory} {shown} "
            f"advantage_shares [{shares}]",
            flush=True,
        )
        warnings = [
            name.removeprefix("range_warning_")
            for name, value in metrics.items()
            if name.startswith("range_warning_") and value > 0
        ]
        if warnings:
            print(
                f"{stage.name} update {index} WARNING value support: {' '.join(warnings)}",
                flush=True,
            )
        if metrics["kl_warning"] > 0:
            epochs = " ".join(
                f"{value:.4f}"
                for name, value in metrics.items()
                if name.startswith("approximate_kl_epoch_")
            )
            print(
                f"{stage.name} update {index} WARNING approximate KL above {config.kl_warning} "
                f"(epochs: {epochs})",
                flush=True,
            )
        if metrics["skipped_steps"] > 0:
            print(
                f"{stage.name} update {index} WARNING {metrics['skipped_steps']:.0f} steps skipped "
                "(gradient norm not finite)",
                flush=True,
            )
        if out is not None and checkpoints.due(before, stage.steps):
            path = out / CHECKPOINTS / f"{stage.name}_{stage.steps:012d}.pt"
            save_run(path, stage, policy, config, generator, plan)
            print(f"{stage.name}: checkpoint {path}", flush=True)
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
        save_checkpoint(
            out / f"{stage.name}.pt",
            policy,
            stage.averages,
            config,
            stage=stage.name,
            updates=stage.index,
            steps=stage.steps,
            adam_steps=stage.adam_steps,
            teacher=dataclasses.asdict(stage.teacher),
        )
        print(f"{stage.name}: saved {out / f'{stage.name}.pt'}", flush=True)
    return stage.averages


def latest_checkpoint(path: Path) -> Path:
    """`path` itself, or the latest checkpoint under path/checkpoints (RL after BC, then by "
    "steps)."""
    if path.is_file():
        return path
    found = sorted(
        (path / CHECKPOINTS).glob("*.pt"), key=lambda item: (item.name.startswith("rl_"), item.name)
    )
    if not found:
        raise FileNotFoundError(f"no checkpoint in {path / CHECKPOINTS}")
    return found[-1]


def smoke_config() -> LearnConfig:
    """Tiny settings for --smoke: 4 environments x 2 segments x 8 ticks = 64 steps an update,
    a snapshot after every update and a checkpoint every second one."""
    return LearnConfig(
        team_size_minimum=2,
        team_size_maximum=2,
        minibatches=2,
        epochs=1,
        segments_per_update=2,
        segment_ticks=8,
        micro_batch_views=4,
        warmup_steps=2,
        bc_updates=2,
        rl_updates=2,
        league_snapshot_steps=(64,),
        checkpoint_every=128,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--stage", choices=("all", "bc", "rl"), default="all")
    parser.add_argument("--smoke", action="store_true", help="a few tiny updates on the CPU")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--envs", type=int, default=None, help="default: LearnConfig.environments")
    parser.add_argument(
        "--bc-updates", type=int, default=None, help="default: LearnConfig.bc_updates"
    )
    parser.add_argument(
        "--updates",
        type=int,
        default=None,
        help="RL updates (default LearnConfig.rl_updates; on --resume, the run's)",
    )
    parser.add_argument(
        "--start",
        type=Path,
        default=None,
        help="RL: start from this BC checkpoint's hand-off weights",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="directory for the stages' checkpoints"
    )
    parser.add_argument(
        "--resume", type=Path, default=None, help="a checkpoint, or a run directory (its latest)"
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--threads", type=int, default=0, help="CPU threads for torch; 0 keeps the default"
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "compile the rollout tick and the update's loss (lean.learn.compiled); "
            "--no-compile runs eagerly"
        ),
    )
    arguments = parser.parse_args()
    if arguments.start is not None and arguments.stage != "rl":
        parser.error("--start is the BC checkpoint RL starts from: it needs --stage rl")

    device = torch.device(arguments.device)
    if arguments.threads > 0:
        torch.set_num_threads(arguments.threads)
    # Training deals use the training spawn; evaluation (lean.eval) plays the standard one.
    game_config = game.GameConfig(training_spawn=True)
    generator = torch.Generator(device=device).manual_seed(arguments.seed)

    resumed = None
    if arguments.resume is not None:
        path = latest_checkpoint(arguments.resume)
        resumed = load_checkpoint(path, device)
        snapshots = (
            path.parent.parent / SNAPSHOTS
        )  # a checkpoint's snapshots sit beside its directory
        config = LearnConfig(**resumed["learn_config"])
        plan = dict(resumed["plan"])
        if arguments.updates is not None:
            plan["updates"] = arguments.updates
        print(
            f"resume: {path} ({resumed['stage']} update {resumed['updates']}, {resumed['steps']} "
            "steps)",
            flush=True,
        )
    else:
        config = smoke_config() if arguments.smoke else LearnConfig()
        envs = arguments.envs or (4 if arguments.smoke else config.environments)
        updates = config.rl_updates if arguments.updates is None else arguments.updates
        bc_updates = config.bc_updates if arguments.bc_updates is None else arguments.bc_updates
        plan = dict(
            stage=arguments.stage,
            envs=envs,
            updates=updates,
            bc_updates=bc_updates,
            seed=arguments.seed,
            # The smoke run stays eager: on the CPU a compile costs more than the run.
            compiled=arguments.compile and not arguments.smoke,
            out=None if arguments.out is None else str(arguments.out),
            start=None if arguments.start is None else str(arguments.start),
        )
    out = None if plan["out"] is None else Path(plan["out"])
    compiled = plan["compiled"]

    torch.manual_seed(plan["seed"])
    policy = model.Policy(model.ModelConfig() if resumed is None else resumed["model_config"]).to(
        device
    )
    stages = {"all": ("bc", "rl"), "bc": ("bc",), "rl": ("rl",)}[plan["stage"]]
    planned = {"bc": plan["bc_updates"], "rl": plan["updates"]}
    if resumed is not None:
        stages = stages[stages.index(resumed["stage"]) :]
    elif plan["start"] is not None:
        checkpoint = load_checkpoint(Path(plan["start"]), device)
        policy = model.Policy(checkpoint["model_config"]).to(device)
        policy.load_state_dict(policy_weights(checkpoint, config.handoff))

    for name in stages:
        if resumed is not None and name == resumed["stage"]:
            stage = restored_stage(
                resumed, config, policy, generator, planned[name], snapshots, out
            )
        else:
            stage = new_stage(
                name, planned[name], config, game_config, policy, plan["envs"], generator
            )
        averages = run_stage(stage, config, game_config, policy, generator, compiled, out, plan)
        if name == "bc":
            policy.load_state_dict(averages.weights(config.handoff, policy))


if __name__ == "__main__":
    main()
