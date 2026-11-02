"""The `bnb` command: `bnb train` trains or clones, `bnb runs` lists runs and uploads
checkpoints, `bnb ship-card` charts a ship card.

An entry point (code style, Layers): it sets the process-wide options its commands need (the
seed, the matmul precision, the charts' backend). The play shell (`python -m bnb.play`) and
`bench` are the other entry points."""

import dataclasses
import json
import os
import pathlib
import subprocess
import sys
from typing import Annotated

import tyro


@dataclasses.dataclass(frozen=True)
class Train:
    """Train a model with PPO (learner spec), or clone the scripted agent (BC), in a run
    directory (runs spec): Yemong, or the toy relational model.

    preset: a training preset (deathmatch, deathmatch_wide, frontline), which sets the mode,
    the sizes and the spawn groups, or mixed, the default training mix of all three in one
    batch; without one, the mode and sizes below.
    precision: bfloat16 (the model's matmuls), float32 or tf32; old_log_probabilities:
    rollout or recomputed (by unroll before the first epoch).
    updates: the updates to train (a resume continues to the run's total).
    seed: the run's seed (TrainConfig.seed: the process and the learner's streams).
    set: TrainConfig fields by dotted path, such as learner.entropy=-0.01 (JSON values);
    fields inside tuples (such as pools) cannot be set.
    resume: continue the run named, from its newest checkpoint or the one at --at, with
    every setting from the checkpoint; only the budget, the precision, the micro-batch
    rows, the checkpoint cadence and --set of OVERRIDABLE fields (the learning rate, the
    warm-up, the loss coefficients, the reward weights) may change.
    fork (--from): a new run with these settings, starting from the named run's checkpoint
    (newest, or at --at): its model, optimizer, return scale, league slot and environment
    steps; --updates counts from the fork.
    init_from: a warm start from a policy file: its weights and return scale, a fresh
    optimizer (the BC -> RL hand-off).
    print_config: print the resolved configuration and exit; probe: run the memory probe
    (the setup and one update), print the peaks and exit, writing no run.
    name: the run's name (default: date, preset or mode, and two words).
    runs_directory: where runs live (default: runs/ at the repository root, or in the
    current directory outside a repository).
    checkpoint_every: environment steps between training checkpoints (default: about 30
    minutes at the run's measured throughput).
    objective: ppo, or bc (behaviour cloning against self-play and the scripted agent until
    a stop rule fires, then the clone is saved as the run's policy file; updates is unused,
    and a BC run is not resumed or forked); teacher_temperature and teacher_floor set the
    labels' entropy; bc_budget_steps: BC's budget in environment steps.
    calibrate_entropy: print the scripted agent's natural entropy over its own games of the
    preset and the temperature whose labels reach this mean entropy (nats per living ship
    step) at teacher_floor, then exit, writing no run; calibration_segments: how many
    rollout segments of 128 ticks it records.
    """

    preset: str | None = None
    mode: str = "team_deathmatch"
    ships_per_team: int = 5
    num_fields: int = 10
    arena_radius: float = 2600.0
    environments: int = 1024
    updates: int = 100
    shards: int = 1
    model: str = "yemong"
    precision: str = "bfloat16"
    old_log_probabilities: str = "rollout"
    micro_batch_rows: int = 160
    seed: int = 0
    device: str = "cuda"
    set: Annotated[tuple[str, ...], tyro.conf.UseAppendAction] = ()
    resume: str | None = None
    fork: Annotated[str | None, tyro.conf.arg(name="from")] = None
    at: int | None = None
    init_from: str | None = None
    print_config: bool = False
    probe: bool = False
    name: str | None = None
    runs_directory: str | None = None
    checkpoint_every: int | None = None
    objective: str = "ppo"
    teacher_temperature: float = 1.0
    teacher_floor: float = 0.01
    bc_budget_steps: int = 500_000_000
    calibrate_entropy: float | None = None
    calibration_segments: int = 4


# What a resume may be given besides --resume: the rest comes from the checkpoint.
RESUME_OPTIONS = (
    "resume",
    "at",
    "updates",
    "precision",
    "micro_batch_rows",
    "set",
    "checkpoint_every",
    "device",
    "runs_directory",
    "print_config",
)
# The resume options that set a TrainConfig field: their paths.
RESUME_FIELDS = {
    "updates": "updates",
    "precision": "precision",
    "micro_batch_rows": "learner.micro_batch_rows",
}


@dataclasses.dataclass(frozen=True)
class Runs:
    """List the runs (`bnb runs`), show one (`bnb runs show RUN`), or upload a chosen
    checkpoint (`bnb runs upload RUN [--at STEP] [--league]`) as a W&B artifact.

    action: list, show or upload; run: the run's name (show and upload).
    at: the checkpoint's step (default: the newest); league: upload the league's snapshots
    instead; to: the target (wandb only, for now); runs_directory: as for bnb train.
    """

    action: tyro.conf.Positional[str] = "list"
    run: tyro.conf.Positional[str | None] = None
    at: int | None = None
    league: bool = False
    to: str = "wandb"
    runs_directory: str | None = None


@dataclasses.dataclass(frozen=True)
class ShipCard:
    """Print every handle, parameter and shown quantity of a ship card per type, and write its
    charts (the Ps curve, the E-M doghouse, the similarity ratios against the P-51D). The
    charts need the optional tools extra (uv sync --extra tools).

    card: a JSON file of ship-card fields (per-type lists; missing fields keep the default);
    the default card without one.
    out: the directory the charts are written to.
    """

    card: str | None = None
    out: str = "ship-card"


COMMANDS = ("train", "runs", "ship-card")


def main() -> None:
    """The `bnb` command: `bnb train ...` trains, `bnb runs ...` lists and shows runs and
    uploads checkpoints, `bnb ship-card ...` charts a card."""
    os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")  # Triton 3.9's default, made explicit
    command, arguments = (sys.argv[1:2] or [""])[0], sys.argv[2:]
    if command == "ship-card":
        os.environ.setdefault("MPLBACKEND", "Agg")  # the charts go to files: no display
        ship_card(tyro.cli(ShipCard, args=arguments, prog="bnb ship-card"))
    elif command == "train":
        train(tyro.cli(Train, args=arguments, prog="bnb train"), given(arguments))
    elif command == "runs":
        runs(tyro.cli(Runs, args=arguments, prog="bnb runs"))
    else:
        raise SystemExit(f"usage: bnb {{{','.join(COMMANDS)}}} ... (bnb train --help)")


def given(arguments: list[str]) -> set[str]:
    """The Train fields the command line set (`--from` is the fork field)."""
    names = {
        argument[2:].partition("=")[0].replace("-", "_")
        for argument in arguments
        if argument.startswith("--")
    }
    return {"fork" if name == "from" else name for name in names}


def runs_directory_of(given: str | None) -> pathlib.Path:
    """`given`, or `runs/` at the repository root (`git rev-parse --show-toplevel`), or in
    the current directory outside a repository."""
    if given is not None:
        return pathlib.Path(given)
    found = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    root = found.stdout.strip() if found.returncode == 0 else ""
    return pathlib.Path(root or ".") / "runs"


def ship_card(options: ShipCard) -> None:
    """`bnb ship-card`: print the card's chart at 30 Hz and write its charts to options.out;
    exits with the reason when matplotlib is missing."""
    from bnb.contracts import GameConfig
    from bnb.game.api import rules_of
    from bnb.game.handling import chart

    rules = rules_of(GameConfig(1, 1, 1000.0))  # flight and weapons need no arena
    types = chart.load_card(options.card)
    card = chart.report(rules, types)
    print("\n".join(chart.lines(rules, types, card)))
    try:
        paths = chart.write_charts(rules, types, card, options.out)
    except ImportError as error:
        raise SystemExit(str(error)) from error
    for path in paths:
        print(f"wrote {path}")


def _requested(options: Train):
    """The TrainConfig the options describe, `--set` applied."""
    from bnb.contracts import GameConfig
    from bnb.contracts.pools import environment_multiple
    from bnb.game.modes.rules import PRESETS
    from bnb.learn import bc
    from bnb.learn.ppo import LearnerConfig
    from bnb.runtime import cloning
    from bnb.runtime.train import LINEUP, TrainConfig, mixed

    cloned = options.objective == "bc"
    pools = cloning.LINEUP if cloned else LINEUP
    if options.preset == "mixed":
        game, pools = mixed(pools)
    elif options.preset is not None:
        if options.preset not in PRESETS:
            raise SystemExit(
                f"unknown preset {options.preset!r}; one of {sorted(PRESETS) + ['mixed']}"
            )
        game = PRESETS[options.preset]
    else:
        game = GameConfig(
            options.ships_per_team, options.num_fields, options.arena_radius, mode=options.mode
        )
    multiple = environment_multiple(pools)
    if options.environments % multiple:
        raise SystemExit(
            f"--environments {options.environments} does not split into the pools' shares: "
            f"use a multiple of {multiple}"
        )
    settings = dict(
        old_log_probabilities=options.old_log_probabilities,
        micro_batch_rows=options.micro_batch_rows,
    )
    learner = bc.config(**settings) if cloned else LearnerConfig(**settings)
    try:
        config = TrainConfig(
            game=game,
            pools=pools,
            environments=options.environments,
            updates=options.updates,
            shards=options.shards,
            model=options.model,
            precision=options.precision,
            learner=learner,
            seed=options.seed,
            objective=options.objective,
            teacher=bc.TeacherConfig(options.teacher_temperature, options.teacher_floor),
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    return with_settings(config, map(parse_setting, options.set))


def parse_setting(setting: str) -> tuple[str, object]:
    """`path=value` as (path, the value as JSON, else the string)."""
    path, separator, text = setting.partition("=")
    if not separator:
        raise SystemExit(f"--set {setting!r}: expected path=value")
    try:
        return path, json.loads(text)
    except json.JSONDecodeError:
        return path, text


def with_settings(config, settings):
    """`config` with each (path, value) of `settings` set."""
    from bnb.runs.serialise import SerialiseError, from_json, to_json

    data = to_json(config)
    for path, value in settings:
        *parents, leaf = path.split(".")
        node = data
        for part in parents:
            if not isinstance(node.get(part), dict):
                raise SystemExit(f"--set {path}: TrainConfig has no {part!r} there")
            node = node[part]
        if leaf not in node:
            raise SystemExit(f"--set {path}: TrainConfig has no field {path}")
        node[leaf] = value
    try:
        return from_json(type(config), data)[0]
    except (SerialiseError, ValueError) as error:
        raise SystemExit(f"--set: {error}") from error


def set_precision(precision: str) -> None:
    """The process-wide matmul precision of a run: TF32 for "tf32", full float32 otherwise
    (bf16 is the model's own autocast)."""
    import torch

    torch.set_float32_matmul_precision("high" if precision == "tf32" else "highest")


def train(options: Train, given_options: set[str]) -> None:
    """`bnb train`: a new run, a resume, a fork or a warm start, logging a line per update;
    exits with the reason for a setting that does not fit."""
    import torch

    from bnb.runs import checkpoint, probe, session
    from bnb.runs.checkpoint import OVERRIDABLE, CheckpointError
    from bnb.runs.probe import MemoryBudgetError
    from bnb.runs.records import Run, run_name
    from bnb.runs.serialise import SerialiseError, to_json
    from bnb.runtime.train import Trainer

    runs_directory = runs_directory_of(options.runs_directory)
    device = torch.device(options.device)
    starts = [name for name in ("resume", "fork", "init_from") if getattr(options, name)]
    if len(starts) > 1:
        raise SystemExit(f"choose one of --resume, --from and --init-from, not {starts}")
    if starts and options.objective == "bc":
        flag = {"resume": "--resume", "fork": "--from", "init_from": "--init-from"}[starts[0]]
        raise SystemExit(f"a BC run starts fresh: --objective bc takes no {flag}")
    if options.at is not None and not (options.resume or options.fork):
        raise SystemExit("--at chooses a checkpoint of --resume or --from")
    try:
        if options.resume:
            run = Run.open(runs_directory, options.resume)
            if run.live():
                raise SystemExit(
                    f"run {options.resume} is running as process "
                    f"{run.record['pid']} on this host; stop it (Ctrl-C) before "
                    f"resuming it"
                )
            source = run.checkpoint_at(options.at)
            fixed = sorted(given_options - set(RESUME_OPTIONS))
            if fixed:
                flags = ", ".join(
                    "--from" if name == "fork" else "--" + name.replace("_", "-") for name in fixed
                )
                raise SystemExit(
                    f"a resume takes {flags} "
                    f"from the checkpoint and cannot change it: start a new run "
                    f"from it with --from {options.resume}"
                )
            settings = [
                (RESUME_FIELDS[name], getattr(options, name))
                for name in RESUME_FIELDS
                if name in given_options
            ]
            settings += [parse_setting(setting) for setting in options.set]
            outside = [path for path, _ in settings if path not in OVERRIDABLE]
            if outside:
                raise SystemExit(
                    f"a resume cannot --set {', '.join(outside)}: it may change "
                    f"only {', '.join(OVERRIDABLE)}; to change more, start a new "
                    f"run with --from {options.resume}"
                )
            config = with_settings(checkpoint.training_config(source)[0], settings)
        else:
            config = _requested(options)
        if options.print_config:
            print(json.dumps(to_json(config), indent=1))
            return
        if options.resume and config.objective == "bc":
            raise SystemExit(
                f"run {options.resume} is a BC run, which is not resumed: start a "
                f"new one, or RL from its policy file with --init-from"
            )
        torch.manual_seed(config.seed)
        set_precision(config.precision)
        if options.calibrate_entropy is not None:
            calibrate(options, config, device)
            return
        if options.probe:
            trainer = Trainer(config, device)
            probe.begin(device)
            trainer.update()
            peaks = probe.measure(device)
            print(probe.describe(peaks))
            probe.check(
                peaks,
                f"{config.environments} environments, "
                f"{config.learner.micro_batch_rows} micro-batch rows",
            )
            return
        if options.resume:
            trainer = checkpoint.load_training(source, device, config)
            checkpoint.restore_default_generators(source, device)
            step = trainer.ppo.environment_steps
            rewound = run.set_aside_after(step)
            overrides = dict(settings)
            if options.checkpoint_every is not None:
                overrides["checkpoint_every"] = options.checkpoint_every
                run.update(checkpoint_every=options.checkpoint_every)
            warning = run.start_segment(step, overrides, rewound_from=max(rewound, default=None))
            if warning:
                print(f"warning: {warning}", flush=True)
        else:
            label = options.preset or config.game.mode
            name = options.name or run_name(label, runs_directory)
            parent = None
            if options.fork:
                parent_run = Run.open(runs_directory, options.fork)
                source = parent_run.checkpoint_at(options.at)
                trainer = checkpoint.fork_training(source, config, device)
                parent = dict(run=options.fork, step=trainer.ppo.environment_steps)
            elif options.init_from:
                trainer = checkpoint.warm_start(options.init_from, config, device)
                parent = dict(policy=str(pathlib.Path(options.init_from).resolve()))
            else:
                trainer = Trainer(config, device)
            run = Run.create(
                runs_directory, name, config, sys.argv, options.checkpoint_every, parent
            )
            run.start_segment(trainer.ppo.environment_steps)
            print(f"run {name} in {run.directory}", flush=True)
    except (
        CheckpointError,
        SerialiseError,
        MemoryBudgetError,
        FileNotFoundError,
        FileExistsError,
    ) as error:
        raise SystemExit(str(error)) from error

    def log(row: dict) -> None:
        rewards = " ".join(
            f"{name} {sum(pool['reward_per_second'].values()):+.3f}"
            for name, pool in row["pools"].items()
        )
        print(
            f"update {row['update']:4d}  rollout {row['rollout_seconds'] * 1e3:6.0f} ms  "
            f"learn {row['learn_seconds'] * 1e3:6.0f} ms  entropy {row['loss/entropy']:.2f}  "
            f"kl {row['approximate_kl']:+.4f}  clip {row['clip_fraction']:.3f}  "
            f"drift {row['ratio_drift_max']:.3f}  nonfinite {row['nonfinite_gradients']}/"
            f"{row['nonfinite_beliefs']}  reward/s: {rewards}",
            flush=True,
        )

    try:
        if config.objective == "bc":
            from bnb.learn import bc

            status = session.clone(
                run, trainer, bc.StopConfig(budget_steps=options.bc_budget_steps), log_cloning
            )
            print(
                f"stopped by {run.record['stopped_by']}; the clone is {run.policy_path}", flush=True
            )
        else:
            status = session.train(run, trainer, log)
    except MemoryBudgetError as error:  # the run is marked failed with the same message
        raise SystemExit(str(error)) from error
    print(f"run {run.record['name']} {status} at step {trainer.ppo.environment_steps}")


def log_cloning(row: dict) -> None:
    scores = " ".join(
        f"{kind} {value['score']:.3f}"
        for kind, value in row["scores"].items()
        if value["score"] is not None
    )
    # the score stop's window against the scripted agent, once full: its score and bound
    window = (
        ""
        if row["window_score"] is None
        else f"  window {row['window_score']:.3f} (bound {row['window_lower_bound']:.3f})"
    )
    print(
        f"update {row['update']:4d}  steps {row['environment_steps']:>11,}  "
        f"rollout {row['rollout_seconds'] * 1e3:6.0f} ms  "
        f"learn {row['learn_seconds'] * 1e3:6.0f} ms  "
        f"bc kl {row['bc_kl']:.4f}  teacher {row['teacher_entropy']:.3f}  "
        f"student {row['student_entropy']:.3f}  scores: {scores}{window}",
        flush=True,
    )


def calibrate(options: Train, config, device) -> None:
    """`bnb train --calibrate-entropy`: the teacher's natural entropy over the scripted
    agent's own games of the configured game and pools, and the temperature for the target."""
    from bnb.learn import bc
    from bnb.runtime import cloning

    tokens = cloning.teacher_distributions(
        config.game, config.pools, config.environments, options.calibration_segments, device
    )
    natural = bc.mean_entropy(tokens, bc.TeacherConfig(1.0, 0.0))
    print(
        f"natural entropy {natural:.4f} nats per living ship-step over "
        f"{tokens.shape[0]} ship-steps",
        flush=True,
    )
    try:
        chosen = bc.calibrate(tokens, options.calibrate_entropy, options.teacher_floor)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    print(
        f"temperature {chosen.temperature:.4f} at floor {chosen.floor} gives "
        f"{bc.mean_entropy(tokens, chosen):.4f}"
    )


def runs(options: Runs) -> None:
    """`bnb runs`: list the runs, show one, or upload a checkpoint."""
    from bnb.runs.records import Run, list_runs

    runs_directory = runs_directory_of(options.runs_directory)
    if options.action == "list":
        print(f"{'run':44s} {'status':16s} {'newest step':>12s}  {'started':25s} commit")
        for run in list_runs(runs_directory):
            record = run.record
            segments = record["segments"]
            commit = (segments[-1]["commit"] or "")[:10] if segments else ""
            started = segments[0]["start_time"] if segments else ""
            newest = record["newest_checkpoint"]
            print(
                f"{record['name']:44s} {run.shown_status():16s} "
                f"{'' if newest is None else newest:>12}  {started:25s} {commit}"
            )
        return
    if options.run is None:
        raise SystemExit(f"bnb runs {options.action} needs a run name")
    try:
        run = Run.open(runs_directory, options.run)
        if options.action == "show":
            print("\n".join(show(run)))
        elif options.action == "upload":
            from bnb.runs import upload

            if options.to != "wandb":
                raise SystemExit(f"--to {options.to!r}: the only target is wandb for now")
            print(upload.upload(run, options.at, options.league))
        else:
            raise SystemExit(f"bnb runs {options.action!r}: the actions are list, show, upload")
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error


def show(run) -> list[str]:
    """`bnb runs show`: the record, readable."""
    record = run.record
    lines = [
        f"run {record['name']}: {run.shown_status()}"
        + (f" ({record['message']})" if record["message"] else ""),
        f"  directory   {run.directory}",
        f"  command     {' '.join(record['command'])}",
        f"  host, pid   {record['host']}, {record['pid']}",
        f"  W&B id      {record['wandb_id']}",
        f"  parent      {record['parent']}",
        f"  cadence     every {record['checkpoint_every']} environment steps",
        f"  probe       {record['probe']}",
        f"  checkpoints {run.checkpoint_steps()} (newest {record['newest_checkpoint']}, "
        f"final {record['final_checkpoint']})",
        "  segments:",
    ]
    for index, segment in enumerate(record["segments"]):
        dirty = " (dirty)" if segment["dirty"] else ""
        lines.append(
            f"    {index}: steps {segment['start_step']} to {segment['end_step']} at "
            f"{segment['commit']}{dirty}, {segment['start_time']} to "
            f"{segment['end_time']}"
        )
        if segment["overrides"]:
            lines.append(
                f"       overrides from step {segment['start_step']}: {segment['overrides']}"
            )
        if segment["rewound_from"] is not None:
            lines.append(f"       rewound from step {segment['rewound_from']}")
    return lines
