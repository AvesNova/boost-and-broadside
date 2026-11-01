"""Command-line entry point: `bnb train ...` trains, `bnb runs ...` lists runs and uploads
checkpoints, `bnb ship-card ...` charts a ship card. The one
place that sets process-wide options (the seed, the matmul precision, the charts' backend)."""

import dataclasses
import json
import os
import sys
from typing import Annotated

import tyro


@dataclasses.dataclass(frozen=True)
class Train:
    """Train a model with PPO (learner spec) in a run directory (runs spec): Yemong, or the
    toy relational model.

    preset: a training preset (deathmatch, deathmatch_wide, frontline), which sets the mode,
    the sizes and the spawn groups, or mixed, the default training mix of all three in one
    batch; without one, the mode and sizes below.
    precision: float32, tf32 or bfloat16 (the model's matmuls); old_log_probabilities:
    rollout or recomputed (by unroll before the first epoch).
    updates: the run's total update budget (a resume continues to it).
    set: TrainConfig fields by dotted path, such as learner.entropy=-0.01 (JSON values).
    resume: continue the run named, from its newest checkpoint or the one at --at, with
    every setting from the checkpoint; only the budget, --set of the learning rate, the
    warm-up and the loss coefficients, and the checkpoint cadence may change.
    fork (--from): a new run with these settings, starting from the named run's checkpoint
    (newest, or at --at): its model, optimizer, return scale and league slot.
    init_from: a warm start from a policy file: its weights and return scale, a fresh
    optimizer (the BC -> RL hand-off).
    print_config: print the resolved configuration and exit; probe: run the memory probe
    (the setup and one update), print the peaks and exit, writing no run.
    name: the run's name (default: date, preset or mode, and two words).
    checkpoint_every: environment steps between training checkpoints (default: about 30
    minutes at the run's measured throughput).
    """

    preset: str | None = None
    mode: str = "team_deathmatch"
    ships_per_team: int = 5
    num_fields: int = 10
    arena_radius: float = 2600.0
    num_envs: int = 1024
    updates: int = 100
    shards: int = 1
    model: str = "yemong"
    precision: str = "float32"
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
    runs_directory: str = "runs"
    checkpoint_every: int | None = None


# What a resume may be given besides --resume: the rest comes from the checkpoint.
RESUME_OPTIONS = ("resume", "at", "updates", "set", "checkpoint_every", "device",
                  "runs_directory", "print_config")


@dataclasses.dataclass(frozen=True)
class Runs:
    """List the runs (`bnb runs`), show one (`bnb runs show RUN`), or upload a chosen
    checkpoint (`bnb runs upload RUN [--at STEP] [--league]`) as a W&B artifact.

    action: list, show or upload; run: the run's name (show and upload).
    at: the checkpoint's step (default: the newest); league: upload the league's snapshots
    instead; to: the target (wandb only, for now).
    """

    action: tyro.conf.Positional[str] = "list"
    run: tyro.conf.Positional[str | None] = None
    at: int | None = None
    league: bool = False
    to: str = "wandb"
    runs_directory: str = "runs"


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
    names = {argument[2:].partition("=")[0].replace("-", "_") for argument in arguments
             if argument.startswith("--")}
    return {"fork" if name == "from" else name for name in names}


def ship_card(args: ShipCard) -> None:
    """`bnb ship-card`: print the card's chart at 30 Hz and write its charts to args.out;
    exits with the reason when matplotlib is missing."""
    from bnb.contracts import GameConfig
    from bnb.game.api import rules_of
    from bnb.game.handling import chart

    rules = rules_of(GameConfig(1, 1, 1000.0))  # flight and weapons need no arena
    types = chart.load_card(args.card)
    card = chart.report(rules, types)
    print("\n".join(chart.lines(rules, types, card)))
    try:
        paths = chart.write_charts(rules, types, card, args.out)
    except ImportError as error:
        raise SystemExit(str(error)) from error
    for path in paths:
        print(f"wrote {path}")


def _requested(args: Train):
    """The TrainConfig the options describe, `--set` applied."""
    from bnb.contracts import GameConfig
    from bnb.contracts.pools import environment_multiple
    from bnb.game.modes.rules import PRESETS
    from bnb.learn.ppo import LearnerConfig
    from bnb.runtime.train import LINEUP, TrainConfig, mixed

    pools = LINEUP
    if args.preset == "mixed":
        game, pools = mixed()
    elif args.preset is not None:
        if args.preset not in PRESETS:
            raise SystemExit(f"unknown preset {args.preset!r}; one of "
                             f"{sorted(PRESETS) + ['mixed']}")
        game = PRESETS[args.preset]
    else:
        game = GameConfig(args.ships_per_team, args.num_fields, args.arena_radius,
                          mode=args.mode)
    multiple = environment_multiple(pools)
    if args.num_envs % multiple:
        raise SystemExit(f"--num-envs {args.num_envs} does not split into the pools' shares: "
                         f"use a multiple of {multiple}")
    learner = LearnerConfig(old_log_probabilities=args.old_log_probabilities,
                            micro_batch_rows=args.micro_batch_rows, seed=args.seed)
    cfg = TrainConfig(game=game, pools=pools, num_envs=args.num_envs, updates=args.updates,
                      shards=args.shards, model=args.model, precision=args.precision,
                      learner=learner, seed=args.seed)
    return with_settings(cfg, args.set)


def parse_setting(setting: str) -> tuple[str, object]:
    """`path=value` as (path, the value as JSON, else the string)."""
    path, separator, text = setting.partition("=")
    if not separator:
        raise SystemExit(f"--set {setting!r}: expected path=value")
    try:
        return path, json.loads(text)
    except json.JSONDecodeError:
        return path, text


def with_settings(cfg, settings: tuple[str, ...]):
    """`cfg` with each `path=value` of `settings` set (the value as JSON, else a string)."""
    from bnb.runs.serialise import SerialiseError, from_json, to_json

    data = to_json(cfg)
    for setting in settings:
        path, value = parse_setting(setting)
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
        return from_json(type(cfg), data)[0]
    except (SerialiseError, ValueError) as error:
        raise SystemExit(f"--set: {error}") from error


def train(args: Train, options: set[str]) -> None:
    """`bnb train`: a new run, a resume, a fork or a warm start, logging a line per update;
    exits with the reason for a setting that does not fit."""
    import pathlib

    import torch

    from bnb.runs import checkpoint, probe, session
    from bnb.runs.checkpoint import OVERRIDABLE, CheckpointError
    from bnb.runs.records import Run, run_name
    from bnb.runs.serialise import to_json
    from bnb.runtime.train import Trainer, set_precision

    runs_directory = pathlib.Path(args.runs_directory)
    device = torch.device(args.device)
    starts = [name for name in ("resume", "fork", "init_from") if getattr(args, name)]
    if len(starts) > 1:
        raise SystemExit(f"choose one of --resume, --from and --init-from, not {starts}")
    if args.at is not None and not (args.resume or args.fork):
        raise SystemExit("--at chooses a checkpoint of --resume or --from")
    try:
        if args.resume:
            run = Run.open(runs_directory, args.resume)
            source = run.checkpoint_at(args.at)
            fixed = sorted(options - set(RESUME_OPTIONS))
            if fixed:
                flags = ", ".join("--from" if name == "fork" else "--" + name.replace("_", "-")
                                  for name in fixed)
                raise SystemExit(f"a resume takes {flags} "
                                 f"from the checkpoint and cannot change it: start a new run "
                                 f"from it with --from {args.resume}")
            outside = [setting.partition("=")[0] for setting in args.set
                       if setting.partition("=")[0] not in OVERRIDABLE]
            if outside:
                raise SystemExit(f"a resume cannot --set {', '.join(outside)}: it may change "
                                 f"only {', '.join(OVERRIDABLE)}; to change more, start a new "
                                 f"run with --from {args.resume}")
            saved, _ = checkpoint.training_config(source)
            cfg = saved
            if "updates" in options:
                cfg = dataclasses.replace(cfg, updates=args.updates)
            cfg = with_settings(cfg, args.set)
        else:
            cfg = _requested(args)
        if args.print_config:
            print(json.dumps(to_json(cfg), indent=1))
            return
        torch.manual_seed(cfg.seed)
        set_precision(cfg.precision)
        if args.probe:
            trainer = Trainer(cfg, device)
            probe.begin(device)
            trainer.update()
            peaks = probe.measure(device)
            print(probe.describe(peaks))
            probe.check(peaks, f"{cfg.num_envs} environments, "
                               f"{cfg.learner.micro_batch_rows} micro-batch rows")
            return
        if args.resume:
            trainer = checkpoint.load_training(source, device, cfg)
            checkpoint.restore_default_generators(source, device)
            step = trainer.ppo.env_steps
            rewound = run.set_aside_after(step)
            overrides = {name: getattr(args, name) for name in ("updates", "checkpoint_every")
                         if name in options}
            overrides.update(parse_setting(setting) for setting in args.set)
            if args.checkpoint_every is not None:
                run.update(checkpoint_every=args.checkpoint_every)
            warning = run.start_segment(step, overrides, rewound_from=max(rewound, default=None))
            if warning:
                print(f"warning: {warning}", flush=True)
        else:
            label = args.preset or cfg.game.mode
            name = args.name or run_name(label, runs_directory)
            parent = None
            if args.fork:
                parent_run = Run.open(runs_directory, args.fork)
                source = parent_run.checkpoint_at(args.at)
                trainer = checkpoint.fork_training(source, cfg, device)
                parent = dict(run=args.fork, step=trainer.ppo.env_steps)
            elif args.init_from:
                trainer = checkpoint.warm_start(args.init_from, cfg, device)
                parent = dict(policy=str(pathlib.Path(args.init_from).resolve()))
            else:
                trainer = Trainer(cfg, device)
            run = Run.create(runs_directory, name, cfg, sys.argv, args.checkpoint_every,
                             parent)
            run.start_segment(trainer.ppo.env_steps)
            print(f"run {name} in {run.directory}", flush=True)
    except (CheckpointError, FileNotFoundError, FileExistsError) as error:
        raise SystemExit(str(error)) from error

    def log(row: dict) -> None:
        rewards = " ".join(f"{name} {sum(pool['reward_per_s'].values()):+.3f}"
                           for name, pool in row["pools"].items())
        print(f"update {row['update']:4d}  rollout {row['rollout_s'] * 1e3:6.0f} ms  "
              f"learn {row['learn_s'] * 1e3:6.0f} ms  entropy {row['loss/entropy']:.2f}  "
              f"kl {row['approximate_kl']:+.4f}  clip {row['clip_fraction']:.3f}  "
              f"reward/s: {rewards}", flush=True)

    status = session.train(run, trainer, log)
    print(f"run {run.record['name']} {status} at step {trainer.ppo.env_steps}")


def runs(args: Runs) -> None:
    """`bnb runs`: list the runs, show one, or upload a checkpoint."""
    import pathlib

    from bnb.runs.records import Run, list_runs

    runs_directory = pathlib.Path(args.runs_directory)
    if args.action == "list":
        print(f"{'run':44s} {'status':16s} {'newest step':>12s}  {'started':25s} commit")
        for run in list_runs(runs_directory):
            record = run.record
            segments = record["segments"]
            commit = (segments[-1]["commit"] or "")[:10] if segments else ""
            started = segments[0]["start_time"] if segments else ""
            newest = record["newest_checkpoint"]
            print(f"{record['name']:44s} {run.shown_status():16s} "
                  f"{'' if newest is None else newest:>12}  {started:25s} {commit}")
        return
    if args.run is None:
        raise SystemExit(f"bnb runs {args.action} needs a run name")
    try:
        run = Run.open(runs_directory, args.run)
        if args.action == "show":
            print("\n".join(show(run)))
        elif args.action == "upload":
            from bnb.runs import upload

            if args.to != "wandb":
                raise SystemExit(f"--to {args.to!r}: the only target is wandb for now")
            print(upload.upload(run, args.at, args.league))
        else:
            raise SystemExit(f"bnb runs {args.action!r}: the actions are list, show, upload")
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error


def show(run) -> list[str]:
    """`bnb runs show`: the record, readable."""
    record = run.record
    lines = [f"run {record['name']}: {run.shown_status()}"
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
             "  segments:"]
    for index, segment in enumerate(record["segments"]):
        dirty = " (dirty)" if segment["dirty"] else ""
        lines.append(f"    {index}: steps {segment['start_step']} to {segment['end_step']} at "
                     f"{segment['commit']}{dirty}, {segment['start_time']} to "
                     f"{segment['end_time']}")
        if segment["overrides"]:
            lines.append(f"       overrides from step {segment['start_step']}: "
                         f"{segment['overrides']}")
        if segment["rewound_from"] is not None:
            lines.append(f"       rewound from step {segment['rewound_from']}")
    return lines
