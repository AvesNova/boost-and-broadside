"""The `bnb` command: `bnb train` trains or clones, `bnb runs` lists runs and uploads
checkpoints, `bnb ship-card` charts a ship card, `bnb ratings` and `bnb head-to-head` rate
policies, `bnb figures` draws what a policy's heads have learned and films it playing.

An entry point (code style, Layers): it sets the process-wide options its commands need (the
seed, the matmul precision, the charts' backend). The play shell (`python -m bnb.play`) and
`bench` are the other entry points."""

import dataclasses
import functools
import json
import os
import pathlib
import subprocess
import sys
from typing import Annotated

import tyro

from bnb.contracts import REWARD_LEVELS
from bnb.contracts.config import TEAM_DEATHMATCH
from bnb.figures.video import VideoConfig


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
    set: TrainConfig fields by dotted path, such as learner.entropy_shoot=-0.01 or
    yemong.width=256 (JSON values); fields inside tuples (such as pools) cannot be set.
    resume: continue the run named (PPO or BC), from its newest checkpoint or the one at
    --at, with every setting from the checkpoint; only the budget (--updates, or
    --bc-budget-steps), the precision, the micro-batch rows, the checkpoint cadence and
    --set of OVERRIDABLE fields (the learning rate, the warm-up, the loss coefficients, the
    reward weights) may change.
    fork (--from): a new run with these settings, starting from the named run's checkpoint
    (newest, or at --at): its model, optimizer, return scale, league slot and environment
    steps; --updates counts from the fork.
    init_from: a warm start from a policy file: its weights and return scale, a fresh
    optimizer (the BC -> RL hand-off). A fork and a warm start take the model's settings
    from the file, unless --set gives one, when the file must match them.
    print_config: print the resolved configuration and exit; probe: run the memory probe
    (the setup and one update), print the peaks and exit, writing no run.
    name: the run's name (default: date, preset or mode, and two words).
    runs_directory: where runs live (default: runs/ at the repository root, or in the
    current directory outside a repository).
    checkpoint_every: environment steps between training checkpoints (default: about 30
    minutes at the run's measured throughput).
    update_policies: write the live, exponential-average and uniform-average policy files
    after every update to the run's `policies/` (about 10 MB each, never pruned; default: on
    for ppo, off for bc; `--update-policies False` turns it off; a resume may change it).
    objective: ppo, or bc (behaviour cloning against self-play and the scripted agent until
    a stop rule fires, then the clone is saved as the run's policy file; updates is unused;
    a BC run keeps training checkpoints and resumes, but is not forked or warm-started);
    teacher_temperature and teacher_floor set the teacher (the scripted agent at its own
    temperature, its labels mixed with uniform by the floor; T = 1 and floor 0 by
    default); bc_budget_steps: BC's budget in environment steps
    (TrainConfig.stop.budget_steps).
    wandb: stream each update's record to W&B (project `boost-and-broadside-v2`, the run's
    name) as well as to the run's metrics.jsonl; without it nothing leaves the machine.
    calibrate_entropy: print the scripted agent's natural entropy over its own games of the
    preset and the temperature whose labels reach this mean entropy (nats per living ship
    step) at teacher_floor, then exit, writing no run; calibration_segments: how many
    rollout segments of 128 ticks it records.
    """

    preset: str | None = None
    mode: str = TEAM_DEATHMATCH
    ships_per_team: int = 5
    num_fields: int = 10
    arena_radius: float = 2600.0
    environments: int = 2048
    updates: int = 100
    shards: int = 4
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
    update_policies: bool | None = None
    objective: str = "ppo"
    teacher_temperature: float = 1.0
    teacher_floor: float = 0.0
    bc_budget_steps: int = 50_000_000
    calibrate_entropy: float | None = None
    calibration_segments: int = 4
    wandb: bool = False


# What a resume may be given besides --resume: the rest comes from the checkpoint.
RESUME_OPTIONS = (
    "resume",
    "at",
    "updates",
    "precision",
    "micro_batch_rows",
    "set",
    "checkpoint_every",
    "update_policies",
    "device",
    "runs_directory",
    "print_config",
    "wandb",
    "bc_budget_steps",
)
# The resume options that set a TrainConfig field: their paths.
RESUME_FIELDS = {
    "updates": "updates",
    "bc_budget_steps": "stop.budget_steps",
    "precision": "precision",
    "micro_batch_rows": "learner.micro_batch_rows",
}


@dataclasses.dataclass(frozen=True)
class Runs:
    """List the runs (`bnb runs`), show one (`bnb runs show RUN`), upload a chosen
    checkpoint (`bnb runs upload RUN [--at STEP] [--league]`) as a W&B artifact, or build the
    run's W&B dashboards (`bnb runs dashboard RUN`, the `tools` extra).

    Or build an exponential moving average of any decay from the run's saved per-update live
    policies (`bnb runs ema RUN --decay 0.9 [--at STEP] [--out DIR]`, `bnb.runs.ema`) and
    write it as a policy file for the tournament; the decay is per update.

    action: list, show, upload, dashboard or ema; run: the run's name (all but list).
    at: the checkpoint's step (default: the newest; for ema, the update the average ends at);
    decay and out: ema's per-update decay and where to write the policy file (default under
    the run's artifacts/ema/); league: upload the league's snapshots
    instead; to: the target (wandb only, for now); runs_directory: as for bnb train.
    """

    action: tyro.conf.Positional[str] = "list"
    run: tyro.conf.Positional[str | None] = None
    at: int | None = None
    league: bool = False
    decay: float | None = None
    out: str | None = None
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


@dataclasses.dataclass(frozen=True)
class Ratings:
    """Published ratings (`bnb ratings RUN`): an offline tournament over the run's snapshots
    and averages (its `league/` directory), the stationary ladder and the players `--with`
    names (policy files, runs' newest checkpoints, ladder entries), game preset by preset,
    until every standard error is under `--target-stderr` or the budget is spent. The
    batches are the match engine's fixed blocks (`--blocks` of `--block-environments`
    games). The leaderboard goes to the run's `artifacts/ratings/` (or `--out`'s).

    A Yemong policy may play ablated: `PATH:rotary`, `PATH:forecast` or
    `PATH:rotary+forecast` (`bnb.league.matches.ablations_of`), the same weights with the
    rotary rotation as the identity or the forecast not fed back, on its own rows only.

    run: the run; with_: more players; presets: the game presets; target_stderr: rating
    points; budget: games per preset at most; blocks and block_environments: the engine's
    layout; device; runs_directory: as for bnb train. ladder: the stationary ladder's rungs
    below the scripted agent (the scripted agent always plays: it anchors the scale at
    1000). league: the run's `league/` policies (else only `--with`'s players and the
    ladder). scripted_plays: the only players the scripted agent plays (by the names given;
    all by default). out: the directory whose `artifacts/ratings/` the result goes to.
    Each preset's batches are recorded as they end in `<preset>.batches.jsonl`, in `--out`
    (else the run's `artifacts/ratings/`), and a tournament that finds its record goes on
    from it (its field must be the same; move it aside to start over).

    reuse: directories of earlier tournaments' records (every `<preset>.batches.jsonl` below
    each, as `--out` writes them): each recorded game whose two players are both in this
    field starts the fit, and batches are played only while an error is above the target
    (`bnb.league.tournament.reuse_record`; the layout does not matter). Players match by
    content: a policy by the SHA-256 of its weights in its checkpoint.json, a scripted
    player by its name. A record written before identities were recorded (a legacy one)
    trusts its paths: a name is resolved as it would be now, and is left out (and
    reported) when it no longer resolves or its policy's checkpoint.json was written after
    the record's first batch ended (a rewind writes a new policy at the old path).
    """

    run: tyro.conf.Positional[str]
    with_: Annotated[tuple[str, ...], tyro.conf.arg(name="with")] = ()
    presets: tuple[str, ...] = ("frontline", "deathmatch", "deathmatch_wide")
    target_stderr: float = 25.0
    budget: int = 20_000
    blocks: int = 8
    block_environments: int = 128
    device: str = "cuda"
    runs_directory: str | None = None
    ladder: bool = True
    league: bool = True
    scripted_plays: tuple[str, ...] | None = None
    out: str | None = None
    reuse: tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class HeadToHead:
    """Head-to-head (`bnb head-to-head A B [C D ...] --games N`): only the named pairs (A
    against B, C against D, ...), each over `--games` games per preset split evenly between
    the seats. A player is a policy file, a run (its newest checkpoint), "scripted",
    "random", "rung_T<t>" (the scripted agent at temperature t), or a teacher,
    "scripted@T=<temperature>,floor=<floor>" (either key optional; T defaults to 1 and
    floor to 0: the scripted agent at its own temperature T, then a uniform joint action
    with probability floor, drawn as BC's teacher opponent does, so "scripted" is BC's
    default teacher and "scripted@T=2" plays as "rung_T2"), or a Yemong policy ablated,
    "PATH:rotary", "PATH:forecast" or "PATH:rotary+forecast". Prints and records, per pair,
    the wins, losses and draws, the score with its standard error and the split by preset;
    the result goes to the first named run's `artifacts/head_to_head/` (else the runs
    directory's).

    players: the pairs; games: games per pair and preset (even); presets; device;
    runs_directory: as for bnb train.
    """

    players: tyro.conf.Positional[tuple[str, ...]]
    games: int = 64
    presets: tuple[str, ...] = ("frontline", "deathmatch", "deathmatch_wide")
    device: str = "cuda"
    runs_directory: str | None = None


@dataclasses.dataclass(frozen=True)
class Figures:
    """Figures of what a policy's heads have learned (`bnb figures SET ... --policy P`):
    capture a self-play rollout and the scenario lab with the policy's model (on the CPU by
    default), store it as one artifact under the output's `captures/`, and draw the figure
    sets from it; or draw them from a stored capture (`--capture FILE`), without the model.
    The sets (`bnb.figures.sets.FIGURE_SETS`, or all) and how to read them:
    docs/guides/figures.md. `bnb figures capture --policy P` only captures. Each PNG
    carries its provenance in its text chunks. Needs the tools extra (matplotlib).
    `bnb figures video` (FiguresVideo) and `bnb figures watch` (FiguresWatch) have their
    own options.

    sets: figure set names, all, or capture; policy: a policy directory or a run's name
    (its newest checkpoint); capture: a stored capture to draw from; out: the output
    directory (default: the run's artifacts/figures/<policy directory>, else beside the
    capture, else figures/); preset, environments, ticks, seed and spread_samples: the
    capture, and game_segments (more segments of the same games, for the critic set:
    `bnb.figures.capture.CaptureConfig`); device; runs_directory: as for bnb train.
    """

    sets: tyro.conf.Positional[tuple[str, ...]] = ("all",)
    policy: str | None = None
    capture: str | None = None
    out: str | None = None
    preset: str = "deathmatch"
    environments: int = 4
    ticks: int = 256
    seed: int = 0
    spread_samples: int = 4
    game_segments: int = 0
    device: str = "cpu"
    runs_directory: str | None = None


@dataclasses.dataclass(frozen=True)
class FiguresVideo:
    """A video of one game (`bnb figures video --policy P`): the policy (team 0) against
    the scripted agent in the preset's mode, drawn by the play shell offscreen (EGL) and
    encoded to MP4 by ffmpeg (`bnb.figures.video`). docs/guides/figures.md.

    policy: a policy directory or a run's name (its newest checkpoint); preset: the game
    (frontline, deathmatch, deathmatch_wide); out: the MP4 file (default: the run's
    artifacts/figures/<policy directory>/videos/<preset>_seed<seed>.mp4, else
    figures/videos/); video: the video's own settings (`--video.seconds` and so on:
    `bnb.figures.video.VideoConfig`); device: where the agent runs (cpu hides the GPU from
    torch; the drawing uses the GPU's OpenGL either way); threads: torch's CPU threads;
    runs_directory: as for bnb train.
    """

    policy: str
    preset: str = "frontline"
    out: str | None = None
    video: VideoConfig = VideoConfig()
    device: str = "cpu"
    threads: int = 4
    runs_directory: str | None = None


@dataclasses.dataclass(frozen=True)
class FiguresWatch:
    """Media of a training run while it trains (`bnb figures watch RUN`): a second process
    that only reads the run, polls its checkpoints and, for each new one, captures each
    mode on the CPU, draws the quick figure sets and films one game per mode against the
    scripted agent, into the run's artifacts/figures/<checkpoint>/ (`bnb.figures.watch`).
    With --wandb it logs them at the checkpoint's environment steps to the companion W&B
    run `<run>-media`, never to the training run's own. It stops once the run has stopped
    and every checkpoint has its media. docs/guides/figures.md.

    run: the run's name (or its directory); sets and modes: the figure sets and the modes
    (presets) made at each checkpoint; environments, ticks and seed: each capture's;
    video: the videos' settings (`--video.seconds` and so on:
    `bnb.figures.video.VideoConfig`); interval: seconds between polls; keep_captures: keep
    each capture artifact (about 60 MB a mode); once: one pass, then stop; wandb: log to
    the companion run; threads: torch's CPU threads; nice: the process's niceness
    increment, so the trainer's CPU work goes first; runs_directory: as for bnb train.
    """

    run: tyro.conf.Positional[str]
    sets: tuple[str, ...] = ("shots", "belief", "next_state", "density", "enemy_action")
    modes: tuple[str, ...] = ("frontline", "deathmatch")
    environments: int = 4
    ticks: int = 256
    seed: int = 0
    video: VideoConfig = VideoConfig()
    interval: float = 60.0
    keep_captures: bool = False
    once: bool = False
    wandb: bool = False
    threads: int = 4
    nice: int = 10
    runs_directory: str | None = None


@dataclasses.dataclass(frozen=True)
class QueueOptions:
    """The GPU job queue (`bnb queue ACTION`, `bnb.runs.queue`): one job at a time, in order.

    `bnb queue add [--after JOB] [--background] KIND ARGUMENTS...` queues a job: KIND train
    (ARGUMENTS are `bnb train`'s: a new run; give it --name to rate or resume it by name),
    resume (a run's name, then `bnb train --resume`'s options, such as a larger --updates
    or --bc-budget-steps) or ratings (`bnb ratings`' arguments). --after: wait until that
    job has finished; --background: the champion's continuation, run while nothing else is
    ready and stopped gracefully (then queued again as a resume) when something is.
    `bnb queue list` shows every job's state, start, end and ETA; `bnb queue remove JOB`
    removes a job (a running one after a graceful stop); `bnb queue log JOB` prints a job's
    log's last --lines lines. `bnb queue run` is the runner, for a systemd user unit:
    `systemd-run --user --unit=bnb-queue -p KillMode=mixed -p TimeoutStopSec=1h
    --working-directory=$PWD .venv/bin/bnb queue run` (docs/guides/queueing-jobs.md).

    action: add, list, remove, log or run; job: the job's number (remove, log); kind and
    arguments: the job (add); after and background: as above; queue_file: the queue
    (default ~/.local/share/bnb/queue.json; the logs go to logs/queue/ beside it); program:
    the command that runs bnb (default this environment's bnb); lines: the log lines
    printed; lock: the bench lock's wrapper the runner runs each job under ("none": no
    lock); poll: the runner's seconds between looks at the queue; until_idle: the runner
    exits once nothing is running or ready.
    """

    action: str
    job: int | None = None
    kind: str | None = None
    arguments: tuple[str, ...] = ()
    after: int | None = None
    background: bool = False
    queue_file: str | None = None
    program: str | None = None
    lines: int = 40
    lock: str | None = None
    poll: float = 2.0
    until_idle: bool = False


# `bnb queue`'s own options, read anywhere on its command line (the rest is the job's), with
# whether each takes a value
QUEUE_FLAGS = {
    "--after": True,
    "--background": False,
    "--queue-file": True,
    "--program": True,
    "--lines": True,
    "--lock": True,
    "--poll": True,
    "--until-idle": False,
}

COMMANDS = ("train", "play", "runs", "ship-card", "ratings", "head-to-head", "figures", "queue")


def main() -> None:
    """The `bnb` command: `bnb train ...` trains, `bnb play ...` opens the play shell,
    `bnb runs ...` lists and shows runs and uploads checkpoints, `bnb ship-card ...` charts a
    card, `bnb ratings ...` and
    `bnb head-to-head ...` rate policies offline, `bnb figures ...` draws what a policy's
    heads have learned (`bnb figures video ...` films a game, `bnb figures watch RUN` makes
    both while a run trains), `bnb queue ...` queues and runs GPU jobs."""
    os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")  # Triton 3.9's default, made explicit
    command, arguments = (sys.argv[1:2] or [""])[0], sys.argv[2:]
    if command == "ship-card":
        os.environ.setdefault("MPLBACKEND", "Agg")  # the charts go to files: no display
        ship_card(tyro.cli(ShipCard, args=arguments, prog="bnb ship-card"))
    elif command == "train":
        train(tyro.cli(Train, args=arguments, prog="bnb train"), given(arguments))
    elif command == "play":
        from bnb.play.__main__ import main as play

        play(arguments)
    elif command == "runs":
        runs(tyro.cli(Runs, args=arguments, prog="bnb runs"))
    elif command == "queue":
        queue(queue_options(arguments))
    elif command == "ratings":
        ratings(tyro.cli(Ratings, args=arguments, prog="bnb ratings"))
    elif command == "head-to-head":
        head_to_head(tyro.cli(HeadToHead, args=arguments, prog="bnb head-to-head"))
    elif command == "figures":
        os.environ.setdefault("MPLBACKEND", "Agg")  # the figures go to files: no display
        kind = (arguments[:1] or [""])[0]
        if kind == "video":
            figures_video(tyro.cli(FiguresVideo, args=arguments[1:], prog="bnb figures video"))
        elif kind == "watch":
            figures_watch(tyro.cli(FiguresWatch, args=arguments[1:], prog="bnb figures watch"))
        else:
            figures(tyro.cli(Figures, args=arguments, prog="bnb figures"))
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
    from bnb.runtime.lanes import LaneConfig
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
            lanes=cloning.LANES if cloned else LaneConfig(),
            model=options.model,
            precision=options.precision,
            learner=learner,
            seed=options.seed,
            objective=options.objective,
            teacher=bc.TeacherConfig(options.teacher_temperature, options.teacher_floor),
            stop=bc.StopConfig(budget_steps=options.bc_budget_steps),
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


def warm_start_entropy(config, settings: tuple[str, ...]):
    """`config` with the warm start's entropy coefficient (`WARM_START_ENTROPY`) on every
    action factor that `--set learner.entropy_<factor>=...` did not choose."""
    from bnb.learn.ppo import ENTROPY_TERMS, WARM_START_ENTROPY

    chosen = {path for path, _ in map(parse_setting, settings)}
    learner = dataclasses.replace(
        config.learner,
        **{term: WARM_START_ENTROPY for term in ENTROPY_TERMS if f"learner.{term}" not in chosen},
    )
    return dataclasses.replace(config, learner=learner)


def set_precision(precision: str) -> None:
    """The process-wide matmul precision of a run: TF32 for "tf32", full float32 otherwise
    (bf16 is the model's own autocast)."""
    import torch

    torch.set_float32_matmul_precision("high" if precision == "tf32" else "highest")


def train(options: Train, given_options: set[str]) -> None:
    """`bnb train`: a new run, a resume, a fork or a warm start, logging a line per update;
    exits with the reason for a setting that does not fit. SIGINT, SIGTERM or the run's
    STOP file stops it after the update in progress, with a checkpoint
    (`bnb.runs.session.StopRequest`), from the start: a request during the setup stops the
    run after no update."""
    from bnb.runs.session import StopRequest

    with StopRequest() as stop:
        _train(options, given_options, stop)


def _train(options: Train, given_options: set[str], stop) -> None:
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
    if starts and starts[0] != "resume" and options.objective == "bc":
        flag = {"fork": "--from", "init_from": "--init-from"}[starts[0]]
        raise SystemExit(
            f"a BC run starts fresh or resumes: --objective bc takes no {flag} "
            f"(--resume RUN continues a BC run)"
        )
    if options.at is not None and not (options.resume or options.fork):
        raise SystemExit("--at chooses a checkpoint of --resume or --from")
    try:
        if options.resume:
            run = Run.open(runs_directory, options.resume)
            if run.live():
                raise SystemExit(
                    f"run {options.resume} is running as process "
                    f"{run.record['pid']} on this host; stop it (Ctrl-C, SIGTERM or a "
                    f"STOP file in its directory) before resuming it"
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
            if options.fork:
                source = Run.open(runs_directory, options.fork).checkpoint_at(options.at)
            elif options.init_from:
                source = pathlib.Path(options.init_from)
                config = warm_start_entropy(config, options.set)
            model_set = any(
                path.split(".")[0] == config.model for path, _ in map(parse_setting, options.set)
            )
            if (options.fork or options.init_from) and not model_set:
                config = checkpoint.with_model_of(config, source)
        if options.print_config:
            print(json.dumps(to_json(config), indent=1))
            return
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
            if options.update_policies is not None:
                run.update(update_policies=options.update_policies)
            warning = run.start_segment(step, overrides, rewound_from=max(rewound, default=None))
            if warning:
                print(f"warning: {warning}", flush=True)
        else:
            label = options.preset or config.game.mode
            name = options.name or run_name(label, runs_directory)
            parent = None
            if options.fork:
                trainer = checkpoint.fork_training(source, config, device)
                parent = dict(run=options.fork, step=trainer.ppo.environment_steps)
            elif options.init_from:
                trainer = checkpoint.warm_start(source, config, device)
                parent = dict(policy=str(pathlib.Path(options.init_from).resolve()))
            else:
                trainer = Trainer(config, device)
            keep_policies = options.update_policies
            if keep_policies is None:
                keep_policies = config.objective != "bc"
            run = Run.create(
                runs_directory,
                name,
                config,
                sys.argv,
                options.checkpoint_every,
                parent,
                keep_policies,
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
        totals = {
            pool: sum(row.get(f"reward/rate/{level}/{pool}", 0.0) for level in REWARD_LEVELS)
            for pool in trainer.pool_names[: len(trainer.config.pools)]
        }
        rewards = " ".join(f"{pool} {total:+.3f}" for pool, total in totals.items())
        print(
            f"update {row['counters/updates']:4d}  "
            f"rollout {row['train/time/rollout_seconds'] * 1e3:6.0f} ms  "
            f"learn {row['train/time/learner_seconds'] * 1e3:6.0f} ms  "
            f"entropy {row.get('policy/entropy/joint', float('nan')):.2f}  "
            f"kl {row['policy/kl']:+.4f}  clip {row['policy/clip_fraction']:.3f}  "
            f"drift {row['policy/ratio_drift/max']:.3f}  skipped {row['train/skipped_steps']}  "
            f"nonfinite beliefs {row['counters/nonfinite_beliefs']}  reward/s: {rewards}",
            flush=True,
        )

    state = checkpoint.league_state(source) if options.resume else None
    if config.objective == "ppo" and trainer.opponents is not None:
        from bnb.league.league import League

        # a warm start's live rating starts from its parent's (the clone's lane ratings)
        ratings = checkpoint.read_policy(source).ratings if options.init_from else None
        League(trainer, run.directory / "league", run, state, ratings)
    elif trainer.lanes or config.objective == "bc":  # BC's hooks keep its averages
        from bnb.league.lanes import LaneHooks

        LaneHooks(trainer, state)

    from bnb.runs.recording import Recorder
    from bnb.runs.sinks import Sinks
    from bnb.runs.triggers import Triggers

    sinks = Sinks(run, to_json(config), wandb=options.wandb)
    rules = Triggers(
        state=checkpoint.triggers_state(source) if options.resume else None,
        objective=config.objective,
    )
    recorder = Recorder(run, trainer, sinks, rules)
    cloned = config.objective == "bc"
    try:
        status = session.train(
            run, trainer, log_cloning if cloned else log, recorder=recorder, stop=stop
        )
        if cloned:
            print(
                f"stopped by {run.record['stopped_by']}; the policy is {run.policy_path} "
                f"(the {run.record['policy_player']})",
                flush=True,
            )
    except MemoryBudgetError as error:  # the run is marked failed with the same message
        raise SystemExit(str(error)) from error
    finally:
        sinks.close()
    print(f"run {run.record['name']} {status} at step {trainer.ppo.environment_steps}")


def log_cloning(row: dict) -> None:
    """Print one behaviour-cloning update's row on one line: the timings (ms), the BC KL and the
    teacher's and student's entropies (nats), each score, and the score stop's window once one
    is full (the average it reads)."""
    scores = " ".join(
        f"{name.split('/')[-1]} {value:.3f}"
        for name, value in row.items()
        if name.startswith("bc/score/")
    )
    # the score stop's window of lane games against the scripted agent, once one is full
    read = [name for name, value in row.items() if name.startswith("bc/stop/reads/") and value]
    window = (
        f"  window {read[0].split('/')[-1]} {row['bc/stop/window_score']:.3f} "
        f"(+- {row['bc/stop/window_stderr']:.3f})"
        if "bc/stop/window_score" in row and read
        else ""
    )
    print(
        f"update {row['counters/updates']:4d}  steps {row['counters/environment_steps']:>11,}  "
        f"rollout {row['train/time/rollout_seconds'] * 1e3:6.0f} ms  "
        f"learn {row['train/time/learner_seconds'] * 1e3:6.0f} ms  "
        f"bc kl {row['bc/kl']:.4f}  teacher {row['bc/entropy/target']:.3f}  "
        f"student {row['bc/entropy/student']:.3f}  scores: {scores}{window}",
        flush=True,
    )


def calibrate(options: Train, config, device) -> None:
    """`bnb train --calibrate-entropy`: the teacher's natural entropy over the scripted
    agent's own games of the configured game and pools, and the temperature for the target."""
    from bnb.learn import bc
    from bnb.runtime import cloning

    distributions = cloning.teacher_distributions(
        config.game,
        config.ship_types(device),
        config.pools,
        config.environments,
        options.calibration_segments,
        device,
    )
    tokens = distributions(1.0)
    natural = bc.mean_entropy(tokens)
    print(
        f"natural entropy {natural:.4f} nats per living ship-step over "
        f"{tokens.shape[0]} ship-steps",
        flush=True,
    )
    try:
        chosen = bc.calibrate(distributions, options.calibrate_entropy, options.teacher_floor)
    except ValueError as error:
        raise SystemExit(str(error)) from None
    print(
        f"temperature {chosen.temperature:.4f} at floor {chosen.floor} gives "
        f"{bc.mean_entropy(distributions(chosen.temperature), chosen.floor):.4f}"
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
        elif options.action == "dashboard":
            print(dashboard(run))
        elif options.action == "ema":
            from bnb.runs import ema

            if options.decay is None:
                raise SystemExit("bnb runs ema needs --decay")
            try:
                print(ema.build(run, options.decay, options.at, options.out))
            except ValueError as error:
                raise SystemExit(str(error)) from error
        elif options.action == "upload":
            from bnb.runs import upload

            if options.to != "wandb":
                raise SystemExit(f"--to {options.to!r}: the only target is wandb for now")
            print(upload.upload(run, options.at, options.league))
        else:
            raise SystemExit(
                f"bnb runs {options.action!r}: the actions are list, show, upload, dashboard, ema"
            )
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error


def dashboard(run) -> str:
    """`bnb runs dashboard RUN`: build or update the run's headline and curated W&B views
    for its objective (`bnb.runtime.dashboards`; the `tools` extra), and their links."""
    import wandb

    from bnb.runs.upload import PROJECT
    from bnb.runtime import dashboards

    entity = os.environ.get("WANDB_ENTITY") or wandb.Api().default_entity
    config = json.loads((run.directory / "config.json").read_text())
    links = dashboards.save(run.record["name"], entity, PROJECT, config.get("objective", "ppo"))
    return "\n".join(f"{name}: {link}" for name, link in links.items())


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
        f"  policies    {'every update' if record.get('update_policies') else 'not per update'}",
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


def _players(names, runs_directory):
    from bnb.league.matches import player

    try:
        return [player(name, runs_directory) for name in names]
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error


def _check_presets(presets: tuple[str, ...]) -> None:
    from bnb.game.api import PRESETS

    unknown = [preset for preset in presets if preset not in PRESETS]
    if unknown:
        raise SystemExit(f"unknown presets {unknown}; choose from {sorted(PRESETS)}")


def ratings(options: Ratings) -> None:
    """`bnb ratings`: the offline tournament and its leaderboard."""
    import torch

    from bnb.league import tournament
    from bnb.league.matches import ArenaConfig, Player, player
    from bnb.league.pool import ladder_entries
    from bnb.runs import provenance
    from bnb.runs.records import Run
    from bnb.runtime.lanes import SCRIPTED_PLAYER
    from bnb.runtime.opponents import LADDER

    _check_presets(options.presets)
    runs_directory = runs_directory_of(options.runs_directory)
    try:
        run = Run.open(runs_directory, options.run)
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error
    league = sorted((run.directory / "league").glob("step_*/*")) if options.league else []
    field = [
        Player(f"{options.run}/{policy.parent.name}/{policy.name}", policy=policy)
        for policy in league
        if (policy / "checkpoint.json").exists()
    ]
    field += [
        Player(e.name, rung=e.rung)
        for e in ladder_entries(LADDER)
        if options.ladder or e.name == SCRIPTED_PLAYER
    ]
    field += [
        p for p in _players(options.with_, runs_directory) if p.name not in {f.name for f in field}
    ]
    try:
        allowed = tournament.allowed_pairs([p.name for p in field], options.scripted_plays)
        config = tournament.TournamentConfig(
            arena=ArenaConfig(blocks=options.blocks, block_environments=options.block_environments),
            target_stderr=options.target_stderr,
            budget=options.budget,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    device = torch.device(options.device)
    owner = run.directory if options.out is None else pathlib.Path(options.out)
    records_directory = run.directory / "artifacts" / "ratings" if options.out is None else owner
    missing = [name for name in options.reuse if not pathlib.Path(name).is_dir()]
    if missing:
        raise SystemExit(f"--reuse: {missing} are not directories")
    reused = {
        preset: tuple(
            sorted(
                {
                    record.resolve()
                    for name in options.reuse
                    for record in pathlib.Path(name).rglob(f"{preset}.batches.jsonl")
                }
            )
        )
        for preset in options.presets
    }
    rows = []
    try:
        for preset in options.presets:
            record = records_directory / f"{preset}.batches.jsonl"
            rows += tournament.rate(
                field,
                preset,
                device,
                config,
                allowed=allowed,
                record=record,
                reuse=reused[preset],
                resolve=lambda name: player(name, runs_directory),
            )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    for row in sorted(rows, key=lambda row: (row["map"], -row["calibrated_elo"])):
        print(
            f"{row['map']:16s} {row['player']:60s} {row['calibrated_elo']:8.1f} "
            f"+- {row['stderr']:5.1f}  {row['games']:6d} games"
        )
    inputs = {player.name: player.policy for player in field if player.policy is not None}
    payload = dict(
        leaderboard=rows,
        config=dataclasses.asdict(config),
        scripted_plays=options.scripted_plays,
        reused={preset: [str(record) for record in records] for preset, records in reused.items()},
    )
    print(provenance.write_result(owner, "ratings", payload, inputs))


def head_to_head(options: HeadToHead) -> None:
    """`bnb head-to-head`: the named pairs, and the result recorded."""
    import torch

    from bnb.league import tournament
    from bnb.runs import provenance, records

    _check_presets(options.presets)
    runs_directory = runs_directory_of(options.runs_directory)
    players = _players(options.players, runs_directory)
    try:
        rows = tournament.head_to_head(
            players, options.games, options.presets, torch.device(options.device)
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    for row in rows:
        print(
            f"{row['first']} against {row['second']}: {row['wins']} wins, {row['losses']} "
            f"losses, {row['draws']} draws; score {row['score']:.3f} +- {row['stderr']:.3f}"
        )
        for preset, part in row["presets"].items():
            print(
                f"  {preset:16s} {part['wins']}-{part['losses']}-{part['draws']}, "
                f"score {part['score']:.3f}"
            )
    # a run named by the players, never a policy directory given by its path
    owner = next(
        (
            runs_directory / name
            for name in options.players
            if (runs_directory / name / records.RECORD).is_file()
        ),
        runs_directory,
    )
    inputs = {player.name: player.policy for player in players if player.policy is not None}
    payload = dict(pairs=rows, games=options.games, presets=list(options.presets))
    print(provenance.write_result(owner, "head_to_head", payload, inputs))


def figures(options: Figures) -> None:
    """`bnb figures`: capture and draw (`bnb.figures.command.figures`)."""
    import torch

    from bnb.figures.command import figures as draw

    torch.manual_seed(options.seed)  # the capture's rollout and spread draws
    try:
        draw(options, runs_directory_of(options.runs_directory))
    except ImportError as error:
        raise SystemExit(f"{error}: bnb figures needs the tools extra (uv sync --extra tools)")


def hide_gpu() -> None:
    """Hide every CUDA device from torch (before anything initialises CUDA), so a process
    that shares the machine with a training run never takes GPU memory or time from it.
    OpenGL over EGL, which the play shell draws with, is not CUDA and still sees the GPU."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def figures_video(options: FiguresVideo) -> None:
    """`bnb figures video`: one game filmed (`bnb.figures.video.render`)."""
    if options.device == "cpu":
        hide_gpu()
    import torch

    from bnb.figures import video
    from bnb.figures.command import policy_path, run_of

    torch.set_num_threads(options.threads)
    policy = policy_path(options.policy, runs_directory_of(options.runs_directory))
    path = options.out
    if path is None:
        run = run_of(policy)
        directory = pathlib.Path("figures") if run is None else run / "artifacts" / "figures"
        directory = directory / (policy.name if run is not None else "") / "videos"
        path = directory / video.video_name(options.preset, options.video)
    numbers = video.render(policy, options.preset, options.video, path)
    print(f"wrote {path}")
    print(f"video: {numbers}")


def figures_watch(options: FiguresWatch) -> None:
    """`bnb figures watch`: media of a training run while it trains
    (`bnb.figures.watch.watch`), on the CPU at a lower priority."""
    hide_gpu()
    import torch

    from bnb.figures import watch
    from bnb.figures.sets import FIGURE_SETS
    from bnb.game.api import PRESETS

    unknown = [name for name in options.sets if name not in FIGURE_SETS]
    unknown += [name for name in options.modes if name not in PRESETS]
    if unknown:
        raise SystemExit(f"unknown figure sets or modes {unknown}")
    os.nice(options.nice)
    torch.set_num_threads(options.threads)
    runs_directory = runs_directory_of(options.runs_directory)
    name = options.run
    if (pathlib.Path(name) / "run.json").is_file():  # a run's directory
        runs_directory, name = pathlib.Path(name).resolve().parent, pathlib.Path(name).name
    config = watch.WatchConfig(
        sets=options.sets,
        modes=options.modes,
        environments=options.environments,
        ticks=options.ticks,
        seed=options.seed,
        video=options.video,
        interval=options.interval,
        keep_captures=options.keep_captures,
    )
    from bnb.runs.records import Run

    log = watch.MediaLog(Run.open(runs_directory, name), config) if options.wandb else None
    try:
        make = functools.partial(
            watch.make_media, reseed=lambda: torch.manual_seed(options.seed)
        )  # every checkpoint's captures start from the same draws
        watch.watch(runs_directory, name, config, log=log, once=options.once, make=make)
    except ImportError as error:
        raise SystemExit(f"{error}: bnb figures needs the tools extra (uv sync --extra tools)")
    finally:
        if log is not None:
            log.close()


def queue_options(arguments: list[str]) -> QueueOptions:
    """`bnb queue`'s command line: the action, its own options (QUEUE_FLAGS, anywhere), and
    for add the kind and the job's arguments (everything else, in order)."""
    if not arguments or arguments[0] in ("-h", "--help"):
        print(QueueOptions.__doc__)
        raise SystemExit(0 if arguments else 2)
    action, rest = arguments[0], arguments[1:]
    values: dict[str, object] = {}
    positional: list[str] = []
    index = 0
    while index < len(rest):
        flag, _, inline = rest[index].partition("=")
        if flag in QUEUE_FLAGS:
            name = flag[2:].replace("-", "_")
            if not QUEUE_FLAGS[flag]:
                values[name] = True
            elif inline:
                values[name] = inline
            elif index + 1 < len(rest):
                index += 1
                values[name] = rest[index]
            else:
                raise SystemExit(f"bnb queue: {flag} takes a value")
        else:
            positional.append(rest[index])
        index += 1
    try:
        for name, kind in (("after", int), ("lines", int), ("poll", float)):
            if name in values:
                values[name] = kind(values[name])
        if action == "add":
            if not positional:
                raise SystemExit("bnb queue add KIND ARGUMENTS... (KIND: train, resume, ratings)")
            values.update(kind=positional[0], arguments=tuple(positional[1:]))
        elif action in ("remove", "log"):
            if len(positional) != 1:
                raise SystemExit(f"bnb queue {action} JOB")
            values["job"] = int(positional[0])
        elif positional:
            raise SystemExit(f"bnb queue {action} takes no {positional}")
    except ValueError as error:
        raise SystemExit(f"bnb queue: {error}") from error
    return QueueOptions(action=action, **values)


def _check_job(kind: str, arguments: tuple[str, ...]) -> None:
    """Parse a job's arguments as its command would, so a typo fails when it is queued."""
    if kind == "train":
        tyro.cli(Train, args=list(arguments), prog="bnb queue add train")
    elif kind == "resume" and arguments:
        tyro.cli(Train, args=["--resume", *arguments], prog="bnb queue add resume")
    elif kind == "ratings":
        tyro.cli(Ratings, args=list(arguments), prog="bnb queue add ratings")


def queue(options: QueueOptions) -> None:
    """`bnb queue`: add, list, remove, show a log, or run the queue."""
    import shlex

    from bnb.runs.queue import LOCK_WRAPPER, QUEUE_FILE, Queue, QueueError, Runner, describe

    store = Queue(pathlib.Path(options.queue_file) if options.queue_file else QUEUE_FILE)
    try:
        if options.action == "add":
            _check_job(options.kind, options.arguments)
            if options.program is not None:
                program = shlex.split(options.program)
            elif (script := pathlib.Path(sys.executable).with_name("bnb")).exists():
                program = [str(script)]
            else:
                program = [sys.executable, "-c", "import sys; from bnb.cli import main; main()"]
            given = next(
                (
                    options.arguments[index + 1]
                    for index, argument in enumerate(options.arguments[:-1])
                    if argument == "--runs-directory"
                ),
                None,
            )
            job = store.add(
                options.kind,
                list(options.arguments),
                program=program,
                cwd=pathlib.Path.cwd(),
                runs_directory=runs_directory_of(given).resolve(),
                after=options.after,
                background=options.background,
            )
            print(f"job {job['id']}: {job['kind']} {' '.join(job['arguments'])}, queued")
        elif options.action == "list":
            print("\n".join(describe(store)))
        elif options.action == "remove":
            print(store.remove(options.job))
        elif options.action == "log":
            from bnb.runs.queue import find

            path = store.log_path(find(store.read(), options.job))
            lines = path.read_text(errors="replace").splitlines() if path.exists() else []
            print("\n".join([f"{path}:", *lines[-options.lines :]]))
        elif options.action == "run":
            lock = LOCK_WRAPPER if options.lock is None else pathlib.Path(options.lock)
            if options.lock == "none":
                lock = None
            elif not lock.exists():
                raise SystemExit(f"no bench lock wrapper at {lock} (--lock PATH, or --lock none)")
            print(f"[queue] runner for {store.path}, lock {lock}", flush=True)
            Runner(store, lock, poll=options.poll).run(until_idle=options.until_idle)
        else:
            raise SystemExit(f"bnb queue {{add,list,remove,log,run}}, not {options.action!r}")
    except QueueError as error:
        raise SystemExit(str(error)) from error
