"""The `bnb` command: `bnb train` trains or clones, `bnb runs` lists runs and uploads
checkpoints, `bnb ship-card` charts a ship card, `bnb ratings` and `bnb head-to-head` rate
policies, `bnb crossover` finds the ship-count handicap at which a policy's score crosses
one half, `bnb figures` draws what a policy's heads have learned and films it playing.

An entry point (code style, Layers): it sets the process-wide options its commands need (the
seed, the matmul precision, the charts' backend). The play shell (`python -m bnb.play`) and
`bench` are the other entry points."""

import dataclasses
import functools
import json
import os
import pathlib
import secrets
import subprocess
import sys
from typing import Annotated, Literal

import tyro

from bnb.contracts import REWARD_LEVELS
from bnb.contracts.config import TEAM_DEATHMATCH
from bnb.figures.video import VideoConfig

# The rating games (`bnb.league.tournament.PRESET_NAMES`, kept here so the CLI starts
# without torch): frontline, elimination and team deathmatch, rated by kill share
# (`bnb.league.tournament.kill_share_scored`) unless --deathmatch-score win_loss.
RATED_PRESETS = ("frontline", "elimination", "deathmatch")


@dataclasses.dataclass(frozen=True)
class Train:
    """Train a model with PPO (learner spec), clone the scripted agent (BC), or chain the
    two, in a run directory (runs spec): Yemong (--model yemong), the toy relational model
    (relational), or the nearest-neighbour MLP (nearest; its settings
    `--set nearest.hidden=...`, `allies`, `enemies`, `fields`).

    preset: a game preset (deathmatch, deathmatch_wide, frontline, real_scale_deathmatch,
    real_scale_frontline), which sets the mode, the sizes, the map and the ship set, or a
    training mix in one batch (`bnb.runtime.train.TRAINING_MIXES`): mixed, frontline and
    deathmatch half each; real_scale_mixed, the same on the real-scale maps and ships;
    without one, the mode and sizes below.
    precision: bfloat16 (the model's matmuls), float32 or tf32; old_log_probabilities:
    rollout or recomputed (by unroll before the first epoch).
    updates: the updates to train (a resume continues to the run's total).
    seed: the run's seed (TrainConfig.seed: every keyed draw and the learner's streams).
    set: TrainConfig fields by dotted path, such as learner.entropy_shoot=-0.01 or
    yemong.width=256 (JSON values); fields inside tuples (such as pools) cannot be set.
    resume: continue the run named (PPO or BC; a chain: see objective), from its newest
    checkpoint or the one at --at, with every setting from the checkpoint; only the budget
    (--updates, or --bc-budget-steps), the precision, the micro-batch rows, the checkpoint
    cadence and --set of OVERRIDABLE fields (the learning rate, the warm-up, the loss
    coefficients, the reward weights) may change.
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
    Or chain (`bnb.runs.chain`): BC until its stop rule fires, then PPO from the clone's
    exponential average (a warm start: a fresh optimizer, its warm-up and
    WARM_START_ENTROPY), to chain_budget_steps environment steps in all (BC's budget is
    the same number, a cap), as two runs, `<name>-bc` and `<name>-rl`, with the chain's
    record in `<name>/`. Every other option applies to both stages; bc_set and rl_set are
    --set lines for one stage (after --set). `--resume NAME` continues the chain at the
    stage it reached; it takes only --chain-budget-steps (raised, or lowered before the
    handover), --device, --runs-directory, --checkpoint-every and --wandb.
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
    chain_budget_steps: int | None = None
    bc_set: Annotated[tuple[str, ...], tyro.conf.UseAppendAction] = ()
    rl_set: Annotated[tuple[str, ...], tyro.conf.UseAppendAction] = ()
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
# What a chain's resume may be given besides --resume: the rest comes from its record.
CHAIN_RESUME_OPTIONS = (
    "resume",
    "objective",
    "chain_budget_steps",
    "device",
    "runs_directory",
    "checkpoint_every",
    "wandb",
)
# The options a chain sets for its stages, which its own command cannot.
CHAIN_SETS = ("updates", "bc_budget_steps", "fork", "init_from", "at", "probe", "calibrate_entropy")
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

    A Yemong policy may play ablated: `PATH:rotary`, `PATH:forecast`, `PATH:absent_zones`
    or a combination, `PATH:rotary+forecast` (`bnb.league.matches.ablations_of`), the same
    weights with the rotary rotation as the identity, the forecast not fed back, or the
    absent zone slots masked out of attention, on its own rows only.
    A network may play at a sampling temperature, `PATH@T=0.7` (after any ablations,
    `PATH:rotary@T=0.7`; `bnb.league.matches.temperature_of`): its joint action drawn from
    logits / T, on its own rows only.

    run: the run; with_: more players; presets: the game presets; target_stderr: rating
    points; budget: games per preset at most; blocks and block_environments: the engine's
    layout; device; runs_directory: as for bnb train. ladder: the stationary ladder's rungs
    below the scripted agent (the scripted agent always plays: it anchors the scale at
    1000). league: the run's `league/` policies (else only `--with`'s players and the
    ladder). cheese: the six cheese agents (`scripted:<name>`, `bnb.controllers.scripted_cheese`)
    play, to check the policies against crude strategies (`--no-cheese` leaves them out).
    scripted_plays: the only players the scripted agent plays (by the names given;
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

    lineup: each slot's ship type by name, both teams, for every player (the scripted agent
    and its ladder too) in every preset (`TrainConfig.lineup`; empty: the game's own). A
    record keeps no lineup: reuse only records played with the same one.
    seed: the games' draws (`bnb.league.matches.ArenaConfig.seed`; None: a fresh one,
    recorded with the result), so several tournaments of one field, run at once and reused
    together, play different games.
    deathmatch_score: how the fit scores a team deathmatch game: kill_share (the default),
    its kill share, or win_loss, a win, draw or loss by the sign of the final kill
    difference (`bnb.league.tournament.kill_share_scored`). Each batch records every
    block's full outcome (wins, draws, losses, summed kill shares and kills), so a record
    is resumed and reused under either; a record written before that, which holds only
    its own scoring's results, is not resumed (choose another --out) or reused under the
    other.

    arena_scale: an evaluation-only map size, every preset played at its own arena radius
    times this scale, at least its mode's minimum (`bnb.game.api.arena_radius_for`, the
    radius a training draw of that scale plays; `rated_game`); None: the preset's own. A
    record keeps no scale: give each scale its own --out.

    A budget of 0 plays no games: the field is refitted from the record in `--out` and the
    `--reuse` records alone, under `--deathmatch-score`, so a tournament can be re-rated
    under the other scoring, or with players added or left out (a field other than the
    record's takes a fresh `--out` and the old directory as `--reuse`).
    """

    run: tyro.conf.Positional[str]
    with_: Annotated[tuple[str, ...], tyro.conf.arg(name="with")] = ()
    presets: tuple[str, ...] = RATED_PRESETS
    target_stderr: float = 25.0
    budget: int = 20_000
    blocks: int = 8
    block_environments: int = 128
    device: str = "cuda"
    runs_directory: str | None = None
    ladder: bool = True
    league: bool = True
    cheese: bool = True
    scripted_plays: tuple[str, ...] | None = None
    out: str | None = None
    reuse: tuple[str, ...] = ()
    lineup: tuple[str, ...] = ()
    seed: int | None = None
    deathmatch_score: Literal["kill_share", "win_loss"] = "kill_share"
    arena_scale: float | None = None


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
    "PATH:rotary", "PATH:forecast" or "PATH:rotary+forecast", or a network at a sampling
    temperature, "PATH@T=0.7" (after any ablations). Prints and records, per pair,
    the wins, losses and draws, the score with its standard error and the split by preset;
    the result goes to the first named run's `artifacts/head_to_head/` (else the runs
    directory's).

    players: the pairs; games: games per pair and preset (even); presets; device;
    runs_directory: as for bnb train; deathmatch_score: how a team deathmatch game is
    scored, kill_share (the default) or win_loss (as for bnb ratings); lineup: each slot's
    ship type by name, both teams, for every player (as for bnb ratings; empty: the game's
    own); blocks and block_environments: the engine's layout at most (as for bnb crossover);
    seed: the games' draws (`bnb.league.matches.ArenaConfig.seed`).
    """

    players: tyro.conf.Positional[tuple[str, ...]]
    games: int = 64
    presets: tuple[str, ...] = RATED_PRESETS
    device: str = "cuda"
    runs_directory: str | None = None
    deathmatch_score: Literal["kill_share", "win_loss"] = "kill_share"
    lineup: tuple[str, ...] = ()
    blocks: int = 8
    block_environments: int = 128
    seed: int = 0


@dataclasses.dataclass(frozen=True)
class Crossover:
    """Crossover fights (`bnb crossover A [B] --pairs a:b ...`): A against B (default
    "scripted") at unequal ship counts, to find the handicap at which A's score crosses one
    half (`bnb.league.crossover`). The game is the preset's, at `--ships-per-team` ships per
    team when given (default the preset's own); each pair a:b plays A with a active ships
    against B with b, the rest of each team's slots benched (they stay dead and out of play,
    and team deathmatch's winning score counts the enemy's active ships), `--games` games
    per pair split evenly between the seats, every pair in the same batches. A and B name
    players as for bnb head-to-head; a player may be named twice (scripted against itself).

    Prints and records, per pair, A's wins, draws and losses and its score (a draw half a
    win) with a 95% Wilson interval, and where the score crosses one half between
    neighbouring pairs (in the order given, interpolated): `pairs.jsonl` (a line per pair)
    and `result.json` (the provenance with the command, the players' SHA-256, the rows and
    the crossings) in `artifacts/crossover/<time>/` of `--out` (else A's run, else the
    runs directory). To refine, play a finer grid or more games around the crossing.

    players: A, and B (default scripted); pairs: the active counts, a:b each (A's, B's;
    each from 1 to the ships per team); preset; ships_per_team: the game's size M (the
    largest count any pair needs, at least); games: per pair (even); blocks and
    block_environments: the engine's layout at most; seed: the games' draws
    (`bnb.league.matches.ArenaConfig.seed`); device; out; runs_directory: as for bnb train;
    lineup: each slot's ship type by name, both teams, for both players (as for bnb ratings;
    empty: the game's own), so a team's benched ships are its lineup's last ones.
    """

    players: tyro.conf.Positional[tuple[str, ...]]
    pairs: tuple[str, ...]
    preset: str = "deathmatch"
    ships_per_team: int | None = None
    games: int = 256
    blocks: int = 8
    block_environments: int = 128
    seed: int = 0
    device: str = "cuda"
    out: str | None = None
    runs_directory: str | None = None
    lineup: tuple[str, ...] = ()


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
    """The GPU job queue (`bnb queue ACTION`, `bnb.runs.queue`): one job at a time per GPU,
    by priority, then in order.

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
    On a machine of several GPUs, one runner per GPU: `bnb queue run --slot K [--gpu G]`
    (the slot's jobs see only GPU G, by default K). `add --priority P`: ready jobs start
    highest first (default 0); `add --warm KEY`: wait while a job of the same key compiles
    (has logged no update yet).

    action: add, list, remove, log or run; job: the job's number (remove, log); kind and
    arguments: the job (add); after and background: as above; queue_file: the queue
    (default ~/.local/share/bnb/queue.json; the logs go to logs/queue/ beside it); program:
    the command that runs bnb (default this environment's bnb); lines: the log lines
    printed; lock: the bench lock's wrapper the runner runs each job under ("none": no
    lock); poll: the runner's seconds between looks at the queue; until_idle: the runner
    exits once nothing is running or ready; slot and gpu: the runner's slot and its GPU;
    priority and warm: the job's (add).
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
    slot: int | None = None
    gpu: str | None = None
    priority: float = 0.0
    warm: str | None = None


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
    "--slot": True,
    "--gpu": True,
    "--priority": True,
    "--warm": True,
}

COMMANDS = (
    "train",
    "play",
    "runs",
    "ship-card",
    "ratings",
    "head-to-head",
    "crossover",
    "figures",
    "queue",
)


def main() -> None:
    """The `bnb` command: `bnb train ...` trains, `bnb play ...` opens the play shell,
    `bnb runs ...` lists and shows runs and uploads checkpoints, `bnb ship-card ...` charts a
    card, `bnb ratings ...` and
    `bnb head-to-head ...` rate policies offline, `bnb crossover ...` plays them at unequal
    ship counts, `bnb figures ...` draws what a policy's
    heads have learned (`bnb figures video ...` films a game, `bnb figures watch RUN` makes
    both while a run trains), `bnb queue ...` queues and runs GPU jobs."""
    os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")  # Triton 3.9's default, made explicit
    # The CUDA caching allocator grows its segments in place rather than caching fixed blocks:
    # at a real-scale PPO update (160 micro-batch rows) 0.73 GB less reserved memory and the
    # steady update about 1% faster (docs/rewrite/sections/compile-overhaul-notes.md). Read
    # at the first CUDA allocation, so it has to be set here, before any.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    command, arguments = (sys.argv[1:2] or [""])[0], sys.argv[2:]
    if command in ("train", "ratings", "head-to-head", "figures"):
        compile_limits()
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
    elif command == "crossover":
        crossover(tyro.cli(Crossover, args=arguments, prog="bnb crossover"))
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


def compile_limits() -> None:
    """Room for one compiled rollout per match engine a process builds. Every `Rollout`
    compiles its stages again (their guards read the instance), and each compile counts
    against the stage function's recompile limit (8 by default): with `fullgraph=True` the
    ninth engine in a process (a ratings run over many presets or sizes) fails with
    `FailOnRecompileLimitHit`."""
    import torch._dynamo

    torch._dynamo.config.recompile_limit = 64


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
    from bnb.runtime.train import LINEUP, TRAINING_MIXES, TrainConfig, mixed

    cloned = options.objective == "bc"
    pools = cloning.LINEUP if cloned else LINEUP
    if options.preset in TRAINING_MIXES:
        game, pools = mixed(pools, TRAINING_MIXES[options.preset])
    elif options.preset is not None:
        if options.preset not in PRESETS:
            raise SystemExit(
                f"unknown preset {options.preset!r}; one of "
                f"{sorted(PRESETS) + sorted(TRAINING_MIXES)}"
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
    """`bnb train`: a new run, a resume, a fork, a warm start or a chain, logging a line per
    update; exits with the reason for a setting that does not fit. SIGINT, SIGTERM or the
    run's STOP file (a chain's: the running stage's) stops it after the update in progress,
    with a checkpoint (`bnb.runs.session.StopRequest`), from the start: a request during the
    setup stops the run after no update."""
    from bnb.runs.chain import Chain
    from bnb.runs.session import StopRequest

    with StopRequest() as stop:
        resumes_chain = options.resume is not None and Chain.exists(
            runs_directory_of(options.runs_directory), options.resume
        )
        if options.objective == "chain" or resumes_chain:
            chain(options, given_options, stop)
        else:
            _train(options, given_options, stop)


def _train(options: Train, given_options: set[str], stop, chain_entry: dict | None = None):
    """One run: a new one, a resume, a fork or a warm start (`train`); returns its `Run`
    (None for --print-config, --calibrate-entropy and --probe). chain_entry: a chain's new
    stage (`bnb.runs.chain.Chain.stage_record`), recorded in its run.json; the stage
    writes a checkpoint at its start, so it resumes from any point."""
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
            if chain_entry is not None:
                run.update(chain=chain_entry)
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
            f"nonfinite beliefs {row['counters/nonfinite_beliefs']}  reward/s: {rewards}"
            f"{alert_text(row)}",
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
    if chain_entry is not None and not options.resume:
        session.save(run, trainer, recorder=recorder)
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
    return run


def chain(options: Train, given_options: set[str], stop) -> None:
    """`bnb train --objective chain` (`bnb.runs.chain`): a new chain, or `--resume` of one,
    continued at the stage it reached. The BC stage trains to its stop rule with the
    combined budget as its own; the handover is recorded; the RL stage is a warm start from
    the BC stage's exponential average whose update budget is what the combined budget
    leaves (`bnb.runs.chain.reinforcement_updates`). A stop request, or a BC stage that did
    not finish, ends the process after the stage."""
    import gc

    import torch

    from bnb.runs.chain import Chain, reinforcement_updates
    from bnb.runs.records import Run, run_name
    from bnb.runs.session import HANDOFF_PLAYER, POLICY_VARIANTS

    runs_directory = runs_directory_of(options.runs_directory)
    if options.resume:
        fixed = sorted(given_options - set(CHAIN_RESUME_OPTIONS))
        if fixed:
            flags = ", ".join("--" + name.replace("_", "-") for name in fixed)
            raise SystemExit(
                f"a chain's resume takes {flags} from the chain's record and cannot change it"
            )
        chained = Chain.open(runs_directory, options.resume)
        if "chain_budget_steps" in given_options:
            chained.update(budget_steps=options.chain_budget_steps)
    else:
        fixed = sorted(given_options & set(CHAIN_SETS))
        if fixed:
            flags = ", ".join(
                "--from" if name == "fork" else "--" + name.replace("_", "-") for name in fixed
            )
            raise SystemExit(f"a chain sets its stages' {flags} itself: leave them out")
        if options.chain_budget_steps is None:
            raise SystemExit("a chain needs --chain-budget-steps, its combined budget")
        if options.print_config:
            print(json.dumps(chain_configs(options), indent=1))
            return
        name = options.name or run_name(options.preset or options.mode, runs_directory)
        try:
            chained = Chain.create(runs_directory, name, sys.argv[2:], options.chain_budget_steps)
        except FileExistsError as error:
            raise SystemExit(str(error)) from error
        print(f"chain {name} in {chained.directory}", flush=True)
    budget = chained.record["budget_steps"]
    original = chained_options(chained)
    # what a resume may change, from its command line, else from the chain's own: passed on
    # to the stage it resumes or begins
    passed = {
        name: getattr(options if name in given_options else original, name)
        for name in ("device", "checkpoint_every", "wandb")
    }
    passed["runs_directory"] = str(runs_directory)
    base = dataclasses.replace(original, **passed)
    if chained.record["handover"] is None:
        stage = chained.stage_run("bc")
        bc_run = Run.open(runs_directory, stage) if (runs_directory / stage).exists() else None
        if bc_run is None or bc_run.record["status"] != "finished":
            if bc_run is None:
                bc_options = stage_options(base, "bc", name=stage, bc_budget_steps=budget)
                given_stage, entry = set(), chained.stage_record("bc", 0)
            else:
                # the stage's own budget: the combined one, or a --bc-set stop.budget_steps
                own = _requested(stage_options(original, "bc", bc_budget_steps=budget))
                bc_options = Train(resume=stage, bc_budget_steps=own.stop.budget_steps, **passed)
                given_stage, entry = {"resume", "bc_budget_steps"}, None
            bc_run = _train(bc_options, given_stage, stop, entry)
        if stop.by is not None or bc_run.record["status"] != "finished":
            return
        step = bc_run.record["segments"][-1]["end_step"]
        policy = bc_run.variant_policy_path(POLICY_VARIANTS[HANDOFF_PLAYER])
        chained.hand_over(step, bc_run.record["stopped_by"], policy)
        print(
            f"handover at step {step:,} (stopped by {bc_run.record['stopped_by']}): RL from "
            f"{policy}",
            flush=True,
        )
        del bc_run
        gc.collect()  # the BC stage's trainer and its compiled graphs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    handover = chained.record["handover"]
    rl_config = chain_configs(base)["rl"]
    steps_per_update = rl_config["environments"] * rl_config["length"] * rl_config["shards"]
    updates = reinforcement_updates(budget, handover["step"], steps_per_update)
    stage = chained.stage_run("rl")
    rl_run = Run.open(runs_directory, stage) if (runs_directory / stage).exists() else None
    if rl_run is None:
        rl_options = stage_options(
            base, "rl", name=stage, updates=updates, init_from=handover["policy"]
        )
        _train(rl_options, set(), stop, chained.stage_record("rl", handover["step"]))
    else:
        rl_options = Train(resume=stage, updates=updates, **passed)
        _train(rl_options, {"resume", "updates"}, stop)


def chained_options(chained) -> Train:
    """The options a chain was started with, parsed from its record's arguments."""
    return tyro.cli(Train, args=chained.record["arguments"], prog="bnb train")


def stage_options(options: Train, stage: str, **changes) -> Train:
    """A chain's options as one stage's ("bc" or "rl"): the stage's objective, its --set
    lines after the shared ones, and `changes` (the run's name among them)."""
    sets = options.bc_set if stage == "bc" else options.rl_set
    return dataclasses.replace(
        options,
        objective="bc" if stage == "bc" else "ppo",
        set=options.set + sets,
        bc_set=(),
        rl_set=(),
        chain_budget_steps=None,
        resume=None,
        print_config=False,
        **changes,
    )


def chain_configs(options: Train) -> dict:
    """Both stages' resolved TrainConfigs as JSON (`--print-config`): BC's with the
    combined budget as its own, RL's with the warm start's entropy coefficients (the model's
    settings, which a warm start takes from the clone, are the same run's)."""
    from bnb.runs.serialise import to_json

    budget = options.chain_budget_steps
    bc_config = _requested(stage_options(options, "bc", bc_budget_steps=budget))
    rl_options = stage_options(options, "rl")
    rl_config = warm_start_entropy(_requested(rl_options), rl_options.set)
    return dict(bc=to_json(bc_config), rl=to_json(rl_config))


def alert_text(row: dict) -> str:
    """The update line's tail: the trigger rules up this update (`alerts/{rule}` at 1), or
    nothing."""
    up = [
        name.removeprefix("alerts/")
        for name, value in row.items()
        if name.startswith("alerts/")
        and name not in ("alerts/active", "alerts/events")
        and isinstance(value, int | float)
        and value
    ]
    return f"  alerts: {' '.join(up)}" if up else ""


def log_cloning(row: dict) -> None:
    """Print one behaviour-cloning update's row on one line: the timings (ms), the BC KL and the
    teacher's and student's entropies (nats), each score, the score stop's window once one
    is full (the average it reads), and the plateau's fraction once it reads one (the
    recent improvement over the improvement since its start) with how long it has held."""
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
    plateau = (
        f"  plateau {row['bc/stop/plateau/fraction']:.3f} "
        f"held {row['bc/stop/plateau/held_steps']:,}"
        if "bc/stop/plateau/fraction" in row
        else ""
    )
    print(
        f"update {row['counters/updates']:4d}  steps {row['counters/environment_steps']:>11,}  "
        f"rollout {row['train/time/rollout_seconds'] * 1e3:6.0f} ms  "
        f"learn {row['train/time/learner_seconds'] * 1e3:6.0f} ms  "
        f"bc kl {row['bc/kl']:.4f}  teacher {row['bc/entropy/target']:.3f}  "
        f"student {row['bc/entropy/student']:.3f}  scores: {scores}{window}{plateau}"
        f"{alert_text(row)}",
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
        seed=config.seed,
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


def with_lineup(field: list, lineup: tuple[str, ...], presets: tuple[str, ...]) -> list:
    """The field with every player that flies the game's lineup flying `lineup` (names of the
    presets' ship set, which must be one set at one team size)."""
    from bnb.game.api import PRESETS
    from bnb.runtime.train import lineup_indices

    games = {(PRESETS[preset].ship_set, PRESETS[preset].ships_per_team) for preset in presets}
    if len(games) != 1:
        raise SystemExit(f"--lineup: the presets {list(presets)} play other ship sets or sizes")
    try:
        indices = lineup_indices(tuple(lineup), PRESETS[presets[0]])
    except ValueError as error:
        raise SystemExit(f"--lineup: {error}") from error
    return [
        dataclasses.replace(one, ship_type=indices) if one.ship_type is None else one
        for one in field
    ]


def rated_game(preset: str, arena_scale: float | None):
    """The preset's game at an evaluation-only arena scale (`Ratings.arena_scale`): its radius
    times the scale, at least the mode's minimum (`arena_radius_for`); None: the preset's
    own game."""
    from bnb.game.api import PRESETS
    from bnb.game.modes.rules import arena_radius_for

    game = PRESETS[preset]
    if arena_scale is None:
        return game
    if not arena_scale > 0.0:
        raise SystemExit(f"--arena-scale {arena_scale}: a scale above 0")
    radius = arena_radius_for(game, game.ships_per_team, arena_scale)
    return dataclasses.replace(game, arena_radius=radius)


def ratings(options: Ratings) -> None:
    """`bnb ratings`: the offline tournament and its leaderboard."""
    import torch

    from bnb.controllers.scripted_cheese import NAMES as CHEESE_NAMES
    from bnb.controllers.scripted_cheese import player_name
    from bnb.league import tournament
    from bnb.league.matches import ArenaConfig, Player, player
    from bnb.runs import provenance
    from bnb.runs.records import Run
    from bnb.runtime.lanes import SCRIPTED_PLAYER
    from bnb.runtime.opponents import LADDER, ladder_name

    _check_presets(options.presets)
    seed = secrets.randbits(31) if options.seed is None else options.seed
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
        Player(ladder_name(rung), rung=rung)
        for rung in LADDER
        if options.ladder or ladder_name(rung) == SCRIPTED_PLAYER
    ]
    if options.cheese:
        field += _players([player_name(name) for name in CHEESE_NAMES], runs_directory)
    field += [
        p for p in _players(options.with_, runs_directory) if p.name not in {f.name for f in field}
    ]
    if options.lineup:
        field = with_lineup(field, options.lineup, options.presets)
    try:
        allowed = tournament.allowed_pairs([p.name for p in field], options.scripted_plays)
        config = tournament.TournamentConfig(
            arena=ArenaConfig(
                blocks=options.blocks, block_environments=options.block_environments, seed=seed
            ),
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
                deathmatch_score=options.deathmatch_score,
                game=rated_game(preset, options.arena_scale),
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
        deathmatch_score=options.deathmatch_score,
        arena_scale=options.arena_scale,
        reused={preset: [str(record) for record in records] for preset, records in reused.items()},
    )
    print(provenance.write_result(owner, "ratings", payload, inputs))


def head_to_head(options: HeadToHead) -> None:
    """`bnb head-to-head`: the named pairs, and the result recorded."""
    import torch

    from bnb.league import tournament
    from bnb.league.matches import ArenaConfig
    from bnb.runs import provenance, records

    _check_presets(options.presets)
    runs_directory = runs_directory_of(options.runs_directory)
    players = _players(options.players, runs_directory)
    if options.lineup:
        players = with_lineup(players, options.lineup, options.presets)
    try:
        rows = tournament.head_to_head(
            players,
            options.games,
            options.presets,
            torch.device(options.device),
            ArenaConfig(
                blocks=options.blocks,
                block_environments=options.block_environments,
                seed=options.seed,
            ),
            deathmatch_score=options.deathmatch_score,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    for row in rows:
        print(
            f"{row['first']} against {row['second']}: {row['wins']:g} wins, {row['losses']:g} "
            f"losses, {row['draws']:g} draws; score {row['score']:.3f} +- {row['stderr']:.3f}"
        )
        for preset, part in row["presets"].items():
            print(
                f"  {preset:16s} {part['wins']:g}-{part['losses']:g}-{part['draws']:g}, "
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
    payload = dict(
        pairs=rows,
        games=options.games,
        presets=list(options.presets),
        deathmatch_score=options.deathmatch_score,
        lineup=list(options.lineup),
        seed=options.seed,
    )
    print(provenance.write_result(owner, "head_to_head", payload, inputs))


def crossover(options: Crossover) -> None:
    """`bnb crossover`: the pairs of ship counts, the table, and the result recorded."""
    import torch

    from bnb.game.api import PRESETS
    from bnb.league import crossover as fights
    from bnb.league.matches import ArenaConfig
    from bnb.runs import provenance, records

    _check_presets((options.preset,))
    if not 1 <= len(options.players) <= 2:
        raise SystemExit("bnb crossover takes one or two players: A [B]")
    names = (*options.players, "scripted")[:2]
    runs_directory = runs_directory_of(options.runs_directory)
    first, second = _players(names, runs_directory)
    if options.lineup:
        first, second = with_lineup([first, second], options.lineup, (options.preset,))
    try:
        pairs = [fights.parse_pair(text) for text in options.pairs]
        game = PRESETS[options.preset]
        if options.ships_per_team is not None:
            game = dataclasses.replace(game, ships_per_team=options.ships_per_team)
        layout = ArenaConfig(
            blocks=options.blocks, block_environments=options.block_environments, seed=options.seed
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    try:
        rows = fights.crossover(
            first, second, pairs, options.games, game, torch.device(options.device), layout
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    found = fights.crossings(rows)
    print(
        f"{first.name} against {second.name}, {options.preset} at "
        f"{game.ships_per_team}v{game.ships_per_team}, {options.games} games per pair"
    )
    print(
        f"{'pair':>7s} {'games':>6s} {'wins':>6s} {'draws':>6s} {'losses':>6s}  score [95% Wilson]"
    )
    for row in rows:
        print(
            f"{row['first_active']:>3d}:{row['second_active']:<3d} {row['games']:6d} "
            f"{row['wins']:6d} {row['draws']:6d} {row['losses']:6d}  {row['score']:.3f} "
            f"[{row['score_low']:.3f}, {row['score_high']:.3f}]"
        )
    for crossing in found:
        print(
            f"the score crosses 0.5 between {' and '.join(crossing['between'])}: at "
            f"{crossing['first_active']:.2f}:{crossing['second_active']:.2f} "
            f"(handicap {crossing['handicap']:+.2f} ships)"
        )
    if not found:
        print("the score does not cross 0.5 along the pairs")
    if options.out is not None:
        owner = pathlib.Path(options.out)
    else:
        owner = next(
            (
                runs_directory / name
                for name in names
                if (runs_directory / name / records.RECORD).is_file()
            ),
            runs_directory,
        )
    inputs = {one.name: one.policy for one in (first, second) if one.policy is not None}
    payload = dict(
        rows=rows,
        crossings=found,
        preset=options.preset,
        game=dataclasses.asdict(game),
        games=options.games,
        seed=options.seed,
    )
    directory = provenance.write_result(owner, "crossover", payload, inputs)
    (directory / "pairs.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    print(directory)


def figures(options: Figures) -> None:
    """`bnb figures`: capture and draw (`bnb.figures.command.figures`)."""
    import torch

    from bnb.figures.command import figures as draw

    torch.manual_seed(options.seed)  # the shot figure's spread samples
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
        for name, kind in (
            ("after", int),
            ("lines", int),
            ("poll", float),
            ("slot", int),
            ("priority", float),
        ):
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
                priority=options.priority,
                warm=options.warm,
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
            slot = "" if options.slot is None else f", slot {options.slot}"
            print(f"[queue] runner for {store.path}, lock {lock}{slot}", flush=True)
            Runner(store, lock, poll=options.poll, slot=options.slot, gpu=options.gpu).run(
                until_idle=options.until_idle
            )
        else:
            raise SystemExit(f"bnb queue {{add,list,remove,log,run}}, not {options.action!r}")
    except QueueError as error:
        raise SystemExit(str(error)) from error
