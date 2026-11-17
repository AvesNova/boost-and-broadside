"""The `bnb` command: `bnb train` trains or clones, `bnb runs` lists runs and uploads
checkpoints, `bnb average` builds averages of a run's weights after training, `bnb ship-card`
charts a ship card, `bnb ratings` and `bnb head-to-head` rate
policies, `bnb crossover` finds the ratio of a policy's ships to the scripted agent's
at which it wins half its games, `bnb evaluate` decides between two policies, `bnb figures`
draws what a policy's heads have learned and films it playing.

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

from bnb.contracts import VALUE_LEVELS
from bnb.contracts.config import TEAM_DEATHMATCH
from bnb.figures.hero import HeroConfig
from bnb.figures.match import MatchConfig
from bnb.figures.trails import StillConfig
from bnb.figures.video import VideoConfig
from bnb.league.evaluation_settings import EvaluationConfig

# The rating games (`bnb.league.tournament.PRESET_NAMES`, kept here so the CLI starts
# without torch): frontline, elimination and team deathmatch, team deathmatch rated by the
# kill lead (a win, draw or loss) unless --deathmatch-score kill_share
# (`bnb.league.tournament.kill_share_scored`).
RATED_PRESETS = ("frontline", "elimination", "deathmatch")


@dataclasses.dataclass(frozen=True)
class Train:
    """Train a model with PPO (learner spec), clone Imugi (BC), or chain the
    two, in a run directory (runs spec): Yemong (--model yemong), the toy relational model
    (relational), or the nearest-neighbour MLP (nearest; its settings
    `--set nearest.hidden=...`, `allies`, `enemies`, `fields`).

    preset: a game preset (deathmatch, deathmatch_wide, frontline, real_scale_deathmatch,
    real_scale_frontline), which sets the mode, the sizes, the map and the ship set, or a
    training mix in one batch (`bnb.runtime.train.TRAINING_MIXES`): mixed, frontline and
    deathmatch half each; real_scale_mixed, the same on the real-scale maps and ships;
    real_scale_all_modes, the three real-scale modes at 6/16 frontline, 5/16 deathmatch and
    5/16 elimination (the arms' mix);
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
    objective: ppo, or bc (behaviour cloning against self-play and Imugi until
    a stop rule fires, then the clone is saved as the run's policy file; updates is unused;
    a BC run keeps training checkpoints and resumes, but is not forked or warm-started);
    teacher_temperature and teacher_floor set the teacher (Imugi at its own
    temperature, its labels mixed with uniform by the floor; T = 1 and floor 0 by
    default); bc_budget_steps: BC's budget in environment steps
    (TrainConfig.stop.budget_steps; None: `bnb.learn.bc.StopConfig`'s, 100M, the same for
    every arm; BC's plateau rule only logs unless `--set stop.plateau_stops=true`).
    Or chain (`bnb.runs.chain`): BC until its stop rule fires, then PPO from the clone's
    exponential average (a warm start: a fresh optimizer, its warm-up and
    WARM_START_ENTROPY), to chain_budget_steps environment steps in all (BC's budget is
    its own, 100M, or the chain's bc_budget_steps, at most the combined one:
    `chain_bc_budget`; `--bc-budget-steps 0` is no BC stage, RL from scratch over the whole
    budget with the RL stage's settings, WARM_START_ENTROPY included), as two runs,
    `<name>-bc` and `<name>-rl`, with the chain's record in `<name>/`. Every other option
    applies to both stages; bc_set and rl_set are --set lines for one stage (after
    --set). `--resume NAME` continues the chain at the
    stage it reached; it takes only --chain-budget-steps (raised, or lowered before the
    handover), --device, --runs-directory, --checkpoint-every and --wandb.
    wandb: stream each update's record to W&B (project `boost-and-broadside-v2`, the run's
    name) as well as to the run's metrics.jsonl; without it nothing leaves the machine.
    calibrate_entropy: print Imugi's natural entropy over its own games of the
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
    objective: str = "ppo"
    teacher_temperature: float = 1.0
    teacher_floor: float = 0.0
    bc_budget_steps: int | None = None
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
CHAIN_SETS = ("updates", "fork", "init_from", "at", "probe", "calibrate_entropy")
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
    checkpoint (`bnb runs upload RUN [--at STEP] [--league]`) as a W&B artifact, build the
    run's W&B dashboards (`bnb runs dashboard RUN`, the `tools` extra), or mirror a run
    trained without --wandb into W&B (`bnb runs mirror RUN [--follow] [--host HOST]`,
    `bnb.runs.mirror`).

    action: list, show, upload, dashboard or mirror; run: the run's (or for mirror a
    chain's) name (all but list).
    at: the checkpoint's step (default: the newest); league: upload the league's snapshots
    instead; to: the target (wandb only, for now); runs_directory: as for bnb train.
    follow: mirror until the run is over, polling every `every` seconds; host: fetch the
    run from this ssh host's `remote_runs` into `cache_directory` (default
    ~/.cache/bnb/mirror/HOST).
    """

    action: tyro.conf.Positional[str] = "list"
    run: tyro.conf.Positional[str | None] = None
    at: int | None = None
    league: bool = False
    to: str = "wandb"
    runs_directory: str | None = None
    follow: bool = False
    every: float = 300.0
    host: str | None = None
    remote_runs: str = "/workspace/runs"
    cache_directory: str | None = None


@dataclasses.dataclass(frozen=True)
class Average:
    """Build averages of a run's weights from its per-update live policies (a run trained
    with `TrainConfig.update_policies` on) and write each as a policy file a tournament
    rates (`bnb.runs.averaging`): uniform over a step window, exponential at a half-life,
    windowed uniform over the last updates, and the run's own two averages rebuilt (checked
    against the run's files where both exist), each at the spec's evaluation points. The
    files are read once each, one at a time. An average written before is kept.

    run: the run; spec: the TOML spec of the averages
    (`docs/rewrite/sections/averaging/suite.toml` is the first suite); runs_directory: as
    for bnb train. The player names (`RUN/average/<name>/step_<steps>`, or the live
    file's path) go to stdout one per line, for `bnb ratings RUN --no-league --with
    $(cat players.txt)` after `bnb average RUN --spec SPEC > players.txt`; the progress and
    the checks go to stderr. Exits with
    an error when a rebuilt training average differs from the run's own.
    """

    run: tyro.conf.Positional[str]
    spec: str
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
    below Imugi (Imugi always plays: it anchors the scale at
    1000). league: the run's `league/` policies (else only `--with`'s players and the
    ladder). cheese: the six cheese agents (`cheese:<name>`, `bnb.controllers.cheese`)
    play, to check the policies against crude strategies (`--no-cheese` leaves them out).
    imugi_plays: the only players Imugi plays (by the names given;
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

    lineup: each slot's ship type by name, both teams, for every player (Imugi
    and its ladder too) in every preset (`TrainConfig.lineup`; empty: the game's own). A
    record keeps no lineup: reuse only records played with the same one.
    seed: the games' draws (`bnb.league.matches.ArenaConfig.seed`; None: a fresh one,
    recorded with the result), so several tournaments of one field, run at once and reused
    together, play different games.
    deathmatch_score: how the fit scores a team deathmatch game: win_loss (the default), a
    win, draw or loss by the sign of the final kill difference, or kill_share, its kill
    share (`bnb.league.tournament.kill_share_scored`). Each batch records every
    block's full outcome (wins, draws, losses, summed kill shares and kills), so a record
    is resumed and reused under either; a record written before that, which holds only
    its own scoring's results, is not resumed (choose another --out) or reused under the
    other.

    arena_scale: an evaluation-only map size, every preset played at its own arena radius
    times this scale, at least its mode's minimum (`bnb.game.api.arena_radius_for`, the
    radius a training draw of that scale plays; `rated_game`); None: the preset's own. A
    record keeps no scale: give each scale its own --out.
    active_fields: evaluation-only field-count sampling, LOW HIGH: every game of every
    preset plays a drawn number of its field slots in that range, the rest absent
    (`bnb.contracts.ActiveFields`; `rated_game`); None: every field. A record keeps no
    field count: give each range its own --out.

    A budget of 0 plays no games: the field is refitted from the record in `--out` and the
    `--reuse` records alone, under `--deathmatch-score`, so a tournament can be re-rated
    under the other scoring, or with players added or left out (a field other than the
    record's takes a fresh `--out` and the old directory as `--reuse`).

    Targets: `--target-stderr` for every player; `--decision NAME ...` players the stricter
    `--decision-stderr`; `--contrast A:B ...` the difference of two players' ratings, with
    `--contrast-stderr` (`bnb.league.tournament.TournamentConfig`). Each batch's blocks
    follow the A-optimal design for the unmet targets (`bnb.league.design.plan_batch`),
    with at most `--players-per-batch` distinct players a batch (None: as many as its
    slots). A target the budget cannot bring in is left out while others can be met; when
    every unmet target needs more than `--hopeless` times the batches left, the tournament
    stops (inf: it plays its budget).
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
    imugi_plays: tuple[str, ...] | None = None
    out: str | None = None
    reuse: tuple[str, ...] = ()
    lineup: tuple[str, ...] = ()
    seed: int | None = None
    deathmatch_score: Literal["win_loss", "kill_share"] = "win_loss"
    arena_scale: float | None = None
    active_fields: tuple[int, int] | None = None
    decision: tuple[str, ...] = ()
    decision_stderr: float = 20.0
    contrast: tuple[str, ...] = ()
    contrast_stderr: float = 25.0
    players_per_batch: int | None = None
    hopeless: float = 10.0


@dataclasses.dataclass(frozen=True)
class HeadToHead:
    """Head-to-head (`bnb head-to-head A B [C D ...] --games N`): only the named pairs (A
    against B, C against D, ...), each over `--games` games per preset split evenly between
    the seats. A player is a policy file, a run (its newest checkpoint), "imugi",
    "random", "rung_T<t>" (Imugi at temperature t), or a teacher,
    "imugi@T=<temperature>,floor=<floor>" (either key optional; T defaults to 1 and
    floor to 0: Imugi at its own temperature T, then a uniform joint action
    with probability floor, drawn as BC's teacher opponent does, so "imugi" is BC's
    default teacher and "imugi@T=2" plays as "rung_T2"), or a Yemong policy ablated,
    "PATH:rotary", "PATH:forecast" or "PATH:rotary+forecast", or a network at a sampling
    temperature, "PATH@T=0.7" (after any ablations). Prints and records, per pair,
    the wins, losses and draws, the score with its standard error and the split by preset;
    the result goes to the first named run's `artifacts/head_to_head/` (else the runs
    directory's).

    players: the pairs; games: games per pair and preset (even); presets; device;
    runs_directory: as for bnb train; deathmatch_score: how a team deathmatch game is
    scored, win_loss (the default) or kill_share (as for bnb ratings); lineup: each slot's
    ship type by name, both teams, for every player (as for bnb ratings; empty: the game's
    own); blocks and block_environments: the engine's layout at most;
    seed: the games' draws (`bnb.league.matches.ArenaConfig.seed`).
    """

    players: tyro.conf.Positional[tuple[str, ...]]
    games: int = 64
    presets: tuple[str, ...] = RATED_PRESETS
    device: str = "cuda"
    runs_directory: str | None = None
    deathmatch_score: Literal["win_loss", "kill_share"] = "win_loss"
    lineup: tuple[str, ...] = ()
    blocks: int = 8
    block_environments: int = 128
    seed: int = 0


@dataclasses.dataclass(frozen=True)
class Crossover:
    """The break-even lineup (`bnb crossover POLICY`): the lineup of POLICY's ships against
    the opponent's (`--opponent`, Imugi) at which POLICY wins half its games, per mode
    (`bnb.league.crossover`), reported as the ratio of the opponent's ships to POLICY's (how
    many times its number POLICY holds). The weaker side fields `--fixed-count` ships (20)
    and the stronger side's count varies downward from it, to 1 ship in every mode: (n, 20)
    for a policy stronger than the opponent, (20, n) for a weaker one, the games deciding
    which side shrinks, starting at 20 against 20. The game is each preset's at
    `--fixed-count` a side, on the preset's own arena (`--map-rule preset`, the default:
    every map fixed-size, the author's rule of 2026-11-15) or the density rule's for that
    fleet (`--map-rule density`, the map-size arm's; `bnb.game.api.arena_radius_for`); each
    team's unused slots are benched (they stay dead and out of play, and team deathmatch's
    winning score counts the enemy's active ships). Every game draws its lineup of ship
    types, each slot one of `--lineup-types` with equal chance, both teams flying it.

    Batch by batch, a logistic fit of the policy's score (a draw half a win) on the log
    ratio r = log(policy's ships / opponent's ships), over the games of both sides of
    parity, gives the break-even ratio (the score one half) and its standard error in r;
    each batch places its blocks at the lineups most informative about it (before a fit,
    over `--bracket` on the side the games point to), until its standard error reaches
    `--target-stderr` or the games reach `--budget-games`. Outcomes separated by ratio (all
    lost below, all won above) give a bound instead, and the range is extended. Prints a
    line per batch with an ETA, then per mode the break-even ratio, the break-even lineup
    (the weaker side at `--fixed-count`) and the games. A break-even with the stronger side
    at 2 ships or fewer is flagged (`suspicious` in the result) with a warning to check for
    a bug.

    Every batch is appended to `artifacts/crossover/<preset>.batches.jsonl` of `--out`
    (else the policy's run, else the runs directory), and a rerun reuses the batches of the
    same measurement (the players' identities, the game, the seed, the fixed count and the
    lineup types) and continues; the result (with the provenance and the
    policy's SHA-256) goes to `artifacts/crossover/<time>/result.json`.

    Fixed lineups (`--lineups 5v5 5v10 ...`, the policy's ships first): those lineups in
    place of the break-even's allocation, each played to `--lineup-games` games (the
    team-size chart's tallies, `bnb figures results team-size`), on the same engine (no
    side above `--fixed-count`), recorded as their own measurement; the fit is reported as
    usual.

    policy: the policy (as for bnb head-to-head); opponent: its opponent; fixed_count: the
    weaker side's ships; presets; map_rule:
    "preset" or "density" (above); bracket: the stronger side's counts before a fit;
    target_stderr: the break-even log ratio's standard error that stops a mode (0.02, about
    0.3 ships at 15 against 20); budget_games: the most games per mode; lineup_types: ship
    type names of the preset's set; blocks and block_environments: the engine's layout (a
    batch is blocks x block_environments games, a block one lineup); seed: the games' draws
    and lineups; device; out; runs_directory: as for bnb train; lineups and lineup_games:
    the fixed lineups and their games each (above).
    """

    policy: tyro.conf.Positional[str]
    opponent: str = "imugi"
    fixed_count: int = 20
    presets: tuple[str, ...] = (
        "real_scale_frontline",
        "real_scale_deathmatch",
        "real_scale_elimination",
    )
    map_rule: Literal["preset", "density"] = "preset"
    bracket: tuple[int, ...] = (6, 10, 14, 20)
    target_stderr: float = 0.02
    budget_games: int = 8192
    lineup_types: tuple[str, ...] = ("Yongma", "Jujak")
    blocks: int = 8
    block_environments: int = 128
    seed: int = 0
    device: str = "cuda"
    out: str | None = None
    runs_directory: str | None = None
    lineups: tuple[str, ...] = ()
    lineup_games: int = 256


def parse_lineups(lineups: tuple[str, ...]) -> tuple[tuple[int, int], ...]:
    """Lineups in words ("5v10": the policy's ships, then the opponent's) as pairs."""
    pairs = []
    for lineup in lineups:
        policy_ships, _, opponent_ships = lineup.partition("v")
        if not (policy_ships.isdigit() and opponent_ships.isdigit()):
            raise SystemExit(f"--lineups {lineup!r}: the policy's ships v the opponent's (5v10)")
        pairs.append((int(policy_ships), int(opponent_ships)))
    return tuple(pairs)


@dataclasses.dataclass(frozen=True)
class Evaluate:
    """The decision suite (`bnb evaluate A B --out DIR`): A against B over the suite's
    conditions, with a group-sequential test deciding A wins, B wins or a tie
    (`bnb.league.evaluation`; docs/rewrite/evaluation-suite.md). A and B are policies (as
    for bnb head-to-head), normally two arms' final exponential-average weights.

    Plays toward each planned look (the games added where they most reduce the overall
    score's variance per engine second), takes the look when its information is reached,
    and stops when a look decides. Every batch is appended to `DIR/batches/<writer>.jsonl`
    and every look to `DIR/looks/<writer>.jsonl`; `DIR/settings.json` holds the players and
    the settings, and a rerun with the same ones continues. Several processes may share
    DIR, each with its own `--play` (for example one condition each): each plays its
    own, waits while another's are short, and any of them takes a look. Prints a line per
    batch with the next look's ETA.

    a, b: the players; out: the directory; play: the conditions this process plays (None:
    every scored one: `--conditions` less `--unscored`, which by default leaves out
    none); games: a pilot instead: play this process's conditions to
    this many games each and take no look; settings: the suite's settings, recorded in
    settings.json (`bnb.league.evaluation_settings.EvaluationConfig`); device;
    runs_directory: as for bnb train; poll_seconds: how long a process with nothing to play
    waits before reading the records again.

    `bnb evaluate decide DIR` (EvaluateDecide) applies the test at a look now;
    `bnb evaluate design DIR ...` (EvaluateDesign) chooses the design from pilots.
    """

    a: tyro.conf.Positional[str]
    b: tyro.conf.Positional[str]
    out: str
    play: tuple[str, ...] | None = None
    games: int | None = None
    settings: tyro.conf.OmitArgPrefixes[EvaluationConfig] = EvaluationConfig()
    device: str = "cuda"
    runs_directory: str | None = None
    poll_seconds: float = 30.0


@dataclasses.dataclass(frozen=True)
class EvaluateDecide:
    """A look now (`bnb evaluate decide DIR`): every process's records read, the scores per
    condition, per mode and overall printed with their standard errors, and the test
    applied at the current information (the look recorded, unless --peek; a look spends
    some of the test's error rates). Once a look has decided, it is shown again.

    directory: the evaluation's directory; peek: show the look without recording it."""

    directory: tyro.conf.Positional[str]
    peek: bool = False


@dataclasses.dataclass(frozen=True)
class EvaluateDesign:
    """The sequential design from pilots (`bnb evaluate design DIR ...`): each pilot
    directory's per-condition per-game variance and engine seconds per game (averaged over
    the directories), and for each number of looks the planned boundaries, the most engine
    hours, and at each true difference the outcomes' chances and the expected hours; with
    --simulations, the whole procedure simulated as well (`bnb.league.evaluation_design`).

    directories: the pilots; looks: the numbers of looks to compare; differences: true
    differences (overall score) to show, besides 0, delta and 2 delta and each pilot's
    observed one; simulations: repeats of the simulated procedure per difference (0: none);
    seed: the simulation's draws."""

    directories: tyro.conf.Positional[tuple[str, ...]]
    looks: tuple[int, ...] = (2, 3, 4, 5)
    differences: tuple[float, ...] = ()
    simulations: int = 0
    seed: int = 0


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
    The trails set draws one game (`bnb.figures.match`): scripted plays it Imugi against
    Imugi with no policy (`bnb figures trails --scripted`); match: that game
    (`--match.preset`, `--match.ships-per-team`, `--match.seconds` and so on:
    `bnb.figures.match.MatchConfig`); still: one still of it in place of the set's
    defaults when any of its options is given (`--still.start`, `--still.end`,
    `--still.fog`, `--still.theme`, `--still.size`, `--still.formats` and so on:
    `bnb.figures.trails.StillConfig`), named still_name.
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
    scripted: bool = False
    match: MatchConfig = MatchConfig()
    still: StillConfig = StillConfig()
    still_name: str = "trails_still"


@dataclasses.dataclass(frozen=True)
class FiguresVideo:
    """A video of one game (`bnb figures video --policy P`): the policy (team 0) against
    Imugi in the preset's mode, drawn by the play shell offscreen (EGL) and
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
class FiguresHero:
    """The README's hero clip (`bnb figures hero --policy P`): `--hero.games` games of the
    policy against Imugi at one lineup (`--hero.ships 5 6`: the policy's ships, then
    Imugi's), each filmed to its end with the death marks, an MP4 and a JSON of its outcome
    per game in `--out`; then the clip is picked (a decisive win, `bnb.figures.hero.pick`)
    and made a GIF with its poster frame (`--gif`, default <out>/hero.gif).
    docs/guides/figures.md.

    policy: a policy directory or a run's name; preset: the game (the real-scale
    elimination); out: the games' directory; gif: the GIF's path; hero: the games, the
    pick's rules and the GIF's settings (`bnb.figures.hero.HeroConfig`, its videos'
    under `--hero.video.*`); device, threads, runs_directory: as for bnb figures video.
    """

    policy: str
    out: str
    preset: str = "real_scale_elimination"
    gif: str | None = None
    hero: HeroConfig = HeroConfig()
    device: str = "cpu"
    threads: int = 4
    runs_directory: str | None = None


@dataclasses.dataclass(frozen=True)
class FiguresWatch:
    """Media of a training run while it trains (`bnb figures watch RUN`): a second process
    that only reads the run, polls its checkpoints and, for each new one, captures each
    mode on the CPU, draws the quick figure sets and films one game per mode against the
    Imugi, into the run's artifacts/figures/<checkpoint>/ (`bnb.figures.watch`).
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
class FiguresResults:
    """A results chart of the write-up (`bnb figures results CHART SOURCE ...`,
    `bnb.figures.results`; docs/guides/results-charts.md): drawn from records the runs and
    the measurements already write, as a PNG with its provenance and the `--formats`' vector
    copies in `--out`, its numbers merged into the values file `--values` (names quoted by
    the docs as {{name}}; "" writes none). Needs the tools extra (matplotlib).

    Charts and their sources: `fan` and `break-even`, a directory of per-snapshot
    `bnb crossover` results (`<run>/step_<n>/artifacts/crossover/...`); `arms`, a forest
    input (JSON or CSV); `team-size`, game tallies (`<preset>.batches.jsonl` files or
    directories); `health`, a run directory, its metrics.jsonl or `wandb:ENTITY/PROJECT/ID`
    (several: a chain, BC first); `game-length`, per-game JSONL files.

    modes: the modes drawn (fan, team-size; default all found); policy_count and
    opponent_counts: the fan's lineups (5 against each); chain: break-even's chain when the
    directory holds several; ratios: team-size's Imugi ships per ours; trained_team_size:
    the size marked; samples: W&B history samples per key (health).
    """

    chart: tyro.conf.Positional[
        Literal["fan", "break-even", "arms", "team-size", "health", "game-length"]
    ]
    sources: tyro.conf.Positional[tuple[str, ...]]
    out: str = "docs/results"
    values: str = "docs/results/values.json"
    formats: tuple[str, ...] = ("png", "pdf")
    modes: tuple[str, ...] = ()
    policy_count: int = 5
    opponent_counts: tuple[int, ...] = (5, 6, 8, 10)
    chain: str | None = None
    ratios: tuple[float, ...] = (1.0, 1.5, 2.0)
    trained_team_size: int = 5
    samples: int = 1000


@dataclasses.dataclass(frozen=True)
class Pool:
    """The shared rating pool (`bnb pool ACTION POOL ...`, `bnb.league.rating_pool`): every
    rated checkpoint of many runs in one directory per pool, which several machines may
    write at once, rated per game preset against Imugi (1000).

    Actions: `create` makes the pool (its games' conditions: `--lineup`, or
    `--lineup-types`, each game's lineup drawn slot by slot from these types as `bnb
    evaluate` draws it, at any team size; `--arena-scale`, `--deathmatch-score`; creating an
    existing pool with the same conditions opens it);
    `add PATH ...` registers policies (`<run>/policies/step_<n>/<kind>` directories, named
    `<run>/step_<n>/<kind>`; `--name` for one elsewhere) or scripted players by name, and
    `--runs RUN ...` each run's snapshots (run directories, or names in the runs
    directory): its policy files of `--kinds` at the snapshot schedule's steps
    (`bnb.league.rating_pool.scheduled_policies`; a run that keeps every update's files
    gives the scheduled ones);
    `import DIR ...` brings in every `<preset>.batches.jsonl` of `bnb ratings` records
    below the directories (once: an imported batch is not imported again); `rate` plays a
    session per preset (`--presets`) among the players that most need games and can be
    played here, until the targets are met, the budget is spent or the targets left are out
    of reach; `table` prints each preset's leaderboard and goodness of fit (observed
    against predicted win rates, and a likelihood-ratio test for loops: Bradley-Terry
    against mElo-2); `curves` writes Elo against steps per run (with standard errors) as CSV
    to `--out`, and with `--figure PNG` draws them as small multiples, one panel per preset
    (`bnb.figures.ladder`; a chain's BC and RL runs on one axis of its steps), with a row of
    break-even panels when `--break-even DIR` names the per-snapshot `bnb crossover`
    results (`DIR/<run>/step_<n>/artifacts/crossover/...`).

    Presets (`--presets`): game presets, or the evaluation suite's head-to-head conditions
    by name (`deathmatch_10v10`, `elimination_1v1`, ...;
    `bnb.league.evaluation_settings.CONDITIONS`): the mode's preset at the condition's team
    size on the preset's own map, as `bnb evaluate` plays it, each its own fit.

    Targets (`rate`): `--target-stderr` for every player (the ladder's floor), a tighter
    `--decision-stderr` for `--decision` players, and `--contrast A:B` (a difference of two
    players' ratings) with `--contrast-stderr`. `--players-per-batch`: the distinct players
    a batch may hold (None: as many as its slots); `--session-players`: the players a
    session's engine holds; `--triangle-share`: a share of each batch for the least-played
    pairs among the decision players; `--hopeless`: the factor on the batches left past
    which targets all out of reach stop the session (inf: it plays its budget). The fit has
    no prior: a player without a finite maximum-likelihood rating is shown as a bound
    ("> 2400.0 (bound)") in `table` and has status "above" or "below" in `curves`, its
    calibrated_elo the bound and no stderr (`bnb.league.design`). A policy is played from
    a path where it was registered, else from `--runs-directory`'s `<run>/policies/`, with
    its weights' hash checked. `--seed`: the games' draws (None: a fresh one, recorded with
    each batch).
    """

    action: tyro.conf.Positional[Literal["create", "add", "import", "rate", "table", "curves"]]
    pool: tyro.conf.Positional[str]
    paths: tyro.conf.Positional[tuple[str, ...]] = ()
    name: str | None = None
    lineup: tuple[str, ...] = ()
    lineup_types: tuple[str, ...] = ()
    arena_scale: float | None = None
    deathmatch_score: Literal["win_loss", "kill_share"] | None = None
    presets: tuple[str, ...] = RATED_PRESETS
    runs: tuple[str, ...] = ()
    kinds: tuple[str, ...] = ("exponential_average",)
    figure: str | None = None
    break_even: str | None = None
    target_stderr: float = 45.0
    decision: tuple[str, ...] = ()
    decision_stderr: float = 20.0
    contrast: tuple[str, ...] = ()
    contrast_stderr: float = 25.0
    budget: int = 768
    blocks: int = 2
    block_environments: int = 96
    players_per_batch: int | None = None
    session_players: int = 16
    triangle_share: float = 0.0
    hopeless: float = 10.0
    device: str = "cuda"
    seed: int | None = None
    runs_directory: str | None = None
    out: str | None = None


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
    "average",
    "ship-card",
    "ratings",
    "head-to-head",
    "crossover",
    "evaluate",
    "pool",
    "figures",
    "queue",
    "unified",
)


@dataclasses.dataclass(frozen=True)
class Unified:
    """The unified evaluator (`bnb unified SPEC --out DIR`): every evaluation target of the
    spec (Elo per condition, break-even per mode, team-size win rates, sequential decisions)
    played by one continuously running engine per size bucket, each environment refilled
    with the most valuable game the moment its game ends (`bnb.league.unified`;
    docs/guides/unified-evaluation.md). A rerun with the same --out resumes."""

    spec: tyro.conf.Positional[pathlib.Path]
    out: pathlib.Path
    device: str = "cuda"
    # each bucket's environments (None: sized from the free device memory)
    environments: int | None = None
    # "auto": the buckets with work share the device when each gets 512 environments
    concurrent: str = "auto"
    # the share of the free device memory a batch may take
    memory_share: float = 0.7
    # the most hours to play
    hours: float = 24.0
    runs_directory: pathlib.Path | None = None
    # a rating pool the Elo targets' games are also appended to (`bnb pool table` reads it)
    pool: pathlib.Path | None = None
    # rating pools whose games the Elo targets start from (their conditions' games, this
    # game version, players matched by weights; pass them again on a resume)
    prior: tuple[pathlib.Path, ...] = ()
    # network players seated at once per build when the field has more (swapped by value)
    residents: int = 12


def unified(options: Unified) -> None:
    """`bnb unified`: load the spec and run it (`bnb.league.unified.run.run`)."""
    import torch

    from bnb.league.unified.run import load_spec, run

    spec = load_spec(options.spec, options.runs_directory)
    run(
        spec,
        options.out,
        torch.device(options.device),
        options.environments,
        options.concurrent,
        options.memory_share,
        hours=options.hours,
        pool=options.pool,
        prior=options.prior,
        residents=options.residents,
    )


def main() -> None:
    """The `bnb` command: `bnb train ...` trains, `bnb play ...` opens the play shell,
    `bnb runs ...` lists and shows runs and uploads checkpoints, `bnb average ...` builds
    averages of a run's weights from its per-update policies, `bnb ship-card ...` charts a
    card, `bnb ratings ...` and `bnb head-to-head ...` rate policies offline, `bnb crossover
    ...` finds a policy's break-even lineup, `bnb evaluate ...` decides between two
    policies, `bnb pool ...` keeps the shared rating pool, `bnb figures ...` draws what a
    policy's heads have learned (`bnb figures video ...` films a game, `bnb figures hero
    ...` the README clip, `bnb figures watch RUN` makes both while a run trains, `bnb
    figures results CHART ...` draws the write-up's results charts), `bnb queue ...` queues
    and runs GPU jobs."""
    os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")  # Triton 3.9's default, made explicit
    # The CUDA caching allocator grows its segments in place rather than caching fixed blocks:
    # at a real-scale PPO update (160 micro-batch rows) 0.73 GB less reserved memory and the
    # steady update about 1% faster (docs/rewrite/sections/compile-overhaul-notes.md). Read
    # at the first CUDA allocation, so it has to be set here, before any.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    command, arguments = (sys.argv[1:2] or [""])[0], sys.argv[2:]
    if command in (
        "train",
        "ratings",
        "head-to-head",
        "crossover",
        "evaluate",
        "pool",
        "figures",
        "unified",
    ):
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
    elif command == "average":
        average(tyro.cli(Average, args=arguments, prog="bnb average"))
    elif command == "queue":
        queue(queue_options(arguments))
    elif command == "ratings":
        ratings(tyro.cli(Ratings, args=arguments, prog="bnb ratings"))
    elif command == "head-to-head":
        head_to_head(tyro.cli(HeadToHead, args=arguments, prog="bnb head-to-head"))
    elif command == "crossover":
        crossover(tyro.cli(Crossover, args=arguments, prog="bnb crossover"))
    elif command == "unified":
        unified(tyro.cli(Unified, args=arguments, prog="bnb unified"))
    elif command == "evaluate":
        kind = (arguments[:1] or [""])[0]
        if kind == "decide":
            evaluate_decide(
                tyro.cli(EvaluateDecide, args=arguments[1:], prog="bnb evaluate decide")
            )
        elif kind == "design":
            evaluate_design(
                tyro.cli(EvaluateDesign, args=arguments[1:], prog="bnb evaluate design")
            )
        else:
            evaluate(tyro.cli(Evaluate, args=arguments, prog="bnb evaluate"))
    elif command == "pool":
        pool(tyro.cli(Pool, args=arguments, prog="bnb pool"))
    elif command == "figures":
        os.environ.setdefault("MPLBACKEND", "Agg")  # the figures go to files: no display
        kind = (arguments[:1] or [""])[0]
        if kind == "video":
            figures_video(tyro.cli(FiguresVideo, args=arguments[1:], prog="bnb figures video"))
        elif kind == "hero":
            figures_hero(tyro.cli(FiguresHero, args=arguments[1:], prog="bnb figures hero"))
        elif kind == "watch":
            figures_watch(tyro.cli(FiguresWatch, args=arguments[1:], prog="bnb figures watch"))
        elif kind == "results":
            figures_results(
                tyro.cli(FiguresResults, args=arguments[1:], prog="bnb figures results")
            )
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
    from bnb.runtime.train import LINEUP, TRAINING_MIXES, TrainConfig, arranged, mixed

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
            stop=bc.StopConfig()
            if options.bc_budget_steps is None
            else bc.StopConfig(budget_steps=options.bc_budget_steps),
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    try:  # the pools the lineup and mix settings name (`arranged`)
        config = arranged(with_settings(config, map(parse_setting, options.set)), options.preset)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    multiple = environment_multiple(config.pools)
    if config.environments % multiple:
        raise SystemExit(
            f"--environments {config.environments} does not split into the pools' shares: "
            f"use a multiple of {multiple}"
        )
    return config


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
            elif chain_entry is not None and chain_entry["stage"] == "rl":
                # a chain's RL stage with no BC stage: the RL stage's settings, as from a clone
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
            run.update(update_policies=config.writes_update_policies)
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
            run = Run.create(
                runs_directory,
                name,
                config,
                sys.argv,
                options.checkpoint_every,
                parent,
                config.writes_update_policies,
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
            pool: sum(row.get(f"reward/rate/{level}/{pool}", 0.0) for level in VALUE_LEVELS)
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
        League(trainer, run.directory / "league", state, ratings)
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

    from bnb.runs.chain import NO_CLONING, Chain, reinforcement_updates
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
    if chained.record["handover"] is None and chain_bc_budget(budget, original) == 0:
        chained.hand_over(0, NO_CLONING, None)  # RL from scratch (A24)
        print("no BC stage: RL from scratch", flush=True)
    if chained.record["handover"] is None:
        stage = chained.stage_run("bc")
        bc_run = Run.open(runs_directory, stage) if (runs_directory / stage).exists() else None
        if bc_run is None or bc_run.record["status"] != "finished":
            if bc_run is None:
                bc_options = stage_options(
                    base, "bc", name=stage, bc_budget_steps=chain_bc_budget(budget, original)
                )
                given_stage, entry = set(), chained.stage_record("bc", 0)
            else:
                # the stage's own budget (`chain_bc_budget`), or a --bc-set stop.budget_steps
                own = _requested(
                    stage_options(original, "bc", bc_budget_steps=chain_bc_budget(budget, original))
                )
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
        )  # no policy (no BC stage): a fresh RL run
        _train(rl_options, set(), stop, chained.stage_record("rl", handover["step"]))
    else:
        rl_options = Train(resume=stage, updates=updates, **passed)
        _train(rl_options, {"resume", "updates"}, stop)


def chain_bc_budget(combined: int, options: Train) -> int:
    """A chain's BC stage budget: the chain's --bc-budget-steps, else BC's own
    (`bnb.learn.bc.StopConfig.budget_steps`, 100M, fixed for every arm: decided 2026-11-14),
    at most the chain's combined budget. 0: no BC stage, the RL stage from scratch."""
    from bnb.learn.bc import StopConfig

    own = StopConfig().budget_steps if options.bc_budget_steps is None else options.bc_budget_steps
    return min(combined, own)


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
    """Both stages' resolved TrainConfigs as JSON (`--print-config`): BC's with its own
    budget (`chain_bc_budget`; None without a BC stage), RL's with the warm start's entropy
    coefficients, from the clone or from scratch (the model's settings, which a warm start
    takes from the clone, are the same run's)."""
    from bnb.runs.serialise import to_json

    budget = options.chain_budget_steps
    bc_budget = chain_bc_budget(budget, options)
    bc_config = (
        to_json(_requested(stage_options(options, "bc", bc_budget_steps=bc_budget)))
        if bc_budget
        else None
    )
    rl_options = stage_options(options, "rl")
    rl_config = warm_start_entropy(_requested(rl_options), rl_options.set)
    return dict(bc=bc_config, rl=to_json(rl_config))


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
    # the score stop's window of lane games against Imugi, once one is full
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
        f"{' (reached)' if row.get('bc/stop/plateau/reached') else ''}"
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

    if options.action == "mirror":
        from bnb.runs import mirror

        if options.run is None:
            raise SystemExit("bnb runs mirror needs a run or chain name")
        cache = None if options.cache_directory is None else pathlib.Path(options.cache_directory)
        try:
            mirror.mirror(
                runs_directory_of(options.runs_directory) if options.host is None else None,
                options.run,
                follow=options.follow,
                every=options.every,
                host=options.host,
                remote_runs=options.remote_runs,
                cache=cache,
            )
        except FileNotFoundError as error:
            raise SystemExit(str(error)) from error
        return
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
        elif options.action == "upload":
            from bnb.runs import upload

            if options.to != "wandb":
                raise SystemExit(f"--to {options.to!r}: the only target is wandb for now")
            print(upload.upload(run, options.at, options.league))
        else:
            raise SystemExit(
                f"bnb runs {options.action!r}: the actions are list, show, upload, dashboard, "
                f"mirror"
            )
    except FileNotFoundError as error:
        raise SystemExit(str(error)) from error


def average(options: Average) -> None:
    """`bnb average RUN --spec SPEC`: build and write the spec's averages; the player names
    to stdout, the progress and the checks to stderr."""
    from bnb.runs import averaging
    from bnb.runs.records import Run

    try:
        run = Run.open(runs_directory_of(options.runs_directory), options.run)
        spec = averaging.read_spec(options.spec)
        names, checks = averaging.build(
            run, spec, report=lambda line: print(line, file=sys.stderr, flush=True)
        )
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print("\n".join(names))
    differing = [
        f"{check.name} at step {check.step}" for check in checks if not check.within_rounding
    ]
    if differing:
        raise SystemExit(f"rebuilt training averages differ from the run's: {', '.join(differing)}")


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
    from bnb.runtime.train import lineup_indices

    found = [named_game(preset) for preset in presets]
    games = {(game.ship_set, game.ships_per_team) for game in found}
    if len(games) != 1:
        raise SystemExit(f"--lineup: the presets {list(presets)} play other ship sets or sizes")
    try:
        indices = lineup_indices(tuple(lineup), found[0])
    except ValueError as error:
        raise SystemExit(f"--lineup: {error}") from error
    return [
        dataclasses.replace(one, ship_type=indices) if one.ship_type is None else one
        for one in field
    ]


def named_game(name: str):
    """A rated game by name: a game preset (`PRESETS`), or a head-to-head condition of the
    evaluation suite (`bnb.league.evaluation_settings.CONDITIONS`), its mode's preset at its
    team size on the suite's map rule (`bnb.league.evaluation.game_of`)."""
    from bnb.game.api import PRESETS
    from bnb.league.evaluation_settings import BY_NAME, HEAD_TO_HEAD, EvaluationConfig

    if name in PRESETS:
        return PRESETS[name]
    condition = BY_NAME.get(name)
    if condition is None or condition.kind != HEAD_TO_HEAD:
        heads = [one for one, kept in BY_NAME.items() if kept.kind == HEAD_TO_HEAD]
        raise SystemExit(
            f"{name!r} is neither a game preset ({sorted(PRESETS)}) nor a head-to-head "
            f"condition ({heads}); break-even is measured by bnb crossover"
        )
    from bnb.league.evaluation import game_of

    return game_of(condition, EvaluationConfig())


def rated_game(
    preset: str, arena_scale: float | None, active_fields: tuple[int, int] | None = None
):
    """The preset's game at an evaluation-only arena scale (`Ratings.arena_scale`): its radius
    times the scale, at least the mode's minimum (`arena_radius_for`), and with an
    evaluation-only field-count range (`Ratings.active_fields`, low and high); None for
    either: the preset's own."""
    from bnb.contracts import ActiveFields
    from bnb.game.modes.rules import arena_radius_for

    game = named_game(preset)
    if arena_scale is not None:
        if not arena_scale > 0.0:
            raise SystemExit(f"--arena-scale {arena_scale}: a scale above 0")
        radius = arena_radius_for(game, game.ships_per_team, arena_scale)
        game = dataclasses.replace(game, arena_radius=radius)
    if active_fields is not None:
        try:
            game = dataclasses.replace(game, active_fields=ActiveFields(*active_fields))
        except ValueError as error:
            raise SystemExit(f"--active-fields {active_fields}: {error}") from error
    return game


def ratings(options: Ratings) -> None:
    """`bnb ratings`: the offline tournament and its leaderboard."""
    import torch

    from bnb.controllers.cheese import NAMES as CHEESE_NAMES
    from bnb.controllers.cheese import player_name
    from bnb.league import tournament
    from bnb.league.matches import ArenaConfig, Player, player
    from bnb.runs import provenance
    from bnb.runs.records import Run
    from bnb.runtime.lanes import IMUGI_PLAYER
    from bnb.runtime.opponents import LADDER, ladder_name, shown_name, stored_name

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
        if options.ladder or ladder_name(rung) == IMUGI_PLAYER
    ]
    if options.cheese:
        field += _players([player_name(name) for name in CHEESE_NAMES], runs_directory)
    field += [
        p for p in _players(options.with_, runs_directory) if p.name not in {f.name for f in field}
    ]
    if options.lineup:
        field = with_lineup(field, options.lineup, options.presets)
    try:
        allowed = tournament.allowed_pairs(
            [p.name for p in field],
            None if options.imugi_plays is None else tuple(map(stored_name, options.imugi_plays)),
        )
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
    contrasts = tuple(tuple(pair.split(":")) for pair in options.contrast)
    if any(len(pair) != 2 for pair in contrasts):
        raise SystemExit(f"--contrast A:B, not {list(options.contrast)}")
    try:  # the decision players' and contrasts' targets, and the batch's limits
        config = dataclasses.replace(
            config,
            decision=options.decision,
            decision_stderr=options.decision_stderr,
            contrasts=contrasts,
            contrast_stderr=options.contrast_stderr,
            players_per_batch=options.players_per_batch,
            hopeless=options.hopeless,
        )
        config.targets([p.name for p in field])
    except ValueError as error:
        raise SystemExit(str(error)) from error
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
                game=rated_game(preset, options.arena_scale, options.active_fields),
            )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    for row in sorted(rows, key=lambda row: (row["map"], tournament.rating_order(row))):
        print(
            f"{row['map']:16s} {shown_name(row['player']):60s} {tournament.describe_rating(row)}  "
            f"{row['games']:6d} games"
        )
    inputs = {player.name: player.policy for player in field if player.policy is not None}
    payload = dict(
        leaderboard=rows,
        config=dataclasses.asdict(config),
        imugi_plays=options.imugi_plays,
        deathmatch_score=options.deathmatch_score,
        arena_scale=options.arena_scale,
        active_fields=options.active_fields,
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
    """`bnb crossover`: the break-even lineup per preset, recorded batch by batch."""
    import torch

    from bnb.game.api import PRESETS, type_names
    from bnb.league import crossover as break_even
    from bnb.league.matches import ArenaConfig
    from bnb.runs import provenance, records

    _check_presets(options.presets)
    runs_directory = runs_directory_of(options.runs_directory)
    policy, opponent = _players((options.policy, options.opponent), runs_directory)
    try:
        config = break_even.CrossoverConfig(
            fixed_count=options.fixed_count,
            bracket=options.bracket,
            target_stderr=options.target_stderr,
            budget_games=options.budget_games,
            fixed_lineups=parse_lineups(options.lineups),
            lineup_games=options.lineup_games,
        )
        layout = ArenaConfig(
            blocks=options.blocks, block_environments=options.block_environments, seed=options.seed
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    # the owner: --out, else the run the policy is (or is inside), else the runs directory
    owner = runs_directory
    if options.out is not None:
        owner = pathlib.Path(options.out)
    elif policy.policy is not None:
        path = pathlib.Path(policy.policy).resolve()
        owner = next(
            (parent for parent in (path, *path.parents) if (parent / records.RECORD).is_file()),
            owner,
        )
    records_directory = owner / "artifacts" / "crossover"
    results = {}
    for number, preset in enumerate(options.presets, 1):
        base = PRESETS[preset]
        names = type_names(base.ship_set)
        unknown = [name for name in options.lineup_types if name not in names]
        if unknown:
            raise SystemExit(f"--lineup-types: {unknown} are not types of {base.ship_set}: {names}")
        game = break_even.break_even_game(base, config, options.map_rule)
        size = game.ships_per_team
        mode_config = dataclasses.replace(
            config, lineup_types=tuple(names.index(name) for name in options.lineup_types)
        )
        print(
            f"{preset} ({number} of {len(options.presets)}): {policy.name} against "
            f"{opponent.name}, the weaker side at {mode_config.fixed_count} ships and the "
            f"stronger from 1, {size}v{size} game, "
            f"arena radius {game.arena_radius:g} m ({options.map_rule})"
        )
        try:
            results[preset] = break_even.measure(
                policy,
                opponent,
                game,
                torch.device(options.device),
                layout,
                mode_config,
                records_directory / f"{preset}.batches.jsonl",
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
    print(f"{'preset':24s} {'ratio (held)':>16s} {'lineup':>12s} {'games':>6s}  stop")
    for preset, result in results.items():
        fitted = result["fit"]
        if fitted["status"] == "crossing":
            ratio = f"{result['ratio']:6.3f} +- {result['ratio_stderr']:5.3f}"
            lineup = break_even.lineup_name([round(count, 1) for count in result["lineup"]])
        else:
            ratio = break_even.describe(break_even.Fit(**fitted), result["fixed_count"])
            lineup = ""
        flag = "  SUSPICIOUS: check for a bug" if result["suspicious"] else ""
        print(
            f"{preset:24s} {ratio:>16s} {lineup:>12s} {result['games']:6d}  {result['stop']}{flag}"
        )
    inputs = {one.name: one.policy for one in (policy, opponent) if one.policy is not None}
    payload = dict(
        presets=results,
        config=dataclasses.asdict(config),
        map_rule=options.map_rule,
        lineup_types=list(options.lineup_types),
        layout=dataclasses.asdict(layout),
        records=str(records_directory),
    )
    print(provenance.write_result(owner, "crossover", payload, inputs))


def evaluate(options: Evaluate) -> None:
    """`bnb evaluate`: the decision suite (or a pilot, with --games)."""
    import torch

    from bnb.league import evaluation

    runs_directory = runs_directory_of(options.runs_directory)
    a, b, opponent = _players((options.a, options.b, options.settings.opponent), runs_directory)
    directory = pathlib.Path(options.out)
    try:
        suite = evaluation.Evaluation(
            directory,
            dict(A=a, B=b),
            opponent,
            options.settings,
            torch.device(options.device),
            own=options.play,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if options.games is not None:
        suite.fixed(options.games)
    else:
        suite.run(options.poll_seconds)
    evaluate_decide(EvaluateDecide(str(directory), peek=True))


def evaluate_decide(options: EvaluateDecide) -> None:
    """`bnb evaluate decide`: the scores and the test at a look now."""
    from bnb.league import evaluation

    directory = pathlib.Path(options.directory)
    try:
        held, config = evaluation.read_settings(directory)
        summary, result, tallies = evaluation.decide(directory, record=not options.peek)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    print(f"A: {held['players']['A']['name']}\nB: {held['players']['B']['name']}")
    for line in evaluation.table(config.scored(), summary, tallies):
        print(line)
    if result is None:
        print(evaluation.describe_summary(summary))
    else:
        print(("peek at " if options.peek else "") + evaluation.describe_look(result))


def evaluate_design(options: EvaluateDesign) -> None:
    """`bnb evaluate design`: the candidate designs from pilots."""
    import torch

    from bnb.league import evaluation_design

    loaded = []
    try:
        for name in options.directories:
            loaded.append(evaluation_design.pilot_models(pathlib.Path(name)))
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error)) from error
    config = loaded[0][0]
    models = evaluation_design.pooled_models([models for _, models, _ in loaded])
    print(f"{'condition':26s} {'var/game':>9s} {'s/game':>8s} {'draws':>6s}")
    for model in models:
        print(
            f"{model.condition.name:26s} {model.per_game:9.4f} {model.seconds:8.4f} "
            f"{model.draws:6.3f}"
        )
    per_information = evaluation_design.seconds_per_information(models)
    print(f"engine seconds per unit of information: {per_information:.1f}")
    observed = tuple(
        abs(summary.overall[0]) for _, _, summary in loaded if summary.overall is not None
    )
    differences = (0.0, config.delta, 2 * config.delta, *observed, *options.differences)
    rows, plans = evaluation_design.design_table(config, models, options.looks, differences)
    for count, found in plans.items():
        print(
            f"K={count}: fractions {[round(t, 3) for t in found.fractions]}, upper "
            f"{[round(u, 3) for u in found.upper]}, inner {[round(v, 3) for v in found.inner]}"
        )
    print(
        f"{'K':>2s} {'difference':>10s} {'A wins':>7s} {'B wins':>7s} {'tie':>7s} "
        f"{'E[hours]':>9s} {'max hours':>9s}"
    )
    for row in rows:
        print(
            f"{row.looks:2d} {row.difference:10.4f} {row.a_wins:7.4f} {row.b_wins:7.4f} "
            f"{row.tie:7.4f} {row.expected_hours:9.2f} {row.maximum_hours:9.2f}"
        )
    if options.simulations:
        generator = torch.Generator().manual_seed(options.seed)
        print(f"the procedure simulated, {options.simulations} times per difference:")
        for count in options.looks:
            for difference in differences:
                found = evaluation_design.simulate(
                    dataclasses.replace(config, looks=count),
                    models,
                    difference,
                    options.simulations,
                    generator,
                )
                print(
                    f"K={count} difference {difference:.4f}: A {found.a_wins:.3f}, B "
                    f"{found.b_wins:.3f}, tie {found.tie:.3f}, {found.hours:.2f} hours, "
                    f"{found.looks:.2f} looks, estimate at the stop {found.estimate_mean:+.4f} "
                    f"+- {found.estimate_stderr:.4f}"
                )


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


def figures_hero(options: FiguresHero) -> None:
    """`bnb figures hero`: games filmed, the clip picked, its GIF (`bnb.figures.hero`)."""
    if options.device == "cpu":
        hide_gpu()
    import torch

    from bnb.figures import hero
    from bnb.figures.command import policy_path

    torch.set_num_threads(options.threads)
    policy = policy_path(options.policy, runs_directory_of(options.runs_directory))
    result = hero.make(policy, options.preset, options.hero, pathlib.Path(options.out), options.gif)
    print(f"pick: {result['reason']}")
    if result["chosen"] is not None:
        print(f"chosen: {result['chosen']['name']}; gif: {result['gif']}")


def figures_results(options: FiguresResults) -> None:
    """`bnb figures results`: one results chart (`bnb.figures.results.command.results`)."""
    from bnb.figures.results.command import results

    results(options)


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


def pool(options: Pool) -> None:
    """`bnb pool`: the shared rating pool's actions (`Pool`)."""
    from bnb.league import rating_pool

    if options.action == "create":
        try:
            created = rating_pool.RatingPool.create(
                options.pool,
                options.lineup,
                options.arena_scale,
                options.deathmatch_score or "win_loss",
                options.lineup_types,
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        print(f"pool {created.root}: {created.settings}")
        return
    try:
        opened = rating_pool.RatingPool(options.pool)
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(f"{error} (bnb pool create {options.pool})") from error
    if options.action == "add":
        _pool_add(opened, options)
    elif options.action == "import":
        _pool_import(opened, options)
    elif options.action == "rate":
        _pool_rate(opened, options)
    else:
        _pool_table(opened, options)


def _pool_scoring(opened, preset: str, deathmatch_score: str | None) -> tuple[str, str]:
    """(how the fit scores a preset's games, the scoring a batch that names none was
    fitted under; `bnb.league.tournament.recorded_results`)."""
    from bnb.league.tournament import scoring_of, unnamed_scoring

    game = named_game(preset)
    chosen = deathmatch_score or opened.settings["deathmatch_score"]
    return scoring_of(game, chosen), unnamed_scoring(game)


def _pool_add(opened, options: Pool) -> None:
    from bnb.league.matches import imugi_play
    from bnb.league.matches import player as resolve
    from bnb.league.rating_pool import PoolPlayer, scheduled_policies

    if options.name is not None and len(options.paths) != 1:
        raise SystemExit("--name names one player: give one path")
    runs_directory = runs_directory_of(options.runs_directory)
    paths = list(options.paths)
    for run in options.runs:
        directory = pathlib.Path(run)
        if not directory.is_dir():
            directory = runs_directory / run
        try:
            paths += [str(path) for path in scheduled_policies(directory, options.kinds)]
        except ValueError as error:
            raise SystemExit(str(error)) from error
    for path in paths:
        try:
            if pathlib.Path(path).is_dir():
                added = opened.add_policy(path, options.name)
            else:
                one = resolve(path)
                if imugi_play(one) is None:
                    raise ValueError(f"{path} is neither a policy directory nor a scripted player")
                added = opened.register([PoolPlayer(one.name, one.name)])[0]
        except (ValueError, OSError) as error:
            raise SystemExit(f"{path}: {error}") from error
        print(f"{added.name}  {added.identity}")


def _pool_import(opened, options: Pool) -> None:
    from bnb.league.matches import player as resolve
    from bnb.league.rating_pool import import_records

    runs_directory = runs_directory_of(options.runs_directory)
    records = sorted(
        {
            record.resolve()
            for directory in options.paths
            for record in pathlib.Path(directory).rglob("*.batches.jsonl")
        }
    )
    if not records:
        raise SystemExit(f"no <preset>.batches.jsonl below {list(options.paths)}")

    def default_scoring(preset: str) -> str:
        return _pool_scoring(opened, preset, None)[1]

    report = import_records(
        opened, records, default_scoring, lambda name: resolve(name, runs_directory)
    )
    for record, found in report.items():
        left_out = "".join(f"; left out {name}: {why}" for name, why in found["left_out"].items())
        print(f"{record}: {found['batches']} batches added{left_out}")


def _pool_table(opened, options: Pool) -> None:
    import csv

    from bnb.league import goodness_of_fit, rating_pool, tournament

    for preset in options.presets:
        named_game(preset)
    every = []
    for preset in options.presets:
        scoring = _pool_scoring(opened, preset, options.deathmatch_score)
        rows, check, notes = rating_pool.table(opened, preset, scoring)
        every += rows
        if options.action == "curves":
            continue
        print(f"{preset} (scored by {scoring[0]})")
        for row in rows:
            print(
                f"  {row['player']:56s} {tournament.describe_rating(row)} {row['games']:7d} games"
            )
        for line in [*notes, *goodness_of_fit.describe(check)]:
            print(f"  {line}")
    if options.action == "curves":
        if options.out is None:
            raise SystemExit("bnb pool curves: --out FILE.csv")
        columns = [
            "preset",
            "run",
            "kind",
            "step",
            "player",
            "calibrated_elo",
            "stderr",
            "status",
            "games",
        ]
        with open(options.out, "w", newline="") as file:
            writer = csv.DictWriter(file, columns, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rating_pool.curves(every))
        print(f"{options.out}: {len(rating_pool.curves(every))} rows")
        if options.figure is not None:
            from bnb.figures import ladder

            points = None
            if options.break_even is not None:
                points = ladder.break_even_points(options.break_even)
            drawn = ladder.draw(rating_pool.curves(every), options.presets, options.figure, points)
            print(f"{drawn}: {len(options.presets)} conditions")


def _pool_player(one, runs_directory: pathlib.Path, lineup: tuple[str, ...], preset: str):
    """A pool player as this machine plays it (`bnb.league.matches.Player`), or None when
    none of its paths (nor the runs directory's) holds its weights. A policy is a player
    with weights (a `sha256:` identity), named by its run's step directory or registered
    under its own name (an average's `<run>/average/<name>/step_<n>`, which has no run);
    any other is a scripted player, played by its name."""
    from bnb.league.matches import Player
    from bnb.league.matches import player as resolve
    from bnb.league.rating_pool import policy_identity
    from bnb.runs.checkpoint import CheckpointError

    if not one.identity.startswith("sha256:"):
        try:
            found = resolve(one.name)
        except ValueError:  # registered elsewhere under a name this checkout cannot play
            return None
    else:
        candidates = list(one.paths)
        if one.run is not None:
            candidates.append(
                str(runs_directory / one.run / "policies" / f"step_{one.step}" / one.kind)
            )
        found = None
        for path in candidates:
            try:
                if pathlib.Path(path).is_dir() and policy_identity(path) == one.identity:
                    found = Player(one.name, policy=pathlib.Path(path))
                    break
            except CheckpointError, OSError, KeyError:
                continue
        if found is None:
            return None
    return with_lineup([found], lineup, (preset,))[0] if lineup else found


def _pool_rate(opened, options: Pool) -> None:
    import secrets

    import torch

    from bnb.game.api import type_names
    from bnb.league import crossover, rating_pool, tournament
    from bnb.league.matches import Arena, ArenaConfig, default_lanes, network_groups

    for preset in options.presets:
        named_game(preset)
    contrasts = tuple(tuple(pair.split(":")) for pair in options.contrast)
    if any(len(pair) != 2 for pair in contrasts):
        raise SystemExit(f"--contrast A:B, not {list(options.contrast)}")
    seed = secrets.randbits(31) if options.seed is None else options.seed
    torch.manual_seed(seed)  # the games' draws
    runs_directory = runs_directory_of(options.runs_directory)
    lineup = tuple(opened.settings.get("lineup", ()))
    lineup_types = tuple(opened.settings.get("lineup_types", ()))
    try:
        config = tournament.TournamentConfig(
            arena=ArenaConfig(blocks=options.blocks, block_environments=options.block_environments),
            target_stderr=options.target_stderr,
            budget=options.budget,
            decision=options.decision,
            decision_stderr=options.decision_stderr,
            contrasts=contrasts,
            contrast_stderr=options.contrast_stderr,
            players_per_batch=options.players_per_batch,
            triangle_share=options.triangle_share,
            hopeless=options.hopeless,
        )
    except ValueError as error:
        raise SystemExit(str(error)) from error
    device = torch.device(options.device)
    for preset in options.presets:
        game = rated_game(preset, opened.settings.get("arena_scale"))
        arenas = []
        names = type_names(game.ship_set)
        unknown = [name for name in lineup_types if name not in names]
        if unknown:
            raise SystemExit(f"the pool's lineup types {unknown} are not types of {names}")
        types = tuple(names.index(name) for name in lineup_types)

        def play_batch(field, matchups, game=game, arenas=arenas, preset=preset, types=types):
            if not arenas:
                lanes = default_lanes(network_groups(field), 2 * config.arena.blocks)
                arenas.append(
                    Arena(field, game, device, config.arena, lanes, game_lineups=bool(types))
                )
            arena = arenas[0]
            drawn = None
            if types:  # each game's lineup, keyed by the session's seed, the preset, the batch
                drawn = crossover.lineup_draw(
                    seed,
                    arena.batch,
                    config.arena.games,
                    game.ships_per_team,
                    types,
                    stream=f"pool_lineup/{preset}",
                ).to(device)
            return tournament.outcomes(arena, matchups, drawn)

        try:
            rating_pool.rate(
                opened,
                preset,
                lambda one, preset=preset: _pool_player(one, runs_directory, lineup, preset),
                play_batch,
                config,
                _pool_scoring(opened, preset, options.deathmatch_score),
                options.session_players,
                extra=dict(seed=seed),
            )
        except ValueError as error:
            raise SystemExit(str(error)) from error
        arenas.clear()
