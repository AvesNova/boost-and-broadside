"""The decision suite, `bnb evaluate` (docs/rewrite/evaluation-suite.md): two players, A and
B (normally two arms' final checkpoints, their exponential-average weights), compared by
win probability over a fixed set of conditions, with a group-sequential test
(`bnb.league.sequential`) deciding A wins, B wins or a tie.

**Conditions** (`CONDITIONS`): head-to-head A against B in each mode at 5v5 (the mode's
preset map), deathmatch and elimination at 1v1 and 2v2, and every mode at 10v10; and the
break-even condition per mode, each player against `opponent_count` (20) scripted ships
with its own count varied (`bnb.league.crossover`). Every condition plays its preset's own
arena radius at every size (`map_rule` "preset": every map fixed-size, the author's rule
of 2026-11-15); the density rule's map (`bnb.game.api.arena_radius_for`) is the map-size
arm's opt-in (`map_rule` "density"). Every game is played with fog of war
(the engine's default), the networks sampling at temperature 1, each game's lineup drawn
slot by slot from `lineup_types` with equal chance and flown by both teams (`Arena`'s
game_lineups, keyed by the seed, the condition and the batch), and the seats alternating
from game to game. A batch's games are keyed by its number (`Arena`'s first_batch), each
condition's numbers continuing past its recorded batches.

**Score.** A head-to-head condition's is A's mean score (a win 1, a draw 1/2) less one
half; its variance the games' sample variance over their number. The break-even
condition's is 1/2 (p_A(N*_B) - 1/2) - 1/2 (p_B(N*_A) - 1/2), from each player's
logistic fit of score on log(count) (p_X its fitted curve, N*_X where it crosses one
half); its variance by the delta method from the two fits' covariances (the fits are
taken as independent). A mode's score is the mean of its conditions' and the overall
score the mean of the three modes' (`weights`: fixed, whatever the games). The overall
variance is the weighted sum of the conditions' variances, each estimate unbiased however
many games it has, so the cost-aware allocation below moves no weight.

**Gate** (`gate`): a mode where a player's score is at least `gate_stderrs` standard errors
below zero; a winner the gate catches is held to a tie (arms-todo, 2026-11-15).

**Sequential test** (`look`): at a look, Z = D / SE and the information fraction
t = SE^-2 / I_max, with I_max from the planned design (`bnb.league.sequential.plan`); the
boundaries are spent at the recorded looks' actual fractions, and a look is the last
when t ≥ 1 or the budget (`budget_hours` of measured engine time, every writer's) is
spent.

**Allocation** (`targets`): the next planned look needs variance 1 / (t_next I_max); batches
are added, one at a time, to the condition whose next batch most reduces the overall
variance per second of engine time (its per-game variance from its games, shrunk toward
1/4 by one pseudo-game for a head-to-head condition, and its measured seconds per game),
until the variance reaches it. A condition plays `initial_batches` first (and the
break-even condition until both fits cross, at most `initial_batch_limit` batches), so
every condition has a variance and a cost.

**Records** (one directory): `settings.json` (the players, their identities and the
settings; written by the first run and checked by every later one), `batches/<writer>.jsonl`
(every batch: its condition, number, blocks with full outcomes, seconds and games) and
`looks/<writer>.jsonl` (every look: its fraction, score, boundaries and decision). Each
process writes only its own files (`bnb.league.rating_pool`'s scheme); a reader merges
them, keeping the first of any repeated (condition, batch) or look number. So each
condition may run as its own process, every process taking a look when the planned
information is reached.
"""

import dataclasses
import datetime
import json
import math
import pathlib
import time
from collections.abc import Callable

import torch

from bnb.contracts import GameConfig
from bnb.game import VERSION as GAME_VERSION
from bnb.runtime.opponents import shown_name, stored_name

from . import crossover, sequential
from .evaluation_settings import (
    BREAK_EVEN,
    HEAD_TO_HEAD,
    MODES,
    Condition,
    EvaluationConfig,
)
from .matches import Arena, Player, network_groups
from .rating_pool import _writer, read_lines
from .tournament import _append, identity

SIDES = ("A", "B")
# The games a head-to-head condition's per-game variance is shrunk by, toward 1/4 (an even
# game with no draws), for the allocation only.
PSEUDO_GAMES = 1.0


def weights(conditions: list[Condition]) -> dict[str, float]:
    """Each condition's weight in the overall score: a third per mode, shared equally among
    the mode's conditions (a mode without conditions drops out, the rest share)."""
    modes = [mode for mode in MODES if any(c.mode == mode for c in conditions)]
    found = {}
    for condition in conditions:
        within = sum(1 for other in conditions if other.mode == condition.mode)
        found[condition.name] = 1.0 / (len(modes) * within)
    return found


def game_of(condition: Condition, config: EvaluationConfig) -> GameConfig:
    """A condition's game: its mode's preset at its size (break-even's, the larger fleet's,
    `crossover.break_even_game`), on the arena of `config.map_rule`
    (`crossover.sized_game`: the preset's own radius at every size by default)."""
    from bnb.game.api import PRESETS

    base = PRESETS[config.presets[MODES.index(condition.mode)]]
    if condition.kind == BREAK_EVEN:
        return crossover.break_even_game(base, config.crossover_config(), config.map_rule)
    return crossover.sized_game(base, condition.ships_per_team, config.map_rule)


# ---- tallies and estimates


@dataclasses.dataclass
class Tally:
    """One condition's games: head-to-head outcomes (A's wins, draws, losses); break-even's
    per side {count: [score summed, games]}; games and engine seconds in all, and the
    seconds and games that measure the cost (an engine's first batch, which compiles, left
    out where there are others)."""

    outcomes: list[float] = dataclasses.field(default_factory=lambda: [0.0, 0.0, 0.0])
    sides: dict[str, dict[int, list[float]]] = dataclasses.field(
        default_factory=lambda: {side: {} for side in SIDES}
    )
    games: int = 0
    batches: int = 0
    seconds: float = 0.0
    steady_seconds: float = 0.0
    steady_games: int = 0
    first_seconds: float = 0.0
    first_games: int = 0

    def add(self, batch: dict) -> None:
        """Add one recorded batch."""
        for block in batch["blocks"]:
            if "side" in block:
                held = self.sides[block["side"]].setdefault(int(block["count"]), [0.0, 0.0])
                held[0] += block["wins"] + 0.5 * block["draws"]
                held[1] += block["wins"] + block["draws"] + block["losses"]
            else:
                self.outcomes[0] += block["wins"]
                self.outcomes[1] += block["draws"]
                self.outcomes[2] += block["losses"]
        self.games += batch["games"]
        self.batches += 1
        self.seconds += batch["seconds"]
        if batch.get("first", False):
            self.first_seconds += batch["seconds"]
            self.first_games += batch["games"]
        else:
            self.steady_seconds += batch["seconds"]
            self.steady_games += batch["games"]

    def seconds_per_game(self) -> float | None:
        """Measured engine seconds per game: steady batches' where there are any."""
        if self.steady_games:
            return self.steady_seconds / self.steady_games
        if self.first_games:
            return self.first_seconds / self.first_games
        return None


@dataclasses.dataclass(frozen=True)
class Estimate:
    """A condition's score (A's side, less one half), its variance and its per-game variance
    for the allocation (the variance times the games; shrunk for head-to-head)."""

    score: float
    variance: float
    per_game: float


def head_to_head_estimate(outcomes: list[float]) -> Estimate | None:
    """A's mean score less one half from (wins, draws, losses), and the sample variance of a
    game's score over the games; None without games."""
    won, drawn, lost = outcomes
    games = won + drawn + lost
    if games == 0:
        return None
    mean = (won + 0.5 * drawn) / games
    spread = max((won + 0.25 * drawn) / games - mean * mean, 0.0)
    shrunk = (games * spread + PSEUDO_GAMES * 0.25) / (games + PSEUDO_GAMES)
    return Estimate(mean - 0.5, spread / games, shrunk)


def _curve_parameters(found: crossover.Fit) -> tuple[torch.Tensor, torch.Tensor]:
    """(intercept, slope) and their covariance in log(count), float64."""
    point = torch.tensor([found.intercept, found.slope], dtype=torch.float64)
    return point, torch.tensor(found.covariance, dtype=torch.float64)


def break_even_score(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """1/2 (p_A(N*_B) - 1/2) - 1/2 (p_B(N*_A) - 1/2) from each curve's (intercept, slope) in
    log(count): N*_X = exp(-intercept / slope)."""
    log_a, log_b = -a[0] / a[1], -b[0] / b[1]
    a_at_b = torch.sigmoid(a[0] + a[1] * log_b)
    b_at_a = torch.sigmoid(b[0] + b[1] * log_a)
    return 0.5 * (a_at_b - 0.5) - 0.5 * (b_at_a - 0.5)


def break_even_estimate(tally: Tally) -> Estimate | None:
    """The break-even score from the two sides' fits, its delta-method variance, and its
    per-game variance (the variance times the games); None unless both fits cross."""
    fits = {side: crossover.fit(tally.sides[side]) for side in SIDES}
    if any(found.status != "crossing" for found in fits.values()):
        return None
    (a, a_covariance), (b, b_covariance) = (_curve_parameters(fits[side]) for side in SIDES)
    a.requires_grad_(True)
    b.requires_grad_(True)
    score = break_even_score(a, b)
    gradient_a, gradient_b = torch.autograd.grad(score, (a, b))
    variance = float(
        gradient_a @ a_covariance @ gradient_a + gradient_b @ b_covariance @ gradient_b
    )
    return Estimate(float(score.detach()), variance, variance * tally.games)


def estimate(condition: Condition, tally: Tally) -> Estimate | None:
    """A condition's estimate (`head_to_head_estimate`, `break_even_estimate`)."""
    if condition.kind == BREAK_EVEN:
        return break_even_estimate(tally)
    return head_to_head_estimate(tally.outcomes)


@dataclasses.dataclass(frozen=True)
class Summary:
    """Every scored condition's estimate (None without one), each mode's score and standard
    error, and the overall's (None until every condition has an estimate)."""

    conditions: dict[str, Estimate | None]
    modes: dict[str, tuple[float, float] | None]
    overall: tuple[float, float] | None

    def information(self) -> float:
        """The overall score's information, 1 / its variance (0 without it)."""
        if self.overall is None or self.overall[1] <= 0.0:
            return 0.0
        return 1.0 / self.overall[1] ** 2


def summarise(conditions: list[Condition], tallies: dict[str, Tally]) -> Summary:
    """The conditions' estimates (`estimate`) combined (`combine`)."""
    found = {c.name: estimate(c, tallies.get(c.name, Tally())) for c in conditions}
    return combine(conditions, found)


def combine(conditions: list[Condition], found: dict[str, Estimate | None]) -> Summary:
    """The conditions' estimates combined with the fixed weights (`weights`)."""
    weight = weights(conditions)
    modes = {}
    for mode in MODES:
        members = [c for c in conditions if c.mode == mode]
        if not members:
            continue
        if any(found[c.name] is None for c in members):
            modes[mode] = None
            continue
        modes[mode] = (
            sum(found[c.name].score for c in members) / len(members),
            math.sqrt(sum(found[c.name].variance for c in members)) / len(members),
        )
    overall = None
    if all(value is not None for value in found.values()):
        overall = (
            sum(weight[name] * value.score for name, value in found.items()),
            math.sqrt(sum(weight[name] ** 2 * value.variance for name, value in found.items())),
        )
    return Summary(found, modes, overall)


def gate(summary: Summary, stderrs: float) -> dict[str, list[str]]:
    """{"A": modes where A is significantly worse, "B": where B is}: a mode's score at least
    `stderrs` standard errors below (A) or above (B) zero."""
    worse = {side: [] for side in SIDES}
    for mode, value in summary.modes.items():
        if value is None:
            continue
        score, stderr = value
        if score <= -stderrs * stderr:
            worse["A"].append(mode)
        if score >= stderrs * stderr:
            worse["B"].append(mode)
    return worse


# ---- allocation


def targets(
    conditions: list[Condition],
    tallies: dict[str, Tally],
    summary: Summary,
    variance_target: float,
    batch_games: dict[str, int],
    default_seconds: float = 1.0,
    limit: int = 100_000,
) -> dict[str, int]:
    """Each condition's games for the overall variance to reach `variance_target`: from the
    games played, batches added one at a time where the next batch most reduces the
    overall variance per engine second (see the module docstring). Every condition needs
    an estimate; a condition without a measured cost takes `default_seconds` per game."""
    weight = weights(conditions)
    games = {c.name: tallies[c.name].games for c in conditions}
    per_game = {c.name: summary.conditions[c.name].per_game for c in conditions}
    seconds = {c.name: tallies[c.name].seconds_per_game() or default_seconds for c in conditions}

    def variance_of(name: str, count: float) -> float:
        return weight[name] ** 2 * per_game[name] / max(count, 1.0)

    total = sum(variance_of(name, games[name]) for name in games)
    for _ in range(limit):
        if total <= variance_target:
            break
        best, gain = None, -1.0
        for name in games:
            added = batch_games[name]
            reduction = variance_of(name, games[name]) - variance_of(name, games[name] + added)
            rate = reduction / (added * seconds[name])
            if rate > gain:
                best, gain = name, rate
        total -= variance_of(best, games[best]) - variance_of(best, games[best] + batch_games[best])
        games[best] += batch_games[best]
    return games


def cost_per_information(
    conditions: list[Condition], per_game: dict[str, float], seconds: dict[str, float]
) -> float:
    """The engine seconds per unit of the overall score's information with the cost-optimal
    allocation (games in proportion to w_c sigma_c / sqrt(cost_c)): (sum_c w_c sigma_c
    sqrt(cost_c))^2."""
    weight = weights(conditions)
    return (
        sum(weight[c.name] * math.sqrt(per_game[c.name] * seconds[c.name]) for c in conditions) ** 2
    )


# ---- records


def read_batches(directory: pathlib.Path) -> tuple[list[dict], int, int]:
    """Every writer's batches (the first of each repeated (condition, batch)), the damaged
    lines and the repeats left out."""
    lines = read_lines(directory / "batches")
    seen, kept, repeats = set(), [], 0
    for entry in lines.entries:
        key = (entry["condition"], entry["batch"])
        if key in seen:
            repeats += 1
            continue
        seen.add(key)
        kept.append(entry)
    return kept, lines.damaged, repeats


def read_looks(directory: pathlib.Path) -> list[dict]:
    """Every look, in look order (the earliest of each number)."""
    by_number = {}
    for entry in sorted(read_lines(directory / "looks").entries, key=lambda e: e["time"]):
        by_number.setdefault(entry["look"], entry)
    return [by_number[number] for number in sorted(by_number)]


def tallies_of(batches: list[dict]) -> dict[str, Tally]:
    """Each condition's tally."""
    found: dict[str, Tally] = {}
    for batch in batches:
        found.setdefault(batch["condition"], Tally()).add(batch)
    return found


def settings_entry(players: dict[str, Player], config: EvaluationConfig) -> dict:
    """What settings.json holds: the game version, each player's name, policy and identity
    (`bnb.league.tournament.identity`), and the settings."""
    return json.loads(
        json.dumps(
            dict(
                game_version=GAME_VERSION,
                players={
                    side: dict(
                        name=one.name,
                        policy=None if one.policy is None else str(one.policy),
                        identity=identity(one),
                    )
                    for side, one in players.items()
                },
                # recorded as the identity, so settings written before the rename still match
                config=dict(dataclasses.asdict(config), opponent=stored_name(config.opponent)),
            )
        )
    )


def open_directory(directory: pathlib.Path, entry: dict) -> None:
    """Write settings.json, or check that the one there is the same (the players by identity,
    the game version and the settings). Raises ValueError otherwise."""
    path = directory / "settings.json"
    if not path.exists():
        directory.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(entry, indent=1) + "\n")
        temporary.replace(path)
        return
    held = json.loads(path.read_text())
    for key in ("game_version", "config"):
        if held[key] != entry[key]:
            raise ValueError(f"{path}: another {key} ({held[key]} there, {entry[key]} here)")
    for side in SIDES:
        if held["players"][side]["identity"] != entry["players"][side]["identity"]:
            raise ValueError(f"{path}: player {side} is another policy than the one recorded")


def read_settings(directory: pathlib.Path) -> tuple[dict, EvaluationConfig]:
    """settings.json and its settings."""
    path = directory / "settings.json"
    if not path.exists():
        raise FileNotFoundError(f"{path}: no evaluation here")
    held = json.loads(path.read_text())
    config = held["config"]
    tuples = {
        field.name: tuple(config[field.name])
        for field in dataclasses.fields(EvaluationConfig)
        if isinstance(config.get(field.name), list)
    }
    # the opponent is recorded as its identity (`settings_entry`); the settings name it
    opponent = shown_name(config["opponent"])
    return held, EvaluationConfig(**{**config, **tuples, "opponent": opponent})


# ---- the test


@dataclasses.dataclass(frozen=True)
class Look:
    """A look's result: its number (from 0), information fraction, score and standard error,
    Z, boundaries (upper, inner), whether it is the last, the test's decision, the gate's
    findings, and the decision after the gate ("A", "B", "tie" or "continue")."""

    look: int
    fraction: float
    score: float
    stderr: float
    z: float
    upper: float
    inner: float
    last: bool
    test: str
    gate: dict[str, list[str]]
    decision: str

    def as_dict(self) -> dict:
        """The look as a plain dict (for the record)."""
        return dataclasses.asdict(self)


def look_of(entry: dict) -> Look:
    """A recorded look."""
    return Look(**{field.name: entry[field.name] for field in dataclasses.fields(Look)})


def maximum_information(config: EvaluationConfig) -> float:
    """I_max: the planned last look's information ((eta_delta / delta)^2)."""
    drift, _ = sequential.plan(config.design())
    return (drift / config.delta) ** 2


def look(
    config: EvaluationConfig,
    summary: Summary,
    earlier: list[float],
    spent_seconds: float,
    maximum: float | None = None,
) -> Look:
    """The test at a look now, after looks at the `earlier` fractions (see the module
    docstring); the gate holds a winner worse in some mode to a tie."""
    if summary.overall is None:
        raise ValueError("a look needs every condition's estimate")
    maximum = maximum_information(config) if maximum is None else maximum
    drift = config.delta * math.sqrt(maximum)
    score, stderr = summary.overall
    fraction = summary.information() / maximum
    if earlier and fraction <= earlier[-1]:
        raise ValueError(
            f"no new information since the last look (fraction {fraction:.3f}, "
            f"then {earlier[-1]:.3f})"
        )
    last = fraction >= 1.0 or spent_seconds >= config.budget_hours * 3600.0
    found = sequential.boundaries(config.design(), [*earlier, fraction], drift, final=last)
    z = score / stderr
    test = found.decision(len(earlier), z)
    worse = gate(summary, config.gate_stderrs)
    decision = test
    if test in SIDES and worse[test]:
        decision = "tie"
    return Look(
        len(earlier),
        fraction,
        score,
        stderr,
        z,
        found.upper[-1],
        found.inner[-1],
        last,
        test,
        worse,
        decision,
    )


def next_variance(config: EvaluationConfig, looks_taken: int, maximum: float) -> float:
    """The overall variance the next planned look needs: 1 / (its fraction I_max); past the
    planned looks, the last's."""
    fraction = min(looks_taken + 1, config.looks) / config.looks
    return 1.0 / (fraction * maximum)


# ---- playing


def lineups(
    config: EvaluationConfig, condition: Condition, batch: int, games: int, types
) -> torch.Tensor:
    """(games, ships per team of the condition's game) the batch's lineups, keyed by the
    seed, the condition and the batch (`crossover.lineup_draw`)."""
    return crossover.lineup_draw(
        config.seed,
        batch,
        games,
        game_of(condition, config).ships_per_team,
        types,
        stream=f"evaluation_lineup/{condition.name}",
    )


def _lanes(field: list[Player], condition: Condition, blocks: int) -> tuple[int, ...]:
    """Each network group's lanes: its slots in a batch (head-to-head: A's and B's in every
    block; break-even: A's in half the blocks, B's in the other half)."""
    groups = network_groups(field)
    count = 1 + max((group for group in groups if group is not None), default=-1)
    per_player = blocks if condition.kind == HEAD_TO_HEAD else blocks // 2
    held = [0] * count
    for group in groups[:2]:
        if group is not None:
            held[group] += per_player
    return tuple(max(lanes, 1) for lanes in held)


class Engine:
    """One condition's engine (`Arena`): head-to-head blocks A against B; break-even's first
    half of blocks A against the opponent, the second B, at counts from each side's fit
    (`crossover.allocate`)."""

    def __init__(
        self,
        condition: Condition,
        players: dict[str, Player],
        opponent: Player,
        config: EvaluationConfig,
        device: torch.device,
        first_batch: int,
        compiled: bool = True,
    ) -> None:
        from bnb.game.api import type_names

        self.condition, self.config = condition, config
        self.game = game_of(condition, config)
        names = type_names(self.game.ship_set)
        unknown = [name for name in config.lineup_types if name not in names]
        if unknown:
            raise ValueError(f"lineup types {unknown} are not types of {self.game.ship_set}")
        self.types = tuple(names.index(name) for name in config.lineup_types)
        self.layout = dataclasses.replace(config.layout(condition), compiled=compiled)
        field = [players["A"], players["B"]]
        if condition.kind == BREAK_EVEN:
            field.append(opponent)
        self.arena = Arena(
            field,
            self.game,
            device,
            self.layout,
            lanes=_lanes(field, condition, self.layout.blocks),
            benching=condition.kind == BREAK_EVEN,
            game_lineups=True,
            first_batch=first_batch,
        )
        self.device = device

    def play(self, tally: Tally) -> tuple[int, list[dict]]:
        """One batch: (its number, its blocks' outcomes)."""
        number = self.arena.batch
        drawn = lineups(self.config, self.condition, number, self.layout.games, self.types)
        blocks = self.layout.blocks
        if self.condition.kind == HEAD_TO_HEAD:
            results = self.arena.play([(0, 1)] * blocks, lineups=drawn.to(self.device))
            return number, [
                dict(wins=int(won), draws=int(tied), losses=int(lost))
                for won, tied, lost in results.tolist()
            ]
        settings = self.config.crossover_config()
        half = blocks // 2
        counts = {
            side: crossover.allocate(
                crossover.fit(tally.sides[side]),
                tally.sides[side],
                settings,
                half,
                self.layout.block_environments,
            )
            for side in SIDES
        }
        matchups = [(0, 2)] * half + [(1, 2)] * half
        active = [(count, settings.opponent_count) for side in SIDES for count in counts[side]]
        results = self.arena.play(matchups, active, drawn.to(self.device))
        sides = [side for side in SIDES for _ in range(half)]
        flat = [count for side in SIDES for count in counts[side]]
        return number, [
            dict(side=side, count=count, wins=int(won), draws=int(tied), losses=int(lost))
            for side, count, (won, tied, lost) in zip(sides, flat, results.tolist(), strict=True)
        ]


def clock(seconds_from_now: float) -> str:
    """The local clock time `seconds_from_now` ahead, HH:MM."""
    return crossover.clock_time(seconds_from_now)


def describe_summary(summary: Summary) -> str:
    """The overall score in words."""
    if summary.overall is None:
        waiting = [name for name, value in summary.conditions.items() if value is None]
        return f"waiting for {', '.join(waiting)}"
    score, stderr = summary.overall
    return f"overall {score:+.4f} +- {stderr:.4f}"


class Evaluation:
    """One process's part of an evaluation in `directory`: the conditions it plays
    (`own`), batch by batch, appending each batch to its own record; see the module
    docstring for the allocation and the looks. compiled: the engines' rollouts compiled
    (else eager, for the CPU)."""

    def __init__(
        self,
        directory: pathlib.Path,
        players: dict[str, Player],
        opponent: Player,
        config: EvaluationConfig,
        device: torch.device,
        own: tuple[str, ...] | None = None,
        progress: Callable[[str], None] = print,
        writer: str | None = None,
        compiled: bool = True,
    ) -> None:
        self.compiled = compiled
        self.directory, self.players, self.opponent = directory, players, opponent
        self.config, self.device, self.progress = config, device, progress
        self.conditions = config.scored()
        own = tuple(c.name for c in self.conditions) if own is None else own
        missing = [name for name in own if name not in config.conditions]
        if missing:
            raise ValueError(f"conditions {missing} are not among the scored {config.conditions}")
        self.own = [c for c in self.conditions if c.name in own]
        self.writer = writer or _writer()
        open_directory(directory, settings_entry(players, config))
        self.engine: Engine | None = None
        self.fresh = False  # whether the engine has played no batch yet (it compiles)
        self.latest: dict[str, int] = {}
        self.maximum = maximum_information(config)
        self.started = time.perf_counter()

    def state(self) -> tuple[dict[str, Tally], list[dict]]:
        """The pooled tallies and the looks; notes each condition's highest batch number
        (`latest`), which the next engine continues past."""
        batches, damaged, repeats = read_batches(self.directory)
        if damaged or repeats:
            self.progress(f"records: {damaged} damaged lines, {repeats} repeated batches left out")
        for batch in batches:
            name = batch["condition"]
            self.latest[name] = max(self.latest.get(name, -1), batch["batch"])
        return tallies_of(batches), read_looks(self.directory)

    def _engine(self, condition: Condition) -> Engine:
        if self.engine is None or self.engine.condition != condition:
            if self.engine is not None:
                del self.engine
                self.engine = None
                if self.device.type == "cuda":
                    torch.cuda.empty_cache()
            self.engine = Engine(
                condition,
                self.players,
                self.opponent,
                self.config,
                self.device,
                first_batch=1 + self.latest.get(condition.name, -1),
                compiled=self.compiled,
            )
            self.fresh = True
        return self.engine

    def play_batch(self, condition: Condition, tally: Tally) -> dict:
        """Play and record one batch of a condition; returns the record's line."""
        engine = self._engine(condition)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        number, blocks = engine.play(tally)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        seconds = time.perf_counter() - started
        line = dict(
            condition=condition.name,
            batch=number,
            writer=self.writer,
            game_version=GAME_VERSION,
            games=engine.layout.games,
            seconds=seconds,
            first=self.fresh,
            blocks=blocks,
            time=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        )
        self.fresh = False
        _append(self.directory / "batches" / f"{self.writer}.jsonl", line)
        tally.add(line)
        return line

    def batch_games(self) -> dict[str, int]:
        """Each condition's games per batch."""
        return {c.name: self.config.layout(c).games for c in self.conditions}

    def _waiting(self, condition: Condition, tally: Tally) -> bool:
        """Whether a condition still plays its initial batches (break-even until both fits
        cross, at most `initial_batch_limit` batches)."""
        if tally.batches < self.config.initial_batches:
            return True
        if condition.kind == BREAK_EVEN and break_even_estimate(tally) is None:
            if tally.batches >= self.config.initial_batch_limit:
                raise RuntimeError(
                    f"{condition.name}: no crossing for both players after {tally.batches} "
                    "batches (a fit separated or falling; see the records)"
                )
            return True
        return False

    def fixed(self, games: int) -> None:
        """Play the own conditions until each has `games` games (a pilot; no looks)."""
        tallies, _ = self.state()
        for condition in self.own:
            tally = tallies.setdefault(condition.name, Tally())
            while tally.games < games:
                line = self.play_batch(condition, tally)
                rate = line["games"] / line["seconds"]
                left = (games - tally.games) / rate
                self.progress(
                    f"{condition.name} batch {line['batch']}: {tally.games} games, "
                    f"{rate:.2f} games/s{' (with compiling)' if line['first'] else ''}"
                    + (f"; ETA {clock(left)}" if left > 0 else "")
                )

    def run(self, poll_seconds: float = 30.0, sleep=time.sleep) -> Look | None:
        """Play toward the next planned look, take it when its information is reached (or the
        budget is spent), and go on until a look decides; returns the deciding look. With
        nothing of its own to play while another process's conditions are short, waits."""
        while True:
            tallies, looks = self.state()
            if looks and looks[-1]["decision"] != "continue":
                self.progress(f"decided at look {looks[-1]['look']}: {looks[-1]['decision']}")
                return look_of(looks[-1])
            for condition in self.conditions:
                tallies.setdefault(condition.name, Tally())
            spent = sum(tally.seconds for tally in tallies.values())
            waiting = [c for c in self.own if self._waiting(c, tallies[c.name])]
            if waiting:
                condition = waiting[0]
                line = self.play_batch(condition, tallies[condition.name])
                self._report(condition, line, tallies, None)
                continue
            summary = summarise(self.conditions, tallies)
            if summary.overall is None:  # another process's condition has no estimate yet
                self.progress(f"{describe_summary(summary)}; checking again in {poll_seconds:g} s")
                sleep(poll_seconds)
                continue
            budget = self.config.budget_hours * 3600.0
            wanted = targets(
                self.conditions,
                tallies,
                summary,
                next_variance(self.config, len(looks), self.maximum),
                self.batch_games(),
            )
            short = [c for c in self.conditions if tallies[c.name].games < wanted[c.name]]
            if short and spent < budget:
                mine = [c for c in short if c in self.own]
                if not mine:
                    self.progress(
                        f"waiting for {', '.join(c.name for c in short)}; checking again in "
                        f"{poll_seconds:g} s"
                    )
                    sleep(poll_seconds)
                    continue
                condition = max(  # the largest shortfall in engine seconds first
                    mine,
                    key=lambda c: (
                        (wanted[c.name] - tallies[c.name].games)
                        * (tallies[c.name].seconds_per_game() or 1.0)
                    ),
                )
                line = self.play_batch(condition, tallies[condition.name])
                self._report(condition, line, tallies, wanted)
                continue
            result = look(
                self.config,
                summary,
                [entry["fraction"] for entry in looks],
                spent,
                self.maximum,
            )
            entry = dict(result.as_dict(), writer=self.writer, time=_now())
            _append(self.directory / "looks" / f"{self.writer}.jsonl", entry)
            self.progress(describe_look(result))
            if result.decision != "continue":
                return result

    def _report(self, condition, line, tallies, wanted) -> None:
        tally = tallies[condition.name]
        rate = line["games"] / line["seconds"]
        summary = summarise(self.conditions, tallies)
        text = (
            f"{condition.name} batch {line['batch']}: {tally.games} games, {rate:.2f} games/s"
            f"{' (with compiling)' if line['first'] else ''}; {describe_summary(summary)}"
        )
        if wanted is not None:
            left = sum(
                max(wanted[c.name] - tallies[c.name].games, 0)
                * (tallies[c.name].seconds_per_game() or 0.0)
                for c in self.own
            )
            spent = sum(t.seconds for t in tallies.values())
            worst = max(self.config.budget_hours * 3600.0 - spent, 0.0)
            text += f"; next look ETA {clock(left)} (budget's end {clock(worst)})"
        self.progress(text)


def _now() -> str:
    return datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")


def describe_look(result: Look) -> str:
    """A look in words."""
    worse = "; ".join(
        f"{side} worse in {', '.join(modes)}" for side, modes in result.gate.items() if modes
    )
    return (
        f"look {result.look}{' (last)' if result.last else ''}: score {result.score:+.4f} +- "
        f"{result.stderr:.4f}, Z {result.z:+.2f}, fraction {result.fraction:.3f}, boundaries "
        f"+-{result.upper:.3f} (tie inside +-{result.inner:.3f}); test: {result.test}"
        + (f"; gate: {worse}" if worse else "")
        + f"; decision: {result.decision}"
    )


def decide(directory: pathlib.Path, record: bool = True) -> tuple[Summary, Look | None, dict]:
    """`bnb evaluate decide DIR`: read every record and apply the test at a look now
    (recorded unless `record` is False); (the summary, the look or None while a condition
    has no estimate, the tallies)."""
    _, config = read_settings(directory)
    conditions = config.scored()
    batches, _, _ = read_batches(directory)
    tallies = tallies_of(batches)
    for condition in conditions:
        tallies.setdefault(condition.name, Tally())
    summary = summarise(conditions, tallies)
    if summary.overall is None:
        return summary, None, tallies
    looks = read_looks(directory)
    if looks and looks[-1]["decision"] != "continue":
        return summary, look_of(looks[-1]), tallies
    spent = sum(tally.seconds for tally in tallies.values())
    result = look(config, summary, [entry["fraction"] for entry in looks], spent)
    if record:
        entry = dict(result.as_dict(), writer=_writer(), time=_now())
        _append(directory / "looks" / f"{entry['writer']}.jsonl", entry)
    return summary, result, tallies


def table(conditions: list[Condition], summary: Summary, tallies: dict[str, Tally]) -> list[str]:
    """The per-condition and per-mode lines: score, standard error, games, per-game
    variance and seconds per game."""
    lines = [
        f"{'condition':26s} {'score':>8s} {'SE':>7s} {'games':>7s} {'var/game':>9s} {'s/game':>8s}"
    ]
    for condition in conditions:
        value, tally = summary.conditions[condition.name], tallies[condition.name]
        cost = tally.seconds_per_game()
        cost_text = "" if cost is None else f"{cost:8.4f}"
        if value is None:
            lines.append(
                f"{condition.name:26s} {'-':>8s} {'-':>7s} {tally.games:7d} {'':9s} {cost_text}"
            )
            continue
        lines.append(
            f"{condition.name:26s} {value.score:+8.4f} {math.sqrt(value.variance):7.4f} "
            f"{tally.games:7d} {value.per_game:9.4f} {cost_text}"
        )
    for mode, value in summary.modes.items():
        if value is not None:
            lines.append(f"{'mode ' + mode:26s} {value[0]:+8.4f} {value[1]:7.4f}")
    if summary.overall is not None:
        lines.append(f"{'overall':26s} {summary.overall[0]:+8.4f} {summary.overall[1]:7.4f}")
    return lines
