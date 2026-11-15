"""The break-even lineup: `bnb crossover`, the ratio of a policy's ships to a fixed
opponent's (by default Imugi's) at which the policy wins half its games.

A lineup is a pair (the policy's ships, the opponent's ships). The weaker side fields
`fixed_count` ships (20); the stronger side's count varies downward from it, to the mode's
smallest team at least (`minimum_count`). So a policy stronger than the opponent plays
(n, 20), a weaker one (20, n), and 20 against 20 belongs to both; no side ever fields more
than `fixed_count`. Which side is the stronger comes from the games, not from a setting.
Every lineup sits on one axis, the log ratio r = log(policy's ships / opponent's ships),
negative where the policy fields fewer, and the measurement works on that axis throughout.

One engine per mode is built at `fixed_count` a side (`break_even_game`: on the preset's own
arena by default, on the density rule's for that fleet as the map-size arm's opt-in;
`bnb.league.matches.Arena` with benching: each team's unused slots are benched, stay dead
and out of play, and team deathmatch's winning score counts the enemy's active ships).
Every game draws its own lineup of ship types, each slot one of `lineup_types` uniformly,
both teams flying it slot for slot (a team's benched ships are its lineup's last slots);
the draw is keyed by the seed, the batch and the environment (`lineup_draw`).

The measurement (`measure`), batch by batch, each batch `blocks` blocks of
`block_environments` games, each block at one lineup, its games split evenly between the
seats; every game scored by the game's own rule (win, draw or loss):

- **Fit** (`fit`): one logistic regression of the policy's score (a draw half a win, as a
  binomial likelihood on the score) on r, over every game so far, whichever side shrank.
  The break-even ratio is where the fitted curve crosses one half, r* = -intercept /
  slope; its standard error in log ratio is the delta method's from the fit's covariance.
  The break-even lineup is the pair at that ratio with the weaker side at `fixed_count`.
  While the outcomes are separated (no lineup both won and lost where the lineups below
  it in r are all lost and those above all won) the fit has no maximum: the result is a
  bound instead (`Fit.lower`, `Fit.upper`, lineups).
- **Allocation** (`allocate`): the first batch plays every block at parity (`fixed_count`
  a side). While the games give no fit, the next batch spreads its blocks over `bracket`
  on the side the games point to: the policy's ships shrink where its mean score is above
  one half, the opponent's where it is below, both at exactly one half. With a crossing,
  each block goes in turn to the lineup that most reduces the predicted variance of r*
  (the fit's covariance with that block's expected information added), averaged over r*
  at the estimate and one standard error either side, over every lineup on the axis (both
  sides of parity); with separated outcomes, the blocks go inside the bound, or halve the
  ratio below the lowest lineup played (every game won) or double it above the highest
  (every game lost) to extend the range.
- **Stop** (`stop_reason`): when r*'s standard error reaches `target_stderr`, when the games
  reach `budget_games` (the batch that reaches it is the last), or when the bound is final
  (every game won with the policy at the minimum against `fixed_count`, or lost with the
  opponent at the minimum).

Every batch is appended to a record (`<preset>.batches.jsonl`, one line per batch: each
block's lineup and its wins, draws and losses, and the fit after it), and a rerun reuses
the record's batches of the same measurement (`record_key`), continuing past them.
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
from bnb.streams import keyed_hash, stream_key

from .evaluation_settings import MAP_RULES
from .matches import Arena, ArenaConfig, Player
from .tournament import identity

# A lineup: (the policy's ships, the opponent's ships).
Lineup = tuple[int, int]
# Which side's ships shrink below `fixed_count`.
POLICY_SHRINKS, OPPONENT_SHRINKS, BOTH_SHRINK = "policy", "opponent", "both"


def log_ratio(lineup: Lineup) -> float:
    """r = log(the policy's ships / the opponent's ships)."""
    return math.log(lineup[0] / lineup[1])


def lineup_name(lineup) -> str:
    """A lineup in words: "14v20" (the policy's ships first)."""
    return f"{lineup[0]:g}v{lineup[1]:g}"


@dataclasses.dataclass(frozen=True)
class CrossoverConfig:
    """fixed_count: the weaker side's ships, fixed (the most either side fields, and the
    game's size); minimum_count: the fewest ships the stronger side may field (the mode's
    smallest team at least: `EvaluationConfig.crossover_config`); bracket: the stronger
    side's counts the batches after parity spread over before a fit (clipped to
    minimum_count to fixed_count); target_stderr: the standard error of the break-even log
    ratio that stops the measurement (0.02: 0.3 ships at 15 against 20); budget_games: the
    most games (the batch that reaches it is the last); lineup_types: the ship types
    (indices into the game's card) each slot of a game's lineup is drawn from,
    uniformly."""

    fixed_count: int = 20
    minimum_count: int = 1
    bracket: tuple[int, ...] = (6, 10, 14, 20)
    target_stderr: float = 0.02
    budget_games: int = 8192
    lineup_types: tuple[int, ...] = (0, 1)

    def __post_init__(self) -> None:
        if not 1 <= self.minimum_count <= self.fixed_count:
            raise ValueError(
                f"counts {self.minimum_count} to {self.fixed_count}: from at least 1 up to the "
                "fixed count"
            )
        if not self.bracket:
            raise ValueError("an empty bracket")
        if not self.target_stderr > 0.0 or self.budget_games < 1:
            raise ValueError("the target standard error and the budget must be positive")
        if not self.lineup_types:
            raise ValueError("no ship types to draw lineups from")

    @property
    def ships_per_team(self) -> int:
        """The game's size: the fixed side's fleet."""
        return self.fixed_count

    @property
    def parity(self) -> Lineup:
        """`fixed_count` a side."""
        return (self.fixed_count, self.fixed_count)

    def lineups(self) -> list[Lineup]:
        """Every lineup the measurement may play, in order of r: the policy's count rising
        to parity, then the opponent's falling from it."""
        low, high = self.minimum_count, self.fixed_count
        return [(count, high) for count in range(low, high + 1)] + [
            (high, count) for count in range(high - 1, low - 1, -1)
        ]

    def bracket_lineups(self, shrinking: str) -> list[Lineup]:
        """The bracket's lineups with the shrinking side's ships at the bracket's counts
        (clipped to the range), in order of r, each once."""
        low, high = self.minimum_count, self.fixed_count
        counts = {min(max(count, low), high) for count in self.bracket}
        found = set()
        if shrinking in (POLICY_SHRINKS, BOTH_SHRINK):
            found |= {(count, high) for count in counts}
        if shrinking in (OPPONENT_SHRINKS, BOTH_SHRINK):
            found |= {(high, count) for count in counts}
        return sorted(found, key=log_ratio)


def sized_game(
    base: GameConfig, ships_per_team: int, map_rule: str = "preset", map_ships: int | None = None
) -> GameConfig:
    """`base` at `ships_per_team` a side on the map rule's arena (`MAP_RULES`): "preset",
    the base's own radius at every size (the default: every map fixed-size, the author's
    rule of 2026-11-15); "density", the density rule's radius for `map_ships` a side
    (default ships_per_team; `bnb.game.api.arena_radius_for`), for the map-size arm."""
    from bnb.game.modes.rules import arena_radius_for

    if map_rule not in MAP_RULES:
        raise ValueError(f"map rule {map_rule!r}: one of {MAP_RULES}")
    radius = base.arena_radius
    if map_rule == "density":
        radius = arena_radius_for(base, ships_per_team if map_ships is None else map_ships)
    if ships_per_team == base.ships_per_team and radius == base.arena_radius:
        return base
    return dataclasses.replace(base, ships_per_team=ships_per_team, arena_radius=radius)


def break_even_game(base: GameConfig, config: CrossoverConfig, map_rule: str = "preset"):
    """Break-even's game: `base` at the fixed side's size (`CrossoverConfig.ships_per_team`),
    on the base's own arena ("preset") or the density rule's for that fleet ("density";
    `sized_game`)."""
    return sized_game(base, config.ships_per_team, map_rule)


@dataclasses.dataclass(frozen=True)
class Fit:
    """The fit of the games so far. status: "crossing" (a fit with the score rising in r:
    break_even, the break-even log ratio r*, with its standard error, and the curve's
    intercept and slope in r and their covariance); "separated" (no fit: r* lies between
    the lineups lower and upper, either None where the played lineups do not bound it);
    "falling" (a fit with the score falling in r: no break-even); "empty" (no games)."""

    status: str
    break_even: float | None = None
    stderr: float | None = None
    lower: Lineup | None = None
    upper: Lineup | None = None
    intercept: float | None = None
    slope: float | None = None
    covariance: tuple[tuple[float, float], tuple[float, float]] | None = None

    def ratio(self) -> tuple[float | None, float | None]:
        """The break-even ratio, the policy's ships over the opponent's (exp r*), and its
        standard error (delta method), or (None, None) without a crossing."""
        if self.status != "crossing":
            return None, None
        ratio = math.exp(self.break_even)
        return ratio, ratio * self.stderr

    def lineup(self, fixed_count: int) -> tuple[float, float] | None:
        """The break-even lineup: the pair at the break-even ratio with the weaker side at
        `fixed_count` (the policy's ships first; the other side's count fractional), or
        None without a crossing."""
        ratio, _ = self.ratio()
        if ratio is None:
            return None
        if ratio <= 1.0:
            return (fixed_count * ratio, float(fixed_count))
        return (float(fixed_count), fixed_count / ratio)

    def as_dict(self) -> dict:
        """The fit as a plain dict (for the record and the result's JSON)."""
        return dataclasses.asdict(self)


def tallies(blocks: list[dict]) -> dict[Lineup, list[float]]:
    """{lineup: [score summed, games]} over blocks of {policy_count, opponent_count, wins,
    draws, losses}."""
    by_lineup: dict[Lineup, list[float]] = {}
    for block in blocks:
        lineup = (int(block["policy_count"]), int(block["opponent_count"]))
        held = by_lineup.setdefault(lineup, [0.0, 0.0])
        held[0] += block["wins"] + 0.5 * block["draws"]
        held[1] += block["wins"] + block["draws"] + block["losses"]
    return by_lineup


def mean_score(by_lineup: dict[Lineup, list[float]]) -> float | None:
    """The policy's mean score over every game, or None without games."""
    games = sum(held[1] for held in by_lineup.values())
    return sum(held[0] for held in by_lineup.values()) / games if games else None


def _information(
    x: torch.Tensor, games: torch.Tensor, intercept: float, slope: float
) -> torch.Tensor:
    """(2, 2) the Fisher information of (intercept, slope) at games played at x."""
    p = torch.sigmoid(intercept + slope * x)
    weight = games * p * (1 - p)
    design = torch.stack([torch.ones_like(x), x], -1)
    return (design * weight[:, None]).T @ design


def _separated_bound(by_lineup: dict[Lineup, list[float]]) -> tuple | None:
    """The bound on r* when the outcomes are separated with the score rising in r (every
    lineup with a loss at or below every lineup with a win, in r), else None: (the highest
    lineup with every game lost, the lowest with every game won; None where there is
    none)."""
    losing = [lineup for lineup, (score, games) in by_lineup.items() if score < games]
    winning = [lineup for lineup, (score, games) in by_lineup.items() if score > 0]
    if losing and winning and max(map(log_ratio, losing)) > min(map(log_ratio, winning)):
        return None
    lost = [lineup for lineup, (score, _) in by_lineup.items() if score == 0]
    won = [lineup for lineup, (score, games) in by_lineup.items() if score == games]
    return (
        max(lost, key=log_ratio) if lost else None,
        min(won, key=log_ratio) if won else None,
    )


def _falling_separated(by_lineup: dict[Lineup, list[float]]) -> bool:
    """Whether the outcomes are separated with the score falling in r."""
    losing = [log_ratio(lineup) for lineup, (score, games) in by_lineup.items() if score < games]
    winning = [log_ratio(lineup) for lineup, (score, games) in by_lineup.items() if score > 0]
    return bool(losing and winning) and max(winning) <= min(losing)


def fit(by_lineup: dict[Lineup, list[float]]) -> Fit:
    """The logistic fit of the score on r from `tallies` (see the module docstring):
    Newton's method on the binomial log-likelihood of the summed scores, in float64, x
    centred for conditioning."""
    by_lineup = {lineup: held for lineup, held in by_lineup.items() if held[1] > 0}
    if not by_lineup:
        return Fit("empty")
    bound = _separated_bound(by_lineup)
    if bound is not None:
        return Fit("separated", lower=bound[0], upper=bound[1])
    if _falling_separated(by_lineup):
        return Fit("falling")
    played = sorted(by_lineup, key=log_ratio)
    x = torch.tensor([log_ratio(lineup) for lineup in played], dtype=torch.float64)
    score = torch.tensor([by_lineup[lineup][0] for lineup in played], dtype=torch.float64)
    games = torch.tensor([by_lineup[lineup][1] for lineup in played], dtype=torch.float64)
    centre = float((x * games).sum() / games.sum())
    xc = x - centre

    def log_likelihood(a: float, s: float) -> float:
        eta = a + s * xc
        return float((score * eta - games * torch.nn.functional.softplus(eta)).sum())

    a, s = 0.0, 0.0
    current = log_likelihood(a, s)
    for _ in range(200):
        p = torch.sigmoid(a + s * xc)
        gradient = torch.stack([(score - games * p).sum(), ((score - games * p) * xc).sum()])
        step = torch.linalg.solve(_information(xc, games, a, s), gradient)
        scale = 1.0
        while True:  # halve the step until the likelihood does not fall
            trial_a, trial_s = a + scale * float(step[0]), s + scale * float(step[1])
            trial = log_likelihood(trial_a, trial_s)
            if trial >= current - 1e-12 or scale < 1e-6:
                break
            scale /= 2
        a, s, previous, current = trial_a, trial_s, current, trial
        if abs(current - previous) < 1e-12 and float(step.abs().max()) * scale < 1e-10:
            break
    covariance = torch.linalg.inv(_information(xc, games, a, s))
    intercept = a - s * centre  # in r itself
    if s <= 0.0:
        return Fit("falling", intercept=intercept, slope=s)
    gradient = torch.tensor([-1.0 / s, a / (s * s)], dtype=torch.float64)  # of -a/s
    # the covariance of (intercept, slope) in r: the centring's linear map
    to_uncentred = torch.tensor([[1.0, -centre], [0.0, 1.0]], dtype=torch.float64)
    uncentred = to_uncentred @ covariance @ to_uncentred.T
    return Fit(
        "crossing",
        break_even=centre - a / s,
        stderr=math.sqrt(float(gradient @ covariance @ gradient)),
        intercept=intercept,
        slope=s,
        covariance=tuple(tuple(float(value) for value in row) for row in uncentred.tolist()),
    )


def _spread(lineups: list[Lineup], blocks: int) -> list[Lineup]:
    """`blocks` blocks dealt over the lineups in turn."""
    return [lineups[block % len(lineups)] for block in range(blocks)]


def shrinking_side(by_lineup: dict[Lineup, list[float]]) -> str:
    """The side whose ships shrink, from the policy's mean score: the policy's above one
    half, the opponent's below, both at one half (or without games)."""
    mean = mean_score(by_lineup)
    if mean is None or mean == 0.5:
        return BOTH_SHRINK
    return POLICY_SHRINKS if mean > 0.5 else OPPONENT_SHRINKS


def _nearest(candidates: list[Lineup], target: float) -> Lineup:
    """The candidate nearest `target` in r."""
    return min(candidates, key=lambda lineup: abs(log_ratio(lineup) - target))


def allocate(
    current: Fit,
    by_lineup: dict[Lineup, list[float]],
    config: CrossoverConfig,
    blocks: int,
    block_games: int,
) -> list[Lineup]:
    """The next batch's lineup per block (see the module docstring): parity without games;
    the bracket on the shrinking side without a fit; inside or beyond the bound with
    separated outcomes; with a crossing, each block in turn at the lineup that most reduces
    r*'s predicted variance, averaged over r* at the estimate and one standard error either
    side."""
    candidates = config.lineups()
    if current.status == "empty":
        return _spread([config.parity], blocks)
    if current.status == "falling":
        return _spread(config.bracket_lineups(shrinking_side(by_lineup)), blocks)
    if current.status == "separated":
        lower, upper = current.lower, current.upper
        if lower is None and upper is None:  # one ratio of mixed games, or only draws
            return _spread(config.bracket_lineups(shrinking_side(by_lineup)), blocks)
        if lower is None:  # every game won down to `upper`: halve the ratio
            below = [one for one in candidates if log_ratio(one) < log_ratio(upper)]
            target = log_ratio(upper) - math.log(2.0)
            return _spread([_nearest(below, target) if below else candidates[0]], blocks)
        if upper is None:  # every game lost up to `lower`: double the ratio
            above = [one for one in candidates if log_ratio(one) > log_ratio(lower)]
            target = log_ratio(lower) + math.log(2.0)
            return _spread([_nearest(above, target) if above else candidates[-1]], blocks)
        inside = [one for one in candidates if log_ratio(lower) < log_ratio(one) < log_ratio(upper)]
        if not inside:
            return _spread([lower, upper], blocks)
        middle = (log_ratio(lower) + log_ratio(upper)) / 2
        nearest = sorted(inside, key=lambda one: abs(log_ratio(one) - middle))[:3]
        return _spread(sorted(nearest, key=log_ratio), blocks)
    # a crossing: greedy c-optimal design for r*
    a, s = current.intercept, current.slope
    played = sorted(by_lineup, key=log_ratio)
    information = _information(
        torch.tensor([log_ratio(lineup) for lineup in played], dtype=torch.float64),
        torch.tensor([by_lineup[lineup][1] for lineup in played], dtype=torch.float64),
        a,
        s,
    )
    x = torch.tensor([log_ratio(lineup) for lineup in candidates], dtype=torch.float64)
    design = torch.stack([torch.ones_like(x), x], -1)  # (C, 2)
    shifts = (-current.stderr, 0.0, current.stderr)
    chosen = []
    for _ in range(blocks):
        criterion = torch.zeros(len(candidates), dtype=torch.float64)
        for shift in shifts:
            centre = current.break_even + shift
            shifted_intercept = -s * centre  # the curve moved to cross one half there
            p = torch.sigmoid(shifted_intercept + s * x)
            added = block_games * p * (1 - p)  # (C,)
            # the gradient of -intercept / slope, at the shifted curve
            gradient = torch.tensor([-1.0 / s, shifted_intercept / (s * s)], dtype=torch.float64)
            after = information[None] + added[:, None, None] * (
                design[:, :, None] * design[:, None, :]
            )
            criterion += (gradient @ torch.linalg.inv(after) @ gradient).reshape(-1)
        best = int(torch.argmin(criterion))
        chosen.append(candidates[best])
        p = torch.sigmoid(torch.tensor(a + s * float(x[best]), dtype=torch.float64))
        information = information + block_games * p * (1 - p) * torch.outer(
            design[best], design[best]
        )
    return sorted(chosen, key=log_ratio)


def stop_reason(current: Fit, games: int, config: CrossoverConfig) -> str | None:
    """Why the measurement stops (see the module docstring): "target", "bound" or
    "budget", or None to play on."""
    if current.status == "crossing" and current.stderr <= config.target_stderr:
        return "target"
    if current.status == "separated":
        candidates = config.lineups()
        if current.lower is None and current.upper is not None:
            if log_ratio(current.upper) <= log_ratio(candidates[0]):
                return "bound"
        if current.upper is None and current.lower is not None:
            if log_ratio(current.lower) >= log_ratio(candidates[-1]):
                return "bound"
    if games >= config.budget_games:
        return "budget"
    return None


def lineup_draw(
    seed: int,
    batch: int,
    games: int,
    ships_per_team: int,
    types: tuple[int, ...],
    stream: str = "crossover_lineup",
) -> torch.Tensor:
    """(games, ships_per_team) int32 each game's lineup of ship types: each slot one of
    `types` uniformly, keyed by the seed, the stream's name, the batch and the game (the
    same for every engine of one seed)."""
    key = stream_key(seed, stream)
    hashed = keyed_hash(
        key,
        torch.tensor(batch, dtype=torch.int64),
        torch.arange(games, dtype=torch.int64)[:, None],
        torch.arange(ships_per_team, dtype=torch.int64)[None, :],
    )
    choice = torch.tensor(types, dtype=torch.int32)
    return choice[(hashed >> 8) % len(types)]


def record_key(policy: Player, opponent: Player, game: GameConfig, seed: int, config) -> dict:
    """What makes a record's batches one measurement: the game version (`bnb.game.VERSION`),
    the players' identities, the game, the seed, the fixed and minimum counts (the lineups
    the measurement may play) and the lineup's types (as the record holds it: JSON)."""
    key = dict(
        game_version=GAME_VERSION,
        policy=identity(policy),
        opponent=identity(opponent),
        game=dataclasses.asdict(game),
        seed=seed,
        fixed_count=config.fixed_count,
        minimum_count=config.minimum_count,
        lineup_types=list(config.lineup_types),
    )
    return json.loads(json.dumps(key))


def stored_batches(record: pathlib.Path, key: dict) -> list[dict]:
    """The record's batches of this measurement (`record_key`), in order; none for a
    missing record. A line of another measurement is left alone."""
    if not record.exists():
        return []
    lines = [json.loads(line) for line in record.read_text().splitlines() if line.strip()]
    return sorted((line for line in lines if line["key"] == key), key=lambda line: line["batch"])


def clock_time(seconds_from_now: float) -> str:
    """The local clock time `seconds_from_now` ahead, HH:MM."""
    return (datetime.datetime.now() + datetime.timedelta(seconds=seconds_from_now)).strftime(
        "%H:%M"
    )


def _remaining_batches(current: Fit, games: int, batch_games: int, config) -> tuple[int, int]:
    """(expected, worst) batches still to play: the expected from the standard error
    shrinking as one over the root of the games (to the budget at most), the worst the
    budget's."""
    worst = max(math.ceil((config.budget_games - games) / batch_games), 0)
    if current.status != "crossing":
        return worst, worst
    needed = games * (current.stderr / config.target_stderr) ** 2 - games
    return min(max(math.ceil(needed / batch_games), 0), worst), worst


def measure(
    policy: Player,
    opponent: Player,
    game: GameConfig,
    device: torch.device,
    layout: ArenaConfig,
    config: CrossoverConfig,
    record: pathlib.Path,
    progress: Callable[[str], None] = print,
) -> dict:
    """The break-even measurement of `policy` against `opponent` in `game` (built at
    `config.ships_per_team`), batches appended to `record` and the record's batches of the
    same measurement reused (see the module docstring). Returns the result: the stop
    reason, the fit, the break-even log ratio and ratio with their standard errors and the
    break-even lineup, the games and batches, and a row per lineup played (games, wins,
    draws, losses, score)."""
    if game.ships_per_team != config.ships_per_team:
        raise ValueError(
            f"a game of {game.ships_per_team} ships per team for a fixed side of "
            f"{config.ships_per_team}"
        )
    key = record_key(policy, opponent, game, layout.seed, config)
    stored = stored_batches(record, key)
    blocks = [block for line in stored for block in line["blocks"]]
    by_lineup = tallies(blocks)
    current = fit(by_lineup)
    games = int(sum(held[1] for held in by_lineup.values()))
    reason = stop_reason(current, games, config)
    if stored:
        progress(f"reusing {len(stored)} batches, {games} games, from {record}")
    arena = None
    started, played = time.perf_counter(), 0
    batch_games = layout.blocks * layout.block_environments
    while reason is None:
        if arena is None:
            # a network player holds one lane per block (`default_lanes`): its blocks run as
            # one stacked forward, as head-to-head's do
            arena = Arena(
                [policy, opponent],
                game,
                device,
                layout,
                benching=True,
                game_lineups=True,
                first_batch=1 + max((line["batch"] for line in stored), default=-1),
            )
        lineups = allocate(current, by_lineup, config, layout.blocks, layout.block_environments)
        number = arena.batch
        types = lineup_draw(
            layout.seed, number, layout.games, game.ships_per_team, config.lineup_types
        )
        results = arena.play([(0, 1)] * layout.blocks, lineups, types.to(device))
        new = [
            dict(
                policy_count=lineup[0],
                opponent_count=lineup[1],
                wins=int(won),
                draws=int(tied),
                losses=int(lost),
            )
            for lineup, (won, tied, lost) in zip(lineups, results.tolist(), strict=True)
        ]
        blocks += new
        by_lineup = tallies(blocks)
        current = fit(by_lineup)
        games += batch_games
        played += 1
        line = dict(
            key=key,
            batch=number,
            blocks=new,
            fit=current.as_dict(),
            time=datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"),
        )
        record.parent.mkdir(parents=True, exist_ok=True)
        with record.open("a") as handle:
            handle.write(json.dumps(line) + "\n")
            handle.flush()
        reason = stop_reason(current, games, config)
        per_batch = (time.perf_counter() - started) / played
        expected, worst = _remaining_batches(current, games, batch_games, config)
        progress(
            f"batch {number}: lineups {' '.join(lineup_name(one) for one in lineups)}, "
            f"{games} games, {describe(current, config.fixed_count)}"
            + (
                ""
                if reason
                else f"; ETA {clock_time(expected * per_batch)} "
                f"(worst {clock_time(worst * per_batch)})"
            )
        )
    ratio, ratio_stderr = current.ratio()
    rows = []
    for lineup in sorted(by_lineup, key=log_ratio):
        mine = [one for one in blocks if (one["policy_count"], one["opponent_count"]) == lineup]
        score, lineup_games = by_lineup[lineup]
        rows.append(
            dict(
                policy_count=lineup[0],
                opponent_count=lineup[1],
                games=int(lineup_games),
                wins=sum(one["wins"] for one in mine),
                draws=sum(one["draws"] for one in mine),
                losses=sum(one["losses"] for one in mine),
                score=score / lineup_games,
            )
        )
    break_even_lineup = current.lineup(config.fixed_count)
    return dict(
        policy=policy.name,
        opponent=opponent.name,
        policy_identity=key["policy"],
        opponent_identity=key["opponent"],
        fixed_count=config.fixed_count,
        minimum_count=config.minimum_count,
        stop=reason,
        fit=current.as_dict(),
        log_ratio=current.break_even,
        log_ratio_stderr=current.stderr,
        ratio=ratio,
        ratio_stderr=ratio_stderr,
        lineup=None if break_even_lineup is None else list(break_even_lineup),
        games=games,
        batches=len(stored) + played,
        lineups=rows,
    )


def describe(current: Fit, fixed_count: int) -> str:
    """A fit in words: the break-even ratio with its standard error and lineup, or the
    bound."""
    if current.status == "crossing":
        ratio, ratio_stderr = current.ratio()
        policy_ships, opponent_ships = current.lineup(fixed_count)
        return f"ratio {ratio:.3f} +- {ratio_stderr:.3f} ({policy_ships:.1f}v{opponent_ships:.1f})"
    if current.status == "separated":
        lower = "-" if current.lower is None else lineup_name(current.lower)
        upper = "-" if current.upper is None else lineup_name(current.upper)
        return f"separated: break-even between {lower} and {upper}"
    return f"no crossing ({current.status})"
