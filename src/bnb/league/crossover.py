"""The break-even ship count: `bnb crossover`, how many of a policy's ships hold a fixed
opponent fleet (by default 20 of the scripted agent's).

The opponent always fields `opponent_count` ships; the policy fields a varying number,
`count`, from `minimum_count` to `maximum_count`. One engine per mode is built at the larger
of the two sizes (`bnb.league.matches.Arena` with benching: the policy's unused slots are
benched, stay dead and out of play, and team deathmatch's winning score counts the enemy's
active ships). Every game draws its own lineup, each slot one of `lineup_types` uniformly,
both teams flying it slot for slot (the policy's benched ships are its lineup's last
slots); the draw is keyed by the seed, the batch and the environment (`lineup_draw`).

The measurement (`measure`), batch by batch, each batch `blocks` blocks of
`block_environments` games, each block at one count, its games split evenly between the
seats; every game scored by the game's own rule (win, draw or loss):

- **Fit** (`fit`): one logistic regression of the policy's score (a draw half a win, as a
  binomial likelihood on the score) on log(count), over every game so far. The break-even
  count N* is where the fitted curve crosses one half; its standard error is the delta
  method's from the fit's covariance. While the outcomes are separated (no count both won
  and lost where the counts below it are all lost and those above all won) the fit has no
  maximum: the result is a bound instead (`Fit.lower`, `Fit.upper`).
- **Allocation** (`allocate`): the first batch spreads its blocks over a coarse bracket
  (`bracket`). After that, with a crossing, each block goes in turn to the count that
  most reduces the predicted variance of log N* (the fit's covariance with that block's
  expected information added), averaged over N* at the estimate and one standard error
  either side; with separated outcomes, the blocks go inside the bound, or halve (double)
  the lowest (highest) count played to extend the range.
- **Stop** (`stop_reason`): when N*'s standard error reaches `target_stderr` ships, when the
  games reach `budget_games` (the batch that reaches it is the last), or when the bound
  is final (every game won at the minimum count, or lost at the maximum).

Every batch is appended to a record (`<preset>.batches.jsonl`, one line per batch: each
block's count and its wins, draws and losses, and the fit after it), and a rerun reuses
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
from bnb.streams import keyed_hash, stream_key

from .matches import Arena, ArenaConfig, Player
from .tournament import identity


@dataclasses.dataclass(frozen=True)
class CrossoverConfig:
    """opponent_count: the opponent's ships (fixed); minimum_count and maximum_count: the
    policy's counts the measurement may play; bracket: the first batch's counts (clipped to
    that range); target_stderr: the standard error of N* (ships) that stops the measurement;
    budget_games: the most games (the batch that reaches it is the last); lineup_types: the
    ship types (indices into the game's card) each slot of a game's lineup is drawn from,
    uniformly."""

    opponent_count: int = 20
    minimum_count: int = 1
    maximum_count: int = 20
    bracket: tuple[int, ...] = (6, 10, 14, 20)
    target_stderr: float = 0.3
    budget_games: int = 8192
    lineup_types: tuple[int, ...] = (0, 1)

    def __post_init__(self) -> None:
        if not 1 <= self.minimum_count <= self.maximum_count:
            raise ValueError(
                f"policy counts {self.minimum_count} to {self.maximum_count}: from at least 1"
            )
        if self.opponent_count < 1:
            raise ValueError(f"{self.opponent_count} opponent ships: at least 1")
        if not self.bracket:
            raise ValueError("an empty bracket")
        if not self.target_stderr > 0.0 or self.budget_games < 1:
            raise ValueError("the target standard error and the budget must be positive")
        if not self.lineup_types:
            raise ValueError("no ship types to draw lineups from")

    @property
    def ships_per_team(self) -> int:
        """The game's size: the larger of the two fleets."""
        return max(self.opponent_count, self.maximum_count)

    def bracket_counts(self) -> list[int]:
        """The bracket's counts inside the policy's range, in order, each once."""
        low, high = self.minimum_count, self.maximum_count
        return sorted({min(max(count, low), high) for count in self.bracket})


@dataclasses.dataclass(frozen=True)
class Fit:
    """The fit of the games so far. status: "crossing" (a fit with the score rising in the
    count: break_even, N*, with its standard error, and the curve's intercept and slope in
    log(count)); "separated" (no fit: N* lies between lower and upper, either None where the
    played counts do not bound it); "falling" (a fit with the score falling in the count:
    no break-even); "empty" (no games)."""

    status: str
    break_even: float | None = None
    stderr: float | None = None
    lower: float | None = None
    upper: float | None = None
    intercept: float | None = None
    slope: float | None = None
    covariance: tuple[tuple[float, float], tuple[float, float]] | None = None

    def ratio(self, opponent_count: int) -> tuple[float | None, float | None]:
        """The force ratio opponent_count / N* and its standard error (delta method), or
        (None, None) without a crossing."""
        if self.status != "crossing":
            return None, None
        ratio = opponent_count / self.break_even
        return ratio, ratio * self.stderr / self.break_even

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


def tallies(blocks: list[dict]) -> dict[int, list[float]]:
    """{count: [score summed, games]} over blocks of {count, wins, draws, losses}."""
    by_count: dict[int, list[float]] = {}
    for block in blocks:
        held = by_count.setdefault(int(block["count"]), [0.0, 0.0])
        held[0] += block["wins"] + 0.5 * block["draws"]
        held[1] += block["wins"] + block["draws"] + block["losses"]
    return by_count


def _information(
    x: torch.Tensor, games: torch.Tensor, intercept: float, slope: float
) -> torch.Tensor:
    """(2, 2) the Fisher information of (intercept, slope) at games played at x."""
    p = torch.sigmoid(intercept + slope * x)
    weight = games * p * (1 - p)
    design = torch.stack([torch.ones_like(x), x], -1)
    return (design * weight[:, None]).T @ design


def _separated_bound(by_count: dict[int, list[float]]) -> tuple[int | None, int | None] | None:
    """The bound on N* when the outcomes are separated with the score rising in the count
    (every count with a loss at or below every count with a win), else None: (the highest
    count with every game lost, the lowest with every game won; None where there is none)."""
    losing = [count for count, (score, games) in by_count.items() if score < games]
    winning = [count for count, (score, games) in by_count.items() if score > 0]
    if losing and winning and max(losing) > min(winning):
        return None
    lost = [count for count, (score, _) in by_count.items() if score == 0]
    won = [count for count, (score, games) in by_count.items() if score == games]
    return (max(lost) if lost else None, min(won) if won else None)


def _falling_separated(by_count: dict[int, list[float]]) -> bool:
    """Whether the outcomes are separated with the score falling in the count."""
    losing = [count for count, (score, games) in by_count.items() if score < games]
    winning = [count for count, (score, games) in by_count.items() if score > 0]
    return bool(losing and winning) and max(winning) <= min(losing)


def fit(by_count: dict[int, list[float]]) -> Fit:
    """The logistic fit of the score on log(count) from `tallies` (see the module
    docstring): Newton's method on the binomial log-likelihood of the summed scores, in
    float64, x centred for conditioning."""
    by_count = {count: held for count, held in by_count.items() if held[1] > 0}
    if not by_count:
        return Fit("empty")
    bound = _separated_bound(by_count)
    if bound is not None:
        return Fit("separated", lower=bound[0], upper=bound[1])
    if _falling_separated(by_count):
        return Fit("falling")
    counts = sorted(by_count)
    x = torch.tensor([math.log(count) for count in counts], dtype=torch.float64)
    score = torch.tensor([by_count[count][0] for count in counts], dtype=torch.float64)
    games = torch.tensor([by_count[count][1] for count in counts], dtype=torch.float64)
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
    intercept = a - s * centre  # in log(count) itself
    if s <= 0.0:
        return Fit("falling", intercept=intercept, slope=s)
    log_break_even = centre - a / s
    gradient = torch.tensor([-1.0 / s, a / (s * s)], dtype=torch.float64)  # of -a/s
    log_stderr = math.sqrt(float(gradient @ covariance @ gradient))
    break_even = math.exp(log_break_even)
    # the covariance of (intercept, slope) in log(count): the centring's linear map
    to_uncentred = torch.tensor([[1.0, -centre], [0.0, 1.0]], dtype=torch.float64)
    uncentred = to_uncentred @ covariance @ to_uncentred.T
    return Fit(
        "crossing",
        break_even=break_even,
        stderr=break_even * log_stderr,
        intercept=intercept,
        slope=s,
        covariance=tuple(tuple(float(value) for value in row) for row in uncentred.tolist()),
    )


def _spread(counts: list[int], blocks: int) -> list[int]:
    """`blocks` blocks dealt over the counts in turn."""
    return [counts[block % len(counts)] for block in range(blocks)]


def allocate(
    current: Fit,
    by_count: dict[int, list[float]],
    config: CrossoverConfig,
    blocks: int,
    block_games: int,
) -> list[int]:
    """The next batch's count per block (see the module docstring): the bracket without
    a fit; inside or beyond the bound with separated outcomes; with a crossing, each block in
    turn at the count that most reduces log N*'s predicted variance, averaged over N* at
    the estimate and one standard error either side."""
    low, high = config.minimum_count, config.maximum_count
    if current.status in ("empty", "falling"):
        return _spread(config.bracket_counts(), blocks)
    if current.status == "separated":
        lower, upper = current.lower, current.upper
        if lower is None and upper is None:  # only draws so far: nothing to steer by
            return _spread(config.bracket_counts(), blocks)
        if lower is None:  # every game won down to `upper`: go lower
            return _spread([max(low, upper // 2)], blocks)
        if upper is None:  # every game lost up to `lower`: go higher
            return _spread([min(high, 2 * lower)], blocks)
        inside = list(range(lower + 1, upper))
        if not inside:
            return _spread([lower, upper], blocks)
        middle = math.sqrt(lower * upper)
        nearest = sorted(inside, key=lambda count: abs(math.log(count / middle)))[:3]
        return _spread(sorted(nearest), blocks)
    # a crossing: greedy c-optimal design for log N*
    a, s = current.intercept, current.slope
    log_break_even, log_stderr = math.log(current.break_even), current.stderr / current.break_even
    played = sorted(by_count)
    information = _information(
        torch.tensor([math.log(count) for count in played], dtype=torch.float64),
        torch.tensor([by_count[count][1] for count in played], dtype=torch.float64),
        a,
        s,
    )
    candidates = list(range(low, high + 1))
    x = torch.tensor([math.log(count) for count in candidates], dtype=torch.float64)
    design = torch.stack([torch.ones_like(x), x], -1)  # (C, 2)
    shifts = (-log_stderr, 0.0, log_stderr)
    chosen = []
    for _ in range(blocks):
        criterion = torch.zeros(len(candidates), dtype=torch.float64)
        for shift in shifts:
            centre = log_break_even + shift
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
    return sorted(chosen)


def stop_reason(current: Fit, games: int, config: CrossoverConfig) -> str | None:
    """Why the measurement stops (see the module docstring): "target", "bound" or
    "budget", or None to play on."""
    if current.status == "crossing" and current.stderr <= config.target_stderr:
        return "target"
    if current.status == "separated":
        if current.lower is None and current.upper == config.minimum_count:
            return "bound"
        if current.upper is None and current.lower == config.maximum_count:
            return "bound"
    if games >= config.budget_games:
        return "budget"
    return None


def lineup_draw(
    seed: int, batch: int, games: int, ships_per_team: int, types: tuple[int, ...]
) -> torch.Tensor:
    """(games, ships_per_team) int32 each game's lineup: each slot one of `types`
    uniformly, keyed by the seed, the batch and the game (the same for every engine of one
    seed)."""
    key = stream_key(seed, "crossover_lineup")
    hashed = keyed_hash(
        key,
        torch.tensor(batch, dtype=torch.int64),
        torch.arange(games, dtype=torch.int64)[:, None],
        torch.arange(ships_per_team, dtype=torch.int64)[None, :],
    )
    choice = torch.tensor(types, dtype=torch.int32)
    return choice[(hashed >> 8) % len(types)]


def record_key(policy: Player, opponent: Player, game: GameConfig, seed: int, config) -> dict:
    """What makes a record's batches one measurement: the players' identities, the game,
    the seed, the opponent's count and the lineup's types (as the record holds it: JSON)."""
    key = dict(
        policy=identity(policy),
        opponent=identity(opponent),
        game=dataclasses.asdict(game),
        seed=seed,
        opponent_count=config.opponent_count,
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
    reason, the fit, N* and the ratio with their standard errors, the games and batches,
    and a row per count played (games, wins, draws, losses, score)."""
    if game.ships_per_team != config.ships_per_team:
        raise ValueError(
            f"a game of {game.ships_per_team} ships per team for fleets up to "
            f"{config.ships_per_team}"
        )
    key = record_key(policy, opponent, game, layout.seed, config)
    stored = stored_batches(record, key)
    blocks = [block for line in stored for block in line["blocks"]]
    by_count = tallies(blocks)
    current = fit(by_count)
    games = int(sum(held[1] for held in by_count.values()))
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
        counts = (
            _spread(config.bracket_counts(), layout.blocks)
            if not stored and played == 0
            else allocate(current, by_count, config, layout.blocks, layout.block_environments)
        )
        number = arena.batch
        lineups = lineup_draw(
            layout.seed, number, layout.games, game.ships_per_team, config.lineup_types
        )
        results = arena.play(
            [(0, 1)] * layout.blocks,
            [(count, config.opponent_count) for count in counts],
            lineups.to(device),
        )
        new = [
            dict(count=count, wins=int(won), draws=int(tied), losses=int(lost))
            for count, (won, tied, lost) in zip(counts, results.tolist(), strict=True)
        ]
        blocks += new
        by_count = tallies(blocks)
        current = fit(by_count)
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
            f"batch {number}: counts {counts}, {games} games, {describe(current)}"
            + (
                ""
                if reason
                else f"; ETA {clock_time(expected * per_batch)} "
                f"(worst {clock_time(worst * per_batch)})"
            )
        )
    ratio, ratio_stderr = current.ratio(config.opponent_count)
    rows = [
        dict(
            count=count,
            games=int(held[1]),
            wins=sum(block["wins"] for block in blocks if block["count"] == count),
            draws=sum(block["draws"] for block in blocks if block["count"] == count),
            losses=sum(block["losses"] for block in blocks if block["count"] == count),
            score=held[0] / held[1],
        )
        for count, held in sorted(by_count.items())
    ]
    return dict(
        policy=policy.name,
        opponent=opponent.name,
        policy_identity=key["policy"],
        opponent_identity=key["opponent"],
        opponent_count=config.opponent_count,
        stop=reason,
        fit=current.as_dict(),
        break_even=current.break_even,
        break_even_stderr=current.stderr,
        ratio=ratio,
        ratio_stderr=ratio_stderr,
        games=games,
        batches=len(stored) + played,
        counts=rows,
    )


def describe(current: Fit) -> str:
    """A fit in words: N* with its standard error, or the bound."""
    if current.status == "crossing":
        return f"N* {current.break_even:.2f} +- {current.stderr:.2f}"
    if current.status == "separated":
        lower = "-" if current.lower is None else current.lower
        upper = "-" if current.upper is None else current.upper
        return f"separated: N* between {lower} and {upper}"
    return f"no crossing ({current.status})"
