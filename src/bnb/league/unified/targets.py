"""The unified evaluator's targets: what the games are for, each with a standard-error goal
(loose by default, tightenable later) or, for a decision, a sequential test.

Every target answers the planner (`bnb.league.unified.planner`) the same two questions:
which cells it can use (`proposals`: cell -> value) and whether it is done (`done`). A
cell's **value** is the share of the target's remaining variance gap one more game there
closes: the predicted drop of each unmet error's variance over its goal's variance, summed.
The predictions count the games in flight (`add_pending`) at their planned information,
so the refills of one round spread over the cells instead of all taking the best one.

- `EloTarget`: Bradley-Terry ratings of a field in one condition (`bnb.league.design`'s
  fit and planning view), each player's standard error and named contrasts to their goals.
- `BreakEvenTarget`: the break-even log ratio of a policy against an opponent in one mode
  (`bnb.league.crossover`'s fit and c-optimal allocation, on the 20-a-side axis).
- `WinRateTarget`: the first player's score in one cell (a team-size lineup).
- `DecisionTarget`: a pairwise decision (A against B in one condition) by the
  group-sequential test (`bnb.league.sequential`) with its looks at pre-set INFORMATION
  fractions: a look comes when the games played reach t_k of the planned maximum, the
  boundary is spent at the fraction actually reached (Lan-DeMets O'Brien-Fleming), and its
  value to the planner depends only on the game counts, never on the observed difference.
"""

import dataclasses
import functools
import math

import numpy

from bnb.league import crossover
from bnb.league.design import (
    MARGIN_CAP,
    PLANNING_STDERR,
    RATED,
    fit_posterior,
    planning_information,
)
from bnb.league.ratings import SCALE
from bnb.league.sequential import SequentialDesign, boundaries, plan

from .cells import Cell, Condition, Outcome

# The planning variance of one game's score (a win or a loss at even odds): the
# outcome-independent information every count-based plan uses.
SCORE_VARIANCE = 0.25


class Target:
    """The interface (see the module docstring)."""

    name: str

    def proposals(self, overtime: bool = False) -> dict[Cell, float]:
        """Cell -> value (module docstring). overtime: the values with every goal taken as
        unmet (a met goal's further precision), for a slice with nothing else wanted; a
        budget still binds, and a decided test proposes nothing."""
        raise NotImplementedError

    def add_pending(self, cell: Cell) -> None:
        """Count a game of `cell` in flight (a cell the target does not use is ignored)."""
        raise NotImplementedError

    def record(self, cell: Cell, outcome: Outcome, pending: bool = True) -> None:
        """A finished game of `cell` (pending: it was planned in flight, and leaves it)."""
        raise NotImplementedError

    def done(self) -> bool:
        """Whether the target is finished (its goals met, its budget spent, or decided)."""
        raise NotImplementedError

    def report(self) -> dict:
        """The target's estimate as a plain dict (report.json)."""
        raise NotImplementedError

    def remaining(self) -> float:
        """The worst error over its goal, squared, less 1 (0 when done): the progress logs'
        and the ETA's measure."""
        raise NotImplementedError


def _drop(planned: int) -> float:
    """The planned variance's drop from one more game after `planned`, with one pseudo-game
    of prior (so the first game's drop is finite and the largest): V/(m+1) - V/(m+2)."""
    return SCORE_VARIANCE / ((planned + 1) * (planned + 2))


def _score(outcome: Outcome, kill_share: bool) -> float:
    return outcome.kill_share if kill_share else outcome.score


# ---- Elo


class EloTarget(Target):
    """Ratings of `players` in `condition` (the anchor first, at 0; an Elo of 1000 is added
    when reported), every non-anchor player to `stderr` Elo and each contrast (a, b, goal)
    to its goal. pairs: the pairs that may be played (None: every pair); kill_share: score
    team deathmatch by kill share; refit: finished games between fits of the planning view
    (the planner's ratings and the p (1 - p) a game is planned at); budget: the most games."""

    def __init__(
        self,
        name: str,
        condition: Condition,
        players: list[str],
        stderr: float = 60.0,
        contrasts: tuple[tuple[str, str, float], ...] = (),
        pairs: list[tuple[str, str]] | None = None,
        kill_share: bool = False,
        refit: int = 64,
        budget: int = 1 << 30,
    ) -> None:
        if len(set(players)) != len(players) or len(players) < 2:
            raise ValueError(f"{name}: at least two distinct players, got {players}")
        self.name, self.condition, self.players = name, condition, list(players)
        self.index = {player: number for number, player in enumerate(players)}
        self.kill_share, self.refit, self.budget = kill_share, refit, budget
        count = len(players)
        if pairs is None:
            pairs = [(a, b) for a in players for b in players if self.index[a] < self.index[b]]
        self.pairs = [(self.index[a], self.index[b]) for a, b in pairs]
        directions, goals, labels = [], [], []
        for number in range(1, count):
            row = numpy.zeros(count)
            row[number] = 1.0
            directions.append(row)
            goals.append(stderr)
            labels.append(players[number])
        for a, b, goal in contrasts:
            row = numpy.zeros(count)
            row[self.index[a]] += 1.0
            row[self.index[b]] -= 1.0
            directions.append(row)
            goals.append(goal)
            labels.append(f"{a} - {b}")
        self.directions = numpy.stack(directions)[:, 1:]  # anchor pinned: its column dropped
        self.goals = numpy.array(goals) / SCALE  # natural units
        self.labels = labels
        self.wins = numpy.zeros((count, count))
        self.games = numpy.zeros((count, count))
        self.pending = numpy.zeros((count, count))
        self.finished = 0
        self.since_fit = 0
        self.theta = numpy.zeros(count)  # planning ratings, natural units
        self.difference_variance = numpy.full((count, count), (MARGIN_CAP / SCALE) ** 2)
        self._weights = None
        self._covariance = None
        self.posterior = None

    # the planning view

    def _fit(self) -> None:
        """Refit the planning ratings from the finished games (outcome-dependent: an
        estimate's plan, never a decision's)."""
        posterior = fit_posterior(self.wins, 0)
        self.posterior = posterior
        theta = numpy.zeros(len(self.players))
        rated = numpy.array([status == RATED for status in posterior.status])
        relative = posterior.ratings.numpy() / SCALE
        theta[rated] = relative[rated]
        self.theta = theta
        covariance = posterior.covariance.numpy() / SCALE**2
        diagonal = numpy.diag(covariance)
        variance = diagonal[:, None] + diagonal[None, :] - 2.0 * covariance
        unrated = ~rated
        variance[unrated, :] = (MARGIN_CAP / SCALE) ** 2
        variance[:, unrated] = (MARGIN_CAP / SCALE) ** 2
        self.difference_variance = variance
        self._weights = None
        self._covariance = None

    def _pair_weights(self) -> numpy.ndarray:
        """(K, K) each pair's planned information per game, p (1 - p) at one standard
        error past the difference (`bnb.league.design.planning_information`)."""
        if self._weights is None:
            import torch

            difference = torch.from_numpy(self.theta[:, None] - self.theta[None, :])
            variance = torch.from_numpy(self.difference_variance)
            self._weights = planning_information(difference, variance, 1.0).numpy()
        return self._weights

    def covariance(self) -> numpy.ndarray:
        """(K - 1, K - 1) the planning covariance of the non-anchor ratings (natural units):
        the finished and in-flight games' planned information, with a weak prior
        (`PLANNING_STDERR`) so an unplayed player has a finite variance."""
        if self._covariance is None:
            weights = self._pair_weights() * (self.games + self.pending)
            precision = numpy.diag(weights.sum(1)) - weights
            precision = (
                precision[1:, 1:]
                + numpy.eye(len(self.players) - 1) * (SCALE / PLANNING_STDERR) ** 2
            )
            self._covariance = numpy.linalg.inv(precision)
        return self._covariance

    def _unmet(self, covariance: numpy.ndarray) -> numpy.ndarray:
        variance = numpy.einsum("tk,kl,tl->t", self.directions, covariance, self.directions)
        return variance > self.goals**2

    def proposals(self, overtime: bool = False) -> dict[Cell, float]:
        """Cell -> value of one more game there (`Target.proposals`)."""
        if self.finished + self.pending.sum() / 2 >= self.budget:
            return {}
        covariance = self.covariance()
        # a goal is played for while the fit leaves it unmet (and off a bound), unless the
        # games in flight are planned to meet it
        unmet = self._open() & (self._unmet(covariance) | (self.pending.sum() == 0))
        if overtime:
            unmet = numpy.ones_like(unmet)
        if not unmet.any():
            return {}
        rows = self.directions[unmet] / self.goals[unmet, None]  # (T, K - 1)
        weights = self._pair_weights()
        out = {}
        for a, b in self.pairs:
            direction = numpy.zeros(len(self.players))
            direction[a], direction[b] = 1.0, -1.0
            e = direction[1:]
            column = covariance @ e
            quadratic = float(e @ column)
            w = float(weights[a, b])
            if w <= 0.0:
                continue
            reduction = w * float(((rows @ column) ** 2).sum()) / (1.0 + w * quadratic)
            if reduction > 0.0:
                out[self.cell(a, b)] = reduction
        return out

    def cell(self, a: int, b: int) -> Cell:
        """The cell of players a and b (indices) in the target's condition."""
        return Cell(self.condition, self.players[a], self.players[b])

    def _pair(self, cell: Cell) -> tuple[int, int] | None:
        if cell.condition != self.condition:
            return None
        a, b = self.index.get(cell.first), self.index.get(cell.second)
        return None if a is None or b is None else (a, b)

    def add_pending(self, cell: Cell) -> None:
        """Count a game of `cell` in flight (a cell the target does not use is ignored)."""
        pair = self._pair(cell)
        if pair is not None:
            a, b = pair
            self.pending[a, b] += 1
            self.pending[b, a] += 1
            self._covariance = None

    def record(self, cell: Cell, outcome: Outcome, pending: bool = True) -> None:
        """A finished game of `cell` (pending: it was counted in flight, and leaves it)."""
        pair = self._pair(cell)
        if pair is None:
            return
        a, b = pair
        if pending and self.pending[a, b] > 0:
            self.pending[a, b] -= 1
            self.pending[b, a] -= 1
        score = _score(outcome, self.kill_share)
        self.wins[a, b] += score
        self.wins[b, a] += 1.0 - score
        self.games[a, b] += 1
        self.games[b, a] += 1
        self.finished += 1
        self.since_fit += 1
        self._covariance = None
        if self.since_fit >= self.refit or self.posterior is None:
            self.since_fit = 0
            self._fit()

    def errors(self, fresh: bool = False) -> numpy.ndarray:
        """(T,) each goal's standard error, in Elo (inf without a rating): the fit's, refitted
        after every `refit` finished games (fresh: whenever a game has finished since)."""
        held = getattr(self, "_errors", None)
        if held is not None and (
            held[0] == self.finished or (not fresh and self.finished - held[0] < self.refit)
        ):
            return held[1]
        posterior = fit_posterior(self.wins, 0)
        self.posterior = posterior
        import torch

        full = numpy.zeros((len(self.directions), len(self.players)))
        full[:, 1:] = self.directions
        errors = posterior.variance(torch.from_numpy(full)).sqrt().numpy()
        self._errors = (self.finished, errors)
        return errors

    def _open(self) -> numpy.ndarray:
        """(T,) the goals the fit leaves open: above their error and not on a bound (a
        player every game of whose went one way is reported as a bound, as the rating
        pool's fit reports it; `bnb.league.design.Posterior.on_bound`)."""
        errors = self.errors()
        import torch

        full = numpy.zeros((len(self.directions), len(self.players)))
        full[:, 1:] = self.directions
        bound = self.posterior.on_bound(torch.from_numpy(full)).numpy()
        # a bound stands once its players have played the games an even pair would need
        # for the goal, 1 / (goal^2 / 4) (else a short sweep would end the target)
        played = self.games.sum(1)
        touched = numpy.abs(full) > 0
        least = numpy.where(touched, played[None, :], numpy.inf).min(1)
        settled = bound & (least >= 4.0 / self.goals**2)
        return ~((errors <= self.goals * SCALE + 1e-9) | settled)

    def done(self) -> bool:
        """Whether the target is finished (its goals met, its budget spent, or decided)."""
        return self.finished >= self.budget or not self._open().any()

    def remaining(self) -> float:
        """The worst error over its goal, squared, less 1 (0 when done; inf with no estimate)."""
        if self.done():
            return 0.0
        errors = numpy.where(self._open(), self.errors(), 0.0)
        worst = float(numpy.max(errors / (self.goals * SCALE)))
        return 0.0 if worst <= 1.0 else (math.inf if not math.isfinite(worst) else worst**2 - 1)

    def report(self) -> dict:
        """The target's estimate as a plain dict (report.json)."""
        errors = self.errors(fresh=True)
        posterior = self.posterior
        return dict(
            kind="elo",
            condition=self.condition.key,
            games=self.finished,
            ratings={
                name: (None if math.isnan(r) else 1000.0 + r)
                for name, r in zip(self.players, posterior.ratings.tolist(), strict=True)
            },
            status=dict(zip(self.players, posterior.status, strict=True)),
            errors={
                label: (None if not math.isfinite(e) else e)
                for label, e in zip(self.labels, errors.tolist(), strict=True)
            },
            goals=dict(zip(self.labels, (self.goals * SCALE).tolist(), strict=True)),
        )


# ---- a cell's score


class WinRateTarget(Target):
    """The first player's score in one cell (a team-size lineup) to `stderr`, planned at
    the score variance of even odds (`SCORE_VARIANCE`, the largest), met on the observed
    variance once `minimum` games are in; at most `budget` games."""

    def __init__(
        self,
        name: str,
        cell: Cell,
        stderr: float = 0.05,
        minimum: int = 16,
        budget: int = 1 << 30,
        kill_share: bool = False,
    ) -> None:
        self.name, self.cell, self.goal = name, cell, stderr
        self.minimum, self.budget, self.kill_share = minimum, budget, kill_share
        self.games, self.pending, self.total, self.squares = 0, 0, 0.0, 0.0

    def _planned(self) -> int:
        return self.games + self.pending

    def proposals(self, overtime: bool = False) -> dict[Cell, float]:
        """Cell -> value of one more game there (`Target.proposals`)."""
        planned = self._planned()
        if planned >= self.budget:
            return {}
        if overtime:
            return {self.cell: _drop(planned) / self.goal**2}
        if self.done():
            return {}
        if self.pending and SCORE_VARIANCE / planned <= self.goal**2:
            return {}  # the games in flight should do (a look after them decides)
        return {self.cell: _drop(planned) / self.goal**2}

    def add_pending(self, cell: Cell) -> None:
        """Count a game of `cell` in flight (a cell the target does not use is ignored)."""
        if cell == self.cell:
            self.pending += 1

    def record(self, cell: Cell, outcome: Outcome, pending: bool = True) -> None:
        """A finished game of `cell` (pending: it was counted in flight, and leaves it)."""
        if cell != self.cell:
            return
        if pending and self.pending:
            self.pending -= 1
        score = _score(outcome, self.kill_share)
        self.games += 1
        self.total += score
        self.squares += score * score

    def mean(self) -> float | None:
        """The mean score so far (None without games)."""
        return self.total / self.games if self.games else None

    def stderr(self) -> float:
        """The observed score's standard error (inf before two games; at least the floor
        of a cell whose every game went one way)."""
        if self.games < 2:
            return math.inf
        mean = self.total / self.games
        variance = max(self.squares / self.games - mean * mean, 0.0) * self.games / (self.games - 1)
        # a cell every game of which went one way still has the planning floor's error
        return math.sqrt(max(variance, 1.0 / (4.0 * self.games)) / self.games)

    def done(self) -> bool:
        """Whether the target is finished (its goals met, its budget spent, or decided)."""
        return self.games >= self.budget or (
            self.games >= self.minimum and self.stderr() <= self.goal
        )

    def remaining(self) -> float:
        """The worst error over its goal, squared, less 1 (0 when done; inf with no estimate)."""
        if self.done():
            return 0.0
        error = self.stderr()
        return math.inf if not math.isfinite(error) else max((error / self.goal) ** 2 - 1, 0.0)

    def report(self) -> dict:
        """The target's estimate as a plain dict (report.json)."""
        return dict(
            kind="win_rate",
            cell=self.cell.key,
            games=self.games,
            score=self.mean(),
            stderr=None if not math.isfinite(self.stderr()) else self.stderr(),
            goal=self.goal,
        )


# ---- break-even


class BreakEvenTarget(Target):
    """The break-even log ratio of `policy` against `opponent` in `preset`'s mode
    (`bnb.league.crossover`, on its fixed-count axis), to `stderr` in log ratio, at most
    `budget` games. Its next lineups come from crossover's allocation over the finished
    games, `queue` pairs at a time (the in-flight ones count toward nothing until they end:
    a pair's two games, one in each seat, are planned together); before a crossing every
    game is worth `EXPLORE`, after one the predicted drop of r*'s variance over the goal's."""

    EXPLORE = 1.0

    def __init__(
        self,
        name: str,
        preset: str,
        policy: str,
        opponent: str,
        stderr: float = 0.1,
        fixed_count: int = 20,
        budget: int = 2048,
        queue: int = 16,
    ) -> None:
        self.name, self.preset, self.policy, self.opponent = name, preset, policy, opponent
        self.config = crossover.CrossoverConfig(
            fixed_count=fixed_count, target_stderr=stderr, budget_games=budget
        )
        self.goal, self.budget, self.queue_pairs = stderr, budget, queue
        self.by_lineup: dict[tuple[int, int], list[float]] = {}
        self.fit = crossover.Fit("empty")
        self.queue: list[tuple[int, int]] = []
        self.pending: dict[tuple[int, int], int] = {}
        self.games = 0

    def _cell(self, lineup: tuple[int, int]) -> Cell:
        return Cell(Condition(self.preset, lineup[0], lineup[1]), self.policy, self.opponent)

    def _planned(self) -> int:
        return self.games + sum(self.pending.values())

    def _value(self, lineup: tuple[int, int]) -> float:
        if self.fit.status != "crossing":
            return self.EXPLORE
        a, s = self.fit.intercept, self.fit.slope
        played = list(self.by_lineup)
        x = numpy.array([crossover.log_ratio(one) for one in played])
        n = numpy.array([self.by_lineup[one][1] for one in played])
        p = 1.0 / (1.0 + numpy.exp(-(a + s * x)))
        design = numpy.stack([numpy.ones_like(x), x], -1)
        information = (design.T * (n * p * (1 - p))) @ design
        gradient = numpy.array([-1.0 / s, a / (s * s)])
        r = crossover.log_ratio(lineup)
        q = 1.0 / (1.0 + math.exp(-(a + s * r)))
        d = numpy.array([1.0, r])
        before = gradient @ numpy.linalg.solve(information, gradient)
        after_information = information + q * (1 - q) * numpy.outer(d, d)
        after = gradient @ numpy.linalg.solve(after_information, gradient)
        return max(float(before - after), 0.0) / self.goal**2

    def proposals(self, overtime: bool = False) -> dict[Cell, float]:
        """Cell -> value of one more game there (`Target.proposals`)."""
        if self._planned() >= self.budget or (self.done() and not overtime):
            return {}
        if not self.queue:
            self.queue = crossover.allocate(
                self.fit, self.by_lineup, self.config, 2 * self.queue_pairs
            )
        lineup = self.queue[0]
        return {self._cell(lineup): self._value(lineup)}

    def _lineup(self, cell: Cell) -> tuple[int, int] | None:
        c = cell.condition
        if (c.preset, cell.first, cell.second, c.arena_scale) != (
            self.preset,
            self.policy,
            self.opponent,
            1.0,
        ):
            return None
        return (c.first, c.second)

    def add_pending(self, cell: Cell) -> None:
        """Count a game of `cell` in flight (a cell the target does not use is ignored)."""
        lineup = self._lineup(cell)
        if lineup is None:
            return
        self.pending[lineup] = self.pending.get(lineup, 0) + 1
        if lineup in self.queue:
            self.queue.remove(lineup)

    def record(self, cell: Cell, outcome: Outcome, pending: bool = True) -> None:
        """A finished game of `cell` (pending: it was counted in flight, and leaves it)."""
        lineup = self._lineup(cell)
        if lineup is None:
            return
        if pending and self.pending.get(lineup, 0) > 0:
            self.pending[lineup] -= 1
        held = self.by_lineup.setdefault(lineup, [0.0, 0.0])
        held[0] += outcome.score
        held[1] += 1
        self.games += 1
        if self.games % 8 == 0 or self.fit.status == "empty":
            self.fit = crossover.fit(self.by_lineup)
            self.queue = []

    def done(self) -> bool:
        """Whether the target is finished (its goals met, its budget spent, or decided)."""
        if self.games >= self.budget:
            return True
        reason = crossover.stop_reason(self.fit, self.games, self.config, self.by_lineup)
        return reason is not None

    def remaining(self) -> float:
        """The worst error over its goal, squared, less 1 (0 when done; inf with no estimate)."""
        if self.done():
            return 0.0
        if self.fit.status != "crossing":
            return math.inf
        return max((self.fit.stderr / self.goal) ** 2 - 1, 0.0)

    def report(self) -> dict:
        """The target's estimate as a plain dict (report.json)."""
        ratio, ratio_error = self.fit.ratio()
        return dict(
            kind="break_even",
            preset=self.preset,
            policy=self.policy,
            opponent=self.opponent,
            games=self.games,
            fit=self.fit.as_dict(),
            ratio=ratio,
            ratio_stderr=ratio_error,
            goal=self.goal,
            lineups={crossover.lineup_name(k): v for k, v in sorted(self.by_lineup.items())},
        )


# ---- decisions


@functools.cache
def _plan(design: SequentialDesign) -> float:
    """The design's planned drift at delta (`bnb.league.sequential.plan`), once per design."""
    return plan(design)[0]


@functools.lru_cache(maxsize=4096)
def _boundaries(design: SequentialDesign, fractions: tuple, drift: float, final: bool):
    return boundaries(design, list(fractions), drift, final)


@dataclasses.dataclass
class Look:
    """One look of a decision: the games at it, its information fraction, the statistic and
    the boundaries spent at it, and the outcome ("A", "B", "tie" or "continue")."""

    games: int
    fraction: float
    z: float
    upper: float
    inner: float
    outcome: str


class InformationLooks:
    """The group-sequential test with looks at pre-set information fractions (the module
    docstring): `design`'s K looks are planned at fractions k / K of I_max (`plan`); look k
    comes at the first check after the information reaches fraction_k of I_max, and its
    boundaries are spent at the fraction actually reached, given the looks before (so a
    late or skipped look keeps the error rate: the spending function, not the schedule,
    sets it). The information is counted, not estimated: games / `variance` (the planned
    per-game variance), so when a look comes depends on the games alone."""

    def __init__(self, design: SequentialDesign, variance: float = SCORE_VARIANCE) -> None:
        self.design, self.variance = design, variance
        drift = _plan(design)
        self.drift = drift
        self.maximum_information = (drift / design.delta) ** 2
        self.maximum_games = math.ceil(self.maximum_information * variance)
        self.planned = [(k + 1) / design.looks for k in range(design.looks)]
        self.looks: list[Look] = []
        self.outcome: str | None = None

    def next_fraction(self) -> float:
        """The planned fraction of the next look: the first planned one past the last look's
        fraction (a look that came late passes the planned ones it reached)."""
        last = self.looks[-1].fraction if self.looks else 0.0
        later = [fraction for fraction in self.planned if fraction > last + 1e-12]
        return later[0] if later else self.planned[-1]

    def due(self, games: int) -> bool:
        """Whether `games` finished games bring the next look (and the test still runs)."""
        if self.outcome is not None:
            return False
        return games / self.maximum_games >= self.next_fraction() - 1e-12

    def look(self, games: int, z: float) -> str:
        """Take the look at `games` games with statistic `z` (the difference over its
        standard error): the outcome, which ends the test unless "continue". The boundary
        is spent at the fraction reached, given the earlier looks' actual fractions; a look
        at or past the last planned fraction is final (it always decides)."""
        fraction = games / self.maximum_games
        final = fraction >= self.planned[-1] - 1e-12
        fractions = [one.fraction for one in self.looks] + [fraction]
        found = _boundaries(self.design, tuple(fractions), self.drift, final)
        upper, inner = found.upper[-1], found.inner[-1]
        outcome = found.decision(len(fractions) - 1, z)
        self.looks.append(Look(games, fraction, z, upper, inner, outcome))
        if outcome != "continue":
            self.outcome = outcome
        return outcome


class DecisionTarget(Target):
    """Whether A (`cell.first`) beats B (`cell.second`) in `cell`'s condition: the score
    difference from one half, tested by `InformationLooks`. The planner's value is the
    drop of the planned variance SCORE_VARIANCE / games toward its value at the maximum,
    by the counts alone; at most the planned maximum's games are planned in all."""

    def __init__(self, name: str, cell: Cell, design: SequentialDesign = SequentialDesign()):
        self.name, self.cell = name, cell
        self.test = InformationLooks(design)
        self.games, self.pending, self.total, self.squares = 0, 0, 0.0, 0.0

    def proposals(self, overtime: bool = False) -> dict[Cell, float]:
        """Cell -> value of one more game there (`Target.proposals`)."""
        planned = self.games + self.pending
        if self.done() or planned >= self.test.maximum_games:
            return {}
        floor = SCORE_VARIANCE / self.test.maximum_games
        return {self.cell: _drop(planned) / floor}

    def add_pending(self, cell: Cell) -> None:
        """Count a game of `cell` in flight (a cell the target does not use is ignored)."""
        if cell == self.cell:
            self.pending += 1

    def statistic(self) -> float:
        """The test's statistic: the mean score less one half over its standard error."""
        mean = self.total / self.games
        variance = max(self.squares / self.games - mean * mean, 1e-12)
        return (mean - 0.5) / math.sqrt(variance / self.games)

    def record(self, cell: Cell, outcome: Outcome, pending: bool = True) -> None:
        """A finished game of `cell` (pending: it was counted in flight, and leaves it)."""
        if cell != self.cell:
            return
        if pending and self.pending:
            self.pending -= 1
        if self.test.outcome is not None:
            return  # decided: later games are not the test's
        self.games += 1
        self.total += outcome.score
        self.squares += outcome.score**2
        if self.test.due(self.games) and self.games >= 2:
            self.test.look(self.games, self.statistic())

    def done(self) -> bool:
        """Whether the target is finished (its goals met, its budget spent, or decided)."""
        return self.test.outcome is not None

    def remaining(self) -> float:
        """The worst error over its goal, squared, less 1 (0 when done; inf with no estimate)."""
        if self.done():
            return 0.0
        return max(self.test.maximum_games / max(self.games, 1) - 1, 0.0)

    def report(self) -> dict:
        """The target's estimate as a plain dict (report.json)."""
        return dict(
            kind="decision",
            cell=self.cell.key,
            games=self.games,
            maximum_games=self.test.maximum_games,
            outcome=self.test.outcome,
            looks=[dataclasses.asdict(look) for look in self.test.looks],
        )
