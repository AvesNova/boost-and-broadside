"""The unified evaluator's planner: one allocation across every target.

When an environment's game ends, it starts the game the planner gives it next: of the
cells its engine can play (its size bucket and its mode's slice of the batch) and whose
players have room in their network lanes, the one with the most value per expected
second of play, the value summed over every target the cell serves (`Target.proposals`:
the drop of each unmet error's variance over its goal's, in flight games counted). The
planner then counts the game in flight with every target, so the next refill of the same
round sees it.

The value depends on variances and planned information only. An estimate's planned
information may use its fit (an Elo target's p (1 - p)); a decision's never reads the
observed difference (`bnb.league.unified.targets.DecisionTarget`), so the allocation
cannot bias its test.

Each cell's games alternate their seats (the first player on team 0, then team 1), and
each cell's expected seconds are the running mean of its condition's finished games (a
prior of half the preset's clock before any).
"""

import collections
import math

from bnb.game.modes.rules import CLOCKS, PRESETS

from .cells import BUCKETS, Cell, Condition, Outcome, bucket_of
from .targets import Target


def clock_seconds(preset: str) -> float:
    """The preset's episode length in game seconds."""
    game = PRESETS[preset]
    return game.episode_seconds if game.episode_seconds is not None else CLOCKS[game.mode][0]


class Planner:
    """targets: every target of the run; buckets: the engines' sizes."""

    def __init__(self, targets: list[Target], buckets: tuple[int, ...] = BUCKETS) -> None:
        names = [target.name for target in targets]
        if len(set(names)) != len(names):
            raise ValueError(f"target names repeat: {names}")
        self.targets, self.buckets = targets, tuple(sorted(buckets))
        self.seconds: dict[Condition, list[float]] = {}  # [sum, games]
        self.seats: collections.Counter = collections.Counter()
        self._cache: dict[str, dict[Cell, float]] = {}
        self.assigned = 0
        self.finished_games = 0

    # ---- costs

    def expected_seconds(self, condition: Condition) -> float:
        """The expected game seconds of a condition: its finished games' mean, shrunk
        toward half the clock while few."""
        held = self.seconds.get(condition)
        if held and held[1] >= 4:
            return max(held[0] / held[1], 1.0)
        prior = 0.5 * clock_seconds(condition.preset)
        if held and held[1]:
            return (held[0] + 4 * prior) / (held[1] + 4)  # shrunk toward the prior
        return prior

    # ---- proposals

    def _proposals(self, target: Target, overtime: bool = False) -> dict[Cell, float]:
        key = (target.name, overtime)
        found = self._cache.get(key)
        if found is None:
            found = target.proposals(overtime)
            self._cache[key] = found
        return found

    def values(self, overtime: bool = False) -> dict[Cell, float]:
        """Every proposed cell's summed value (overtime: `Target.proposals`')."""
        total: dict[Cell, float] = collections.defaultdict(float)
        for target in self.targets:
            for cell, value in self._proposals(target, overtime).items():
                total[cell] += value
        return dict(total)

    def _touched(self, cell: Cell) -> None:
        for key in list(self._cache):
            if cell in self._cache[key]:
                self._cache.pop(key, None)
        for target in self.targets:  # a target that proposed nothing may propose now
            for overtime in (False, True):
                if not self._cache.get((target.name, overtime), True):
                    self._cache.pop((target.name, overtime), None)

    def choose(
        self, bucket: int, preset: str, fits=lambda cell: True
    ) -> tuple[Cell, int, bool] | None:
        """The next game of an environment of size bucket `bucket` playing `preset`: (the
        cell, the seat its first player takes, 0 or 1, and whether it is overtime), counted
        in flight with every target. When no unmet goal wants a game in the slice, the
        overtime values choose (a met goal's further precision: the slice's modes are fixed
        while the bucket runs, `bnb.league.unified.engine`); None when even those want none
        (every cell's budget spent, or no room: `fits(cell)`, the lanes')."""
        for overtime in (False, True):
            best, best_rate = None, 0.0
            for cell, value in self.values(overtime).items():
                condition = cell.condition
                if condition.preset != preset or bucket_of(condition, self.buckets) != bucket:
                    continue
                rate = value / self.expected_seconds(condition)
                if rate > best_rate and fits(cell):
                    best, best_rate = cell, rate
            if best is not None:
                self.pending(best)
                seat = self.seats[best] % 2
                self.seats[best] += 1
                return best, seat, overtime
        return None

    def pending(self, cell: Cell) -> None:
        """Count a game of `cell` in flight with every target."""
        for target in self.targets:
            target.add_pending(cell)
        self._touched(cell)
        self.assigned += 1

    def finished(self, cell: Cell, outcome: Outcome, pending: bool = True) -> None:
        """A finished game: every target records it (pending: it was in flight), and its
        seconds join its condition's mean."""
        for target in self.targets:
            target.record(cell, outcome, pending)
        held = self.seconds.setdefault(cell.condition, [0.0, 0])
        held[0] += outcome.seconds
        held[1] += 1
        self._touched(cell)
        self.finished_games += 1

    def wanted(self) -> set[tuple[int, str]]:
        """The (bucket, preset) slices some target still proposes a cell in."""
        return {
            (bucket_of(cell.condition, self.buckets), cell.condition.preset)
            for cell in self.values()
        }

    def done(self) -> bool:
        """Whether the target is finished (its goals met, its budget spent, or decided)."""
        return all(target.done() for target in self.targets)

    def demand(self) -> dict[tuple[int, str], float]:
        """Each (bucket, preset) slice's share of the work in hand: its proposals' summed
        value times their expected seconds, normalised (the batch's split over the modes;
        a slice with nothing proposed is absent)."""
        work: dict[tuple[int, str], float] = collections.defaultdict(float)
        for cell, value in self.values().items():
            key = (bucket_of(cell.condition, self.buckets), cell.condition.preset)
            work[key] += min(value, 1.0) * self.expected_seconds(cell.condition)
        total = sum(work.values())
        return {key: value / total for key, value in work.items()} if total else {}

    def remaining(self) -> dict[str, float]:
        """Each target's worst error over its goal, squared, less 1 (0 when done)."""
        return {target.name: target.remaining() for target in self.targets}


def eta_seconds(progress: list[tuple[float, float]], now: float) -> float | None:
    """The seconds to finish from (time, remaining work) samples: the remaining work's
    log against time, fitted over the samples (variance falls as 1 / games, so the work
    left, error^2 / goal^2 - 1, falls about linearly in games once the games flow); None
    before two samples or while the work does not fall."""
    finite = [(t, w) for t, w in progress if math.isfinite(w)]
    if len(finite) < 2:
        return None
    (t0, w0), (t1, w1) = finite[0], finite[-1]
    if w1 <= 0.0:
        return 0.0
    if w0 <= w1 or t1 <= t0:
        return None
    rate = (w0 - w1) / (t1 - t0)
    return max(w1 / rate - (now - t1), 0.0)
