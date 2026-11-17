"""The static Elo curve target (`static_design`'s design, the author's 2026-11-17 decision):
an `EloTarget` whose games are fixed counts per pair, in two stages, then stop.

- **Stage 1**: at the first call for proposals (after the prior games and a resume's are
  recorded), the A-optimal counts under the design prior (`static_design.a_optimal`), the
  games already in hand counted at their prior information (they reduce the counts).
- **Stage 2**: once every stage-1 count is played and no game is in flight, the fit
  (`design.fit_posterior`; the planning ratings `targets.planning_ratings` place a bound
  past its opponents) and the A-optimal top-up at the FITTED ratings: the games only where
  a player's error exceeds the goal (the games of a player outside the anchor's fitted
  component count nothing toward it, as `EloTarget`'s planning does).
- Then the target is done: its report carries every player's error, and a player still
  above its goal is a WARNING line (`run.short_of_goal`), never a silent stop.

Every cell short of its count is proposed (no probe caps, no overtime, no filler); the
engine plays them as its resident window reaches their players (`stream`, the fixed
streaming order; `engine.BucketEngine`). A rerun designs again from every game in hand, so
a resume continues where the counts left off.
"""

import math
import pathlib

import numpy

from bnb.league.design import RATED, fit_posterior
from bnb.league.ratings import SCALE

from . import static_design
from .cells import Cell, Condition, Outcome
from .targets import EloTarget, planning_ratings

# Stage 2 (the last) designs to the goal times this: its games move the fitted ratings, and
# with them the errors, by a few percent either way (bench/elo_static_design.py)
MARGIN = 0.95


class StaticEloTarget(EloTarget):
    """Ratings of a learning curve's field (the anchor first, then checkpoints) by the
    static design (module docstring). order: the checkpoints' streaming order (training
    order); pairs: the candidate pairs (`static_design.candidate_pairs`); prior: the design
    prior's Elo per player relative to the anchor (`static_design.prior_ratings`)."""

    def __init__(
        self,
        name: str,
        condition: Condition,
        players: list[str],
        stderr: float,
        order: list[str],
        pairs: list[tuple[str, str]],
        prior: dict[str, float],
        pilot: float = 1.0,
        margin: float = MARGIN,
        reach: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(name, condition, players, stderr, (), pairs, **kwargs)
        self.margin = margin  # stage 2 designs to the goal times this (the fit moves after)
        self.pilot = pilot  # stage 1's goal is the goal times this (1: the full prior design)
        self.reach = reach  # chain neighbours a side (`static_design.candidate_pairs`)
        self.order = list(order)
        self.rank = {name: number for number, name in enumerate(order)}
        self.prior_theta = numpy.array([prior.get(p, 0.0) / SCALE for p in players])
        self.goal = stderr
        self.stage = 0  # 0 not designed, 1 and 2 the stages, 3 done
        self.required = numpy.zeros((len(players), len(players)))  # games wanted in all
        self.designs: dict[int, static_design.Design] = {}
        self.thetas: dict[int, numpy.ndarray] = {}
        self.started_with = 0  # the games in hand when the current stage was designed
        first, second = self._pair_index
        # the cells' values: the earlier player in the order first (its lane frees first)
        self._value = numpy.array(
            [
                1.0 + 1.0 / (1.0 + min(self.rank.get(players[a], 0), self.rank.get(players[b], 0)))
                for a, b in zip(first.tolist(), second.tolist(), strict=True)
            ]
        )

    # ---- the design

    def _laplacian(self, theta: numpy.ndarray, games: numpy.ndarray) -> numpy.ndarray:
        probability = 1.0 / (1.0 + numpy.exp(-(theta[:, None] - theta[None, :])))
        weight = games * probability * (1.0 - probability)
        numpy.fill_diagonal(weight, 0.0)
        return numpy.diag(weight.sum(1)) - weight

    def _design_stage(self, stage: int) -> None:
        if stage == 1:
            theta = self.prior_theta
            existing = self._laplacian(theta, self.games)
        else:
            posterior = fit_posterior(self.wins, 0)
            self.posterior = posterior
            # the fit's ratings; a player without one at its planning rating (a bound past
            # its opponents, `planning_ratings`)
            theta = planning_ratings(self.wins, self.neighbours)
            fitted = posterior.ratings.numpy() / SCALE
            rated_now = numpy.array([s == RATED for s in posterior.status])
            theta = numpy.where(rated_now & numpy.isfinite(fitted), fitted, theta)
            component = posterior.component.numpy()
            inside = component[:, None] == component[None, :]
            rated = numpy.array([s == RATED for s in posterior.status])
            # a game counts toward its players' errors only within the anchor's component
            existing = self._laplacian(theta, self.games * inside * rated[:, None] * rated[None, :])
        goal = self.goal * (self.pilot if stage == 1 else self.margin)
        design = static_design.a_optimal(len(self.players), self.pairs, theta, goal, existing)
        self.designs[stage], self.thetas[stage] = design, theta
        for (a, b), games in zip(self.pairs, design.counts.tolist(), strict=True):
            self.required[a, b] = self.required[b, a] = self.games[a, b] + games
        self.stage = stage
        self.started_with = self.finished
        if design.rank < len(self.players) - 1:
            raise ValueError(
                f"{self.name}: the stage-{stage} design leaves players unconnected to the "
                f"anchor (information rank {design.rank} of {len(self.players) - 1})"
            )

    def design(self) -> None:
        """Design stage 1 if not yet (and move on while a stage has nothing to play)."""
        if self.stage == 0:
            self._design_stage(1)
        self._advance()

    def _outstanding(self) -> numpy.ndarray:
        """(P,) each candidate pair's games still to start."""
        first, second = self._pair_index
        return (
            self.required[first, second] - self.games[first, second] - self.pending[first, second]
        )

    def _advance(self) -> None:
        first, second = self._pair_index
        while self.stage in (1, 2):
            if (self._outstanding() > 0).any() or self.pending[first, second].sum() > 0:
                return
            if self.stage == 1:
                self._design_stage(2)
            else:
                self.stage = 3

    # ---- the planner's interface

    def proposals(self, overtime: bool = False) -> dict[Cell, float]:
        """Every cell short of its count, an earlier player's first (`Target.proposals`);
        nothing in overtime (the counts are the design)."""
        if self.stage == 0:
            self.design()
        if overtime or self.stage == 3 or self.finished + self.pending.sum() / 2 >= self.budget:
            return {}
        outstanding = self._outstanding()
        return {
            cell: float(value)
            for cell, value, left in zip(self._cells, self._value, outstanding, strict=True)
            if left > 0
        }

    def filler(self) -> dict[Cell, float]:
        """None: a static design plays its counts only."""
        return {}

    def record(self, cell: Cell, outcome: Outcome, pending: bool = True) -> None:
        """A finished game (`EloTarget.record`); a stage's last result designs the next."""
        super().record(cell, outcome, pending)
        if self.stage in (1, 2):
            self._advance()

    def done(self) -> bool:
        """Whether both stages are played (or the safety budget spent)."""
        return self.stage == 3 or self.finished >= self.budget

    def remaining(self) -> float:
        """The games still to play over those played in this stage (0 when done)."""
        if self.done():
            return 0.0
        if self.stage == 0:
            return math.inf
        first, second = self._pair_index
        left = float(
            numpy.maximum(self._outstanding(), 0).sum() + self.pending[first, second].sum()
        )
        return left / max(self.finished - self.started_with, 1)

    def stream(self) -> list[str]:
        """The streaming order of the field's checkpoints (`BucketEngine`'s resident window)."""
        return self.order

    def cells_left(self) -> dict[tuple[str, str], int]:
        """(first, second) -> the games each candidate pair still has to start."""
        return {
            (self.players[a], self.players[b]): int(left)
            for (a, b), left in zip(self.pairs, self._outstanding().tolist(), strict=True)
            if left > 0
        }

    def schedule(self, residents: int) -> list[dict]:
        """The current stage's streaming schedule (`static_design.streaming_schedule`)."""
        return static_design.streaming_schedule(self.order, self.cells_left(), residents)

    def design_table(self, seconds: float) -> str:
        """The current stage's per-pair table, expected errors and totals."""
        design = self.designs[self.stage if self.stage in self.designs else max(self.designs)]
        theta = self.thetas[max(self.thetas)]
        return static_design.table(self.players, design, theta, seconds, self.games)

    def report(self) -> dict:
        """`EloTarget.report` with the design: each stage's games and expected errors."""
        held = super().report()
        held["design"] = dict(
            kind="static",
            stage=self.stage,
            stages={
                str(stage): dict(
                    games=design.games,
                    expected_stderr=dict(
                        zip(self.players[1:], design.stderr[1:].round(1).tolist(), strict=True)
                    ),
                )
                for stage, design in self.designs.items()
            },
        )
        if self.done():
            held["stopped"] = "met" if self.met() else ("design" if self.stage == 3 else "budget")
        return held


def static_elo(
    name: str,
    condition: Condition,
    field: list[str],
    players,
    stderr: float,
    anchor: str,
    reach: int = static_design.NEIGHBOURS,
    pilot: float = 1.0,
    **kwargs,
) -> StaticEloTarget:
    """A spec's `[[elo]]` with `design = "static"` (`run.load_spec`): the field's
    checkpoints chained by their runs (`static_design.checkpoints`), each run resolved on
    its own against the anchor (no cross-run pairs: a later linking stage)."""
    seated, paths = [], set()
    for one in players:  # the same weights at two names (a champion): the first only
        if one.name not in field:
            continue
        where = None if one.policy is None else str(pathlib.Path(one.policy).resolve())
        if where is not None and where in paths:
            continue
        paths.add(where)
        seated.append(one)
    found, other = static_design.checkpoints(seated)
    other = [n for n in other if n != anchor]
    if other:
        raise ValueError(
            f"{name}: a static design rates checkpoints against the anchor only; "
            f"not on a run: {other}"
        )
    chains = static_design.chains_of(found)
    order = static_design.design_order(chains)
    pairs = static_design.candidate_pairs(chains, anchor, neighbours=reach)
    prior = static_design.prior_ratings(chains)
    names = [anchor] + order
    # the planning ratings' neighbours (stage 2): along each chain, across its transition
    kwargs["neighbours"] = [
        (a.name, b.name) for chain in chains.values() for a, b in zip(chain, chain[1:])
    ]
    return StaticEloTarget(
        name, condition, names, stderr, order, pairs, prior, pilot, reach=reach, **kwargs
    )
