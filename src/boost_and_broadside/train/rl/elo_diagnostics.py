"""Instrumentation for the live Elo estimator.

Phase 0 of ``docs/internal/live-elo-plan.md``. Everything here is read-only: it
observes the K-factor filter and changes nothing about how the filter, opponent
sampling, or ladder promotion behave. The point is to have a baseline recorded
under the current estimator before a replacement is proposed, and to catch the
failure that motivated the work — run 727's live rating stepped +85 across a
resume seam and stayed there, while its win rate against the scripted controller
implied roughly a third of that.

Four questions, one metric family each.

*Is the rating consistent with the games actually played?* The gauge's
stationary references — random, the semi-random rungs, and scripted — have
ratings that are **defined** rather than estimated (see ``config/live_elo``), so
a rating fitted to the live policy's record against those alone is independent
of the run's self-generated ladder. The gap between that and ``live_elo`` is the
drift detector.

**Read the drift as a paired comparison between runs, never as an absolute
error.** It carries a large shared bias. Replayed over run 719 — whose filter
the post-hoc calibration shows tracking the truth to −4 Elo on average — the
drift still reads +53 on average and rises past +75, because a rating fitted
only against the floor saturates: once the policy beats every defined reference
almost always, the record stops being able to say how far above them it is, and
the estimate settles below the truth. Two runs under the same physics share that
bias almost exactly, so the *difference* between two runs at matched steps is
meaningful even though neither number is. On 719 against 727 that difference is
+4.6 ± 33 Elo before 727's resume at 128M and +60.5 ± 30 after it — the same
regime change three independent instruments found, with the bias divided out.

*How precisely can the floor-anchored offset be known at all?* The Fisher
information of a Bradley-Terry model is a weighted graph Laplacian with edge
weight ``games_ij · c²·p_ij(1−p_ij)``, and ``Var(r_i − r_j)`` is the effective
resistance between i and j in that graph. Both facts were checked numerically
against the calibrator's own standard errors before this module was written.
The resulting standard error grows over a run as the direct live-vs-scripted
edge saturates, and how fast is a measurement nobody has taken here.

*Is the filter moving more than the evidence licenses?* One update's games carry
a finite amount of information; convert it to a standard error and report the
update's rating change in units of it. Calibrate expectations against a healthy
run before treating a number as an alarm: 719 sits at a median of 1.33 with a
95th percentile of 4.34 and single updates as high as 14.8, so this is a
distribution to watch rather than a threshold to trip. It is a *scale-free*
statistic, which is its point — a large move on thin evidence and a small one on
thick evidence can both be wrong, and neither shows up in the raw delta.

*Is the fit anywhere near a degenerate corner?* Complete separation sends a
maximum-likelihood rating to infinity, which is the most likely cause of the
meaningless multi-hundred-Elo swings earlier attempts at an online fitter hit.
The largest absolute rating on the ladder trips first and trips immediately.

One structural caveat is worth stating rather than discovering later. The
evaluator records match counts only from the live policy's perspective, so the
graph this module builds today is a **star** centred on the live policy: every
reference connects to it and to nothing else. Effective resistance to scripted
therefore reduces to the reciprocal of that single edge's weight, and the
Fiedler value reduces to a statement about the weakest spoke. Both are still the
right numbers to watch — the star topology *is* the diagnosis — and the code is
written for a general graph because Phases 1 and 4 add the missing edges.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping, Sequence

import numpy as np

from boost_and_broadside.train.rl.bradley_terry import (
    RATING_SCALE,
    fisher_information,
    fit_single_rating,
    win_probability,
)

# Rating points per factor-of-10 in odds, as a logistic slope. Matches
# bradley_terry._C; re-derived rather than imported to keep the private name
# private.
_C = float(np.log(10.0) / RATING_SCALE)

# Live-policy node label in the graph. Not a legal roster label, so it cannot
# collide with a real opponent.
LIVE_NODE = "__live__"

# Updates pooled before fitting the live policy's record. Deliberately not a
# config knob: it trades the same bias against variance the K-factor does — the
# policy improves *within* the window, so a boxcar of length W describes what it
# was worth about W/2 updates ago — and Phase 2 chooses the number that matters
# by replaying 719's recorded counts against its calibrated curve rather than by
# taste. Eight is enough episodes to fit against a saturated rung without
# reaching back far enough for the policy to have visibly moved. Note this is
# counted in updates, unlike EloEvalConfig.window_size, which counts episodes.
DEFAULT_WINDOW_UPDATES = 8


def pair_information(rating: float, opponent_rating: float, games: float) -> float:
    """Fisher information about ``rating − opponent_rating`` from ``games`` games.

    ``games`` counts every rated episode including draws. Draws are scored as
    half a win throughout this codebase, which models no draw process at all and
    so leaves them as ordinary observations of the expected score; dropping them
    instead would throw away most of the information in tie-heavy matchups. On a
    real whole-run record against the random agent (2794W/10L/1120T) the two
    conventions differ by a factor of nearly 50.
    """
    probability = float(win_probability(rating, opponent_rating))
    return _C**2 * float(games) * probability * (1.0 - probability)


def potential(laplacian: np.ndarray, source: int, sink: int) -> np.ndarray | None:
    """Solve ``L φ = e_source − e_sink`` for the unit-current potential.

    Returns None when the two nodes sit in different components of the graph, in
    which case the rating difference is not identified and no finite answer
    exists. A least-squares solve would otherwise return a plausible-looking
    vector that solves nothing, so the residual is checked rather than trusted.

    ``φ`` is reused by the allocator in later phases: the potential drop across
    an edge is what makes a game on it worth playing.
    """
    size = laplacian.shape[0]
    if not 0 <= source < size or not 0 <= sink < size or source == sink:
        raise ValueError(f"source and sink must be distinct nodes in [0, {size}), got {source}, {sink}")
    current = np.zeros(size, dtype=np.float64)
    current[source] = 1.0
    current[sink] = -1.0
    solution, *_ = np.linalg.lstsq(laplacian, current, rcond=None)
    scale = max(float(np.abs(laplacian).max()), 1.0)
    if not np.allclose(laplacian @ solution, current, atol=1e-8 * scale):
        return None
    return solution


def effective_resistance(laplacian: np.ndarray, source: int, sink: int) -> float:
    """Return ``Var(r_source − r_sink)`` in squared rating points, or inf."""
    solution = potential(laplacian, source, sink)
    if solution is None:
        return float("inf")
    return float(solution[source] - solution[sink])


def fiedler_value(laplacian: np.ndarray) -> float:
    """Return the algebraic connectivity — the second-smallest eigenvalue.

    Zero (to numerical precision) means the pool has split into components that
    share no games, so their ratings float relative to each other and small
    count changes move them arbitrarily. It is the early warning for the
    connectivity failure that information-weighted allocation can cause.
    """
    if laplacian.shape[0] < 2:
        return float("nan")
    eigenvalues = np.linalg.eigvalsh(laplacian)
    return float(eigenvalues[1])


class LiveEloDiagnostics:
    """Per-update diagnostics for the live rating, accumulated over a window.

    A single update's games are too few to fit a rating against the weak end of
    the ladder — the live policy may complete no episodes at all against a
    saturated rung — so the record is pooled over a sliding window of updates.
    The window is a diagnostic convenience here and carries no lag cost, because
    nothing reads these numbers back into training.
    """

    def __init__(
        self, window_updates: int = DEFAULT_WINDOW_UPDATES, scripted_label: str = "scripted"
    ) -> None:
        if window_updates < 1:
            raise ValueError(f"window_updates must be at least 1, got {window_updates}")
        self.scripted_label = scripted_label
        self._window: deque[dict[str, tuple[int, int, int]]] = deque(maxlen=window_updates)
        self._previous_live_elo: float | None = None

    def _windowed(self) -> dict[str, tuple[float, float, float]]:
        """Sum the window into one win/loss/tie record per opponent label."""
        totals: dict[str, tuple[float, float, float]] = {}
        for record in self._window:
            for label, (win, loss, tie) in record.items():
                previous = totals.get(label, (0.0, 0.0, 0.0))
                totals[label] = (previous[0] + win, previous[1] + loss, previous[2] + tie)
        return totals

    def update(
        self,
        *,
        live_elo: float,
        match_counts: Mapping[str, tuple[int, int, int]],
        ratings: Mapping[str, float],
        stationary: Mapping[str, bool],
    ) -> dict[str, float]:
        """Record one update's outcomes and return the diagnostic metrics.

        Args:
            live_elo:     The filter's current rating for the live policy.
            match_counts: This update's win/loss/tie per opponent label, from
                          the live policy's perspective.
            ratings:      Current rating of every roster entry, by label.
            stationary:   Whether each label is a defined gauge reference. Only
                          these are used for the drift detector, because only
                          their ratings are independent of the live policy.

        Returns:
            Metrics under the ``elo_diag/`` prefix. Non-finite values are
            dropped rather than logged, so a saturated window shows up as a
            missing point instead of a spike that ruins every chart's y-axis.
        """
        self._window.append({label: tuple(wlt) for label, wlt in match_counts.items()})  # type: ignore[misc]
        previous = self._previous_live_elo
        self._previous_live_elo = live_elo

        metrics: dict[str, float] = {}
        totals = self._windowed()
        rated = {label: wlt for label, wlt in totals.items() if label in ratings}

        metrics["elo_diag/window_games"] = float(
            sum(win + loss + tie for win, loss, tie in rated.values())
        )
        if ratings:
            metrics["elo_diag/max_abs_rating"] = max(
                abs(float(value)) for value in (*ratings.values(), live_elo)
            )

        self._add_drift(metrics, live_elo, rated, ratings, stationary)
        self._add_resistance(metrics, live_elo, rated, ratings)
        self._add_movement(metrics, live_elo, previous, match_counts, ratings)
        return {key: value for key, value in metrics.items() if np.isfinite(value)}

    def _add_drift(
        self,
        metrics: dict[str, float],
        live_elo: float,
        rated: Mapping[str, tuple[float, float, float]],
        ratings: Mapping[str, float],
        stationary: Mapping[str, bool],
    ) -> None:
        """Fit the live rating against defined references and report the gap."""
        gauge = [label for label in rated if stationary.get(label, False)]
        if gauge:
            implied, stderr = self._implied(gauge, rated, ratings)
            if np.isfinite(implied):
                metrics["elo_diag/implied_gauge_elo"] = implied
                metrics["elo_diag/implied_gauge_stderr"] = stderr
                metrics["elo_diag/drift_vs_gauge"] = live_elo - implied
        if self.scripted_label in rated:
            implied, stderr = self._implied([self.scripted_label], rated, ratings)
            if np.isfinite(implied):
                metrics["elo_diag/implied_scripted_elo"] = implied
                metrics["elo_diag/implied_scripted_stderr"] = stderr
                metrics["elo_diag/drift_vs_scripted"] = live_elo - implied

    @staticmethod
    def _implied(
        labels: Sequence[str],
        rated: Mapping[str, tuple[float, float, float]],
        ratings: Mapping[str, float],
    ) -> tuple[float, float]:
        """Maximum-likelihood rating explaining the record against ``labels``."""
        opponents = np.array([ratings[label] for label in labels], dtype=np.float64)
        # Half-win draws: the convention every other caller of the fitter uses.
        wins = np.array([rated[label][0] + 0.5 * rated[label][2] for label in labels])
        losses = np.array([rated[label][1] + 0.5 * rated[label][2] for label in labels])
        return fit_single_rating(opponents, wins, losses)

    def _add_resistance(
        self,
        metrics: dict[str, float],
        live_elo: float,
        rated: Mapping[str, tuple[float, float, float]],
        ratings: Mapping[str, float],
    ) -> None:
        """Report how precisely the floor-anchored offset is currently known."""
        if self.scripted_label not in rated:
            return
        labels = [LIVE_NODE, *rated]
        node_ratings = [live_elo, *(ratings[label] for label in rated)]
        size = len(labels)
        games = np.zeros((size, size), dtype=np.float64)
        for index, label in enumerate(rated, start=1):
            win, loss, tie = rated[label]
            games[0, index] = games[index, 0] = win + loss + tie
        # fisher_information is the Fisher matrix of a BT fit, which for
        # pairwise comparisons *is* the weighted graph Laplacian: off-diagonal
        # −w_ij, diagonal the row sum. Same object under two names.
        laplacian = fisher_information(games, np.asarray(node_ratings, dtype=np.float64))
        variance = effective_resistance(laplacian, 0, labels.index(self.scripted_label))
        if np.isfinite(variance) and variance > 0.0:
            metrics["elo_diag/se_live_vs_scripted"] = float(np.sqrt(variance))
        metrics["elo_diag/fiedler"] = fiedler_value(laplacian)
        metrics["elo_diag/pool_size"] = float(size)

    def _add_movement(
        self,
        metrics: dict[str, float],
        live_elo: float,
        previous: float | None,
        match_counts: Mapping[str, tuple[int, int, int]],
        ratings: Mapping[str, float],
    ) -> None:
        """Report this update's rating change in units of its own evidence."""
        if previous is None:
            return
        delta = live_elo - previous
        metrics["elo_diag/live_elo_delta"] = delta
        information = sum(
            pair_information(live_elo, ratings[label], win + loss + tie)
            for label, (win, loss, tie) in match_counts.items()
            if label in ratings
        )
        if information <= 0.0:
            return
        stderr = float(1.0 / np.sqrt(information))
        metrics["elo_diag/update_stderr"] = stderr
        metrics["elo_diag/movement_z"] = abs(delta) / stderr
