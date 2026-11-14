"""How well one rating per player fits a field's games (league spec 6): each directly played
pair's observed score against the Bradley-Terry prediction, and a likelihood-ratio test of
Bradley-Terry against the multidimensional Elo with one cyclic component (mElo-2,
`bnb.league.cyclic`; Balduzzi et al. 2018).

**Observed against predicted** (`pair_table`, `deviance`). For each pair with games, the
first player's score s over n games (a draw half a win, a kill share its fraction) and the
posterior's predicted win probability p, with the signed deviance residual

    sign(s - p) sqrt(2 n [s log(s / p) + (1 - s) log((1 - s) / (1 - p))]).

The deviance is the sum of their squares over the played pairs, against a chi-square with
P - (K - 1) degrees of freedom (P played pairs among K players with games): a large one says
the pairs' results do not sit on one line of ratings. The deviance residual is used, not
Pearson's (s - p) / sqrt(p (1 - p) / n), because a pair predicted at p near 0 or 1 that
loses or draws one game in hundreds gives a Pearson residual of many units on one game.
The prediction is the posterior's (with its prior).

**Loops** (`cyclic_test`). Bradley-Terry (k = 0) and mElo-2 (k = 1: each player's rating and
a 2-vector, the log-odds r_i - r_j + c_i' W c_j) fitted by maximum likelihood on the same
games; the statistic is twice the log-likelihood gained, against a chi-square with
min(2K - 3, P - (K - 1)) degrees of freedom (the 2K cyclic coordinates less the three that a
unimodular map of the plane leaves the cyclic term unchanged by, at most the residual
degrees of freedom). A small p-value is evidence of a rock-paper-scissors structure among
the directly played pairs. Without a pair outside a spanning tree (P = K - 1) no loop can
show, and the test is not made.

Both read a fractional outcome as the binomial likelihood reads a draw's half: a kill share
varies less than a win or a loss of the same mean, so with kill-share games the statistics
are conservative (they find structure less often than their nominal rate). Both also take
the games as independent given the ratings; games of one block share a batch's conditions.

Offline statistics in float64 on the CPU.
"""

import math

import torch

from . import cyclic
from .design import Posterior
from .ratings import SCALE

DTYPE = torch.float64
SEEDS = (0, 1, 2)  # mElo's random starts (its likelihood is not concave): the best is kept
CYCLIC_PENALTY = 1e-6  # a tiny L2 on the cyclic vectors, which keeps a sweep finite


def _played(games: torch.Tensor) -> torch.Tensor:
    """(K,) bool: the players with any games."""
    return games.sum(1) > 0


def _relative_entropy_term(observed: float, predicted: float) -> float:
    """observed log(observed / predicted), 0 at observed 0."""
    return 0.0 if observed <= 0.0 else observed * math.log(observed / predicted)


def pair_table(wins: torch.Tensor, posterior: Posterior) -> list[dict]:
    """Every played pair i < j: its games, i's observed score, the predicted probability and
    the signed deviance residual (module docstring), by player index."""
    wins = wins.to(DTYPE)
    games = wins + wins.T
    theta = posterior.ratings / SCALE
    rows = []
    count = wins.shape[0]
    for first in range(count):
        for second in range(first + 1, count):
            played = float(games[first, second])
            if played <= 0.0:
                continue
            score = min(max(float(wins[first, second]) / played, 0.0), 1.0)
            predicted = float(torch.sigmoid(theta[first] - theta[second]))
            predicted = min(max(predicted, 1e-300), 1.0 - 1e-16)
            deviance = (
                2.0
                * played
                * (
                    _relative_entropy_term(score, predicted)
                    + _relative_entropy_term(1.0 - score, 1.0 - predicted)
                )
            )
            residual = math.copysign(math.sqrt(max(deviance, 0.0)), score - predicted)
            rows.append(
                dict(
                    first=first,
                    second=second,
                    games=played,
                    score=score,
                    predicted=predicted,
                    residual=residual,
                )
            )
    return rows


def chi_square_survival(statistic: float, degrees: int) -> float:
    """P(chi-square with `degrees` degrees of freedom > statistic); nan without degrees."""
    if degrees <= 0:
        return math.nan
    half = torch.tensor(degrees / 2.0, dtype=DTYPE)
    return float(
        torch.special.gammaincc(half, torch.tensor(max(statistic, 0.0) / 2.0, dtype=DTYPE))
    )


def deviance(wins: torch.Tensor, posterior: Posterior) -> dict:
    """The deviance over the played pairs, its degrees of freedom and p-value."""
    table = pair_table(wins, posterior)
    players = int(_played(wins + wins.T).sum())
    degrees = len(table) - (players - 1)
    statistic = sum(row["residual"] ** 2 for row in table)
    return dict(
        statistic=statistic,
        degrees=degrees,
        p_value=chi_square_survival(statistic, degrees),
        pairs=len(table),
    )


def cyclic_test(wins: torch.Tensor) -> dict:
    """The likelihood-ratio test of Bradley-Terry against mElo-2 (module docstring): the
    statistic, its degrees of freedom and the p-value (nan where no loop can show)."""
    wins = wins.to(DTYPE)
    games = wins + wins.T
    kept = _played(games)
    wins, games = wins[kept][:, kept], games[kept][:, kept]
    players = wins.shape[0]
    pairs = int((torch.triu(games, 1) > 0).sum())
    degrees = min(2 * players - 3, pairs - (players - 1))
    if players < 3 or degrees <= 0:
        return dict(statistic=0.0, degrees=max(degrees, 0), p_value=math.nan)
    transitive = cyclic.fit(wins, games, 0)
    loop = min(
        (cyclic.fit(wins, games, 1, CYCLIC_PENALTY, seed) for seed in SEEDS),
        key=lambda model: model.loss,
    )
    statistic = max(2.0 * (transitive.loss - loop.loss), 0.0)
    return dict(
        statistic=statistic, degrees=degrees, p_value=chi_square_survival(statistic, degrees)
    )


def report(wins: torch.Tensor, posterior: Posterior, names: list[str], worst: int = 5) -> dict:
    """The field's goodness of fit: the deviance test, the loop test, and the `worst` pairs
    by absolute residual (with names)."""
    table = pair_table(wins, posterior)
    table.sort(key=lambda row: -abs(row["residual"]))
    return dict(
        deviance=deviance(wins, posterior),
        loops=cyclic_test(wins),
        worst_pairs=[
            dict(row, first=names[row["first"]], second=names[row["second"]])
            for row in table[:worst]
        ],
    )


def describe(fit: dict) -> list[str]:
    """`report`'s result as lines for a table's foot."""
    fitted, loops = fit["deviance"], fit["loops"]
    lines = [
        f"observed against predicted: deviance {fitted['statistic']:.1f} on "
        f"{fitted['degrees']} degrees of freedom over {fitted['pairs']} played pairs, "
        f"p = {fitted['p_value']:.3g}",
    ]
    if math.isnan(loops["p_value"]):
        lines.append("loops: not tested (no pair outside a spanning tree of the played pairs)")
    else:
        lines.append(
            f"loops (Bradley-Terry against mElo-2): likelihood ratio {loops['statistic']:.1f} "
            f"on {loops['degrees']} degrees of freedom, p = {loops['p_value']:.3g}"
        )
    for row in fit["worst_pairs"]:
        lines.append(
            f"  {row['first']} against {row['second']}: {row['score']:.3f} observed, "
            f"{row['predicted']:.3f} predicted over {row['games']:.0f} games "
            f"(residual {row['residual']:+.1f})"
        )
    return lines
