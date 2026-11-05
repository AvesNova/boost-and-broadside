"""The rating estimators (league spec 4 and 6): the Bradley-Terry fit with standard errors (the
run's ratings, `bnb.league.lanes.RunRatings`, and the offline tournament's) and the offline
tournament's game allocation.

Offline tools in float64 on the CPU (code style, Dtypes). Ratings are on the Elo scale: 400
points per factor of 10 in odds, `P(i beats j) = 1 / (1 + 10^((r_j - r_i)/400))`. Every fit
is a maximum of a concave log-likelihood, found by Newton's method with a backtracking line
search and run to convergence, so the result is the estimator's optimum whatever the start.
"""

import math

import torch

SCALE = 400.0 / math.log(10.0)  # rating points per natural unit of log-odds
DTYPE = torch.float64
NEWTON_STEPS = 200
# On the Newton decrement, in natural units: far past float64 rating noise.
DECREMENT_TOLERANCE = 1e-13


def win_probability(rating: torch.Tensor, opponent: torch.Tensor) -> torch.Tensor:
    """P(a player rated `rating` beats one rated `opponent`), elementwise."""
    return torch.sigmoid((rating - opponent) / SCALE)


def _as_float(values) -> torch.Tensor:
    return torch.as_tensor(values, dtype=DTYPE).cpu()


def _log_likelihood(theta: torch.Tensor, wins: torch.Tensor) -> torch.Tensor:
    """sum_ij wins[i, j] log sigmoid(theta_i - theta_j)."""
    difference = theta[:, None] - theta[None, :]
    return (wins * torch.nn.functional.logsigmoid(difference)).sum()


def _maximise(theta: torch.Tensor, wins: torch.Tensor, free: torch.Tensor) -> torch.Tensor:
    """The Bradley-Terry maximum over the `free` players' strengths (natural units), the others
    held at `theta`. wins (K, K): fractional wins of row over column."""
    theta = theta.clone()
    games = wins + wins.T
    for _ in range(NEWTON_STEPS):
        difference = theta[:, None] - theta[None, :]
        probability = torch.sigmoid(difference)
        gradient = (wins - games * probability).sum(1)[free]
        weight = games * probability * (1.0 - probability)
        hessian = (torch.diag(weight.sum(1)) - weight)[free][:, free]
        # least squares, not solve: a player whose every game is lost to underflow (or one with
        # no games) makes the Hessian singular, and that player's step is then zero
        step = torch.linalg.lstsq(hessian, gradient[:, None], driver="gelsd").solution[:, 0]
        decrement = float(gradient @ step)
        if decrement < DECREMENT_TOLERANCE:
            break
        current = _log_likelihood(theta, wins)
        length = 1.0
        while True:
            trial = theta.clone()
            trial[free] += length * step
            if _log_likelihood(trial, wins) >= current - 1e-15 * abs(float(current)):
                break
            length *= 0.5
            if length < 1e-12:
                return theta
        theta = trial
    return theta


def _fisher(theta: torch.Tensor, games: torch.Tensor, free: torch.Tensor) -> torch.Tensor:
    """The Fisher information over the free players (natural units): the graph Laplacian with
    edge weights n_ij p_ij (1 - p_ij)."""
    probability = torch.sigmoid(theta[:, None] - theta[None, :])
    weight = games * probability * (1.0 - probability)
    return (torch.diag(weight.sum(1)) - weight)[free][:, free]


def fit_ratings(wins, anchor: int, prior_games: float) -> tuple[torch.Tensor, torch.Tensor]:
    """The Bradley-Terry ratings (K,) and their standard errors (K,), with player `anchor`
    pinned at 0 (its error 0). wins (K, K): row's wins over column, draws scored as half a
    win each. Each other player gets `prior_games` virtual games against the anchor, half
    won and half lost, which keeps a clean sweep finite. The errors are the inverse Fisher
    information's diagonal, the prior's games included."""
    wins = _as_float(wins)
    count = wins.shape[0]
    prior = torch.zeros_like(wins)
    others = torch.arange(count) != anchor
    prior[others, anchor] = prior_games / 2.0
    prior[anchor, others] = prior_games / 2.0
    wins = wins + prior
    theta = _maximise(torch.zeros(count, dtype=DTYPE), wins, others)
    stderr = torch.zeros(count, dtype=DTYPE)
    covariance = torch.linalg.inv(_fisher(theta, wins + wins.T, others))
    stderr[others] = covariance.diagonal().sqrt() * SCALE
    return (theta - theta[anchor]) * SCALE, stderr


def allocate(pair_games, ratings, anchor: int, budget: int, minimum_per_pair: int) -> torch.Tensor:
    """How many of `budget` games each pair plays next, (K, K) symmetric int64.

    Each pair's share is its A-optimal value: the rate at which more games between them
    reduce the summed rating variance (the trace of the inverse Fisher information C, with
    the anchor's row and column removed), `p (1 - p) v' C^2 v` with `v = e_i - e_j`, at the
    given ratings (K,) and the pair games played so far (K, K). Shares are floored, raised
    to at least `minimum_per_pair`, and the remainder goes one game each to the pairs with
    the largest shares. With no games yet, the budget is spread evenly over the pairs."""
    pair_games, ratings = _as_float(pair_games), _as_float(ratings)
    count = pair_games.shape[0]
    upper = torch.triu_indices(count, count, 1)
    pairs = upper.shape[1]
    if float(pair_games.sum()) == 0.0:
        value = torch.ones(pairs, dtype=DTYPE)
    else:
        theta = ratings / SCALE
        free = torch.arange(count) != anchor
        covariance = torch.zeros((count, count), dtype=DTYPE)
        index = free.nonzero().flatten()
        covariance[index[:, None], index[None, :]] = torch.linalg.inv(
            _fisher(theta, pair_games, free)
        )
        probability = torch.sigmoid(theta[upper[0]] - theta[upper[1]])
        weight = probability * (1.0 - probability)
        # -d trace(C) / d games_ij = p (1 - p) v' C^2 v, with v = e_i - e_j.
        direction = torch.zeros((pairs, count), dtype=DTYPE)
        direction[torch.arange(pairs), upper[0]] = 1.0
        direction[torch.arange(pairs), upper[1]] = -1.0
        projected = direction @ covariance  # (pairs, K)
        value = weight * (projected * projected).sum(1)
    share = budget * value / value.sum()
    games = share.floor().clamp_min(minimum_per_pair)
    remainder = int(budget - games.sum())
    if remainder > 0:
        order = torch.argsort(share, descending=True, stable=True)
        games[order[:remainder]] += 1.0
    allocation = torch.zeros((count, count), dtype=torch.int64)
    allocation[upper[0], upper[1]] = games.long()
    return allocation + allocation.T
