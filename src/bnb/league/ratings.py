"""The rating estimators (league spec 4 and 6): the Bradley-Terry fit with standard errors (the
run's ratings, `bnb.league.lanes.RunRatings`, and the offline tournament's) and the offline
tournament's allocation of its matchup blocks.

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


def allocate_blocks(
    pair_games,
    ratings,
    anchor: int,
    blocks: int,
    block_games: int,
    allowed,
    prior_games: float,
) -> list[tuple[int, int]]:
    """Which pair each of `blocks` blocks of `block_games` games plays next: (i, j) with
    i < j, chosen greedily by A-optimal value.

    Each block goes to the allowed pair whose games most reduce the summed rating variance:
    the trace of C, the inverse Fisher information over the players other than the anchor,
    at the given ratings (K,), the pair games played so far (K, K) and the blocks already
    chosen. The information holds the fit's prior games against the anchor (`fit_ratings`'s
    `prior_games`), so it is defined before any game is played. n more games between i and j,
    whose win probability is p, add n w v v' to the information (w = p (1 - p), v = e_i - e_j,
    the anchor's entry dropped), which lowers the trace by n w v'C^2 v / (1 + n w v'C v)
    (Sherman-Morrison); C is updated the same way. allowed (K, K) bool: the pairs that may
    play (symmetric)."""
    pair_games, ratings = _as_float(pair_games), _as_float(ratings)
    allowed = torch.as_tensor(allowed, dtype=torch.bool).cpu()
    count = pair_games.shape[0]
    upper = torch.triu_indices(count, count, 1)
    upper = upper[:, allowed[upper[0], upper[1]]]
    if upper.shape[1] == 0:
        raise ValueError("no pair is allowed to play")
    free = torch.arange(count) != anchor
    games = pair_games.clone()
    games[free, anchor] += prior_games
    games[anchor, free] += prior_games
    theta = ratings / SCALE
    covariance = torch.linalg.inv(_fisher(theta, games, free))
    # each pair's direction v over the free players (the anchor's entry dropped)
    position = torch.cumsum(free.long(), 0) - 1
    direction = torch.zeros((upper.shape[1], count - 1), dtype=DTYPE)
    for side, sign in ((upper[0], 1.0), (upper[1], -1.0)):
        kept = free[side]
        direction[torch.arange(upper.shape[1])[kept], position[side[kept]]] = sign
    probability = torch.sigmoid(theta[upper[0]] - theta[upper[1]])
    weight = block_games * probability * (1.0 - probability)
    chosen = []
    for _ in range(blocks):
        projected = direction @ covariance  # (pairs, K - 1): C v per pair
        spread = (projected * direction).sum(1)  # v'C v
        value = weight * (projected * projected).sum(1) / (1.0 + weight * spread)
        best = int(torch.argmax(value))
        chosen.append((int(upper[0, best]), int(upper[1, best])))
        column = projected[best]
        covariance = covariance - weight[best] * torch.outer(column, column) / (
            1.0 + weight[best] * spread[best]
        )
    return chosen
