"""Win rates, their standard errors, the adaptive allocation of games and a Bradley-Terry
(Elo) fit."""

import heapq
import math

import torch

ELO_PER_NAT = 400.0 / math.log(10.0)


def standard_error(score: float, games: int) -> float:
    """The standard error of a score (wins plus half the draws) over `games` games, from the
    score shrunk half a game towards 0.5 (finite at 0 and at every game won); infinite with
    no games."""
    if games <= 0:
        return math.inf
    rate = (score + 0.5) / (games + 1.0)
    return math.sqrt(rate * (1.0 - rate) / games)


def allocate(pairings: list[tuple[float, int]], games: int, step: int = 2) -> list[int]:
    """New games per pairing, `games` in all, in lots of `step` (a lot keeps the two sides
    level), each lot to the pairing whose standard error is the largest at that point.
    `pairings` holds each pairing's (score, games) so far; as games are added a pairing's
    error is projected with its current shrunk rate."""

    def projected(index: int, total: int) -> float:
        score, played = pairings[index]
        if total == 0:
            return math.inf
        rate = (score + 0.5) / (played + 1.0)
        return math.sqrt(rate * (1.0 - rate) / total)

    added = [0] * len(pairings)
    heap = [(-projected(index, played), index) for index, (_, played) in enumerate(pairings)]
    heapq.heapify(heap)
    for _ in range(games // step):
        _, index = heapq.heappop(heap)
        added[index] += step
        heapq.heappush(heap, (-projected(index, pairings[index][1] + added[index]), index))
    return added


def bradley_terry(
    results: list[tuple[int, int, int, float]], agents: int, anchor: int = 0, anchor_elo: float = 1000.0,
    prior_elo: float = 1000.0, iterations: int = 50,
) -> tuple[list[float], list[float]]:
    """Elo ratings and their standard errors from pairings `results`: (agent A, agent B,
    games, A's score), agents numbered 0 .. agents - 1. P(A beats B) = sigmoid((r_A - r_B) /
    ELO_PER_NAT); a draw is half a win. `anchor` is fixed at `anchor_elo` (standard error
    0). A Gaussian prior of standard deviation `prior_elo` about the anchor keeps unbeaten or
    unconnected agents finite. Newton's method on the log-likelihood."""
    rating = torch.zeros(agents, dtype=torch.float64)
    results = [row for row in results if row[0] != row[1]]  # self-play carries no rating information
    if not results or agents < 2:
        return [anchor_elo] * agents, [0.0] * agents
    first = torch.tensor([row[0] for row in results])
    second = torch.tensor([row[1] for row in results])
    games = torch.tensor([float(row[2]) for row in results], dtype=torch.float64)
    score = torch.tensor([float(row[3]) for row in results], dtype=torch.float64)
    precision = (ELO_PER_NAT / prior_elo) ** 2
    free = torch.ones(agents, dtype=torch.bool)
    free[anchor] = False
    for _ in range(iterations):
        expected = torch.sigmoid(rating[first] - rating[second])
        residual = score - games * expected
        gradient = torch.zeros(agents, dtype=torch.float64).index_add_(0, first, residual).index_add_(0, second, -residual)
        gradient -= precision * rating
        weight = games * expected * (1.0 - expected)
        hessian = torch.zeros(agents, agents, dtype=torch.float64)
        hessian.index_put_((first, first), weight, accumulate=True)
        hessian.index_put_((second, second), weight, accumulate=True)
        hessian.index_put_((first, second), -weight, accumulate=True)
        hessian.index_put_((second, first), -weight, accumulate=True)
        hessian += precision * torch.eye(agents, dtype=torch.float64)
        sub = hessian[free][:, free]
        step = torch.linalg.solve(sub, gradient[free])
        rating[free] += step
        if step.abs().max() < 1e-10:
            break
    covariance = torch.zeros(agents, agents, dtype=torch.float64)
    covariance[free.nonzero()[:, 0][:, None], free.nonzero()[:, 0]] = torch.linalg.inv(sub)
    errors = (covariance.diagonal().sqrt() * ELO_PER_NAT).tolist()
    return (rating * ELO_PER_NAT + anchor_elo).tolist(), errors
