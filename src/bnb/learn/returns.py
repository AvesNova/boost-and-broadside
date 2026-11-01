"""Returns (learner spec 3): per-level GAE, team mixing with the level sum, and the outcome
level's categorical targets. v1's math; fixtures `learner_gae` and `learner_team_aggregate`.

Shapes are time-major, (L, R, S, ...) per ship, with the K reward levels last.
"""

import torch


def per_level_gae(rewards: torch.Tensor, values: torch.Tensor, terminated: torch.Tensor,
                  next_value: torch.Tensor, next_done: torch.Tensor, discount: torch.Tensor,
                  trace_decay: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Advantages and returns (L, R, S, K): plain GAE per level with its own discounts.

    rewards, values (L, R, S, K); terminated (L, R): the episode ended on step t, so the
    trace is cut after t (a timeout is a real end); next_value (R, S, K) and next_done (R,):
    the bootstrap after the last step, cut where either the last step or next_done ends the
    episode. discount (gamma) and trace_decay (lambda): (K,).
    """
    keep = (~terminated).to(values.dtype)[..., None, None]  # (L, R, 1, 1)
    last_keep = keep[-1] * (~next_done).to(values.dtype)[:, None, None]
    advantages = torch.empty_like(values)
    running = torch.zeros_like(next_value)
    following = next_value
    for t in reversed(range(values.shape[0])):
        cut = last_keep if t == values.shape[0] - 1 else keep[t]
        delta = rewards[t] + discount * following * cut - values[t]
        running = delta + discount * trace_decay * cut * running
        advantages[t] = running
        following = values[t]
    return advantages, advantages + values


def team_aggregate(team: torch.Tensor, alive: torch.Tensor, advantages: torch.Tensor,
                   returns: torch.Tensor, team_spirit: torch.Tensor
                   ) -> tuple[torch.Tensor, torch.Tensor]:
    """The level sum with team mixing, (..., N) each:
    A_i = sum_k (A_ik + s_k mean_{living teammates j != i} A_jk), and the same of the
    returns. team (..., N) ids, alive (..., N); advantages, returns (..., N, K); team_spirit
    (K,). A ship with no living teammate mixes in nothing."""
    ships = team.shape[-1]
    others = ~torch.eye(ships, dtype=torch.bool, device=team.device)
    mates = (team[..., :, None] == team[..., None, :]) & others & alive[..., None, :]
    weight = mates.to(advantages.dtype)
    weight = weight / weight.sum(-1, keepdim=True).clamp_min(1.0)  # (..., N, N)

    def mixed(per_level: torch.Tensor) -> torch.Tensor:
        mean = weight @ per_level  # (..., N, K): the mean over living teammates
        return (per_level + team_spirit * mean).sum(-1)

    return mixed(advantages), mixed(returns)


def outcome_values(probabilities: torch.Tensor, weight: float) -> torch.Tensor:
    """The outcome level's value in reward units: (P(win) - P(loss)) times its weight."""
    return (probabilities[..., 0] - probabilities[..., 2]) * weight
