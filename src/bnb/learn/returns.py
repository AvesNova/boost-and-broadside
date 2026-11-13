"""Returns (learner spec 3): each value level's reward, per-level GAE, and team mixing with
the level sum. v1's math; fixtures `learner_gae` and `learner_team_aggregate`.

Shapes are time-major, (L, R, S, ...) per ship, with the critic's value levels last.
"""

import torch

from bnb.contracts import REWARD_LEVELS, SUMMED_LEVEL

# The critic's value levels the returns support: one per reward level, or the single level
# over their sum.
VALUE_LAYOUTS = (REWARD_LEVELS, (SUMMED_LEVEL,))


def value_rewards(rewards: torch.Tensor, levels: tuple[str, ...]) -> torch.Tensor:
    """The reward of each of the critic's value levels (..., len(levels)), from the
    weighted rewards per reward level (..., len(REWARD_LEVELS)): the reward levels as
    they are, or for the single level (`SUMMED_LEVEL`) their sum."""
    if levels == REWARD_LEVELS:
        return rewards
    if levels == (SUMMED_LEVEL,):
        return rewards.sum(-1, keepdim=True)
    raise ValueError(f"the critic's levels {levels} are not one of {VALUE_LAYOUTS}")


def per_level_gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    terminated: torch.Tensor,
    next_value: torch.Tensor,
    next_done: torch.Tensor,
    discount: torch.Tensor,
    trace_decay: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advantages and returns (L, R, S, levels): plain GAE per level with its own discounts.

    rewards, values (L, R, S, levels); terminated (L, R): the episode ended on step t, so the
    trace is cut after t (a timeout is a real end); next_value (R, S, levels) and next_done (R,):
    the bootstrap after the last step, cut where either the last step or next_done ends the
    episode. discount (gamma) and trace_decay (lambda): (levels,).
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


def team_weights(team: torch.Tensor, alive: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """(..., N, N) [i, j]: 1 / the living teammates of ship i where j is one of them (j != i),
    else 0. team (..., N) ids, alive (..., N)."""
    ships = team.shape[-1]
    others = ~torch.eye(ships, dtype=torch.bool, device=team.device)
    mates = (team[..., :, None] == team[..., None, :]) & others & alive[..., None, :]
    weight = mates.to(dtype)
    return weight / weight.sum(-1, keepdim=True).clamp_min(1.0)


def team_mixed_levels(
    team: torch.Tensor, alive: torch.Tensor, per_level: torch.Tensor, team_spirit: torch.Tensor
) -> torch.Tensor:
    """(..., N, levels): each level's A_ik + s_k mean_{living teammates j != i} A_jk, the
    parts `team_aggregate` sums. per_level (..., N, levels)."""
    weight = team_weights(team, alive, per_level.dtype)
    return per_level + team_spirit * (weight @ per_level)


def team_aggregate(
    team: torch.Tensor,
    alive: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    team_spirit: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The level sum with team mixing, (..., N) each:
    A_i = sum_k (A_ik + s_k mean_{living teammates j != i} A_jk), and the same of the
    returns. team (..., N) ids, alive (..., N); advantages, returns (..., N, levels); team_spirit
    (levels,). A ship with no living teammate mixes in nothing."""
    weight = team_weights(team, alive, advantages.dtype)  # (..., N, N)

    def mixed(per_level: torch.Tensor) -> torch.Tensor:
        mean = weight @ per_level  # (..., N, levels): the mean over living teammates
        return (per_level + team_spirit * mean).sum(-1)

    return mixed(advantages), mixed(returns)
