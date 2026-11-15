"""Returns (learner spec 3): per-level GAE over the critic's value levels, each reward
level's offence and defence (`bnb.contracts.VALUE_LEVELS`), and team mixing with the
weighted level sum. v1's math; fixtures `learner_gae` and `learner_team_aggregate`.

Shapes are time-major, (L, R, S, ...) per ship, with the critic's value levels last.

The advantage weighs each part: A = sum_k w_k A_k over the value levels, w_k 1 but on the
offence of the premium levels (`PREMIUM_LEVELS`), where it is the offence premium
(`bnb.learn.ppo.LearnerConfig.offence_premium`). Defence keeps its sign (a cost is
negative), so with every weight 1 the sum is the zero-sum levels' advantage, and a premium
p is p A_offence + A_defence, the advantage of p times what the ship earns from the enemy
less what the enemy takes from it. The weights touch only the advantage: each level's
returns, the critic's targets, are its part's own.
"""

import torch

from bnb.contracts import REWARD_LEVELS, REWARD_PARTS, VALUE_LEVELS, level_part

# The levels whose offence the premium weighs (decided 2026-11-14): damage and kill/death.
# The objective levels and the outcome stay zero-sum: a premium on captures would pay both
# teams for trading zones in self-play.
PREMIUM_LEVELS = ("damage", "kill_death")


def part_weights(offence_premium: float) -> tuple[float, ...]:
    """Each value level's weight in the advantage, in VALUE_LEVELS order: `offence_premium`
    on the offence of the premium levels, 1 on every other part."""
    premium = {level_part(level, "offence") for level in PREMIUM_LEVELS}
    return tuple(offence_premium if name in premium else 1.0 for name in VALUE_LEVELS)


def per_part(per_level: tuple[float, ...]) -> tuple[float, ...]:
    """A value per reward level (REWARD_LEVELS order) for each of its parts, in VALUE_LEVELS
    order: a level's discount, trace decay or team spirit is both its parts'."""
    if len(per_level) != len(REWARD_LEVELS):
        raise ValueError(f"{per_level} needs one value per reward level, {REWARD_LEVELS}")
    return tuple(value for value in per_level for _ in REWARD_PARTS)


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
    parts `team_aggregate` weighs and sums. per_level (..., N, levels)."""
    weight = team_weights(team, alive, per_level.dtype)
    return per_level + team_spirit * (weight @ per_level)


def team_aggregate(
    team: torch.Tensor,
    alive: torch.Tensor,
    advantages: torch.Tensor,
    returns: torch.Tensor,
    team_spirit: torch.Tensor,
    weights: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The weighted level sum with team mixing, (..., N) each:
    A_i = sum_k w_k (A_ik + s_k mean_{living teammates j != i} A_jk), and the unweighted
    sum of the returns the same way (the return scale's). team (..., N) ids, alive (..., N);
    advantages, returns (..., N, levels); team_spirit and weights (levels,). A ship with no
    living teammate mixes in nothing."""
    weight = team_weights(team, alive, advantages.dtype)  # (..., N, N)

    def mixed(per_level: torch.Tensor) -> torch.Tensor:
        mean = weight @ per_level  # (..., N, levels): the mean over living teammates
        return per_level + team_spirit * mean

    return (mixed(advantages) * weights).sum(-1), mixed(returns).sum(-1)
