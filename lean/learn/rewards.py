"""Per-ship rewards on the four levels."""

import torch
from torch import Tensor

from lean.learn.config import DAMAGE, KILL, LEVELS, OUTCOME, SHAPING, LearnConfig


def team_mean(values: Tensor, team: Tensor, exists: Tensor) -> Tensor:
    """Mean of `values` (E, N) over the ships that exist on each ship's own team, given
    back per ship (E, N). Ships that do not exist get 0."""
    result = torch.zeros_like(values)
    for side in (0, 1):
        members = (exists & (team == side)).to(values.dtype)
        mean = (values * members).sum(1, keepdim=True) / members.sum(1, keepdim=True).clamp(min=1.0)
        result = result + mean * members
    return result


def shaping_terms(state) -> tuple[Tensor, Tensor]:
    """Facing and closing terms toward the nearest living enemy, each (E, N) in [-1, 1].

    Facing is the cosine between the nose direction and the direction to that enemy;
    closing is the velocity component toward it in metres per second (not yet scaled).
    Ships with no living enemy get 0.
    """
    alive = state.alive & state.exists
    enemy = alive[:, None, :] & alive[:, :, None] & (state.team[:, :, None] != state.team[:, None, :])
    delta_x = state.x[:, None, :] - state.x[:, :, None]
    delta_y = state.y[:, None, :] - state.y[:, :, None]
    distance = torch.sqrt(delta_x**2 + delta_y**2).masked_fill(~enemy, float("inf"))
    nearest_distance, nearest = distance.min(dim=2)
    found = torch.isfinite(nearest_distance)
    scale = torch.where(found, nearest_distance, torch.ones_like(nearest_distance)).clamp(min=1e-6)
    unit_x = delta_x.gather(2, nearest[:, :, None]).squeeze(2) / scale
    unit_y = delta_y.gather(2, nearest[:, :, None]).squeeze(2) / scale
    facing = (state.angle_cos * unit_x + state.angle_sin * unit_y) * found
    closing = state.speed * (state.heading_cos * unit_x + state.heading_sin * unit_y) * found
    return facing, closing


def rewards(step_out, state, config: LearnConfig, shaping: float = 0.0) -> Tensor:
    """Rewards of one tick, (E, N, LEVELS) float32, for every ship of both teams.

    `state` is the state the tick started from (its `exists` and `team` are those of the
    episode; its `alive` says who could act). `shaping` is the taper in [0, 1].

    How each level is kept independent of fleet size:

    - outcome: on the tick the episode ends, each ship that existed (dead included) gets
      +win_reward if its team won, -win_reward if it lost, 0 for a draw. One value per
      ship, never summed over the team.
    - kill and death, damage: an event term per ship (own death, own kills through
      `killer`, own damage dealt and taken, own friendly fire and friendly kills as
      penalties; damage in units of `damage_unit`). The reward is
      individual_share * own term + (1 - individual_share) * mean of the term over the
      ships that exist on the ship's team. A team mean divides by the team's size at
      reset, so a ship's return is bounded by events per ship, not events per fleet.
    - shaping: facing and closing speed toward the nearest living enemy for living ships,
      times `shaping`; already per ship.

    v1 instead divided every reward by the total ship count.
    """
    exists = state.exists
    team = state.team
    dtype = step_out.damage_dealt.dtype
    reward = torch.zeros(*exists.shape, LEVELS, dtype=dtype, device=exists.device)

    result = step_out.result[:, None]
    decided = step_out.done[:, None] & (result >= 0) & (result < 2)
    sign = torch.where(result == team, 1.0, -1.0)
    reward[..., OUTCOME] = config.win_reward * sign * (decided & exists)

    died = step_out.died & exists
    slots = exists.shape[1]
    has_killer = died & (step_out.killer >= 0)
    killer = step_out.killer.clamp(min=0)
    killer_team = team.gather(1, killer)
    enemy_kill = (has_killer & (killer_team != team)).to(dtype)
    friendly_kill = (has_killer & (killer_team == team) & (killer != torch.arange(slots, device=team.device))).to(dtype)
    kill_credit = torch.zeros_like(enemy_kill).scatter_add(1, killer, enemy_kill - friendly_kill)
    kill_term = config.kill_reward * kill_credit - config.death_penalty * died.to(dtype)

    damage_term = (
        config.damage_dealt_reward * step_out.damage_dealt
        - config.damage_taken_penalty * step_out.damage_taken
        - config.friendly_damage_penalty * step_out.friendly_damage
    ) / config.damage_unit

    share = config.individual_share
    for level, term in ((KILL, kill_term), (DAMAGE, damage_term)):
        term = term * exists
        reward[..., level] = share * term + (1.0 - share) * team_mean(term, team, exists)

    if shaping > 0.0 and (config.facing_reward > 0.0 or config.closing_reward > 0.0):
        facing, closing = shaping_terms(state)
        closing = (closing / config.speed_unit).clamp(-1.0, 1.0)
        reward[..., SHAPING] = shaping * (config.facing_reward * facing + config.closing_reward * closing)
    return reward
