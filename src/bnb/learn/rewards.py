"""Rewards (rewards spec): the game's per-tick Events to five zero-sum reward levels.

A pure function of (ledger, events) to (rewards per ship per level, ledger'), called by the
rollout every tick. The ledger (B, N + 1, N), [attacker, target], holds each pair's damage
that has not healed; row N is the boundary, so each living ship's column sums to its
shield deficit. The component functions are v1's math (fixtures in tests/reference/rewards)
and take any leading axes; `team` is each ship's team, 0 or 1, broadcasting against them.
"""

import dataclasses

import torch

from bnb.contracts import REWARD_LEVELS, Events

__all__ = ["REWARD_LEVELS", "RewardConfig", "accumulate_ledger", "attribute_recharge",
           "blank_ledger", "damage_events", "enemy_team_share", "kill_death_events",
           "outcome_events", "reward_step"]


@dataclasses.dataclass(frozen=True)
class RewardConfig:
    """Level weights (rewards spec 1, v1's values), in REWARD_LEVELS order: damage per
    damage_unit points, kill/death per death, each objective level per unit of the mode's
    events, the outcome per match result."""

    damage: float = 0.274
    kill_death: float = 0.283
    objective_progress: float = 2.0
    objective: float = 2.0
    outcome: float = 1.0
    damage_unit: float = 100.0  # points, for every ship type

    @property
    def weights(self) -> tuple[float, ...]:
        return tuple(getattr(self, level) for level in REWARD_LEVELS)


def blank_ledger(batch: int, num_ships: int, device=None) -> torch.Tensor:
    """(B, N + 1, N) zeros: no outstanding damage."""
    return torch.zeros((batch, num_ships + 1, num_ships), device=device)


def _shares(values: torch.Tensor) -> torch.Tensor:
    """values (..., A, T) as fractions of each target's column total (0 for an empty one)."""
    total = values.sum(-2, keepdim=True)
    return torch.where(total > 0.0, values / torch.where(total > 0.0, total, 1.0), 0.0)


def _enemies(team: torch.Tensor) -> torch.Tensor:
    """(..., N, N) [i, j]: ships i and j are on different teams."""
    return team[..., :, None] != team[..., None, :]


# ------------------------------------------------------------------ the ledger

def accumulate_ledger(ledger: torch.Tensor, raw: torch.Tensor,
                      combat_applied: torch.Tensor) -> torch.Tensor:
    """The ledger plus each attacker's share of the applied damage (..., T), in proportion
    to its raw impact raw (..., A, T)."""
    return ledger + _shares(raw) * combat_applied[..., None, :]


def attribute_recharge(ledger: torch.Tensor, recharge: torch.Tensor, deficit: torch.Tensor):
    """Recharge (..., T) attributed pro rata over the ledger's rows (..., R, T), with the
    fraction min(recharge / deficit, 1) (0 for a zero deficit). Returns (attributed
    (..., R, T), unattributed = recharge - attributed over the rows (..., T), ledger)."""
    positive = deficit > 0.0
    fraction = torch.where(positive, (recharge / torch.where(positive, deficit, 1.0))
                           .clamp_max(1.0), 0.0)
    attributed = ledger * fraction[..., None, :]
    return attributed, recharge - attributed.sum(-2), ledger - attributed


# ------------------------------------------------------------------ the levels' events

def enemy_team_share(value: torch.Tensor, team: torch.Tensor) -> torch.Tensor:
    """Each ship's value (..., N) split over the opposing team's slots (the dead included):
    every ship receives the other team's total over its own team's size."""
    enemies = _enemies(team)
    own_size = (~enemies).sum(-1)
    return torch.where(enemies, value[..., None, :], 0.0).sum(-1) / own_size


def outcome_events(team: torch.Tensor, result: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
    """(..., N, 1): +result for team 0 and -result for team 1 on done environments, the
    living and the dead alike. result (...,) is +1 a team-0 win, -1 a team-1 win, 0 a draw."""
    sign = torch.where(team == 0, 1.0, -1.0)
    return (sign * torch.where(done, result, 0.0)[..., None])[..., None]


def kill_death_events(team: torch.Tensor, combat_death: torch.Tensor,
                      boundary_death: torch.Tensor, raw: torch.Tensor,
                      ledger: torch.Tensor) -> torch.Tensor:
    """(..., N, 4) death, kill, ally kill, unforced.

    A dead ship's kill is shared half by the final tick's raw hits and half by the ship
    rows' outstanding damage (by the hits alone when nothing is outstanding), so every
    share is held. Shares held by enemies are kills; shares held by allies blame them and
    pay the opposing team twice. A boundary death pays the opposing team.
    """
    raw_share = _shares(raw)
    outstanding = ledger.sum(-2, keepdim=True) > 0.0
    held = torch.where(outstanding, _shares(ledger), raw_share)
    share = combat_death[..., None, :] * (0.5 * raw_share + 0.5 * held)  # (..., A, T)
    enemies = _enemies(team)
    ally_share = torch.where(enemies, 0.0, share)
    kill = torch.where(enemies, share, 0.0).sum(-1)
    value = boundary_death + 2.0 * ally_share.sum(-2)
    death = -(combat_death | boundary_death).to(share.dtype)
    return torch.stack([death, kill, -ally_share.sum(-1), enemy_team_share(value, team)], -1)


def damage_events(team: torch.Tensor, combat_applied: torch.Tensor,
                  boundary_applied: torch.Tensor, raw: torch.Tensor, recharge: torch.Tensor,
                  recharge_attributed: torch.Tensor, unattributed: torch.Tensor,
                  unit: float) -> torch.Tensor:
    """(..., N, 6) taken, dealt, friendly fire, recharge, charge-back, unforced, in units of
    `unit` points.

    Applied damage (..., T) is split over its attackers by raw impact. Damage to an enemy
    pays the shooter; to an ally it blames the shooter and pays the opposing team twice.
    Recharge refunds the target and charges back each attacker pro rata (an ally's blame
    is refunded). Boundary damage pays, and its unattributed healing charges, the opposing
    team.
    """
    applied = _shares(raw) * combat_applied[..., None, :]  # (..., A, T)
    enemies = _enemies(team)
    ally_applied = torch.where(enemies, 0.0, applied).sum(-1)
    ally_attributed = torch.where(enemies, 0.0, recharge_attributed).sum(-1)
    enemy_attributed = torch.where(enemies, recharge_attributed, 0.0).sum(-1)
    value = 2.0 * ally_applied + boundary_applied - unattributed - 2.0 * ally_attributed
    events = torch.stack([-(combat_applied + boundary_applied),
                          torch.where(enemies, applied, 0.0).sum(-1), -ally_applied, recharge,
                          ally_attributed - enemy_attributed, enemy_team_share(value, team)], -1)
    return events / unit


# ------------------------------------------------------------------ one tick (v2)

def reward_step(config: RewardConfig, ledger: torch.Tensor, events: Events,
                team: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(rewards (B, N, 5) in REWARD_LEVELS order, ledger') for one tick (rewards spec 3).

    ledger (B, N + 1, N); events in sim order; team (N,) or (B, N).
    """
    N = events.died.shape[-1]
    raw, boundary = events.damage, events.boundary_damage
    hit = raw.sum(-2)  # raw hit damage per target
    total = hit + boundary
    lost = torch.where(total > 0.0, events.shield_lost / torch.where(total > 0.0, total, 1.0), 0.0)
    combat_applied, boundary_applied = hit * lost, boundary * lost

    ships = accumulate_ledger(ledger[..., :N, :], raw, combat_applied)
    ledger = torch.cat([ships, (ledger[..., N, :] + boundary_applied)[..., None, :]], -2)

    died = events.died
    combat_death = died & (hit > 0.0)
    kill_death = kill_death_events(team, combat_death, died & ~combat_death, raw, ships)

    restored = events.shield_restored
    attributed, _, ledger = attribute_recharge(ledger, restored, ledger.sum(-2))
    ship_attributed = attributed[..., :N, :]
    damage = damage_events(team, combat_applied, boundary_applied, raw, restored,
                           ship_attributed, restored - ship_attributed.sum(-2),
                           config.damage_unit)
    outcome = outcome_events(team, events.result.to(raw.dtype), events.done)[..., 0]

    levels = (damage.sum(-1), kill_death.sum(-1), events.objective_progress, events.objective,
              outcome)
    rewards = torch.stack([weight * level for weight, level in zip(config.weights, levels)], -1)
    # A death makes the damage done to the ship stick; a finished match starts afresh.
    # A ship's row (its claims on others) survives its own death.
    ledger = torch.where(died[..., None, :] | events.done[..., None, None], 0.0, ledger)
    return rewards, ledger
