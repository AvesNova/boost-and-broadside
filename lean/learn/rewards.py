"""Per-ship rewards on v2's five levels (v2's src/bnb/learn/rewards.py at commit 4cfd1e86).

Levels, in this order: damage, kill_death, objective_progress, objective, outcome. Each is
zero-sum between the teams as v2's is: every event a team earns pairs with the same amount
charged to the other team. A ledger per environment holds each attacker's damage to each
target that has not healed; recharge charges it back, and a kill is shared by it (assists).
The objective levels are zero until a mode with objectives exists; their columns are kept.

v2 splits each level into an offence and a defence part (ten critic parts). Lean keeps
the five levels (one critic target each, the author's decision); the parts' sum is the
level, so the rewards per level are v2's.

The components below are v2's functions with the same mathematics; they take any leading
axes and `team` (..., N) broadcasting against them.
"""

from dataclasses import dataclass

import torch
from torch import Tensor

from lean.learn.config import LearnConfig


@dataclass
class Ledger:
    """Outstanding damage: ships (E, N, N) [attacker, target], each ship's damage to each
    target that has not healed; boundary (E, N), the boundary's. Each living ship's column
    plus its boundary entry is its shield deficit."""

    ships: Tensor
    boundary: Tensor


def blank_ledger(envs: int, slots: int, device=None) -> Ledger:
    """No outstanding damage."""
    return Ledger(ships=torch.zeros(envs, slots, slots, device=device), boundary=torch.zeros(envs, slots, device=device))


def clear_ledger(ledger: Ledger, chosen: Tensor) -> Ledger:
    """The ledger with the environments in `chosen` (E,) bool emptied (re-dealt games)."""
    return Ledger(
        ships=torch.where(chosen[:, None, None], 0.0, ledger.ships),
        boundary=torch.where(chosen[:, None], 0.0, ledger.boundary),
    )


@dataclass
class TickEvents:
    """One tick's events in the form v2's rewards read (v2's `Events`).

    raw (E, N, N) [attacker, target]: raw hit damage of real bullets. boundary (E, N): raw
    boundary damage. shield_lost, shield_restored (E, N): what the shield lost and regained.
    died (E, N) bool, done (E,) bool, win_loss (E, 2) team 0's win and loss weights (a
    result one-hot, a draw neither). present (E, N) bool: the slot is in the match.
    """

    raw: Tensor
    boundary: Tensor
    shield_lost: Tensor
    shield_restored: Tensor
    died: Tensor
    done: Tensor
    win_loss: Tensor
    present: Tensor


def tick_events(step_out, state) -> TickEvents:
    """v2's events from the game's StepOut and the state at the tick's start: the
    attacker-by-target hit damage, the boundary damage and the shield's restoration come
    straight from the game (exact); the shield lost is the damage up to the shield the ship
    had."""
    alive = state.alive
    total = step_out.damage_taken
    lost = torch.where(alive, torch.minimum(state.shield, total), 0.0)
    result = step_out.result
    win_loss = torch.stack([result == 0, result == 1], -1).to(total.dtype)
    return TickEvents(
        raw=step_out.hit_damage, boundary=step_out.boundary_damage, shield_lost=lost,
        shield_restored=step_out.shield_restored, died=step_out.died, done=step_out.done,
        win_loss=win_loss, present=state.exists,
    )


def _column_total(values: Tensor) -> Tensor:
    """(..., 1, N): each target's total over the rows."""
    return values.sum(-2, keepdim=True)


def _shares(values: Tensor, total: Tensor | None = None) -> Tensor:
    """values (..., N, N) as fractions of each target's column total (0 for an empty one)."""
    total = _column_total(values) if total is None else total
    return torch.where(total > 0.0, values / torch.where(total > 0.0, total, 1.0), 0.0)


def _enemies(team: Tensor) -> Tensor:
    """(..., N, N) [i, j]: ships i and j are on different teams."""
    return team[..., :, None] != team[..., None, :]


def attribute_recharge(ledger: Tensor, recharge: Tensor, deficit: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Recharge (..., N) attributed pro rata over the ledger's rows (..., rows, N), with the
    fraction min(recharge / deficit, 1). Returns (attributed (..., rows, N), unattributed
    (..., N), the ledger less the attributed)."""
    positive = deficit > 0.0
    fraction = torch.where(positive, (recharge / torch.where(positive, deficit, 1.0)).clamp_max(1.0), 0.0)
    attributed = ledger * fraction[..., None, :]
    return attributed, recharge - _column_total(attributed)[..., 0, :], ledger - attributed


def enemy_team_share(value: Tensor, team: Tensor, present: Tensor) -> Tensor:
    """Each ship's value (..., N) paid to the opposing team: every present ship receives the
    other team's total over its own team's present count; a slot out of the match none."""
    enemies = _enemies(team)
    total = torch.where(enemies, value[..., None, :], 0.0).sum(-1)
    own_size = (~enemies & present[..., None, :]).sum(-1).clamp_min(1)
    return torch.where(present, total / own_size, 0.0)


def outcome_reward(team: Tensor, win_loss: Tensor, done: Tensor) -> Tensor:
    """(..., N): on done environments each ship's team's win weight less its loss weight,
    the living and the dead alike (+1 a win, -1 a loss, 0 a draw). win_loss (..., 2) is
    team 0's; team 1's is it flipped."""
    win, loss = torch.where(done[..., None], win_loss, 0.0)[..., None, :].unbind(-1)
    on_team0 = team == 0
    return torch.where(on_team0, win, loss) - torch.where(on_team0, loss, win)


def kill_death_reward(team, combat_death, boundary_death, raw, ledger, present) -> Tensor:
    """(..., N): v2's death, kill, ally kill and unforced events summed.

    Death -1 (in combat or at the boundary). A dead ship's kill is shared half by the final
    tick's raw hits and half by the ledger's outstanding damage (by the hits alone when
    nothing is outstanding), so assists earn a share. Shares held by enemies are kills;
    shares held by allies blame them (-share) and pay the opposing team twice. A boundary
    death pays the opposing team 1.
    """
    raw_share = _shares(raw)
    ledger_total = _column_total(ledger)
    held = torch.where(ledger_total > 0.0, _shares(ledger, ledger_total), raw_share)
    share = combat_death[..., None, :] * (0.5 * raw_share + 0.5 * held)
    enemies = _enemies(team)
    ally_share = torch.where(enemies, 0.0, share)
    kill = torch.where(enemies, share, 0.0).sum(-1)
    unforced = enemy_team_share(boundary_death + 2.0 * ally_share.sum(-2), team, present)
    death = (combat_death | boundary_death).to(share.dtype)
    return kill - death - ally_share.sum(-1) + unforced


def damage_reward(team, combat_applied, boundary_applied, raw, recharge, recharge_attributed, unattributed, present) -> Tensor:
    """(..., N) in damage points: v2's taken, dealt, friendly fire, recharge, charge-back,
    refund and unforced events summed.

    Applied damage is split over its attackers by raw impact. Damage to an enemy pays the
    shooter; to an ally it blames the shooter and pays the opposing team twice. Recharge
    refunds the target, charges back each enemy attacker pro rata and refunds each ally
    attacker's blame. Boundary damage pays, and its unattributed healing charges, the
    opposing team.
    """
    applied = _shares(raw) * combat_applied[..., None, :]
    enemies = _enemies(team)
    ally_applied = torch.where(enemies, 0.0, applied).sum(-1)
    ally_attributed = torch.where(enemies, 0.0, recharge_attributed).sum(-1)
    enemy_attributed = torch.where(enemies, recharge_attributed, 0.0).sum(-1)
    unforced = 2.0 * ally_applied + boundary_applied - unattributed - 2.0 * ally_attributed
    return (
        -(combat_applied + boundary_applied)
        + torch.where(enemies, applied, 0.0).sum(-1)
        - ally_applied
        + recharge
        - enemy_attributed
        + ally_attributed
        + enemy_team_share(unforced, team, present)
    )


def reward_step(config: LearnConfig, ledger: Ledger, events: TickEvents, team: Tensor) -> tuple[Tensor, Ledger]:
    """Rewards of one tick, (E, N, LEVELS) float32 for every ship of both teams, weighted
    (damage per `damage_unit` points), and the ledger after the tick (v2's reward_step).
    A slot out of the match is rewarded nothing."""
    raw, present = events.raw, events.present
    hit = _column_total(raw)[..., 0, :]
    total = hit + events.boundary
    lost = torch.where(total > 0.0, events.shield_lost / torch.where(total > 0.0, total, 1.0), 0.0)
    combat_applied, boundary_applied = hit * lost, events.boundary * lost

    ships = ledger.ships + _shares(raw) * combat_applied[..., None, :]
    boundary_row = (ledger.boundary + boundary_applied)[..., None, :]

    died = events.died
    combat_death = died & (hit > 0.0)
    kill_death = kill_death_reward(team, combat_death, died & ~combat_death, raw, ships, present)

    restored = events.shield_restored
    deficit = (_column_total(ships) + boundary_row)[..., 0, :]
    ship_attributed, unattributed, ships = attribute_recharge(ships, restored, deficit)
    _, _, boundary_row = attribute_recharge(boundary_row, restored, deficit)
    damage = damage_reward(team, combat_applied, boundary_applied, raw, restored, ship_attributed, unattributed, present)
    outcome = outcome_reward(team, events.win_loss, events.done)

    zero = torch.zeros_like(outcome)
    levels = (damage / config.damage_unit, kill_death, zero, zero, outcome)
    reward = torch.stack([weight * level for weight, level in zip(config.reward_weights, levels, strict=True)], -1)
    reward = torch.where(present[..., None], reward, 0.0)
    # A death makes the damage done to the ship stick; a finished match starts afresh. A
    # ship's row (its claims on others) survives its own death.
    clear = died[..., None, :] | events.done[..., None, None]
    return reward, Ledger(
        ships=torch.where(clear, 0.0, ships), boundary=torch.where(clear[..., 0, :], 0.0, boundary_row[..., 0, :])
    )
