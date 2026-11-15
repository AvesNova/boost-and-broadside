"""Rewards (rewards spec): the game's per-tick Events to five zero-sum reward levels, each
in two parts, offence and defence (`bnb.contracts.REWARD_PARTS`).

A pure function of (ledger, events) to (rewards per ship per level part, ledger'), called by
the rollout every tick. The ledger holds each pair's damage that has not healed, [attacker,
target], with the boundary as a pseudo-attacker: each living ship's column of ship rows
plus its boundary entry sums to its shield deficit. It is the spec's (B, N + 1, N) ledger
kept as its ship rows and its boundary row, so the step writes each without a copy.

The component functions are v1's math (fixtures in tests/reference/rewards) and take any
leading axes; `team` is each ship's team, 0 or 1, broadcasting against them.

Every event has one part, by whose ships it is about (`REWARD_EVENTS`). Offence: the
enemy's ships, damage dealt and its charge-back when the target heals, kills, and what the
enemy team's own mistakes pay (its friendly fire and boundary losses, the unforced events);
defence: the own team's ships, damage taken, healing, deaths, and the blame for the own
team's friendly fire with its refund. Each pairs with an event of the other part on the
other team, so a team's offence is the opposing team's defence negated, and the level stays
zero-sum. The objective levels and the outcome split by sign, the gain offence and the loss
defence: a ship's objective amounts are the mode's credits for its team's progress and its
charges for the enemy's, each event in its part before the tick's sum (the game gives them
apart, `Events.objective`, so a credit and a charge on one tick do not net), and its outcome
is its team's win weight less its loss weight.

The step is memory-bound on its (B, N, N) matrices, so it reads each as few times as it
can: the components optionally take the column totals (..., 1, N) the step already has,
instead of summing a derived matrix again.
"""

import dataclasses

import torch

from bnb.contracts import REWARD_LEVELS, REWARD_PARTS, VALUE_LEVELS, Events, level_part
from bnb.contracts.tree import contract
from bnb.game.modes.rules import DEATHMATCH_OUTCOMES

__all__ = [
    "REWARD_LEVELS",
    "REWARD_PARTS",
    "VALUE_LEVELS",
    "Ledger",
    "RewardConfig",
    "accumulate_ledger",
    "attribute_recharge",
    "blank_ledger",
    "damage_events",
    "enemy_team_share",
    "kill_death_events",
    "level_rewards",
    "outcome_events",
    "REWARD_EVENTS",
    "reward_step",
    "reward_step_events",
]


@dataclasses.dataclass(frozen=True)
class RewardConfig:
    """Level weights (rewards spec 1), in REWARD_LEVELS order: damage per damage_unit
    points, kill/death per death, each objective level per unit of the mode's events, the
    outcome per match result. v1's values times 10, the same factor for every level: with
    v1's the returns were about a tenth of the units the critic's bins and the return
    scale's floor assume (each level's 5th to 95th percentile within +-0.6, so the bins near
    0 resolved little; compile-overhaul notes, "Return units"). The advantages are divided
    by the return scale, so the factor leaves the policy's gradient as it was.

    deathmatch_outcome: what the outcome level (and the outcome head's target) counts as
    team deathmatch's win and loss (`bnb.game.modes.rules.DEATHMATCH_OUTCOMES`):
    "kill_lead" (the default), the result one-hot by the sign of the final kill difference,
    a draw neither, as the ratings score it by default; or "kill_share", the kill share
    (share, 1 - share) (the rollout sets it on the rules: `ModeRules.deathmatch_outcome`)."""

    damage: float = 2.74
    kill_death: float = 2.83
    objective_progress: float = 20.0
    objective: float = 20.0
    outcome: float = 10.0
    damage_unit: float = 100.0  # points, for every ship type
    deathmatch_outcome: str = "kill_lead"

    def __post_init__(self) -> None:
        if self.deathmatch_outcome not in DEATHMATCH_OUTCOMES:
            raise ValueError(
                f"reward.deathmatch_outcome {self.deathmatch_outcome!r} is not one of "
                f"{DEATHMATCH_OUTCOMES}"
            )

    @property
    def weights(self) -> tuple[float, ...]:
        """The level weights in REWARD_LEVELS order."""
        return tuple(getattr(self, level) for level in REWARD_LEVELS)


@contract
class Ledger:
    """Outstanding damage (rewards spec 2): ships (B, N, N) [attacker, target], each ship's
    damage to each target that has not healed; boundary (B, N), the boundary's (the spec's
    row N). Reward state, carried by the runtime."""

    ships: torch.Tensor
    boundary: torch.Tensor


def blank_ledger(batch: int, num_ships: int, device=None) -> Ledger:
    """No outstanding damage."""
    return Ledger(
        ships=torch.zeros((batch, num_ships, num_ships), device=device),
        boundary=torch.zeros((batch, num_ships), device=device),
    )


def _column_total(values: torch.Tensor) -> torch.Tensor:
    """(..., 1, N): each target's total over the rows."""
    return values.sum(-2, keepdim=True)


def _shares(values: torch.Tensor, total: torch.Tensor | None = None) -> torch.Tensor:
    """values (..., N, N) as fractions of each target's column total (0 for an empty one)."""
    total = _column_total(values) if total is None else total
    return torch.where(total > 0.0, values / torch.where(total > 0.0, total, 1.0), 0.0)


def _enemies(team: torch.Tensor) -> torch.Tensor:
    """(..., N, N) [i, j]: ships i and j are on different teams."""
    return team[..., :, None] != team[..., None, :]


# ------------------------------------------------------------------ the ledger


def accumulate_ledger(
    ledger: torch.Tensor,
    raw: torch.Tensor,
    combat_applied: torch.Tensor,
    *,
    raw_total: torch.Tensor | None = None,
) -> torch.Tensor:
    """The ledger plus each attacker's share of the applied damage (..., N), in proportion
    to its raw impact raw (..., N, N)."""
    return ledger + _shares(raw, raw_total) * combat_applied[..., None, :]


def attribute_recharge(
    ledger: torch.Tensor,
    recharge: torch.Tensor,
    deficit: torch.Tensor,
    *,
    ledger_total: torch.Tensor | None = None,
):
    """Recharge (..., N) attributed pro rata over the ledger's rows (..., rows, N), with the
    fraction min(recharge / deficit, 1) (0 for a zero deficit). Returns (attributed
    (..., rows, N), unattributed = recharge - attributed over the rows (..., N), ledger)."""
    positive = deficit > 0.0
    fraction = torch.where(
        positive, (recharge / torch.where(positive, deficit, 1.0)).clamp_max(1.0), 0.0
    )
    attributed = ledger * fraction[..., None, :]
    total = (
        _column_total(attributed) if ledger_total is None else ledger_total * fraction[..., None, :]
    )
    return attributed, recharge - total[..., 0, :], ledger - attributed


# ------------------------------------------------------------------ the levels' events


def enemy_team_share(
    value: torch.Tensor, team: torch.Tensor, present: torch.Tensor | None = None
) -> torch.Tensor:
    """Each ship's value (..., N) split over the opposing team's slots (the dead included):
    every ship receives the other team's total over its own team's size. present (..., N)
    bool: the slots in the match (a benched slot is not, `bnb.game.api.benched_of`; None:
    every slot); the share then goes to the own team's present slots only, over their
    count, and a slot out of the match receives none, so the level stays zero-sum."""
    enemies = _enemies(team)
    total = torch.where(enemies, value[..., None, :], 0.0).sum(-1)
    if present is None:
        return total / (~enemies).sum(-1)
    own_size = (~enemies & present[..., None, :]).sum(-1).clamp_min(1)
    return torch.where(present, total / own_size, 0.0)


def outcome_events(team: torch.Tensor, win_loss: torch.Tensor, done: torch.Tensor) -> torch.Tensor:
    """(..., N, 2) win and loss: each ship's team's win weight and its loss weight negated, on
    done environments, the living and the dead alike. win_loss (..., 2) is team 0's
    (`Events.win_loss`; team 1's is it flipped): a result one-hot, a draw neither, or team
    deathmatch's kill share (share, 1 - share). Their sum is the ship's team's score in
    [-1, 1]: +1 a win, -1 a loss, 0 a draw, team deathmatch's 2 share - 1."""
    win, loss = torch.where(done[..., None], win_loss, 0.0)[..., None, :].unbind(-1)
    on_team0 = team == 0
    return torch.stack([torch.where(on_team0, win, loss), -torch.where(on_team0, loss, win)], -1)


def kill_death_events(
    team: torch.Tensor,
    combat_death: torch.Tensor,
    boundary_death: torch.Tensor,
    raw: torch.Tensor,
    ledger: torch.Tensor,
    *,
    raw_total: torch.Tensor | None = None,
    ledger_total: torch.Tensor | None = None,
    present: torch.Tensor | None = None,
) -> torch.Tensor:
    """(..., N, 4) death, kill, ally kill, unforced (split over the own team's present slots,
    `enemy_team_share`).

    A dead ship's kill is shared half by the final tick's raw hits and half by the ship
    rows' outstanding damage (by the hits alone when nothing is outstanding), so every
    share is held. Shares held by enemies are kills; shares held by allies blame them and
    pay the opposing team twice. A boundary death pays the opposing team.
    """
    raw_share = _shares(raw, raw_total)
    ledger_total = _column_total(ledger) if ledger_total is None else ledger_total
    held = torch.where(ledger_total > 0.0, _shares(ledger, ledger_total), raw_share)
    share = combat_death[..., None, :] * (0.5 * raw_share + 0.5 * held)  # (..., N, N)
    enemies = _enemies(team)
    ally_share = torch.where(enemies, 0.0, share)
    kill = torch.where(enemies, share, 0.0).sum(-1)
    value = boundary_death + 2.0 * ally_share.sum(-2)
    death = -(combat_death | boundary_death).to(share.dtype)
    unforced = enemy_team_share(value, team, present)
    return torch.stack([death, kill, -ally_share.sum(-1), unforced], -1)


def damage_events(
    team: torch.Tensor,
    combat_applied: torch.Tensor,
    boundary_applied: torch.Tensor,
    raw: torch.Tensor,
    recharge: torch.Tensor,
    recharge_attributed: torch.Tensor,
    unattributed: torch.Tensor,
    unit: float,
    *,
    raw_total: torch.Tensor | None = None,
    present: torch.Tensor | None = None,
) -> torch.Tensor:
    """(..., N, 7) taken, dealt, friendly fire, recharge, charge-back, refund, unforced, in
    units of `unit` points (the unforced split over the own team's present slots,
    `enemy_team_share`).

    Applied damage (..., N) is split over its attackers by raw impact. Damage to an enemy
    pays the shooter; to an ally it blames the shooter and pays the opposing team twice.
    Recharge refunds the target and charges back each enemy attacker pro rata (the
    charge-back) and refunds each ally attacker's blame (the refund). Boundary damage pays,
    and its unattributed healing charges, the opposing team.
    """
    applied = _shares(raw, raw_total) * combat_applied[..., None, :]  # (..., N, N)
    enemies = _enemies(team)
    ally_applied = torch.where(enemies, 0.0, applied).sum(-1)
    ally_attributed = torch.where(enemies, 0.0, recharge_attributed).sum(-1)
    enemy_attributed = torch.where(enemies, recharge_attributed, 0.0).sum(-1)
    value = 2.0 * ally_applied + boundary_applied - unattributed - 2.0 * ally_attributed
    events = torch.stack(
        [
            -(combat_applied + boundary_applied),
            torch.where(enemies, applied, 0.0).sum(-1),
            -ally_applied,
            recharge,
            -enemy_attributed,
            ally_attributed,
            enemy_team_share(value, team, present),
        ],
        -1,
    )
    return events / unit


# ------------------------------------------------------------------ one tick (v2)


# Each event's level and part (diagnostics audit, group 15), in the order
# `reward_step_events` gives them: the damage level's seven, the kill/death level's four,
# then each other level's gain and loss.
REWARD_EVENTS = (
    ("damage", "defence", "damage_taken"),
    ("damage", "offence", "damage_dealt"),
    ("damage", "defence", "friendly_fire"),
    ("damage", "defence", "recharge"),
    ("damage", "offence", "charge_back"),
    ("damage", "defence", "friendly_fire_refund"),
    ("damage", "offence", "damage_unforced"),
    ("kill_death", "defence", "death"),
    ("kill_death", "offence", "kill"),
    ("kill_death", "defence", "ally_kill"),
    ("kill_death", "offence", "kill_death_unforced"),
    ("objective_progress", "offence", "objective_progress_gain"),
    ("objective_progress", "defence", "objective_progress_loss"),
    ("objective", "offence", "objective_gain"),
    ("objective", "defence", "objective_loss"),
    ("outcome", "offence", "win"),
    ("outcome", "defence", "loss"),
)


def level_rewards(rewards: torch.Tensor) -> torch.Tensor:
    """(..., len(REWARD_LEVELS)): each reward level's reward, the sum of its parts, from the
    rewards per level part (..., len(VALUE_LEVELS))."""
    return rewards.unflatten(-1, (len(REWARD_LEVELS), len(REWARD_PARTS))).sum(-1)


def reward_step(
    config: RewardConfig,
    ledger: Ledger,
    events: Events,
    team: torch.Tensor,
    present: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Ledger]:
    """(rewards (B, N, len(REWARD_LEVELS)) in REWARD_LEVELS order, ledger') for one tick (rewards
    spec 3): each level's reward, its parts summed (`level_rewards`).

    Events in sim order; team (N,) or (B, N); present (B, N) bool or None, as for
    `reward_step_events`.
    """
    rewards, ledger, _ = reward_step_events(config, ledger, events, team, present)
    return level_rewards(rewards), ledger


def reward_step_events(
    config: RewardConfig,
    ledger: Ledger,
    events: Events,
    team: torch.Tensor,
    present: torch.Tensor | None = None,
) -> tuple[torch.Tensor, Ledger, torch.Tensor]:
    """(rewards (B, N, len(VALUE_LEVELS)) per level part in VALUE_LEVELS order, ledger', each
    event's weighted reward (B, N, len(REWARD_EVENTS))) for one tick: a part's reward is the
    sum of its events', a level's of its parts'. present (B, N) bool: the slots in the match
    (None: every slot); a benched slot (team-size sampling) takes no share of the enemy's
    unforced events (`enemy_team_share`) and is rewarded nothing (its outcome included):
    it never plays."""
    raw, boundary = events.damage, events.boundary_damage
    raw_total = _column_total(raw)
    old_total = _column_total(ledger.ships)
    hit = raw_total[..., 0, :]  # raw hit damage per target
    total = hit + boundary
    lost = torch.where(total > 0.0, events.shield_lost / torch.where(total > 0.0, total, 1.0), 0.0)
    combat_applied, boundary_applied = hit * lost, boundary * lost

    # The accumulated rows' totals follow without a sum: a hit target's shares sum to one.
    ships = accumulate_ledger(ledger.ships, raw, combat_applied, raw_total=raw_total)
    ships_total = old_total + combat_applied[..., None, :]
    boundary_row = (ledger.boundary + boundary_applied)[..., None, :]

    died = events.died
    combat_death = died & (hit > 0.0)
    kill_death = kill_death_events(
        team,
        combat_death,
        died & ~combat_death,
        raw,
        ships,
        raw_total=raw_total,
        ledger_total=ships_total,
        present=present,
    )

    restored = events.shield_restored
    deficit = (ships_total + boundary_row)[..., 0, :]
    ship_attributed, unattributed, ships = attribute_recharge(
        ships, restored, deficit, ledger_total=ships_total
    )
    _, _, boundary_row = attribute_recharge(
        boundary_row, restored, deficit, ledger_total=boundary_row
    )
    damage = damage_events(
        team,
        combat_applied,
        boundary_applied,
        raw,
        restored,
        ship_attributed,
        unattributed,
        config.damage_unit,
        raw_total=raw_total,
        present=present,
    )
    outcome = outcome_events(team, events.win_loss.to(raw.dtype), events.done)

    by_level = (
        damage,
        kill_death,
        events.objective_progress,
        events.objective,
        outcome,
    )
    event_rewards = torch.cat(
        [weight * level for weight, level in zip(config.weights, by_level, strict=True)], -1
    )
    if present is not None:
        event_rewards = torch.where(present[..., None], event_rewards, 0.0)
    # each part's events, summed in the events' order (static indices: one graph)
    columns = {name: [] for name in VALUE_LEVELS}
    for index, (level, part, _) in enumerate(REWARD_EVENTS):
        columns[level_part(level, part)].append(event_rewards[..., index])
    rewards = torch.stack([sum(columns[name][1:], columns[name][0]) for name in VALUE_LEVELS], -1)
    # A death makes the damage done to the ship stick; a finished match starts afresh.
    # A ship's row (its claims on others) survives its own death.
    clear = died[..., None, :] | events.done[..., None, None]
    return (
        rewards,
        Ledger(
            ships=torch.where(clear, 0.0, ships),
            boundary=torch.where(clear[..., 0, :], 0.0, boundary_row[..., 0, :]),
        ),
        event_rewards,
    )
