"""Five zero-sum reward levels, one critic head each.

Events are grouped into levels by time scale and each level is one reward
component, read by one critic head: the match outcome, completed captures,
capture-meter progress, kills and deaths, and damage. A component returns the
sum of its level's events, already multiplied by the level weight, so rewards
are in win units: a win is 1. Events are still computed one by one and logged
through :meth:`RewardComponent.events`.

Every level is zero-sum by construction. An event charges the side it happens
to ``w`` and pays the side that caused it ``ratio * w``; if nobody on the
opposing side caused it (the boundary, friendly fire), the counterpart is split
evenly over the opposing team, dead slots included. At every ratio 1 each
level's reward summed over both teams is zero on every tick, which
``tests/env/test_reward_heads.py`` checks against real rollouts.

Rewards own spatial credit: each one is written onto the ship that earned it,
and the match outcome is the only quantity broadcast across a team. PPO's
per-head GAE then solves temporal credit only.

Adding an event
---------------
Add it to the ``events`` of the level whose time scale it shares. A new level
is a new critic head: add a component, list its name in
``REWARD_COMPONENT_NAMES``, give it a weight in ``RewardConfig``, a gamma and
lambda in ``config/defaults.py`` and a tier in ``_TIER`` in train/rl/ppo.py.
"""

import dataclasses
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any

import torch

from boost_and_broadside.config import RewardConfig, ShipConfig, ZoneRole
from boost_and_broadside.constants import EPS
from boost_and_broadside.env.frontline import zone_membership
from boost_and_broadside.env.outcome import outcome_masks
from boost_and_broadside.env.state import TensorState

#: The levels, in critic-head order.
REWARD_COMPONENT_NAMES: tuple[str, ...] = (
    "outcome",
    "zone_capture",
    "zone_progress",
    "kill_death",
    "damage",
)

#: Levels whose reward is identical for every ship on a side by construction --
#: a function of team and match result alone, paid to the living and the dead
#: alike -- so their value is estimated once per environment off the global
#: token. ``tests/train/test_global_value.py`` checks the claim on rollouts.
GLOBAL_VALUE_COMPONENTS: frozenset[str] = frozenset({"outcome"})


class RewardComponent(ABC):
    """One reward level: the weighted sum of its events, per ship.

    ``weight`` is the level weight times the schedule's tier scale, and
    ``payout_ratio`` the ratio the paying side receives; ppo.py rewrites both
    once per update and the env wrapper re-reads them through
    ``refresh_component_weights()``.
    """

    name: str
    #: Event names, in the order :meth:`events` stacks them.
    event_names: tuple[str, ...]

    def __init__(self, weight: float, payout_ratio: float = 1.0) -> None:
        self.weight = weight
        self.payout_ratio = payout_ratio

    @abstractmethod
    def events(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """Per-ship events of this level, before the level weight.

        Args:
            prev_state: State snapshot immediately before physics/damage.
            actions: (B, N, 3) actions that produced next_state.
            next_state: State after physics + damage (before env reset).
            dones: (B,) bool — game-over flags.

        Returns:
            (B, N, E) float32, one column per entry of ``event_names``.
        """

    def compute(
        self,
        prev_state: TensorState,
        actions: torch.Tensor,
        next_state: TensorState,
        dones: torch.Tensor,
    ) -> torch.Tensor:
        """(B, N) reward: the weighted sum of this level's events."""
        return self.weight * self.events(prev_state, actions, next_state, dones).sum(-1)


# ---------------------------------------------------------------------------
# Payee helpers
# ---------------------------------------------------------------------------


def enemy_team_share(value: torch.Tensor, state: TensorState) -> torch.Tensor:
    """Split each ship's ``value`` evenly over the opposing team's slots.

    Args:
        value: (B, N) amounts, one per ship.

    Returns:
        (B, N): what each ship receives from the opposing team's amounts. The
        divisor is the recipient team's slot count, dead slots included.
    """
    enemy = state.ship_team_id[:, :, None] != state.ship_team_id[:, None, :]
    recipients = (~enemy).sum(-1).clamp_min(1)  # (B, N) own team size
    return (enemy * value[:, None, :]).sum(-1) / recipients


def _enemy_pairs(state: TensorState) -> torch.Tensor:
    """(B, N_a, N_t) True where a and t are on opposing teams."""
    return state.ship_team_id[:, :, None] != state.ship_team_id[:, None, :]


def _route_by_cause(amount: torch.Tensor, state: TensorState) -> tuple[torch.Tensor, torch.Tensor]:
    """Route a per-pair payment by who caused it.

    An enemy attacker is paid directly. An allied attacker is charged the same
    amount as blame, and the opposing team is paid it twice: once for the event
    nobody on its side caused, once as the counterpart of the blame.

    Args:
        amount: (B, N_a, N_t) payment owed for what a did to t.

    Returns:
        ``(direct, opposing)``: (B, N) paid to each attacker (negative for
        blame), and (B, N) per ship of the amount owed to its opposing team,
        indexed by the ship whose team the amount is charged against.
    """
    enemy = _enemy_pairs(state)
    direct = torch.where(enemy, amount, -amount).sum(2)  # (B, N_a)
    friendly = torch.where(enemy, torch.zeros_like(amount), amount).sum(2)  # (B, N_a)
    return direct, 2.0 * friendly


# ---------------------------------------------------------------------------
# The five levels
# ---------------------------------------------------------------------------


class OutcomeReward(RewardComponent):
    """The match result, +1 won, -1 lost, 0 drew, broadcast to the team.

    Paid on the terminal transition; a timeout is a real result, never a
    bootstrap.
    """

    name = "outcome"
    event_names = ("result",)

    def events(self, prev_state, actions, next_state, dones):
        del prev_state, actions
        team0_won, team1_won, _ = outcome_masks(next_state, dones)
        result = (team0_won.float() - team1_won.float()).unsqueeze(1)  # (B, 1)
        team0 = next_state.ship_team_id == 0
        return torch.where(team0, result, -result).unsqueeze(-1)


class ZoneCaptureReward(RewardComponent):
    """A completed capture, priced by who held the attempt.

    The attacking side is paid ``ratio`` and the defending side charged one per
    completed zone, each split half by presence on the completing tick and half
    by the attempt's ledgers (see ``env/frontline._update_capture_ledgers``).
    """

    name = "zone_capture"
    event_names = ("captured", "lost")

    def events(self, prev_state, actions, next_state, dones):
        del prev_state, actions, dones
        gained = self.payout_ratio * next_state.ship_capture_gain
        return torch.stack([gained, -next_state.ship_capture_loss], dim=-1)


class ZoneProgressReward(RewardComponent):
    """Every movement of a capture meter, split by who showed up.

    The side a meter favours splits ``ratio * |delta|`` among its ships inside
    the zone, and the other side splits the charge ``|delta|`` among its ships
    outside it, dead ones included. Ships of the losing side who are inside are
    contesting and are not charged. The completion tick is the zone-capture
    level's, so it pays nothing here.
    """

    name = "zone_progress"
    event_names = ("held", "absent")

    def __init__(self, weight: float, payout_ratio: float, world_size: tuple[float, float]):
        super().__init__(weight, payout_ratio)
        self.world_size = world_size

    def events(self, prev_state, actions, next_state, dones):
        del actions, dones
        delta = next_state.zone_capture_progress - prev_state.zone_capture_progress  # (B, Z)
        captured = (next_state.team0_captured | next_state.team1_captured).unsqueeze(1)
        delta = torch.where(captured, torch.zeros_like(delta), delta)
        # Progress runs toward the attacker, who is the side not holding the zone.
        roles = next_state.zone_roles
        t0_defense = (roles == int(ZoneRole.TEAM0_DEFENSE)).float()
        t1_defense = (roles == int(ZoneRole.TEAM1_DEFENSE)).float()
        credit = delta * t1_defense - delta * t0_defense  # (B, Z), + favours team 0

        inside = zone_membership(
            next_state.ship_pos, next_state.zone_pos, next_state.zone_radius, self.world_size
        )  # (B, N, Z)
        team0 = (next_state.ship_team_id == 0).unsqueeze(-1)  # (B, N, 1)
        team1 = ~team0
        favours0 = (credit > 0).unsqueeze(1)  # (B, 1, Z)
        favours1 = (credit < 0).unsqueeze(1)
        magnitude = credit.abs().unsqueeze(1)  # (B, 1, Z)

        paid = (favours0 & team0 & inside) | (favours1 & team1 & inside)
        charged = (favours0 & team1 & ~inside) | (favours1 & team0 & ~inside)
        opponent = (favours0 & team1) | (favours1 & team0)
        charged = torch.where(charged.any(1, keepdim=True), charged, opponent)
        paid_n = paid.sum(dim=1, keepdim=True).clamp(min=1)
        charged_n = charged.sum(dim=1, keepdim=True).clamp(min=1)
        held = (paid.float() * magnitude * self.payout_ratio / paid_n).sum(-1)
        absent = -(charged.float() * magnitude / charged_n).sum(-1)
        return torch.stack([held, absent], dim=-1)


class KillDeathReward(RewardComponent):
    """Deaths, and the kills credited for them.

    A dying ship is charged 1, whatever the cause. A combat death is shared
    among its attackers, half by the raw impact of the final tick (the killing
    hit lands on an empty shield, so its *applied* damage is zero) and half by
    outstanding applied damage; with no outstanding damage the raw hit takes
    the whole share. An enemy attacker is paid ``ratio * share``; an allied one
    is blamed the same, and the opposing team paid it twice over. A boundary
    death, and any share nobody holds, pays the opposing team. A ship is never
    blamed for its own death.
    """

    name = "kill_death"
    event_names = ("death", "kill", "ally_kill", "unforced")

    def events(self, prev_state, actions, next_state, dones):
        del prev_state, actions, dones
        combat = next_state.ship_combat_death.float()  # (B, N_t)
        boundary = next_state.ship_boundary_death.float()
        death = -(combat + boundary)

        raw = next_state.damage_matrix  # (B, N_a, N_t)
        ledger = next_state.cumulative_damage_matrix
        raw_total = raw.sum(1, keepdim=True)  # (B, 1, N_t)
        ledger_total = ledger.sum(1, keepdim=True)
        raw_weight = (raw_total > 0).float()
        ledger_weight = (ledger_total > 0).float()
        halves = (raw_weight + ledger_weight).clamp(min=1.0)
        share = (
            raw_weight * raw / raw_total.clamp(min=EPS)
            + ledger_weight * ledger / ledger_total.clamp(min=EPS)
        ) / halves  # (B, N_a, N_t), columns sum to 1 or 0
        share = share * combat.unsqueeze(1)

        enemy = _enemy_pairs(next_state)
        num_ships = enemy.shape[-1]
        itself = torch.eye(num_ships, dtype=torch.bool, device=enemy.device).unsqueeze(0)
        ratio = self.payout_ratio
        kill = ratio * (share * enemy).sum(2)  # (B, N_a)
        ally_amount = ratio * share * (~enemy & ~itself)
        ally_kill = -ally_amount.sum(2)
        unheld = ratio * (combat - share.sum(1)).clamp(min=0.0) + ratio * boundary  # (B, N_t)
        opposing = 2.0 * ally_amount.sum(2) + unheld  # owed by each ship's team
        unforced = enemy_team_share(opposing, next_state)
        return torch.stack([death, kill, ally_kill, unforced], dim=-1)


class DamageReward(RewardComponent):
    """Damage taken and dealt, shield recharge, and its charge-back.

    Every applied unit of damage charges the ship that took it 1, from any
    source; the boundary is the reason it covers more than combat. Damage dealt
    to an enemy pays the shooter ``ratio``, shared among simultaneous hits by
    applied damage; dealt to an ally it blames the shooter ``ratio`` and pays
    the opposing team twice that; boundary damage pays the opposing team.
    Recharge pays the recovering ship 1 per unit and reverses, pro rata over
    the outstanding damage, exactly the payment the recovered damage earned --
    the charge-back. Recharge of damage nobody caused charges the opposing team.
    """

    name = "damage"
    event_names = ("taken", "dealt", "friendly_fire", "recharge", "charge_back", "unforced")

    def events(self, prev_state, actions, next_state, dones):
        del prev_state, actions, dones
        ratio = self.payout_ratio
        combat = next_state.ship_combat_damage  # (B, N_t) applied
        boundary = next_state.ship_boundary_damage
        taken = -(combat + boundary)

        raw = next_state.damage_matrix  # (B, N_a, N_t) raw impact
        applied = raw * (combat / raw.sum(1).clamp(min=EPS)).unsqueeze(1)
        dealt_direct, dealt_opposing = _route_by_cause(ratio * applied, next_state)
        enemy = _enemy_pairs(next_state)
        dealt = (ratio * applied * enemy).sum(2)
        friendly_fire = dealt_direct - dealt  # the blame half of the direct payment

        recharge = next_state.ship_shield_recharge  # (B, N_t)
        back_direct, back_opposing = _route_by_cause(
            -ratio * next_state.recharge_matrix, next_state
        )
        opposing = (
            dealt_opposing
            + back_opposing
            + ratio * boundary
            - ratio * next_state.ship_unattributed_recharge
        )
        unforced = enemy_team_share(opposing, next_state)
        return torch.stack([taken, dealt, friendly_fire, recharge, back_direct, unforced], dim=-1)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def _scaled_ratio(raw: Mapping[str, Any], name: str, offensive_bias: float) -> float:
    """A payout ratio after the offensive bias: ``1 + bias * (ratio - 1)``."""
    return 1.0 + offensive_bias * (float(raw.get(name, 1.0)) - 1.0)


def component_weights(rewards: "RewardConfig | Mapping[str, Any]") -> dict[str, float]:
    """Each level's weight, by component name.

    Args:
        rewards: A ``RewardConfig``, or the plain mapping a checkpoint stores in
            ``train_config["rewards"]``.
    """
    raw = rewards if isinstance(rewards, Mapping) else dataclasses.asdict(rewards)
    return {
        "outcome": float(raw["win_weight"]),
        "zone_capture": float(raw["zone_capture_weight"]),
        "zone_progress": float(raw["zone_progress_weight"]),
        "kill_death": float(raw["death_weight"]),
        "damage": float(raw["damage_weight"]),
    }


def component_payout_ratios(
    rewards: "RewardConfig | Mapping[str, Any]", offensive_bias: float = 1.0
) -> dict[str, float]:
    """Each level's payout ratio after the offensive bias anneals it toward 1."""
    raw = rewards if isinstance(rewards, Mapping) else dataclasses.asdict(rewards)
    capture = _scaled_ratio(raw, "capture_payout_ratio", offensive_bias)
    return {
        "outcome": 1.0,
        "zone_capture": capture,
        "zone_progress": capture,
        "kill_death": _scaled_ratio(raw, "kill_payout_ratio", offensive_bias),
        "damage": _scaled_ratio(raw, "damage_payout_ratio", offensive_bias),
    }


def build_reward_components(
    rewards: RewardConfig,
    ship_config: ShipConfig,
) -> list[RewardComponent]:
    """One component per level, in ``REWARD_COMPONENT_NAMES`` order.

    Built at full offensive bias; ppo.py applies the schedule each update.
    """
    w = component_weights(rewards)
    r = component_payout_ratios(rewards)
    return [
        OutcomeReward(w["outcome"]),
        ZoneCaptureReward(w["zone_capture"], r["zone_capture"]),
        ZoneProgressReward(w["zone_progress"], r["zone_progress"], ship_config.world_size),
        KillDeathReward(w["kill_death"], r["kill_death"]),
        DamageReward(w["damage"], r["damage"]),
    ]


_NAME_TO_K: dict[str, int] = {name: k for k, name in enumerate(REWARD_COMPONENT_NAMES)}


def compute_per_component_rewards(
    components: list[RewardComponent],
    prev_state: TensorState,
    actions: torch.Tensor,
    next_state: TensorState,
    dones: torch.Tensor,
) -> torch.Tensor:
    """(B, N, K) weighted per-ship reward of every level, in registry order."""
    B, N = next_state.ship_health.shape
    result = torch.zeros(
        B, N, len(REWARD_COMPONENT_NAMES), device=next_state.device, dtype=torch.float32
    )
    for comp in components:
        result[:, :, _NAME_TO_K[comp.name]] = comp.compute(prev_state, actions, next_state, dones)
    return result
