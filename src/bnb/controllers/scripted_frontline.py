"""The scripted agent's Frontline layer (scripted spec A5): which zone each ship goes to,
separation from allies, recovery toward its spawn when hurt and threatened, and the blend
with the combat layer by the nearest enemy's distance.

Strength is a sum of Gaussian kernels over visible, living ships, each weighted by its
health 0.2 + 0.8 shield / max_shield (a ship at full shield counts 1).

The goal (defend or attack) and recovery are plan choices the agent commits to
(`scripted_memory.commit`): each rule gives a margin, positive for the first option, and the
committed option switches at its rate on the other's margin.
"""

import math
from typing import NamedTuple

import torch

from bnb.contracts.obs import ZONE
from bnb.game.handling.characteristics import Characteristics
from bnb.game.ship_types import per_ship

from .scripted_combat import (
    TINY,
    Combat,
    Context,
    Marginals,
    blend,
    column,
    either,
    far,
    power_marginal,
    signed_angle,
    slow,
    steer,
    stick_marginal,
)
from .scripted_doctrine import per_ship as per_ship_doctrine
from .scripted_memory import commit
from .scripted_parameters import Ramp, ScriptedParameters, in_own_time, ramp

OWN_SPAWN, OWN_DEFENSE, ENEMY_DEFENSE = 0, 1, 3  # zone roles for the viewer (ZONE_COLUMNS)
# The doctrine's lock and wave terms shift the goal's margin by their weight times this
# many ship strengths: more than a team's own (a weight of 1 decides the goal).
LOCK_STRENGTHS = 10.0


def _health(rows: torch.Tensor, types: torch.Tensor, characteristics: Characteristics):
    return 0.2 + 0.8 * column(rows, "shield") / per_ship(characteristics.max_shield, types)


def _kernel(offset_x: torch.Tensor, offset_y: torch.Tensor, width: torch.Tensor):
    return torch.exp(-(offset_x * offset_x + offset_y * offset_y) / (2.0 * width * width))


def _pairs(x: torch.Tensor, y: torch.Tensor, to_x: torch.Tensor, to_y: torch.Tensor):
    """Offsets (R, S, K) from each own ship (R, S) to each point (R, K)."""
    return to_x[:, None, :] - x[..., None], to_y[:, None, :] - y[..., None]


class Frontline(NamedTuple):
    """The Frontline layer's marginals and the plan choices it made per own ship (R, S): the
    turn side, the goal (defending) and recovery."""

    marginals: Marginals
    left: torch.Tensor
    defending: torch.Tensor
    recovering: torch.Tensor


def _choice(fleet: Context, committed: torch.Tensor, margin: torch.Tensor, shape) -> torch.Tensor:
    """A two-way plan choice from its rule's margin (positive: the first option, True):
    the committed option switches at `shape`'s rate on the other's margin."""
    advantage = torch.where(committed, -margin, margin)
    return commit(
        committed,
        margin > 0.0,
        advantage,
        fleet.carry.fresh,
        in_own_time(shape, fleet.time_unit),
        fleet.temperature,
        fleet.seconds,
        fleet.switching,
    )


def frontline(
    fleet: Context,
    zones: torch.Tensor,
    characteristics: Characteristics,
    parameters: ScriptedParameters,
    fighting: Combat,
    doctrines: torch.Tensor | None = None,
) -> Frontline:
    """The Frontline layer per own ship (R, S): power and stick toward the assigned zone of
    zones (R, Z, columns), or toward the spawn when recovering, blended with the combat
    layer's marginals by its target's distance, times the target's weight, where it has a
    target; the shoot marginal is the combat layer's. doctrines: each ship type's doctrine
    table (`bnb.controllers.scripted_doctrine`), whose lock and wave terms shift the goal
    (`lock_and_wave`); None, or the weights at 0, leave it as it was."""
    combat_marginals, has_target = fighting.marginals, fighting.has_target
    target_distance = fighting.distance
    own, enemy = fleet.own, fleet.enemy
    own_alive = column(own, "alive") > 0.5
    own_health = torch.where(own_alive, _health(own, fleet.own_type, characteristics), 0.0)
    enemy_health = torch.where(
        fleet.enemy_seen, _health(enemy, fleet.enemy_type, characteristics), 0.0
    )
    combat_radius = parameters.combat_radius_in_gun_range * fleet.gun_range  # (R, S)

    # The threat at each own ship: the enemies' strength there.
    enemy_x, enemy_y = _pairs(fleet.x, fleet.y, column(enemy, "x"), column(enemy, "y"))
    threat = (enemy_health[:, None, :] * _kernel(enemy_x, enemy_y, combat_radius[..., None])).sum(
        -1
    )

    # The zones the layer reads, as offsets (R, S) from each own ship.
    zone_x, zone_y = zones[..., ZONE["x"]], zones[..., ZONE["y"]]
    role, present = zones[..., ZONE["role"]], zones[..., ZONE["present"]] > 0.5
    to_zone_x, to_zone_y = _pairs(fleet.x, fleet.y, zone_x, zone_y)  # (R, S, Z)

    def of_role(values: torch.Tensor, wanted: int) -> torch.Tensor:
        """The value of the zone with this role: (R, S, Z) -> (R, S), (R, Z) -> (R,)."""
        mask = present & (role == wanted)
        return torch.where(mask[:, None, :] if values.dim() == 3 else mask, values, 0.0).sum(-1)

    to_defense = of_role(to_zone_x, OWN_DEFENSE), of_role(to_zone_y, OWN_DEFENSE)
    to_attack = of_role(to_zone_x, ENEMY_DEFENSE), of_role(to_zone_y, ENEMY_DEFENSE)
    to_spawn = of_role(to_zone_x, OWN_SPAWN), of_role(to_zone_y, OWN_SPAWN)
    support = (parameters.support_radius_in_zone_radius * zones[..., ZONE["radius"]])[:, None, :]
    enemy_near = (
        enemy_health[..., None]
        * _kernel(*_pairs(column(enemy, "x"), column(enemy, "y"), zone_x, zone_y), support)
    ).sum(1)  # (R, Z): the enemy's strength near each zone
    # The defense urgency: while the own defense zone's meter moves against the viewer, the
    # more so the further the enemy's capture has gone (progress in [0, 1]).
    against = (zones[..., ZONE["direction"]] < 0.0).float()
    urgency = of_role(against * (1.0 + zones[..., ZONE["progress"]]), OWN_DEFENSE)  # (R,)

    # Recovery: hurt and threatened, fall back to the own spawn.
    # Recovery's margin: the smaller log margin of its two conditions.
    shield_fraction = column(own, "shield") / per_ship(characteristics.max_shield, fleet.own_type)
    recovery_margin = torch.minimum(
        _log(parameters.recovery_shield_fraction) - torch.log(shield_fraction.clamp_min(TINY)),
        torch.log(threat.clamp_min(TINY)) - _log(parameters.recovery_threat),
    )
    recovering = _choice(
        fleet, fleet.carry.recovering, recovery_margin, parameters.recovery_switch_in_log_ratio
    )

    # Who defends: the defense wants the floor, or the enemy's strength near it plus a
    # margin, and more by the urgency. Ships nearer the defense
    # zone than the attack zone go first (ties: lowest row), until their strength covers
    # the want; the rest attack, unless their strength times exp(aggression) is below the
    # enemy's near the attack zone, when everyone defends. Every ship of a view reaches
    # the same assignment, so the team splits without memory.
    assignable = own_alive & ~recovering
    available = torch.where(assignable, own_health, 0.0)

    def per_view(value) -> torch.Tensor:  # a float or a per-view (R, 1) parameter
        # a float is filled on the device: a host copy cannot run inside a graph capture
        if isinstance(value, torch.Tensor):
            return value.to(available.dtype)
        return available.new_full((), value)

    wanted = (
        torch.maximum(
            of_role(enemy_near, OWN_DEFENSE)[:, None] + parameters.defensive_margin,
            per_view(parameters.defensive_floor),
        )
        + parameters.defense_urgency * urgency[:, None]
    )  # (R, 1)
    attackers = available.sum(-1, keepdim=True) - wanted
    # hopeless where positive: the enemy near the attack zone outweighs the attackers
    hopeless = (
        of_role(enemy_near, ENEMY_DEFENSE)[:, None] * torch.exp(-per_view(parameters.aggression))
        - attackers
    )
    order = torch.sqrt(to_defense[0] ** 2 + to_defense[1] ** 2) - torch.sqrt(
        to_attack[0] ** 2 + to_attack[1] ** 2
    )
    rows = torch.arange(order.shape[-1], device=order.device)
    ahead = (order[:, None, :] < order[..., None]) | (
        (order[:, None, :] == order[..., None]) & (rows[None, :] < rows[:, None])
    )
    strength_ahead = torch.where(ahead, available[:, None, :], 0.0).sum(-1)  # (R, S)
    # Defend's margin, in ship strengths: the larger of the two rules', shifted by the
    # doctrine's lock and wave terms.
    margin = torch.maximum(hopeless, wanted - strength_ahead)
    if doctrines is not None:
        margin = margin + lock_and_wave(fleet, zones, own_alive, doctrines)
    defending = _choice(fleet, fleet.carry.defending, margin, parameters.goal_switch_in_strength)
    goal_x = torch.where(defending, to_defense[0], to_attack[0])
    goal_y = torch.where(defending, to_defense[1], to_attack[1])

    # Separation from allies within its radius, stronger the nearer.
    ally_x, ally_y = _pairs(fleet.x, fleet.y, fleet.x, fleet.y)  # (R, S, S): ship -> ally
    ally_distance = torch.sqrt(ally_x * ally_x + ally_y * ally_y)
    separation_radius = (parameters.separation_radius_in_gun_range * fleet.gun_range)[..., None]
    others = ~torch.eye(own.shape[1], dtype=torch.bool, device=own.device)
    near = others & own_alive[:, None, :] & (ally_distance < separation_radius)
    push = torch.where(
        near, (1.0 - ally_distance / separation_radius) / ally_distance.clamp_min(1e-3), 0.0
    )
    goal_length = torch.sqrt(goal_x * goal_x + goal_y * goal_y).clamp_min(1e-6)
    wanted_x = goal_x / goal_length - parameters.separation_weight * (push * ally_x).sum(-1)
    wanted_y = goal_y / goal_length - parameters.separation_weight * (push * ally_y).sum(-1)
    wanted_x = torch.where(recovering, to_spawn[0], wanted_x)
    wanted_y = torch.where(recovering, to_spawn[1], wanted_y)
    goal_distance = torch.where(
        recovering, torch.sqrt(to_spawn[0] ** 2 + to_spawn[1] ** 2), goal_length
    )

    stick, left = stick_marginal(
        fleet,
        signed_angle(fleet.cos, fleet.sin, *steer(fleet, wanted_x, wanted_y)),
        parameters,
        fleet.carry.frontline_left,
    )
    power = power_marginal(
        either(slow(fleet, parameters), far(fleet, goal_distance, parameters)),
        torch.zeros_like(goal_distance),
    )
    # Combat takes over from the shooting range's centre in to the combat radius.
    handover = Ramp(
        parameters.shoot_in_gun_range.centre, parameters.combat_radius_in_gun_range, 1.0, 0.0
    )
    weight = torch.where(
        has_target, fighting.weight * ramp(target_distance / fleet.gun_range, handover), 0.0
    )
    blended_power = blend(combat_marginals.power, power, weight)
    blended_stick = blend(combat_marginals.stick, stick, weight)
    keep = recovering[..., None]
    marginals = Marginals(
        torch.where(keep, power, blended_power),
        torch.where(keep, stick, blended_stick),
        combat_marginals.shoot,
    )
    return Frontline(marginals, left, defending, recovering)


def lock_and_wave(
    fleet: Context, zones: torch.Tensor, own_alive: torch.Tensor, doctrines: torch.Tensor
) -> torch.Tensor:
    """(R, S) the doctrine's shift of the defend margin, in ship strengths (positive:
    defend). The capture lock: the enemy locked out of a zone it lost for longer than the
    ship's travel back there (its distance over its speed, at least the corner speed)
    presses the attack (-lock_press); the own team locked out of a zone for longer than the
    travel to it holds the defense (+lock_press). The wave: an own ship waiting to respawn,
    with the team's next wave within wave_hold_seconds, holds the defense (+wave_hold) until
    the wave lands. Each term is LOCK_STRENGTHS ship strengths times its weight.

    A ship waits to respawn when it is dead and its respawn wait is at 0 (its wave to come)
    or below the last decision's (counting down). A benched slot (fleet sizes, the play
    shell's padded teams: `bnb.game.physics.step`) is a dead ship whose wait holds at the
    full respawn length, so it never counts; nor does a ship on the tick it died."""
    doctrine = per_ship_doctrine(doctrines, fleet.own_type)
    present = zones[..., ZONE["present"]] > 0.5  # (R, Z)
    to_x, to_y = _pairs(fleet.x, fleet.y, zones[..., ZONE["x"]], zones[..., ZONE["y"]])
    pace = torch.maximum(fleet.speed, fleet.corner_speed).clamp_min(1.0)[..., None]
    travel = torch.sqrt(to_x * to_x + to_y * to_y) / pace  # (R, S, Z) s to each zone
    lock_own = zones[..., ZONE["lock_own"]][:, None, :]
    lock_enemy = zones[..., ZONE["lock_enemy"]][:, None, :]
    held_out = (present[:, None, :] & (lock_own > travel)).any(-1)  # we cannot retake one
    pressing = (present[:, None, :] & (lock_enemy > travel)).any(-1)  # they cannot
    wait = column(fleet.own, "respawn_wait")
    counting = (wait <= 0.0) | (wait < fleet.carry.respawn_seen - 1e-6)
    waiting = (~own_alive & counting).any(-1, keepdim=True)  # an own ship waits for the wave
    landing = (fleet.next_wave[:, None] < doctrine.wave_hold_seconds) & waiting
    shift = doctrine.lock_press * (held_out.float() - pressing.float())
    shift = shift + doctrine.wave_hold * landing.float()
    return LOCK_STRENGTHS * shift


def _log(value):
    """The natural logarithm of a float or a per-view (R, 1) parameter."""
    return torch.log(value) if isinstance(value, torch.Tensor) else math.log(value)
