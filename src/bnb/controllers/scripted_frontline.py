"""The scripted agent's Frontline layer (scripted spec A5): which zone each ship goes to,
separation from allies, recovery toward its spawn when hurt and threatened, and the blend
with the combat layer by the nearest enemy's distance.

Strength is a sum of Gaussian kernels over visible, living ships, each weighted by its
health 0.2 + 0.8 shield / max_shield (a ship at full shield counts 1).
"""

import torch

from bnb.contracts.obs import ZONE
from bnb.game.handling import Characteristics
from bnb.game.ship_types import per_ship

from .scripted_combat import (
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
from .scripted_parameters import Ramp, ScriptedParameters, ramp

OWN_SPAWN, OWN_DEFENSE, ENEMY_DEFENSE = 0, 1, 3  # zone roles for the viewer (ZONE_COLUMNS)


def _health(rows: torch.Tensor, types: torch.Tensor, characteristics: Characteristics):
    return 0.2 + 0.8 * column(rows, "shield") / per_ship(characteristics.max_shield, types)


def _kernel(offset_x: torch.Tensor, offset_y: torch.Tensor, width: torch.Tensor):
    return torch.exp(-(offset_x * offset_x + offset_y * offset_y) / (2.0 * width * width))


def _pairs(x: torch.Tensor, y: torch.Tensor, to_x: torch.Tensor, to_y: torch.Tensor):
    """Offsets (R, S, K) from each own ship (R, S) to each point (R, K)."""
    return to_x[:, None, :] - x[..., None], to_y[:, None, :] - y[..., None]


def frontline(fleet: Context, zones: torch.Tensor, characteristics: Characteristics,
              parameters: ScriptedParameters, combat_marginals: Marginals,
              has_target: torch.Tensor, target_distance: torch.Tensor) -> Marginals:
    own, enemy = fleet.own, fleet.enemy
    own_alive = column(own, "alive") > 0.5
    own_health = torch.where(own_alive, _health(own, fleet.own_type, characteristics), 0.0)
    enemy_health = torch.where(fleet.enemy_seen,
                               _health(enemy, fleet.enemy_type, characteristics), 0.0)
    combat_radius = parameters.combat_radius_in_gun_range * fleet.gun_range  # (R, S)

    # The threat at each own ship: the enemies' strength there.
    enemy_x, enemy_y = _pairs(fleet.x, fleet.y, column(enemy, "x"), column(enemy, "y"))
    threat = (enemy_health[:, None, :]
              * _kernel(enemy_x, enemy_y, combat_radius[..., None])).sum(-1)

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
    enemy_near = (enemy_health[..., None]
                  * _kernel(*_pairs(column(enemy, "x"), column(enemy, "y"), zone_x, zone_y),
                            support)).sum(1)  # (R, Z): the enemy's strength near each zone
    under_attack = of_role((zones[..., ZONE["direction"]] < 0.0).float(), OWN_DEFENSE) > 0.5

    # Recovery: hurt and threatened, fall back to the own spawn.
    shield_fraction = column(own, "shield") / per_ship(characteristics.max_shield, fleet.own_type)
    recovering = ((shield_fraction < parameters.recovery_shield_fraction)
                  & (threat > parameters.recovery_threat))

    # Who defends: the defense wants the floor, or the enemy's strength near it plus a
    # margin, and more while its meter moves against the viewer. Ships nearer the defense
    # zone than the attack zone go first (ties: lowest row), until their strength covers
    # the want; the rest attack, unless their strength times exp(aggression) is below the
    # enemy's near the attack zone, when everyone defends. Every ship of a view reaches
    # the same assignment, so the team splits without memory.
    assignable = own_alive & ~recovering
    available = torch.where(assignable, own_health, 0.0)

    def per_view(value) -> torch.Tensor:  # a float or a per-view (R, 1) parameter
        return torch.as_tensor(value, dtype=available.dtype, device=available.device)

    wanted = (torch.maximum(of_role(enemy_near, OWN_DEFENSE)[:, None]
                            + parameters.defensive_margin, per_view(parameters.defensive_floor))
              + parameters.defense_urgency * under_attack[:, None].float())  # (R, 1)
    attackers = available.sum(-1, keepdim=True) - wanted
    hopeless = (attackers * torch.exp(per_view(parameters.aggression))
                < of_role(enemy_near, ENEMY_DEFENSE)[:, None])
    order = (torch.sqrt(to_defense[0] ** 2 + to_defense[1] ** 2)
             - torch.sqrt(to_attack[0] ** 2 + to_attack[1] ** 2))
    rows = torch.arange(order.shape[-1], device=order.device)
    ahead = ((order[:, None, :] < order[..., None])
             | ((order[:, None, :] == order[..., None]) & (rows[None, :] < rows[:, None])))
    strength_ahead = torch.where(ahead, available[:, None, :], 0.0).sum(-1)  # (R, S)
    defending = hopeless | (strength_ahead < wanted)
    goal_x = torch.where(defending, to_defense[0], to_attack[0])
    goal_y = torch.where(defending, to_defense[1], to_attack[1])

    # Separation from allies within its radius, stronger the nearer.
    ally_x, ally_y = _pairs(fleet.x, fleet.y, fleet.x, fleet.y)  # (R, S, S): ship -> ally
    ally_distance = torch.sqrt(ally_x * ally_x + ally_y * ally_y)
    separation_radius = (parameters.separation_radius_in_gun_range * fleet.gun_range)[..., None]
    others = ~torch.eye(own.shape[1], dtype=torch.bool, device=own.device)
    near = others & own_alive[:, None, :] & (ally_distance < separation_radius)
    push = torch.where(near, (1.0 - ally_distance / separation_radius)
                       / ally_distance.clamp_min(1e-3), 0.0)
    goal_length = torch.sqrt(goal_x * goal_x + goal_y * goal_y).clamp_min(1e-6)
    wanted_x = goal_x / goal_length - parameters.separation_weight * (push * ally_x).sum(-1)
    wanted_y = goal_y / goal_length - parameters.separation_weight * (push * ally_y).sum(-1)
    wanted_x = torch.where(recovering, to_spawn[0], wanted_x)
    wanted_y = torch.where(recovering, to_spawn[1], wanted_y)
    goal_distance = torch.where(recovering, torch.sqrt(to_spawn[0] ** 2 + to_spawn[1] ** 2),
                                goal_length)

    stick = stick_marginal(fleet, signed_angle(fleet.cos, fleet.sin,
                                               *steer(fleet, wanted_x, wanted_y)), parameters)
    power = power_marginal(either(slow(fleet, parameters), far(fleet, goal_distance, parameters)),
                           torch.zeros_like(goal_distance))
    # Combat takes over from the inner shooting range in to the combat radius.
    handover = Ramp(parameters.shoot_ramp_in_gun_range.low, parameters.combat_radius_in_gun_range,
                    1.0, 0.0)
    weight = torch.where(has_target, ramp(target_distance / fleet.gun_range, handover), 0.0)
    blended_power = blend(combat_marginals.power, power, weight)
    blended_stick = blend(combat_marginals.stick, stick, weight)
    keep = recovering[..., None]
    return Marginals(torch.where(keep, power, blended_power),
                     torch.where(keep, stick, blended_stick), combat_marginals.shoot)
