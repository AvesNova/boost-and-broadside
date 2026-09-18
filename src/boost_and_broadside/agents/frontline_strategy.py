"""Memoryless Frontline navigation from visible health, geometry, and zone roles."""

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.config import ShipConfig, ZoneRole
from boost_and_broadside.env.frontline import toroidal_displacement
from boost_and_broadside.env.state import TensorState


@dataclass
class FrontlineStrategy:
    """Navigation plus interpretable diagnostics, all on the state's device."""

    distance: torch.Tensor
    bearing: torch.Tensor
    zone_need: torch.Tensor
    zone_preference: torch.Tensor
    combat_score: torch.Tensor
    separation: torch.Tensor
    recovery: torch.Tensor


def frontline_strategy(
    state: TensorState,
    ship: ShipConfig,
    config: StochasticAgentConfig,
    visibility: torch.Tensor,
    frontline_parameters: dict[str, torch.Tensor] | None = None,
) -> FrontlineStrategy:
    """Combine marginal objective demand, local strength, separation and recovery.

    Ship axes are (batch, observer, contributor); zone axes are (batch, ship, zone).
    Visibility is the authoritative (batch, team, ship) mask. No hidden memory,
    random identities, ship ranks, or fleet-size-dependent thresholds are used.
    """
    def parameter(name: str) -> float | torch.Tensor:
        return getattr(config, name) if frontline_parameters is None else frontline_parameters[name]

    # A tournament search may give the two sides of an environment different
    # Frontline settings. These are (B, N) tensors in that path, and the final
    # singleton axis broadcasts them across enemy or zone comparisons.
    combat_radius = parameter("frontline_combat_radius")
    combat_radius_for_pairs = (
        combat_radius if not torch.is_tensor(combat_radius) else combat_radius.unsqueeze(-1)
    )
    aggression = parameter("frontline_aggression")
    zone_radius = parameter("frontline_zone_radius")
    zone_margin = parameter("frontline_zone_margin")
    separation_radius = parameter("frontline_separation_radius")
    recovery_health = parameter("frontline_recovery_health")

    health = (state.ship_health / ship.max_health).clamp(0, 1)
    shield_fraction = health
    # Depleted ships still shoot and capture; zero shields is not zero strength.
    health = torch.where(state.ship_alive, 0.2 + 0.8 * health, 0.0)
    allied = state.ship_team_id[:, :, None] == state.ship_team_id[:, None, :]
    visible = visibility.gather(1, state.ship_team_id.long()[:, :, None].expand_as(allied))
    enemies = ~allied & visible & state.ship_alive[:, None, :]
    allies = allied & state.ship_alive[:, None, :]
    delta = toroidal_displacement(
        state.ship_pos[:, None, :] - state.ship_pos[:, :, None], ship.world_size
    )
    distance = delta.abs()
    unit = delta / distance.clamp_min(1e-8)
    kernel = torch.exp(-(distance / combat_radius_for_pairs).square())
    allied_strength = torch.where(allies, kernel * health[:, None, :], 0).sum(-1)
    enemy_weight = torch.where(enemies, kernel * health[:, None, :], 0)
    enemy_strength = enemy_weight.sum(-1)
    combat = torch.tanh(
        torch.log((allied_strength + 1e-6) / (enemy_strength + 1e-6)) + aggression
    )
    enemy_direction = (enemy_weight * unit).sum(-1) / enemy_strength.clamp_min(1e-8)
    # Vanishes in empty space; bounded even when a large enemy fleet is present.
    combat_force = combat * enemy_direction * (1 - torch.exp(-enemy_strength))

    zone_delta = toroidal_displacement(
        state.zone_pos[:, None, :] - state.ship_pos[:, :, None], ship.world_size
    )
    zone_distance = zone_delta.abs()
    support_radius = (
        2 * state.zone_radius[:, None, :]
        if zone_radius is None
        else zone_radius if not torch.is_tensor(zone_radius) else zone_radius.unsqueeze(-1)
    )
    contribution = health[:, :, None] * torch.exp(-(zone_distance / support_radius).square())
    # Sum once per team, then gather for each observer and subtract self.
    # Team-shared sight makes both pressure totals identical for allied observers.
    team_members = torch.stack([state.ship_team_id == 0, state.ship_team_id == 1], dim=1)
    team_allied_zone = torch.bmm(team_members.float(), contribution)
    team_enemy_zone = torch.bmm((~team_members & visibility).float(), contribution)
    observer_team = state.ship_team_id.long()[:, :, None].expand_as(contribution)
    allied_zone = team_allied_zone.gather(1, observer_team)
    enemy_zone = team_enemy_zone.gather(1, observer_team)
    without_self = (allied_zone - contribution).clamp_min(0)
    roles = state.zone_roles[:, None, :]
    team0 = state.ship_team_id[:, :, None] == 0
    own_defense = torch.where(
        team0, roles == int(ZoneRole.TEAM0_DEFENSE), roles == int(ZoneRole.TEAM1_DEFENSE)
    )
    offense = torch.where(
        team0, roles == int(ZoneRole.TEAM1_DEFENSE), roles == int(ZoneRole.TEAM0_DEFENSE)
    )
    # Keep one ship-equivalent of demand on our defense even when no attacker is
    # visible.  Aggression increases offensive demand without suppressing that
    # defensive floor.  The old symmetric exp(+a)/exp(-a) rule made offense
    # 7.4x as attractive as defense at the shipped aggression of 1.0.
    offensive_margin = (
        math.exp(aggression)
        if not torch.is_tensor(aggression)
        else torch.exp(aggression).unsqueeze(-1)
    )
    margin_scale = zone_margin if not torch.is_tensor(zone_margin) else zone_margin.unsqueeze(-1)
    margin = margin_scale * torch.where(
        offense, offensive_margin, 1.0
    )
    # Capture state is public map information.  Use it to recruit defenders even
    # when an opaque zone hides the attacker that is moving the meter.
    defense_attacked = own_defense & (state.zone_capture_progress[:, None, :] > 0)
    defense_urgency = defense_attacked * (1.0 + 3.0 * state.zone_capture_progress[:, None, :])
    margin = margin + defense_urgency
    # Fixed half-ship softplus width; normalize later so pressure cannot explode.
    need = F.softplus(2 * (margin + enemy_zone - without_self)) / 2
    need = torch.where(own_defense | offense, need, 0)
    utility = need / (1 + (zone_distance / support_radius).square())
    preference = utility / utility.sum(-1, keepdim=True).clamp_min(1e-8)
    zone_unit = zone_delta / zone_distance.clamp_min(1e-8)
    objective_force = (preference * zone_unit).sum(-1)

    separation_radius = (
        4 * ship.collision_radius if separation_radius is None else separation_radius
    )
    separation_radius_for_pairs = (
        separation_radius
        if not torch.is_tensor(separation_radius)
        else separation_radius.unsqueeze(-1)
    )
    # delta/R is smooth even at coincident positions; normalizing by mass bounds
    # dense fleets without diluting repulsion when unrelated distant ships exist.
    sep_weight = torch.where(
        allies & (distance > 0), torch.exp(-(distance / separation_radius_for_pairs).square()), 0
    )
    separation = -(sep_weight * delta / separation_radius_for_pairs).sum(-1)
    separation = separation / sep_weight.sum(-1).clamp_min(1)

    own_spawn = torch.where(
        team0, roles == int(ZoneRole.TEAM0_SPAWN), roles == int(ZoneRole.TEAM1_SPAWN)
    )
    spawn_delta = torch.where(own_spawn, zone_delta, 0).sum(-1)
    spawn_distance = spawn_delta.abs()
    recovery = (1 - shield_fraction).square() / (
        (1 - shield_fraction).square()
        + (shield_fraction / recovery_health).square()
        + 1e-8
    )
    # Recharge anywhere: do not travel all the way home when no enemy threatens us.
    threat = 1 - torch.exp(-enemy_strength)
    recovery = recovery * threat
    escape = -enemy_direction
    home = spawn_delta / spawn_distance.clamp_min(1e-8)
    retreat = escape + 0.25 * home
    retreat = retreat / retreat.abs().clamp_min(1e-8)
    force = (1 - recovery) * (objective_force + combat_force) + recovery * retreat + separation
    bearing = force / force.abs().clamp_min(1e-8)
    # At a balanced/zero force, retain heading without inventing an identity.
    bearing = torch.where(force.abs() > 1e-8, bearing, state.ship_attitude)
    travel = (1 - recovery) * (preference * zone_distance).sum(-1) + recovery * spawn_distance
    return FrontlineStrategy(travel, bearing, need, preference, combat, separation, recovery)
