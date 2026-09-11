"""Vectorized five-zone frontline mechanics and toroidal map geometry."""

import math
from dataclasses import replace

import torch

from boost_and_broadside.config import FrontlineConfig, MatchResult, ShipConfig, ZoneRole
from boost_and_broadside.config.core import NUM_FRONTLINE_ZONES
from boost_and_broadside.env.field_physics import evaluate_fields
from boost_and_broadside.env.state import TensorState

FRONTLINE_WORLD_SIZE = (16384.0, 16384.0)
FRONTLINE_FIELD_RADIUS_MAX = 750.0
# Re-exported from config, which owns it so the launch arithmetic can size a
# batch without importing the environment. Importers here keep working.
__all__ = ["NUM_FRONTLINE_ZONES"]


def frontline_ship_config(config: ShipConfig) -> ShipConfig:
    """Apply the common provisional world, field-size, and integrator contract."""

    return replace(
        config,
        world_size=FRONTLINE_WORLD_SIZE,
        field_radius_max=FRONTLINE_FIELD_RADIUS_MAX,
        dt=1.0 / 30.0,
        # At <=6 px of travel per 30 Hz tick versus a 40 px interface, one
        # explicit optical step resolves the transition while avoiding four
        # field evaluations per ship tick in the latency-sensitive play loop.
        field_integrator="two_step",
        field_integration_substeps=1,
    )


def toroidal_displacement(
    displacement: torch.Tensor,
    world_size: tuple[float, float],
) -> torch.Tensor:
    """Return minimum-image complex displacements on a rectangular torus."""

    world_w, world_h = world_size
    return torch.complex(
        (displacement.real + world_w / 2.0) % world_w - world_w / 2.0,
        (displacement.imag + world_h / 2.0) % world_h - world_h / 2.0,
    )


def wrap_positions(
    position: torch.Tensor,
    world_size: tuple[float, float],
) -> torch.Tensor:
    """Wrap complex positions into the configured physical toroid."""

    world_w, world_h = world_size
    return torch.complex(position.real % world_w, position.imag % world_h)


# The role pattern is a constant, and building it from a Python list on every
# call copies it from the host, which drains the CUDA queue. `roles_from_front`
# runs on every physics tick and on every reset, so the tensor is cached per
# device instead (the same idiom `physics._lookup_tables` uses).
_BASE_ROLE_CACHE: dict[str, torch.Tensor] = {}


def _base_roles(device: torch.device) -> torch.Tensor:
    """The physical zone-role pattern at front position zero, cached per device."""

    key = str(device)
    roles = _BASE_ROLE_CACHE.get(key)
    if roles is None:
        roles = torch.tensor(
            [
                ZoneRole.NEUTRAL,
                ZoneRole.TEAM0_SPAWN,
                ZoneRole.TEAM0_DEFENSE,
                ZoneRole.TEAM1_DEFENSE,
                ZoneRole.TEAM1_SPAWN,
            ],
            dtype=torch.int8,
            device=device,
        )
        _BASE_ROLE_CACHE[key] = roles
    return roles


def roles_from_front(front_position: torch.Tensor) -> torch.Tensor:
    """Derive physical-zone roles from an arbitrary unwrapped front coordinate.

    Team 0 advances rotate the role pattern one physical location clockwise;
    Team 1 advances rotate it anticlockwise. The unwrapped input is never
    reduced or replaced by the visible modulo-five assignment.
    """

    base_roles = _base_roles(front_position.device)
    physical_index = torch.arange(NUM_FRONTLINE_ZONES, device=front_position.device)
    source_index = (physical_index.unsqueeze(0) - front_position.unsqueeze(1)) % 5
    return base_roles[source_index]


def zone_membership(
    ship_pos: torch.Tensor,
    zone_pos: torch.Tensor,
    zone_radius: torch.Tensor,
    world_size: tuple[float, float],
) -> torch.Tensor:
    """Return ``(B, N, Z)`` membership using shortest toroidal distance."""

    displacement = toroidal_displacement(
        ship_pos.unsqueeze(2) - zone_pos.unsqueeze(1),
        world_size,
    )
    return displacement.abs() <= zone_radius.unsqueeze(1)


def initialize_frontline_map(
    state: TensorState,
    reset_mask: torch.Tensor,
    config: FrontlineConfig,
    world_size: tuple[float, float],
) -> None:
    """Reset translated map-local frontline state for selected environments."""

    batch_size = state.num_envs
    world_w, world_h = world_size
    map_center = torch.complex(
        torch.rand((batch_size,), device=state.device) * world_w,
        torch.rand((batch_size,), device=state.device) * world_h,
    )
    state.map_center = torch.where(reset_mask, map_center, state.map_center)

    angles = (
        torch.arange(NUM_FRONTLINE_ZONES, device=state.device, dtype=torch.float32)
        * (2.0 * math.pi / NUM_FRONTLINE_ZONES)
        - math.pi / 2.0
    )
    offsets = torch.polar(torch.full_like(angles, config.zone_ring_radius), angles)
    translated = wrap_positions(state.map_center.unsqueeze(1) + offsets, world_size)
    reset_z = reset_mask.unsqueeze(1)
    state.zone_pos = torch.where(reset_z, translated, state.zone_pos)
    state.zone_radius = torch.where(reset_z, config.zone_radius, state.zone_radius)
    state.playable_boundary_radius = torch.where(
        reset_mask,
        config.playable_radius,
        state.playable_boundary_radius,
    )
    state.front_position = torch.where(reset_mask, 0, state.front_position)
    state.front_delta = torch.where(reset_mask, 0, state.front_delta)
    state.match_result = torch.where(
        reset_mask,
        int(MatchResult.ONGOING),
        state.match_result,
    )
    state.zone_roles = torch.where(
        reset_z,
        roles_from_front(state.front_position),
        state.zone_roles,
    )
    state.zone_capture_progress = torch.where(reset_z, 0.0, state.zone_capture_progress)
    state.zone_capture_direction = torch.where(reset_z, 0, state.zone_capture_direction)
    state.team0_captured &= ~reset_mask
    state.team1_captured &= ~reset_mask
    state.simultaneous_capture &= ~reset_mask


def _zone_for_role(state: TensorState, role: ZoneRole) -> torch.Tensor:
    """Gather the physical zone position currently carrying ``role``."""

    zone_index = (state.zone_roles == int(role)).long().argmax(dim=1)
    return state.zone_pos.gather(1, zone_index.unsqueeze(1)).squeeze(1)


def place_ships_at_spawns(
    state: TensorState,
    ship_mask: torch.Tensor,
    ship_config: ShipConfig,
    health: torch.Tensor,
) -> None:
    """Place selected slots at their team's current spawn with fresh flight state."""

    batch_size, num_ships = state.ship_pos.shape
    team0_spawn = _zone_for_role(state, ZoneRole.TEAM0_SPAWN)
    team1_spawn = _zone_for_role(state, ZoneRole.TEAM1_SPAWN)
    spawn_center = torch.where(
        state.ship_team_id == 0,
        team0_spawn.unsqueeze(1),
        team1_spawn.unsqueeze(1),
    )

    slot = torch.arange(num_ships, device=state.device, dtype=torch.float32).unsqueeze(0)
    phase = (slot * 0.61803398875 + state.step_count.unsqueeze(1) * 0.38196601125) % 1.0
    angle = phase * (2.0 * math.pi)
    spawn_radius = state.zone_radius[:, :1] * 0.35
    spawn_pos = wrap_positions(
        spawn_center + torch.polar(spawn_radius.expand(batch_size, num_ships), angle),
        ship_config.world_size,
    )

    team0_target = _zone_for_role(state, ZoneRole.TEAM1_DEFENSE)
    team1_target = _zone_for_role(state, ZoneRole.TEAM0_DEFENSE)
    target = torch.where(
        state.ship_team_id == 0,
        team0_target.unsqueeze(1),
        team1_target.unsqueeze(1),
    )
    bearing = toroidal_displacement(target - spawn_pos, ship_config.world_size)
    bearing = bearing / bearing.abs().clamp(min=1e-6)

    field_eval = evaluate_fields(
        spawn_pos,
        state.field_pos,
        state.field_radius,
        state.field_transition_width,
        state.field_index,
        ship_config.world_size,
    )
    velocity = ship_config.default_speed * bearing / field_eval.index
    ship_field_mask = ship_mask.unsqueeze(-1)
    state.ship_pos = torch.where(ship_mask, spawn_pos, state.ship_pos)
    state.ship_attitude = torch.where(ship_mask, bearing, state.ship_attitude)
    state.ship_vel = torch.where(ship_mask, velocity, state.ship_vel)
    state.ship_ang_vel = torch.where(ship_mask, 0.0, state.ship_ang_vel)
    state.ship_health = torch.where(ship_mask, health, state.ship_health)
    state.ship_cooldown = torch.where(ship_mask, 0.0, state.ship_cooldown)
    state.ship_is_shooting &= ~ship_mask
    state.ship_alive |= ship_mask
    state.ship_field_alpha = torch.where(
        ship_field_mask,
        field_eval.alpha,
        state.ship_field_alpha,
    )
    state.ship_local_index = torch.where(ship_mask, field_eval.index, state.ship_local_index)
    state.ship_field_gradient = torch.where(
        ship_mask,
        field_eval.grad_index,
        state.ship_field_gradient,
    )


def clear_previous_life_attribution(state: TensorState) -> None:
    """Clear combat history involving slots that respawned on the prior tick."""

    respawned = state.ship_respawned
    keep = ~(respawned.unsqueeze(2) | respawned.unsqueeze(1))
    state.cumulative_damage_matrix *= keep
    state.ship_respawned.zero_()


def _apply_damage_source(
    state: TensorState,
    requested_damage: torch.Tensor,
    damage_output: torch.Tensor,
    death_output: torch.Tensor,
) -> None:
    """Apply one ordered environmental source with exclusive death attribution."""

    health_before = state.ship_health
    alive_before = state.ship_alive
    health_after = (health_before - requested_damage * alive_before.float()).clamp(min=0.0)
    damage_output.copy_(health_before - health_after)
    death_output.copy_(alive_before & (health_after <= 0.0))
    state.ship_health = health_after
    state.ship_alive = alive_before & ~death_output


def _apply_frontline_hazards(
    state: TensorState,
    membership: torch.Tensor,
    config: FrontlineConfig,
    ship_config: ShipConfig,
) -> None:
    """Apply spawn healing and ordered defense/spawn/boundary damage sources."""

    team = state.ship_team_id
    roles = state.zone_roles
    team0 = team == 0
    team1 = team == 1
    t0_spawn = (roles == int(ZoneRole.TEAM0_SPAWN)).unsqueeze(1)
    t1_spawn = (roles == int(ZoneRole.TEAM1_SPAWN)).unsqueeze(1)
    friendly_spawn = (
        membership & ((team0.unsqueeze(2) & t0_spawn) | (team1.unsqueeze(2) & t1_spawn))
    ).any(dim=2)
    hostile_spawn = (
        membership & ((team0.unsqueeze(2) & t1_spawn) | (team1.unsqueeze(2) & t0_spawn))
    ).any(dim=2)

    heal_request = config.spawn_heal_per_second * ship_config.dt
    missing_health = (ship_config.max_health - state.ship_health).clamp(min=0.0)
    healing = torch.minimum(missing_health, torch.full_like(missing_health, heal_request))
    healing = healing * (friendly_spawn & state.ship_alive).float()
    state.ship_health = state.ship_health + healing
    state.ship_spawn_healing.copy_(healing)

    defense = (
        (roles == int(ZoneRole.TEAM0_DEFENSE)) | (roles == int(ZoneRole.TEAM1_DEFENSE))
    ).unsqueeze(1)
    in_defense = (membership & defense).any(dim=2)
    _apply_damage_source(
        state,
        in_defense.float() * config.defense_damage_per_second * ship_config.dt,
        state.ship_zone_damage,
        state.ship_zone_death,
    )
    _apply_damage_source(
        state,
        hostile_spawn.float() * config.enemy_spawn_damage_per_second * ship_config.dt,
        state.ship_spawn_damage,
        state.ship_spawn_death,
    )

    from_center = toroidal_displacement(
        state.ship_pos - state.map_center.unsqueeze(1),
        ship_config.world_size,
    ).abs()
    outside = (from_center - state.playable_boundary_radius.unsqueeze(1)).clamp(min=0.0)
    boundary_rate = torch.where(
        outside > 0.0,
        config.boundary_damage_per_second + config.boundary_damage_per_pixel_second * outside,
        0.0,
    )
    _apply_damage_source(
        state,
        boundary_rate * ship_config.dt,
        state.ship_boundary_damage,
        state.ship_boundary_death,
    )


def _advance_capture_state(
    state: TensorState,
    membership: torch.Tensor,
    config: FrontlineConfig,
    ship_config: ShipConfig,
) -> None:
    """Advance both defense meters and apply simultaneous completion atomically."""

    alive_in_zone = membership & state.ship_alive.unsqueeze(2)
    team0_count = (alive_in_zone & (state.ship_team_id == 0).unsqueeze(2)).sum(dim=1)
    team1_count = (alive_in_zone & (state.ship_team_id == 1).unsqueeze(2)).sum(dim=1)
    # Deliberately discard the size of the advantage: every non-tied majority
    # applies one fixed capture/stabilization rate.
    majority = torch.sign(team0_count - team1_count).to(torch.int8)

    roles = state.zone_roles
    t0_defense = roles == int(ZoneRole.TEAM0_DEFENSE)
    t1_defense = roles == int(ZoneRole.TEAM1_DEFENSE)
    active_defense = t0_defense | t1_defense
    direction = torch.where(active_defense, majority, 0)
    state.zone_capture_direction.copy_(direction)

    attacker_direction = torch.where(t1_defense, 1, torch.where(t0_defense, -1, 0))
    signed_motion = direction * attacker_direction
    delta = signed_motion.float() * (ship_config.dt / config.capture_seconds)
    progress = (state.zone_capture_progress + delta).clamp(0.0, 1.0)
    progress = torch.where(active_defense, progress, 0.0)

    state.team0_captured.copy_((progress >= 1.0).logical_and(t1_defense).any(dim=1))
    state.team1_captured.copy_((progress >= 1.0).logical_and(t0_defense).any(dim=1))
    state.simultaneous_capture.copy_(state.team0_captured & state.team1_captured)
    state.front_delta.copy_(
        state.team0_captured.to(torch.int8) - state.team1_captured.to(torch.int8)
    )
    state.front_position = state.front_position + state.front_delta.long()

    any_capture = state.team0_captured | state.team1_captured
    state.zone_capture_progress = torch.where(any_capture.unsqueeze(1), 0.0, progress)
    state.zone_capture_direction = torch.where(
        any_capture.unsqueeze(1),
        0,
        state.zone_capture_direction,
    )
    state.zone_roles = roles_from_front(state.front_position)


def apply_frontline_tick(
    state: TensorState,
    config: FrontlineConfig,
    ship_config: ShipConfig,
) -> torch.Tensor:
    """Apply objectives, hazards, victory, and immediate same-slot respawn."""

    state.front_delta.zero_()
    state.team0_captured.zero_()
    state.team1_captured.zero_()
    state.simultaneous_capture.zero_()
    state.ship_zone_damage.zero_()
    state.ship_spawn_damage.zero_()
    state.ship_boundary_damage.zero_()
    state.ship_zone_death.zero_()
    state.ship_spawn_death.zero_()
    state.ship_boundary_death.zero_()
    state.ship_spawn_healing.zero_()

    membership = zone_membership(
        state.ship_pos,
        state.zone_pos,
        state.zone_radius,
        ship_config.world_size,
    )
    # Capture membership is sampled before the same zone's health commitment is
    # charged. This is provisional and intentionally explicit for Gate-1 review.
    _advance_capture_state(state, membership, config, ship_config)
    _apply_frontline_hazards(state, membership, config, ship_config)

    threshold = config.front_win_threshold
    team0_win = state.front_position >= threshold
    team1_win = state.front_position <= -threshold
    state.match_result = torch.where(
        team0_win,
        int(MatchResult.TEAM0_WIN),
        torch.where(team1_win, int(MatchResult.TEAM1_WIN), state.match_result),
    )

    respawned = (
        state.ship_field_death
        | state.ship_combat_death
        | state.ship_zone_death
        | state.ship_spawn_death
        | state.ship_boundary_death
    )
    state.ship_respawned.copy_(respawned)
    respawn_health = torch.full_like(state.ship_health, config.respawn_health)
    place_ships_at_spawns(state, respawned, ship_config, respawn_health)
    return state.match_result != int(MatchResult.ONGOING)


def apply_timeout_result(state: TensorState, timed_out: torch.Tensor) -> None:
    """Set the configured timeout winner from the sign of unwrapped front state."""

    unresolved = timed_out & (state.match_result == int(MatchResult.ONGOING))
    result = torch.where(
        state.front_position > 0,
        int(MatchResult.TEAM0_WIN),
        torch.where(
            state.front_position < 0,
            int(MatchResult.TEAM1_WIN),
            int(MatchResult.DRAW),
        ),
    )
    state.match_result = torch.where(unresolved, result, state.match_result)
