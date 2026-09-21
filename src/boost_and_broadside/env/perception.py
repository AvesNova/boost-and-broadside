"""GPU-vectorized team perception and opaque-core line-of-sight occlusion."""

from dataclasses import dataclass

import torch

from boost_and_broadside.config import EnvConfig, ShipConfig
from boost_and_broadside.constants import EPS
from boost_and_broadside.env.frontline import toroidal_displacement
from boost_and_broadside.env.state import TensorState


@dataclass(frozen=True)
class TeamVisibility:
    """Visibility diagnostics and masks for both team perspectives.

    Shapes use B environments, N ships, K bullets/ship. ``observer_ship`` is
    per allied observer before team sharing; the other ship masks are indexed
    by team. Allies are always known to their own team. Enemy bullets are
    independently range/LOS tested as dynamic world objects.
    """

    observer_ship: torch.Tensor  # (B, N, N) bool, source observer -> target
    range_only_observer_ship: torch.Tensor  # (B, N, N) bool
    ship: torch.Tensor  # (B, 2, N) bool
    range_only_ship: torch.Tensor  # (B, 2, N) bool
    los_ship: torch.Tensor  # (B, 2, N) bool, before successful-shot reveal
    # ``None`` when the caller declared that nothing would read projectile
    # perception, so it was never computed. Readers must not treat that as
    # "no bullets are visible" -- it is the absence of an answer.
    bullet: torch.Tensor | None  # (B, 2, N, K) bool
    vision_range: float | None

    def for_team(self, team: int) -> torch.Tensor:
        if team not in (0, 1):
            raise ValueError(f"team perspective must be 0 or 1, got {team}")
        return self.ship[:, team]


def _occluder_cores(
    state: TensorState,
    env_config: EnvConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(B, M)`` centres and radii of every opaque core in the world.

    A field's core is its nominal interface radius less half the transition
    band, so the gradual part of the interface stays transparent and only the
    flat interior blocks. A zone has no band, so its core is its full radius.
    Zones contribute only where the environment declares them opaque.
    """

    centers = [state.field_pos]
    radii = [(state.field_radius - 0.5 * state.field_transition_width).clamp(min=0.0)]
    if env_config.zones_occlude and state.num_zones > 0:
        centers.append(state.zone_pos)
        radii.append(state.zone_radius)
    if len(centers) == 1:
        return centers[0], radii[0]
    return torch.cat(centers, dim=1), torch.cat(radii, dim=1)


def _line_of_sight_clear(
    observer: torch.Tensor,
    target: torch.Tensor,
    core_pos: torch.Tensor,
    core_radius: torch.Tensor,
    world_size: tuple[float, float],
) -> torch.Tensor:
    """Return clear shortest-path sight lines for ``(B, O)`` x ``(B, T)`` points.

    A core blocks the line whenever the line crosses its boundary. That single
    test covers all three ways sight is lost -- looking into a core from
    outside, out of one from inside, and past one that lies between -- because
    each of them crosses the boundary an odd or even number of times but always
    at least once.

    The one exemption is a line with *both* endpoints inside the same core.
    A core is a disk and therefore convex, so such a line never leaves it and
    nothing is occluded: ships sharing a field still see each other.

    Each core is located by minimum image from the observer, which is the same
    copy both ends of a line would choose only while ``vision_range`` plus the
    largest core radius stays within half a world. Frontline is 1024 + 750
    against a half-world of 8192, so the test is exact there; a configuration
    that violated the bound could return asymmetric verdicts near the seam.
    """

    batch, num_observers = observer.shape
    num_targets = target.shape[1]
    if core_pos.shape[1] == 0:
        return torch.ones(
            (batch, num_observers, num_targets), dtype=torch.bool, device=observer.device
        )

    segment = toroidal_displacement(
        target.unsqueeze(1) - observer.unsqueeze(2), world_size
    )  # (B, O, T)
    center = toroidal_displacement(
        core_pos[:, None, None, :] - observer[:, :, None, None],
        world_size,
    )  # (B, O, 1, M)
    segment_m = segment.unsqueeze(-1)
    denom = segment.abs().square().unsqueeze(-1).clamp(min=EPS)
    projection = (center.real * segment_m.real + center.imag * segment_m.imag) / denom
    # Clamping the projection into the segment makes ``closest`` the nearest
    # point *on the segment*, so this one distance also answers the endpoint
    # cases: a line that stops short of a core never crosses it.
    closest = center - projection.clamp(0.0, 1.0) * segment_m

    radius = core_radius[:, None, None, :]
    observer_inside = center.abs() < radius
    target_inside = (center - segment_m).abs() < radius
    blocked = (closest.abs() < radius) & ~(observer_inside & target_inside)
    return ~blocked.any(dim=-1)


def _team_share(
    per_observer: torch.Tensor,
    state: TensorState,
) -> torch.Tensor:
    """Reduce ``(B, N, T)`` per-observer sight into ``(B, 2, T)`` team sight."""

    observer_team = state.ship_team_id[:, :, None]
    observer_alive = state.ship_alive[:, :, None]
    return torch.stack(
        [(per_observer & observer_alive & (observer_team == team)).any(dim=1) for team in (0, 1)],
        dim=1,
    )


def team_visibility_from_state(
    state: TensorState,
    ship_config: ShipConfig,
    env_config: EnvConfig,
    perceive_bullets: bool = True,
) -> TeamVisibility:
    """Compute current team-shared visibility without host synchronization.

    ``vision_range=None`` is the explicit omniscient compatibility mode. A
    finite range enables both distance and opaque-core LOS checks using shortest
    toroidal paths.

    ``perceive_bullets=False`` leaves ``TeamVisibility.bullet`` as None instead
    of testing every projectile. Projectile LOS is the widest tensor in the
    environment step -- ``(B, N, N*K, M)`` against the ships' ``(B, N, N, M)``
    -- so a profile whose policy never reads bullets should not pay for it. The
    result is None rather than an empty mask so that a reader which does need it
    fails immediately instead of silently seeing an empty sky.
    """

    batch, num_ships = state.ship_pos.shape
    device = state.device
    target_alive = state.ship_alive.unsqueeze(1)
    if env_config.vision_range is None:
        observer = target_alive.expand(batch, num_ships, num_ships)
        team_ship = state.ship_alive[:, None, :].expand(batch, 2, num_ships)
        bullet = (
            state.bullet_active[:, None, :, :].expand(batch, 2, num_ships, state.max_bullets)
            if perceive_bullets
            else None
        )
        return TeamVisibility(
            observer,
            observer,
            team_ship,
            team_ship,
            team_ship,
            bullet,
            env_config.vision_range,
        )

    displacement = toroidal_displacement(
        state.ship_pos.unsqueeze(1) - state.ship_pos.unsqueeze(2),
        ship_config.world_size,
    )
    in_range = displacement.abs() <= env_config.vision_range
    range_observer = in_range & target_alive
    core_pos, core_radius = _occluder_cores(state, env_config)
    los_clear = _line_of_sight_clear(
        state.ship_pos, state.ship_pos, core_pos, core_radius, ship_config.world_size
    )
    los_observer_ship = range_observer & los_clear
    los_team_ship = _team_share(los_observer_ship, state)
    # A successful shot is an observable event: it reveals the firing ship to
    # both teams for this state sample even beyond ordinary range or through a
    # field shadow. Dead or cooldown-blocked shoot commands never set this flag.
    shooting_target = (state.ship_is_shooting & state.ship_alive).unsqueeze(1)
    observer_ship = los_observer_ship | shooting_target
    team_ship = _team_share(observer_ship, state)
    range_team_ship = _team_share(range_observer, state)

    # A team always knows its own slots, including a currently dead slot. Enemy
    # alive state remains masked unless an allied observer actually sees it.
    for team in (0, 1):
        allies = state.ship_team_id == team
        team_ship[:, team] |= allies
        range_team_ship[:, team] |= allies
        los_team_ship[:, team] |= allies

    if env_config.deploy_reveal_steps > 0:
        # Deployment reveal: both fleets see the whole board for the opening
        # ticks. Applied to the operative mask only -- ``range_only_ship`` and
        # ``los_ship`` stay pure geometry, so the fog diagnostics keep measuring
        # what range and line of sight actually occlude rather than reporting
        # the reveal back to us as a perception result.
        deployed = (state.step_count < env_config.deploy_reveal_steps).view(batch, 1, 1)
        team_ship = team_ship | deployed

    if not perceive_bullets:
        bullet = None
    elif state.max_bullets == 0:
        bullet = torch.zeros((batch, 2, num_ships, 0), dtype=torch.bool, device=device)
    else:
        flat_bullets = state.bullet_pos.reshape(batch, num_ships * state.max_bullets)
        bullet_disp = toroidal_displacement(
            flat_bullets.unsqueeze(1) - state.ship_pos.unsqueeze(2),
            ship_config.world_size,
        )
        bullet_in_range = bullet_disp.abs() <= env_config.vision_range
        bullet_los = _line_of_sight_clear(
            state.ship_pos, flat_bullets, core_pos, core_radius, ship_config.world_size
        )
        seen = _team_share(bullet_in_range & bullet_los, state).reshape(
            batch, 2, num_ships, state.max_bullets
        )
        active = state.bullet_active[:, None]
        shooter_team = state.ship_team_id[:, None, :, None]
        perspective = torch.arange(2, device=device).view(1, 2, 1, 1)
        own = shooter_team == perspective
        bullet = active & (own | seen)

    return TeamVisibility(
        observer_ship=observer_ship,
        range_only_observer_ship=range_observer,
        ship=team_ship,
        range_only_ship=range_team_ship,
        los_ship=los_team_ship,
        bullet=bullet,
        vision_range=env_config.vision_range,
    )
