"""GPU-vectorized team perception and refractive-field line-of-sight occlusion."""

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
    bullet: torch.Tensor  # (B, 2, N, K) bool
    vision_range: float | None

    def for_team(self, team: int) -> torch.Tensor:
        if team not in (0, 1):
            raise ValueError(f"team perspective must be 0 or 1, got {team}")
        return self.ship[:, team]


def _line_of_sight_clear(
    observer: torch.Tensor,
    target: torch.Tensor,
    state: TensorState,
    ship_config: ShipConfig,
) -> torch.Tensor:
    """Return clear shortest-path sight lines for ``(B, O)`` × ``(B, T)`` points.

    A field blocks only when its flat core lies strictly between the endpoints.
    Endpoints inside that same core are exempt, so entering a field does not make
    every unit within it globally blind. This conservative provisional rule
    supplies natural occlusion without creating a discontinuous self-blindness
    shell at field entry.
    """

    batch, num_observers = observer.shape
    num_targets = target.shape[1]
    if state.num_fields == 0:
        return torch.ones(
            (batch, num_observers, num_targets), dtype=torch.bool, device=state.device
        )

    segment = toroidal_displacement(
        target.unsqueeze(1) - observer.unsqueeze(2), ship_config.world_size
    )  # (B, O, T)
    center = toroidal_displacement(
        state.field_pos[:, None, None, :] - observer[:, :, None, None],
        ship_config.world_size,
    )  # (B, O, 1, M)
    segment_m = segment.unsqueeze(-1)
    denom = segment.abs().square().unsqueeze(-1).clamp(min=EPS)
    projection = (center.real * segment_m.real + center.imag * segment_m.imag) / denom
    closest = center - projection.clamp(0.0, 1.0) * segment_m

    core_radius = (
        state.field_radius - 0.5 * state.field_transition_width
    ).clamp(min=0.0)[:, None, None, :]
    observer_inside = center.abs() < core_radius
    target_to_center = center - segment_m
    target_inside = target_to_center.abs() < core_radius
    strictly_between = (projection > 0.0) & (projection < 1.0)
    blocked = (
        strictly_between
        & (closest.abs() < core_radius)
        & ~observer_inside
        & ~target_inside
    )
    return ~blocked.any(dim=-1)


def _team_share(
    per_observer: torch.Tensor,
    state: TensorState,
) -> torch.Tensor:
    """Reduce ``(B, N, T)`` per-observer sight into ``(B, 2, T)`` team sight."""

    observer_team = state.ship_team_id[:, :, None]
    observer_alive = state.ship_alive[:, :, None]
    return torch.stack(
        [
            (per_observer & observer_alive & (observer_team == team)).any(dim=1)
            for team in (0, 1)
        ],
        dim=1,
    )


def team_visibility_from_state(
    state: TensorState,
    ship_config: ShipConfig,
    env_config: EnvConfig,
) -> TeamVisibility:
    """Compute current team-shared visibility without host synchronization.

    ``vision_range=None`` is the explicit omniscient compatibility mode. A
    finite range enables both distance and field-core LOS checks using shortest
    toroidal paths.
    """

    batch, num_ships = state.ship_pos.shape
    device = state.device
    target_alive = state.ship_alive.unsqueeze(1)
    if env_config.vision_range is None:
        observer = target_alive.expand(batch, num_ships, num_ships)
        team_ship = state.ship_alive[:, None, :].expand(batch, 2, num_ships)
        bullet = state.bullet_active[:, None, :, :].expand(
            batch, 2, num_ships, state.max_bullets
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
    los_clear = _line_of_sight_clear(
        state.ship_pos, state.ship_pos, state, ship_config
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

    if state.max_bullets == 0:
        bullet = torch.zeros((batch, 2, num_ships, 0), dtype=torch.bool, device=device)
    else:
        flat_bullets = state.bullet_pos.reshape(batch, num_ships * state.max_bullets)
        bullet_disp = toroidal_displacement(
            flat_bullets.unsqueeze(1) - state.ship_pos.unsqueeze(2),
            ship_config.world_size,
        )
        bullet_in_range = bullet_disp.abs() <= env_config.vision_range
        bullet_los = _line_of_sight_clear(
            state.ship_pos, flat_bullets, state, ship_config
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
