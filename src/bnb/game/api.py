"""The game's public functions over the contracts.

    rules = rules_of(config); types = ship_types(device)
    state = reset(rules, types, uniforms(rules, B), fogged)
    moved, sweep = move(rules, types, state, action, spread(rules, B))
    state, events = resolve(rules, types, moved, sweep, uniforms(rules, B))
    state, obs, privileged = perceive(rules, state)

Every function is pure (tensors in, tensors out) and works eagerly or under
torch.compile, on any device: at any batch size for one game (`rules_of(config)`), and at
the pools' batch size for several (`rules_of(config, pools, num_envs)`). The state is
opaque outside `bnb.game`.
The tick's randomness is an explicit input; `spread` and `uniforms` draw it. The ship card
`types` is always passed: it is data, which a sweep may change between calls.

For display (the play shell), with no rules of their own: `log_index` (the medium at
points), `core_radius` (the discs the sight test treats as cores), `zone_sides`
(Frontline's zone owners and attackers), `visible` (each team's observation in sim order),
`vision_range` and `per_ship`; and the constants `TEAMS`, `LN_SQRT2` (ln n per field level),
`MAP_SCALE_RADIUS` (the arena radius of map scale 1) and the game presets `PRESETS`.
"""

import torch

from bnb.contracts import Obs
from bnb.contracts.obs import SHIP

from .modes import frontline
from .modes.rules import PRESETS
from .perception.observe import perceive, view_order
from .perception.sight import vision_range
from .physics.medium import medium
from .physics.step import move, reset, resolve, step
from .rules import LN_SQRT2, MAP_SCALE_RADIUS, TEAMS, Rules, rules_of
from .ship_types import ShipTypes, per_ship, ship_types
from .state import Fields, State, Sweep, replace, sweep_like

__all__ = [
    "LN_SQRT2",
    "MAP_SCALE_RADIUS",
    "PRESETS",
    "TEAMS",
    "Rules",
    "ShipTypes",
    "State",
    "Sweep",
    "core_radius",
    "log_index",
    "move",
    "per_ship",
    "perceive",
    "reset",
    "resolve",
    "rules_of",
    "ship_types",
    "spread",
    "start_clocks",
    "step",
    "sweep_like",
    "uniforms",
    "vision_range",
    "visible",
    "zone_sides",
]


def spread(rules: Rules, batch: int, device=None, generator=None) -> torch.Tensor:
    """(B, N, 2) standard normals: the bullet spread of one tick."""
    return torch.randn((batch, rules.num_ships, 2), device=device, generator=generator)


def uniforms(rules: Rules, batch: int, device=None, generator=None) -> torch.Tensor:
    """(B, U) uniforms in [0, 1): a tick's map, formation rotations and keys."""
    return torch.rand((batch, rules.tick_uniform_count), device=device, generator=generator)


def start_clocks(state: State, episode_tick: torch.Tensor) -> State:
    """The state with its episode clocks set to episode_tick (B,) (to stagger resets)."""
    return replace(state, match=replace(state.match, episode_tick=episode_tick.to(torch.int32)))


def log_index(
    centre_x: torch.Tensor,
    centre_y: torch.Tensor,
    radius: torch.Tensor,
    band_width: torch.Tensor,
    level: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    """(B, P) ln n, the medium's log refractive index, at points x, y (B, P) for the fields
    given as (B, M) centres, radii and band widths (m) and int32 levels: the game's
    `medium`, for a caller that holds fields outside a state (such as a subset of them)."""
    fields = Fields(x=centre_x, y=centre_y, radius=radius, band_width=band_width, level=level)
    return medium(fields, x, y)[0]


def core_radius(state: State) -> torch.Tensor:
    """(B, M) each field's core radius (m): the disc the sight test treats as the core
    (game spec 10.1), the radius less half the band, and 0 for a field with no flat
    interior. A test holds it to `visibility`, which builds the same cores."""
    fields = state.fields
    return (fields.radius - 0.5 * fields.band_width).clamp_min(0.0)


def zone_sides(state: State) -> tuple[torch.Tensor, torch.Tensor]:
    """Frontline's zones by their roles, (B, Z) int32 each: owner, the team a spawn or
    defense zone belongs to (-1 for the neutral zone), and attacker, the team whose capture
    the zone's meter counts (-1 for a zone without a meter). Only meaningful in a Frontline
    environment."""
    role = state.mode.role
    teams = torch.arange(TEAMS, device=role.device)
    attacker, defender, _, _ = frontline._sides(teams, role)  # (B, Z, 2) by team
    spawn = torch.stack([role == frontline.TEAM0_SPAWN, role == frontline.TEAM1_SPAWN], -1)

    def team_of(members: torch.Tensor) -> torch.Tensor:
        return torch.where(members.any(-1), members.int().argmax(-1), -1).to(torch.int32)

    return team_of(defender | spawn), team_of(attacker)


def visible(rules: Rules, observation: Obs) -> torch.Tensor:
    """(B, 2, N) bool: ship n is visible in team T's observation (B, 2, ...), with the
    ships put back in sim order (team-major) from each view's rows."""
    seen = observation.ships[..., SHIP["visible"]] > 0.5  # (B, 2, N) by view row
    order = view_order(rules.ships_per_team, seen.device)  # (2, N) the sim index of each row
    ordered = torch.zeros_like(seen)
    ordered[:, torch.arange(TEAMS, device=seen.device)[:, None], order] = seen
    return ordered
