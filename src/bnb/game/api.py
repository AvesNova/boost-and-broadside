"""The game's public functions over the contracts.

    rules = rules_of(config); types = ship_types(device)
    state = reset(rules, types, uniforms(rules, B), fogged)
    moved, sweep = move(rules, types, state, action, spread(rules, B))
    state, events = resolve(rules, types, moved, sweep, uniforms(rules, B))
    state, obs, privileged = perceive(rules, state)

Every function is pure (tensors in, tensors out) and works eagerly or under
torch.compile, on any device: at any batch size for one game (`rules_of(config)`), and at
the pools' batch size for several (`rules_of(config, pools, environments)`). The state is
opaque outside `bnb.game`: other packages hold it and pass it back, and read it only through
these functions.
The tick's randomness is an explicit input; `spread` and `uniforms` draw it. The ship card
`types` is an argument of every function that reads it (`reset`, `move`, `resolve`, `step`):
it is data, which a sweep may change between calls.

For a runtime that keeps the state in persistent buffers: `perceived` (the part of the state
`perceive` changes, to copy back), `episode_ticks` (each environment's episode length),
`launch_capture` (what the learner's shot re-simulation stores of a tick), `ship_team` and
`view_order` (each ship's team, and the sim index of each observation row), `start_clocks`
and `sweep_like`.

For display (the play shell), with no rules of their own: `display` (the state's values a
display draws, by name), `log_index` (the medium at
points), `core_radius` (the discs the sight test treats as cores), `zone_sides`
(Frontline's zone owners and attackers), `visible` (each team's observation in sim order),
`vision_range` and `per_ship`; and the constants `TEAMS`, `LN_SQRT2` (ln n per field level),
`MAP_SCALE_RADIUS` (the arena radius of map scale 1) and the game presets `PRESETS`.
"""

import torch

from bnb.contracts import Obs
from bnb.contracts.obs import SHIP

from .layout import environment
from .modes import frontline
from .modes.rules import PRESETS
from .perception import sight
from .perception.observe import perceive, view_order
from .perception.sight import vision_range
from .physics.bullets import Launch, launch_state
from .physics.medium import medium
from .physics.step import move, reset, resolve, step
from .rules import LN_SQRT2, MAP_SCALE_RADIUS, TEAMS, Rules, rules_of
from .ship_types import ShipTypes, per_ship, ship_types
from .state import Fields, State, Sweep, replace, ship_team, sweep_like

__all__ = [
    "DISPLAYED",
    "LN_SQRT2",
    "Launch",
    "MAP_SCALE_RADIUS",
    "PRESETS",
    "Rules",
    "ShipTypes",
    "State",
    "Sweep",
    "TEAMS",
    "core_radius",
    "display",
    "episode_ticks",
    "launch_capture",
    "log_index",
    "move",
    "per_ship",
    "perceive",
    "perceived",
    "reset",
    "resolve",
    "rules_of",
    "ship_team",
    "ship_types",
    "spread",
    "start_clocks",
    "step",
    "sweep_like",
    "uniforms",
    "view_order",
    "visible",
    "vision_range",
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


def perceived(state: State) -> tuple[torch.Tensor, ...]:
    """The part of the state that `perceive` changes, as a pytree of tensors: a caller that
    keeps the state in persistent buffers copies the new state's part onto its own, leaf by
    leaf, without naming them."""
    return (state.match.ticks_since_seen,)


def episode_ticks(rules: Rules, batch: int, device=None) -> torch.Tensor:
    """(B,) int32 each environment's episode length in ticks."""
    return environment(rules, batch, device).episode_ticks


def launch_capture(
    state: State, sweep: Sweep
) -> tuple[Launch, torch.Tensor, torch.Tensor, torch.Tensor]:
    """What the learner's shot re-simulation stores of a tick, in sim order (B, N): each
    ship's launch state at the tick's start (`Launch`), its straight-line move this tick
    (x, y, m) and whether it was alive at the tick's start."""
    return (
        launch_state(state.ships),
        sweep.ship_displacement_x,
        sweep.ship_displacement_y,
        state.ships.alive,
    )


DISPLAYED = {
    "ships": (
        "x", "y", "heading_cos", "heading_sin", "angle_of_attack", "nose_rate", "shield",
        "energy", "alive", "ship_type", "fired", "spawned", "respawn_wait",
    ),
    "bullets": ("x", "y", "direction_x", "direction_y", "moves_left"),
    "fields": ("x", "y", "radius", "band_width", "level"),
    "mode": ("role", "zone_x", "zone_y", "zone_radius", "progress", "score", "front"),
    "match": ("episode_tick", "fogged"),
}  # fmt: skip


def display(state: State) -> dict[str, torch.Tensor]:
    """The state's values a display draws, named `part.value` (`DISPLAYED`), as the state
    holds them (B, ...): read-only access for the play shell, which keeps no rules."""
    return {
        f"{part}.{name}": getattr(getattr(state, part), name)
        for part, names in DISPLAYED.items()
        for name in names
    }


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
    interior."""
    return sight.core_radius(state.fields)


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
