"""The game's public functions over the contracts.

    rules = rules_of(config); types = ship_types(device)
    state = reset(rules, types, uniforms(rules, B), fogged)
    moved, sweep = move(rules, types, state, action, spread(rules, B))
    state, events = resolve(rules, types, moved, sweep, action, uniforms(rules, B))
    state, obs, privileged = perceive(rules, state)

Every function is pure (tensors in, tensors out) and works eagerly or under
torch.compile, on any device: at any batch size for one game (`rules_of(config)`), and at
the pools' batch size for several (`rules_of(config, pools, environments)`). The state is
opaque outside `bnb.game`: other packages hold it and pass it back, and read it only through
these functions.
Unequal teams at one static size: `reset`, `resolve`, `step` and `perceive` take `active`
(B, 2) int32, each team's active ships per environment (None: every ship); a team's last
S - active slots are benched, kept as dead ships waiting to respawn (`bnb.game.physics.step`).
The tick's randomness is an explicit input; `spread` and `uniforms` draw it. The ship card
`types` is an argument of every function that reads it (`reset`, `move`, `resolve`, `step`):
it is data, which a sweep may change between calls; `default_card` gives the default
card's values in the form a configuration holds.

For a runtime that keeps the state in persistent buffers: `perceived` (the part of the state
`perceive` changes, to copy back), `episode_ticks` (each environment's episode length),
`launch_capture` (what the learner's shot re-simulation stores of a tick), `ship_team` and
`view_order` (each ship's team, and the sim index of each observation row), `start_clocks`
and `sweep_like`; `retune`, the state carried over to new rules or a new card of the same
sizes (live tuning).

For live tuning (the play shell's tuning panel): `tuning` (the rules a running match can
change, by key, and the rules with changed values), the ship card's handles (`Handles`,
`HANDLE_NAMES`, `FLOWN_HANDLES`, `HANDLE_UNITS`), `measure_handles` (a card's handles),
`solve_handles` (the card meeting handles, exact, with `SolveProgress` reports; its
acceptance `SOLVE_ACCEPTANCE` against `solve_error_floor`), `FLIGHT_ONLY_HANDLES` (the
handles that change the flight, not the guns), `combat_card` (a card's combat fields in
closed form), `Estimate`
(an instant estimate of that card, anchored on an exact one), `bleed_rate` (a card's bleed
rate in closed form, shown beside the handles) and `Infeasible`; `ships` (the ship files:
one per type, tracked in git, in ship sets: `ships.sets()`; `STANDARD_SHIP_SET` is the
game's) and `reload_ships` (read them again after writing); `similarity` (every handle,
card field and rule value by its units, for the time scale);
`flight_tick` (one tick of flight, which the handles' flights repeat: the runtime compiles
it for them).

For display (the play shell), with no rules of their own: `display` (the state's values a
display draws, by name), `log_index` and `log_index_gradient` (the medium at points),
`core_radius` (the discs the sight test treats as cores), `zone_sides`
(Frontline's zone owners and attackers), `visible` (each team's observation in sim order),
`vision_range`, `per_ship` and `type_names` (the ship types' names, from the ship files);
`bullet_paths` (each live bullet's flight ahead, by the game's own transport); and the
constants `TEAMS`, `LN_SQRT2` (ln n per field level),
`MAP_SCALE_RADIUS` (the arena radius of map scale 1), the game presets `PRESETS` and the
lineups of those whose lineup is their own (`LINEUPS`, as type indices by `preset_lineup`).
"""

import torch

from bnb.contracts import Obs
from bnb.contracts.obs import SHIP

from . import ships, tuning
from .handling import FLOWN as FLOWN_HANDLES
from .handling import NAMES as HANDLE_NAMES
from .handling import Handles, Infeasible, similarity
from .handling import measure as measure_handles
from .handling import solve as solve_handles
from .handling.estimate import Estimate
from .handling.estimate import bleed_rate_now as bleed_rate
from .handling.handles import FLIGHT as FLIGHT_ONLY_HANDLES
from .handling.handles import UNITS as HANDLE_UNITS
from .handling.solver import ACCEPTANCE as SOLVE_ACCEPTANCE
from .handling.solver import Progress as SolveProgress
from .handling.solver import error_floor as solve_error_floor
from .handling.weapons import combat_card, combat_handles
from .layout import environment, per_mode
from .modes import frontline, phase
from .modes.rules import FLEET_PRESETS, LINEUPS, PRESETS
from .perception import sight
from .perception.observe import perceive, view_order
from .perception.sight import vision_range
from .physics.bullets import Launch, launch_state, move_bullets
from .physics.flight import flight_tick
from .physics.medium import medium
from .physics.step import move, reset, resolve, step
from .rules import LN_SQRT2, MAP_SCALE_RADIUS, TEAMS, Rules, bullet_slots_for, rules_of
from .ship_types import STANDARD as STANDARD_SHIP_SET
from .ship_types import ShipTypes, default_card, per_ship, ship_types, type_names
from .ship_types import reload as reload_ships
from .state import Fields, State, Sweep, replace, ship_team, sweep_like

__all__ = [
    "DISPLAYED",
    "Estimate",
    "FLIGHT_ONLY_HANDLES",
    "FLOWN_HANDLES",
    "HANDLE_NAMES",
    "HANDLE_UNITS",
    "Handles",
    "Infeasible",
    "SOLVE_ACCEPTANCE",
    "SolveProgress",
    "STANDARD_SHIP_SET",
    "LN_SQRT2",
    "Launch",
    "MAP_SCALE_RADIUS",
    "PRESETS",
    "FLEET_PRESETS",
    "LINEUPS",
    "Rules",
    "ShipTypes",
    "State",
    "Sweep",
    "TEAMS",
    "bleed_rate",
    "bullet_paths",
    "bullet_slots_for",
    "combat_card",
    "gun_range",
    "core_radius",
    "default_card",
    "display",
    "episode_ticks",
    "flight_tick",
    "launch_capture",
    "log_index",
    "log_index_gradient",
    "measure_handles",
    "move",
    "per_ship",
    "perceive",
    "perceived",
    "preset_lineup",
    "reset",
    "reload_ships",
    "resolve",
    "retune",
    "rules_of",
    "ship_team",
    "ship_types",
    "ships",
    "similarity",
    "solve_error_floor",
    "solve_handles",
    "spread",
    "start_clocks",
    "step",
    "sweep_like",
    "tuning",
    "type_names",
    "uniforms",
    "view_order",
    "visible",
    "vision_range",
    "zone_sides",
]


def spread(rules: Rules, batch: int, device=None, generator=None) -> torch.Tensor:
    """(B, N, 2) standard normals: the bullet spread of one tick."""
    return torch.randn((batch, rules.num_ships, 2), device=device, generator=generator)


def preset_lineup(name: str) -> tuple[int, ...] | None:
    """The lineup of the preset `name` as ship type indices of its set (each slot's, the
    same for both teams: `Rules.team_types`), or None for one that plays the game's
    default."""
    if name not in LINEUPS:
        return None
    names = type_names(PRESETS[name].ship_set)
    return tuple(names.index(type_name) for type_name in LINEUPS[name])


def uniforms(rules: Rules, batch: int, device=None, generator=None) -> torch.Tensor:
    """(B, U) uniforms in [0, 1): a tick's map, formation rotations and keys."""
    return torch.rand((batch, rules.tick_uniform_count), device=device, generator=generator)


def retune(rules: Rules, types: ShipTypes, state: State) -> State:
    """The state carried over to new rules or a new card of the same sizes (live tuning):
    Frontline's zone ring rebuilt for the rules' zone radius, ring fraction and arena, and
    each ship's shield and store capped at its type's new maximum. The map and the spawns
    are left as drawn; they follow the rules from the next episode."""
    batch = state.ships.x.shape[0]
    arena = environment(rules, batch, state.ships.x.device)
    mode = per_mode(
        rules,
        batch,
        lambda mode_rules, arena, mode: phase.refit_zones(
            mode_rules, arena.arena_radius, arena.map_scale, mode
        ),
        arena,
        state.mode,
    )
    ships = state.ships
    ships = replace(
        ships,
        shield=torch.minimum(ships.shield, per_ship(types.max_shield, ships.ship_type)),
        energy=torch.minimum(ships.energy, per_ship(types.max_energy, ships.ship_type)),
    )
    return replace(state, ships=ships, mode=mode)


def gun_range(rules: Rules, types: ShipTypes, cruise_speed: torch.Tensor) -> torch.Tensor:
    """(num_types,) each type's gun range handle: how far a bullet fired at cruise along the
    flight path travels in the rules' bullet life (`combat_handles`)."""
    return combat_handles(rules, types, cruise_speed)["gun_range"]


def start_clocks(state: State, episode_tick: torch.Tensor) -> State:
    """The state with its episode clocks set to episode_tick (B,) (to stagger resets)."""
    return replace(state, match=replace(state.match, episode_tick=episode_tick.to(torch.int32)))


def perceived(state: State) -> tuple[torch.Tensor, ...]:
    """The part of the state that `perceive` changes, as a pytree of tensors: a caller that
    keeps the state in persistent buffers copies the new state's part onto its own, leaf by
    leaf, without naming them."""
    match = state.match
    return (
        match.ticks_since_seen,
        match.previous_velocity_x,
        match.previous_velocity_y,
        match.previous_nose_rate,
        match.previous_alive,
        match.previously_seen,
    )


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


def log_index_gradient(
    centre_x: torch.Tensor,
    centre_y: torch.Tensor,
    radius: torch.Tensor,
    band_width: torch.Tensor,
    level: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(ln n, d/dx ln n, d/dy ln n), each (B, P), at points x, y (B, P): `log_index` with
    the gradient a ship's row shows (`log_index_gradient_x`, `log_index_gradient_y`)."""
    fields = Fields(x=centre_x, y=centre_y, radius=radius, band_width=band_width, level=level)
    return medium(fields, x, y)


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


def bullet_paths(
    rules: Rules, types: ShipTypes, state: State
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Each live bullet's flight ahead through the fields, by the game's own transport
    (`move_bullets`, as the learner's shot re-simulation flies them), ignoring the ships:
    x, y (B, N, K, A) m after each of the next A = bullet_moves moves, and valid (B, N, K,
    A), the bullet still flies that move (it has that many moves left)."""
    bullets, owner_type = state.bullets, state.ships.ship_type
    xs, ys, valid = [], [], []
    for _ in range(rules.bullet_moves):
        flying = bullets.moves_left > 0
        bullets = move_bullets(rules, types, bullets, state.fields, owner_type)
        bullets = replace(bullets, moves_left=torch.where(flying, bullets.moves_left - 1, 0))
        xs.append(bullets.x)
        ys.append(bullets.y)
        valid.append(flying)
    return torch.stack(xs, -1), torch.stack(ys, -1), torch.stack(valid, -1)


def visible(rules: Rules, observation: Obs) -> torch.Tensor:
    """(B, 2, N) bool: ship n is visible in team T's observation (B, 2, ...), with the
    ships put back in sim order (team-major) from each view's rows."""
    seen = observation.ships[..., SHIP["visible"]] > 0.5  # (B, 2, N) by view row
    order = view_order(rules.ships_per_team, seen.device)  # (2, N) the sim index of each row
    ordered = torch.zeros_like(seen)
    ordered[:, torch.arange(TEAMS, device=seen.device)[:, None], order] = seen
    return ordered
