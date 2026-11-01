"""The render snapshot: what drawing needs from one tick of one match, as numpy arrays.

`snapshot` is the only place the play shell reads the game state. It is pure: it copies
every value it takes, so a snapshot never changes when the sim steps on. Rendering reads
snapshots and never imports torch.
"""

import dataclasses
import math

import numpy as np
import torch

from bnb.contracts import Obs
from bnb.contracts.obs import GLOBAL, SHIP
from bnb.game.api import Rules, ShipTypes, State
from bnb.game.modes.frontline import TEAM0_DEFENSE, TEAM0_SPAWN, TEAM1_DEFENSE, TEAM1_SPAWN
from bnb.game.perception.observe import view_order
from bnb.game.perception.sight import vision_range
from bnb.game.rules import LN_SQRT2, TEAMS
from bnb.game.ship_types import per_ship


@dataclasses.dataclass(frozen=True)
class Ships:
    """(N,) per ship, in sim order (team-major). Angles in radians, counter-clockwise from
    +x: heading (the nose), flight_path (the velocity), angle_of_attack (heading minus
    flight path). shield and energy are fractions of the ship type's maxima; respawn_wait
    in seconds; hit_radius and vision_range in metres. visible (2, N): team T sees ship n
    (from team T's observation)."""

    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    flight_path: np.ndarray
    angle_of_attack: np.ndarray
    nose_rate: np.ndarray
    shield: np.ndarray
    energy: np.ndarray
    alive: np.ndarray
    team: np.ndarray
    ship_type: np.ndarray
    fired: np.ndarray
    spawned: np.ndarray
    respawn_wait: np.ndarray
    hit_radius: np.ndarray
    vision_range: np.ndarray
    visible: np.ndarray


@dataclasses.dataclass(frozen=True)
class Bullets:
    """(N K,) per bullet slot, by owner then slot: position, unit direction, active,
    moves_left, and the shooter (sim index) and its team."""

    x: np.ndarray
    y: np.ndarray
    direction_x: np.ndarray
    direction_y: np.ndarray
    active: np.ndarray
    moves_left: np.ndarray
    owner: np.ndarray
    team: np.ndarray


@dataclasses.dataclass(frozen=True)
class Fields:
    """(M,): centre, radius and band width (m), level, and the core radius (m), the disc
    that blocks sight (`core_radius`)."""

    x: np.ndarray
    y: np.ndarray
    radius: np.ndarray
    band_width: np.ndarray
    level: np.ndarray
    core_radius: np.ndarray


@dataclasses.dataclass(frozen=True)
class Zones:
    """(Z,) Frontline's zones (none present in other modes): centre and radius (m); role
    in team 0's frame (`bnb.game.modes.frontline`); meter, the capture progress in [0, 1];
    owner, the team the zone belongs to (-1 for the neutral zone); attacker, the team
    whose capture the meter counts (-1 for a zone without a meter); present."""

    x: np.ndarray
    y: np.ndarray
    radius: np.ndarray
    role: np.ndarray
    meter: np.ndarray
    owner: np.ndarray
    attacker: np.ndarray
    present: np.ndarray


@dataclasses.dataclass(frozen=True)
class Snapshot:
    """One tick of one match. tick: ticks into the episode; clock and episode_length in
    seconds. score (2,) by team and score_target (team deathmatch); front (positive for
    team 0) and front_threshold (Frontline). reset: the episode (and its map) started on
    this tick, so the static layers rebuild and nothing interpolates into it.
    log_index_per_level: ln n per field level; stall_angle (rad): the game's constants the
    drawing scales by."""

    ships: Ships
    bullets: Bullets
    fields: Fields
    zones: Zones
    arena_radius: float
    mode: str
    tick: int
    clock: float
    episode_length: float
    tick_length: float
    score: np.ndarray
    score_target: int
    front: int
    front_threshold: int
    reset: bool
    log_index_per_level: float
    stall_angle: float


def core_radius(radius: torch.Tensor, band_width: torch.Tensor) -> torch.Tensor:
    """The radius of a field's core, the disc that blocks sight: the one
    `bnb.game.perception.sight.visibility` builds from the fields (a test holds the two
    together)."""
    return (radius - 0.5 * band_width).clamp_min(0.0)


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def _ships(rules: Rules, types: ShipTypes, state: State, observation: Obs) -> Ships:
    ships = state.ships
    heading = torch.atan2(ships.heading_sin, ships.heading_cos)
    angle_of_attack = ships.angle_of_attack
    flight_path = heading - angle_of_attack
    order = view_order(rules.ships_per_team, ships.x.device)  # (2, N) sim index of each row
    seen = observation.ships[0, :, :, SHIP["visible"]] > 0.5  # (2, N) by row
    visible = torch.zeros_like(seen)
    visible[torch.arange(TEAMS)[:, None], order] = seen
    team = torch.arange(rules.num_ships) // rules.ships_per_team
    return Ships(
        x=_numpy(ships.x[0]), y=_numpy(ships.y[0]), heading=_numpy(heading[0]),
        flight_path=_numpy(torch.remainder(flight_path[0] + math.pi, 2 * math.pi) - math.pi),
        angle_of_attack=_numpy(angle_of_attack[0]), nose_rate=_numpy(ships.nose_rate[0]),
        shield=_numpy((ships.shield / per_ship(types.max_shield, ships.ship_type))[0]),
        energy=_numpy((ships.energy / per_ship(types.max_energy, ships.ship_type))[0]),
        alive=_numpy(ships.alive[0]), team=_numpy(team), ship_type=_numpy(ships.ship_type[0]),
        fired=_numpy(ships.fired[0]), spawned=_numpy(ships.spawned[0]),
        respawn_wait=_numpy(ships.respawn_wait[0].float() * rules.tick_length),
        hit_radius=_numpy(per_ship(types.hit_radius, ships.ship_type)[0]),
        vision_range=_numpy(vision_range(rules, state)[0]), visible=_numpy(visible))


def _bullets(rules: Rules, state: State) -> Bullets:
    bullets = state.bullets
    N, K = rules.num_ships, rules.bullet_slots
    owner = torch.arange(N).repeat_interleave(K)
    return Bullets(x=_numpy(bullets.x[0].reshape(-1)), y=_numpy(bullets.y[0].reshape(-1)),
                   direction_x=_numpy(bullets.direction_x[0].reshape(-1)),
                   direction_y=_numpy(bullets.direction_y[0].reshape(-1)),
                   active=_numpy(bullets.moves_left[0].reshape(-1) > 0),
                   moves_left=_numpy(bullets.moves_left[0].reshape(-1)),
                   owner=_numpy(owner), team=_numpy(owner // rules.ships_per_team))


def _fields(state: State) -> Fields:
    fields = state.fields
    return Fields(x=_numpy(fields.x[0]), y=_numpy(fields.y[0]), radius=_numpy(fields.radius[0]),
                  band_width=_numpy(fields.band_width[0]), level=_numpy(fields.level[0]),
                  core_radius=_numpy(core_radius(fields.radius, fields.band_width)[0]))


def _zones(rules: Rules, state: State) -> Zones:
    mode = state.mode
    role = mode.role[0]
    present = torch.full_like(role, rules.pools[0].mode.name == "frontline", dtype=torch.bool)
    owner = torch.where((role == TEAM0_SPAWN) | (role == TEAM0_DEFENSE), 0,
                        torch.where((role == TEAM1_SPAWN) | (role == TEAM1_DEFENSE), 1, -1))
    attacker = torch.where(role == TEAM0_DEFENSE, 1, torch.where(role == TEAM1_DEFENSE, 0, -1))
    return Zones(x=_numpy(mode.zone_x[0]), y=_numpy(mode.zone_y[0]),
                 radius=_numpy(mode.zone_radius[0]), role=_numpy(role),
                 meter=_numpy(mode.progress[0]), owner=_numpy(torch.where(present, owner, -1)),
                 attacker=_numpy(torch.where(present, attacker, -1)), present=_numpy(present))


def snapshot(rules: Rules, types: ShipTypes, state: State, observation: Obs) -> Snapshot:
    """The snapshot of a one-match state (batch size 1) and its observation (1, 2, ...), as
    `bnb.runtime.realtime.Realtime` holds them. types: the ship card the match plays with
    (for the shield and energy maxima and the hit radius)."""
    pool = rules.pools[0]
    tick = int(state.match.episode_tick[0])
    globals_ = observation.globals[0, 0]
    return Snapshot(
        ships=_ships(rules, types, state, observation), bullets=_bullets(rules, state),
        fields=_fields(state), zones=_zones(rules, state), arena_radius=pool.arena_radius,
        mode=pool.mode.name, tick=tick, clock=tick * rules.tick_length,
        episode_length=pool.episode_ticks * rules.tick_length, tick_length=rules.tick_length,
        score=_numpy(state.mode.score[0]), score_target=int(globals_[GLOBAL["score_target"]]),
        front=int(state.mode.front[0]), front_threshold=int(globals_[GLOBAL["front_threshold"]]),
        reset=tick == 0, log_index_per_level=LN_SQRT2, stall_angle=rules.flight.stall_angle)
