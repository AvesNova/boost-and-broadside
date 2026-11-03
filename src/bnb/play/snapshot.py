"""The render snapshot: what drawing needs from one tick of one match, as numpy arrays.

`snapshot` is the only place the play shell reads the game state, through `bnb.game.api`
(`display` and the display helpers), never its fields. It is pure: it copies
every value it takes, so a snapshot never changes when the sim steps on. Rendering reads
snapshots and never imports torch.
"""

import dataclasses
import math

import numpy as np
import torch

from bnb.contracts import Obs
from bnb.contracts.obs import GLOBAL
from bnb.game import api
from bnb.game.api import Rules, ShipTypes, State


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
    that blocks sight (`bnb.game.api.core_radius`)."""

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
    team 0) and front_threshold (Frontline). fogged: the match hides what each team cannot
    see (without it a team's view has no fog). log_index_per_level: ln n per field level;
    stall_angle (rad): the game's constants the drawing scales by."""

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
    fogged: bool
    log_index_per_level: float
    stall_angle: float


def _numpy(value: torch.Tensor) -> np.ndarray:
    return value.detach().cpu().numpy().copy()


def _ships(rules: Rules, types: ShipTypes, state: State, values: dict, observation: Obs) -> Ships:
    def value(name: str) -> torch.Tensor:
        return values[f"ships.{name}"]

    heading = torch.atan2(value("heading_sin"), value("heading_cos"))
    angle_of_attack = value("angle_of_attack")
    flight_path = heading - angle_of_attack
    ship_type = value("ship_type")
    team = torch.arange(rules.num_ships) // rules.ships_per_team
    return Ships(
        x=_numpy(value("x")[0]),
        y=_numpy(value("y")[0]),
        heading=_numpy(heading[0]),
        flight_path=_numpy(torch.remainder(flight_path[0] + math.pi, 2 * math.pi) - math.pi),
        angle_of_attack=_numpy(angle_of_attack[0]),
        nose_rate=_numpy(value("nose_rate")[0]),
        shield=_numpy((value("shield") / api.per_ship(types.max_shield, ship_type))[0]),
        energy=_numpy((value("energy") / api.per_ship(types.max_energy, ship_type))[0]),
        alive=_numpy(value("alive")[0]),
        team=_numpy(team),
        ship_type=_numpy(ship_type[0]),
        fired=_numpy(value("fired")[0]),
        spawned=_numpy(value("spawned")[0]),
        respawn_wait=_numpy(value("respawn_wait")[0].float() * rules.tick_length),
        hit_radius=_numpy(api.per_ship(types.hit_radius, ship_type)[0]),
        vision_range=_numpy(api.vision_range(rules, state)[0]),
        visible=_numpy(api.visible(rules, observation)[0]),
    )


def _bullets(rules: Rules, values: dict) -> Bullets:
    N, K = rules.num_ships, rules.bullet_slots
    owner = torch.arange(N).repeat_interleave(K)

    def value(name: str) -> np.ndarray:
        return _numpy(values[f"bullets.{name}"][0].reshape(-1))

    moves_left = value("moves_left")
    return Bullets(
        x=value("x"),
        y=value("y"),
        direction_x=value("direction_x"),
        direction_y=value("direction_y"),
        active=moves_left > 0,
        moves_left=moves_left,
        owner=_numpy(owner),
        team=_numpy(owner // rules.ships_per_team),
    )


def _fields(state: State, values: dict) -> Fields:
    def value(name: str) -> np.ndarray:
        return _numpy(values[f"fields.{name}"][0])

    return Fields(
        x=value("x"),
        y=value("y"),
        radius=value("radius"),
        band_width=value("band_width"),
        level=value("level"),
        core_radius=_numpy(api.core_radius(state)[0]),
    )


def _zones(rules: Rules, state: State, values: dict) -> Zones:
    role = values["mode.role"][0]
    present = torch.full_like(role, rules.pools[0].mode.name == "frontline", dtype=torch.bool)
    owner, attacker = (side[0] for side in api.zone_sides(state))
    return Zones(
        x=_numpy(values["mode.zone_x"][0]),
        y=_numpy(values["mode.zone_y"][0]),
        radius=_numpy(values["mode.zone_radius"][0]),
        role=_numpy(role),
        meter=_numpy(values["mode.progress"][0]),
        owner=_numpy(torch.where(present, owner, -1)),
        attacker=_numpy(torch.where(present, attacker, -1)),
        present=_numpy(present),
    )


def snapshot(rules: Rules, types: ShipTypes, state: State, observation: Obs) -> Snapshot:
    """The snapshot of a one-match state (batch size 1) and its observation (1, 2, ...), as
    `bnb.runtime.realtime.Realtime` holds them. types: the ship card the match plays with
    (for the shield and energy maxima and the hit radius)."""
    pool = rules.pools[0]
    values = api.display(state)
    tick = int(values["match.episode_tick"][0])
    globals_ = observation.globals[0, 0]
    return Snapshot(
        ships=_ships(rules, types, state, values, observation),
        bullets=_bullets(rules, values),
        fields=_fields(state, values),
        zones=_zones(rules, state, values),
        arena_radius=pool.arena_radius,
        mode=pool.mode.name,
        tick=tick,
        clock=tick * rules.tick_length,
        episode_length=pool.episode_ticks * rules.tick_length,
        tick_length=rules.tick_length,
        score=_numpy(values["mode.score"][0]),
        score_target=int(globals_[GLOBAL["score_target"]]),
        front=int(values["mode.front"][0]),
        front_threshold=int(globals_[GLOBAL["front_threshold"]]),
        fogged=bool(values["match.fogged"][0]),
        log_index_per_level=api.LN_SQRT2,
        stall_angle=rules.flight.stall_angle,
    )
