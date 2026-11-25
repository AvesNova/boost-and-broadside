"""The adaption layer's snapshot: what drawing needs from one tick of one lean environment,
as numpy arrays, and `to_v2`, which fills the snapshot dataclasses of v2's play shell (the
copy in `lean.view.play`, NEEDS REVIEW) from it.

`snapshot` (with `index_map` and `match`) is the only place the view reads the game: a lean
`State` and its `Constants`, by name. It copies every value it takes, so a snapshot never
changes when the game steps on. Lean's extras that v2's snapshot has no place for (the sight
ellipse's eccentricity, the tracking circle, collisions, kills, the result) stay here.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import torch

from lean.contract import State, StepOut
from lean.game import card, core

SPECTATOR = -1  # the view that sees everything; a team's view is 0 or 1


@dataclasses.dataclass(frozen=True)
class Ships:
    """(N,) per ship slot that exists, in slot order (team 0 first). Angles in radians,
    counter-clockwise from +x: heading (the nose), flight_path (the velocity's direction) and
    angle_of_attack (heading minus flight path). speed: proper speed (m/s). shield and energy:
    fractions of the type's maxima (energy is the store). hit_radius, vision_range (the sight
    ellipse's side reach p) in metres, vision_eccentricity e. visible (2, N): team T sees ship n
    (its own ships always; with fog off every ship). in_sight (2, N): team T sees ship n by
    the sight ellipse alone (no tracking circle, no reveal). tracked (2, N): team T held ship n
    on this tick, so on the next it holds it with the tracking circle (core.tracked_by_enemy;
    False for its own ships)."""

    x: np.ndarray
    y: np.ndarray
    heading: np.ndarray
    flight_path: np.ndarray
    angle_of_attack: np.ndarray
    speed: np.ndarray
    shield: np.ndarray
    energy: np.ndarray
    alive: np.ndarray
    team: np.ndarray
    ship_type: np.ndarray
    fired: np.ndarray
    hit_radius: np.ndarray
    vision_range: np.ndarray
    vision_eccentricity: np.ndarray
    visible: np.ndarray
    in_sight: np.ndarray
    tracked: np.ndarray


@dataclasses.dataclass(frozen=True)
class Bullets:
    """(N K,) every bullet slot of the ships that exist, by owner then slot (v2's layout, so
    the play shell's interpolation pairs slots across ticks): position, unit direction, speed
    (m/s), active (a real bullet in flight: ghost bullets do no damage and are not drawn),
    moves_left, the shooter (ship index) and its team, and age, the seconds since it was fired
    (0 for an idle slot)."""

    x: np.ndarray
    y: np.ndarray
    direction_x: np.ndarray
    direction_y: np.ndarray
    speed: np.ndarray
    active: np.ndarray
    moves_left: np.ndarray
    owner: np.ndarray
    team: np.ndarray
    age: np.ndarray


@dataclasses.dataclass(frozen=True)
class Fields:
    """(F,) the environment's fields: centre, radius and band width (m), level (half-octaves of
    the refractive index, signed: + thick air) and log_index (its target ln n), and the band's two edges where the sight test
    changes, the core radius max(r - w / 2, 0) and the outer radius r + w / 2 (m)."""

    x: np.ndarray
    y: np.ndarray
    radius: np.ndarray
    band_width: np.ndarray
    level: np.ndarray
    log_index: np.ndarray
    core_radius: np.ndarray
    outer_radius: np.ndarray


@dataclasses.dataclass(frozen=True)
class Events:
    """What happened on the tick that led to this snapshot. died (N,): the ship died on it;
    killer (N,): the ship that did it the most damage on that tick (-1: none, the boundary).
    Collisions (C,), the pairs that first touched (core.collisions): the two ships, the
    contact point (the midpoint after the move) and the damage each took."""

    died: np.ndarray
    killer: np.ndarray
    collision_first: np.ndarray
    collision_second: np.ndarray
    collision_x: np.ndarray
    collision_y: np.ndarray
    collision_damage_first: np.ndarray
    collision_damage_second: np.ndarray


@dataclasses.dataclass(frozen=True)
class Snapshot:
    """One tick of one environment. tick: ticks into the episode; clock and episode_length in
    seconds. arena_radius: the boundary (damage starts beyond it), boundary_ramp_width: the
    width over which the damage rises to its maximum (m). fogged: the game hides what each team
    cannot see; tracking: the tracking circle is on (GameConfig.fog_tracking). result: -1 while
    playing, 0 or 1 the winning team, 2 a draw. log_index_per_level: ln n per field level;
    stall_angle: the angle of attack the trails scale their vortices by (rad)."""

    ships: Ships
    bullets: Bullets
    fields: Fields
    events: Events
    arena_radius: float
    boundary_ramp_width: float
    tick: int
    clock: float
    episode_length: float
    tick_length: float
    fogged: bool
    tracking: bool
    result: int
    log_index_per_level: float
    stall_angle: float
    type_names: tuple[str, ...]


TYPE_LABELS = {"p51d": "P-51D"}  # lean.game.card.TYPE_NAMES as the interface writes them


def numpy(value: torch.Tensor) -> np.ndarray:
    """A detached CPU copy of a tensor."""
    return value.detach().cpu().numpy().copy()


def per_slot(constants, name: str, like: torch.Tensor) -> torch.Tensor:
    """A ship constant as (E, N): a uniform number broadcast, or the per-slot row."""
    return torch.as_tensor(constants[name], dtype=torch.float32, device=like.device).expand_as(like)


def _ships(config: core.GameConfig, state: State, environment: int, exists: torch.Tensor) -> Ships:
    constants = state.constants
    heading = torch.atan2(state.heading_sin, state.heading_cos)
    angle = state.angle
    flight_path = torch.remainder(heading - angle + torch.pi, 2.0 * torch.pi) - torch.pi
    team = state.team
    if state.seen is None:  # fog off: every ship visible to both teams
        enemy_sees = in_sight = torch.ones_like(state.alive)
        tracked = torch.zeros_like(state.alive)
    else:
        enemy_sees = (state.ticks_since_seen == 0) & (state.seen[:, core.SEEN_KNOWN] > 0.5)
        in_sight = core.sighted_by_enemy(state)
        tracked = core.tracked_by_enemy(state) if config.fog_tracking else torch.zeros_like(
            state.alive
        )
    # (E, 2, N): row T is team T's view; a team always sees its own ships.
    own = team[:, None, :] == torch.arange(2, device=team.device)[None, :, None]
    visible = own | enemy_sees[:, None, :]
    sight = own | in_sight[:, None, :]
    held = ~own & tracked[:, None, :]

    def take(value: torch.Tensor) -> np.ndarray:
        return numpy(value[environment][..., exists])

    return Ships(
        x=take(state.x),
        y=take(state.y),
        heading=take(heading),
        flight_path=take(flight_path),
        angle_of_attack=take(angle),
        speed=take(state.speed),
        shield=take(state.shield / per_slot(constants, "max_shield", state.x)),
        energy=take(state.energy * per_slot(constants, "inverse_max_energy", state.x)),
        alive=take(state.alive),
        team=take(team),
        ship_type=take(state.type),
        fired=take(state.fired),
        hit_radius=take(per_slot(constants, "hit_radius", state.x)),
        vision_range=take(per_slot(constants, "vision_range", state.x)),
        vision_eccentricity=take(per_slot(constants, "vision_eccentricity", state.x)),
        visible=take(visible),
        in_sight=take(sight),
        tracked=take(held),
    )


def _bullets(
    config: core.GameConfig, state: State, environment: int, exists: torch.Tensor
) -> Bullets:
    moves = state.bullet_moves_left[environment][exists]  # (N, K)
    active = (moves > 0) & state.bullet_real[environment][exists]
    index = torch.arange(active.shape[0], device=active.device)[:, None].expand_as(active)
    fired = state.bullet_fired_tick[environment][exists]
    age = (state.tick[environment] - fired).to(torch.float32) * config.tick_length

    def take(value: torch.Tensor) -> np.ndarray:
        return numpy(value[environment][exists].reshape(-1))

    return Bullets(
        x=take(state.bullet_x),
        y=take(state.bullet_y),
        direction_x=take(state.bullet_dir_x),
        direction_y=take(state.bullet_dir_y),
        speed=take(state.bullet_speed),
        active=numpy(active.reshape(-1)),
        moves_left=numpy(moves.reshape(-1)),
        owner=numpy(index.reshape(-1)),
        team=numpy(state.team[environment][exists][:, None].expand_as(active).reshape(-1)),
        age=numpy(torch.where(active, age, 0.0).reshape(-1)),
    )


def fields_of(state: State, environment: int = 0) -> Fields:
    """The environment's fields (fixed for an episode)."""
    rows = state.fields[environment]  # (5, F)
    radius, band = rows[core.FIELD_RADIUS], rows[core.FIELD_BAND]
    return Fields(
        x=numpy(rows[core.FIELD_X]),
        y=numpy(rows[core.FIELD_Y]),
        radius=numpy(radius),
        band_width=numpy(band),
        level=numpy(torch.round(rows[core.FIELD_TARGET] / card.LN_SQRT2)),
        log_index=numpy(rows[core.FIELD_TARGET]),
        core_radius=numpy((radius - 0.5 * band).clamp_min(0.0)),
        outer_radius=numpy(radius + 0.5 * band),
    )


def no_events(ships: int) -> Events:
    """The events of a tick where nothing happened (a deal)."""
    empty = np.zeros(0, np.float32)
    return Events(
        died=np.zeros(ships, bool),
        killer=np.full(ships, -1, np.int64),
        collision_first=np.zeros(0, np.int64),
        collision_second=np.zeros(0, np.int64),
        collision_x=empty,
        collision_y=empty,
        collision_damage_first=empty,
        collision_damage_second=empty,
    )


def _events(
    config: core.GameConfig,
    previous: State,
    state: State,
    step_out: StepOut,
    environment: int,
    exists: torch.Tensor,
) -> Events:
    died = step_out.died[environment]
    damage = step_out.hit_damage[environment]  # (attacker, target)
    killer = torch.where(damage.amax(0) > 0.0, damage.argmax(0), -1)
    # The pairs that first touched: the game's own test on the same moves (lean.game.core).
    crash = core.collisions(
        previous,
        previous.alive,
        config.tick_length,
        state.x,
        state.y,
        state.heading_cos,
        state.heading_sin,
    )
    pairs = torch.nonzero(torch.triu(crash.contact[environment], 1))
    first, second = pairs[:, 0], pairs[:, 1]
    x, y = state.x[environment], state.y[environment]
    # Slot indices to ship indices (the snapshot keeps the slots that exist, in order).
    index = torch.cumsum(exists.long(), 0) - 1
    return Events(
        died=numpy(died[exists]),
        killer=numpy(torch.where(killer >= 0, index[killer.clamp_min(0)], -1)[exists]),
        collision_first=numpy(index[first]),
        collision_second=numpy(index[second]),
        collision_x=numpy(0.5 * (x[first] + x[second])),
        collision_y=numpy(0.5 * (y[first] + y[second])),
        collision_damage_first=numpy(crash.damage[environment][first, second]),
        collision_damage_second=numpy(crash.damage[environment][second, first]),
    )


@torch.no_grad()
def snapshot(
    config: core.GameConfig,
    state: State,
    environment: int = 0,
    step_out: StepOut | None = None,
    previous: State | None = None,
    fields: Fields | None = None,
) -> Snapshot:
    """The snapshot of one environment of `state`. step_out and previous: the tick that led to
    it and the state before it (its deaths, kills and collisions); None at a deal. fields: the
    episode's fields when already taken (they are fixed within an episode)."""
    exists = state.exists[environment]
    ships = _ships(config, state, environment, exists)
    if step_out is None or previous is None:
        events = no_events(len(ships.x))
        result = -1
    else:
        events = _events(config, previous, state, step_out, environment, exists)
        result = int(step_out.result[environment])
    tick = int(state.tick[environment])
    return Snapshot(
        ships=ships,
        bullets=_bullets(config, state, environment, exists),
        fields=fields_of(state, environment) if fields is None else fields,
        events=events,
        arena_radius=config.arena_radius,
        boundary_ramp_width=config.boundary_ramp_width,
        tick=tick,
        clock=tick * config.tick_length,
        episode_length=config.clock_ticks * config.tick_length,
        tick_length=config.tick_length,
        fogged=state.seen is not None,
        tracking=config.fog and config.fog_tracking,
        result=result,
        log_index_per_level=card.LN_SQRT2,
        stall_angle=float(torch.as_tensor(state.constants["stall"]).flatten()[0]),
        type_names=tuple(TYPE_LABELS.get(name, name) for name in card.TYPE_NAMES),
    )


def to_v2(frame: Snapshot):
    """v2's play-shell snapshot (lean.view.play.snapshot.Snapshot) of a lean frame: elimination
    with no shrink, no zones, no respawns; a ghost bullet's slot is idle."""
    from .play import snapshot as v2

    ships, bullets, fields = frame.ships, frame.bullets, frame.fields
    count = len(ships.x)
    empty = np.zeros(0, np.float32)
    return v2.Snapshot(
        ships=v2.Ships(
            x=ships.x,
            y=ships.y,
            heading=ships.heading,
            flight_path=ships.flight_path,
            angle_of_attack=ships.angle_of_attack,
            nose_rate=np.zeros(count, np.float32),
            shield=ships.shield,
            energy=ships.energy,
            alive=ships.alive,
            team=ships.team,
            ship_type=ships.ship_type,
            fired=ships.fired,
            spawned=np.zeros(count, bool),
            respawn_wait=np.zeros(count, np.float32),
            hit_radius=ships.hit_radius,
            vision_range=ships.vision_range,
            visible=ships.visible,
        ),
        bullets=v2.Bullets(
            x=bullets.x,
            y=bullets.y,
            direction_x=bullets.direction_x,
            direction_y=bullets.direction_y,
            active=bullets.active,
            moves_left=bullets.moves_left,
            owner=bullets.owner,
            team=bullets.team,
            age=bullets.age,
            speed=bullets.speed,
        ),
        fields=v2.Fields(
            x=fields.x,
            y=fields.y,
            radius=fields.radius,
            band_width=fields.band_width,
            level=fields.level.astype(np.int64),
            core_radius=fields.core_radius,
            outer_radius=fields.outer_radius,
        ),
        zones=v2.Zones(
            x=empty,
            y=empty,
            radius=empty,
            role=np.zeros(0, np.int64),
            meter=empty,
            owner=np.zeros(0, np.int64),
            attacker=np.zeros(0, np.int64),
            present=np.zeros(0, bool),
            lock=np.zeros((2, 0), np.float32),
            lock_length=empty,
        ),
        arena_radius=frame.arena_radius,
        mode="elimination",
        tick=frame.tick,
        clock=frame.clock,
        episode_length=frame.episode_length,
        tick_length=frame.tick_length,
        score=np.zeros(2, np.int64),
        score_target=0,
        front=0,
        front_threshold=0,
        fogged=frame.fogged,
        log_index_per_level=frame.log_index_per_level,
        stall_angle=frame.stall_angle,
        next_wave=np.zeros(2, np.float32),
        boundary_radius=frame.arena_radius,
        boundary_time_to_zero=0.0,
    )
