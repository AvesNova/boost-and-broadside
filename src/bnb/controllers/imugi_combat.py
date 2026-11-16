"""Imugi's combat layer (scripted spec A4) and the geometry both layers share.

Everything is per own ship, (R, S), from one team's observation: the own rows are the first
S, the enemy rows the rest. An enemy row is read only where it is visible and alive, so a
hidden row's zeroed columns never matter; the target is chosen from the enemies the agent
recalls (`imugi_memory`), seen or dead reckoned.
"""

import math
import os
from typing import NamedTuple

import torch

from bnb.contracts import Obs
from bnb.contracts.obs import GLOBAL, SHIP, ZONE
from bnb.game.handling.characteristics import Characteristics, time_of_flight
from bnb.game.rules import Rules
from bnb.game.ship_types import per_ship
from bnb.streams import purpose_key, row_uniforms

from .imugi_doctrine import Doctrine
from .imugi_doctrine import per_ship as doctrine_of
from .imugi_memory import (
    TARGET_LOST_SECONDS,
    ImugiCarry,
    Recall,
    commit,
    explore_cells,
    fresh_carry,
    remember,
    time_unit,
    ways_round,
)
from .imugi_parameters import (
    ImugiParameters,
    Logistic,
    floored,
    in_own_time,
    logistic,
    ramp,
)

COAST, BOOST, REVERSE = 0, 1, 2  # power indices (bnb.game.rules.POWER_MODES)
TINY = 1e-6  # m: keeps a direction's length away from zero
STICK_MAGNITUDES = (0.0, 0.4, 0.75, 1.0)  # bnb.game.rules.STICK_LEVELS by level
TURNING_RATE = 0.05  # rad/s: a nose rate below this is not a turn (the sense maneuver)
# The edge steer fades out where the boundary's radius is a few turn radii (the ship's own,
# at its speed): from all of it at 4 to none at 2, where a turn barely fits inside and
# running from one side of the edge only meets the other; the ship fights instead.
CRITICAL_RADIUS_IN_TURN_RADII = (2.0, 4.0)
# Whether the edge steer reads the boundary's closing speed (its moving frame) and fades out
# at the critical radius; off, it is the earlier steer for a still edge (for comparison).
MOVING_EDGE = True
# What a ship does while its team knows no enemy (none seen, none remembered):
# "centre" heads for the arena's centre and holds near it (the old rule: both teams agree to
# meet there); "straight" flies on (the edge steer turns it back from the boundary);
# "explore" heads, with the team, for the nearest cell its team has not seen lately
# (`explore_target`), each ship pulled toward its team's centre of mass when it strays and
# pushed off allies that come too close.
SEARCH = "explore"  # decided 2026-11-15 (the author), with the wide spacing
SEARCH_MODES = ("centre", "straight", "explore")
# Whether the boundary's steer fades the doctrine's maneuvers (`maneuvers`: the sense and
# the extensions) by its weight. Off, a post-merge sense near the edge overrides the turn
# back and, following the target's own turn, flips the committed side every few ticks, so
# the ship flies on out (diagnosis 2026-11-16, bench/boundary_deaths.py).
BOUNDARY_OVER_MANEUVERS = os.environ.get("BNB_BOUNDARY_OVER_MANEUVERS", "1") == "1"
# the cohesion pull: none within COHESION_IN_GUN_RANGE[0] own gun ranges of the team's
# centre of mass, all of the steer beyond [1]; the separation push: from allies within
# SEPARATION_IN_GUN_RANGE, growing to all of the steer at none. SPACINGS: the variants
# measured (2026-11-15; a gun range is about 640 m and vision 1 km at real scale): "tight",
# allies about half a gun range apart; "wide", about a vision range apart, so the team
# sweeps a wider strip.
SPACINGS = {"tight": ((1.0, 3.0), 0.5), "wide": ((2.5, 4.5), 1.6)}
COHESION_IN_GUN_RANGE, SEPARATION_IN_GUN_RANGE = SPACINGS["wide"]
# The explore target: a cell counts as unseen once its seen value (`imugi_memory.track_seen`)
# has decayed below EXPLORE_UNSEEN; only cells within EXPLORE_INSIDE of the boundary's
# radius count. Of those, the team heads for the nearest to its centre of mass, distances
# shortened by up to EXPLORE_AHEAD for cells ahead of the team's mean velocity (so it keeps
# going the way it goes and no absolute direction is preferred).
EXPLORE_UNSEEN = 0.5
EXPLORE_INSIDE = 0.85
EXPLORE_AHEAD = 0.5


class Marginals(NamedTuple):
    """Per own ship: power (R, S, 3), stick (R, S, 7) and shoot (R, S, 2) probabilities."""

    power: torch.Tensor
    stick: torch.Tensor
    shoot: torch.Tensor


class Context(NamedTuple):
    """What both layers read, per own ship (R, S) unless shown; enemy rows (R, E)."""

    own: torch.Tensor  # (R, S, columns)
    enemy: torch.Tensor  # (R, E, columns)
    own_type: torch.Tensor
    enemy_type: torch.Tensor
    enemy_seen: torch.Tensor  # visible and alive
    x: torch.Tensor
    y: torch.Tensor
    velocity_x: torch.Tensor
    velocity_y: torch.Tensor
    cos: torch.Tensor
    sin: torch.Tensor
    speed: torch.Tensor  # proper
    energy_fraction: torch.Tensor
    gun_range: torch.Tensor
    stall_angle: torch.Tensor
    corner_speed: torch.Tensor
    arena_radius: torch.Tensor  # (R, 1)
    # (R, 1) the boundary's radius now: the arena radius, or less where it shrinks
    # (elimination), which the boundary's steer and the centre hold read
    boundary_radius: torch.Tensor
    distance_from_centre: torch.Tensor
    boundary_weight: torch.Tensor  # the steer toward the centre, in [0, 1]
    boundary_speed: torch.Tensor  # (R, 1) m/s: the boundary's closing speed (`track_boundary`)
    temperature: torch.Tensor | float  # the agent's: a float, or (R, 1) per view
    carry: ImugiCarry  # the memory, before this decision's plan choices
    recall: Recall  # the enemies as the agent believes them
    seconds: float  # a decision's length
    time_unit: torch.Tensor  # the ship's own time unit in seconds (`time_unit`)
    # the decision's key (R,) int64 (`Controller.step`): the plan choices may switch,
    # drawn from it (`choice_uniforms`); None: only a forced re-choice
    decision_key: torch.Tensor | None
    # Frontline's timers, s (0 in the other modes): the own team's next respawn wave (R,)
    # (the enemy's is hidden), and each zone's capture lock on the own team and on the
    # enemy's (R, Z), by the observation's zone rows. Read by the Frontline layer's
    # capture-lock term (`imugi_frontline.lock_press`).
    next_wave: torch.Tensor
    zone_lock_own: torch.Tensor
    zone_lock_enemy: torch.Tensor


def choice_uniforms(fleet: Context, name: str, like: torch.Tensor) -> torch.Tensor | None:
    """Uniforms shaped like `like` (R, ...) for the decision's draw `name` (a plan choice's
    switch), keyed by the decision's key; None for a decision that does not
    switch (`Context.decision_key`)."""
    if fleet.decision_key is None:
        return None
    # the salt keeps its recorded text, so the draws are those of every earlier run
    return row_uniforms(purpose_key(f"scripted {name}"), fleet.decision_key, like.shape[1:])


def column(rows: torch.Tensor, name: str) -> torch.Tensor:
    """The named SHIP column of ship rows (..., columns), shaped (...)."""
    return rows[..., SHIP[name]]


def context(
    obs: Obs,
    characteristics: Characteristics,
    parameters: ImugiParameters,
    temperature: torch.Tensor | float = 1.0,
    carry: ImugiCarry | None = None,
    recall: Recall | None = None,
    seconds: float = Rules.tick_length,
    decision_key: torch.Tensor | None = None,
) -> Context:
    """The Context both layers read, from one team's views (R, ...), own rows first, at the
    agent's temperature (a float, or (R, 1) per view), with its memory (`remember`'s carry
    and recall; None for a fresh one's: the memoryless agent)."""
    if carry is None:
        ships = obs.ships
        S = ships.shape[-2] // 2
        reset = torch.zeros(ships.shape[0], dtype=torch.bool, device=ships.device)
        carry, recall = remember(
            fresh_carry(ships.shape[0], S, ships.shape[-2] - S, ships.device),
            obs,
            reset,
            characteristics,
            parameters,
            temperature,
        )
    ships = obs.ships
    S = ships.shape[-2] // 2  # two teams, own rows first
    own, enemy = ships[:, :S], ships[:, S:]
    own_type = column(own, "ship_type").to(torch.int32)
    enemy_type = column(enemy, "ship_type").to(torch.int32)
    enemy_seen = (column(enemy, "visible") > 0.5) & (column(enemy, "alive") > 0.5)
    x, y = column(own, "x"), column(own, "y")
    velocity_x, velocity_y = column(own, "velocity_x"), column(own, "velocity_y")
    speed = column(own, "proper_speed")
    arena_radius = obs.globals[:, None, GLOBAL["arena_radius"]]
    boundary_radius = arena_radius * obs.globals[:, None, GLOBAL["boundary_fraction"]]
    # The boundary, in its own frame (it closes at v_b, as if steadily): within a margin of
    # turn radii of the edge and flying outward relative to it. A ship flying straight out
    # at speed v that turns back with radius r gets deepest relative to the edge when its
    # outward speed equals the edge's inward speed, r f(b) beyond where it began, b = v_b / v
    # and f(b) = sqrt(1 - b^2) + b arccos(-b) (1 for a still edge, pi when the edge is as
    # fast as the ship, which then cannot get away). The margin is the parameter's number of
    # those depths; with a still edge it is exactly the turn radii it was. Where the
    # boundary is only a few turn radii across (`CRITICAL_RADIUS_IN_TURN_RADII`) the steer
    # fades out, and the ship fights.
    distance_from_centre = torch.sqrt(x * x + y * y).clamp_min(TINY)
    turn_radius = speed * speed / per_ship(characteristics.max_acceleration, own_type)
    boundary_speed = carry.boundary_speed[:, None] * float(MOVING_EDGE)
    ratio = (boundary_speed / speed.clamp_min(TINY)).clamp(0.0, 1.0)
    depth_factor = torch.sqrt(1.0 - ratio * ratio) + ratio * torch.acos(-ratio)
    margin = (parameters.boundary_margin_in_turn_radius * turn_radius * depth_factor).clamp_min(1.0)
    depth = (distance_from_centre - (boundary_radius - margin)) / margin
    outward = (x * velocity_x + y * velocity_y) / distance_from_centre + boundary_speed > 0.0
    low, high = CRITICAL_RADIUS_IN_TURN_RADII
    room = ((boundary_radius / turn_radius.clamp_min(TINY) - low) / (high - low)).clamp(0.0, 1.0)
    room = room if MOVING_EDGE else torch.ones_like(room)
    boundary_weight = (
        torch.where(outward, ramp(depth, parameters.boundary_steer_ramp_in_margin), 0.0) * room
    )
    return Context(
        own=own,
        enemy=enemy,
        own_type=own_type,
        enemy_type=enemy_type,
        enemy_seen=enemy_seen,
        x=x,
        y=y,
        velocity_x=velocity_x,
        velocity_y=velocity_y,
        cos=column(own, "heading_cos"),
        sin=column(own, "heading_sin"),
        speed=speed,
        energy_fraction=column(own, "energy") / per_ship(characteristics.max_energy, own_type),
        gun_range=per_ship(characteristics.gun_range, own_type),
        stall_angle=per_ship(characteristics.stall_angle, own_type),
        corner_speed=per_ship(characteristics.corner_speed, own_type),
        arena_radius=arena_radius,
        boundary_radius=boundary_radius,
        distance_from_centre=distance_from_centre,
        boundary_weight=boundary_weight,
        boundary_speed=boundary_speed,
        temperature=temperature,
        carry=carry,
        recall=recall,
        seconds=seconds,
        time_unit=time_unit(characteristics, own_type, parameters),
        decision_key=decision_key,
        next_wave=obs.globals[:, GLOBAL["next_wave_own"]],
        zone_lock_own=obs.zones[..., ZONE["lock_own"]],
        zone_lock_enemy=obs.zones[..., ZONE["lock_enemy"]],
    )


def signed_angle(
    cos: torch.Tensor, sin: torch.Tensor, x: torch.Tensor, y: torch.Tensor
) -> torch.Tensor:
    """The angle from the nose (cos, sin) to the direction (x, y), positive to the left."""
    return torch.atan2(cos * y - sin * x, cos * x + sin * y)


def steer(
    fleet: Context, direction_x: torch.Tensor, direction_y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """A wanted direction with the boundary's steer toward the centre blended in."""
    length = torch.sqrt(direction_x * direction_x + direction_y * direction_y).clamp_min(TINY)
    weight = fleet.boundary_weight
    centre_x = -fleet.x / fleet.distance_from_centre
    centre_y = -fleet.y / fleet.distance_from_centre
    return (
        (1.0 - weight) * direction_x / length + weight * centre_x,
        (1.0 - weight) * direction_y / length + weight * centre_y,
    )


def stick_marginal(
    fleet: Context,
    error: torch.Tensor,
    parameters: ImugiParameters,
    committed_left: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """((R, S, 7) the stick, the committed side (R, S) bool) for an angle error (positive
    left): P(turn) and the level by two further logistics, all in stall angles, each
    floored. The side: the committed side (`committed_left`, None for the error's sign),
    which switches by side_switch_in_radians on how much shorter the other way round is,
    outweighs the error's sign by side_commit_in_stall_angle on the error, then floored (an
    error of exactly 0 picks either side evenly). Indices: 0 straight, then by level, odd
    right and even left."""
    size = error.abs() / fleet.stall_angle
    temperature = fleet.temperature
    turn = floored(logistic(size, parameters.turn_in_stall_angle, temperature), "turn")
    above_one = floored(
        logistic(size, parameters.level_two_in_stall_angle, temperature), "level_two"
    )
    above_two = floored(
        logistic(size, parameters.level_three_in_stall_angle, temperature), "level_three"
    )
    levels = torch.stack(
        [
            turn * (1.0 - above_one),
            turn * above_one * (1.0 - above_two),
            turn * above_one * above_two,
        ],
        -1,
    )  # (R, S, 3)
    sign_left = error > 0.0
    if committed_left is None:
        committed_left, forced = sign_left, torch.ones_like(sign_left)
    else:
        forced = fleet.carry.fresh
    left_way, right_way = ways_round(error)
    longer = torch.where(committed_left, left_way - right_way, right_way - left_way)
    committed_left = commit(
        committed_left,
        sign_left,
        longer,
        forced,
        in_own_time(parameters.side_switch_in_radians, fleet.time_unit),
        temperature,
        fleet.seconds,
        choice_uniforms(fleet, "side", longer),
    )
    weight = logistic(size, parameters.side_commit_in_stall_angle, temperature)
    left = weight * committed_left.float() + (1.0 - weight) * sign_left.float()
    left = torch.where(error == 0.0, 0.5, floored(left, "side"))[..., None]
    sides = torch.stack([(1.0 - left) * levels, left * levels], -1)
    return torch.cat([(1.0 - turn)[..., None], sides.flatten(-2)], -1), committed_left


def power_marginal(boost: torch.Tensor, reverse: torch.Tensor) -> torch.Tensor:
    """(R, S, 3) from P(reverse) and P(boost | not reverse), each floored, coast otherwise."""
    reverse = floored(reverse, "reverse")
    boost = (1.0 - reverse) * floored(boost, "boost")
    return torch.stack([1.0 - boost - reverse, boost, reverse], -1)


def slow(fleet: Context, parameters: ImugiParameters) -> torch.Tensor:
    """P(boost) for speed, unfloored: high when slow, in corner speeds."""
    return logistic(
        fleet.speed / fleet.corner_speed, parameters.boost_slow_in_corner_speed, fleet.temperature
    )


def far(fleet: Context, distance: torch.Tensor, parameters: ImugiParameters) -> torch.Tensor:
    """P(boost) for distance, unfloored: the store fraction times the distance in gun
    ranges."""
    return logistic(
        fleet.energy_fraction * distance / fleet.gun_range,
        parameters.boost_far_in_store_gun_ranges,
        fleet.temperature,
    )


def either(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    """P(first or second) for independent events: 1 - (1 - first)(1 - second)."""
    return 1.0 - (1.0 - first) * (1.0 - second)


def arc(
    velocity_x: torch.Tensor, velocity_y: torch.Tensor, rate: torch.Tensor, time: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """The displacement over `time` of a ship flying a constant-speed turn at `rate` (rad/s,
    positive left) from `velocity`: (sin(w t) v + (1 - cos(w t)) v_left) / w, the straight
    line's v t as w -> 0."""
    angle = rate * time
    small = angle.abs() < 1e-3
    safe = torch.where(small, 1.0, rate)
    along = torch.where(small, time * (1.0 - angle * angle / 6.0), torch.sin(angle) / safe)
    across = torch.where(small, 0.5 * angle * time, (1.0 - torch.cos(angle)) / safe)
    return (
        along * velocity_x - across * velocity_y,
        along * velocity_y + across * velocity_x,
    )


# A ship's store below this fraction of its most cannot hold more than its sustained turn
# (`turn_now`).
LOW_STORE_FRACTION = 0.1
# Two circles need my turn rate now above the target's by this factor (`maneuvers`).
TWO_CIRCLE_RATE_MARGIN = 1.05
# The turning fight's commitment (`turn_commitment`; the author's rules of 2026-11-16): the
# turn sense is chosen once, at the merge, and held for the fight; the advantaged ship
# commits first, just before a head-on merge. Off (BNB_TURN_COMMIT=0): the sense
# re-decided every tick from the target's current turn (the old behaviour).
TURN_COMMIT = os.environ.get("BNB_TURN_COMMIT", "1") == "1"
# Stick by energy (`maneuvers`): in a committed two-circle fight below the corner speed, the
# sustained stick level (the card's `sustained_stick` lift fraction: the turn that holds
# speed) until back near the corner speed; full stick otherwise. Off
# (BNB_STICK_BY_ENERGY=0): full stick throughout (the old behaviour).
STICK_BY_ENERGY = os.environ.get("BNB_STICK_BY_ENERGY", "1") == "1"
# s: a commitment holds at least this long before the next merge may release it
COMMIT_HOLD_SECONDS = 2.5
# own over target energy (kinetic plus store) above this is an energy advantage
ENERGY_ADVANTAGE_RATIO = 1.1
# The initiative turn's onset: a full-stick turn from straight flight reaches 90% of its rate
# in about this many nose periods (1 / the nose frequency, which scales with the speed up to
# the corner speed): 3.3 / omega_c is 0.30 s at the corner speed on both real-scale ships
# (docs/rewrite/sections/imugi-initiative/turn_onset.py).
TURN_ONSET_NOSE_PERIODS = 3.3
# m: a lateral offset within this is no offset (a dead-on merge: the tie-break)
TIE_OFFSET = 1.0
# below this fraction of the corner speed the two-circle turn eases to the sustained level
EASE_BELOW_CORNER = 0.95


def turn_now(
    characteristics: Characteristics,
    ship_type: torch.Tensor,
    speed: torch.Tensor,
    store_fraction: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """((R, S) the full-stick turn rate (rad/s) and radius (m) a ship of this type has at
    `speed` now, from the card: the g limit's acceleration above the corner speed, the
    lift's (a_max (v / v_corner)^2) below it, over the speed; with the store below
    LOW_STORE_FRACTION, at most its sustained rate (it cannot pay for more)."""
    limit = per_ship(characteristics.max_acceleration, ship_type)
    corner = per_ship(characteristics.corner_speed, ship_type)
    speed = speed.clamp_min(1.0)
    acceleration = limit * (torch.minimum(speed, corner) / corner) ** 2
    rate = acceleration / speed
    sustained = per_ship(characteristics.turn_rate[..., 2], ship_type)
    rate = torch.where(store_fraction < LOW_STORE_FRACTION, torch.minimum(rate, sustained), rate)
    return rate, speed / rate


def offset_approach(
    fleet: Context,
    characteristics: Characteristics,
    target_of,
    has_target: torch.Tensor,
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    aim_x: torch.Tensor,
    aim_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The aim point on a head-on approach (the target seen, its nose within 45 degrees of
    the ship, the ship's within 60 of it, closing): one own turn radius at the current speed
    (`turn_now`) to the side of the target away from the side it is already on, so the two
    pass that far apart; while the range is above one turn diameter and the line of sight
    turns slower than the ship can. Then the offset drops and the ship turns onto its lead
    (the lead turn), before the pass. Elsewhere `aim` unchanged."""
    rate_now, radius_now = turn_now(
        characteristics, fleet.own_type, fleet.speed, fleet.energy_fraction
    )
    distance = torch.sqrt(target_x * target_x + target_y * target_y).clamp_min(TINY)
    seen = has_target & (target_of(fleet.enemy_seen.float()) > 0.5)
    target_vx = target_of(column(fleet.enemy, "velocity_x"))
    target_vy = target_of(column(fleet.enemy, "velocity_y"))
    target_speed = torch.sqrt(target_vx * target_vx + target_vy * target_vy).clamp_min(TINY)
    # the target's nose (track) toward the ship: cos of the angle to the line from it
    toward = -(target_vx * target_x + target_vy * target_y) / (target_speed * distance)
    bearing = signed_angle(fleet.cos, fleet.sin, target_x, target_y)
    relative_x, relative_y = target_vx - fleet.velocity_x, target_vy - fleet.velocity_y
    closing = (relative_x * target_x + relative_y * target_y) < 0.0
    sight_rate = (target_x * relative_y - target_y * relative_x) / (distance * distance)
    approach = (
        seen
        & (toward > math.cos(math.radians(45.0)))
        & (bearing.abs() < math.radians(60.0))
        & closing
        & (distance > 2.0 * radius_now)
        & (sight_rate.abs() < rate_now)
    )
    # the target left of the nose: aim to its right (the two pass left to left), and back
    side = torch.where(bearing >= 0.0, -1.0, 1.0)
    offset_x = target_x - side * radius_now * target_y / distance
    offset_y = target_y + side * radius_now * target_x / distance
    return torch.where(approach, offset_x, aim_x), torch.where(approach, offset_y, aim_y)


def lead(
    fleet: Context,
    characteristics: Characteristics,
    parameters: ImugiParameters,
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    target_velocity_x: torch.Tensor,
    target_velocity_y: torch.Tensor,
    target_rate: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The aim point (interception on the target's turn), relative to the shooter: where the
    target is when a bullet fired now arrives, flying its constant-turn arc at
    `target_rate` (rad/s, positive left; 0 a straight line), less the shooter's own
    velocity times the flight time (the bullet inherits it). The flight time is
    `time_of_flight` at the distance capped at lead_cap_in_gun_range, then once more at the
    predicted point's distance. No field bending."""
    cap = parameters.lead_cap_in_gun_range * fleet.gun_range

    def at(distance: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flight_time = time_of_flight(characteristics, fleet.own_type, torch.minimum(distance, cap))
        moved_x, moved_y = arc(target_velocity_x, target_velocity_y, target_rate, flight_time)
        return (
            target_x + moved_x - fleet.velocity_x * flight_time,
            target_y + moved_y - fleet.velocity_y * flight_time,
        )

    aim_x, aim_y = at(torch.sqrt(target_x * target_x + target_y * target_y))
    return at(torch.sqrt(aim_x * aim_x + aim_y * aim_y))


def path_rate(
    fleet: Context, characteristics: Characteristics, target_of, has_target: torch.Tensor
) -> torch.Tensor:
    """(R, S) the target's flight-path turn rate (rad/s, positive left), from what the team
    sees: the velocity's turn (v x a / |v|^2) where its acceleration is valid, else its
    nose rate while seen, else 0 (a hidden target flies straight); held within its g limit
    at its speed."""
    vx, vy = (
        target_of(column(fleet.enemy, "velocity_x")),
        target_of(column(fleet.enemy, "velocity_y")),
    )
    ax = target_of(column(fleet.enemy, "acceleration_x"))
    ay = target_of(column(fleet.enemy, "acceleration_y"))
    speed_squared = (vx * vx + vy * vy).clamp_min(1.0)
    seen = has_target & (target_of(fleet.enemy_seen.float()) > 0.5)
    valid = seen & (target_of(column(fleet.enemy, "acceleration_valid")) > 0.5)
    rate = torch.where(
        valid,
        (vx * ay - vy * ax) / speed_squared,
        torch.where(seen, target_of(column(fleet.enemy, "nose_rate")), 0.0),
    )
    limit = target_of(per_ship(characteristics.max_acceleration, fleet.enemy_type))
    limit = limit / speed_squared.sqrt().clamp_min(1.0)
    return torch.maximum(torch.minimum(rate, limit), -limit)


class Combat(NamedTuple):
    """The combat layer's result: its marginals; per own ship (R, S): has_target, the
    target's distance (0 without one) and its weight (1 seen, fading when dead reckoned, 0
    without one); the plan choices it made, the target (-1 none) and the turn side; and the
    target's doubt, the extension's phase and the turning fight's commitment (its sense and
    clock) after the decision (`ImugiCarry`)."""

    marginals: Marginals
    has_target: torch.Tensor
    distance: torch.Tensor
    weight: torch.Tensor
    target: torch.Tensor
    left: torch.Tensor
    target_doubt: torch.Tensor
    extend_phase: torch.Tensor
    fight_sense: torch.Tensor
    fight_clock: torch.Tensor


# The committed target (`choose_target`) holds until it is dead, unseen for
# TARGET_LOST_SECONDS, or for TARGET_DOUBT_SECONDS without a break either farther than
# TARGET_FAR_IN_GUN_RANGES own gun ranges or TARGET_SWITCH_RATIO times farther than the best
# known enemy (its distance over its weight): a fight is pressed, not re-scored each tick.
TARGET_DOUBT_SECONDS = 2.0
TARGET_FAR_IN_GUN_RANGES = 3.0
TARGET_SWITCH_RATIO = 2.0
# The extend-and-return rule (`extend_and_return`); off, an energy fighter stays in the turn
# (for ablations).
EXTEND_AND_RETURN = True
# The offset approach and its lead turn (`offset_approach`); off, a head-on approach aims
# at the lead point all the way in (for ablations).
OFFSET_APPROACH = True


def choose_target(
    fleet: Context, offset_x: torch.Tensor, offset_y: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """(the committed target (R, S) int64, -1 without one; the doubt after this decision
    (R, S) s). A fresh ship, or one whose target is dead or unseen for TARGET_LOST_SECONDS,
    takes the best known enemy: the nearest by its distance over its weight (ties: lowest
    row). Otherwise the target holds; the doubt counts the seconds that it has been beyond
    TARGET_FAR_IN_GUN_RANGES or TARGET_SWITCH_RATIO times the best's score, without a
    break, and at TARGET_DOUBT_SECONDS the ship takes the best. Deterministic; without a
    decision key (a re-reading) the doubt holds and only a forced choice changes the
    target."""
    recall = fleet.recall
    known = recall.known[:, None, :]  # (R, 1, E)
    distance = torch.sqrt(offset_x * offset_x + offset_y * offset_y).clamp_min(TINY)
    score = torch.log(distance) - torch.log(recall.weight.clamp_min(1e-30))[:, None, :]
    score = torch.where(known, score, math.inf)  # (R, S, E)
    best = score.argmin(-1)
    any_known = recall.known.any(-1, keepdim=True)
    best = torch.where(any_known, best, -1)
    committed = fleet.carry.target
    held = committed.clamp_min(0)
    # known: alive, and seen, or hidden but pursued (`imugi_memory.remember`: up to
    # TARGET_LOST_SECONDS, and not where it would be seen)
    since = torch.where(fleet.enemy_seen, 0.0, column(fleet.enemy, "time_since_seen"))
    valid = (
        (committed >= 0)
        & recall.known.gather(1, held)
        & (since.gather(1, held) < TARGET_LOST_SECONDS)
    )
    forced = fleet.carry.fresh | ~valid
    held_distance = distance.gather(-1, held[..., None])[..., 0]
    best_score = score.gather(-1, best.clamp_min(0)[..., None])[..., 0]
    far_away = held_distance > TARGET_FAR_IN_GUN_RANGES * fleet.gun_range
    outscored = any_known & (torch.log(held_distance) > best_score + math.log(TARGET_SWITCH_RATIO))
    if fleet.decision_key is None:
        return torch.where(forced, best, committed), fleet.carry.target_doubt
    doubt = torch.where(far_away | outscored, fleet.carry.target_doubt + fleet.seconds, 0.0)
    switch = forced | (doubt >= TARGET_DOUBT_SECONDS)
    target = torch.where(switch, best, committed)
    return target, torch.where(switch, 0.0, doubt)


def extend_and_return(
    fleet: Context,
    characteristics: Characteristics,
    target_of,
    has_target: torch.Tensor,
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    distance: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The one extension, from the card: after a pass (the target behind the beam inside
    the own gun range, flying away from the ship), a ship whose sustained turn rate w (the
    settled full-stick turn) is below its seen target's and whose energy (kinetic plus
    store) is above it flies straight away from the target, boosting, until it is one own
    sustained turn diameter away (2 v / w at its cruise speed v) or 2 / w + 1 s have passed,
    then comes straight back at the same target: no other extension starts for pi / w (a
    reversal) plus that bound. Otherwise it stays in the turn. Returns ((R, S) bool
    extending, the phase after this decision: `ImugiCarry.extend_phase`)."""
    own_rate = per_ship(characteristics.turn_rate[..., 2], fleet.own_type)
    target_rate = target_of(per_ship(characteristics.turn_rate[..., 2], fleet.enemy_type))
    cruise = per_ship(characteristics.cruise_speed, fleet.own_type)
    seen = target_of(fleet.enemy_seen.float()) > 0.5
    target_speed = target_of(column(fleet.enemy, "proper_speed"))
    own_energy = 0.5 * fleet.speed * fleet.speed + column(fleet.own, "energy")
    target_energy = 0.5 * target_speed * target_speed + target_of(column(fleet.enemy, "energy"))
    outturned = has_target & seen & (own_rate < target_rate) & (own_energy > target_energy)
    bearing = signed_angle(fleet.cos, fleet.sin, target_x, target_y)
    # passed: the target behind the beam inside the gun range and flying away from the ship
    # (the two opening after a merge, not the target on the ship's tail)
    opening = (
        target_of(column(fleet.enemy, "velocity_x")) * target_x
        + target_of(column(fleet.enemy, "velocity_y")) * target_y
    ) > 0.0
    passed = (bearing.abs() > 0.5 * math.pi) & (distance < fleet.gun_range) & opening
    bound = 2.0 / own_rate + 1.0
    diameter = 2.0 * cruise / own_rate
    phase = torch.where(fleet.carry.fresh, 0.0, fleet.carry.extend_phase)
    if not EXTEND_AND_RETURN:
        return torch.zeros_like(phase, dtype=torch.bool), torch.zeros_like(phase)
    if fleet.decision_key is None:  # a re-reading: the phase the decision left
        return phase > 0.0, phase
    extending = phase > 0.0
    finished = (phase <= fleet.seconds) | (distance >= diameter) | ~has_target
    after = torch.where(
        extending,
        torch.where(finished, -(math.pi / own_rate + bound), phase - fleet.seconds),
        torch.where(phase < 0.0, (phase + fleet.seconds).clamp_max(0.0), 0.0),
    )
    after = torch.where((phase == 0.0) & outturned & passed, bound, after)
    return after > 0.0, after


def advantage_weight(
    fleet: Context,
    characteristics: Characteristics,
    doctrine: Doctrine,
    target_of,
    has_target: torch.Tensor,
) -> torch.Tensor:
    """The doctrine's weight per own ship (R, S) against its committed target, in [0,
    strength]: its strength times in_advantage at the doctrine's advantage, the product of
    the own-over-target ratios of the sustained turn rate (the settled full-stick turn), the
    climb power, the cruise speed and the gun range, each raised to its exponent. The
    target's type is always observed (`ship_type`), and the card is common knowledge. 0
    without a target. target_of: the committed target's value from per-enemy values (R, E)."""
    log_advantage = torch.zeros_like(fleet.x)
    for values, exponent in (
        (characteristics.turn_rate[..., 2], doctrine.turn_exponent),
        (characteristics.climb_power, doctrine.climb_exponent),
        (characteristics.cruise_speed, doctrine.speed_exponent),
        (characteristics.gun_range, doctrine.range_exponent),
    ):
        own = per_ship(values, fleet.own_type)
        theirs = torch.where(has_target, target_of(per_ship(values, fleet.enemy_type)), own)
        log_advantage = log_advantage + exponent * torch.log(own / theirs)
    weight = doctrine.strength * logistic(
        torch.exp(log_advantage), doctrine.in_advantage, fleet.temperature
    )
    return torch.where(has_target, weight, 0.0)


def rotated(x: torch.Tensor, y: torch.Tensor, angle: torch.Tensor):
    """(x, y) turned by `angle` (positive left)."""
    cos, sin = torch.cos(angle), torch.sin(angle)
    return cos * x - sin * y, sin * x + cos * y


class Doctrinal(NamedTuple):
    """What a doctrine makes of the attack: the aim point and P(boost), P(reverse) on the
    attack, and the weight of extending (flying away, boosting)."""

    aim_x: torch.Tensor
    aim_y: torch.Tensor
    boost: torch.Tensor
    reverse: torch.Tensor
    extending: torch.Tensor
    close: torch.Tensor


def doctrinal(
    fleet: Context,
    characteristics: Characteristics,
    doctrine: Doctrine,
    target_of,
    has_target: torch.Tensor,
    geometry: tuple,
    boost: torch.Tensor,
    reverse: torch.Tensor,
) -> Doctrinal:
    """The doctrine's elements (`bnb.controllers.imugi_doctrine`) on the attack: geometry
    is (target_x, target_y, distance, aim_x, aim_y, aim_error, target velocity x, y)."""
    target_x, target_y, distance, aim_x, aim_y, aim_error, target_vx, target_vy = geometry
    temperature = fleet.temperature
    weight = advantage_weight(fleet, characteristics, doctrine, target_of, has_target)
    safe = distance.clamp_min(TINY)
    in_gun_ranges = distance / fleet.gun_range
    off_nose = aim_error.abs() / fleet.stall_angle
    close = logistic(in_gun_ranges, doctrine.fight_in_gun_range, temperature)

    # Extend: off the nose, or without the energy to engage, fly away while close.
    store = column(fleet.own, "energy")
    own_energy = 0.5 * fleet.speed * fleet.speed + store
    seen = target_of(fleet.enemy_seen.float()) > 0.5
    target_speed = target_of(column(fleet.enemy, "proper_speed"))
    target_energy = 0.5 * target_speed * target_speed + target_of(column(fleet.enemy, "energy"))
    ratio = torch.where(seen, own_energy / target_energy.clamp_min(1.0), 1.0)
    engage = logistic(ratio, doctrine.engage_in_energy_ratio, temperature) * logistic(
        fleet.energy_fraction, doctrine.engage_in_store_fraction, temperature
    )
    turning = logistic(off_nose, doctrine.extend_in_stall_angle, temperature)
    extending = weight * doctrine.extend * close * either(turning, 1.0 - engage)

    # Lag: aim behind the target along its track, the more the farther from its tail.
    target_speed_now = torch.sqrt(target_vx * target_vx + target_vy * target_vy).clamp_min(TINY)
    # cos of the angle between the target's track and the line from it to the ship: -1
    # dead astern
    astern = -(target_vx * target_x + target_vy * target_y) / (target_speed_now * safe)
    behind = doctrine.lag_in_gun_range * fleet.gun_range / target_speed_now
    lag_weight = weight * doctrine.lag * 0.5 * (1.0 + astern)
    aim_x = aim_x + lag_weight * (target_x - behind * target_vx - aim_x)
    aim_y = aim_y + lag_weight * (target_y - behind * target_vy - aim_y)

    # Deny: with the target's nose on the ship, turn off the line to it, to the side the
    # ship is already off its nose.
    target_heading = torch.atan2(target_vy, target_vx)
    off_its_nose = signed_angle(
        torch.cos(target_heading), torch.sin(target_heading), -target_x, -target_y
    )
    target_stall = target_of(per_ship(characteristics.stall_angle, fleet.enemy_type))
    on_its_nose = logistic(
        off_its_nose.abs() / target_stall.clamp_min(TINY), doctrine.deny_in_stall_angle, temperature
    )
    deny_weight = (
        weight
        * doctrine.deny
        * on_its_nose
        * logistic(in_gun_ranges, doctrine.deny_in_gun_range, temperature)
    )
    # the ship is off the target's nose to the target's left when off_its_nose > 0: seen
    # from the ship, that side is its right, so it turns right (negative)
    side = torch.where(off_its_nose >= 0.0, -1.0, 1.0)
    denied_x, denied_y = rotated(aim_x, aim_y, side * doctrine.deny_angle)
    aim_x = aim_x + deny_weight * (denied_x - aim_x)
    aim_y = aim_y + deny_weight * (denied_y - aim_y)
    # extending overrides the attack's aim: away from the target
    away = torch.sqrt(aim_x * aim_x + aim_y * aim_y) / safe  # aim length over distance
    aim_x = aim_x + extending * (-target_x * away - aim_x)
    aim_y = aim_y + extending * (-target_y * away - aim_y)

    # Slash: boost into the attack, never reverse; bleed: reverse to the corner speed in a
    # turning fight.
    slashing = weight * doctrine.slash
    boost = either(
        boost, slashing * logistic(in_gun_ranges, doctrine.slash_in_gun_range, temperature)
    )
    bleeding = (
        weight
        * doctrine.bleed
        * close
        * logistic(off_nose, doctrine.bleed_in_stall_angle, temperature)
        * logistic(fleet.speed / fleet.corner_speed, doctrine.bleed_in_corner_speed, temperature)
    )
    reverse = either((1.0 - slashing) * reverse, bleeding)
    return Doctrinal(aim_x, aim_y, boost, reverse, extending, close)


def commanded_stick(left: torch.Tensor, fraction) -> torch.Tensor:
    """(R, S, 7) the stick distribution flying `fraction` of full stick (in [0, 1], between
    the two nearest levels of STICK_MAGNITUDES, linearly) to the left where `left`, else to
    the right."""
    # A number is filled on the device (a host-to-device copy would break CUDA graph capture)
    if isinstance(fraction, torch.Tensor):
        fraction = fraction.clamp(0.0, 1.0).expand(left.shape)
    else:
        fraction = torch.full(left.shape, min(max(float(fraction), 0.0), 1.0), device=left.device)
    hats = []
    for index, magnitude in enumerate(STICK_MAGNITUDES):
        below = STICK_MAGNITUDES[index - 1] if index > 0 else None
        above = STICK_MAGNITUDES[index + 1] if index + 1 < len(STICK_MAGNITUDES) else None
        rising = (
            (fraction - below) / (magnitude - below)
            if below is not None
            else torch.ones_like(fraction)
        )
        falling = (
            (above - fraction) / (above - magnitude)
            if above is not None
            else torch.ones_like(fraction)
        )
        hats.append(torch.where(fraction <= magnitude, rising, falling).clamp(0.0, 1.0))
    on_left = left.float()
    parts = [hats[0]]
    for hat in hats[1:]:
        parts += [(1.0 - on_left) * hat, on_left * hat]
    return torch.stack(parts, -1)


class Commitment(NamedTuple):
    """The turning fight's commitment per own ship (R, S): the sense (+1 left, -1 right, 0
    none), the seconds since it was made, and where the committed turn is flown now (the
    target behind the beam, or the initiative turn just before the merge)."""

    sense: torch.Tensor
    clock: torch.Tensor
    flown: torch.Tensor


def turn_onset(
    fleet: Context, characteristics: Characteristics, speed: torch.Tensor
) -> torch.Tensor:
    """(R, S) s: the time a full-stick turn from straight flight takes to reach its full
    rate at `speed`: TURN_ONSET_NOSE_PERIODS over the nose frequency (the card's at the
    corner speed, `Characteristics.nose_frequency`, scaled by the speed up to it; at least
    the rules' floor of 1 rad/s)."""
    frequency = per_ship(characteristics.nose_frequency, fleet.own_type)
    scale = (speed / fleet.corner_speed).clamp(max=1.0)
    return TURN_ONSET_NOSE_PERIODS / (frequency * scale).clamp_min(1.0)


def tie_side(fleet: Context, lateral: torch.Tensor) -> torch.Tensor:
    """(R, S) +1 left or -1 right: the lateral offset's side (positive left), and where it
    is within TIE_OFFSET (a dead-on merge) toward the map centre, and where that too is
    exactly ambiguous (the ship at the centre or flying through it), left. Deterministic."""
    centre = fleet.cos * -fleet.y - fleet.sin * -fleet.x  # the centre's side, positive left
    side = torch.where(centre < 0.0, -1.0, 1.0)
    return torch.where(lateral.abs() > TIE_OFFSET, torch.sign(lateral), side)


def turn_commitment(
    fleet: Context,
    characteristics: Characteristics,
    target_of,
    has_target: torch.Tensor,
    geometry: tuple,
    released: tuple,
    advantage: tuple,
) -> Commitment:
    """The turning fight's committed sense (the author's rules of 2026-11-16).

    Release (`released`: (released, blocked), from the caller): no target, a new target,
    the target unseen, an extension or the boundary steer (all but the new target also
    block a commitment this decision), or the next merge (a head-on approach) once the
    commitment has held COMMIT_HOLD_SECONDS. Commit (an uncommitted ship, once):
    - initiative: the advantaged ship (`advantage`: its full-stick turn rate now above the
      target's by TWO_CIRCLE_RATE_MARGIN, or its energy above by ENERGY_ADVANTAGE_RATIO),
      on a head-on approach, at the latest tick that still has its turn at full rate at the
      crossing: the time to the closest approach within its turn onset (`turn_onset`), to
      the sense that closes on the target's tail (`tie_side` of the tail point's offset);
    - at the merge (the target behind the beam, inside four gun ranges), once: where one
      ship of the pair holds an advantage and the target is turning, to the card's choice
      against the target's observed sense (`maneuvers`: two circles where the own rate now
      is clearly the higher, else one circle), so the outmatched ship reacts to the
      initiative turn; an even pair, or a target not turning, by the geometry (the tail
      point's side).
    Without a decision key (a re-reading) the carry's commitment holds unchanged."""
    target_x, target_y, distance = geometry[:3]
    target_vx, target_vy = geometry[6], geometry[7]
    released, blocked = released
    advantaged, outmatched, target_rate, one_circle = advantage
    carry = fleet.carry
    if fleet.decision_key is None:
        sense = torch.where(carry.fresh, 0.0, carry.fight_sense)
        clock = carry.fight_clock
    else:
        # a dead ship is fresh at its next observation: none for it either
        dead = column(fleet.own, "alive") < 0.5
        sense = torch.where(carry.fresh | released | dead, 0.0, carry.fight_sense)
        blocked = blocked | dead
        clock = carry.fight_clock + fleet.seconds
    seen = has_target & (target_of(fleet.enemy_seen.float()) > 0.5)
    bearing = signed_angle(fleet.cos, fleet.sin, target_x, target_y)
    target_speed = torch.sqrt(target_vx * target_vx + target_vy * target_vy).clamp_min(TINY)
    # head-on: the tracks within 60 degrees of opposite (the line of sight swings wide
    # close to an offset pass, the tracks do not)
    opposed = -(target_vx * fleet.velocity_x + target_vy * fleet.velocity_y) / (
        target_speed * fleet.speed.clamp_min(TINY)
    )
    relative_x, relative_y = target_vx - fleet.velocity_x, target_vy - fleet.velocity_y
    closing_rate = relative_x * target_x + relative_y * target_y  # negative closing
    approaching = (
        seen
        & (closing_rate < 0.0)
        & (bearing.abs() < 0.5 * math.pi)
        & (opposed > math.cos(math.radians(60.0)))
    )
    relative_squared = (relative_x * relative_x + relative_y * relative_y).clamp_min(TINY)
    crossing = (-closing_rate / relative_squared).clamp_min(0.0)  # s to the closest approach
    onset = turn_onset(fleet, characteristics, fleet.speed)
    # the latest tick: the crossing within the onset (rounded to the decision's length)
    initiative_window = approaching & (crossing <= onset + 0.5 * fleet.seconds)
    # the tail point: one own turn radius behind the target along its track
    _, radius_now = turn_now(characteristics, fleet.own_type, fleet.speed, fleet.energy_fraction)
    tail_x = target_x - radius_now * target_vx / target_speed
    tail_y = target_y - radius_now * target_vy / target_speed
    lateral = fleet.cos * tail_y - fleet.sin * tail_x  # positive: the tail on the left
    geometric = tie_side(fleet, lateral)
    abeam = seen & (bearing.abs() > 0.5 * math.pi)
    in_range = distance < 4.0 * fleet.gun_range
    if fleet.decision_key is not None:
        held_long = clock >= COMMIT_HOLD_SECONDS
        sense = torch.where(approaching & held_long, 0.0, sense)
        free = sense == 0.0
        initiative = free & advantaged & initiative_window & ~blocked
        turning = target_rate.abs() > TURNING_RATE
        reaction = torch.where((target_rate > 0.0) ^ one_circle, 1.0, -1.0)
        at_merge = free & ~initiative & abeam & in_range & ~blocked
        merged = torch.where((advantaged | outmatched) & turning, reaction, geometric)
        new = torch.where(initiative, geometric, merged)
        committing = initiative | at_merge
        sense = torch.where(committing, new, sense)
        clock = torch.where(committing, 0.0, clock)
    committed = sense != 0.0
    flown = committed & (abeam | (initiative_window & advantaged))
    return Commitment(sense, clock, flown)


def maneuvers(
    fleet: Context,
    characteristics: Characteristics,
    doctrine: Doctrine,
    target_of,
    has_target: torch.Tensor,
    geometry: tuple,
    stick: torch.Tensor,
    left: torch.Tensor,
    power: torch.Tensor,
    others: tuple,
    released: tuple | None = None,
):
    """The doctrine's maneuver (`bnb.controllers.imugi_doctrine`: the sense, the duel
    extension and the crowd extension) on the combat layer's stick, committed side and
    power. others: the other known enemies' distances in own gun ranges (R, S, E; inf
    elsewhere), the angles between their tracks and the lines from them to the ship (R, S,
    E), and the own ships' distances in own gun ranges (R, S, S; inf for the ship itself and
    the dead). released: ((R, S) bool the turning fight's commitment released this
    decision, (R, S) bool none made this decision) (`turn_commitment`; None: neither).
    Returns (stick, left, power, commitment)."""
    target_x, target_y, distance = geometry[:3]
    temperature = fleet.temperature
    boost = power_marginal(torch.ones_like(distance), torch.zeros_like(distance))
    bearing = signed_angle(fleet.cos, fleet.sin, target_x, target_y)

    # Sense: after the merge, the turn's rotation sense against the target's.
    target_rate = target_of(column(fleet.enemy, "nose_rate"))
    seen = target_of(fleet.enemy_seen.float()) > 0.5
    lift = characteristics.max_acceleration / (
        characteristics.corner_speed * characteristics.corner_speed
    )  # the maximum lift coefficient by type
    lift_ratio = per_ship(lift, fleet.own_type) / target_of(
        per_ship(lift, fleet.enemy_type)
    ).clamp_min(TINY)
    same_type = fleet.own_type == target_of(fleet.enemy_type.float()).round().to(torch.int32)
    geometry_gate = torch.where(
        has_target & seen & (bearing.abs() > 0.5 * math.pi) & (target_rate.abs() > TURNING_RATE),
        logistic(distance / fleet.gun_range, doctrine.sense_in_gun_range, temperature),
        0.0,
    )
    if BOUNDARY_OVER_MANEUVERS:  # the edge steer outranks the maneuvers' forced turn sense
        geometry_gate = geometry_gate * (1.0 - fleet.boundary_weight)
    # the choice, from the card and the state now (`turn_now`): two circles (the target's
    # own rotation sense) only where my turn rate at my speed is clearly the higher
    # (TWO_CIRCLE_RATE_MARGIN), else one circle (the other sense), where the smaller radius
    # wins and a mirror does not settle into an even two-circle fight
    own_rate_now, _ = turn_now(characteristics, fleet.own_type, fleet.speed, fleet.energy_fraction)
    target_type = target_of(fleet.enemy_type.float()).round().to(torch.int32)
    target_rate_now, _ = turn_now(
        characteristics,
        target_type,
        target_of(column(fleet.enemy, "proper_speed")),
        target_of(column(fleet.enemy, "energy"))
        / per_ship(characteristics.max_energy, target_type),
    )
    one_circle = own_rate_now <= TWO_CIRCLE_RATE_MARGIN * target_rate_now
    sense_left = (target_rate > 0.0) ^ one_circle
    # the situation: own over target specific energy, shield and maximum lift coefficient
    target_speed_now = target_of(column(fleet.enemy, "proper_speed"))
    own_energy = 0.5 * fleet.speed * fleet.speed + column(fleet.own, "energy")
    target_energy = 0.5 * target_speed_now * target_speed_now + target_of(
        column(fleet.enemy, "energy")
    )
    energy_ratio = torch.where(seen, own_energy / target_energy.clamp_min(1.0), 1.0)
    commitment = Commitment(
        torch.zeros_like(distance), torch.zeros_like(distance), torch.zeros_like(seen)
    )
    if TURN_COMMIT:
        if released is None:
            released = (torch.zeros_like(seen), torch.zeros_like(seen))
        blocked = released[1] | ~seen | (fleet.boundary_weight > 0.5)
        released = (released[0] | blocked, blocked)
        advantaged = (own_rate_now > TWO_CIRCLE_RATE_MARGIN * target_rate_now) | (
            energy_ratio > ENERGY_ADVANTAGE_RATIO
        )
        outmatched = (target_rate_now > TWO_CIRCLE_RATE_MARGIN * own_rate_now) | (
            energy_ratio * ENERGY_ADVANTAGE_RATIO < 1.0
        )
        commitment = turn_commitment(
            fleet,
            characteristics,
            target_of,
            has_target,
            geometry,
            released,
            (advantaged, outmatched, target_rate, one_circle),
        )
        sense_left = commitment.sense > 0.0
        geometry_gate = torch.where(
            commitment.flown,
            logistic(distance / fleet.gun_range, doctrine.sense_in_gun_range, temperature),
            0.0,
        )
        if BOUNDARY_OVER_MANEUVERS:
            geometry_gate = geometry_gate * (1.0 - fleet.boundary_weight)
    # the stick by energy: a two-circle turn (the target's rotation sense) below the corner
    # speed eases to the sustained level, the turn that holds speed; else full stick
    two_circle = (sense_left == (target_rate > 0.0)) & (target_rate.abs() > TURNING_RATE)
    easing = two_circle & (fleet.speed < EASE_BELOW_CORNER * fleet.corner_speed)
    if STICK_BY_ENERGY:
        level = torch.where(easing, STICK_MAGNITUDES[2], 1.0)
    else:
        level = 1.0
    # alone: no other threatening known enemy near (a soft count of them inside the crowd's
    # radius, their tracks toward the ship), less the weighted count of own ships near
    enemies, tracks, allies = others
    radius = doctrine.sense_crowd_in_gun_range
    per_enemy = Logistic(radius.centre[..., None], radius.scale[..., None], radius.rising)
    threat = doctrine.sense_crowd_in_nose_angle
    per_track = Logistic(threat.centre[..., None], threat.scale[..., None], threat.rising)
    heat = temperature[..., None] if isinstance(temperature, torch.Tensor) else temperature
    near = logistic(enemies, per_enemy, heat) * logistic(tracks, per_track, heat)
    crowd = torch.where(torch.isfinite(enemies), near, 0.0).sum(-1)
    friends = torch.where(torch.isfinite(allies), logistic(allies, per_enemy, heat), 0.0).sum(-1)
    crowd = crowd - doctrine.sense_crowd_ally_weight * friends
    alone = logistic(crowd, doctrine.sense_in_crowd, temperature)
    shield_ratio = torch.where(
        seen,
        column(fleet.own, "shield").clamp_min(TINY)
        / target_of(column(fleet.enemy, "shield")).clamp_min(TINY),
        1.0,
    )
    stay_out = logistic(energy_ratio, doctrine.crowd_sense_in_energy_ratio, temperature) * logistic(
        shield_ratio, doctrine.crowd_sense_in_shield_ratio, temperature
    )
    behind = either(
        either(
            logistic(energy_ratio, doctrine.duel_extend_in_energy_ratio, temperature),
            logistic(shield_ratio, doctrine.duel_extend_in_shield_ratio, temperature),
        ),
        logistic(lift_ratio, doctrine.duel_extend_in_lift_ratio, temperature),
    )
    anywhere = torch.isfinite(enemies).float().sum(-1)  # the other known enemies
    cross = torch.where(
        doctrine.duel_extend_mirror_only <= 0.5,
        logistic(anywhere, doctrine.duel_extend_cross_in_others, temperature),
        0.0,
    )
    duel_extending = (
        doctrine.duel_extend * geometry_gate * alone * behind * torch.where(same_type, 1.0, cross)
    )
    sensing = (
        doctrine.sense
        * geometry_gate
        * (alone * (1.0 - duel_extending) + (1.0 - alone) * (1.0 - stay_out))
    )
    in_crowd = doctrine.crowd_sense * geometry_gate * (1.0 - alone) * stay_out + duel_extending
    stick = stick + sensing[..., None] * (commanded_stick(sense_left, level) - stick)
    left = torch.where(sensing > 0.5, sense_left, left)
    stick = stick + in_crowd[..., None] * (
        commanded_stick(sense_left, doctrine.crowd_sense_stick) - stick
    )
    power = power + (in_crowd * doctrine.crowd_sense_boost)[..., None] * (boost - power)
    return stick, left, power, commitment


def combat(
    fleet: Context,
    characteristics: Characteristics,
    parameters: ImugiParameters,
    doctrines: torch.Tensor | None = None,
) -> Combat:
    """The combat layer. The target is the committed one (`choose_target`), aimed at where
    the agent believes it is; P(shoot) is scaled by its weight, so a dead-reckoned target
    draws fewer shots the staler it is. doctrines: a doctrine table per type (T, D) or per
    view and type (R, T, D) (`bnb.controllers.imugi_doctrine`), each own ship flying its
    type's; None flies none (the agent without doctrines)."""
    recall = fleet.recall
    offset_x = recall.x[:, None, :] - fleet.x[..., None]  # (R, S, E)
    offset_y = recall.y[:, None, :] - fleet.y[..., None]
    chosen, doubt = choose_target(fleet, offset_x, offset_y)
    has_target = chosen >= 0
    pick = torch.nn.functional.one_hot(chosen.clamp_min(0), offset_x.shape[-1]).bool()
    pick = pick & has_target[..., None]

    def target(values: torch.Tensor) -> torch.Tensor:
        """The target's value from per-pair (R, S, E) or per-enemy (R, E) values."""
        values = values if values.dim() == 3 else values[:, None, :]
        return torch.where(pick, values, 0.0).sum(-1)

    target_x, target_y = target(offset_x), target(offset_y)
    weight = target(recall.weight)
    distance = torch.sqrt(target_x * target_x + target_y * target_y)
    aim_x, aim_y = lead(
        fleet,
        characteristics,
        parameters,
        target_x,
        target_y,
        target(recall.velocity_x),
        target(recall.velocity_y),
        path_rate(fleet, characteristics, target, has_target),
    )
    lead_x, lead_y = aim_x, aim_y
    aim_error = signed_angle(fleet.cos, fleet.sin, aim_x, aim_y)  # the gun's, to the lead
    if OFFSET_APPROACH:
        aim_x, aim_y = offset_approach(
            fleet, characteristics, target, has_target, target_x, target_y, aim_x, aim_y
        )
    boost = either(
        slow(fleet, parameters), torch.where(has_target, far(fleet, distance, parameters), 0.0)
    )
    reverse = torch.where(
        has_target,
        logistic(distance / fleet.gun_range, parameters.reverse_in_gun_range, fleet.temperature),
        0.0,
    )
    power, doctrine, geometry, shaped = None, None, None, None
    if doctrines is not None:  # the own types' doctrines shape the aim point and the power
        geometry = (
            target_x,
            target_y,
            distance,
            aim_x,
            aim_y,
            aim_error,
            target(recall.velocity_x),
            target(recall.velocity_y),
        )
        doctrine = doctrine_of(doctrines, fleet.own_type)
        shaped = doctrinal(
            fleet, characteristics, doctrine, target, has_target, geometry, boost, reverse
        )
        aim_x, aim_y = shaped.aim_x, shaped.aim_y
        attack = power_marginal(shaped.boost, shaped.reverse)
        extend = power_marginal(
            either(slow(fleet, parameters), shaped.close), torch.zeros_like(boost)
        )
        power = attack + shaped.extending[..., None] * (extend - attack)

    # The extend-and-return rule: away from the target, boosting.
    extending, extend_phase = extend_and_return(
        fleet, characteristics, target, has_target, target_x, target_y, distance
    )
    aim_x = torch.where(extending, -target_x, aim_x)
    aim_y = torch.where(extending, -target_y, aim_y)
    if power is None:
        power = power_marginal(boost, reverse)
    power = torch.where(
        extending[..., None],
        power_marginal(torch.ones_like(boost), torch.zeros_like(boost)),
        power,
    )

    # Without a target: the search (`SEARCH`).
    idle_x, idle_y, may_hold = idle_direction(fleet)
    toward_centre = fleet.distance_from_centre > (
        parameters.centre_hold_in_arena_radius * fleet.boundary_radius
    )
    if not may_hold:
        toward_centre = torch.ones_like(toward_centre)
    wanted_x, wanted_y = steer(
        fleet, torch.where(has_target, aim_x, idle_x), torch.where(has_target, aim_y, idle_y)
    )
    steered, left = stick_marginal(
        fleet,
        signed_angle(fleet.cos, fleet.sin, wanted_x, wanted_y),
        parameters,
        fleet.carry.combat_left,
    )
    hold = (~has_target & ~toward_centre)[..., None]
    held, _ = stick_marginal(fleet, torch.zeros_like(fleet.x), parameters)
    stick = torch.where(hold, held, steered)
    fight_sense, fight_clock = torch.zeros_like(distance), torch.zeros_like(distance)
    if doctrine is not None:
        # the other known enemies' distances in own gun ranges (inf for the target and the
        # unknown): the sense's crowd
        apart = torch.sqrt(offset_x * offset_x + offset_y * offset_y)
        others = apart / fleet.gun_range[..., None]
        others = torch.where(recall.known[:, None, :] & ~pick, others, math.inf)
        # the angle between each enemy's track and the line from it to the ship
        track_speed = torch.sqrt(
            recall.velocity_x * recall.velocity_x + recall.velocity_y * recall.velocity_y
        ).clamp_min(TINY)[:, None, :]
        toward = -(
            recall.velocity_x[:, None, :] * offset_x + recall.velocity_y[:, None, :] * offset_y
        ) / (track_speed * apart.clamp_min(TINY))
        tracks = torch.acos(toward.clamp(-1.0, 1.0)).clamp_min(TINY)
        # the own ships' distances (none to itself or the dead)
        own_alive = column(fleet.own, "alive") > 0.5
        ally_x = fleet.x[:, None, :] - fleet.x[..., None]
        ally_y = fleet.y[:, None, :] - fleet.y[..., None]
        allies = torch.sqrt(ally_x * ally_x + ally_y * ally_y) / fleet.gun_range[..., None]
        itself = torch.eye(allies.shape[-1], dtype=torch.bool, device=allies.device)
        allies = torch.where(own_alive[:, None, :] & ~itself, allies, math.inf)
        # the turning fight's commitment ends with the target, its change, or an extension
        # (and none is made without a target or while extending)
        blocked = ~has_target | extending | (shaped.extending > 0.5)
        released = (blocked | (chosen != fleet.carry.target), blocked)
        stick, left, power, commitment = maneuvers(
            fleet,
            characteristics,
            doctrine,
            target,
            has_target,
            geometry,
            stick,
            left,
            power,
            (others, tracks, allies),
            released,
        )
        fight_sense, fight_clock = commitment.sense, commitment.clock

    hit_radius = target(per_ship(characteristics.hit_radius, fleet.enemy_type))
    # the predicted miss: how far the nose's line passes from the lead point, in the
    # target's hit radii (the lead point's whole distance when it is behind the beam)
    reach = torch.sqrt(lead_x * lead_x + lead_y * lead_y)
    miss = torch.where(aim_error.abs() < 0.5 * math.pi, reach * torch.sin(aim_error.abs()), reach)
    aligned = logistic(
        miss / hit_radius.clamp_min(TINY), parameters.aim_in_target_size, fleet.temperature
    )
    in_range = logistic(
        distance / fleet.gun_range, parameters.shoot_in_gun_range, fleet.temperature
    )
    cost = per_ship(characteristics.bullet_energy_cost, fleet.own_type)
    ready = (column(fleet.own, "gun_cooldown") <= 0.0) & (column(fleet.own, "energy") >= cost)
    shoot = floored(torch.where(has_target & ready, weight * aligned * in_range, 0.0), "shoot")
    shoot = torch.stack([1.0 - shoot, shoot], -1)
    return Combat(
        Marginals(power, stick, shoot),
        has_target,
        distance,
        weight,
        chosen,
        left,
        doubt,
        extend_phase,
        fight_sense,
        fight_clock,
    )


def team_centre(fleet: Context) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(R, 1) x, y of the own living ships' centre of mass, and (R, 1) whether any lives."""
    alive = (column(fleet.own, "alive") > 0.5).float()
    count = alive.sum(-1, keepdim=True)
    centre_x = (fleet.x * alive).sum(-1, keepdim=True) / count.clamp_min(1.0)
    centre_y = (fleet.y * alive).sum(-1, keepdim=True) / count.clamp_min(1.0)
    return centre_x, centre_y, count > 0.0


def unit(x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The direction (x, y) scaled to unit length (a zero vector stays near zero)."""
    length = torch.sqrt(x * x + y * y).clamp_min(TINY)
    return x / length, y / length


def idle_direction(fleet: Context) -> tuple[torch.Tensor, torch.Tensor, bool]:
    """(R, S) the direction a ship without a target wants, by `SEARCH`, and whether it may
    hold its course near the centre (only "centre" does)."""
    if SEARCH == "centre":
        return -fleet.x, -fleet.y, True
    if SEARCH == "straight":
        moving = fleet.speed > 1.0
        return (
            torch.where(moving, fleet.velocity_x, fleet.cos),
            torch.where(moving, fleet.velocity_y, fleet.sin),
            False,
        )
    if SEARCH == "explore":
        target_x, target_y = explore_target(fleet)
        goal_x, goal_y = unit(target_x - fleet.x, target_y - fleet.y)
        centre_x, centre_y, _ = team_centre(fleet)
        apart_x, apart_y = centre_x - fleet.x, centre_y - fleet.y
        apart = torch.sqrt(apart_x * apart_x + apart_y * apart_y) / fleet.gun_range
        low, high = COHESION_IN_GUN_RANGE
        pull = ((apart - low) / (high - low)).clamp(0.0, 1.0)
        toward_x, toward_y = unit(apart_x, apart_y)
        wanted_x = (1.0 - pull) * goal_x + pull * toward_x
        wanted_y = (1.0 - pull) * goal_y + pull * toward_y
        # separation: away from each living ally within half a gun range, more the nearer
        alive = column(fleet.own, "alive") > 0.5
        away_x = fleet.x[..., None] - fleet.x[:, None, :]  # (R, S, S): from ally to ship
        away_y = fleet.y[..., None] - fleet.y[:, None, :]
        gap = torch.sqrt(away_x * away_x + away_y * away_y).clamp_min(TINY)
        reach = SEPARATION_IN_GUN_RANGE * fleet.gun_range[..., None]
        itself = torch.eye(gap.shape[-1], dtype=torch.bool, device=gap.device)
        push = torch.where(alive[:, None, :] & ~itself, (1.0 - gap / reach).clamp_min(0.0), 0.0)
        push_x = (push * away_x / gap).sum(-1)
        push_y = (push * away_y / gap).sum(-1)
        return wanted_x + push_x, wanted_y + push_y, False
    raise ValueError(f"unknown search {SEARCH!r}; one of {SEARCH_MODES}")


def explore_target(fleet: Context) -> tuple[torch.Tensor, torch.Tensor]:
    """(R, 1) x, y of the team's explore target: the nearest unseen cell inside the
    boundary to its centre of mass (distances shortened for cells ahead of the team's mean
    velocity), or the least recently seen one inside if none is unseen, or the centre if
    no cell is inside."""
    cells = explore_cells(fleet.x.device)  # (C, 2) in arena radii
    cell_x = cells[:, 0] * fleet.arena_radius  # (R, C)
    cell_y = cells[:, 1] * fleet.arena_radius
    seen = fleet.carry.seen
    inside = cell_x * cell_x + cell_y * cell_y < (EXPLORE_INSIDE * fleet.boundary_radius) ** 2
    centre_x, centre_y, _ = team_centre(fleet)
    off_x, off_y = cell_x - centre_x, cell_y - centre_y
    distance = torch.sqrt(off_x * off_x + off_y * off_y).clamp_min(TINY)
    alive = (column(fleet.own, "alive") > 0.5).float()
    count = alive.sum(-1, keepdim=True).clamp_min(1.0)
    heading_x = (fleet.velocity_x * alive).sum(-1, keepdim=True) / count
    heading_y = (fleet.velocity_y * alive).sum(-1, keepdim=True) / count
    heading = torch.sqrt(heading_x * heading_x + heading_y * heading_y).clamp_min(TINY)
    ahead = (off_x * heading_x + off_y * heading_y) / (distance * heading)  # cos, (R, C)
    cost = distance * (1.0 - EXPLORE_AHEAD * ahead)
    unseen = inside & (seen < EXPLORE_UNSEEN)
    nearest = torch.where(unseen, cost, math.inf).argmin(-1)
    stalest = torch.where(inside, seen, math.inf).argmin(-1)
    pick = torch.where(unseen.any(-1), nearest, stalest)[:, None]
    target_x = cell_x.gather(-1, pick)
    target_y = cell_y.gather(-1, pick)
    anywhere = inside.any(-1, keepdim=True)
    return torch.where(anywhere, target_x, 0.0), torch.where(anywhere, target_y, 0.0)


def blend(first: torch.Tensor, second: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The log-probability blend first^w second^(1 - w), renormalized; where the two have
    no common support, their mixture. weight (R, S) in [0, 1]."""
    weight = weight[..., None]
    geometric = first.pow(weight) * second.pow(1.0 - weight)
    total = geometric.sum(-1, keepdim=True)
    return torch.where(
        total > 0.0, geometric / total.clamp_min(1e-30), weight * first + (1.0 - weight) * second
    )
