"""The scripted agent's combat layer (scripted spec A4) and the geometry both layers share.

Everything is per own ship, (R, S), from one team's observation: the own rows are the first
S, the enemy rows the rest. An enemy row is read only where it is visible and alive, so a
hidden row's zeroed columns never matter; the target is chosen from the enemies the agent
recalls (`scripted_memory`), seen or dead reckoned.
"""

import math
from typing import NamedTuple

import torch

from bnb.contracts import Obs
from bnb.contracts.obs import GLOBAL, SHIP
from bnb.game.handling.characteristics import Characteristics, time_of_flight
from bnb.game.rules import Rules
from bnb.game.ship_types import per_ship

from .scripted_doctrine import Doctrine
from .scripted_doctrine import per_ship as doctrine_of
from .scripted_memory import (
    Recall,
    ScriptedCarry,
    commit,
    fresh_carry,
    remember,
    time_unit,
    ways_round,
)
from .scripted_parameters import ScriptedParameters, floored, in_own_time, logistic, ramp

TARGET_SIZE_RANGE = (math.radians(1.0), math.radians(45.0))  # the clamp on its angular size
COAST, BOOST, REVERSE = 0, 1, 2  # power indices (bnb.game.rules.POWER_MODES)
TINY = 1e-6  # m: keeps a direction's length away from zero
# The maneuvers' stalemate clock (`maneuvers`): it runs while the ship's committed target is
# within this many own gun ranges and no angle is gained, an angle gained being the nose
# within GAINED_NOSE stall angles of the aim point while the target's nose is more than
# GAINED_OFF_ITS_NOSE of its stall angles off the ship (an offensive shot, not a joust).
STALEMATE_IN_GUN_RANGES = 4.0
GAINED_NOSE_IN_STALL_ANGLES = 1.0
GAINED_OFF_ITS_NOSE_IN_STALL_ANGLES = 2.0
STICK_MAGNITUDES = (0.0, 0.4, 0.75, 1.0)  # bnb.game.rules.STICK_LEVELS by level


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
    distance_from_centre: torch.Tensor
    boundary_weight: torch.Tensor  # the steer toward the centre, in [0, 1]
    temperature: torch.Tensor | float  # the agent's: a float, or (R, 1) per view
    carry: ScriptedCarry  # the memory, before this decision's plan choices
    recall: Recall  # the enemies as the agent believes them
    seconds: float  # a decision's length
    time_unit: torch.Tensor  # the ship's own time unit in seconds (`time_unit`)
    switching: bool  # the plan choices may switch (else only a forced re-choice)


def column(rows: torch.Tensor, name: str) -> torch.Tensor:
    """The named SHIP column of ship rows (..., columns), shaped (...)."""
    return rows[..., SHIP[name]]


def context(
    obs: Obs,
    characteristics: Characteristics,
    parameters: ScriptedParameters,
    temperature: torch.Tensor | float = 1.0,
    carry: ScriptedCarry | None = None,
    recall: Recall | None = None,
    seconds: float = Rules.tick_length,
    switching: bool = False,
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
    # The boundary: within a margin of turn radii of the edge and flying outward.
    distance_from_centre = torch.sqrt(x * x + y * y).clamp_min(TINY)
    turn_radius = speed * speed / per_ship(characteristics.max_acceleration, own_type)
    margin = (parameters.boundary_margin_in_turn_radius * turn_radius).clamp_min(1.0)
    depth = (distance_from_centre - (arena_radius - margin)) / margin
    outward = x * velocity_x + y * velocity_y > 0.0
    boundary_weight = torch.where(
        outward, ramp(depth, parameters.boundary_steer_ramp_in_margin), 0.0
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
        distance_from_centre=distance_from_centre,
        boundary_weight=boundary_weight,
        temperature=temperature,
        carry=carry,
        recall=recall,
        seconds=seconds,
        time_unit=time_unit(characteristics, own_type, parameters),
        switching=switching,
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
    parameters: ScriptedParameters,
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
        fleet.switching,
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


def slow(fleet: Context, parameters: ScriptedParameters) -> torch.Tensor:
    """P(boost) for speed, unfloored: high when slow, in corner speeds."""
    return logistic(
        fleet.speed / fleet.corner_speed, parameters.boost_slow_in_corner_speed, fleet.temperature
    )


def far(fleet: Context, distance: torch.Tensor, parameters: ScriptedParameters) -> torch.Tensor:
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


def lead(
    fleet: Context,
    characteristics: Characteristics,
    parameters: ScriptedParameters,
    target_x: torch.Tensor,
    target_y: torch.Tensor,
    target_velocity_x: torch.Tensor,
    target_velocity_y: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The aim point (first-order interception), relative to the shooter: the target's
    displacement when a bullet fired now arrives, `t = time_of_flight(distance)` with the
    distance capped at lead_cap_in_gun_range. The bullet inherits the shooter's velocity, so
    the relative velocity carries the target. No field bending."""
    distance = torch.sqrt(target_x * target_x + target_y * target_y)
    cap = parameters.lead_cap_in_gun_range * fleet.gun_range
    flight_time = time_of_flight(characteristics, fleet.own_type, torch.minimum(distance, cap))
    return (
        target_x + (target_velocity_x - fleet.velocity_x) * flight_time,
        target_y + (target_velocity_y - fleet.velocity_y) * flight_time,
    )


class Combat(NamedTuple):
    """The combat layer's result: its marginals; per own ship (R, S): has_target, the
    target's distance (0 without one) and its weight (1 seen, fading when dead reckoned, 0
    without one); the plan choices it made, the target (-1 none) and the turn side; and the
    maneuvers' clocks after the decision (`ScriptedCarry.stalemate_seconds`,
    `yoyo_seconds`)."""

    marginals: Marginals
    has_target: torch.Tensor
    distance: torch.Tensor
    weight: torch.Tensor
    target: torch.Tensor
    left: torch.Tensor
    stalemate_seconds: torch.Tensor
    yoyo_seconds: torch.Tensor


def choose_target(fleet: Context, offset_x: torch.Tensor, offset_y: torch.Tensor, parameters):
    """The committed target (R, S) int64, -1 without a known enemy: the nearest known enemy
    by its distance over its weight (ties: lowest row) when the ship is fresh or its target
    is gone, else the committed one, switching to that nearest at
    target_switch_in_log_distance's rate on the log ratio of the two."""
    recall = fleet.recall
    known = recall.known[:, None, :]  # (R, 1, E)
    distance = torch.sqrt(offset_x * offset_x + offset_y * offset_y).clamp_min(TINY)
    score = torch.log(distance) - torch.log(recall.weight.clamp_min(1e-30))[:, None, :]
    score = torch.where(known, score, math.inf)  # (R, S, E)
    best = score.argmin(-1)
    committed = fleet.carry.target
    held = committed.clamp_min(0)[..., None]
    valid = (committed >= 0) & known.expand_as(score).gather(-1, held)[..., 0]
    advantage = torch.where(
        valid, score.gather(-1, held)[..., 0] - score.gather(-1, best[..., None])[..., 0], 0.0
    )
    target = commit(
        committed,
        best,
        advantage,
        fleet.carry.fresh | ~valid,
        in_own_time(parameters.target_switch_in_log_distance, fleet.time_unit),
        fleet.temperature,
        fleet.seconds,
        fleet.switching,
    )
    return torch.where(recall.known.any(-1, keepdim=True), target, -1)


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
    """The doctrine's elements (`bnb.controllers.scripted_doctrine`) on the attack: geometry
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
    fraction = torch.as_tensor(fraction, device=left.device).clamp(0.0, 1.0)
    fraction = fraction.expand(left.shape)
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


def maneuvers(
    fleet: Context,
    characteristics: Characteristics,
    doctrine: Doctrine,
    target_of,
    has_target: torch.Tensor,
    chosen: torch.Tensor,
    geometry: tuple,
    stick: torch.Tensor,
    left: torch.Tensor,
    power: torch.Tensor,
):
    """The doctrine's maneuvers (`bnb.controllers.scripted_doctrine`: the lead turn, the
    yo-yo and the brake) on the combat layer's stick, committed side and power; with the
    stalemate clock and the yo-yo's clock after this decision (advanced only when the plan
    choices may switch, so a re-read carry gives the same marginals; reset on a fresh ship).
    Returns (stick, left, power, stalemate_seconds, yoyo_seconds)."""
    target_x, target_y, distance, _, _, aim_error, target_vx, target_vy = geometry
    temperature = fleet.temperature
    carry = fleet.carry
    safe = distance.clamp_min(TINY)
    target_stall = target_of(per_ship(characteristics.stall_angle, fleet.enemy_type))
    target_heading = torch.atan2(target_vy, target_vx)
    off_its_nose = signed_angle(
        torch.cos(target_heading), torch.sin(target_heading), -target_x, -target_y
    ).abs() / target_stall.clamp_min(TINY)
    relative_x = target_vx - fleet.velocity_x
    relative_y = target_vy - fleet.velocity_y
    closing = -(target_x * relative_x + target_y * relative_y) / safe  # m/s
    approaching = (closing > 0.0) & has_target

    # The clocks: the stalemate's and the yo-yo's.
    engaged = has_target & (distance < STALEMATE_IN_GUN_RANGES * fleet.gun_range)
    gained = (aim_error.abs() < GAINED_NOSE_IN_STALL_ANGLES * fleet.stall_angle) & (
        off_its_nose > GAINED_OFF_ITS_NOSE_IN_STALL_ANGLES
    )
    fresh = carry.fresh
    if fleet.switching:
        same = chosen == carry.target
        stalemate = torch.where(
            engaged & ~gained & same & ~fresh, carry.stalemate_seconds + fleet.seconds, 0.0
        )
        running = carry.yoyo_seconds >= 0.0
        clock = torch.where(running, carry.yoyo_seconds + fleet.seconds, -1.0)
        # the first phase ends early at the wanted speed
        fast = fleet.speed > doctrine.yoyo_to_corner_speed * fleet.corner_speed
        clock = torch.where(
            running & fast & (clock < doctrine.yoyo_climb_seconds),
            doctrine.yoyo_climb_seconds,
            clock,
        )
        done = clock > doctrine.yoyo_climb_seconds + doctrine.yoyo_dive_seconds
        start = (
            ~running
            & (doctrine.yoyo > 0.0)
            & engaged
            & (stalemate > doctrine.yoyo_after_seconds)
            & (aim_error.abs() < doctrine.yoyo_phase)
            & (off_its_nose > GAINED_OFF_ITS_NOSE_IN_STALL_ANGLES)  # never under its guns
        )
        clock = torch.where(start, 0.0, torch.where(done, -1.0, clock))
        stalemate = torch.where(done, 0.0, stalemate)
        clock = torch.where(fresh | ~has_target, -1.0, clock)
    else:
        stalemate = torch.where(fresh, 0.0, carry.stalemate_seconds)
        clock = torch.where(fresh | ~has_target, -1.0, carry.yoyo_seconds)

    reverse = power_marginal(torch.zeros_like(distance), torch.ones_like(distance))
    boost = power_marginal(torch.ones_like(distance), torch.zeros_like(distance))

    # Yo-yo: the high phase eases the stick and climbs, the low phase dives.
    # (the low yo-yo, `yoyo_low` above 1/2: the first phase dives, the second leaves the power
    # to the combat layer)
    first = ((clock >= 0.0) & (clock < doctrine.yoyo_climb_seconds)).float() * doctrine.yoyo
    second = (clock >= doctrine.yoyo_climb_seconds).float() * doctrine.yoyo
    low = doctrine.yoyo_low > 0.5
    stick = stick + first[..., None] * (commanded_stick(left, doctrine.yoyo_ease) - stick)
    first_power = torch.where(low[..., None], boost, reverse)
    power = power + first[..., None] * (first_power - power)
    power = power + torch.where(low, 0.0, second)[..., None] * (boost - power)

    # Lead turn: on a head-on approach, turn before the merge.
    own_speed = torch.sqrt(
        fleet.velocity_x * fleet.velocity_x + fleet.velocity_y * fleet.velocity_y
    ).clamp_min(TINY)
    target_speed = torch.sqrt(target_vx * target_vx + target_vy * target_vy).clamp_min(TINY)
    crossing = (fleet.velocity_x * target_vx + fleet.velocity_y * target_vy) / (
        own_speed * target_speed
    )
    from_opposed = math.pi - torch.acos(crossing.clamp(-1.0, 1.0))
    time_to_merge = safe / closing.clamp_min(TINY)
    lead_weight = torch.where(
        approaching,
        doctrine.lead_turn
        * logistic(time_to_merge, doctrine.lead_turn_in_seconds, temperature)
        * logistic(from_opposed, doctrine.lead_turn_in_crossing, temperature),
        0.0,
    )
    # the side the target passes on: its offset at the closest approach, from the nose
    relative_squared = (relative_x * relative_x + relative_y * relative_y).clamp_min(TINY)
    along = (target_x * relative_x + target_y * relative_y) / relative_squared
    miss_x, miss_y = target_x - along * relative_x, target_y - along * relative_y
    passes_left = fleet.cos * miss_y - fleet.sin * miss_x > 0.0
    lead_left = torch.where(doctrine.lead_turn_one_circle > 0.5, ~passes_left, passes_left)
    stick = stick + lead_weight[..., None] * (
        commanded_stick(lead_left, doctrine.lead_turn_stick) - stick
    )
    left = torch.where(lead_weight > 0.5, lead_left, left)

    # Brake: the target closing in the rear hemisphere with its nose on the ship.
    bearing = signed_angle(fleet.cos, fleet.sin, target_x, target_y)
    braking = torch.where(
        approaching & (bearing.abs() > 0.5 * math.pi),
        doctrine.brake
        * logistic(off_its_nose, doctrine.brake_in_stall_angle, temperature)
        * logistic(distance / fleet.gun_range, doctrine.brake_in_gun_range, temperature),
        0.0,
    )
    power = power + braking[..., None] * (reverse - power)
    return stick, left, power, stalemate, clock


def combat(
    fleet: Context,
    characteristics: Characteristics,
    parameters: ScriptedParameters,
    doctrines: torch.Tensor | None = None,
) -> Combat:
    """The combat layer. The target is the committed one (`choose_target`), aimed at where
    the agent believes it is; P(shoot) is scaled by its weight, so a dead-reckoned target
    draws fewer shots the staler it is. doctrines: a doctrine table per type (T, D) or per
    view and type (R, T, D) (`bnb.controllers.scripted_doctrine`), each own ship flying its
    type's; None flies none (the agent without doctrines)."""
    recall = fleet.recall
    offset_x = recall.x[:, None, :] - fleet.x[..., None]  # (R, S, E)
    offset_y = recall.y[:, None, :] - fleet.y[..., None]
    chosen = choose_target(fleet, offset_x, offset_y, parameters)
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
    safe_distance = distance.clamp_min(TINY)
    aim_x, aim_y = lead(
        fleet,
        characteristics,
        parameters,
        target_x,
        target_y,
        target(recall.velocity_x),
        target(recall.velocity_y),
    )
    aim_error = signed_angle(fleet.cos, fleet.sin, aim_x, aim_y)
    boost = either(
        slow(fleet, parameters), torch.where(has_target, far(fleet, distance, parameters), 0.0)
    )
    reverse = torch.where(
        has_target,
        logistic(distance / fleet.gun_range, parameters.reverse_in_gun_range, fleet.temperature),
        0.0,
    )
    power, doctrine, geometry = None, None, None
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

    # Without a target: head for the centre unless already near it.
    toward_centre = fleet.distance_from_centre > (
        parameters.centre_hold_in_arena_radius * fleet.arena_radius
    )
    wanted_x, wanted_y = steer(
        fleet, torch.where(has_target, aim_x, -fleet.x), torch.where(has_target, aim_y, -fleet.y)
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
    if power is None:
        power = power_marginal(boost, reverse)
    if doctrine is not None:
        stick, left, power, stalemate, yoyo = maneuvers(
            fleet,
            characteristics,
            doctrine,
            target,
            has_target,
            chosen,
            geometry,
            stick,
            left,
            power,
        )
    else:
        fresh = fleet.carry.fresh
        stalemate = torch.where(fresh, 0.0, fleet.carry.stalemate_seconds)
        yoyo = torch.where(fresh, -1.0, fleet.carry.yoyo_seconds)

    hit_radius = target(per_ship(characteristics.hit_radius, fleet.enemy_type))
    size = (2.0 * torch.atan(hit_radius / safe_distance)).clamp(*TARGET_SIZE_RANGE)
    aligned = logistic(aim_error.abs() / size, parameters.aim_in_target_size, fleet.temperature)
    in_range = logistic(
        distance / fleet.gun_range, parameters.shoot_in_gun_range, fleet.temperature
    )
    cost = per_ship(characteristics.bullet_energy_cost, fleet.own_type)
    ready = (column(fleet.own, "gun_cooldown") <= 0.0) & (column(fleet.own, "energy") >= cost)
    shoot = floored(torch.where(has_target & ready, weight * aligned * in_range, 0.0), "shoot")
    shoot = torch.stack([1.0 - shoot, shoot], -1)
    return Combat(
        Marginals(power, stick, shoot), has_target, distance, weight, chosen, left, stalemate, yoyo
    )


def blend(first: torch.Tensor, second: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """The log-probability blend first^w second^(1 - w), renormalized; where the two have
    no common support, their mixture. weight (R, S) in [0, 1]."""
    weight = weight[..., None]
    geometric = first.pow(weight) * second.pow(1.0 - weight)
    total = geometric.sum(-1, keepdim=True)
    return torch.where(
        total > 0.0, geometric / total.clamp_min(1e-30), weight * first + (1.0 - weight) * second
    )
