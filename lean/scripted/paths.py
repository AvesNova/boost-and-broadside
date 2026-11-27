"""Candidate paths: where each ship would be, flying each of the 7 stick indices at its table
power for a short look-ahead, from the game's own flight functions; and its power outlook,
the speed and store of a longer look at each of the 3 powers (round 2, author 2026-11-26: the
stick is chosen from the pair tables at the current power, the power from the own-ship terms).

Per ship and stick candidate the state at the look-ahead points `LOOK_AHEAD` (seconds):
position, flight-path direction, nose direction, proper speed, store, and the medium there. The
speed and store are stepped segment by segment (one segment per point) from the game's
steady-flight rates at the segment's start speed and store, so a Snap's speed loss shows: read
from the game's steady-flight table (`lean.game.core.steady_lookup`, `core.steady_flight`
tabulated once per ship type with the card; round 3), not the flight functions themselves.
The turn is the steady path rate plus the nose's lag: the angle of attack follows the game's
nose spring toward the level's steady angle in closed form (`nose_response`), with the
stiffness the tick uses (the level's going out, the full spring going back) and its fixed
friction, so Sustain, Max and Snap differ in onset; the spring's
target is the steady angle at each segment's start speed (Snap's deep angle closes as the speed
bleeds, as the tick's `_holding_angle` does). In a medium every rate is scaled by 1 / n and the
path bends with the gradient, as in the tick (n and the gradient at each segment's start).

Everything is read from what the team's observation carries (positions, headings, speed,
angle of attack, nose rate, store, the fields) and the ship card. Shapes: (E, N) ships, C = 7
stick candidates, P = len(LOOK_AHEAD) points.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import Tensor

from lean.contract import POWERS, STICKS, device_constant
from lean.game import card
from lean.game.core import current_path_rate, maximum, medium, steady_lookup

# The look-ahead points, seconds from now; each closes one segment of the stepped path. A
# module setting (read at call time), so a measurement can sweep it.
LOOK_AHEAD = (0.25, 0.5, 1.0, 1.5, 2.0)
# The power outlook's points (seconds): longer, only speed and store.
POWER_LOOK = (1.0, 2.5, 5.0)
# The nose spring's target recomputed at each segment's start from the speed there (Snap's deep
# angle closes as the speed bleeds); off: the steady angle at the speed now throughout.
RETARGET = True
CANDIDATES = STICKS  # the stick indices (lean.contract: 0 straight, then Sustain, Max, Snap)
FLIGHTS = STICKS * POWERS  # flight index = stick * POWERS + power (lean.contract.join_action)
# The stick indices by |level| (card.STICK_LEVELS: 0 straight, then Sustain, Max, Snap): each
# index's level, and the first index of each level (a level's two sides fly the same speeds).
LEVEL_OF_STICK = tuple(int(abs(level)) for level in card.STICK_LEVELS)
LEVELS = max(LEVEL_OF_STICK) + 1
FIRST_OF_LEVEL = tuple(LEVEL_OF_STICK.index(level) for level in range(LEVELS))
_SMALL = 1e-6
# The action tables indexed by power.
POWER_TABLES = ("commanded_table", "demand_table")


@dataclass
class Paths:
    """Each ship's candidate paths, (E, N, C, P) each; `now` the same quantities at t = 0, (E,
    N) each (no candidate axis)."""

    x: Tensor  # metres
    y: Tensor
    path_cos: Tensor  # the flight path's direction (the world velocity's)
    path_sin: Tensor
    nose_cos: Tensor  # the nose's direction
    nose_sin: Tensor
    speed: Tensor  # proper speed, metres per second
    world_speed: Tensor  # speed over the map: proper speed / n
    energy: Tensor  # the store
    log_index: Tensor  # ln n at the point (0 without fields)
    now: SimpleNamespace
    # The own-only rates per point (lean.scripted.opponent.own_components), once attached.
    boundary: Tensor | None = None
    slow: Tensor | None = None
    # Every field above stacked (lean.scripted.opponent.PATH_FIELDS' order), once
    # materialised: (P, E, views x N, fields, C), the kernel's layout (both views).
    stacked: Tensor | None = None


def candidate_constants(constants, power: Tensor | None = None) -> SimpleNamespace:
    """The per-ship constants laid out for the (level, power) axes: a number as it is, an (E, N)
    column as (E, N, 1, 1), an action-indexed (E, width, N) table as (E, N, 1, 1, width), so the
    game's steady-flight functions broadcast over (E, N, levels, powers). The power tables are
    (E, N, 1, POWERS), or at `power` (E, N) long each ship's own (E, N, 1, 1)."""
    laid = dict(constants.uniform)
    for name, (_, width) in constants.columns.items():
        column = constants[name]
        if width == 1:
            laid[name] = column[..., None, None]
        elif name in POWER_TABLES:  # the power axis is the last: (E, N, 1, POWERS)
            table = column.movedim(1, -1)
            if power is not None:
                table = table.gather(-1, power[..., None])
            laid[name] = table[:, :, None, :]
        else:
            laid[name] = column.movedim(1, -1)[:, :, None, None, :]
    return SimpleNamespace(**laid)


def level_shapes(constants) -> tuple[Tensor, Tensor, Tensor]:
    """Per ship (E, N, LEVELS) the stick levels' nose stiffness going out (the game's table) for
    straight, Sustain, Max and Snap; and (STICKS,) long the |level| of each stick index,
    (STICKS,) float its side (+1 left, the game's stick table sign: positive is a left turn).
    The levels' flight itself is the game's steady-flight table (`core.steady_lookup`)."""
    device = constants["stick_table"].device
    columns = device_constant(FIRST_OF_LEVEL, device)
    stiffness = constants["nose_stiffness_table"].index_select(1, columns).movedim(1, -1)
    side = torch.sign(device_constant(card.STICK_LEVELS, device))
    return stiffness, device_constant(LEVEL_OF_STICK, device), side


def nose_response(offset, rate, sigma, stiffness, time):
    """The nose spring in closed form: the angle's offset e from its target obeys e'' + 2 sigma
    e' + stiffness e = 0 (the tick's spring: total friction 2 nose_damping omega, stiffness the
    level's factor x omega^2). From e(0) = `offset`, e'(0) = `rate`, at `time` (a number or a
    tensor): (e, e', the integral of e from 0). Under-, critically and over-damped alike (a
    series near critical damping, where Sustain's spring sits)."""
    b = stiffness - sigma * sigma  # beta^2: > 0 oscillates, < 0 overdamped
    x = b * (time * time)
    small = x.abs() < 0.05
    root = torch.sqrt(b.abs().clamp_min(_SMALL))
    phase = root * time
    # C(t), S(t): C'' = -b C, C(0) = 1, C'(0) = 0; S'' = -b S, S(0) = 0, S'(0) = 1.
    oscillating_c, oscillating_s = torch.cos(phase), torch.sin(phase) / root
    growing = torch.exp(phase.clamp_max(30.0))
    shrinking = 1.0 / growing
    hyperbolic_c = 0.5 * (growing + shrinking)
    hyperbolic_s = 0.5 * (growing - shrinking) / root
    series_c = 1.0 - x * (0.5 - x / 24.0)
    series_s = time * (1.0 - x * (1.0 / 6.0 - x / 120.0))
    c = torch.where(small, series_c, torch.where(b > 0.0, oscillating_c, hyperbolic_c))
    s = torch.where(small, series_s, torch.where(b > 0.0, oscillating_s, hyperbolic_s))
    decay = torch.exp(-sigma * time)
    carried = rate + sigma * offset
    value = decay * (offset * c + carried * s)
    derivative = decay * (offset * (-b * s) + carried * c) - sigma * value
    integral = -(derivative - rate + 2.0 * sigma * (value - offset)) / stiffness
    return value, derivative, integral


def _start(constants, fields: Tensor, ships):
    """What every path starts from: the speed (kept above a quarter of the stall speed: a
    never-seen enemy's row is all zeros, and the steady analysis divides by the speed), the
    path direction, 1 / n, and the angle's own rate (the nose rate less the path's: the
    observation has no angle rate)."""
    has_fields = fields.shape[-1] > 0
    speed0 = maximum(ships.speed, 0.25 * constants["stall_speed"])
    heading = torch.atan2(ships.heading_sin, ships.heading_cos)
    path0 = heading - ships.angle
    inverse_index0 = torch.exp(-ships.log_index) if has_fields else torch.ones_like(speed0)
    return speed0, path0, inverse_index0


def predict(constants, fields: Tensor, ships, power: Tensor, start: Tensor | float = 0.0) -> Paths:
    """The 7 stick candidates' paths of every slot of `ships` (anything with the State's
    per-ship columns, (E, N) each: x, y, heading_cos/sin, speed, angle, nose_rate, energy,
    log_index, gradient_x, gradient_y) at its table power `power` (E, N) long, with its
    constants and the environment's fields (E, 5, F). `start` (a number or (E, N), seconds,
    below LOOK_AHEAD[0]): how far after now `ships` is (an own ship flown one tick under its
    pending action, which the game applies first); its candidates start there and the points
    stay at the times LOOK_AHEAD from now (the first segment is that much shorter)."""
    points_at = LOOK_AHEAD
    p = candidate_constants(constants, power)  # power tables (E, N, 1, 1)
    out_stiffness, level, side = level_shapes(constants)
    has_fields = fields.shape[-1] > 0
    speed0, path0, inverse_index0 = _start(constants, fields, ships)
    angle0 = ships.angle
    path_rate0 = current_path_rate(
        SimpleNamespace(**{k: _squeeze(v) for k, v in vars(p).items()}),
        speed0,
        angle0,
        ships.energy,
    )
    bend0 = _bend(ships.gradient_x, ships.gradient_y, path0, speed0) if has_fields else 0.0
    path_rate0 = (path_rate0 + bend0) * inverse_index0
    angle_rate0 = ships.nose_rate - path_rate0

    levels = torch.arange(LEVELS, device=speed0.device)[:, None]  # (LEVELS, 1)
    ship_type = ships.type[..., None, None]
    table_power = power[..., None, None]

    def steady(speed, energy):
        """(rate, d speed / dt, d store / dt, angle) per level, (E, N, LEVELS) each, at the
        levels' speed and energy (E, N, LEVELS): the game's steady-flight table
        (`core.steady_lookup`) at the table power."""
        flown = steady_lookup(
            p,
            constants["steady_table"],
            ship_type,
            speed[..., None],
            levels,
            table_power,
            energy[..., None],
        )
        return tuple(value[..., 0] for value in flown)

    def per_stick(value: Tensor) -> Tensor:  # (E, N, LEVELS) -> (E, N, STICKS)
        return value.index_select(-1, level)

    first_of_level = device_constant(FIRST_OF_LEVEL, speed0.device)
    shape = speed0.shape + (STICKS,)
    omega = (constants["nose_frequency"] * speed0).clamp_min(card.NOSE_FREQUENCY_FLOOR)
    omega = omega * inverse_index0
    sigma = (constants["nose_damping"] * omega)[..., None]
    omega_squared = (omega * omega)[..., None]
    stiffness_out = per_stick(out_stiffness) * omega_squared
    x = ships.x[..., None].expand(shape)
    y = ships.y[..., None].expand(shape)
    path = path0[..., None].expand(shape)
    angle = angle0[..., None].expand(shape)
    angle_rate = angle_rate0[..., None].expand(shape)
    inverse_index = inverse_index0[..., None].expand(shape)
    gradient_x = ships.gradient_x[..., None].expand(shape) if has_fields else None
    gradient_y = ships.gradient_y[..., None].expand(shape) if has_fields else None
    speed = speed0[..., None].expand(shape)
    energy = ships.energy[..., None].expand(shape)
    log_index = ships.log_index[..., None].expand(shape)
    path_rate = path_rate0[..., None].expand(shape)  # the path rate now (the angle's)
    started = start[..., None] if isinstance(start, Tensor) else start
    points = []
    for index, time in enumerate(points_at):
        duration = time - started
        if index == 0:
            rate, speed_rate, store_rate, target = steady(
                speed0[..., None].expand(shape[:-1] + (LEVELS,)),
                ships.energy[..., None].expand(shape[:-1] + (LEVELS,)),
            )
        else:  # the steady state at this segment's start (each level's own speed and store)
            rate, speed_rate, store_rate, target = steady(
                speed.index_select(-1, first_of_level), energy.index_select(-1, first_of_level)
            )
        # The spring's target this segment: the level's steady angle at the segment's start.
        if index == 0 or RETARGET:
            target_now = per_stick(target) * side
        target = target_now
        steady_rate = per_stick(rate) * side * inverse_index
        if has_fields:
            steady_rate = steady_rate + _bend(gradient_x, gradient_y, path, speed) * inverse_index
        going_out = (target * angle >= 0.0) & (target.abs() > angle.abs())
        stiffness = torch.where(going_out, stiffness_out, omega_squared)
        offset0 = angle - target
        offset, offset_rate, integral = nose_response(
            offset0, angle_rate, sigma, stiffness, duration
        )
        # The path rate linear in the angle's offset, between now and the steady state.
        slope = (path_rate - steady_rate) * offset0 / (offset0 * offset0 + 1e-6)
        new_path = path + steady_rate * duration + slope * integral
        new_speed = (speed + duration * per_stick(speed_rate) * inverse_index).clamp_min(
            card.SPEED_FLOOR
        )
        new_energy = (energy + duration * per_stick(store_rate) * inverse_index).clamp(
            0.0, _grid(constants["max_energy"])
        )
        # The segment's chord: the mean map speed for its length, the mean direction, the
        # arc's shortening (sinc of half the turn).
        half = 0.5 * (new_path - path)
        middle = path + half
        chord = torch.where(half.abs() < 1e-3, 1.0 - half * half / 6.0, torch.sin(half) / half)
        length = 0.5 * (speed + new_speed) * inverse_index * duration * chord
        x = x + length * torch.cos(middle)
        y = y + length * torch.sin(middle)
        if has_fields:
            log_index, gradient_x, gradient_y = medium(fields, x.flatten(1), y.flatten(1))
            log_index = log_index.view(shape)
            gradient_x, gradient_y = gradient_x.view(shape), gradient_y.view(shape)
            inverse_index = torch.exp(-log_index)
        angle = target + offset
        angle_rate = offset_rate
        path_rate = steady_rate + slope * offset
        nose = new_path + angle
        points.append(
            (x, y, new_path, nose, new_speed, new_speed * inverse_index, new_energy, log_index)
        )
        path, speed, energy = new_path, new_speed, new_energy
        started = time
    names = ("x", "y", "path", "nose", "speed", "world_speed", "energy", "log_index")
    stacked = {
        name: torch.stack([point[i] for point in points], -1) for i, name in enumerate(names)
    }
    now = SimpleNamespace(
        x=ships.x,
        y=ships.y,
        path_cos=torch.cos(path0),
        path_sin=torch.sin(path0),
        nose_cos=ships.heading_cos,
        nose_sin=ships.heading_sin,
        speed=speed0,
        world_speed=speed0 * inverse_index0,
        energy=ships.energy,
        log_index=ships.log_index,
    )
    return Paths(
        x=stacked["x"],
        y=stacked["y"],
        path_cos=torch.cos(stacked["path"]),
        path_sin=torch.sin(stacked["path"]),
        nose_cos=torch.cos(stacked["nose"]),
        nose_sin=torch.sin(stacked["nose"]),
        speed=stacked["speed"],
        world_speed=stacked["world_speed"],
        energy=stacked["energy"],
        log_index=stacked["log_index"],
        now=now,
    )


def power_outlook(constants, fields: Tensor, ships, stick: Tensor) -> tuple[Tensor, Tensor]:
    """Each ship's speed and store (E, N, POWERS, len(POWER_LOOK)) holding stick index `stick`
    (E, N) long at each of the 3 powers for the power look (`POWER_LOOK`), stepped per segment
    from the game's steady-flight rates (its table) at the segment's start (the medium's 1 / n
    held at its value now)."""
    p = candidate_constants(constants)  # columns (E, N, 1, 1)
    level = device_constant(LEVEL_OF_STICK, stick.device)
    stick_level = level[stick][..., None, None]  # (E, N, 1, 1)
    ship_type = ships.type[..., None, None]
    powers = torch.arange(POWERS, device=stick.device)
    speed0, _, inverse_index = _start(constants, fields, ships)
    inverse_index = inverse_index[..., None, None]
    speed = speed0[..., None, None].expand(speed0.shape + (1, POWERS))
    energy = ships.energy[..., None, None].expand_as(speed)
    started = 0.0
    speeds, energies = [], []
    for time in POWER_LOOK:
        duration = time - started
        _, speed_rate, store_rate, _ = steady_lookup(
            p, constants["steady_table"], ship_type, speed, stick_level, powers, energy
        )
        speed = (speed + duration * speed_rate * inverse_index).clamp_min(card.SPEED_FLOOR)
        energy = (energy + duration * store_rate * inverse_index).clamp(
            0.0, _grid(constants["max_energy"])
        )
        speeds.append(speed[:, :, 0])
        energies.append(energy[:, :, 0])
        started = time
    return torch.stack(speeds, -1), torch.stack(energies, -1)


def _bend(gradient_x, gradient_y, path, speed):
    """The medium's bend of the flight path (rad/s before the 1 / n): proper speed times the
    gradient of ln n across the path (its left normal), as the tick bends it."""
    return speed * (torch.cos(path) * gradient_y - torch.sin(path) * gradient_x)


def _grid(value):
    """A per-ship value (E, N) laid along the candidate axis (and a power axis); a number as
    it is."""
    return value[..., None, None] if isinstance(value, Tensor) else value


def _squeeze(value):
    """A candidate-laid constant back to the ships' shape: (E, N, 1, 1) to (E, N), a table (E,
    N, 1, 1, width) or a power table (E, N, 1, POWERS) to (E, N, width); a number as it is."""
    if not isinstance(value, Tensor):
        return value
    return value[:, :, 0] if value.dim() == 4 else value[:, :, 0, 0]
