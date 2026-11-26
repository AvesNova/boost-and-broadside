"""Candidate paths: where each ship would be, flying each of the 21 flight actions (7 stick
indices x 3 powers) for a short look-ahead, from the game's own flight functions.

Per ship and candidate the state at the look-ahead points `LOOK_AHEAD` (seconds): position,
flight-path direction, nose direction, proper speed, store, and the medium there. The speed and
store are stepped segment by segment (one segment per point) from the game's steady-flight
rates (`lean.game.core.steady_flight`) at the segment's start speed and store, so a Snap's speed
loss shows. The turn is the steady path rate plus the nose's lag: the angle of attack follows the
game's nose spring toward the level's steady angle in closed form (`nose_response`), with the
stiffness the tick uses (the level's going out, the full spring going back) and its fixed
friction, so Sustain, Max and Snap differ in onset. In a medium every rate is scaled by 1 / n
and the path bends with the gradient, as in the tick (n and the gradient at each segment's
start).

Everything is read from what the team's observation carries (positions, headings, speed,
angle of attack, nose rate, store, the fields) and the ship card. Shapes: (E, N) ships, C = 21
candidates, P = len(LOOK_AHEAD) points.
"""

from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import Tensor

from lean.contract import POWERS, STICKS
from lean.game import card
from lean.game.core import current_path_rate, maximum, medium, steady_flight

# The look-ahead points, seconds from now; each closes one segment of the stepped path.
LOOK_AHEAD = (0.25, 0.5, 1.0, 2.0)
CANDIDATES = STICKS * POWERS  # flight index = stick * POWERS + power (lean.contract.join_action)
# The stick indices by |level| (card.STICK_LEVELS: 0 straight, then Sustain, Max, Snap): each
# index's level, and the first index of each level (a level's two sides fly the same speeds).
LEVEL_OF_STICK = tuple(int(abs(level)) for level in card.STICK_LEVELS)
LEVELS = max(LEVEL_OF_STICK) + 1
FIRST_OF_LEVEL = tuple(LEVEL_OF_STICK.index(level) for level in range(LEVELS))
_SMALL = 1e-6
# The action tables indexed by power (`core.steady_flight` with every power at once).
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


def candidate_constants(constants) -> SimpleNamespace:
    """The per-ship constants laid out for the (level, power) axes: a number as it is, an (E, N)
    column as (E, N, 1, 1), an action-indexed (E, width, N) table as (E, N, 1, 1, width), so the
    game's steady-flight functions broadcast over (E, N, levels, powers)."""
    laid = dict(constants.uniform)
    for name, (_, width) in constants.columns.items():
        column = constants[name]
        if width == 1:
            laid[name] = column[..., None, None]
        elif name in POWER_TABLES:  # the power axis is the last: (E, N, 1, POWERS)
            laid[name] = column.movedim(1, -1)[:, :, None, :]
        else:
            laid[name] = column.movedim(1, -1)[:, :, None, None, :]
    return SimpleNamespace(**laid)


def level_shapes(constants) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Per ship (E, N, LEVELS) the stick levels' (size, sustain, deep, stiffness going out) as
    the game's tables hold them (`steady_flight` takes the first three), for straight, Sustain,
    Max and Snap; and (STICKS,) long the |level| of each stick index, (STICKS,) float its side
    (+1 left, the game's stick table sign: positive is a left turn)."""
    device = constants["stick_table"].device
    columns = torch.tensor(FIRST_OF_LEVEL, device=device)

    def table(name: str) -> Tensor:
        return constants[name].index_select(1, columns).movedim(1, -1)  # (E, N, LEVELS)

    side = torch.sign(torch.tensor(card.STICK_LEVELS, device=device))
    return (
        table("stick_table").abs(),
        table("sustain_stick_table"),
        table("deep_stall_table"),
        table("nose_stiffness_table"),
        torch.tensor(LEVEL_OF_STICK, device=device),
        side,
    )


def nose_response(offset, rate, sigma, stiffness, time):
    """The nose spring in closed form: the angle's offset e from its target obeys e'' + 2 sigma
    e' + stiffness e = 0 (the tick's spring: total friction 2 nose_damping omega, stiffness the
    level's factor x omega^2). From e(0) = `offset`, e'(0) = `rate`, at `time` (a number):
    (e, e', the integral of e from 0). Under-, critically and over-damped alike (a series near
    critical damping, where Sustain's spring sits)."""
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


def predict(constants, fields: Tensor, ships) -> Paths:
    """The candidate paths of every slot of `ships` (anything with the State's per-ship columns,
    (E, N) each: x, y, heading_cos/sin, speed, angle, nose_rate, energy, log_index, gradient_x,
    gradient_y), with its constants and the environment's fields (E, 5, F)."""
    p = candidate_constants(constants)
    size, sustain, deep, out_stiffness, level, side = level_shapes(constants)
    has_fields = fields.shape[-1] > 0
    # Kept above a quarter of the stall speed: a never-seen enemy's row is all zeros, and the
    # steady analysis divides by the speed (no living ship flies that slowly).
    speed0 = maximum(ships.speed, 0.25 * constants["stall_speed"])
    angle0 = ships.angle
    # The path direction: the nose turned back by the angle of attack (State's velocity rows).
    heading = torch.atan2(ships.heading_sin, ships.heading_cos)
    path0 = heading - angle0
    inverse_index0 = torch.exp(-ships.log_index) if has_fields else torch.ones_like(speed0)
    # The angle's own rate: the nose rate less the path's (the observation has no angle rate).
    path_rate0 = current_path_rate(
        SimpleNamespace(**{k: _squeeze(v) for k, v in vars(p).items()}),
        speed0,
        angle0,
        ships.energy,
    )
    bend0 = _bend(ships.gradient_x, ships.gradient_y, path0, speed0) if has_fields else 0.0
    path_rate0 = path_rate0 * inverse_index0 + bend0 * inverse_index0
    angle_rate0 = ships.nose_rate - path_rate0

    # Per (E, N, LEVELS, POWERS): the steady angle at the speed now (the spring's target).
    def steady(speed, energy):
        """(rate, d speed / dt, d store / dt, angle) per level and power, (E, N, LEVELS,
        POWERS) each, at speed and energy (E, N, LEVELS, POWERS)."""
        return steady_flight(
            p, speed, size[..., None], None, energy, sustain[..., None], deep[..., None]
        )

    shape = speed0.shape + (LEVELS, POWERS)
    speed = speed0[..., None, None].expand(shape)
    energy = ships.energy[..., None, None].expand(shape)
    rate, speed_rate, store_rate, angle = steady(speed, energy)

    # Out to the stick indices: (E, N, STICKS, POWERS), the side signing the turn.
    def per_stick(value: Tensor) -> Tensor:
        return value.index_select(2, level)

    signs = side[:, None]
    target = per_stick(angle) * signs
    stiffness_factor = per_stick(out_stiffness[..., None].expand(shape))
    going_out = (target * angle0[..., None, None] >= 0.0) & (
        target.abs() > angle0.abs()[..., None, None]
    )
    omega = (constants["nose_frequency"] * speed0).clamp_min(card.NOSE_FREQUENCY_FLOOR)
    omega = omega * inverse_index0
    sigma = (constants["nose_damping"] * omega)[..., None, None]
    stiffness = torch.where(going_out, stiffness_factor, 1.0) * (omega * omega)[..., None, None]
    offset0 = angle0[..., None, None] - target
    angle_rate = angle_rate0[..., None, None].expand_as(offset0)
    # The path rate's response to the angle, linear between now and the steady state.
    steady_rate0 = per_stick(rate) * signs * inverse_index0[..., None, None]
    slope = (path_rate0[..., None, None] - steady_rate0) * offset0 / (offset0 * offset0 + 1e-6)

    x = ships.x[..., None, None].expand(shape[:2] + (STICKS, POWERS))
    y = ships.y[..., None, None].expand_as(x)
    path = path0[..., None, None].expand_as(x)
    inverse_index = inverse_index0[..., None, None].expand_as(x)
    gradient_x = ships.gradient_x[..., None, None].expand_as(x) if has_fields else None
    gradient_y = ships.gradient_y[..., None, None].expand_as(x) if has_fields else None
    speed_now = speed0[..., None, None].expand_as(x)
    energy_now = ships.energy[..., None, None].expand_as(x)
    log_index = ships.log_index[..., None, None].expand_as(x)
    steady_turned = torch.zeros_like(x)  # the integral of the steady path rate so far
    first_of_level = torch.tensor(FIRST_OF_LEVEL, device=x.device)
    started = 0.0
    points = []
    for index, time in enumerate(LOOK_AHEAD):
        duration = time - started
        if index > 0:  # the steady rates at this segment's start (the first: computed above)
            rate, speed_rate, store_rate, _ = steady(
                speed_now.index_select(2, first_of_level),
                energy_now.index_select(2, first_of_level),
            )
        turn_rate = per_stick(rate) * signs * inverse_index
        if has_fields:
            turn_rate = turn_rate + _bend(gradient_x, gradient_y, path, speed_now) * inverse_index
        offset, _, integral = nose_response(offset0, angle_rate, sigma, stiffness, time)
        steady_turned = steady_turned + turn_rate * duration
        new_path = path0[..., None, None] + steady_turned + slope * integral
        new_speed = (speed_now + duration * per_stick(speed_rate) * inverse_index).clamp_min(
            card.SPEED_FLOOR
        )
        new_energy = (energy_now + duration * per_stick(store_rate) * inverse_index).clamp(
            0.0, _grid(constants["max_energy"])
        )
        # The segment's chord: the mean map speed for its length, the mean direction, the
        # arc's shortening (sinc of half the turn).
        turned = new_path - path
        middle = path + 0.5 * turned
        half = 0.5 * turned
        chord = torch.where(half.abs() < 1e-3, 1.0 - half * half / 6.0, torch.sin(half) / half)
        length = 0.5 * (speed_now + new_speed) * inverse_index * duration * chord
        x = x + length * torch.cos(middle)
        y = y + length * torch.sin(middle)
        if has_fields:
            log_index, gradient_x, gradient_y = medium(fields, x.flatten(2), y.flatten(2))
            log_index = log_index.view_as(x)
            gradient_x, gradient_y = gradient_x.view_as(x), gradient_y.view_as(x)
            inverse_index = torch.exp(-log_index)
        nose = new_path + target + offset
        points.append(
            (x, y, new_path, nose, new_speed, new_speed * inverse_index, new_energy, log_index)
        )
        path, speed_now, energy_now = new_path, new_speed, new_energy
        started = time
    names = ("x", "y", "path", "nose", "speed", "world_speed", "energy", "log_index")
    stacked = {
        name: torch.stack([point[i] for point in points], -1).flatten(2, 3)
        for i, name in enumerate(names)
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


def _bend(gradient_x, gradient_y, path, speed):
    """The medium's bend of the flight path (rad/s before the 1 / n): proper speed times the
    gradient of ln n across the path (its left normal), as the tick bends it."""
    return speed * (torch.cos(path) * gradient_y - torch.sin(path) * gradient_x)


def _grid(value):
    """A per-ship value (E, N) laid along the (stick, power) axes; a number as it is."""
    return value[..., None, None] if isinstance(value, Tensor) else value


def _squeeze(value):
    """A candidate-laid constant back to the ships' shape: (E, N, 1, 1) to (E, N), a table (E,
    N, 1, 1, width) or a power table (E, N, 1, POWERS) to (E, N, width); a number as it is."""
    if not isinstance(value, Tensor):
        return value
    return value[:, :, 0] if value.dim() == 4 else value[:, :, 0, 0]
