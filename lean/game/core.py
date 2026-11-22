"""The game: reset, one tick, and the observation, as pure functions of tensors.

Sizes: E environments, N ship slots, K bullet slots per ship, F fields. Bullet o * K + k
belongs to ship o. Units: metres, seconds, radians, joules; clocks in ticks.

Tick order: fire, flight, bullet motion, swept hit test, damage and shields, boundary,
deaths, win check and clock, counters. Nothing here resets an environment: the caller
passes StepOut.done to reset_done.
"""

import dataclasses
import math
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import Tensor

from lean.contract import (
    BULLET_SLOTS,
    GLOBAL_FEATURES,
    SHIP_FEATURES,
    SHIP_TYPES,
    Obs,
    State,
    StepOut,
)

from . import card


@dataclass
class GameConfig:
    """The rules of one run.

    arena_radius: metres; boundary damage starts beyond it. boundary_ramp_width: metres over
    which boundary damage rises to its maximum. clock_ticks: the episode length. spawn:
    "sides" (each team in a disc on its side of a random axis, noses toward the enemy) or
    "mixed" (every ship uniform in the arena, uniform heading). fields: fixed refractive
    fields, each (x, y, radius, band_width, level) with level in half-octaves of the index;
    empty for no medium. lineup: type names by rank within a team, cycled, the same for both
    teams. tick_length: seconds.
    """

    arena_radius: float = 4200.0
    boundary_ramp_width: float = 300.0
    clock_ticks: int = 5400
    spawn: str = "sides"
    fields: tuple[tuple[float, float, float, float, float], ...] = ()
    lineup: tuple[str, ...] = ("yongma", "jujak")
    tick_length: float = 1.0 / 30.0


# --------------------------------------------------------------------------- constants


def ship_constants(config: GameConfig, ship_type: Tensor) -> dict[str, Tensor]:
    """Per-ship constants for types (E, N) long: every value (E, N) float32 except the stick
    table (E, N, 7), the power tables (E, N, 3), the integer counts (E, N) long and the
    field arrays (F,)."""
    device = ship_type.device

    def of(name: str) -> Tensor:
        return torch.tensor(card.CARD[name], dtype=torch.float32, device=device)[ship_type]

    lift, acceleration, max_energy = of("max_lift_coefficient"), of("max_acceleration"), of("max_energy")
    stall = card.STALL_ANGLE * of("stall_angle_scale")
    knee = card.ENGINE_KNEE * of("engine_knee_scale")
    climb = card.CLIMB_EXCHANGE * of("climb_exchange_scale")
    dive = card.DIVE_EXCHANGE * of("dive_exchange_scale")
    fine, sustained = of("fine_stick"), of("sustained_stick")
    zeros, ones = torch.zeros_like(fine), torch.ones_like(fine)
    levels = torch.tensor(card.STICK_LEVELS, device=device)
    magnitude = torch.stack([zeros, fine, fine, sustained, sustained, ones, ones], -1)
    width = card.GRAVITY / 32.0
    raw = torch.stack([torch.full_like(climb, card.GRAVITY), card.GRAVITY - dive, card.GRAVITY + climb], -1)
    fields = torch.tensor(config.fields, dtype=torch.float32, device=device).reshape(-1, 5)
    hit_radius = of("hit_radius")
    return {
        "lift": lift,
        "inverse_lift": 1.0 / lift,
        "inverse_lift_squared": 1.0 / (lift * lift),
        "lift_over_stall": lift / stall,
        "max_acceleration": acceleration,
        "zero_lift_drag": of("zero_lift_drag"),
        "induced_drag": card.INDUCED_DRAG_FACTOR * of("induced_drag_scale"),
        "power_full_store": of("engine_power_full_store"),
        "power_surplus": of("engine_power") - of("engine_power_full_store"),
        "static_thrust": of("static_thrust"),
        "max_energy": max_energy,
        "inverse_max_energy": 1.0 / max_energy,
        "inverse_fade": 1.0 / (card.STORE_FADE_BAND * max_energy),
        "stall": stall,
        "inverse_stall": 1.0 / stall,
        "knee": knee,
        "inverse_above_knee": 1.0 / (1.0 - knee),
        "nose_frequency": of("corner_nose_frequency") / torch.sqrt(acceleration / lift),
        "stick_table": torch.sign(levels) * magnitude,
        "commanded_table": torch.stack([zeros, -dive, climb], -1),
        "demand_table": 0.5 * (raw + torch.sqrt(raw * raw + width * width)),
        "muzzle_speed": of("muzzle_speed"),
        "spread_speed": card.BULLET_SPREAD * of("bullet_spread_scale"),
        "bullet_cost": of("bullet_energy_cost"),
        "bullet_drag": card.BULLET_DRAG * of("bullet_drag_scale"),
        "bullet_moves": torch.round(of("bullet_life") / config.tick_length).long(),
        "fire_interval": torch.round(1.0 / (config.tick_length * of("fire_rate"))).long(),
        "hit_damage": of("hit_damage"),
        "explosive_share": of("explosive_share"),
        "inverse_reference_bullet_speed": 1.0 / of("reference_bullet_speed"),
        "hit_radius": hit_radius,
        "hit_radius_squared": hit_radius * hit_radius,
        "core_radius": of("core_radius"),
        "head_on": card.HEAD_ON_FRACTION * of("head_on_fraction_scale"),
        "inverse_head_on_angle": 1.0 / (card.HEAD_ON_ANGLE * of("head_on_angle_scale")),
        "head_on_sharpness": of("head_on_sharpness"),
        "max_shield": of("max_shield"),
        "shield_rate": card.SHIELD_RECHARGE_RATE * of("shield_recharge_rate_scale"),
        "shield_delay": card.SHIELD_RECHARGE_DELAY * of("shield_recharge_delay_scale"),
        "spawn_energy": of("spawn_energy"),
        "field_x": fields[:, 0],
        "field_y": fields[:, 1],
        "field_radius": fields[:, 2],
        "field_band": fields[:, 3],
        "field_target": fields[:, 4] * card.LN_SQRT2,
    }


# --------------------------------------------------------------------------- numerics


def softmin(a: Tensor, b: Tensor, exponent: float) -> Tensor:
    """low (1 + (low / high)^q)^(-1/q) for a, b >= 0: a smooth minimum."""
    low, high = torch.minimum(a, b), torch.maximum(a, b)
    return low * (1.0 + (low / high.clamp_min(1e-30)) ** exponent) ** (-1.0 / exponent)


def small_angle_cos_sin(angle: Tensor) -> tuple[Tensor, Tensor]:
    """(cos, sin) of a small angle by the game's truncated series."""
    square = angle * angle
    cos = 1.0 - square * (0.5 - square * (1.0 / 24.0 - square / 720.0))
    sin = angle * (1.0 - square * (1.0 / 6.0 - square * (1.0 / 120.0 - square / 5040.0)))
    return cos, sin


def smootherstep(t: Tensor) -> Tensor:
    """t^3 (t (6 t - 15) + 10) for t in [0, 1]."""
    return t * t * t * (t * (6.0 * t - 15.0) + 10.0)


def medium(constants: dict[str, Tensor], x: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(ln n, d/dx ln n, d/dy ln n) at points (E, P) for the run's fixed fields (F,).
    With no fields the medium is exactly 0."""
    if constants["field_x"].shape[0] == 0:
        zeros = torch.zeros_like(x)
        return zeros, zeros, zeros
    target, band = constants["field_target"], constants["field_band"]
    offset_x, offset_y = x[..., None] - constants["field_x"], y[..., None] - constants["field_y"]
    distance = torch.sqrt(offset_x * offset_x + offset_y * offset_y)
    depth = 0.5 - (distance - constants["field_radius"]) / band
    inward = depth.clamp(0.0, 1.0)
    outward = 1.0 - inward
    profile = smootherstep(inward)
    complement = smootherstep(outward)
    in_band = (depth > 0.0) & (depth < 1.0) & (distance > 1e-6)
    slope = torch.where(in_band, -30.0 * (inward * outward) ** 2 / (band * distance.clamp_min(1e-6)), 0.0)
    profile_gradient_x, profile_gradient_y = slope * offset_x, slope * offset_y
    weight = profile.sum(-1)
    weighted_target = (profile * target).sum(-1)
    clear = complement.prod(-1)
    leave_one_out = torch.where(in_band, slope / torch.where(in_band, complement, 1.0), 0.0)
    coverage_gradient_x = clear * (leave_one_out * offset_x).sum(-1)
    coverage_gradient_y = clear * (leave_one_out * offset_y).sum(-1)
    weight_gradient_x, weight_gradient_y = profile_gradient_x.sum(-1), profile_gradient_y.sum(-1)
    target_gradient_x = (profile_gradient_x * target).sum(-1)
    target_gradient_y = (profile_gradient_y * target).sum(-1)
    coverage = 1.0 - clear
    covered = weight > 1e-12
    safe_weight = torch.where(covered, weight, 1.0)
    mean = torch.where(covered, weighted_target / safe_weight, 0.0)
    mean_gradient_x = torch.where(covered, (target_gradient_x - mean * weight_gradient_x) / safe_weight, 0.0)
    mean_gradient_y = torch.where(covered, (target_gradient_y - mean * weight_gradient_y) / safe_weight, 0.0)
    return (
        coverage * mean,
        mean * coverage_gradient_x + coverage * mean_gradient_x,
        mean * coverage_gradient_y + coverage * mean_gradient_y,
    )


# --------------------------------------------------------------------------- flight


def _setting(p, control, speed, energy):
    """The controls resolved at a speed and store: signed lateral acceleration, the
    exchange between speed and store, the vertical acceleration held, thrust and charge."""
    t = ((energy * p.inverse_max_energy - p.knee) * p.inverse_above_knee).clamp(0.0, 1.0)
    fall = t * t * (3.0 - 2.0 * t)
    reach = p.lift * speed * speed
    q = card.LIFT_SOFTMIN_EXPONENT
    budget = softmin(reach, p.max_acceleration, q).clamp_min(1e-12)
    command = softmin(control.stick_size * reach, p.max_acceleration, q)
    sink_room = (energy * p.inverse_fade).clamp(0.0, 1.0)
    budget_squared = budget * budget
    held = softmin(control.demand * (1.0 - sink_room), budget, q)
    lateral_size = softmin(command, torch.sqrt((budget_squared - held * held).clamp_min(0.0)), q)
    vertical = softmin(control.demand, torch.sqrt((budget_squared - lateral_size * lateral_size).clamp_min(0.0)), q)
    exchange = control.commanded - (control.demand - vertical)
    share = (lateral_size / budget).clamp(0.0, 1.0)
    headroom = ((p.max_energy - energy) * p.inverse_fade).clamp(0.0, 1.0)
    exchange = exchange * torch.where(exchange > 0.0, headroom, sink_room)
    charge = (
        p.power_surplus * (1.0 - fall) * torch.sqrt(1.0 - share * share)
        * (vertical * control.inverse_demand) * headroom
    )
    thrust = softmin(p.power_full_store / speed, p.static_thrust, card.THRUST_SOFTMIN_EXPONENT)
    return control.stick_sign * lateral_size, exchange, vertical, thrust, charge


def _rates(p, setting, inverse_index, speed, lift, angle_cos, inverse_speed_squared):
    """(d speed / dt, d energy / dt) for a setting, a lift coefficient and an angle of attack."""
    _, exchange, vertical, thrust, charge = setting
    induced = vertical * inverse_speed_squared
    load = lift * lift + induced * induced
    drag = (
        p.zero_lift_drag * (1.0 + (speed * (1.0 / card.CRITICAL_SPEED)) ** card.DRAG_RISE_EXPONENT)
        + p.induced_drag * load
        * (1.0 + card.SEPARATION_DRAG * (load * p.inverse_lift_squared) ** (0.5 * card.SEPARATION_EXPONENT))
    ) * speed * speed
    return (thrust * angle_cos - drag - exchange) * inverse_index, (speed * exchange + charge) * inverse_index


def _lift_coefficient(p, angle):
    """The lift coefficient at an angle of attack: a parabola peaking at the stall angle."""
    ratio = (angle * p.inverse_stall).clamp(-2.0, 2.0)
    return p.lift * (2.0 * ratio - ratio * ratio.abs())


def _matrix_exponential(a11, a12, a21, a22, duration):
    """exp(duration * A) for 2x2 A given by its entries (E, N): a degree-4 series, squared once."""
    b = torch.stack([torch.stack([a11, a12], -1), torch.stack([a21, a22], -1)], -2) * (duration / 2.0)
    eye = torch.eye(2, dtype=b.dtype, device=b.device)
    h = eye + b / 4.0
    h = eye + (b @ h) / 3.0
    h = eye + (b @ h) / 2.0
    h = eye + b @ h
    h = h @ h
    return h[..., 0, 0], h[..., 0, 1], h[..., 1, 0], h[..., 1, 1]


def _nose_frame(s):
    """The medium's gradient in the nose frame: (along, across)."""
    return (
        s.heading_cos * s.gradient_x + s.heading_sin * s.gradient_y,
        s.heading_cos * s.gradient_y - s.heading_sin * s.gradient_x,
    )


def _kick(p, control, s, inverse_index, gradient_along, gradient_across, duration):
    """Advance speed, angle of attack, nose rate and energy by duration at a fixed position."""
    speed0, angle0, cos0, sin0, energy, nose_rate = s.speed, s.angle, s.angle_cos, s.angle_sin, s.energy, s.nose_rate
    floor = card.SPEED_FLOOR
    safe0 = speed0.clamp_min(floor)
    first = _setting(p, control, safe0, energy)
    lift0 = _lift_coefficient(p, angle0)
    speed_rate0, energy_rate0 = _rates(p, first, inverse_index, speed0, lift0, cos0, 1.0 / (safe0 * safe0))
    speed = (speed0 + 0.5 * duration * speed_rate0).clamp_min(floor)
    current = _setting(p, control, speed, energy + 0.5 * duration * energy_rate0)
    lateral, _, _, thrust, _ = current
    speed_squared = speed * speed
    inverse_speed_squared = 1.0 / speed_squared
    over = inverse_index / speed
    path_rate = (lift0 * speed_squared + thrust * sin0) * over + (speed * inverse_index) * (
        cos0 * gradient_across + sin0 * gradient_along
    )
    ratio = (angle0 * p.inverse_stall).abs()
    slope = (p.lift_over_stall * (2.0 - 2.0 * ratio)).clamp_min(0.0)
    kappa = (slope * speed_squared + thrust * cos0) * over
    omega = (p.nose_frequency * speed).clamp_min(card.NOSE_FREQUENCY_FLOOR) * inverse_index
    reach = (lateral.abs() * inverse_speed_squared * p.inverse_lift).clamp(0.0, 1.0)
    step_angle = torch.sign(lateral) * p.stall * (1.0 - torch.sqrt(1.0 - reach)) - angle0
    damping = 2.0 * card.NOSE_DAMPING * omega
    m11, m12, m21, m22 = _matrix_exponential(
        -kappa, torch.ones_like(kappa), damping * kappa - omega * omega, -damping, duration
    )
    steady = kappa * step_angle
    offset = (nose_rate - path_rate) - steady
    turn = step_angle - m11 * step_angle + m12 * offset
    new_rate = path_rate + steady - m21 * step_angle + m22 * offset
    half_cos, half_sin = small_angle_cos_sin(0.5 * turn)
    middle_cos = cos0 * half_cos - sin0 * half_sin
    middle_lift = _lift_coefficient(p, angle0 + 0.5 * turn)
    speed_rate1, _ = _rates(p, current, inverse_index, speed0, middle_lift, middle_cos, inverse_speed_squared)
    speed_rate2, energy_rate2 = _rates(
        p, current, inverse_index, (speed0 + 0.5 * duration * speed_rate1).clamp_min(floor),
        middle_lift, middle_cos, inverse_speed_squared,
    )
    turn_cos, turn_sin = small_angle_cos_sin(turn)
    s.speed = (speed0 + duration * speed_rate2).clamp_min(floor)
    s.angle = angle0 + turn
    s.angle_cos, s.angle_sin = cos0 * turn_cos - sin0 * turn_sin, cos0 * turn_sin + sin0 * turn_cos
    s.nose_rate = new_rate
    s.energy = torch.minimum((energy + duration * energy_rate2).clamp_min(0.0), p.max_energy)
    return s


def flight_tick(constants: dict[str, Tensor], tick_length: float, s: SimpleNamespace, stick: Tensor, power: Tensor):
    """One tick of flight for every slot (E, N): half a kick, a drift, half a kick.

    s carries x, y, heading_cos, heading_sin, speed, angle, angle_cos, angle_sin, nose_rate,
    energy and the medium at the ship (log_index, gradient_x, gradient_y); it is updated in
    place and returned. stick, power: (E, N) long action indices.
    """
    p = SimpleNamespace(**constants)
    control = SimpleNamespace()
    signed = torch.gather(p.stick_table, -1, stick[..., None])[..., 0]
    control.stick_size, control.stick_sign = signed.abs(), torch.sign(signed)
    control.commanded = torch.gather(p.commanded_table, -1, power[..., None])[..., 0]
    control.demand = torch.gather(p.demand_table, -1, power[..., None])[..., 0]
    control.inverse_demand = 1.0 / control.demand
    h = tick_length
    inverse_index = torch.exp(-s.log_index)
    s = _kick(p, control, s, inverse_index, *_nose_frame(s), 0.5 * h)
    along, across = _nose_frame(s)
    velocity = s.speed * inverse_index
    velocity_x, velocity_y = velocity * s.angle_cos, -velocity * s.angle_sin
    correction = 0.5 * h * (along * velocity_x + across * velocity_y)
    stretch = 1.0 / (1.0 + correction * (1.0 + correction * (0.5 + correction / 6.0)))
    velocity_x, velocity_y = velocity_x * stretch, velocity_y * stretch
    arc = s.nose_rate * h
    arc_squared = arc * arc
    forward = (1.0 - arc_squared * (1.0 / 6.0 - arc_squared / 120.0)) * h
    sideways = arc * (0.5 - arc_squared * (1.0 / 24.0 - arc_squared / 720.0)) * h
    body_x = velocity_x * forward - velocity_y * sideways
    body_y = velocity_x * sideways + velocity_y * forward
    s.x = s.x + (s.heading_cos * body_x - s.heading_sin * body_y)
    s.y = s.y + (s.heading_sin * body_x + s.heading_cos * body_y)
    arc_cos, arc_sin = small_angle_cos_sin(arc)
    heading_cos = s.heading_cos * arc_cos - s.heading_sin * arc_sin
    heading_sin = s.heading_cos * arc_sin + s.heading_sin * arc_cos
    norm = torch.rsqrt(heading_cos * heading_cos + heading_sin * heading_sin)
    s.heading_cos, s.heading_sin = heading_cos * norm, heading_sin * norm
    norm = torch.rsqrt(s.angle_cos * s.angle_cos + s.angle_sin * s.angle_sin)
    s.angle_cos, s.angle_sin = s.angle_cos * norm, s.angle_sin * norm
    s.log_index, s.gradient_x, s.gradient_y = medium(constants, s.x, s.y)
    return _kick(p, control, s, torch.exp(-s.log_index), *_nose_frame(s), 0.5 * h)


# --------------------------------------------------------------------------- bullets


def _half_bend(tick_length, direction_x, direction_y, speed, index, gradient_x, gradient_y):
    """The direction turned toward the higher index by half a tick."""
    turn = 0.5 * tick_length * (speed / index) * (direction_x * gradient_y - direction_y * gradient_x)
    cos, sin = small_angle_cos_sin(turn)
    return direction_x * cos - direction_y * sin, direction_x * sin + direction_y * cos


def bullet_motion(constants, tick_length, x, y, direction_x, direction_y, speed, drag):
    """One move of every bullet (E, P): half drag, half bend, the step, half bend, half
    drag. drag: (E, P) the shooter's drag constant, 1/m. Returns x, y, direction, speed."""
    log_index, gradient_x, gradient_y = medium(constants, x, y)
    index = torch.exp(log_index)
    half = drag * 0.5 * tick_length
    speed = speed / (1.0 + half * speed / index)
    direction_x, direction_y = _half_bend(tick_length, direction_x, direction_y, speed, index, gradient_x, gradient_y)
    through = speed / index
    correction = 0.5 * tick_length * through * (direction_x * gradient_x + direction_y * gradient_y)
    step = through * tick_length / (1.0 + correction * (1.0 + correction * (0.5 + correction / 6.0)))
    x, y = x + direction_x * step, y + direction_y * step
    log_index, gradient_x, gradient_y = medium(constants, x, y)
    index = torch.exp(log_index)
    direction_x, direction_y = _half_bend(tick_length, direction_x, direction_y, speed, index, gradient_x, gradient_y)
    speed = speed / (1.0 + half * speed / index)
    return x, y, direction_x, direction_y, speed


# --------------------------------------------------------------------------- the tick


def draw_noise(config: GameConfig, envs: int, slots: int, generator: torch.Generator, device) -> dict[str, Tensor]:
    """The random draws one tick needs: "spread" (E, N, 2) standard normals."""
    return {"spread": torch.randn((envs, slots, 2), generator=generator, device=device)}


def step(config: GameConfig, state: State, action: Tensor, noise: dict[str, Tensor]) -> tuple[State, StepOut]:
    """One tick. action: (E, N, 3) long [stick, power, fire]; noise from draw_noise.

    Slots that are not alive at tick start neither act nor change. The returned state of a
    finished environment (StepOut.done) is the final state of its episode.
    """
    c = state.constants
    h = config.tick_length
    E, N, K = state.bullet_x.shape
    device = state.x.device
    stick, power, shoot = action[..., 0], action[..., 1], action[..., 2] == 1
    alive0 = state.alive
    log_index, gradient_x, gradient_y = medium(c, state.x, state.y)

    # Fire, from the tick-start state.
    fired = alive0 & shoot & (state.cooldown == 0) & (state.energy >= c["bullet_cost"])
    path_x = state.speed * (state.heading_cos * state.angle_cos + state.heading_sin * state.angle_sin)
    path_y = state.speed * (state.heading_sin * state.angle_cos - state.heading_cos * state.angle_sin)
    spread = noise["spread"]
    launch_x = path_x + c["muzzle_speed"] * state.heading_cos + c["spread_speed"] * spread[..., 0]
    launch_y = path_y + c["muzzle_speed"] * state.heading_sin + c["spread_speed"] * spread[..., 1]
    launch_speed = torch.sqrt(launch_x * launch_x + launch_y * launch_y)
    slot = torch.arange(K, device=device)
    written = fired[..., None] & (slot == state.next_slot[..., None])

    def write(new: Tensor, old: Tensor) -> Tensor:
        return torch.where(written, new[..., None], old)

    bullet_x0, bullet_y0 = write(state.x, state.bullet_x), write(state.y, state.bullet_y)
    direction_x0 = write(launch_x / launch_speed, state.bullet_dir_x)
    direction_y0 = write(launch_y / launch_speed, state.bullet_dir_y)
    bullet_speed0 = write(launch_speed, state.bullet_speed)
    moves_left0 = write(c["bullet_moves"], state.bullet_moves_left)
    energy = torch.where(fired, state.energy - c["bullet_cost"], state.energy)
    cooldown = torch.where(fired, c["fire_interval"], state.cooldown)
    next_slot = torch.where(fired, (state.next_slot + 1) % K, state.next_slot)

    # Flight; slots not alive at tick start keep everything.
    flown = flight_tick(
        c, h,
        SimpleNamespace(
            x=state.x, y=state.y, heading_cos=state.heading_cos, heading_sin=state.heading_sin,
            speed=state.speed, angle=state.angle, angle_cos=state.angle_cos, angle_sin=state.angle_sin,
            nose_rate=state.nose_rate, energy=energy,
            log_index=log_index, gradient_x=gradient_x, gradient_y=gradient_y,
        ),
        stick, power,
    )

    def flew(new: Tensor, old: Tensor) -> Tensor:
        return torch.where(alive0, new, old)

    x, y = flew(flown.x, state.x), flew(flown.y, state.y)
    heading_cos, heading_sin = flew(flown.heading_cos, state.heading_cos), flew(flown.heading_sin, state.heading_sin)

    # Bullet motion; inactive bullets are frozen.
    flying = moves_left0 > 0
    flat = (E, N * K)
    moved = bullet_motion(
        c, h, bullet_x0.reshape(flat), bullet_y0.reshape(flat), direction_x0.reshape(flat),
        direction_y0.reshape(flat), bullet_speed0.reshape(flat),
        c["bullet_drag"][..., None].expand(E, N, K).reshape(flat),
    )
    bullet_x, bullet_y, direction_x, direction_y, bullet_speed = (
        torch.where(flying, new.reshape(E, N, K), old)
        for new, old in zip(moved, (bullet_x0, bullet_y0, direction_x0, direction_y0, bullet_speed0))
    )

    # The swept test between the two motions, every bullet against every ship: (E, N K, N).
    per_bullet = (E, N * K, 1)
    start_x = bullet_x0.reshape(per_bullet) - state.x[:, None, :]
    start_y = bullet_y0.reshape(per_bullet) - state.y[:, None, :]
    relative_x = (bullet_x - bullet_x0).reshape(per_bullet) - (x - state.x)[:, None, :]
    relative_y = (bullet_y - bullet_y0).reshape(per_bullet) - (y - state.y)[:, None, :]
    relative_squared = (relative_x * relative_x + relative_y * relative_y).clamp_min(1e-12)
    tau = (-(start_x * relative_x + start_y * relative_y) / relative_squared).clamp(0.0, 1.0)
    miss_x, miss_y = start_x + tau * relative_x, start_y + tau * relative_y
    miss_squared = miss_x * miss_x + miss_y * miss_y
    ship = torch.arange(N, device=device)
    owner = ship.repeat_interleave(K)
    reach = torch.where(alive0, c["hit_radius_squared"], -1.0)
    contact = (miss_squared < reach[:, None, :]) & (owner[:, None] != ship) & flying.reshape(per_bullet)

    # First contact: the smallest tau, ties within 1e-6 to the closer pass.
    infinity = torch.inf
    contact_tau = torch.where(contact, tau, infinity)
    first_tau = contact_tau.min(-1).values
    tied = contact & (contact_tau <= first_tau[..., None] + 1e-6)
    target = torch.where(tied, miss_squared, infinity).argmin(-1)
    hit = first_tau < infinity

    # Damage per bullet on its target (the target is slot 0 and unused for a miss).
    def of_target(values: Tensor) -> Tensor:
        return values.gather(1, target)

    def of_owner(values: Tensor) -> Tensor:
        return values[..., None].expand(E, N, K).reshape(flat)

    flat_direction_x, flat_direction_y = direction_x.reshape(flat), direction_y.reshape(flat)
    flat_speed = bullet_speed.reshape(flat)
    nose_cos, nose_sin = of_target(heading_cos), of_target(heading_sin)
    theta = torch.atan2(
        (flat_direction_x * nose_sin - flat_direction_y * nose_cos).abs(),
        -(flat_direction_x * nose_cos + flat_direction_y * nose_sin),
    )
    speed_ratio = flat_speed * of_owner(c["inverse_reference_bullet_speed"])
    explosive = of_owner(c["explosive_share"])
    full = of_owner(c["hit_damage"]) * (explosive + (1.0 - explosive) * speed_ratio * speed_ratio)
    target_start_x = bullet_x0.reshape(flat) - of_target(state.x)
    target_start_y = bullet_y0.reshape(flat) - of_target(state.y)
    target_relative_x = (bullet_x - bullet_x0).reshape(flat) - of_target(x - state.x)
    target_relative_y = (bullet_y - bullet_y0).reshape(flat) - of_target(y - state.y)
    cross = target_start_x * target_relative_y - target_start_y * target_relative_x
    line = cross * cross / (target_relative_x * target_relative_x + target_relative_y * target_relative_y).clamp_min(1e-12)
    impact_squared = torch.minimum(line, target_start_x * target_start_x + target_start_y * target_start_y)
    radius, core = of_target(c["hit_radius"]), of_target(c["core_radius"])
    band = ((impact_squared.sqrt() - core) / (radius - core).clamp_min(1e-6)).clamp(0.0, 1.0)
    closeness = 1.0 - band * band * (3.0 - 2.0 * band)
    head_on = of_target(c["head_on"])
    aspect_ratio = theta * of_target(c["inverse_head_on_angle"])
    aspect = 1.0 - (1.0 - head_on) * torch.exp(-card.LN_2 * aspect_ratio.pow(of_target(c["head_on_sharpness"])))
    bullet_damage = torch.where(hit, full * closeness * aspect, 0.0)

    enemy = state.team[:, owner] != of_target(state.team)
    hit_taken = torch.zeros((E, N), dtype=bullet_damage.dtype, device=device).scatter_add(1, target, bullet_damage)
    damage_dealt = torch.where(enemy, bullet_damage, 0.0).reshape(E, N, K).sum(-1)
    friendly_damage = torch.where(enemy, 0.0, bullet_damage).reshape(E, N, K).sum(-1)

    # The earliest hit on each ship names its killer: smallest tau, then lowest bullet index.
    hit_tau = torch.where(hit, first_tau, infinity)
    earliest_tau = torch.full((E, N), infinity, device=device).scatter_reduce(1, target, hit_tau, "amin")
    earliest = hit & (hit_tau <= earliest_tau.gather(1, target))
    no_bullet = N * K
    bullet_index = torch.arange(N * K, device=device).expand(E, -1)
    earliest_bullet = torch.full((E, N), no_bullet, dtype=torch.long, device=device).scatter_reduce(
        1, target, torch.where(earliest, bullet_index, no_bullet), "amin"
    )
    first_shooter = torch.where(earliest_bullet < no_bullet, earliest_bullet // K, -1)

    # Boundary damage after the move, then shields and deaths.
    depth = torch.sqrt(x * x + y * y) - config.arena_radius
    boundary = torch.where(
        alive0,
        card.BOUNDARY_MAX_DAMAGE * smootherstep((depth / config.boundary_ramp_width).clamp(0.0, 1.0)) * h,
        0.0,
    )
    total = hit_taken + boundary
    quiet = (power == 0) & (stick <= card.QUIET_STICK_MAX) & ~shoot
    hurt = total > 0.0
    died = alive0 & hurt & (state.shield <= 0.0)
    shield = (state.shield - total).clamp_min(0.0)
    recharge = torch.minimum(c["max_shield"] - shield, c["shield_rate"] * (h - state.shield_wait).clamp(0.0, h))
    restored = torch.where(~hurt & quiet, recharge, 0.0)
    wait = torch.where(hurt | ~quiet, c["shield_delay"], (state.shield_wait - h).clamp_min(0.0))
    alive = alive0 & ~died

    # Win check and clock.
    team_one = state.team == 1
    left_zero = (alive & ~team_one).sum(-1)
    left_one = (alive & team_one).sum(-1)
    wiped = (left_zero == 0) | (left_one == 0)
    wipe_result = torch.where(left_zero > 0, 0, torch.where(left_one > 0, 1, 2))
    health = torch.where(alive, shield / c["max_shield"], 0.0)
    lead = torch.where(team_one, -health, health).sum(-1)
    clock_result = torch.where(lead > 0.0, 0, torch.where(lead < 0.0, 1, 2))
    tick = state.tick + 1
    done = wiped | (tick >= config.clock_ticks)
    result = torch.where(done, torch.where(wiped, wipe_result, clock_result), -1)

    new_state = dataclasses.replace(
        state,
        x=x,
        y=y,
        heading_cos=heading_cos,
        heading_sin=heading_sin,
        speed=flew(flown.speed, state.speed),
        angle=flew(flown.angle, state.angle),
        angle_cos=flew(flown.angle_cos, state.angle_cos),
        angle_sin=flew(flown.angle_sin, state.angle_sin),
        nose_rate=flew(flown.nose_rate, state.nose_rate),
        energy=flew(flown.energy, state.energy),
        shield=torch.where(alive0, shield + restored, state.shield),
        shield_wait=torch.where(alive0, wait, state.shield_wait),
        cooldown=(cooldown - 1).clamp_min(0),
        next_slot=next_slot,
        alive=alive,
        bullet_x=bullet_x,
        bullet_y=bullet_y,
        bullet_dir_x=direction_x,
        bullet_dir_y=direction_y,
        bullet_speed=bullet_speed,
        bullet_moves_left=torch.where(hit.reshape(E, N, K), 0, torch.where(flying, moves_left0 - 1, moves_left0)),
        tick=tick,
    )
    out = StepOut(
        damage_dealt=damage_dealt,
        damage_taken=total,
        friendly_damage=friendly_damage,
        died=died,
        fired=fired,
        killer=torch.where(died, first_shooter, -1),
        done=done,
        result=result,
    )
    return new_state, out


# --------------------------------------------------------------------------- reset


def _edge_safe(config: GameConfig, x: Tensor, y: Tensor, angle: Tensor) -> tuple[Tensor, Tensor]:
    """(cos, sin) of a heading angle turned the least that leaves room to turn away from
    the boundary: within the spawn turn distance D of the edge, its angle off the outward
    normal is at least acos(d / D)."""
    distance = torch.sqrt(x * x + y * y)
    room = (config.arena_radius - distance) / card.SPAWN_TURN_DISTANCE
    least = torch.acos(room.clamp(0.0, 1.0))
    normal = torch.atan2(y, x)
    off_normal = torch.remainder(angle - normal + math.pi, 2.0 * math.pi) - math.pi
    off_normal = torch.copysign(torch.maximum(off_normal.abs(), least), off_normal)
    return torch.cos(normal + off_normal), torch.sin(normal + off_normal)


def _deal(config: GameConfig, envs: int, slots: int, team_sizes: Tensor, generator, device) -> State:
    """A fresh state of (E, N): team 0 in the first team_sizes[:, 0] slots, team 1 in the
    next team_sizes[:, 1], the rest not existing (parked at the centre at spawn speed)."""
    team_sizes = team_sizes.to(device=device, dtype=torch.long)
    slot = torch.arange(slots, device=device).expand(envs, slots)
    first, both = team_sizes[:, :1], team_sizes.sum(-1, keepdim=True)
    exists = slot < both
    team = ((slot >= first) & exists).long()
    rank = torch.where(team == 1, slot - first, slot)
    names = torch.tensor([card.TYPE_NAMES.index(name) for name in config.lineup], device=device)
    ship_type = torch.where(exists, names[rank % len(config.lineup)], 0)
    constants = ship_constants(config, ship_type)

    def uniform(*shape):
        return torch.rand(shape, generator=generator, device=device)

    radius_draw, angle_draw, heading_draw = uniform(envs, slots), uniform(envs, slots), uniform(envs, slots)
    axis = 2.0 * math.pi * uniform(envs, 1)
    polar = 2.0 * math.pi * angle_draw
    if config.spawn == "sides":
        side = (2 * team - 1).to(torch.float32)
        offset = card.SPAWN_OFFSET * config.arena_radius
        centre_x, centre_y = side * offset * torch.cos(axis), side * offset * torch.sin(axis)
        spread = card.SIDE_AREA_FRACTION * config.arena_radius * torch.sqrt(radius_draw)
        x, y = centre_x + spread * torch.cos(polar), centre_y + spread * torch.sin(polar)
        toward = torch.atan2(-centre_y - y, -centre_x - x)
        angle = toward + card.SIDE_HEADING_JITTER * (2.0 * heading_draw - 1.0)
    elif config.spawn == "mixed":
        spread = card.MIXED_AREA_FRACTION * config.arena_radius * torch.sqrt(radius_draw)
        x, y = spread * torch.cos(polar), spread * torch.sin(polar)
        angle = 2.0 * math.pi * heading_draw
    else:
        raise ValueError(f"spawn rule {config.spawn!r}: need 'sides' or 'mixed'")
    heading_cos, heading_sin = _edge_safe(config, x, y, angle)
    x, y = torch.where(exists, x, 0.0), torch.where(exists, y, 0.0)
    heading_cos, heading_sin = torch.where(exists, heading_cos, 1.0), torch.where(exists, heading_sin, 0.0)
    log_index, _, _ = medium(constants, x, y)
    zeros = torch.zeros((envs, slots), device=device)
    whole = torch.zeros((envs, slots), dtype=torch.long, device=device)
    bullets = torch.zeros((envs, slots, BULLET_SLOTS), device=device)
    return State(
        x=x,
        y=y,
        heading_cos=heading_cos,
        heading_sin=heading_sin,
        speed=card.SPAWN_SPEED * torch.exp(log_index),
        angle=zeros,
        angle_cos=torch.ones_like(zeros),
        angle_sin=zeros.clone(),
        nose_rate=zeros.clone(),
        energy=constants["spawn_energy"].clone(),
        shield=constants["max_shield"].clone(),
        shield_wait=zeros.clone(),
        cooldown=whole,
        next_slot=whole.clone(),
        alive=exists.clone(),
        exists=exists,
        team=team,
        type=ship_type,
        constants=constants,
        bullet_x=bullets,
        bullet_y=bullets.clone(),
        bullet_dir_x=torch.ones_like(bullets),
        bullet_dir_y=bullets.clone(),
        bullet_speed=bullets.clone(),
        bullet_moves_left=torch.zeros((envs, slots, BULLET_SLOTS), dtype=torch.long, device=device),
        tick=torch.zeros(envs, dtype=torch.long, device=device),
    )


def reset(config: GameConfig, envs: int, team_sizes: Tensor, generator: torch.Generator, device) -> State:
    """A fresh state. team_sizes: (E, 2) long, ships per team, each at least 1. N is the
    largest total over the environments; unused slots have exists = False."""
    slots = int(team_sizes.sum(-1).max())
    return _deal(config, envs, slots, team_sizes, generator, device)


def reset_done(config: GameConfig, state: State, done: Tensor, team_sizes: Tensor, generator: torch.Generator) -> State:
    """The state with the environments in done (E,) bool dealt afresh at team_sizes (E, 2),
    whose totals must fit the state's N. Every environment's draws are made, so nothing
    reads done on the host."""
    envs, slots = state.x.shape
    fresh = _deal(config, envs, slots, team_sizes, generator, state.x.device)

    def select(new: Tensor, old: Tensor) -> Tensor:
        return torch.where(done.reshape((envs,) + (1,) * (new.dim() - 1)), new, old)

    values = {}
    for field in dataclasses.fields(State):
        new, old = getattr(fresh, field.name), getattr(state, field.name)
        if field.name == "constants":
            values[field.name] = {
                key: old[key] if key.startswith("field_") else select(new[key], old[key]) for key in old
            }
        else:
            values[field.name] = select(new, old)
    return State(**values)


# --------------------------------------------------------------------------- observation

SPEED_SCALE = 400.0  # m/s
NOSE_RATE_SCALE = 3.0  # rad/s
ANGLE_SCALE = 0.27  # rad, about the stall angle
SHIELD_WAIT_SCALE = 3.0  # s, the recharge delay
COOLDOWN_CLIP = 0.5  # s
BOUNDARY_CLIP = 1000.0  # m
SHIELD_SCALE = 150.0


def observe(config: GameConfig, state: State, team: int) -> Obs:
    """Team `team`'s view of every slot, full information.

    ship (E, N, SHIP_FEATURES): 20 scalars [velocity x, y and speed over 400 m/s; angle of
    attack over 0.27 rad, its cos and sin; nose rate over 3 rad/s; energy and shield as
    fractions of the type's full values; shield over 150; shield wait over 3 s, clipped to
    1; gun cooldown in seconds clipped to 0.5; distance inside the boundary over 1000 m
    clipped to [-1, 1]; the outward unit direction x, y; heading cos, sin; position over the
    arena radius x, y; the share of bullet slots in flight], then flags [same team as the
    viewer, alive, exists, fired last tick], then the ship-type one-hot. Rows of slots that
    do not exist are zero. Fired last tick is read from the bullet ring (a bullet in the
    previous slot with one move spent), so a shot that hit on its first move is not flagged.
    """
    c = state.constants
    h = config.tick_length
    exists, alive = state.exists, state.alive
    velocity_x = state.speed * (state.heading_cos * state.angle_cos + state.heading_sin * state.angle_sin)
    velocity_y = state.speed * (state.heading_sin * state.angle_cos - state.heading_cos * state.angle_sin)
    distance = torch.sqrt(state.x * state.x + state.y * state.y)
    inverse_distance = 1.0 / distance.clamp_min(1.0)
    K = state.bullet_moves_left.shape[-1]
    previous = ((state.next_slot - 1) % K)[..., None]
    fired_last = state.bullet_moves_left.gather(-1, previous)[..., 0] == c["bullet_moves"] - 1
    same_team = state.team == team
    scalars = [
        velocity_x / SPEED_SCALE,
        velocity_y / SPEED_SCALE,
        state.speed / SPEED_SCALE,
        state.angle / ANGLE_SCALE,
        state.angle_cos,
        state.angle_sin,
        state.nose_rate / NOSE_RATE_SCALE,
        state.energy * c["inverse_max_energy"],
        state.shield / c["max_shield"],
        state.shield / SHIELD_SCALE,
        (state.shield_wait / SHIELD_WAIT_SCALE).clamp(0.0, 1.0),
        (state.cooldown.to(torch.float32) * h).clamp(0.0, COOLDOWN_CLIP),
        ((config.arena_radius - distance) / BOUNDARY_CLIP).clamp(-1.0, 1.0),
        state.x * inverse_distance,
        state.y * inverse_distance,
        state.heading_cos,
        state.heading_sin,
        state.x / config.arena_radius,
        state.y / config.arena_radius,
        (state.bullet_moves_left > 0).to(torch.float32).mean(-1),
    ]
    flags = [same_team, alive, exists, fired_last & alive]
    one_hot = torch.nn.functional.one_hot(state.type, SHIP_TYPES)
    ship = torch.cat(
        [torch.stack(scalars, -1), torch.stack(flags, -1).to(torch.float32), one_hot.to(torch.float32)], -1
    ) * exists[..., None]
    assert ship.shape[-1] == SHIP_FEATURES

    def alive_fraction(mask: Tensor) -> Tensor:
        return (alive & mask).sum(-1) / (exists & mask).sum(-1).clamp_min(1)

    globals_ = torch.stack(
        [state.tick.to(torch.float32) / config.clock_ticks, alive_fraction(same_team), alive_fraction(~same_team)], -1
    )
    assert globals_.shape[-1] == GLOBAL_FEATURES
    return Obs(
        ship=ship,
        pos=torch.stack([state.x, state.y], -1),
        heading=torch.atan2(state.heading_sin, state.heading_cos),
        key_mask=exists & alive,
        own=same_team & alive,
        globals=globals_,
    )
