"""Ship flight (game spec 3): the flight setting, the kick, and one kick-drift-kick tick.

`constants` holds what every type shares (Python numbers); `airframe` the flying ships'
own ship-card values (per-ship tensors). A parameter the card scales (`ship_types.SCALED`)
is the shared value times the ship's scale. `stick` and `power` are the stick level sigma
and the power mode pi as floats, not the action indices (`controls` converts).
"""

from typing import NamedTuple

import torch

from ..rules import POWER_MODES, STICK_LEVELS, FlightConstants
from ..ship_types import ShipTypes, per_ship
from ..state import Fields, Ships, replace
from .medium import medium
from .numerics import lookup, rotate, small_angle_cos_sin, smoothstep, softmin

SUSTAINED = sorted(level for level in STICK_LEVELS if level > 0.0)[1]  # the middle level


class Airframe(NamedTuple):
    """Each ship's flight parameters from the ship card, shaped like its ship_type."""

    max_lift_coefficient: torch.Tensor
    max_acceleration: torch.Tensor
    zero_lift_drag: torch.Tensor
    engine_power: torch.Tensor
    engine_power_full_store: torch.Tensor
    static_thrust: torch.Tensor
    max_energy: torch.Tensor
    corner_nose_frequency: torch.Tensor
    bleed_exponent: torch.Tensor
    induced_drag_scale: torch.Tensor
    stall_angle_scale: torch.Tensor
    coast_exchange_scale: torch.Tensor
    dive_exchange_scale: torch.Tensor
    climb_exchange_scale: torch.Tensor
    engine_knee_scale: torch.Tensor
    fine_stick: torch.Tensor
    sustained_stick: torch.Tensor
    corner_speed: torch.Tensor  # sqrt(max_acceleration / max_lift_coefficient)


def airframe_of(types: ShipTypes, ship_type: torch.Tensor) -> Airframe:
    """Each ship's Airframe from the ship card, shaped like ship_type, with its corner_speed (m/s)
    derived."""
    names = Airframe._fields[:-1]
    values = {name: per_ship(getattr(types, name), ship_type) for name in names}
    corner = torch.sqrt(values["max_acceleration"] / values["max_lift_coefficient"])
    return Airframe(**values, corner_speed=corner)


def controls(airframe: Airframe, power_index: torch.Tensor, stick_index: torch.Tensor):
    """Action indices to (stick level sigma, power mode pi), game spec 3.2. The stick's
    three levels each way (`rules.STICK_LEVELS` by index: straight, then fine, sustained
    and max, right then left) take each ship's own lift fractions: the card's fine_stick
    and sustained_stick, and 1."""
    level = lookup(STICK_LEVELS, stick_index)
    magnitude = torch.where(
        level.abs() == 1.0,
        1.0,
        torch.where(level.abs() == SUSTAINED, airframe.sustained_stick, airframe.fine_stick),
    )
    return torch.sign(level) * magnitude, lookup(POWER_MODES, power_index)


def engine_fall(constants: FlightConstants, airframe: Airframe, energy: torch.Tensor):
    """phi(E): 0 below the knee, rising smoothly to 1 at a full store."""
    fraction = energy / airframe.max_energy
    knee = constants.engine_knee * airframe.engine_knee_scale
    return smoothstep(((fraction - knee) / (1.0 - knee)).clamp(0.0, 1.0))


def stall_angle(constants: FlightConstants, airframe: Airframe) -> torch.Tensor:
    """alpha_s of each ship."""
    return constants.stall_angle * airframe.stall_angle_scale


def lift_coefficient(constants: FlightConstants, airframe: Airframe, angle: torch.Tensor):
    """The lift coefficient at angle of attack angle (rad): max_lift_coefficient (2r - r|r|), r the
    angle in stall angles clamped to [-2, 2]; it peaks at the stall angle and is zero from twice
    it."""
    ratio = (angle / stall_angle(constants, airframe)).clamp(-2.0, 2.0)
    return airframe.max_lift_coefficient * (2.0 * ratio - ratio * ratio.abs())


def lift_slope(constants: FlightConstants, airframe: Airframe, angle: torch.Tensor):
    """The lift coefficient's derivative in the angle of attack (1/rad), zero from twice the stall
    angle."""
    stall = stall_angle(constants, airframe)
    ratio = (angle / stall).abs()
    return torch.where(
        ratio < 2.0, (airframe.max_lift_coefficient / stall) * (2.0 - 2.0 * ratio), 0.0
    )


class Setting(NamedTuple):
    """Game spec 3.4: what the controls ask of the airframe, held over a half kick."""

    lateral_lift: torch.Tensor  # l (m/s^2)
    exchange: torch.Tensor  # a, the speed-for-store exchange (m/s^2)
    vertical_lift: torch.Tensor  # z (m/s^2)
    thrust: torch.Tensor  # T (m/s^2)
    charge: torch.Tensor | None = None  # c, the store's direct charge (W/kg); None without


def setting(constants: FlightConstants, airframe: Airframe, speed, energy, stick, power) -> Setting:
    """The Setting (game spec 3.4) at speed (m/s) and store energy, for stick level sigma and power
    mode pi as floats: the lateral lift within the airframe's budget, the exchange faded at the
    store's ends, the vertical lift and the thrust.

    With `direct_charge`, the engine's charge, the power above its null power (less as the
    store fills past the knee), goes into the store directly in every power mode, faded like
    the exchange in a turn and at the store's top; the thrust is the null power's, and the
    coast exchanges nothing. At cruise this flies as the coast exchange does; away from it the
    charge is a power, not an acceleration taken from the speed."""
    fall = engine_fall(constants, airframe, energy)
    dive = constants.dive_exchange * airframe.dive_exchange_scale  # a_d
    coast = (
        torch.zeros_like(fall)
        if constants.direct_charge
        else constants.coast_exchange * airframe.coast_exchange_scale * (1.0 - fall)
    )
    commanded = torch.where(
        power == 0.0,
        coast,
        torch.where(power > 0.0, constants.climb_exchange * airframe.climb_exchange_scale, -dive),
    )
    lift_reach = airframe.max_lift_coefficient * speed * speed
    budget = softmin(lift_reach, airframe.max_acceleration, constants.lift_softmin_exponent)  # L
    safe_budget = budget.clamp_min(1e-12)
    lateral_command = torch.sign(stick) * softmin(
        stick.abs() * lift_reach, airframe.max_acceleration, constants.lift_softmin_exponent
    )
    need = torch.sqrt(
        lateral_command * lateral_command + constants.gravity * constants.gravity
    )  # N
    flown = softmin(need, safe_budget, constants.lift_softmin_exponent) / need.clamp_min(1e-12)  # q
    lateral = lateral_command * flown
    lateral_share = (lateral.abs() / safe_budget).clamp(0.0, 1.0)  # chi
    level = torch.sqrt(1.0 - lateral_share * lateral_share)  # the lift's share left to climb
    exchange = commanded * level * flown - (1.0 - flown) * dive
    # Fade the exchange out at the ends of the store: charging stops at the maximum,
    # spending at 0.
    fade = constants.store_fade_band * airframe.max_energy
    headroom = ((airframe.max_energy - energy) / fade).clamp(0.0, 1.0)
    room = torch.where(exchange > 0.0, headroom, (energy / fade).clamp(0.0, 1.0))
    exchange = exchange * room
    if constants.direct_charge:
        charge = (airframe.engine_power - airframe.engine_power_full_store) * (1.0 - fall)
        thrust = softmin(airframe.engine_power_full_store / speed, airframe.static_thrust, 4.0)
        return Setting(
            lateral,
            exchange,
            constants.gravity * flown,
            thrust / constants.mass,
            charge * level * flown * headroom / constants.mass,
        )
    engine = (
        airframe.engine_power - (airframe.engine_power - airframe.engine_power_full_store) * fall
    )
    thrust = softmin(engine / speed, airframe.static_thrust, 4.0) / constants.mass
    return Setting(lateral, exchange, constants.gravity * flown, thrust)


def rates(
    constants: FlightConstants,
    airframe: Airframe,
    current: Setting,
    index,
    speed,
    lift,
    angle_cos,
    inverse_speed_squared,
):
    """(d speed / dt, d energy / dt) for a speed, lift coefficient, cos(angle of attack) and
    1/u^2 (game spec 3.4). `index` is the refractive index n."""
    induced = current.vertical_lift * inverse_speed_squared
    drag = (
        (
            zero_lift_drag(constants, airframe, speed)
            + constants.induced_drag_factor
            * airframe.induced_drag_scale
            * bleed(airframe, lift * lift + induced * induced)
        )
        * speed
        * speed
    )
    energy_rate = constants.mass * speed * current.exchange
    if current.charge is not None:
        energy_rate = energy_rate + constants.mass * current.charge
    return (
        (current.thrust * angle_cos - drag - current.exchange) / index,
        energy_rate / index,
    )


def zero_lift_drag(constants: FlightConstants, airframe: Airframe, speed: torch.Tensor):
    """The zero-lift drag coefficient (1/m) at a speed: the card's, times the drag rise
    1 + (u / u_c)^q when the rules give a critical speed u_c."""
    if constants.critical_speed <= 0.0:
        return airframe.zero_lift_drag
    rise = torch.pow(speed.abs() / constants.critical_speed, constants.drag_rise_exponent)
    return airframe.zero_lift_drag * (1.0 + rise)


def bleed(airframe: Airframe, lift_squared: torch.Tensor) -> torch.Tensor:
    """(C^2 + (z/u^2)^2)^(e/2) with the bleed exponent e (ship-card spec 2), written as
    s * s^(e/2 - 1) so that it is exactly s at e = 2 (pow(x, 0) is exactly 1). The clamp
    keeps s = 0 at 0 for any e."""
    return lift_squared * torch.pow(
        lift_squared.clamp_min(1e-30), 0.5 * airframe.bleed_exponent - 1.0
    )


def exponential_2x2(a11, a12, a21, a22, dt: float):
    """The normative 2x2 exponential of A dt: Horner series of exp(A dt/2), squared."""
    b11, b12, b21, b22 = (v * (dt / 2.0) for v in (a11, a12, a21, a22))
    h11, h12, h21, h22 = 1.0, 0.0, 0.0, 1.0  # innermost first: I + B/4, then I + (B/k)(...)
    for k in (4.0, 3.0, 2.0, 1.0):
        h11, h12, h21, h22 = (
            1.0 + (b11 * h11 + b12 * h21) / k,
            (b11 * h12 + b12 * h22) / k,
            (b21 * h11 + b22 * h21) / k,
            1.0 + (b21 * h12 + b22 * h22) / k,
        )
    return (
        h11 * h11 + h12 * h21,
        h11 * h12 + h12 * h22,
        h21 * h11 + h22 * h21,
        h21 * h12 + h22 * h22,
    )


def kick(
    constants: FlightConstants,
    airframe: Airframe,
    ship: Ships,
    index,
    gradient_nose_x,
    gradient_nose_y,
    stick,
    power,
    dt: float,
) -> Ships:
    """Advance speed, angle of attack, nose rate and energy by dt with the pose held (game
    spec 3.5, normative). gradient_nose_x, _y: grad ln n in the nose frame (gamma)."""
    speed0, angle0 = ship.proper_speed, ship.angle_of_attack
    cos0, sin0, energy = ship.angle_of_attack_cos, ship.angle_of_attack_sin, ship.energy
    safe_speed0 = speed0.clamp_min(constants.speed_floor)
    first = setting(constants, airframe, safe_speed0, energy, stick, power)
    lift0 = lift_coefficient(constants, airframe, angle0)
    speed_rate0, energy_rate0 = rates(
        constants, airframe, first, index, speed0, lift0, cos0, 1.0 / (safe_speed0 * safe_speed0)
    )

    speed = (speed0 + 0.5 * dt * speed_rate0).clamp_min(constants.speed_floor)  # mid speed
    current = setting(constants, airframe, speed, energy + 0.5 * dt * energy_rate0, stick, power)
    inverse_speed_squared = 1.0 / (speed * speed)
    path_rate = (lift0 * speed * speed + current.thrust * sin0) / (speed * index) + (
        speed / index
    ) * (cos0 * gradient_nose_y + sin0 * gradient_nose_x)
    path_slope = (
        lift_slope(constants, airframe, angle0) * speed * speed + current.thrust * cos0
    ) / (speed * index)  # kappa
    frequency = (airframe.corner_nose_frequency * speed / airframe.corner_speed).clamp_min(
        constants.nose_frequency_floor
    ) / index  # Omega
    reach = (
        current.lateral_lift.abs() * inverse_speed_squared / airframe.max_lift_coefficient
    ).clamp(0.0, 1.0)
    angle_step = (
        torch.sign(current.lateral_lift)
        * stall_angle(constants, airframe)
        * (1.0 - torch.sqrt(1.0 - reach))
        - angle0
    )

    # The nose oscillator, linearized about the commanded angle of attack, solved exactly.
    m11, m12, m21, m22 = exponential_2x2(
        -path_slope,
        1.0,
        -frequency * frequency + 2.0 * constants.nose_damping * frequency * path_slope,
        -2.0 * constants.nose_damping * frequency,
        dt,
    )
    steady_rate = path_slope * angle_step
    y1 = -angle_step
    y2 = (ship.nose_rate - path_rate) - steady_rate
    turn = angle_step + m11 * y1 + m12 * y2  # e
    nose_rate = path_rate + steady_rate + m21 * y1 + m22 * y2

    half_cos, half_sin = small_angle_cos_sin(0.5 * turn)
    mid_cos = cos0 * half_cos - sin0 * half_sin
    mid_lift = lift_coefficient(constants, airframe, angle0 + 0.5 * turn)
    speed_rate1, _ = rates(
        constants, airframe, current, index, speed0, mid_lift, mid_cos, inverse_speed_squared
    )
    speed_rate2, energy_rate2 = rates(
        constants,
        airframe,
        current,
        index,
        (speed0 + 0.5 * dt * speed_rate1).clamp_min(constants.speed_floor),
        mid_lift,
        mid_cos,
        inverse_speed_squared,
    )

    turn_cos, turn_sin = small_angle_cos_sin(turn)
    angle_cos, angle_sin = rotate(cos0, sin0, turn_cos, turn_sin)
    return replace(
        ship,
        proper_speed=(speed0 + dt * speed_rate2).clamp_min(constants.speed_floor),
        angle_of_attack=angle0 + turn,
        angle_of_attack_cos=angle_cos,
        angle_of_attack_sin=angle_sin,
        nose_rate=nose_rate,
        energy=torch.minimum((energy + dt * energy_rate2).clamp_min(0.0), airframe.max_energy),
    )


def nose_frame(ship: Ships):
    """grad ln n rotated into the nose frame (gamma_x, gamma_y)."""
    cos, sin = ship.heading_cos, ship.heading_sin
    gradient_x, gradient_y = ship.log_index_gradient_x, ship.log_index_gradient_y
    return cos * gradient_x + sin * gradient_y, cos * gradient_y - sin * gradient_x


def flight_tick(
    constants: FlightConstants,
    airframe: Airframe,
    tick_length: float,
    ship: Ships,
    stick,
    power,
    fields: Fields,
) -> Ships:
    """One tick of flight (game spec 3.6): kick, drift along the exact nose arc, re-sample
    the medium, kick. Changes only the flight variables."""
    ship = kick(
        constants,
        airframe,
        ship,
        torch.exp(ship.log_refractive_index),
        *nose_frame(ship),
        stick,
        power,
        0.5 * tick_length,
    )

    index = torch.exp(ship.log_refractive_index)
    gradient_nose_x, gradient_nose_y = nose_frame(ship)
    speed = ship.proper_speed / index
    velocity_x, velocity_y = speed * ship.angle_of_attack_cos, -speed * ship.angle_of_attack_sin
    # Correct the speed to the index at the step midpoint (spec: xi).
    correction = 0.5 * tick_length * (gradient_nose_x * velocity_x + gradient_nose_y * velocity_y)
    stretch = 1.0 + correction * (1.0 + correction * (0.5 + correction / 6.0))
    velocity_x, velocity_y = velocity_x / stretch, velocity_y / stretch
    arc = ship.nose_rate * tick_length  # the nose turns by this angle over the tick
    arc2 = arc * arc
    along = (1.0 - arc2 * (1.0 / 6.0 - arc2 / 120.0)) * tick_length  # A_1
    across = arc * (0.5 - arc2 * (1.0 / 24.0 - arc2 / 720.0)) * tick_length  # A_2
    body_x = velocity_x * along - velocity_y * across
    body_y = velocity_x * across + velocity_y * along
    cos, sin = ship.heading_cos, ship.heading_sin
    x, y = ship.x + (cos * body_x - sin * body_y), ship.y + (sin * body_x + cos * body_y)
    cos, sin = rotate(cos, sin, *small_angle_cos_sin(arc))
    norm = torch.rsqrt(cos * cos + sin * sin)
    angle_cos, angle_sin = ship.angle_of_attack_cos, ship.angle_of_attack_sin
    angle_norm = torch.rsqrt(angle_cos * angle_cos + angle_sin * angle_sin)
    log_index, gradient_x, gradient_y = medium(fields, x, y)
    ship = replace(
        ship,
        x=x,
        y=y,
        heading_cos=cos * norm,
        heading_sin=sin * norm,
        angle_of_attack_cos=angle_cos * angle_norm,
        angle_of_attack_sin=angle_sin * angle_norm,
        log_refractive_index=log_index,
        log_index_gradient_x=gradient_x,
        log_index_gradient_y=gradient_y,
    )
    return kick(
        constants,
        airframe,
        ship,
        torch.exp(log_index),
        *nose_frame(ship),
        stick,
        power,
        0.5 * tick_length,
    )
