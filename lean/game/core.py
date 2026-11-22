"""The game: reset, one tick, and the observation, as pure functions of tensors.

Sizes: E environments, N ship slots, K bullet slots per ship, F fields. Bullet o * K + k
belongs to ship o. Units: metres, seconds, radians, joules; clocks in ticks.

Tick order: fire, flight, bullet motion, swept hit test, damage and shields, boundary,
deaths, win check and clock, counters. Nothing here resets an environment: the caller
passes StepOut.done to reset_done.
"""

from __future__ import annotations

import dataclasses
import functools
import math
from dataclasses import dataclass
from types import SimpleNamespace

import torch
from torch import Tensor

from lean.contract import (
    BULLET_COLUMNS,
    BULLET_COUNT_COLUMNS,
    BULLET_SLOTS,
    FIELD_FEATURES,
    GLOBAL_FEATURES,
    SHIP_COLUMNS,
    SHIP_FEATURES,
    SHIP_TYPES,
    Constants,
    Obs,
    State,
    StepOut,
)

from . import card

# Rows of the packed state read by index here (lean.contract's *_COLUMNS).
X, Y = SHIP_COLUMNS.index("x"), SHIP_COLUMNS.index("y")
LOG_INDEX, GRADIENT_Y = SHIP_COLUMNS.index("log_index"), SHIP_COLUMNS.index("gradient_y")
BULLET_X, BULLET_Y = BULLET_COLUMNS.index("bullet_x"), BULLET_COLUMNS.index("bullet_y")
BULLET_DIR_X, BULLET_DIR_Y = BULLET_COLUMNS.index("bullet_dir_x"), BULLET_COLUMNS.index("bullet_dir_y")
BULLET_SPEED, BULLET_LOG_INDEX = BULLET_COLUMNS.index("bullet_speed"), BULLET_COLUMNS.index("bullet_log_index")
MOVES_LEFT = BULLET_COUNT_COLUMNS.index("bullet_moves_left")
FIRED_TICK = BULLET_COUNT_COLUMNS.index("bullet_fired_tick")


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
    # Flight model: "v2" (v2's flight, tested equal to v2) or "core" (the agreed new physics,
    # docs/rewrite/core-decisions.md "Flight physics"): weight always on the wing, the turn
    # limited unless diving; the engine drives speed only and never feeds the store; a store of
    # `store_loops` vertical loops at corner speed; dive and climb equal and opposite; a
    # smootherstep taper over `store_edge_band` of the store at each end; firing scales thrust
    # by `fire_thrust_fraction` on the ticks it fires and costs no store.
    # Fields drawn per environment at every reset (v2's random map, src/bnb/game/physics/
    # spawning.py new_map, real-scale elimination's values): `field_count` fields in
    # point-symmetric pairs (and one at the centre for an odd count), centres uniform in
    # `field_disc_fraction` of the arena, radius log-normal (median, sigma) clamped to
    # [min, max], all times the map scale (arena radius / 2,600 m), band `field_band_width`,
    # level -2, -1, +1 or +2 half-octaves. 0: the fixed `fields` above instead.
    field_count: int = 0
    field_disc_fraction: float = 0.85
    field_radius_median: float = 400.0
    field_radius_log_sigma: float = 0.6
    field_radius_min: float = 120.0
    field_radius_max: float = 1200.0
    field_band_width: float = 160.0
    field_high_share: float = 0.5
    field_strong_share: float = 0.5
    # Damage form: "v2" (v2's) or "core" (the agreed cheaper form with the same effects: the
    # head-on term written in the hit angle's cosine, (1 - c) / (1 - cos theta_0) in place of
    # (theta / theta_0)^2, which keeps the half-damage angle and differs from v2's curve by at
    # most 0.024; no atan2 or cross product). The closeness stays v2's: the hit test's own miss
    # distance (closest approach within the tick) read 6.8% less damage than v2's impact
    # parameter (the distance to the bullet's line), so it is not the same effect.
    # Ghost bullets (the hit head's labels, author 2026-11-22): every living ship puts a bullet
    # in its ring every tick; only a real shot costs, damages and kills; a ghost ends at its
    # first contact like a real bullet. Every bullet's outcome is reported when its flight ends.
    ghost_bullets: bool = False
    damage_model: str = "v2"
    flight_model: str = "v2"
    store_loops: float = 2.0
    store_edge_band: float = 0.05
    fire_thrust_fraction: float = 0.0
    # Fog of war (docs/rewrite/core-decisions.md): an enemy is visible in the team's sight (v2's
    # sight test: `sighted_by_enemy`) and on the tick it fires or is hit by a bullet; a hidden
    # enemy is its last-seen row. False: full information, exactly as before.
    fog: bool = False

    @classmethod
    def core(cls, **overrides) -> GameConfig:
        """The core game as decided so far: the new flight model, one ship (Vizia), v2's ten
        fields drawn per environment, and fog of war."""
        settings = {"flight_model": "core", "damage_model": "core", "ghost_bullets": True}
        settings.update(lineup=("vizia",), field_count=10, fog=True)
        return cls(**{**settings, **overrides})


# --------------------------------------------------------------------------- constants

# Constants counted in whole ticks: they must be equal for every type of a run.
INTEGER_CONSTANTS = ("bullet_moves", "fire_interval")
# Action-indexed tables (by stick index, by power index): always rows of the per-slot table.
ACTION_TABLES = ("stick_table", "commanded_table", "demand_table")


def _per_type(config: GameConfig) -> dict[str, Tensor]:
    """Every ship constant for every card type: float32 (T,), the action tables (T, width), the
    whole-tick counts long; computed with the game's own float32 arithmetic."""
    types = torch.arange(len(card.TYPE_NAMES))

    def of(name: str) -> Tensor:
        return torch.tensor(card.CARD[name], dtype=torch.float32)[types]

    lift, acceleration, max_energy = of("max_lift_coefficient"), of("max_acceleration"), of("max_energy")
    stall = card.STALL_ANGLE * of("stall_angle_scale")
    knee = card.ENGINE_KNEE * of("engine_knee_scale")
    climb = card.CLIMB_EXCHANGE * of("climb_exchange_scale")
    dive = card.DIVE_EXCHANGE * of("dive_exchange_scale")
    bullet_cost = of("bullet_energy_cost")
    spawn_energy = of("spawn_energy")
    fade_band = card.STORE_FADE_BAND
    if config.flight_model == "core":
        # The store: `store_loops` vertical loops at corner speed. A loop climbs twice the corner
        # turn radius, whose sideways acceleration is the g limit with 1 g held on the wing.
        corner_speed_squared = acceleration / lift
        corner_radius = corner_speed_squared / torch.sqrt(acceleration * acceleration - card.GRAVITY**2)
        max_energy = config.store_loops * card.GRAVITY * 2.0 * corner_radius
        spawn_energy = 0.5 * max_energy
        # Dive and climb equal and opposite: the mean of the card's two rates.
        dive = climb = 0.5 * (climb + dive)
        bullet_cost = torch.zeros_like(bullet_cost)
        fade_band = config.store_edge_band
    elif config.flight_model != "v2":
        raise ValueError(f"flight_model {config.flight_model!r}: need 'v2' or 'core'")
    fine, sustained = of("fine_stick"), of("sustained_stick")
    zeros, ones = torch.zeros_like(fine), torch.ones_like(fine)
    levels = torch.tensor(card.STICK_LEVELS)
    magnitude = torch.stack([zeros, fine, fine, sustained, sustained, ones, ones], -1)
    width = card.GRAVITY / 32.0
    raw = torch.stack([torch.full_like(climb, card.GRAVITY), card.GRAVITY - dive, card.GRAVITY + climb], -1)
    hit_radius = of("hit_radius")
    head_on_angle = card.HEAD_ON_ANGLE * of("head_on_angle_scale")
    values = {
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
        "inverse_fade": 1.0 / (fade_band * max_energy),
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
        "bullet_cost": bullet_cost,
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
        "inverse_one_minus_cos_head_on": 1.0 / (1.0 - torch.cos(head_on_angle)),
        "max_shield": of("max_shield"),
        "shield_rate": card.SHIELD_RECHARGE_RATE * of("shield_recharge_rate_scale"),
        "shield_delay": card.SHIELD_RECHARGE_DELAY * of("shield_recharge_delay_scale"),
        "spawn_energy": spawn_energy,
        "vision_range": card.VISION_RANGE * of("vision_range_scale"),
    }
    values.update(_derived_characteristics(config, values))
    return values


def _derived_characteristics(config: GameConfig, values: dict[str, Tensor]) -> dict[str, Tensor]:
    """Per ship type (card order): cruise speed (straight, coasting, thrust equal to drag),
    corner speed (the g limit meets the lift limit), bullet launch speed (cruise plus muzzle)
    and gun range (how far a bullet fired at cruise travels in its life under drag: v2's
    `api.gun_range`). Solved with the game's own flight functions, so the scripted opponent is
    told these, not guessed."""
    c = SimpleNamespace(**values)
    count = len(card.TYPE_NAMES)
    control = SimpleNamespace(
        model=config.flight_model, thrust_scale=1.0, stick_size=torch.zeros(count),
        stick_sign=torch.ones(count), commanded=c.commanded_table[:, 0],
        demand=c.demand_table[:, 0],
    )
    control.inverse_demand = 1.0 / control.demand
    low, high = torch.full((count,), 20.0), torch.full((count,), 1000.0)
    with torch.no_grad():
        for _ in range(60):  # bisection on d speed / dt = 0
            middle = 0.5 * (low + high)
            setting = _setting(c, control, middle, c.spawn_energy)
            one, zero = torch.ones_like(middle), torch.zeros_like(middle)
            rate, _ = _rates(c, setting, one, middle, zero, one, 1.0 / (middle * middle))
            low, high = torch.where(rate > 0.0, middle, low), torch.where(rate > 0.0, high, middle)
    cruise = 0.5 * (low + high)
    launch = cruise + c.muzzle_speed
    life = c.bullet_moves.float() * config.tick_length
    return {
        "cruise_speed": cruise,
        "corner_speed": torch.sqrt(c.max_acceleration / c.lift),
        "bullet_launch_speed": launch,
        "gun_range": torch.log1p(c.bullet_drag * launch * life) / c.bullet_drag,
    }


@dataclass(frozen=True, eq=False)
class ShipCard:
    """The run's per-ship constants before they are dealt to slots: the uniform ones as Python
    numbers, the rest as a per-type table (C, T) float32 on the CPU whose rows `columns` names."""

    uniform: dict[str, float | int]
    columns: dict[str, tuple[int, int]]
    table: Tensor


@functools.cache
def _ship_card(flight_model: str, store_loops: float, store_edge_band: float, tick_length: float,
               lineup: tuple[str, ...]) -> ShipCard:
    config = GameConfig(flight_model=flight_model, store_loops=store_loops, store_edge_band=store_edge_band,
                        tick_length=tick_length, lineup=lineup)
    used = torch.tensor([card.TYPE_NAMES.index(name) for name in lineup])
    uniform, columns, rows = {}, {}, []
    for name, values in _per_type(config).items():
        chosen = values[used]
        if name not in ACTION_TABLES and bool((chosen == chosen[0]).all()):
            uniform[name] = chosen[0].item()
            continue
        if name in INTEGER_CONSTANTS:
            raise ValueError(f"{name} differs between the lineup's types {lineup}; it must be equal")
        block = values.reshape(len(card.TYPE_NAMES), -1).T  # (width, T)
        columns[name] = (sum(len(row) for row in rows), block.shape[0])
        rows.append(block)
    return ShipCard(uniform, columns, torch.cat(rows, 0).contiguous())


def ship_card(config: GameConfig) -> ShipCard:
    """The run's ship card, built once per flight settings and lineup."""
    return _ship_card(config.flight_model, config.store_loops, config.store_edge_band, config.tick_length,
                      tuple(config.lineup))


@functools.cache
def _card_table(ship_card_: ShipCard, device: torch.device) -> Tensor:
    return ship_card_.table.to(device)


def ship_constants(config: GameConfig, ship_type: Tensor) -> Constants:
    """The per-ship constants of slots of types ship_type (E, N) long, each a type of the lineup:
    the card's uniform numbers, and its per-type rows gathered once into (E, C, N)."""
    ship_card_ = ship_card(config)
    table = _card_table(ship_card_, ship_type.device)
    envs, slots = ship_type.shape
    rows = table.shape[0]
    gathered = table[None].expand(envs, rows, -1).gather(2, ship_type[:, None].expand(envs, rows, slots))
    return Constants(ship_card_.uniform, ship_card_.columns, gathered)


# --------------------------------------------------------------------------- numerics


def minimum(a: Tensor, b: Tensor | float) -> Tensor:
    """torch.minimum for a constant that may be a Python number."""
    return torch.minimum(a, b) if isinstance(b, Tensor) else a.clamp_max(b)


def maximum(a: Tensor, b: Tensor | float) -> Tensor:
    """torch.maximum for a constant that may be a Python number."""
    return torch.maximum(a, b) if isinstance(b, Tensor) else a.clamp_min(b)


def softmin(a: Tensor, b: Tensor | float, exponent: float) -> Tensor:
    """low (1 + (low / high)^q)^(-1/q) for a, b >= 0: a smooth minimum."""
    low, high = minimum(a, b), maximum(a, b)
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


FIELD_X, FIELD_Y, FIELD_RADIUS, FIELD_BAND, FIELD_TARGET = range(5)  # rows of State.fields


def medium(fields: Tensor, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """(ln n, d/dx ln n, d/dy ln n) at points (E, ...) for each environment's fields (E, 5, F).
    With no fields the medium is exactly 0."""
    count = fields.shape[-1]
    if count == 0:
        zeros = torch.zeros_like(x)
        return zeros, zeros, zeros
    shape = (fields.shape[0],) + (1,) * (x.dim() - 1) + (count,)
    field_x, field_y, radius, band, target = (fields[:, row].reshape(shape) for row in range(5))
    offset_x, offset_y = x[..., None] - field_x, y[..., None] - field_y
    distance = torch.sqrt(offset_x * offset_x + offset_y * offset_y)
    depth = 0.5 - (distance - radius) / band
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
    core = getattr(control, "model", "v2") == "core"
    t = ((energy * p.inverse_max_energy - p.knee) * p.inverse_above_knee).clamp(0.0, 1.0)
    fall = t * t * (3.0 - 2.0 * t)
    reach = p.lift * speed * speed
    q = card.LIFT_SOFTMIN_EXPONENT
    budget = softmin(reach, p.max_acceleration, q).clamp_min(1e-12)
    command = softmin(control.stick_size * reach, p.max_acceleration, q)
    sink_room = (energy * p.inverse_fade).clamp(0.0, 1.0)
    budget_squared = budget * budget
    demand, commanded = control.demand, control.commanded
    if core:
        # Weight always on the wing: the vertical demand is held first and the turn gets the rest
        # (dive lowers the demand and frees lift for the turn). Smootherstep edges of the store;
        # at an edge, dive or climb fades to coast in its effect on the wing as well as in the
        # trade (no free turn from diving on the deck, no lost turn from climbing at the top).
        sink_room = smootherstep(sink_room)
        top_room = smootherstep(((p.max_energy - energy) * p.inverse_fade).clamp(0.0, 1.0))
        room = torch.where(commanded > 0.0, top_room, torch.where(commanded < 0.0, sink_room, 1.0))
        demand = card.GRAVITY + (demand - card.GRAVITY) * room
        commanded = commanded * room
        held = softmin(demand, budget, q)
    else:
        held = softmin(demand * (1.0 - sink_room), budget, q)
    lateral_size = softmin(command, torch.sqrt((budget_squared - held * held).clamp_min(0.0)), q)
    if core:
        vertical = held  # held first; the turn took only what was left
    else:
        vertical = softmin(control.demand, torch.sqrt((budget_squared - lateral_size * lateral_size).clamp_min(0.0)), q)
    share = (lateral_size / budget).clamp(0.0, 1.0)
    headroom = ((p.max_energy - energy) * p.inverse_fade).clamp(0.0, 1.0)
    if core:
        # The commanded trade is already faded at the edges; a shortfall in the vertical (the
        # wing cannot hold the demand) sinks the ship, and that needs store to sink.
        exchange = commanded - (demand - vertical).clamp_min(0.0) * sink_room
    else:
        exchange = commanded - (demand - vertical)
        exchange = exchange * torch.where(exchange > 0.0, headroom, sink_room)
    thrust = softmin(p.power_full_store / speed, p.static_thrust, card.THRUST_SOFTMIN_EXPONENT)
    if core:
        # The engine drives speed only (constant power, the card's full-store power) and never
        # feeds the store; firing scales the thrust on the ticks it fires.
        charge = torch.zeros_like(exchange)
        thrust = thrust * getattr(control, "thrust_scale", 1.0)
    else:
        charge = (
            p.power_surplus * (1.0 - fall) * torch.sqrt(1.0 - share * share)
            * (vertical * control.inverse_demand) * headroom
        )
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
    # The lift coefficient's slope: negative past the stall, zero from twice the stall angle (v2's
    # lift_slope, src/bnb/game/physics/flight.py).
    slope = torch.where(ratio < 2.0, p.lift_over_stall * (2.0 - 2.0 * ratio), 0.0)
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
    s.energy = minimum((energy + duration * energy_rate2).clamp_min(0.0), p.max_energy)
    return s


def flight_tick(
    constants: Constants,
    fields: Tensor,
    tick_length: float,
    s: SimpleNamespace,
    stick: Tensor,
    power: Tensor,
    model: str = "v2",
    thrust_scale: Tensor | float = 1.0,
):
    """One tick of flight for every slot (E, N): half a kick, a drift, half a kick.

    s carries x, y, heading_cos, heading_sin, speed, angle, angle_cos, angle_sin, nose_rate,
    energy and the medium at the ship (log_index, gradient_x, gradient_y); it is updated in
    place and returned. fields: (E, 5, F). stick, power: (E, N) long action indices.
    """
    p = constants
    control = SimpleNamespace(model=model, thrust_scale=thrust_scale)
    signed = p.stick_table.gather(1, stick[:, None])[:, 0]
    control.stick_size, control.stick_sign = signed.abs(), torch.sign(signed)
    control.commanded = p.commanded_table.gather(1, power[:, None])[:, 0]
    control.demand = p.demand_table.gather(1, power[:, None])[:, 0]
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
    s.log_index, s.gradient_x, s.gradient_y = medium(fields, s.x, s.y)
    return _kick(p, control, s, torch.exp(-s.log_index), *_nose_frame(s), 0.5 * h)


# --------------------------------------------------------------------------- bullets


def _half_bend(tick_length, direction_x, direction_y, speed, index, gradient_x, gradient_y):
    """The direction turned toward the higher index by half a tick."""
    turn = 0.5 * tick_length * (speed / index) * (direction_x * gradient_y - direction_y * gradient_x)
    cos, sin = small_angle_cos_sin(turn)
    return direction_x * cos - direction_y * sin, direction_x * sin + direction_y * cos


def bullet_motion(fields, tick_length, x, y, direction_x, direction_y, speed, drag, start_medium):
    """One move of every bullet (E, ...): half drag, half bend, the step, half bend, half
    drag. fields: (E, 5, F). drag: the shooter's drag constant, 1/m, a number or broadcast
    against the bullets. start_medium: (ln n, d/dx ln n, d/dy ln n) at the start (cached).
    Returns x, y, direction x and y, speed and the medium at the end (three tensors)."""
    log_index, gradient_x, gradient_y = start_medium
    index = torch.exp(log_index)
    half = drag * 0.5 * tick_length
    speed = speed / (1.0 + half * speed / index)
    direction_x, direction_y = _half_bend(tick_length, direction_x, direction_y, speed, index, gradient_x, gradient_y)
    through = speed / index
    correction = 0.5 * tick_length * through * (direction_x * gradient_x + direction_y * gradient_y)
    step = through * tick_length / (1.0 + correction * (1.0 + correction * (0.5 + correction / 6.0)))
    x, y = x + direction_x * step, y + direction_y * step
    log_index, gradient_x, gradient_y = medium(fields, x, y)
    index = torch.exp(log_index)
    direction_x, direction_y = _half_bend(tick_length, direction_x, direction_y, speed, index, gradient_x, gradient_y)
    speed = speed / (1.0 + half * speed / index)
    return x, y, direction_x, direction_y, speed, log_index, gradient_x, gradient_y


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
    E, _, N, K = state.bullets.shape
    device = state.ships.device
    stick, power, shoot = action[..., 0], action[..., 1], action[..., 2] == 1
    alive0 = state.alive

    def of_ship(values: Tensor | float) -> Tensor | float:  # a per-ship constant against (E, N, K)
        return values[..., None] if isinstance(values, Tensor) else values

    # Fire, from the tick-start state.
    fired = alive0 & shoot & (state.cooldown == 0) & (state.energy >= c["bullet_cost"])
    path_x = state.speed * (state.heading_cos * state.angle_cos + state.heading_sin * state.angle_sin)
    path_y = state.speed * (state.heading_sin * state.angle_cos - state.heading_cos * state.angle_sin)
    spread = noise["spread"]
    launch_x = path_x + c["muzzle_speed"] * state.heading_cos + c["spread_speed"] * spread[..., 0]
    launch_y = path_y + c["muzzle_speed"] * state.heading_sin + c["spread_speed"] * spread[..., 1]
    launch_speed = torch.sqrt(launch_x * launch_x + launch_y * launch_y)
    slot = torch.arange(K, device=device)
    emits = alive0 if config.ghost_bullets else fired  # a ghost every tick, or real shots only
    written = emits[..., None] & (slot == state.next_slot[..., None])
    # A new bullet starts at its ship, with the ship's cached medium; the others keep theirs.
    launched = torch.cat(
        [state.ships[:, X : Y + 1], torch.stack([launch_x / launch_speed, launch_y / launch_speed, launch_speed], 1),
         state.ships[:, LOG_INDEX : GRADIENT_Y + 1]],
        1,
    )
    bullets0 = torch.where(written[:, None], launched[..., None], state.bullets)
    counts0 = torch.where(
        written[:, None],
        torch.stack([torch.full_like(state.tick, c["bullet_moves"]), state.tick], 1)[:, :, None, None],
        state.bullet_counts,
    )
    bullet_real = torch.where(written, fired[..., None], state.bullet_real)
    energy = torch.where(fired, state.energy - c["bullet_cost"], state.energy)
    cooldown = torch.where(fired, c["fire_interval"], state.cooldown)
    next_slot = torch.where(emits, (state.next_slot + 1) % K, state.next_slot)

    # Flight; slots not alive at tick start keep everything. In the core model firing cuts thrust.
    core_model = config.flight_model == "core"
    firing_thrust = torch.where(fired, config.fire_thrust_fraction, 1.0) if core_model else 1.0
    flown = flight_tick(
        c, state.fields, h,
        SimpleNamespace(
            x=state.x, y=state.y, heading_cos=state.heading_cos, heading_sin=state.heading_sin,
            speed=state.speed, angle=state.angle, angle_cos=state.angle_cos, angle_sin=state.angle_sin,
            nose_rate=state.nose_rate, energy=energy,
            log_index=state.log_index, gradient_x=state.gradient_x, gradient_y=state.gradient_y,
        ),
        stick, power,
        model=config.flight_model,
        thrust_scale=firing_thrust,
    )

    def flew(new: Tensor, old: Tensor) -> Tensor:
        return torch.where(alive0, new, old)

    x, y = flew(flown.x, state.x), flew(flown.y, state.y)
    heading_cos, heading_sin = flew(flown.heading_cos, state.heading_cos), flew(flown.heading_sin, state.heading_sin)

    # Bullet motion; inactive bullets are frozen.
    flying = counts0[:, MOVES_LEFT] > 0
    x0, y0, direction_x0, direction_y0, speed0 = bullets0[:, :BULLET_LOG_INDEX].unbind(1)
    moved = bullet_motion(
        state.fields, h, x0, y0, direction_x0, direction_y0, speed0, of_ship(c["bullet_drag"]),
        bullets0[:, BULLET_LOG_INDEX:].unbind(1),
    )
    bullets = torch.where(flying[:, None], torch.stack(moved, 1), bullets0)
    direction_x, direction_y, bullet_speed = bullets[:, BULLET_DIR_X : BULLET_SPEED + 1].unbind(1)

    # The swept test between the two motions, every bullet against every ship: (E, 2, N K, N),
    # x and y on the second axis.
    per_bullet = (E, 2, N * K, 1)
    start = bullets0[:, BULLET_X : BULLET_Y + 1].reshape(per_bullet) - state.ships[:, X : Y + 1, None, :]
    moved_xy = torch.stack([x, y], 1) - state.ships[:, X : Y + 1]
    relative = (bullets[:, BULLET_X : BULLET_Y + 1] - bullets0[:, BULLET_X : BULLET_Y + 1]).reshape(per_bullet)
    relative = relative - moved_xy[:, :, None, :]
    relative_squared = (relative * relative).sum(1).clamp_min(1e-12)
    tau = (-(start * relative).sum(1) / relative_squared).clamp(0.0, 1.0)
    miss = start + tau[:, None] * relative
    miss_squared = (miss * miss).sum(1)
    ship = torch.arange(N, device=device)
    owner = ship.repeat_interleave(K)
    reach = torch.where(alive0, c["hit_radius_squared"], -1.0)
    contact = (miss_squared < reach[:, None, :]) & (owner[:, None] != ship) & flying.reshape(E, N * K, 1)

    # First contact: the smallest tau, ties within 1e-6 to the closer pass.
    infinity = torch.inf
    contact_tau = torch.where(contact, tau, infinity)
    first_tau = contact_tau.min(-1).values
    tied = contact & (contact_tau <= first_tau[..., None] + 1e-6)
    target = torch.where(tied, miss_squared, infinity).argmin(-1)
    hit = first_tau < infinity

    # Damage per bullet on its target (the target is slot 0 and unused for a miss). A constant
    # equal for every ship is a number and needs no gather.
    def of_target(values: Tensor | float) -> Tensor | float:
        return values.gather(1, target) if isinstance(values, Tensor) else values

    flat = (E, N * K)
    flat_direction_x, flat_direction_y = direction_x.reshape(flat), direction_y.reshape(flat)
    nose_cos, nose_sin = of_target(heading_cos), of_target(heading_sin)
    core_damage = config.damage_model == "core"
    if not core_damage:
        theta = torch.atan2(
            (flat_direction_x * nose_sin - flat_direction_y * nose_cos).abs(),
            -(flat_direction_x * nose_cos + flat_direction_y * nose_sin),
        )
    # The owner's constants broadcast over its K bullets.
    speed_ratio = bullet_speed * of_ship(c["inverse_reference_bullet_speed"])
    explosive = of_ship(c["explosive_share"])
    full = (of_ship(c["hit_damage"]) * (explosive + (1.0 - explosive) * speed_ratio * speed_ratio)).reshape(flat)
    # The swept test's start and relative motion against the target.
    on_target = target[:, None, :, None].expand(E, 2, -1, 1)
    target_start_x, target_start_y = start.gather(3, on_target)[..., 0].unbind(1)
    target_relative_x, target_relative_y = relative.gather(3, on_target)[..., 0].unbind(1)
    cross = target_start_x * target_relative_y - target_start_y * target_relative_x
    line = cross * cross / relative_squared.gather(2, target[..., None])[..., 0]
    impact_squared = torch.minimum(line, target_start_x * target_start_x + target_start_y * target_start_y)
    radius, core = of_target(c["hit_radius"]), of_target(c["core_radius"])
    band_width = (radius - core).clamp_min(1e-6) if isinstance(radius - core, Tensor) else max(radius - core, 1e-6)
    band = ((impact_squared.sqrt() - core) / band_width).clamp(0.0, 1.0)
    closeness = 1.0 - band * band * (3.0 - 2.0 * band)
    head_on = of_target(c["head_on"])
    if core_damage:
        # cos of the hit angle (1 head-on); (1 - c) / (1 - cos theta_0) stands for
        # (theta / theta_0)^2.
        hit_cos = -(flat_direction_x * nose_cos + flat_direction_y * nose_sin)
        inverse = of_target(c["inverse_one_minus_cos_head_on"])
        squared_ratio = ((1.0 - hit_cos) * inverse).clamp_min(0.0)
        powered = squared_ratio.pow(0.5 * of_target(c["head_on_sharpness"]))
    else:
        ratio = theta * of_target(c["inverse_head_on_angle"])
        powered = ratio.pow(of_target(c["head_on_sharpness"]))
    aspect = 1.0 - (1.0 - head_on) * torch.exp(-card.LN_2 * powered)
    real = bullet_real.reshape(flat)
    bullet_damage = torch.where(hit & real, full * closeness * aspect, 0.0)

    enemy = state.team[:, owner] != of_target(state.team)
    hit_taken = torch.zeros((E, N), dtype=bullet_damage.dtype, device=device).scatter_add(1, target, bullet_damage)
    damage_dealt = torch.where(enemy, bullet_damage, 0.0).reshape(E, N, K).sum(-1)
    friendly_damage = torch.where(enemy, 0.0, bullet_damage).reshape(E, N, K).sum(-1)

    # The earliest hit on each ship names its killer: smallest tau, then lowest bullet index.
    real_hit = hit & real
    hit_tau = torch.where(real_hit, first_tau, infinity)
    earliest_tau = torch.full((E, N), infinity, device=device).scatter_reduce(1, target, hit_tau, "amin")
    earliest = real_hit & (hit_tau <= earliest_tau.gather(1, target))
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
    # Clock rule (v2's clock_result, src/bnb/game/modes/phase.py): the team with more health over
    # its living ships wins, equal health a draw; health is the shield after this tick's recharge.
    # A lead within float32 rounding of the sums is a draw: the two sums add their ships in
    # different orders, so an exact tie can differ by a rounding residue.
    health = torch.where(alive, (shield + restored) / c["max_shield"], 0.0)
    left_health_zero = torch.where(team_one, 0.0, health).sum(-1)
    left_health_one = torch.where(team_one, health, 0.0).sum(-1)
    lead = left_health_zero - left_health_one
    tie = lead.abs() <= 1e-5 * (left_health_zero + left_health_one).clamp_min(1.0)
    clock_result = torch.where(tie, 2, torch.where(lead > 0.0, 0, 1))
    tick = state.tick + 1
    done = wiped | (tick >= config.clock_ticks)
    result = torch.where(done, torch.where(wiped, wipe_result, clock_result), -1)

    # Every bullet whose flight ends this tick: its first contact, or its last move without one.
    hit_slots = hit.reshape(E, N, K)
    moves_left0 = counts0[:, MOVES_LEFT]
    moves_left = torch.where(hit_slots, 0, torch.where(flying, moves_left0 - 1, moves_left0))
    ended = hit_slots | (flying & (moves_left == 0))
    outcome = torch.where(hit_slots, torch.where(enemy.reshape(E, N, K), 1, 2), 0)
    shot_outcome = torch.where(ended, outcome, -1)
    fired_tick = counts0[:, FIRED_TICK]

    ships = torch.stack(
        [
            x, y, heading_cos, heading_sin, flew(flown.speed, state.speed), flew(flown.angle, state.angle),
            flew(flown.angle_cos, state.angle_cos), flew(flown.angle_sin, state.angle_sin),
            flew(flown.nose_rate, state.nose_rate), flew(flown.energy, state.energy),
            flew(shield + restored, state.shield), flew(wait, state.shield_wait), (cooldown - 1).clamp_min(0),
            flew(flown.log_index, state.log_index), flew(flown.gradient_x, state.gradient_x),
            flew(flown.gradient_y, state.gradient_y), next_slot,
        ],
        1,
    )
    new_state = dataclasses.replace(
        state,
        ships=ships,
        flags=torch.stack([alive, state.exists, fired], 1),
        bullets=bullets,
        bullet_counts=torch.stack([moves_left, fired_tick], 1),
        bullet_real=bullet_real,
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
        shot_outcome=shot_outcome,
        shot_tick=fired_tick,
        hit_damage=torch.zeros((E, N * N), dtype=bullet_damage.dtype, device=device)
        .scatter_add(1, owner[None, :].expand(E, -1) * N + target, bullet_damage)
        .reshape(E, N, N),
        boundary_damage=boundary,
        shield_restored=torch.where(alive0, restored, 0.0),
    )
    if config.fog:
        # A shot or a real bullet's hit reveals the ship for this tick (boundary damage does not).
        struck = earliest_bullet < no_bullet
        new_state = perceive(new_state, fired | struck)
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


MAP_SCALE_RADIUS = 2600.0  # m: the arena radius of map scale 1 (v2's rules.py)


def draw_fields(config: GameConfig, envs: int, generator, device) -> Tensor:
    """Each environment's fields, (E, 5, F) (v2's new_map; rows as State.fields): point-symmetric pairs,
    field P + i is field i rotated by 180 degrees about the centre, and for an odd count the
    last field sits at the centre."""
    count, pairs = config.field_count, config.field_count // 2
    scale = config.arena_radius / MAP_SCALE_RADIUS
    uniforms = torch.rand((envs, count, 4), generator=generator, device=device)
    drawn = torch.cat([uniforms[:, :pairs], uniforms[:, :pairs], uniforms[:, 2 * pairs :]], 1)
    disc = config.field_disc_fraction * config.arena_radius
    distance = disc * torch.sqrt(uniforms[:, :pairs, 0])
    angle = 2.0 * math.pi * uniforms[:, :pairs, 1]
    x, y = distance * torch.cos(angle), distance * torch.sin(angle)
    centre = torch.zeros((envs, count - 2 * pairs), device=device)
    z = math.sqrt(2.0) * torch.erfinv(2.0 * drawn[..., 2].clamp(1e-6, 1.0 - 1e-6) - 1.0)
    radius = config.field_radius_median * torch.exp(config.field_radius_log_sigma * z)
    radius = scale * radius.clamp(config.field_radius_min, config.field_radius_max)
    high, strong = config.field_high_share, config.field_strong_share
    edges = ((1.0 - high) * strong, 1.0 - high, 1.0 - high + high * (1.0 - strong))
    index = sum((drawn[..., 3] >= edge).long() for edge in edges)
    level = torch.tensor([-2.0, -1.0, 1.0, 2.0], device=device)[index]
    return torch.stack(
        [
            torch.cat([x, -x, centre], 1),
            torch.cat([y, -y, centre], 1),
            radius,
            torch.full_like(radius, config.field_band_width),
            level * card.LN_SQRT2,
        ],
        1,
    )


@functools.cache
def _fixed_fields(fields: tuple[tuple[float, float, float, float, float], ...], device: torch.device) -> Tensor:
    table = torch.tensor(fields, dtype=torch.float32).reshape(-1, 5).T.contiguous()
    table[FIELD_TARGET] *= card.LN_SQRT2
    return table.to(device)


def fixed_fields(config: GameConfig, envs: int, device) -> Tensor:
    """The config's fixed fields as State.fields (E, 5, F) (an expanded view; the level turned
    into the target ln n)."""
    table = _fixed_fields(config.fields, torch.device(device))
    return table.expand(envs, *table.shape)


def _per_slot(value: Tensor | float, like: Tensor) -> Tensor:
    """A per-ship constant as a fresh (E, N) tensor."""
    return value.clone() if isinstance(value, Tensor) else torch.full_like(like, value)


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
    # Slots that do not exist take the lineup's first type, so a constant equal for the lineup
    # stays one number.
    ship_type = names[torch.where(exists, rank % len(config.lineup), 0)]
    constants = ship_constants(config, ship_type)
    if config.field_count > 0:
        fields = draw_fields(config, envs, generator, device)
    else:
        fields = fixed_fields(config, envs, device)

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
    log_index, gradient_x, gradient_y = medium(fields, x, y)
    zeros = torch.zeros((envs, slots), device=device)
    ones = torch.ones((envs, slots), device=device)
    named = {
        "x": x, "y": y, "heading_cos": heading_cos, "heading_sin": heading_sin,
        "speed": card.SPAWN_SPEED * torch.exp(log_index), "angle": zeros, "angle_cos": ones, "angle_sin": zeros,
        "nose_rate": zeros, "energy": _per_slot(constants["spawn_energy"], zeros),
        "shield": _per_slot(constants["max_shield"], zeros), "shield_wait": zeros, "cooldown": zeros,
        "log_index": log_index, "gradient_x": gradient_x, "gradient_y": gradient_y, "next_slot": zeros,
    }
    bullets = torch.zeros((envs, len(BULLET_COLUMNS), slots, BULLET_SLOTS), device=device)
    bullets[:, BULLET_DIR_X] = 1.0
    state = State(
        ships=torch.stack([named[name] for name in SHIP_COLUMNS], 1),
        flags=torch.stack([exists, exists, torch.zeros_like(exists)], 1),  # alive, exists, fired
        identity=torch.stack([team, ship_type], 1),
        constants=constants,
        bullets=bullets,
        bullet_counts=torch.zeros((envs, len(BULLET_COUNT_COLUMNS), slots, BULLET_SLOTS), dtype=torch.long,
                                  device=device),
        bullet_real=torch.zeros((envs, slots, BULLET_SLOTS), dtype=torch.bool, device=device),
        tick=torch.zeros(envs, dtype=torch.long, device=device),
        fields=fields,
    )
    if not config.fog:
        return state
    # Nothing seen yet: every row unknown and taken as alive until seen dead. Ships in sight at
    # the spawn are seen at once (no other reveal at a spawn).
    seen = torch.zeros((envs, len(SEEN_COLUMNS), slots), device=device)
    seen[:, SEEN_ALIVE] = exists.to(torch.float32)
    state = dataclasses.replace(
        state, seen=seen, ticks_since_seen=torch.zeros((envs, slots), dtype=torch.long, device=device)
    )
    return perceive(state, torch.zeros_like(exists))


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
    fresh = _deal(config, envs, slots, team_sizes, generator, state.ships.device)

    def select(new: Tensor, old: Tensor) -> Tensor:
        return torch.where(done.reshape((envs,) + (1,) * (new.dim() - 1)), new, old)

    values = {}
    for field in dataclasses.fields(State):
        new, old = getattr(fresh, field.name), getattr(state, field.name)
        if new is None:  # fog's fields with fog off
            values[field.name] = None
        elif field.name == "constants":
            values[field.name] = dataclasses.replace(old, table=select(new.table, old.table))
        else:
            values[field.name] = select(new, old)
    return State(**values)


# --------------------------------------------------------------------------- fog

# The raw values a team keeps of an enemy ship from its last sighting (State.seen's columns): the
# ship's row of the packed state as it was, then its share of bullet slots in flight, alive and
# known.
SEEN_COLUMNS = SHIP_COLUMNS + ("bullet_share", "alive", "known")
SEEN_SHARE = SEEN_COLUMNS.index("bullet_share")
SEEN_ALIVE = SEEN_COLUMNS.index("alive")
SEEN_KNOWN = SEEN_COLUMNS.index("known")
TICKS_SINCE_SEEN_CAP = 1 << 30  # v2's cap (src/bnb/game/perception/observe.py)


def sighted_by_enemy(state: State) -> Tensor:
    """(E, N) bool: some living ship of the other team sees the ship (v2's sight test,
    src/bnb/game/perception/sight.py `sighted`, game spec 10.1). Sight is all round, out to the
    observer's vision range (d^2 <= V^2), and blocked only by fields' cores: field i, with core
    radius max(r - w / 2, 0) and outer radius r + w / 2, blocks the line when exactly one end is
    strictly inside its core and the two ends are not both inside its outer disc (core and band
    see each other), or when neither end is inside its core and the segment passes strictly
    within the core radius of the centre. Dead targets are tested like living ones (a wreck in
    sight is seen); dead observers see nothing. Bullets are not perceived."""
    c = state.constants
    x, y = state.x, state.y
    line_x = x[:, :, None] - x[:, None, :]  # (E, target, observer): target minus observer, d
    line_y = y[:, :, None] - y[:, None, :]
    length_squared = line_x * line_x + line_y * line_y
    vision = c["vision_range"]
    vision_squared = vision * vision
    clear = (
        (length_squared <= (vision_squared[:, None, :] if isinstance(vision, Tensor) else vision_squared))
        & state.alive[:, None, :]
        & (state.team[:, :, None] != state.team[:, None, :])
    )
    count = state.fields.shape[-1]
    if count > 0:
        def per_field(row: int) -> Tensor:  # (E, 1, F)
            return state.fields[:, row, None, :]

        radius, half_band = per_field(FIELD_RADIUS), 0.5 * per_field(FIELD_BAND)
        core = (radius - half_band).clamp_min(0.0)
        core_squared, outer = core * core, radius + half_band
        offset_x = x[..., None] - per_field(FIELD_X)  # (E, N, F)
        offset_y = y[..., None] - per_field(FIELD_Y)
        distance_squared = offset_x * offset_x + offset_y * offset_y
        in_core, in_field = distance_squared < core_squared, distance_squared < outer * outer
        core_target, core_observer = in_core[:, :, None], in_core[:, None]  # (E, N, N, F)
        ends = (core_target ^ core_observer) & ~(in_field[:, :, None] & in_field[:, None])
        # u = centre - observer = -(observer's offset); along = u . d, across = u x d.
        along = -(offset_x[:, None] * line_x[..., None] + offset_y[:, None] * line_y[..., None])
        across = offset_x[:, None] * line_y[..., None] - offset_y[:, None] * line_x[..., None]
        length = length_squared[..., None]
        passes = (
            ~(core_target | core_observer)
            & (across * across < core_squared[:, None] * length)
            & (along > 0.0)
            & (along < length)
        )
        clear = clear & ~(ends | passes).any(-1)
    return clear.any(-1)


def _snapshot(state: State) -> Tensor:
    """(E, SEEN_COLUMNS, N) float32: each ship's raw values now, as a sighting records them.
    Columns on the middle axis, so each column read back is an (E, N) slice with unit stride
    along N."""
    bullet_share = ((state.bullet_moves_left > 0) & state.bullet_real).to(torch.float32).mean(-1)
    alive = state.alive.to(torch.float32)
    return torch.cat([state.ships, torch.stack([bullet_share, alive, torch.ones_like(alive)], 1)], 1)


def perceive(state: State, revealed: Tensor) -> State:
    """The state with each ship's sighting by its enemy team brought up to date: a ship in the
    enemy's sight or `revealed` (E, N) bool (it fired or was hit this tick) is recorded as it is
    now and its ticks since seen set to 0; any other ship keeps its last-seen row and counts up
    to the cap. Each ship has one enemy team, so one row per ship serves."""
    visible = (sighted_by_enemy(state) | revealed) & state.exists
    seen = torch.where(visible[:, None], _snapshot(state), state.seen)
    since = torch.where(visible, 0, (state.ticks_since_seen + 1).clamp_max(TICKS_SINCE_SEEN_CAP))
    return dataclasses.replace(state, seen=seen, ticks_since_seen=since)


def team_view(state: State, team: int) -> State:
    """The game as team `team` knows it: its own ships as they are, every enemy as last seen
    (a never-seen or seen-dead enemy is not alive), and no bullets (bullets are not perceived;
    the bullet fields are None). With fog off (state.seen is None) the state itself. Fields
    that are fixed and told at reset (exists, team, type, constants, tick) are shared."""
    if state.seen is None:
        return state
    enemy = (state.team != team)[:, None]
    known = torch.where(enemy, state.seen, _snapshot(state))
    alive = (known[:, SEEN_ALIVE] > 0.5) & (known[:, SEEN_KNOWN] > 0.5)
    return dataclasses.replace(
        state,
        ships=known[:, : len(SHIP_COLUMNS)],
        # A shot reveals its ship on that tick, so the true flag is known.
        flags=torch.stack([alive, state.exists, state.fired], 1),
        bullets=None, bullet_counts=None, bullet_real=None, seen=None, ticks_since_seen=None,
    )


# --------------------------------------------------------------------------- observation

SPEED_SCALE = 400.0  # m/s
NOSE_RATE_SCALE = 3.0  # rad/s
ANGLE_SCALE = 0.27  # rad, about the stall angle
SHIELD_WAIT_SCALE = 3.0  # s, the recharge delay
COOLDOWN_CLIP = 0.5  # s
BOUNDARY_CLIP = 1000.0  # m
SHIELD_SCALE = 150.0
FIELD_RADIUS_SCALE = 1000.0  # m (v2's inputs.py)
FIELD_BAND_SCALE = 40.0  # m
FIELD_LEVEL_SCALE = 2.0  # half-octaves


def observe(config: GameConfig, state: State, team: int) -> Obs:
    """Team `team`'s view of every slot: full information with fog off; with fog on, built
    from `team_view` only (enemies as last seen), so nothing in it depends on what the team
    cannot see.

    ship (E, N, SHIP_FEATURES): 20 scalars [velocity x, y and speed over 400 m/s; angle of
    attack over 0.27 rad, its cos and sin; nose rate over 3 rad/s; energy and shield as
    fractions of the type's full values; shield over 150; shield wait over 3 s, clipped to
    1; gun cooldown in seconds clipped to 0.5; distance inside the boundary over 1000 m
    clipped to [-1, 1]; the outward unit direction x, y; heading cos, sin; position over the
    arena radius x, y; the share of bullet slots in flight], then flags [same team as the
    viewer, alive, exists, fired last tick], then the ship-type one-hot. Rows of slots that
    do not exist are zero. Fired last tick is the step's stored flag; the share of bullet slots
    in flight counts real bullets only (ghosts are not observed).

    Fog: an enemy row holds its values at its last sighting, alive is "not seen dead", and
    key_mask leaves out never-seen and seen-dead enemies; never-seen rows are zero. visible is
    True for own rows and enemies seen this tick; since_seen is symlog of the seconds since the
    last sighting (v2's time_since_seen as its model reads it), 0 for own rows. The globals'
    enemy alive fraction counts enemies not seen dead.
    """
    c = state.constants
    h = config.tick_length
    same_team = state.team == team
    if config.fog:
        view = team_view(state, team)
        enemy_seen = torch.where(same_team[:, None], 0.0, state.seen[:, SEEN_SHARE:])
        known = same_team | (enemy_seen[:, SEEN_KNOWN - SEEN_SHARE] > 0.5)
        # Own rows: the live share; enemy rows: as last seen.
        bullet_share = torch.where(
            same_team,
            ((state.bullet_moves_left > 0) & state.bullet_real).to(torch.float32).mean(-1),
            enemy_seen[:, 0],
        )
        # For the globals: own ships alive, enemies not seen dead (never-seen count as alive).
        counted = torch.where(same_team, state.alive, enemy_seen[:, SEEN_ALIVE - SEEN_SHARE] > 0.5)
    else:
        view = state
        in_flight = (state.bullet_moves_left > 0) & state.bullet_real
        bullet_share = in_flight.to(torch.float32).mean(-1)
    exists, alive = state.exists, view.alive
    # World-frame motion, as v2 observes it: the proper speed over the refractive index, so a
    # field that slows a ship shows as a slower ship (src/bnb/game/perception/observe.py).
    speed = view.speed * torch.exp(-view.log_index)
    velocity_x = speed * (view.heading_cos * view.angle_cos + view.heading_sin * view.angle_sin)
    velocity_y = speed * (view.heading_sin * view.angle_cos - view.heading_cos * view.angle_sin)
    distance = torch.sqrt(view.x * view.x + view.y * view.y)
    inverse_distance = 1.0 / distance.clamp_min(1.0)
    fired_last = state.fired  # stored by the step (agreed fix 12), not inferred from the ring
    scalars = [
        velocity_x / SPEED_SCALE,
        velocity_y / SPEED_SCALE,
        speed / SPEED_SCALE,
        view.angle / ANGLE_SCALE,
        view.angle_cos,
        view.angle_sin,
        view.nose_rate / NOSE_RATE_SCALE,
        view.energy * c["inverse_max_energy"],
        view.shield / c["max_shield"],
        view.shield / SHIELD_SCALE,
        (view.shield_wait / SHIELD_WAIT_SCALE).clamp(0.0, 1.0),
        (view.cooldown * h).clamp(0.0, COOLDOWN_CLIP),
        ((config.arena_radius - distance) / BOUNDARY_CLIP).clamp(-1.0, 1.0),
        view.x * inverse_distance,
        view.y * inverse_distance,
        view.heading_cos,
        view.heading_sin,
        view.x / config.arena_radius,
        view.y / config.arena_radius,
        bullet_share,
    ]
    flags = [same_team, alive, exists, fired_last & alive]
    one_hot = torch.nn.functional.one_hot(state.type, SHIP_TYPES)
    shown = (exists & known) if config.fog else exists
    ship = torch.cat(
        [torch.stack(scalars, -1), torch.stack(flags, -1).to(torch.float32), one_hot.to(torch.float32)], -1
    ) * shown[..., None]
    assert ship.shape[-1] == SHIP_FEATURES

    def alive_fraction(mask: Tensor, living: Tensor) -> Tensor:
        return (living & mask).sum(-1) / (exists & mask).sum(-1).clamp_min(1)

    enemy_living = counted if config.fog else alive
    globals_ = torch.stack(
        [
            state.tick.to(torch.float32) / config.clock_ticks,
            alive_fraction(same_team, alive),
            alive_fraction(~same_team, enemy_living),
        ],
        -1,
    )
    assert globals_.shape[-1] == GLOBAL_FEATURES
    fields = state.fields
    field_level = fields[:, FIELD_TARGET] / card.LN_SQRT2
    field = torch.stack(
        [
            fields[:, FIELD_RADIUS] / FIELD_RADIUS_SCALE,
            fields[:, FIELD_BAND] / FIELD_BAND_SCALE,
            field_level / FIELD_LEVEL_SCALE,
        ],
        -1,
    )
    assert field.shape[-1] == FIELD_FEATURES
    if config.fog:
        hidden_ticks = torch.where(same_team, 0, state.ticks_since_seen)
        visible = known & (hidden_ticks == 0)
        since_seen = torch.log1p(hidden_ticks.to(torch.float32) * h)
    else:
        visible = torch.ones_like(exists)
        since_seen = torch.zeros_like(state.x)
    return Obs(
        ship=ship,
        pos=torch.stack([view.x, view.y], -1),  # a never-seen enemy's last-seen row is zero
        heading=torch.atan2(view.heading_sin, view.heading_cos),
        key_mask=exists & alive,
        own=same_team & alive,
        globals=globals_,
        field=field,
        field_pos=torch.stack([fields[:, FIELD_X], fields[:, FIELD_Y]], -1),
        field_mask=torch.ones(field.shape[:2], dtype=torch.bool, device=field.device),  # all present
        visible=visible,
        since_seen=since_seen,
    )
