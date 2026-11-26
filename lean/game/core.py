"""The game: reset, one tick, and the observation, as pure functions of tensors.

Sizes: E environments, N ship slots, K bullet slots per ship, F fields. Bullet o * K + k
belongs to ship o. Units: metres, seconds, radians, joules; clocks in ticks.

Tick order (`step`): fire, flight, bullet motion, the swept hit test, damage, the boundary and
shields, the end check, the bullets' bookkeeping, then the fog's sighting. Nothing here resets
an environment: the caller deals afresh (`deal`) and takes the deals where StepOut.done
(`take_deal`).
"""

from __future__ import annotations

import dataclasses
import functools
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import NamedTuple

import torch
from torch import Tensor

from lean.contract import (
    BULLET_COLUMNS,
    BULLET_COUNT_COLUMNS,
    BULLET_SLOTS,
    FIELD_FEATURES,
    GLOBAL_FEATURES,
    NO_PENDING,
    SHIP_COLUMNS,
    SHIP_FEATURES,
    SHIP_TYPES,
    Constants,
    Obs,
    State,
    StepOut,
    join_action,
    random_keywords,
    symlog,
)

from . import card

# Rows of the packed state read by index here (lean.contract's *_COLUMNS).
X, Y = SHIP_COLUMNS.index("x"), SHIP_COLUMNS.index("y")
LOG_INDEX, GRADIENT_Y = SHIP_COLUMNS.index("log_index"), SHIP_COLUMNS.index("gradient_y")
BULLET_X, BULLET_Y = BULLET_COLUMNS.index("bullet_x"), BULLET_COLUMNS.index("bullet_y")
BULLET_DIR_X, BULLET_DIR_Y = (
    BULLET_COLUMNS.index("bullet_dir_x"),
    BULLET_COLUMNS.index("bullet_dir_y"),
)
BULLET_SPEED, BULLET_LOG_INDEX = (
    BULLET_COLUMNS.index("bullet_speed"),
    BULLET_COLUMNS.index("bullet_log_index"),
)
MOVES_LEFT = BULLET_COUNT_COLUMNS.index("bullet_moves_left")
FIRED_TICK = BULLET_COUNT_COLUMNS.index("bullet_fired_tick")


@dataclass
class GameConfig:
    """The rules of one run: the core game (docs/rewrite/core-decisions.md).

    arena_radius: metres; boundary damage starts beyond it. boundary_ramp_width: metres over
    which boundary damage rises to its maximum. clock_ticks: the episode length. spawn, the
    rule chosen at launch (`deal`): "mixed" (every ship uniform over the spawn disc, uniform
    heading) or "sides" (each team uniform over a half disc on its side of a random axis,
    noses toward the other side); under both, no ship starts near the edge facing outward.
    lineup: type names (lean.game.card.TYPE_NAMES) by rank within a team, cycled, the same for
    both teams. tick_length: seconds.

    training_spawn (on in training, off for evaluation, play and the sandbox): each ship of an
    environment starts at a proper speed uniform from its stall speed to its top level speed,
    a shield uniform from 0 to full and a store uniform from 0 to full, except that a share
    `standard_spawn_share` of environments (drawn per environment at the deal) takes the
    standard spawn (full shield, half store, the spawn speed). Off: the standard spawn for all.
    """

    arena_radius: float = 4200.0
    boundary_ramp_width: float = 300.0
    # 60 s: about 85% of scripted 4v4 games end before it (docs/rewrite/clock-measurement.md).
    clock_ticks: int = 1800
    spawn: str = "mixed"
    lineup: tuple[str, ...] = ("p51d",)
    tick_length: float = 1.0 / 30.0
    training_spawn: bool = False
    standard_spawn_share: float = 0.125
    # Fields drawn per environment at every reset (v2's random map, src/bnb/game/physics/
    # spawning.py new_map, real-scale elimination's values): `field_count` fields in
    # point-symmetric pairs (and one at the centre for an odd count), centres uniform in
    # `field_disc_fraction` of the arena, radius log-normal (median, sigma) clamped to
    # [min, max], all times the map scale (arena radius / 2,600 m), band `field_band_width`,
    # level -2, -1, +1 or +2 half-octaves. 0: no fields (no medium).
    field_count: int = 10
    field_disc_fraction: float = 0.85
    field_radius_median: float = 400.0
    field_radius_log_sigma: float = 0.6
    field_radius_min: float = 120.0
    field_radius_max: float = 1200.0
    field_band_width: float = 160.0
    field_high_share: float = 0.5
    field_strong_share: float = 0.5
    # Ghost bullets (the hit head's labels, author 2026-11-22): every living ship puts a bullet
    # in its ring every tick; only a real shot damages and kills; a ghost ends at its first
    # contact like a real bullet. Every bullet's outcome is reported when its flight ends.
    # False (an ablation): real shots only.
    ghost_bullets: bool = True
    # Flight (docs/rewrite/core-decisions.md "Flight physics"): weight always on the wing, the
    # turn limited unless diving; the engine drives speed only and never feeds the store; a
    # store of `store_loops` vertical loops at corner speed; dive and climb equal and opposite;
    # a smootherstep taper over `store_edge_band` of the store at each end; firing scales thrust
    # by `fire_thrust_fraction` on the ticks it fires and costs no store.
    store_loops: float = 2.0
    store_edge_band: float = 0.05
    fire_thrust_fraction: float = 0.0
    # Firing never takes a ship below its stall speed (author 2026-11-26; `_fire`, `_fly`): the
    # thrust a shot withholds is paid from the store instead, wholly at stall speed and below,
    # wholly by the speed from (1 + fire_store_band) x stall speed up, a smootherstep between.
    # With an empty store a ship below that speed cannot fire (its gun stays ready).
    fire_store_band: float = 0.1
    # The post-stall regime (author 2026-11-26; `_setting`, `_lift_coefficient`, `_rates`): below
    # corner speed full stick asks for the card's deep_stall_angle (opening from the stall angle
    # over deep_stall_band of corner speed, closing to the stall angle toward stall speed), the
    # lift past the stall returns toward the flat plate's and the drag rises toward it. False
    # (for comparison): full stick stays on the top line, the lift past the stall is the old
    # parabola's tail (0 at twice the stall angle) and no post-stall drag. Either way the middle
    # stick's shape and the nose's damping are the new.
    post_stall: bool = True
    # Fog of war: an enemy is visible in the team's sight (v2's sight test: `sighted_by_enemy`)
    # and on the tick it fires or is hit by a bullet; a hidden enemy is its last-seen row.
    # False (an ablation): full information.
    fog: bool = True
    # The tracking circle (author 2026-11-26; `sight_pairs`): an enemy visible on the tick before
    # (by sight or a reveal) stays visible while an observer has it within the circle of the
    # observer's forward reach p / (1 - e), with a clear line. False (an ablation): the ellipse
    # every tick. At e = 0 the circle is the ellipse, so the switch changes nothing there.
    fog_tracking: bool = True


def built_once(build):
    """Cache a builder of fixed tensors per argument (agreed fix 4: built once, not every tick).
    While torch.compile traces, the builder runs in the graph instead (constants there, and no
    cache keyed on traced sizes)."""
    cached = functools.cache(build)

    @functools.wraps(build)
    def get(*args):
        return build(*args) if torch.compiler.is_compiling() else cached(*args)

    return get


# --------------------------------------------------------------------------- constants

# Constants counted in whole ticks: they must be equal for every type of a run.
INTEGER_CONSTANTS = ("bullet_moves", "fire_interval")
# Action-indexed tables (by stick index, by power index): always rows of the per-slot table.
ACTION_TABLES = (
    "stick_table",
    "stick_cap_table",
    "deep_stall_table",
    "commanded_table",
    "demand_table",
)


def _per_type(config: GameConfig) -> dict[str, Tensor]:
    """Every ship constant for every card type: float32 (T,), the action tables (T, width), the
    whole-tick counts long; computed with the game's own float32 arithmetic."""

    def of(name: str) -> Tensor:
        return torch.tensor(card.CARD[name], dtype=torch.float32)

    lift, acceleration = of("max_lift_coefficient"), of("max_acceleration")
    stall = card.STALL_ANGLE * of("stall_angle_scale")
    # The store: `store_loops` vertical loops at corner speed. A loop climbs twice the corner
    # turn radius, whose sideways acceleration is the g limit with 1 g held on the wing.
    corner_speed_squared = acceleration / lift
    corner_radius = corner_speed_squared / torch.sqrt(acceleration * acceleration - card.GRAVITY**2)
    max_energy = config.store_loops * card.GRAVITY * 2.0 * corner_radius
    exchange = of("exchange_acceleration")  # dive and climb, equal and opposite
    zeros = torch.zeros_like(exchange)
    width = card.GRAVITY / 32.0
    gravity = torch.full_like(exchange, card.GRAVITY)
    raw = torch.stack([gravity, card.GRAVITY - exchange, card.GRAVITY + exchange], -1)
    hit_radius = of("hit_radius")
    nose_inner_cos, nose_outer_cos = (
        torch.cos(of("nose_inner_angle")),
        torch.cos(of("nose_outer_angle")),
    )
    if bool((nose_outer_cos >= nose_inner_cos).any()):
        raise ValueError("the nose cone's outer angle must be larger than its inner angle")
    values = {
        "lift": lift,
        "max_acceleration": acceleration,
        "zero_lift_drag": of("zero_lift_drag"),
        "induced_drag": card.INDUCED_DRAG_FACTOR * of("induced_drag_scale"),
        "power": of("engine_power"),
        "static_thrust": of("static_thrust"),
        "max_energy": max_energy,
        "inverse_max_energy": 1.0 / max_energy,
        "inverse_fade": 1.0 / (config.store_edge_band * max_energy),
        "stall": stall,
        "inverse_stall": 1.0 / stall,
        "nose_frequency": of("corner_nose_frequency") / torch.sqrt(acceleration / lift),
        "commanded_table": torch.stack([zeros, -exchange, exchange], -1),
        "demand_table": 0.5 * (raw + torch.sqrt(raw * raw + width * width)),
        "muzzle_speed": of("muzzle_speed"),
        "spread_speed": card.BULLET_SPREAD * of("bullet_spread_scale"),
        "bullet_drag": card.BULLET_DRAG * of("bullet_drag_scale"),
        "bullet_moves": torch.round(of("bullet_life") / config.tick_length).long(),
        "fire_interval": torch.round(1.0 / (config.tick_length * of("fire_rate"))).long(),
        "hit_damage": of("hit_damage"),
        "explosive_share": of("explosive_share"),
        "hit_radius": hit_radius,
        "hit_radius_squared": hit_radius * hit_radius,
        "core_radius": of("core_radius"),
        "nose_inner_cos": nose_inner_cos,
        "inverse_nose_width": 1.0 / (nose_inner_cos - nose_outer_cos),
        "max_shield": of("max_shield"),
        "shield_rate": card.SHIELD_RECHARGE_RATE * of("shield_recharge_rate_scale"),
        "shield_delay": card.SHIELD_RECHARGE_DELAY * of("shield_recharge_delay_scale"),
        "spawn_energy": 0.5 * max_energy,
        "vision_range": card.VISION_RANGE * of("vision_range_scale"),
        "vision_eccentricity": of("vision_eccentricity"),
        "inverse_critical_speed": 1.0 / of("critical_speed"),
        "drag_rise_exponent": of("drag_rise_exponent"),
        "separation_drag": of("separation_drag"),
        "half_separation_exponent": 0.5 * of("separation_exponent"),
        "lift_softmin_exponent": of("lift_softmin_exponent"),
        "thrust_softmin_exponent": of("thrust_softmin_exponent"),
        "nose_damping": of("nose_damping"),
    }
    values.update(_post_stall_constants(config, values))
    values.update(_derived_characteristics(config, values))
    return values


def _post_stall_constants(config: GameConfig, values: dict[str, Tensor]) -> dict[str, Tensor]:
    """The post-stall regime's constants (GameConfig.post_stall; `_commanded_angle`,
    `_lift_coefficient`, `_rates`). Past the stall the lift falls from the maximum to the flat
    plate's C (card.POST_STALL_LIFT_FRACTION of the maximum) at 45 deg by a smootherstep in the
    angle (monotonic, zero slope at both ends, no dip), then is C sin(2 a) (0 at 90 deg). The
    post-stall drag coefficient: post_stall_drag x 2 u^2 / (1 + u^2), u the angle's
    (cos stall - cos a) over (cos stall - cos 45 deg); post_stall_drag set so the drag
    coefficient at 45 deg (the flat plate's lift, separated) is post_stall_drag_ratio x the one
    at the stall (both without the drag rise and the held 1 g's induced drag)."""
    lift, stall = values["lift"], values["stall"]
    count = len(lift)
    on = torch.full((count,), 1.0 if config.post_stall else 0.0)
    flat = card.POST_STALL_LIFT_FRACTION * lift
    induced, separation = values["induced_drag"], values["separation_drag"]
    zero_lift = values["zero_lift_drag"]
    exponent = 2.0 * values["half_separation_exponent"]
    stall_drag = zero_lift + induced * lift * lift * (1.0 + separation)
    near = card.POST_STALL_LIFT_FRACTION**2
    reference_drag = zero_lift + induced * flat * flat * (1.0 + separation * near ** (exponent / 2))
    post_stall_drag = (card_of("post_stall_drag_ratio") * stall_drag - reference_drag).clamp_min(0)
    # By stick index (card.STICK_LEVELS' layout): the middle level's g cap, full stick's deep angle.
    level = torch.tensor(card.STICK_LEVELS).abs().expand(count, -1)
    cap = torch.where(level == 2.0, card_of("middle_g_fraction")[:, None], 1.0)
    deep = (level == 3.0).float()
    stall_cos = torch.cos(stall)
    stall_speed = torch.sqrt(card.GRAVITY / lift)
    # The lift at the card's deep angle (`_lift_and_slope` past the stall).
    deep_angle = card_of("deep_stall_angle")
    toward = ((deep_angle - stall) / (math.radians(45.0) - stall)).clamp(0.0, 1.0)
    deep_lift = torch.where(
        deep_angle > math.radians(45.0),
        flat * torch.sin(2.0 * deep_angle),
        lift - (lift - flat) * smootherstep(toward),
    )
    return {
        "post_stall": on,
        "deep_stall_angle": card_of("deep_stall_angle"),
        "inverse_corner_speed": torch.sqrt(lift / values["max_acceleration"]),
        "inverse_stall_speed_squared": lift / card.GRAVITY,
        "deep_stall_start": card_of("deep_stall_start"),
        "inverse_deep_stall_band": 1.0 / card_of("deep_stall_band"),
        # Full stick's deep angle closes to the stall angle from where its lift is 1 g down to
        # stall speed (`_holding_angle`).
        "holding_start_speed": stall_speed,
        "inverse_holding_span": 1.0
        / (stall_speed * (torch.sqrt(lift / deep_lift) - 1.0)).clamp_min(1e-6),
        "flat_plate_lift": flat,
        "inverse_tail_span": 1.0 / (math.radians(45.0) - stall),
        "inverse_share_width": torch.full((count,), 1.0 / card.POST_STALL_SHARE_WIDTH),
        "post_stall_drag": post_stall_drag * on,
        "stall_cos": stall_cos,
        "inverse_drag_cos_span": 1.0 / (stall_cos - math.cos(card.POST_STALL_DRAG_ANGLE)),
        "stick_cap_table": cap,
        "deep_stall_table": deep * on[:, None],
    }


def card_of(name: str) -> Tensor:
    """A card column (one value per type) as float32."""
    return torch.tensor(card.CARD[name], dtype=torch.float32)


def stick_table(count: int) -> Tensor:
    """(count, 7) signed stick levels by stick index (card.STICK_LEVELS' layout): straight, the
    gradual level, the middle level, full stick (card.GRADUAL_STICK, MIDDLE_STICK, FULL_STICK).
    Each level is that fraction of the most lateral acceleration the ship can give at its speed
    (`_setting`)."""
    magnitude = torch.tensor(
        [
            0.0,
            card.GRADUAL_STICK,
            card.GRADUAL_STICK,
            card.MIDDLE_STICK,
            card.MIDDLE_STICK,
            card.FULL_STICK,
            card.FULL_STICK,
        ]
    )
    row = torch.sign(torch.tensor(card.STICK_LEVELS)) * magnitude
    return row.expand(count, -1).clone()


def _derived_characteristics(config: GameConfig, values: dict[str, Tensor]) -> dict[str, Tensor]:
    """Per ship type (card order), solved with the game's own flight functions, so the scripted
    opponent is told these, not guessed (its references, docs/rewrite/scripted-bfm-design.md):

    - cruise speed (straight, coasting, thrust equal to drag: the top level speed), stall speed
      (the wing's lift limit holds 1 g), corner speed (the g limit meets the lift limit), bullet
      launch speed (cruise plus muzzle), gun range (how far a bullet fired at cruise travels
      in its life under drag: v2's `api.gun_range`) and the damage's reference speed (that
      bullet's speed card.DAMAGE_REFERENCE_DISTANCE out: v2's rule, ships/real-scale);
    - the best sustained turn (coasting, mid store): its rate, its speed and its radius (speed
      over rate), and the time to turn 90 degrees at that rate;
    - the corner rate (full stick at corner speed) and the store's turn: the turn a full store
      buys over the best sustained rate when spent holding corner speed at full stick,
      (corner rate - sustained rate) x full store / |energy rate at corner, full stick|;
    - the stick levels (`stick_table`: the shared thirds of full stick, card.GRADUAL_STICK,
      MIDDLE_STICK, FULL_STICK);
    - the middle stick speed, corner speed / sqrt(middle level): the scripted agent's
      reference for its level cap (the speed where the middle fraction of the lift-limited
      line would reach the g limit; the levels are fractions of the top line, so the middle
      level turns at its fraction of full stick at every speed)."""
    c = SimpleNamespace(**values)
    count = len(c.lift)  # every row of the card
    control = SimpleNamespace(
        thrust_scale=1.0,
        stick_size=torch.zeros(count),
        stick_sign=torch.ones(count),
        stick_cap=torch.ones(count),
        deep=torch.zeros(count),
        commanded=c.commanded_table[:, 0],
        demand=c.demand_table[:, 0],
    )
    low, high = torch.full((count,), 20.0), torch.full((count,), 1000.0)
    with torch.no_grad():
        for _ in range(60):  # bisection on d speed / dt = 0
            middle = 0.5 * (low + high)
            setting = _setting(c, control, middle, c.spawn_energy)
            one, zero = torch.ones_like(middle), torch.zeros_like(middle)
            rate, _ = _rates(
                c, setting, one, middle, zero, one, 1.0 / (middle * middle), 1.0 / (c.lift * c.lift)
            )
            low, high = torch.where(rate > 0.0, middle, low), torch.where(rate > 0.0, high, middle)
    cruise = 0.5 * (low + high)
    launch = cruise + c.muzzle_speed
    life = c.bullet_moves.float() * config.tick_length
    stall_speed = torch.sqrt(card.GRAVITY / c.lift)
    corner_speed = torch.sqrt(c.max_acceleration / c.lift)
    sustained_rate, sustained_speed = _best_sustained_turn(values, stall_speed, cruise)
    # Per row, as (T, 1) columns, for one speed per row.
    column = SimpleNamespace(**{name: value[:, None] for name, value in values.items()})
    with torch.no_grad():
        corner = corner_speed[:, None]
        corner_rate, _, corner_energy_rate = quasi_steady(
            column, corner, torch.ones_like(corner), 0, column.spawn_energy
        )
    corner_rate, corner_energy_rate = corner_rate[:, 0], corner_energy_rate[:, 0]
    return {
        "cruise_speed": cruise,
        "stall_speed": stall_speed,
        "corner_speed": corner_speed,
        "bullet_launch_speed": launch,
        # the damage's reference speed: a bullet fired at cruise, DAMAGE_REFERENCE_DISTANCE out
        "inverse_reference_bullet_speed": torch.exp(c.bullet_drag * card.DAMAGE_REFERENCE_DISTANCE)
        / launch,
        "gun_range": torch.log1p(c.bullet_drag * launch * life) / c.bullet_drag,
        "sustained_speed": sustained_speed,
        "sustained_rate": sustained_rate,
        "sustained_radius": sustained_speed / sustained_rate,
        "quarter_turn_time": (0.5 * math.pi) / sustained_rate,
        "corner_rate": corner_rate,
        # the middle stick level reaches its g cap (middle_g_fraction) above this speed
        "middle_stick_speed": corner_speed * torch.sqrt(c.stick_cap_table[:, 3]),
        "stick_table": stick_table(count),
        "store_turn": (corner_rate - sustained_rate) * c.max_energy / corner_energy_rate.abs(),
        # a collision's reference closing speed: two ships head-on at cruise (`collisions`)
        "inverse_collision_speed": 0.5 / cruise,
    }


def steady_rates(
    p,
    speed: Tensor,
    stick: Tensor,
    power: int,
    energy: Tensor,
    cap: Tensor | float = 1.0,
    deep: Tensor | float = 0.0,
):
    """A ship flying steadily (no field, the nose's lag settled) at `speed` with stick size
    `stick` (0 to 1) and power index `power` (0 coast, 1 dive, 2 climb) at store `energy`: its
    turn rate (rad/s), d speed / dt and d store / dt, from the flight functions the tick uses.
    `p` holds the ship constants broadcastable against `speed` (the action tables with a last
    axis of width 3). Every quasi-steady analysis (the derived characteristics, lean.game.tuning's
    performance and envelope) goes through here. By default `stick` is a fraction of the
    envelope's top line (the wing's maximum lift against the g limit); `cap` (the g cap, a
    fraction of the g limit: the middle level's middle_g_fraction) and `deep` (1: full stick's
    deep angle below corner) give a stick level's own shape (`stick_level_shape`)."""
    control = SimpleNamespace(
        thrust_scale=1.0,
        stick_size=stick,
        stick_sign=torch.ones_like(speed),
        stick_cap=torch.as_tensor(cap, dtype=speed.dtype).expand_as(speed),
        deep=torch.as_tensor(deep, dtype=speed.dtype).expand_as(speed),
        commanded=p.commanded_table[..., power],
        demand=p.demand_table[..., power],
    )
    inverse_lift = 1.0 / p.lift
    setting = _setting(p, control, speed, energy)
    lateral, _, vertical, thrust = setting
    angle = _commanded_angle(p, control, lateral, speed, inverse_lift, vertical)
    lift = _lift_coefficient(p, angle)
    rate = (
        _path_lift(p, lift, angle, vertical, speed * speed, thrust, thrust * torch.sin(angle))[0]
        / speed
    )
    speed_rate, store_rate = _rates(
        p,
        setting,
        torch.ones_like(speed),
        speed,
        lift,
        torch.cos(angle),
        1.0 / (speed * speed),
        inverse_lift * inverse_lift,
    )
    return rate, speed_rate, store_rate


def quasi_steady(p, speed: Tensor, stick: Tensor, power: int, energy: Tensor, cap=1.0, deep=0.0):
    """`steady_rates` with the store's rate folded into the total energy rate: turn rate (rad/s),
    d speed / dt, and speed x d speed / dt + d store / dt."""
    rate, speed_rate, store_rate = steady_rates(p, speed, stick, power, energy, cap, deep)
    return rate, speed_rate, speed * speed_rate + store_rate


def stick_level_shape(p, index: int) -> tuple[float, float, float]:
    """A stick index's (stick size, g cap, deep) as `steady_rates` takes them (card.STICK_LEVELS'
    layout; from the constants' tables, so the post-stall switch is read): straight (0, 1, 0),
    gradual (GRADUAL_STICK, 1, 0), middle (MIDDLE_STICK, middle_g_fraction, 0), full (1, 1, 1 with
    GameConfig.post_stall, else 0)."""
    return tuple(
        float(table.reshape(-1, len(card.STICK_LEVELS))[0, index].abs())
        for table in (p.stick_table, p.stick_cap_table, p.deep_stall_table)
    )


SUSTAINED_GRID = 257  # speeds per search pass
SUSTAINED_STICK_STEPS = 40  # bisection steps on the stick holding energy


def _best_sustained_turn(values: dict[str, Tensor], stall_speed: Tensor, cruise: Tensor):
    """Per row (T,): the best sustained turn rate and its speed, coasting at mid store. At each
    speed the stick that holds energy (total energy rate zero; full stick when
    full stick gains), over speeds above stall that hold energy straight; a grid from stall to
    cruise speed, then a second grid one step either side of the best (resolution about
    0.01 m/s)."""
    column = SimpleNamespace(**{name: value[:, None] for name, value in values.items()})
    energy = column.spawn_energy

    def sustained(speed: Tensor) -> Tensor:
        zero, one = torch.zeros_like(speed), torch.ones_like(speed)
        low, high = zero, one
        for _ in range(SUSTAINED_STICK_STEPS):
            middle = 0.5 * (low + high)
            gains = quasi_steady(column, speed, middle, 0, energy)[2] > 0.0
            low, high = torch.where(gains, middle, low), torch.where(gains, high, middle)
        full_gains = quasi_steady(column, speed, one, 0, energy)[2] >= 0.0
        stick = torch.where(full_gains, one, 0.5 * (low + high))
        rate = quasi_steady(column, speed, stick, 0, energy)[0]
        straight = quasi_steady(column, speed, zero, 0, energy)[2]
        return torch.where((straight >= 0.0) & (speed > stall_speed[:, None]), rate, 0.0)

    fraction = torch.linspace(0.0, 1.0, SUSTAINED_GRID)
    with torch.no_grad():
        low, high = stall_speed[:, None], cruise[:, None]
        for _ in range(2):
            speeds = low + (high - low) * fraction
            rates = sustained(speeds)
            best = rates.argmax(-1, keepdim=True)
            step = (high - low) / (SUSTAINED_GRID - 1)
            centre = speeds.gather(-1, best)
            low, high = centre - step, centre + step
        return rates.gather(-1, best)[:, 0], centre[:, 0]


def per_type_constants(config: GameConfig, ship_type: Tensor) -> SimpleNamespace:
    """For analysis tools (charts, tuning, measurements): every ship constant of card rows
    `ship_type` (any shape S, card indices), float32 S-shaped (action tables S + (width,)),
    read from the card each call: not the run's lineup and not cached, so a tool may change
    the card (lean.game.tuning appends a row)."""
    return SimpleNamespace(**{name: value[ship_type] for name, value in _per_type(config).items()})


def _build_ship_card(config: GameConfig) -> Constants:
    lineup = tuple(config.lineup)
    used = torch.tensor([card.TYPE_NAMES.index(name) for name in lineup])
    # The arena's size and the tick's length, for the scripted opponent (it reads every number
    # from the game).
    uniform = {"arena_radius": config.arena_radius, "tick_length": config.tick_length}
    columns, rows = {}, []
    for name, values in _per_type(config).items():
        chosen = values[used]
        if name not in ACTION_TABLES and bool((chosen == chosen[0]).all()):
            uniform[name] = chosen[0].item()
            continue
        if name in INTEGER_CONSTANTS:
            raise ValueError(
                f"{name} differs between the lineup's types {lineup}; it must be equal"
            )
        block = values.reshape(len(card.TYPE_NAMES), -1).T  # (width, T)
        columns[name] = (sum(len(row) for row in rows), block.shape[0])
        rows.append(block)
    return Constants(uniform, columns, torch.cat(rows, 0)[None].contiguous())  # the tables at least


# Built ship cards by (store loops, store edge band, tick length, lineup, arena radius).
_SHIP_CARDS: dict[tuple, Constants] = {}


def ship_card(config: GameConfig) -> Constants:
    """The run's ship card: its constants per ship type, before they are dealt to slots (the
    table (1, C, T) on the CPU, one column per card type). Built once per flight settings,
    lineup and arena. The build reads numbers back to the host (`.item()`), so it runs eagerly:
    the first deal of a run (`reset`, always before any compiled tick) builds it, and a traced
    deal only looks it up."""
    key = (
        config.store_loops,
        config.store_edge_band,
        config.tick_length,
        tuple(config.lineup),
        config.arena_radius,
        config.post_stall,
    )
    if key not in _SHIP_CARDS:
        _SHIP_CARDS[key] = _build_ship_card(config)
    return _SHIP_CARDS[key]


@built_once
def _card_table(ship_card_: Constants, device: torch.device) -> Tensor:
    return ship_card_.table.to(device)


def ship_constants(config: GameConfig, ship_type: Tensor) -> Constants:
    """The per-ship constants of slots of types ship_type (E, N) long, each a type of the lineup:
    the card's uniform numbers, and its per-type rows gathered once into (E, C, N)."""
    ship_card_ = ship_card(config)
    table = _card_table(ship_card_, ship_type.device)
    envs, slots = ship_type.shape
    rows = table.shape[1]
    gathered = table.expand(envs, rows, -1).gather(2, ship_type[:, None].expand(envs, rows, slots))
    return Constants(ship_card_.uniform, ship_card_.columns, gathered)


# --------------------------------------------------------------------------- fixed tensors
# Index tensors and fill values that depend only on the sizes, built once per size and device
# (agreed fix 4), not every tick. They are shared: never written, never handed out in a state.


@built_once
def slot_indices(slots: int, device: torch.device) -> SimpleNamespace:
    """For N slots: ship (N,) 0..N-1; bullet_slot (K,); owner (N K,) the ship of each bullet;
    other (N K, N) bool, the bullet is not that ship's own; bullet_index (N K,); pair (N K,)
    the owner's row start in a flattened (N, N) matrix."""
    ship = torch.arange(slots, device=device)
    owner = ship.repeat_interleave(BULLET_SLOTS)
    return SimpleNamespace(
        ship=ship,
        bullet_slot=torch.arange(BULLET_SLOTS, device=device),
        owner=owner,
        other=owner[:, None] != ship,
        bullet_index=torch.arange(slots * BULLET_SLOTS, device=device),
        pair=owner * slots,
    )


@built_once
def filled(
    shape: tuple[int, ...], value: float | int | bool, dtype: torch.dtype, device: torch.device
) -> Tensor:
    """A constant tensor (the start of a scatter, a fixed row of a fresh deal), built once."""
    return torch.full(shape, value, dtype=dtype, device=device)


@built_once
def _eye(dtype: torch.dtype, device: torch.device) -> Tensor:
    return torch.eye(2, dtype=dtype, device=device)


# --------------------------------------------------------------------------- numerics


def minimum(a: Tensor, b: Tensor | float) -> Tensor:
    """torch.minimum for a constant that may be a Python number."""
    return torch.minimum(a, b) if isinstance(b, Tensor) else a.clamp_max(b)


def maximum(a: Tensor, b: Tensor | float) -> Tensor:
    """torch.maximum for a constant that may be a Python number."""
    return torch.maximum(a, b) if isinstance(b, Tensor) else a.clamp_min(b)


def softmin(a: Tensor, b: Tensor | float, exponent: Tensor | float) -> Tensor:
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
    slope = torch.where(
        in_band, -30.0 * (inward * outward) ** 2 / (band * distance.clamp_min(1e-6)), 0.0
    )
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
    mean_gradient_x = torch.where(
        covered, (target_gradient_x - mean * weight_gradient_x) / safe_weight, 0.0
    )
    mean_gradient_y = torch.where(
        covered, (target_gradient_y - mean * weight_gradient_y) / safe_weight, 0.0
    )
    return (
        coverage * mean,
        mean * coverage_gradient_x + coverage * mean_gradient_x,
        mean * coverage_gradient_y + coverage * mean_gradient_y,
    )


# --------------------------------------------------------------------------- flight


def _setting(p, control, speed, energy):
    """The controls resolved at a speed and store: signed lateral acceleration, the exchange
    between speed and store, the vertical acceleration held, and thrust.

    Weight always on the wing. The lift budget goes, in order: 1 g held (dive: its lower demand,
    which frees lift for the turn), then the turn, then a climb's extra demand gets what is left
    (the turn has precedence over the climb). Smootherstep edges of the store; at an edge, dive
    or climb fades to coast in its effect on the wing as well as in the trade (no free turn from
    diving on the deck, no lost turn from climbing at the top). The engine drives speed only
    (constant power) and never feeds the store; firing scales the thrust on the ticks it fires.

    The stick (author 2026-11-24, the middle level 2026-11-26): a stick level is its size times
    sqrt(level^2 - held^2) over its g cap, the level the wing's maximum lift against the cap's
    share of the g limit (one soft minimum: the true corner) and the budget's the same at the
    full g limit. Gradual and full (cap 1): their fraction of the envelope's top line; middle
    (size and cap card.MIDDLE_STICK): the wing's maximum lift against two thirds of the g limit.
    Below corner full stick's angle of attack goes past the stall (`_commanded_angle`).
    """
    speed_squared = speed * speed
    q = p.lift_softmin_exponent
    # The corner, smooth: the wing's lift against the g limit (the only soft minimum of the turn).
    budget = softmin(p.lift * speed_squared, p.max_acceleration, q).clamp_min(1e-12)
    sink_room = smootherstep((energy * p.inverse_fade).clamp(0.0, 1.0))
    budget_squared = budget * budget
    top_room = smootherstep(((p.max_energy - energy) * p.inverse_fade).clamp(0.0, 1.0))
    climbing = control.commanded > 0.0
    room = torch.where(climbing, top_room, torch.where(control.commanded < 0.0, sink_room, 1.0))
    demand = card.GRAVITY + (control.demand - card.GRAVITY) * room
    commanded = control.commanded * room
    first = torch.where(climbing, card.GRAVITY, demand)  # held before the turn
    held = softmin(first, budget, q)
    # The stick: its fraction of the most lateral acceleration the ship can give at this speed
    # (full stick: all of it, the envelope's top line; lift-limited below corner, g-limited
    # above): the budget with what is held taken first.
    # A level's own line: the wing's maximum lift against its g cap (the middle's
    # middle_g_fraction of the g limit; 1 for the others), the same soft minimum.
    level = softmin(p.lift * speed_squared, control.stick_cap * p.max_acceleration, q)
    lateral_size = control.stick_size * torch.sqrt((level * level - held * held).clamp_min(0.0))
    # The climb's extra lift, from what the turn left: never more than its demand.
    room_left = torch.sqrt((budget_squared - lateral_size * lateral_size).clamp_min(0.0))
    left = (room_left - held).clamp_min(0.0)
    vertical = held + torch.where(climbing, softmin(demand - first, left, q), 0.0)
    # The commanded trade is already faded at the edges. A climb's unmet extra demand is climb
    # not made; a shortfall below what is held first (the wing cannot hold 1 g, or the dive's
    # demand) sinks the ship, and that needs store to sink.
    unmet = minimum((demand - maximum(vertical, first)).clamp_min(0.0), commanded)
    unmet_climb = torch.where(climbing, unmet, 0.0)
    sinking = (first - vertical).clamp_min(0.0)
    exchange = commanded - unmet_climb - sinking * sink_room
    thrust = softmin(p.power / speed, p.static_thrust, p.thrust_softmin_exponent)
    thrust = thrust * control.thrust_scale
    return control.stick_sign * lateral_size, exchange, vertical, thrust


def _rates(
    p,
    setting,
    inverse_index,
    speed,
    lift,
    angle_cos,
    inverse_speed_squared,
    inverse_lift_squared,
    post_stall_drag=None,
):
    """(d speed / dt, d energy / dt) for a setting, a lift coefficient and an angle of attack.
    inverse_index: 1 / n, or None where there is no medium (n = 1). inverse_lift_squared: 1 over
    the squared maximum lift coefficient: the separation drag rises toward the wing's maximum and
    stops there (flying steadily the load never passes the maximum; in a transient the nose's
    lagging lift and the held vertical together can ask for up to twice it, which a steep
    exponent made 2^(exponent / 2) times the separation drag: 2026-11-24, exponent 24, a ship
    stopped dead in one tick). Past the stall (GameConfig.post_stall) the post-stall drag adds
    post_stall_drag x 2 u^2 / (1 + u^2), u = (cos stall - cos angle) / (cos stall - cos 45 deg)
    (`_post_stall_constants`): smooth from the stall, bounded at twice its 45 deg value
    (`post_stall_drag`: that term when the caller has it, else made here)."""
    _, exchange, vertical, thrust = setting
    induced = vertical * inverse_speed_squared
    load = lift * lift + induced * induced
    near = (load * inverse_lift_squared).clamp_max(1.0)  # of the wing's maximum, squared
    coefficient = p.zero_lift_drag * (
        1.0 + (speed * p.inverse_critical_speed) ** p.drag_rise_exponent
    ) + p.induced_drag * load * (1.0 + p.separation_drag * near**p.half_separation_exponent)
    if post_stall_drag is None:
        post_stall_drag = _post_stall_drag(p, angle_cos)
    if post_stall_drag is not None:
        coefficient = coefficient + post_stall_drag
    drag = coefficient * speed * speed
    store = speed * exchange
    speed_rate = thrust * angle_cos - drag - exchange
    if inverse_index is None:
        return speed_rate, store
    return speed_rate * inverse_index, store * inverse_index


def _post_stall_drag(p, angle_cos):
    """The post-stall drag coefficient at an angle's cosine (`_rates`); None with the regime
    off."""
    if not _switched_on(p.post_stall):
        return None
    past = ((p.stall_cos - angle_cos) * p.inverse_drag_cos_span).clamp_min(0.0)
    past_squared = past * past
    return p.post_stall_drag * (2.0 * past_squared) / (1.0 + past_squared)


def _switched_on(flag) -> bool:
    """A uniform on/off constant (1.0 or 0.0): a Python number in a run's constants, a tensor of
    equal values in the analysis tools' (`per_type_constants`)."""
    return bool(flag > 0.5) if not isinstance(flag, Tensor) else bool(flag.reshape(-1)[0] > 0.5)


def _lift_and_slope(p, angle, slope: bool = True):
    """(lift coefficient, its slope d C / d angle or None) at an angle of attack: a parabola
    peaking at the stall angle at the wing's maximum; past the stall the flat plate's return
    (GameConfig.post_stall, `_post_stall_constants`: a dip, about the flat plate's from twice
    the dip width on, 0 at 90 deg), or with the regime off the parabola's tail (0 from twice
    the stall angle; its slope negative past the stall, zero from twice the stall angle: v2's
    lift_slope, src/bnb/game/physics/flight.py)."""
    steepness = p.lift * p.inverse_stall
    if not _switched_on(p.post_stall):
        ratio = (angle * p.inverse_stall).clamp(-2.0, 2.0)
        lift = p.lift * (2.0 * ratio - ratio * ratio.abs())
        if not slope:
            return lift, None
        size = ratio.abs()
        return lift, torch.where(size < 2.0, steepness * (2.0 - 2.0 * size), 0.0)
    ratio = (angle * p.inverse_stall).clamp(-1.0, 1.0)
    size = ratio.abs()
    below = p.lift * ratio * (2.0 - size)
    magnitude = angle.abs()
    past_stall = magnitude > p.stall
    beyond = magnitude > _FORTY_FIVE
    toward = ((magnitude - p.stall) * p.inverse_tail_span).clamp(0.0, 1.0)
    drop = p.lift - p.flat_plate_lift
    double = 2.0 * magnitude
    tail = torch.where(
        beyond, p.flat_plate_lift * torch.sin(double), p.lift - drop * smootherstep(toward)
    )
    lift = torch.where(past_stall, torch.sign(angle) * tail, below)
    if not slope:
        return lift, None
    smooth_slope = 30.0 * (toward * (1.0 - toward)) ** 2  # d smootherstep / d t
    tail_slope = torch.where(
        beyond,
        2.0 * p.flat_plate_lift * torch.cos(double),
        -drop * smooth_slope * p.inverse_tail_span,
    )
    return lift, torch.where(past_stall, tail_slope, steepness * (2.0 - 2.0 * size))


_FORTY_FIVE = math.pi / 4.0


def _lift_coefficient(p, angle):
    """The lift coefficient at an angle of attack (`_lift_and_slope`)."""
    return _lift_and_slope(p, angle, slope=False)[0]


def _path_lift(p, lift, angle, vertical, speed_squared, thrust, thrust_normal):
    """The lateral acceleration of the normal force: the lift plus the thrust's normal part
    (thrust_normal = thrust x sin angle; the thrust's size never depends on the angle, its part
    along the path, thrust x cos angle, drives the speed in `_rates`).

    Below the stall the angle of attack is the turn's own: lift plus thrust_normal, the held
    vertical carried apart as induced drag. Past it (GameConfig.post_stall) the angle becomes
    the wing's whole, so the held vertical comes out of the normal force first, like the lift
    (author 2026-11-26: one force budget): lateral^2 = normal^2 - share x vertical^2, share a
    smootherstep over card.POST_STALL_SHARE_WIDTH past the stall; what is left turns the path.
    Where the normal force barely carries the vertical the square is floored smoothly,
    (x + sqrt(x^2 + e^2)) / 2, e = card.PATH_LIFT_SOFTNESS^2 x share x vertical^2 (0 below the
    stall, so ordinary flight is exact), so the rate has no infinite slope where it reaches 0.
    The turn is never more than the envelope's at that speed (its lateral lift, the wing's
    maximum lift against the g limit less the held vertical, plus its thrust tilted by its own
    angle). Returns (lateral, its share of the normal force): the share scales the path's
    response to the angle (`_kick`'s kappa)."""
    normal = lift * speed_squared + thrust_normal
    if not _switched_on(p.post_stall):
        return normal, 1.0
    share = smootherstep(((angle.abs() - p.stall) * p.inverse_share_width).clamp(0.0, 1.0))
    held = vertical * vertical
    taken = share * held
    left = normal * normal - taken
    softness = card.PATH_LIFT_SOFTNESS**2 * taken
    lateral = torch.sqrt(0.5 * (left + torch.sqrt(left * left + softness * softness)))
    # Never more than the envelope's turn at this speed: its lift's lateral share plus its
    # thrust's (the thrust tilted by the envelope's own angle).
    most = softmin(p.lift * speed_squared, p.max_acceleration, p.lift_softmin_exponent)
    envelope_lateral = torch.sqrt((most * most - held).clamp_min(0.0))
    reach = (envelope_lateral / (p.lift * speed_squared)).clamp(0.0, 1.0)
    envelope_angle = p.stall * (1.0 - torch.sqrt(1.0 - reach))
    turn = torch.minimum(lateral, envelope_lateral + thrust * torch.sin(envelope_angle))
    fraction = (turn / normal.abs().clamp_min(1e-6)).clamp_max(1.0)
    return torch.sign(normal) * turn, fraction


def _commanded_angle(p, control, lateral, speed, inverse_lift, vertical):
    """The angle of attack the stick asks for (signed): the one whose lift (below the stall)
    gives the setting's lateral acceleration; for full stick (control.deep) below corner it
    opens toward the deep angle that still holds the vertical (`_holding_angle`: coasting, the
    card's deep_stall_angle from stall speed / sqrt(POST_STALL_LIFT_FRACTION) up, shrinking
    smoothly to the stall angle at stall speed), a smootherstep from deep_stall_start x corner
    speed (the lift's angle there, so no jump) down over deep_stall_band of corner speed
    (GameConfig.post_stall). Full stick never gives up the 1 g (author 2026-11-26). vertical:
    the setting's held vertical (`_setting`)."""
    reach = (lateral.abs() / (speed * speed) * inverse_lift).clamp(0.0, 1.0)
    angle = p.stall * (1.0 - torch.sqrt(1.0 - reach))
    if _switched_on(p.post_stall):
        below = (
            (p.deep_stall_start - speed * p.inverse_corner_speed) * p.inverse_deep_stall_band
        ).clamp(0.0, 1.0)
        opening = smootherstep(below) * control.deep
        angle = angle + (_holding_angle(p, speed, vertical) - angle) * opening
    return control.stick_sign * angle


def _holding_angle(p, speed, vertical):
    """Full stick's deep angle at this speed: the card's deep_stall_angle from the speed where
    that angle's wing lift holds the vertical (coasting 1 g: stall speed x sqrt(C_max / C(deep
    angle)), stall speed / sqrt(POST_STALL_LIFT_FRACTION) at 45 deg) up, shrinking to the stall
    angle at the speed where the wing's maximum lift holds it (stall speed coasting) by a
    smootherstep in speed, the stall angle below (author 2026-11-26: full stick never gives up
    the 1 g). The vertical is the setting's, at most 1 g (a climb's extra comes after the turn):
    a dive's lower demand moves the span down by sqrt(vertical / 1 g); with the 1 g span a dive's
    line rose about 1.2 deg/s toward the stall near 98 m/s (P-51D). In between the normal force
    (the lift plus thrust x sin angle, `_path_lift`) holds the vertical; the wing alone dips a
    little below it near the top of the span, where the thrust's part carries the rest. The
    exact deepest angle whose lift alone holds it has an infinite slope in speed at both ends
    (the lift past the stall is a smootherstep in the angle), so the turn line had a corner."""
    held = minimum(vertical, card.GRAVITY).clamp_min(1e-6)
    equivalent = speed * torch.sqrt(card.GRAVITY / held)  # the coasting speed of the same load
    closing = ((equivalent - p.holding_start_speed) * p.inverse_holding_span).clamp(0.0, 1.0)
    return p.stall + (p.deep_stall_angle - p.stall) * smootherstep(closing)


def _matrix_exponential(a11, a12, a21, a22, duration):
    """exp(duration * A) for 2x2 A given by its entries (E, N): a degree-4 series, squared once."""
    b = torch.stack([torch.stack([a11, a12], -1), torch.stack([a21, a22], -1)], -2) * (
        duration / 2.0
    )
    eye = _eye(b.dtype, b.device)
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
    """Advance speed, angle of attack, nose rate and energy by duration at a fixed position.
    With no medium inverse_index and the gradient are None (n = 1, no gradient)."""
    speed0, angle0, cos0, sin0, energy, nose_rate = (
        s.speed,
        s.angle,
        s.angle_cos,
        s.angle_sin,
        s.energy,
        s.nose_rate,
    )
    floor = card.SPEED_FLOOR
    safe0 = speed0.clamp_min(floor)
    inverse_lift = 1.0 / p.lift
    inverse_lift_squared = inverse_lift * inverse_lift
    first = _setting(p, control, safe0, energy)
    lift0, slope = _lift_and_slope(p, angle0)
    speed_rate0, energy_rate0 = _rates(
        p, first, inverse_index, speed0, lift0, cos0, 1.0 / (safe0 * safe0), inverse_lift_squared
    )
    speed = (speed0 + 0.5 * duration * speed_rate0).clamp_min(floor)
    current = _setting(p, control, speed, energy + 0.5 * duration * energy_rate0)
    lateral, _, vertical, thrust = current
    speed_squared = speed * speed
    inverse_speed_squared = 1.0 / speed_squared
    path_lift, fraction = _path_lift(
        p, lift0, angle0, vertical, speed_squared, thrust, thrust * sin0
    )
    if inverse_index is None:
        over = 1.0 / speed
        path_rate = path_lift * over
    else:
        over = inverse_index / speed
        path_rate = path_lift * over + (speed * inverse_index) * (
            cos0 * gradient_across + sin0 * gradient_along
        )
    # kappa: the flight path's own response to the angle (the lift's slope: negative past the
    # stall). The nose is a spring toward the commanded angle with the angle's total damping
    # ratio the card's nose_damping at every angle (author 2026-11-26, fly-by-wire: the nose's
    # own damping is what the path's response leaves), so it never runs away past the stall.
    kappa = (slope * speed_squared + thrust * cos0) * fraction * over
    omega = (p.nose_frequency * speed).clamp_min(card.NOSE_FREQUENCY_FLOOR)
    if inverse_index is not None:
        omega = omega * inverse_index
    damping = 2.0 * p.nose_damping * omega - kappa
    step_angle = _commanded_angle(p, control, lateral, speed, inverse_lift, vertical) - angle0
    m11, m12, m21, m22 = _matrix_exponential(
        -kappa,
        filled((), 1.0, kappa.dtype, kappa.device).expand_as(kappa),
        damping * kappa - omega * omega,
        -damping,
        duration,
    )
    steady = kappa * step_angle
    offset = (nose_rate - path_rate) - steady
    turn = step_angle - m11 * step_angle + m12 * offset
    new_rate = path_rate + steady - m21 * step_angle + m22 * offset
    half_cos, half_sin = small_angle_cos_sin(0.5 * turn)
    middle_cos = cos0 * half_cos - sin0 * half_sin
    middle_lift = _lift_coefficient(p, angle0 + 0.5 * turn)
    middle_drag = _post_stall_drag(p, middle_cos)
    speed_rate1, _ = _rates(
        p,
        current,
        inverse_index,
        speed0,
        middle_lift,
        middle_cos,
        inverse_speed_squared,
        inverse_lift_squared,
        middle_drag,
    )
    speed_rate2, energy_rate2 = _rates(
        p,
        current,
        inverse_index,
        (speed0 + 0.5 * duration * speed_rate1).clamp_min(floor),
        middle_lift,
        middle_cos,
        inverse_speed_squared,
        inverse_lift_squared,
        middle_drag,
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
    thrust_scale: Tensor | float = 1.0,
):
    """One tick of flight for every slot (E, N): half a kick, a drift, half a kick.

    s carries x, y, heading_cos, heading_sin, speed, angle, angle_cos, angle_sin, nose_rate,
    energy and the medium at the ship (log_index, gradient_x, gradient_y); it is updated in
    place and returned. fields: (E, 5, F); with none (F = 0) the medium's arithmetic is
    skipped (n = 1, no gradient: the same values). stick, power: (E, N) long action indices.
    thrust_scale: the thrust's factor (firing cuts it).
    """
    p = constants
    index = stick[:, None]
    signed = p.stick_table.gather(1, index)[:, 0]
    control = SimpleNamespace(
        thrust_scale=thrust_scale,
        stick_size=signed.abs(),
        stick_sign=torch.sign(signed),
        stick_cap=p.stick_cap_table.gather(1, index)[:, 0],
        deep=p.deep_stall_table.gather(1, index)[:, 0],
        commanded=p.commanded_table.gather(1, power[:, None])[:, 0],
        demand=p.demand_table.gather(1, power[:, None])[:, 0],
    )
    h = tick_length
    if fields.shape[-1] == 0:  # no medium (agreed fix 8)
        s = _kick(p, control, s, None, None, None, 0.5 * h)
        velocity_x, velocity_y = s.speed * s.angle_cos, -s.speed * s.angle_sin
    else:
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
    if fields.shape[-1] == 0:
        return _kick(p, control, s, None, None, None, 0.5 * h)
    s.log_index, s.gradient_x, s.gradient_y = medium(fields, s.x, s.y)
    return _kick(p, control, s, torch.exp(-s.log_index), *_nose_frame(s), 0.5 * h)


def path_velocity(
    speed: Tensor, heading_cos: Tensor, heading_sin: Tensor, angle_cos: Tensor, angle_sin: Tensor
) -> tuple[Tensor, Tensor]:
    """The velocity at proper speed along the flight path: the nose turned back by the angle of
    attack (State's velocity rows)."""
    return (
        speed * (heading_cos * angle_cos + heading_sin * angle_sin),
        speed * (heading_sin * angle_cos - heading_cos * angle_sin),
    )


def with_velocity(state: State) -> State:
    """The state with its stored velocity recomputed from speed, heading and angle of attack,
    after the motion was set by hand."""
    velocity_x, velocity_y = path_velocity(
        state.speed, state.heading_cos, state.heading_sin, state.angle_cos, state.angle_sin
    )
    return state.replace(velocity_x=velocity_x, velocity_y=velocity_y)


# --------------------------------------------------------------------------- bullets


def _half_bend(tick_length, direction_x, direction_y, speed, index, gradient_x, gradient_y):
    """The direction turned toward the higher index by half a tick."""
    turn = (
        0.5 * tick_length * (speed / index) * (direction_x * gradient_y - direction_y * gradient_x)
    )
    cos, sin = small_angle_cos_sin(turn)
    return direction_x * cos - direction_y * sin, direction_x * sin + direction_y * cos


def bullet_motion(fields, tick_length, x, y, direction_x, direction_y, speed, drag, start_medium):
    """One move of every bullet (E, ...): half drag, half bend, the step, half bend, half
    drag. fields: (E, 5, F). drag: the shooter's drag constant, 1/m, a number or broadcast
    against the bullets. start_medium: (ln n, d/dx ln n, d/dy ln n) at the start (cached).
    Returns x, y, direction x and y, speed and the medium at the end (three tensors). With no
    fields the medium's arithmetic is skipped (n = 1, no bend: the same values)."""
    log_index, gradient_x, gradient_y = start_medium
    half = drag * 0.5 * tick_length
    if fields.shape[-1] == 0:  # no medium (agreed fix 8)
        speed = speed / (1.0 + half * speed)
        step = speed * tick_length
        x, y = x + direction_x * step, y + direction_y * step
        speed = speed / (1.0 + half * speed)
        return x, y, direction_x, direction_y, speed, log_index, gradient_x, gradient_y
    index = torch.exp(log_index)
    speed = speed / (1.0 + half * speed / index)
    direction_x, direction_y = _half_bend(
        tick_length, direction_x, direction_y, speed, index, gradient_x, gradient_y
    )
    through = speed / index
    correction = 0.5 * tick_length * through * (direction_x * gradient_x + direction_y * gradient_y)
    step = (
        through * tick_length / (1.0 + correction * (1.0 + correction * (0.5 + correction / 6.0)))
    )
    x, y = x + direction_x * step, y + direction_y * step
    log_index, gradient_x, gradient_y = medium(fields, x, y)
    index = torch.exp(log_index)
    direction_x, direction_y = _half_bend(
        tick_length, direction_x, direction_y, speed, index, gradient_x, gradient_y
    )
    speed = speed / (1.0 + half * speed / index)
    return x, y, direction_x, direction_y, speed, log_index, gradient_x, gradient_y


# --------------------------------------------------------------------------- the tick


def draw_noise(
    config: GameConfig, envs: int, slots: int, generator: torch.Generator, device
) -> dict[str, Tensor]:
    """The random draws one tick needs: "spread" (E, N, 2) standard normals."""
    return {"spread": torch.randn((envs, slots, 2), **random_keywords(generator, device))}


def tick(config: GameConfig, state: State, action: Tensor, generator) -> tuple[State, StepOut]:
    """One tick with its noise drawn (draw_noise from `generator`, torch's default generator of
    the state's device when None), then step: the game half of every caller's tick (training's
    rollout, the evaluation runner, the sandbox)."""
    envs, slots = state.exists.shape
    noise = draw_noise(config, envs, slots, generator, state.exists.device)
    return step(config, state, action, noise)


def step(
    config: GameConfig, state: State, action: Tensor, noise: dict[str, Tensor]
) -> tuple[State, StepOut]:
    """One tick. action: (E, N, 3) long [stick, power, fire], every ship's action chosen on
    this tick; noise from draw_noise.

    The one-tick action delay: the tick applies each ship's pending action (state.pending,
    chosen on the previous tick; neutral after a deal) and `action` becomes the new pending
    action, applied by the next tick. Slots that are not alive at tick start neither act nor
    change (their pending action is kept as given, never applied). The returned state of a
    finished environment (StepOut.done) is the final state of its episode.
    """
    E, _, N, K = state.bullets.shape
    device = state.ships.device
    index = slot_indices(N, device)
    applied = state.pending
    stick, power, shoot = applied[..., 0], applied[..., 1], applied[..., 2] == 1
    alive0 = state.alive

    shots = _fire(config, state, shoot, noise["spread"], index)
    flown = _fly(config, state, stick, power, shots.fired)
    flying = shots.counts[:, MOVES_LEFT] > 0
    bullets = _move_bullets(config, state, shots.bullets, flying)
    contact = _hit_test(state, flown, shots.bullets, bullets, flying, index)
    real = shots.bullet_real.reshape(E, N * K)
    bullet_damage = _damage(state, flown, bullets, real, contact)
    no_damage = filled((E, N), 0.0, bullet_damage.dtype, device)
    hit_taken = no_damage.scatter_add(1, contact.target, bullet_damage)
    boundary = _boundary_damage(config, flown, alive0)
    # Collisions (after the move, with the bullet hits and the boundary, before the shields):
    # each ship's collision damage is credited to the other ship of the pair.
    crash = collisions(
        state, alive0, config.tick_length, flown.x, flown.y, flown.heading_cos, flown.heading_sin
    ).damage
    crash_taken = crash.sum(-1)
    total = hit_taken + boundary + crash_taken
    died, shield, restored, wait = _shields(config, state, total, stick, power, shoot)
    alive = alive0 & ~died
    recharged = shield + restored
    tick, done, result = _end_check(config, state, alive, recharged)
    enemy = state.team[:, index.owner] != _on_target(state.team, contact.target)
    moves_left, shot_outcome = _bullet_ends(shots.counts, flying, contact.hit, enemy)

    # Slots not alive at tick start keep their whole row (agreed fix 3: one select, not one per
    # field).
    ships = torch.where(
        alive0[:, None],
        torch.stack(
            [
                flown.x,
                flown.y,
                flown.heading_cos,
                flown.heading_sin,
                flown.speed,
                flown.angle,
                flown.angle_cos,
                flown.angle_sin,
                flown.nose_rate,
                flown.energy,
                recharged,
                wait,
                (shots.cooldown - 1).clamp_min(0),
                flown.log_index,
                flown.gradient_x,
                flown.gradient_y,
                *path_velocity(
                    flown.speed,
                    flown.heading_cos,
                    flown.heading_sin,
                    flown.angle_cos,
                    flown.angle_sin,
                ),
                shots.next_slot,
            ],
            1,
        ),
        state.ships,
    )
    fired_tick = shots.counts[:, FIRED_TICK]
    new_state = dataclasses.replace(
        state,
        ships=ships,
        flags=torch.stack([alive, state.exists, shots.fired], 1),
        bullets=bullets,
        bullet_counts=torch.stack([moves_left, fired_tick], 1),
        bullet_real=shots.bullet_real,
        tick=tick,
        pending=action,
    )
    out = StepOut(
        damage_taken=total,
        died=died,
        done=done,
        result=result,
        shot_outcome=shot_outcome,
        shot_tick=fired_tick,
        hit_damage=filled((E, N * N), 0.0, bullet_damage.dtype, device)
        .scatter_add(1, index.pair + contact.target, bullet_damage)
        .reshape(E, N, N)
        + crash.transpose(1, 2),
        boundary_damage=boundary,
        shield_restored=torch.where(alive0, restored, 0.0),
    )
    if config.fog:
        # A shot, a real bullet's hit or a collision reveals the ship for this tick (boundary
        # damage does not).
        real_hit = (contact.hit & real).to(bullet_damage.dtype)
        real_hits = no_damage.scatter_add(1, contact.target, real_hit)
        revealed = shots.fired | (real_hits > 0.0) | (crash_taken > 0.0)
        new_state = perceive(new_state, revealed, config.fog_tracking)
    return new_state, out


def _per_bullet(values: Tensor | float) -> Tensor | float:
    """A per-ship constant (E, N) laid against its ship's K bullets (E, N, K); a number as it is."""
    return values[..., None] if isinstance(values, Tensor) else values


def _on_target(values: Tensor | float, target: Tensor) -> Tensor | float:
    """A per-ship value (E, N) read at each bullet's target (E, N K); a number as it is (a
    constant equal for every ship needs no gather)."""
    return values.gather(1, target) if isinstance(values, Tensor) else values


class Shots(NamedTuple):
    """The fire stage's result: who fired a real shot, and the gun and bullet blocks after it."""

    fired: Tensor  # (E, N) bool
    bullets: Tensor  # (E, BULLET_COLUMNS, N, K) with the new bullets written
    counts: Tensor  # (E, BULLET_COUNT_COLUMNS, N, K) long
    bullet_real: Tensor  # (E, N, K) bool
    cooldown: Tensor  # (E, N), the gun's whole ticks to wait, before this tick's count down
    next_slot: Tensor  # (E, N)


def _fire(
    config: GameConfig, state: State, shoot: Tensor, spread: Tensor, index: SimpleNamespace
) -> Shots:
    """Fire, from the tick-start state. A living ship with the trigger held and the gun ready
    fires a real shot (a shot costs no store). The new bullet goes into the ship's next slot:
    with ghost bullets every living ship writes one every tick (a ghost unless it fired), else
    only real shots. A new bullet starts at its ship with the ship's cached medium, at the
    ship's velocity plus the muzzle speed along the nose and the spread."""
    c = state.constants
    E, _, _, K = state.bullets.shape
    alive = state.alive
    # Below the band's top the store pays (part of) the shot's cost: an empty store blocks it.
    blocked = (state.speed < c["stall_speed"] * (1.0 + config.fire_store_band)) & (
        state.energy <= 0.0
    )
    fired = alive & shoot & (state.cooldown == 0) & ~blocked
    launch_x = (
        state.velocity_x
        + c["muzzle_speed"] * state.heading_cos
        + c["spread_speed"] * spread[..., 0]
    )
    launch_y = (
        state.velocity_y
        + c["muzzle_speed"] * state.heading_sin
        + c["spread_speed"] * spread[..., 1]
    )
    launch_speed = torch.sqrt(launch_x * launch_x + launch_y * launch_y)
    emits = alive if config.ghost_bullets else fired
    written = emits[..., None] & (index.bullet_slot == state.next_slot[..., None])
    launched = torch.cat(
        [
            state.ships[:, X : Y + 1],
            torch.stack([launch_x / launch_speed, launch_y / launch_speed, launch_speed], 1),
            state.ships[:, LOG_INDEX : GRADIENT_Y + 1],
        ],
        1,
    )
    moves = filled((E,), c["bullet_moves"], torch.long, state.ships.device)
    new_counts = torch.stack([moves, state.tick], 1)
    return Shots(
        fired=fired,
        bullets=torch.where(written[:, None], launched[..., None], state.bullets),
        counts=torch.where(written[:, None], new_counts[:, :, None, None], state.bullet_counts),
        bullet_real=torch.where(written, fired[..., None], state.bullet_real),
        cooldown=torch.where(fired, c["fire_interval"], state.cooldown),
        next_slot=torch.where(emits, (state.next_slot + 1) % K, state.next_slot),
    )


def _fly(
    config: GameConfig, state: State, stick: Tensor, power: Tensor, fired: Tensor
) -> SimpleNamespace:
    """Flight for every slot (a slot not alive at tick start is flown too and its row dropped by
    the step's one select; a dead ship's flight is never read: it is no target and takes no
    boundary damage). Firing withholds 1 - fire_thrust_fraction of the thrust on the tick it
    fires: paid by the speed (the thrust cut) above the store band, by the store (the thrust
    kept, the withheld thrust's power x the tick taken from the store) at stall speed and below,
    split by a smootherstep across the band (GameConfig.fire_store_band)."""
    c = state.constants
    stall = c["stall_speed"]
    over = (state.speed - stall) / (stall * config.fire_store_band)
    by_speed = smootherstep(over.clamp(0.0, 1.0))
    withheld = torch.where(fired, 1.0 - config.fire_thrust_fraction, 0.0)
    speed = state.speed.clamp_min(card.SPEED_FLOOR)
    thrust = softmin(c["power"] / speed, c["static_thrust"], c["thrust_softmin_exponent"])
    store_cost = withheld * (1.0 - by_speed) * thrust * speed * config.tick_length
    flown = flight_tick(
        state.constants,
        state.fields,
        config.tick_length,
        SimpleNamespace(
            x=state.x,
            y=state.y,
            heading_cos=state.heading_cos,
            heading_sin=state.heading_sin,
            speed=state.speed,
            angle=state.angle,
            angle_cos=state.angle_cos,
            angle_sin=state.angle_sin,
            nose_rate=state.nose_rate,
            energy=state.energy,
            log_index=state.log_index,
            gradient_x=state.gradient_x,
            gradient_y=state.gradient_y,
        ),
        stick,
        power,
        thrust_scale=1.0 - withheld * by_speed,
    )
    flown.energy = (flown.energy - store_cost).clamp_min(0.0)
    return flown


def _move_bullets(config: GameConfig, state: State, start: Tensor, flying: Tensor) -> Tensor:
    """Every flying bullet's move (bullet_motion); inactive bullets are frozen. start: the bullet
    block after the fire stage; returns the block after the move."""
    x, y, direction_x, direction_y, speed = start[:, :BULLET_LOG_INDEX].unbind(1)
    moved = bullet_motion(
        state.fields,
        config.tick_length,
        x,
        y,
        direction_x,
        direction_y,
        speed,
        _per_bullet(state.constants["bullet_drag"]),
        start[:, BULLET_LOG_INDEX:].unbind(1),
    )
    return torch.where(flying[:, None], torch.stack(moved, 1), start)


class Contact(NamedTuple):
    """The hit test's result, per bullet (E, N K) and per bullet and ship (E, [2,] N K, N)."""

    target: Tensor  # (E, N K) long: the ship of the first contact (slot 0 for a miss)
    hit: Tensor  # (E, N K) bool
    start: Tensor  # (E, 2, N K, N): bullet minus ship at tick start, x and y
    relative: Tensor  # (E, 2, N K, N): the bullet's move minus the ship's
    relative_squared: Tensor  # (E, N K, N), clamped away from 0


def _hit_test(
    state: State,
    flown: SimpleNamespace,
    start_bullets: Tensor,
    bullets: Tensor,
    flying: Tensor,
    index: SimpleNamespace,
) -> Contact:
    """The swept test between the two straight motions, every bullet against every ship
    ((E, 2, N K, N), x and y on the second axis). A contact is a closest approach within the hit
    radius of a ship alive at tick start (a dead ship is no target) that is not the bullet's
    owner, by a flying bullet. First contact: the smallest tau, ties within 1e-6 to the closer
    pass."""
    E, _, N, K = bullets.shape
    per_bullet = (E, 2, N * K, 1)
    ships = state.ships[:, X : Y + 1]
    start_xy = start_bullets[:, BULLET_X : BULLET_Y + 1]
    start = start_xy.reshape(per_bullet) - ships[:, :, None, :]
    moved = torch.stack([flown.x, flown.y], 1) - ships
    relative = (bullets[:, BULLET_X : BULLET_Y + 1] - start_xy).reshape(per_bullet)
    relative = relative - moved[:, :, None, :]
    relative_squared = (relative * relative).sum(1).clamp_min(1e-12)
    tau = (-(start * relative).sum(1) / relative_squared).clamp(0.0, 1.0)
    miss = start + tau[:, None] * relative
    miss_squared = (miss * miss).sum(1)
    reach = torch.where(state.alive, state.constants["hit_radius_squared"], -1.0)
    contact = (miss_squared < reach[:, None, :]) & index.other & flying.reshape(E, N * K, 1)
    contact_tau = torch.where(contact, tau, torch.inf)
    first_tau = contact_tau.min(-1).values
    tied = contact & (contact_tau <= first_tau[..., None] + 1e-6)
    target = torch.where(tied, miss_squared, torch.inf).argmin(-1)
    return Contact(target, first_tau < torch.inf, start, relative, relative_squared)


def _damage(
    state: State, flown: SimpleNamespace, bullets: Tensor, real: Tensor, contact: Contact
) -> Tensor:
    """Each real bullet's damage on its target, (E, N K) (0 for a miss or a ghost): kinetic (the
    owner's hit damage, the explosive share plus the rest by the squared ratio of the bullet's
    speed to its reference speed) x closeness (the impact parameter, the distance from the
    target to the bullet's line, across the band between the core and hit radii: `graze`) x
    aspect (the target's nose cone at the hit angle, the bullet's path against its nose:
    `aspect_factor`)."""
    c = state.constants
    E, _, N, K = bullets.shape
    target = contact.target
    flat = (E, N * K)
    direction_x, direction_y, bullet_speed = bullets[:, BULLET_DIR_X : BULLET_SPEED + 1].unbind(1)
    flat_direction_x, flat_direction_y = direction_x.reshape(flat), direction_y.reshape(flat)
    nose_cos = _on_target(flown.heading_cos, target)
    nose_sin = _on_target(flown.heading_sin, target)
    # The owner's constants broadcast over its K bullets.
    speed_ratio = bullet_speed * _per_bullet(c["inverse_reference_bullet_speed"])
    explosive = _per_bullet(c["explosive_share"])
    kinetic = explosive + (1.0 - explosive) * speed_ratio * speed_ratio
    full = (_per_bullet(c["hit_damage"]) * kinetic).reshape(flat)
    # The swept test's start and relative motion against the target.
    on_target = target[:, None, :, None].expand(E, 2, -1, 1)
    target_start_x, target_start_y = contact.start.gather(3, on_target)[..., 0].unbind(1)
    target_relative_x, target_relative_y = contact.relative.gather(3, on_target)[..., 0].unbind(1)
    cross = target_start_x * target_relative_y - target_start_y * target_relative_x
    line = cross * cross / contact.relative_squared.gather(2, target[..., None])[..., 0]
    impact_squared = torch.minimum(
        line, target_start_x * target_start_x + target_start_y * target_start_y
    )
    radius, core = _on_target(c["hit_radius"], target), _on_target(c["core_radius"], target)
    closeness = graze(impact_squared.sqrt(), core, radius)
    hit_cos = -(flat_direction_x * nose_cos + flat_direction_y * nose_sin)  # 1 head-on
    aspect = aspect_factor(
        hit_cos,
        _on_target(c["nose_inner_cos"], target),
        _on_target(c["inverse_nose_width"], target),
    )
    return torch.where(contact.hit & real, full * closeness * aspect, 0.0)


def graze(distance: Tensor, core: Tensor | float, radius: Tensor | float) -> Tensor:
    """The closeness of a pass `distance` from a centre: 1 inside `core`, a smoothstep down to 0
    at `radius` (a bullet: the target's core and hit radii; a collision: the two ships' summed
    core and hit radii)."""
    width = radius - core
    width = width.clamp_min(1e-6) if isinstance(width, Tensor) else max(width, 1e-6)
    band = ((distance - core) / width).clamp(0.0, 1.0)
    return 1.0 - band * band * (3.0 - 2.0 * band)


def aspect_factor(
    hit_cos: Tensor, nose_inner_cos: Tensor | float, inverse_nose_width: Tensor | float
) -> Tensor:
    """The nose cone (author 2026-11-25) at the cosine `hit_cos` of the angle off the nose (1
    dead ahead, -1 dead astern), with the ship's constants of those names: 0 inside the inner
    angle, 1 beyond the outer, a smootherstep of t = (cos inner - c) / (cos inner - cos outer)
    between (no acos). A bullet's damage factor (`_damage`, the bullet's path against the
    target's nose) and, as 1 - aspect / 2, a collision's blame (`collisions`). Shared with the
    scripted opponent's gun."""
    return smootherstep(((nose_inner_cos - hit_cos) * inverse_nose_width).clamp(0.0, 1.0))


class Collisions(NamedTuple):
    """The collision test per ordered pair (E, me, other)."""

    contact: Tensor  # bool: the pair first touches in this tick, both alive (symmetric)
    closing: Tensor  # m/s: |w|, the relative speed
    miss: Tensor  # m: b, the closest distance of the tick's relative line, extended
    nose_cos: Tensor  # cos(my nose, the line of centres at first touch, me to the other)
    blame: Tensor  # 1 - aspect / 2: 1 inside my nose cone's inner angle, 0.5 beyond the outer
    damage: Tensor  # the damage I take (0 without a contact)


@built_once
def _not_self(slots: int, device: torch.device) -> Tensor:
    return ~torch.eye(slots, dtype=torch.bool, device=device)


def collisions(
    state: State,
    alive: Tensor,
    tick_length: float,
    end_x: Tensor,
    end_y: Tensor,
    nose_cos: Tensor,
    nose_sin: Tensor,
) -> Collisions:
    """Every pair of living ships (any teams), on straight moves from the tick-start positions
    to (end_x, end_y) (E, N), the noses (nose_cos, nose_sin) after the move. p the gap (other
    minus me), w the relative move over the tick, R the two hit radii, R_core the two core
    radii. A pair is scored once per overlap, on the tick of first touch (author 2026-11-26):
    |p| >= R at tick start and the swept move p + s w, 0 <= s <= 1, comes within R. A pair
    that starts a tick inside R is not scored; it must start a tick at or beyond R again
    first. b = |m|, m = p + tau w, tau = -p.w / |w|^2 unclamped: the closest distance of the
    tick's relative line, extended past the tick (the pass's predicted closest approach). The
    line of centres at first touch n = m / R - sqrt(max(0, 1 - b^2 / R^2)) u, u = w / |w|; my
    blame 1 - aspect(nose . n) / 2 (aspect_factor: 1 inside the nose cone's inner angle, 0.5
    beyond the outer). I take max shield x (|w| / tick) / (2 x cruise) x graze(b; R_core, R)
    x blame: linear in the closing speed, a full shield's worth head-on at cruise with full
    blame; full closeness when the cores would meet, a graze when only the hit radii do."""
    c = state.constants
    N = end_x.shape[1]
    x, y = state.x, state.y
    move_x, move_y = end_x - x, end_y - y
    gap_x = x[:, None, :] - x[:, :, None]  # (E, me, other): other minus me
    gap_y = y[:, None, :] - y[:, :, None]
    relative_x = move_x[:, None, :] - move_x[:, :, None]
    relative_y = move_y[:, None, :] - move_y[:, :, None]
    relative_squared = (relative_x * relative_x + relative_y * relative_y).clamp_min(1e-12)
    tau = -(gap_x * relative_x + gap_y * relative_y) / relative_squared
    # |tau w| <= |p|: the line's closest point stays bounded as w goes to 0.
    miss_x, miss_y = gap_x + tau * relative_x, gap_y + tau * relative_y
    miss_squared = miss_x * miss_x + miss_y * miss_y
    clamped = tau.clamp(0.0, 1.0)
    swept_x, swept_y = gap_x + clamped * relative_x, gap_y + clamped * relative_y
    swept_squared = swept_x * swept_x + swept_y * swept_y
    start_squared = gap_x * gap_x + gap_y * gap_y
    radius, core = c["hit_radius"], c["core_radius"]
    reach = _pair_sum(radius)
    core_reach = _pair_sum(core)
    reach_squared = reach * reach
    inverse_reach = 1.0 / reach
    inverse_relative = torch.rsqrt(relative_squared)
    inside = (1.0 - miss_squared * inverse_reach * inverse_reach).clamp_min(0.0).sqrt()
    back = inside * inverse_relative  # along -u, over the relative move's length
    line_x = miss_x * inverse_reach - back * relative_x
    line_y = miss_y * inverse_reach - back * relative_y
    cosine = nose_cos[:, :, None] * line_x + nose_sin[:, :, None] * line_y
    blame = 1.0 - 0.5 * aspect_factor(
        cosine, _per_me(c["nose_inner_cos"]), _per_me(c["inverse_nose_width"])
    )
    miss = miss_squared.sqrt()
    closing = relative_squared * inverse_relative / tick_length
    contact = (
        (start_squared >= reach_squared)
        & (swept_squared < reach_squared)
        & alive[:, :, None]
        & alive[:, None, :]
        & _not_self(N, end_x.device)
    )
    scale = _per_me(c["max_shield"] * c["inverse_collision_speed"])
    closeness = graze(miss, core_reach, reach)
    damage = scale * closing * closeness * blame
    return Collisions(contact, closing, miss, cosine, blame, torch.where(contact, damage, 0.0))


def _per_me(values: Tensor | float) -> Tensor | float:
    """A per-ship value (E, N) laid along the "me" axis of (E, me, other) pairs."""
    return values[:, :, None] if isinstance(values, Tensor) else values


def _pair_sum(values: Tensor | float) -> Tensor | float:
    """A per-ship value summed over each pair (E, me, other); a number doubled."""
    return values[:, :, None] + values[:, None, :] if isinstance(values, Tensor) else 2.0 * values


def _boundary_damage(config: GameConfig, flown: SimpleNamespace, alive: Tensor) -> Tensor:
    """(E, N): damage from the arena boundary after the move, rising by a smootherstep over the
    ramp width beyond the arena radius to its maximum (per second); none for a dead ship."""
    depth = torch.sqrt(flown.x * flown.x + flown.y * flown.y) - config.arena_radius
    ramp = smootherstep((depth / config.boundary_ramp_width).clamp(0.0, 1.0))
    return torch.where(alive, card.BOUNDARY_MAX_DAMAGE * ramp * config.tick_length, 0.0)


def _shields(
    config: GameConfig, state: State, damage: Tensor, stick: Tensor, power: Tensor, shoot: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The tick's damage (E, N) against the shields. A living ship hurt with its shield already
    at 0 dies; the shield takes the damage down to 0; a ship not hurt and quiet (coast, at most
    the gradual stick, no trigger) recharges once its wait has run out; being hurt or not quiet
    restarts the wait. Returns died (E, N) bool, the shield after the damage, the recharge
    (restored) and the wait."""
    c = state.constants
    h = config.tick_length
    quiet = (power == 0) & (stick <= card.QUIET_STICK_MAX) & ~shoot
    hurt = damage > 0.0
    died = state.alive & hurt & (state.shield <= 0.0)
    shield = (state.shield - damage).clamp_min(0.0)
    recharge = torch.minimum(
        c["max_shield"] - shield, c["shield_rate"] * (h - state.shield_wait).clamp(0.0, h)
    )
    restored = torch.where(~hurt & quiet, recharge, 0.0)
    wait = torch.where(hurt | ~quiet, c["shield_delay"], (state.shield_wait - h).clamp_min(0.0))
    return died, shield, restored, wait


def _end_check(
    config: GameConfig, state: State, alive: Tensor, shield: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """The tick count after this tick, done (E,) and result (E,) (0 or 1 the winner, 2 a draw, -1
    not done). A wipe ends the game: the team with ships left wins, both wiped is a draw. At the
    clock: more living ships wins; equal living ships, more total shield over the living ships
    wins (the shield after this tick's recharge, `shield`); equal on both, a draw. A shield lead
    within float32 rounding of the sums is equal: the two sums add their ships in different
    orders, so an exact tie can differ by a rounding residue."""
    team_one = state.team == 1
    left_zero = (alive & ~team_one).sum(-1)
    left_one = (alive & team_one).sum(-1)
    wiped = (left_zero == 0) | (left_one == 0)
    wipe_result = torch.where(left_zero > 0, 0, torch.where(left_one > 0, 1, 2))
    living_shield = torch.where(alive, shield, 0.0)
    shield_zero = torch.where(team_one, 0.0, living_shield).sum(-1)
    shield_one = torch.where(team_one, living_shield, 0.0).sum(-1)
    lead = shield_zero - shield_one
    level = lead.abs() <= 1e-5 * (shield_zero + shield_one).clamp_min(1.0)
    by_shield = torch.where(level, 2, torch.where(lead > 0.0, 0, 1))
    clock_result = torch.where(
        left_zero == left_one, by_shield, torch.where(left_zero > left_one, 0, 1)
    )
    tick = state.tick + 1
    done = wiped | (tick >= config.clock_ticks)
    return tick, done, torch.where(done, torch.where(wiped, wipe_result, clock_result), -1)


def _bullet_ends(
    counts: Tensor, flying: Tensor, hit: Tensor, enemy: Tensor
) -> tuple[Tensor, Tensor]:
    """Each bullet's moves left after this tick (0 once it hits), and the outcome of every bullet
    whose flight ends this tick (its first contact, or its last move without one), (E, N, K):
    -1 none ended, 0 missed, 1 hit an enemy, 2 hit a team-mate. hit, enemy: (E, N K) bool."""
    shape = flying.shape
    hit = hit.reshape(shape)
    moves_left0 = counts[:, MOVES_LEFT]
    moves_left = torch.where(hit, 0, torch.where(flying, moves_left0 - 1, moves_left0))
    ended = hit | (flying & (moves_left == 0))
    outcome = torch.where(hit, torch.where(enemy.reshape(shape), 1, 2), 0)
    return moves_left, torch.where(ended, outcome, -1)


# --------------------------------------------------------------------------- reset


def _edge_safe(config: GameConfig, x: Tensor, y: Tensor, angle: Tensor) -> tuple[Tensor, Tensor]:
    """(cos, sin) of a spawn heading angle with the edge rule: a ship within card.SPAWN_EDGE_BAND
    of the boundary whose heading points outward (a positive component along the outward
    normal) has it mirrored in the tangent, so it points inward by as much as it pointed out.
    Elsewhere the heading is unchanged."""
    heading_cos, heading_sin = torch.cos(angle), torch.sin(angle)
    distance = torch.sqrt(x * x + y * y)
    inverse = 1.0 / distance.clamp_min(1e-6)
    normal_x, normal_y = x * inverse, y * inverse
    outward = heading_cos * normal_x + heading_sin * normal_y
    turned = (distance > config.arena_radius - card.SPAWN_EDGE_BAND) & (outward > 0.0)
    mirror = torch.where(turned, 2.0 * outward, 0.0)
    return heading_cos - mirror * normal_x, heading_sin - mirror * normal_y


def _spawn_places(config: GameConfig, team: Tensor, uniform) -> tuple[Tensor, Tensor, Tensor]:
    """Spawn positions x, y and heading angles (E, N) by config.spawn (GameConfig), before the
    edge rule. The spawn disc has radius card.SPAWN_AREA_FRACTION x the arena radius.

    "mixed": every ship uniform over the spawn disc, heading uniform. "sides": a random axis
    per environment; each team uniform over a half disc on its own side whose flat edge lies
    card.SIDE_GAP_FRACTION x the arena radius from the dividing line and whose radius is the
    spawn disc's less that gap (so it stays inside the spawn disc), noses along the axis
    toward the other side within card.SIDE_HEADING_JITTER."""
    envs, slots = team.shape
    radius_draw, polar_draw, heading_draw = (
        uniform(envs, slots),
        uniform(envs, slots),
        uniform(envs, slots),
    )
    polar = 2.0 * math.pi * polar_draw
    disc = card.SPAWN_AREA_FRACTION * config.arena_radius
    if config.spawn == "mixed":
        spread = disc * torch.sqrt(radius_draw)
        return spread * torch.cos(polar), spread * torch.sin(polar), 2.0 * math.pi * heading_draw
    if config.spawn != "sides":
        raise ValueError(f"spawn rule {config.spawn!r}: need 'sides' or 'mixed'")
    axis = 2.0 * math.pi * uniform(envs, 1)
    axis_cos, axis_sin = torch.cos(axis), torch.sin(axis)
    gap = card.SIDE_GAP_FRACTION * config.arena_radius
    spread = (disc - gap) * torch.sqrt(radius_draw)
    side = (2 * team - 1).to(torch.float32)  # team 0 on the axis's negative side
    along = side * (gap + (spread * torch.cos(polar)).abs())
    across = spread * torch.sin(polar)
    x, y = along * axis_cos - across * axis_sin, along * axis_sin + across * axis_cos
    toward = axis + torch.where(team == 1, math.pi, 0.0)
    return x, y, toward + card.SIDE_HEADING_JITTER * (2.0 * heading_draw - 1.0)


def _spawn_motion(
    config: GameConfig, constants: Constants, log_index: Tensor, uniform
) -> tuple[Tensor, Tensor, Tensor]:
    """Proper speed, store and shield (E, N) at a deal. Standard: the spawn speed over the map
    (card.SPAWN_SPEED times the refractive index), half the store, a full shield. With
    config.training_spawn, every environment not drawn standard (a share
    config.standard_spawn_share) draws per ship a proper speed uniform from the stall speed to
    the top level speed (cruise_speed), a store and a shield each uniform from 0 to full."""
    zeros = torch.zeros_like(log_index)
    speed = card.SPAWN_SPEED * torch.exp(log_index)
    energy = _per_slot(constants["spawn_energy"], zeros)
    shield = _per_slot(constants["max_shield"], zeros)
    if not config.training_spawn:
        return speed, energy, shield
    envs, slots = log_index.shape
    standard = uniform(envs, 1) < config.standard_spawn_share
    speed_draw, energy_draw, shield_draw = (
        uniform(envs, slots),
        uniform(envs, slots),
        uniform(envs, slots),
    )
    stall = constants["stall_speed"]
    drawn_speed = stall + (constants["cruise_speed"] - stall) * speed_draw
    return (
        torch.where(standard, speed, drawn_speed),
        torch.where(standard, energy, constants["max_energy"] * energy_draw),
        torch.where(standard, shield, constants["max_shield"] * shield_draw),
    )


MAP_SCALE_RADIUS = 2600.0  # m: the arena radius of map scale 1 (v2's rules.py)


def draw_fields(config: GameConfig, envs: int, generator, device) -> Tensor:
    """Each environment's fields, (E, 5, F) (v2's new_map; rows as State.fields): point-symmetric
    pairs,
    field P + i is field i rotated by 180 degrees about the centre, and for an odd count the
    last field sits at the centre."""
    count, pairs = config.field_count, config.field_count // 2
    scale = config.arena_radius / MAP_SCALE_RADIUS
    uniforms = torch.rand((envs, count, 4), **random_keywords(generator, device))
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


@built_once
def _lineup_types(lineup: tuple[str, ...], device: torch.device) -> Tensor:
    return torch.tensor([card.TYPE_NAMES.index(name) for name in lineup], device=device)


@built_once
def _no_bullets(envs: int, slots: int, device: torch.device) -> Tensor:
    """The bullet block of a fresh deal: every bullet inactive, at the origin, pointing along x."""
    bullets = torch.zeros((envs, len(BULLET_COLUMNS), slots, BULLET_SLOTS), device=device)
    bullets[:, BULLET_DIR_X] = 1.0
    return bullets


def _per_slot(value: Tensor | float, like: Tensor) -> Tensor:
    """A per-ship constant as an (E, N) tensor (a shared fixed one for a number)."""
    return (
        value if isinstance(value, Tensor) else filled(like.shape, value, like.dtype, like.device)
    )


def _uniform_draws(generator, device):
    """uniform(*shape): uniform draws on [0, 1) from `generator` on `device`."""

    def uniform(*shape):
        return torch.rand(shape, **random_keywords(generator, device))

    return uniform


def _spawned(
    config: GameConfig,
    constants: Constants,
    fields: Tensor,
    x: Tensor,
    y: Tensor,
    heading_cos: Tensor,
    heading_sin: Tensor,
    uniform,
) -> dict[str, Tensor]:
    """Every SHIP_COLUMNS row (E, N) of ships spawned at (x, y) with the heading, by the spawn
    rule (_spawn_motion: speed, store and shield): zero angle of attack and nose rate, no
    shield wait or cooldown, the medium at the place cached, the velocity along the nose."""
    envs, slots = x.shape
    zeros = filled((envs, slots), 0.0, torch.float32, x.device)
    ones = filled((envs, slots), 1.0, torch.float32, x.device)
    if fields.shape[-1] == 0:  # no medium: n = 1 everywhere
        log_index = gradient_x = gradient_y = zeros
    else:
        log_index, gradient_x, gradient_y = medium(fields, x, y)
    speed, energy, shield = _spawn_motion(config, constants, log_index, uniform)
    named = {
        "x": x,
        "y": y,
        "heading_cos": heading_cos,
        "heading_sin": heading_sin,
        "speed": speed,
        "angle": zeros,
        "angle_cos": ones,
        "angle_sin": zeros,
        "nose_rate": zeros,
        "energy": energy,
        "shield": shield,
        "shield_wait": zeros,
        "cooldown": zeros,
        "log_index": log_index,
        "gradient_x": gradient_x,
        "gradient_y": gradient_y,
        "next_slot": zeros,
    }
    # At zero angle of attack the flight path is the nose.
    named["velocity_x"], named["velocity_y"] = speed * heading_cos, speed * heading_sin
    return named


def spawn(
    config: GameConfig,
    state: State,
    chosen: Tensor,
    x: Tensor,
    y: Tensor,
    heading_cos: Tensor,
    heading_sin: Tensor,
    generator,
) -> State:
    """The state with the ships in `chosen` (E, N) bool spawned alive at (x, y) (E, N) with the
    heading, by the deal's spawn rule (_spawned); the other ships as they were. For a caller
    that places ships itself (the sandbox): a deal draws its own places."""
    uniform = _uniform_draws(generator, state.ships.device)
    named = _spawned(config, state.constants, state.fields, x, y, heading_cos, heading_sin, uniform)
    ships = torch.stack([named[name] for name in SHIP_COLUMNS], 1)
    return state.replace(
        ships=torch.where(chosen[:, None], ships, state.ships),
        alive=state.alive | chosen,
    )


def deal(config: GameConfig, envs: int, slots: int, team_sizes: Tensor, generator, device) -> State:
    """A fresh state of (E, N): team 0 in the first team_sizes[:, 0] slots, team 1 in the
    next team_sizes[:, 1], the rest not existing (parked at the centre at spawn speed). The rows
    that every fresh deal starts with (no bullets, tick 0) are shared fixed tensors: `reset`
    copies them, `take_deal` only selects from them, so a deal may be held and taken from
    later (lean.learn draws one per environment once per segment)."""
    device = torch.device(device)
    team_sizes = team_sizes.to(device=device, dtype=torch.long)
    slot = slot_indices(slots, device).ship.expand(envs, slots)
    first, both = team_sizes[:, :1], team_sizes.sum(-1, keepdim=True)
    exists = slot < both
    team = ((slot >= first) & exists).long()
    rank = torch.where(team == 1, slot - first, slot)
    names = _lineup_types(tuple(config.lineup), device)
    # Slots that do not exist take the lineup's first type, so a constant equal for the lineup
    # stays one number.
    ship_type = names[torch.where(exists, rank % len(config.lineup), 0)]
    constants = ship_constants(config, ship_type)
    if config.field_count > 0:
        fields = draw_fields(config, envs, generator, device)
    else:  # no medium
        fields = filled((envs, 5, 0), 0.0, torch.float32, device)

    uniform = _uniform_draws(generator, device)
    x, y, angle = _spawn_places(config, team, uniform)
    heading_cos, heading_sin = _edge_safe(config, x, y, angle)
    x, y = torch.where(exists, x, 0.0), torch.where(exists, y, 0.0)
    heading_cos, heading_sin = (
        torch.where(exists, heading_cos, 1.0),
        torch.where(exists, heading_sin, 0.0),
    )
    named = _spawned(config, constants, fields, x, y, heading_cos, heading_sin, uniform)
    bullet_shape = (envs, slots, BULLET_SLOTS)
    state = State(
        ships=torch.stack([named[name] for name in SHIP_COLUMNS], 1),
        # alive, exists, fired
        flags=torch.stack([exists, exists, filled((envs, slots), False, torch.bool, device)], 1),
        identity=torch.stack([team, ship_type], 1),
        constants=constants,
        bullets=_no_bullets(envs, slots, device),
        bullet_counts=filled(
            (envs, len(BULLET_COUNT_COLUMNS), slots, BULLET_SLOTS), 0, torch.long, device
        ),
        bullet_real=filled(bullet_shape, False, torch.bool, device),
        tick=filled((envs,), 0, torch.long, device),
        fields=fields,
        # The neutral action (straight, coast, no fire) is every ship's first pending action.
        pending=filled((envs, slots, 3), 0, torch.long, device),
    )
    if not config.fog:
        return state
    # Nothing seen yet: every row unknown and taken as alive until seen dead. Ships in sight at
    # the spawn are seen at once (no other reveal at a spawn).
    seen = torch.zeros((envs, len(SEEN_COLUMNS), slots), device=device)
    seen[:, SEEN_ALIVE] = exists.to(torch.float32)
    state = dataclasses.replace(
        state, seen=seen, ticks_since_seen=filled((envs, slots), 0, torch.long, device)
    )
    return perceive(state, torch.zeros_like(exists), config.fog_tracking)


def reset(
    config: GameConfig, envs: int, team_sizes: Tensor, generator: torch.Generator, device
) -> State:
    """A fresh state. team_sizes: (E, 2) long, ships per team, each at least 1. N is the
    largest total over the environments; unused slots have exists = False."""
    slots = int(team_sizes.sum(-1).max())
    state = deal(config, envs, slots, team_sizes, generator, device)
    # The state is the caller's: its own copies of the deal's shared fixed rows.
    shared = ("bullets", "bullet_counts", "bullet_real", "tick", "pending")
    return dataclasses.replace(state, **{name: getattr(state, name).clone() for name in shared})


def take_deal(state: State, done: Tensor, fresh: State, index: Tensor | None = None) -> State:
    """The state with each environment in done (E,) bool replaced by a fresh deal: environment e
    takes row index[e] of `fresh` (a `deal` of the same N; any number of environments), or row
    e when index is None (fresh has E environments). Nothing reads done on the host."""
    envs = done.shape[0]

    def select(new: Tensor, old: Tensor) -> Tensor:
        new = new if index is None else new.index_select(0, index)
        return torch.where(done.reshape((envs,) + (1,) * (new.dim() - 1)), new, old)

    values = {}
    for field in dataclasses.fields(state):
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


def sighted_by_enemy(state: State, tracked: Tensor | None = None) -> Tensor:
    """(E, N) bool: some living ship of the other team sees the ship (`sight_pairs`)."""
    return sight_pairs(state, tracked).any(-1)


def tracked_by_enemy(state: State) -> Tensor:
    """(E, N) bool: the ship was visible to its enemy team on the tick before (by sight or a
    reveal: ticks since seen 0 and seen at least once; the deal's "nothing seen yet" rows have
    ticks since seen 0 but are not known), so its enemies hold it with the tracking circle."""
    return (state.ticks_since_seen == 0) & (state.seen[:, SEEN_KNOWN] > 0.5)


def sight_pairs(state: State, tracked: Tensor | None = None) -> Tensor:
    """(E, target, observer) bool: the observer, a living ship of the target's other team, sees the
    target (v2's sight test, src/bnb/game/perception/sight.py `sighted`, game spec 10.1, with the
    sight ellipse, author 2026-11-25). The observer sees inside an ellipse with itself at one focus,
    stretched along its nose: with d the offset target - observer and n the observer's unit nose,
    |d| - e (d . n) <= p, p the observer's vision range (the side reach) and e its
    vision_eccentricity (0 <= e < 1): p / (1 - e) dead ahead, p / (1 + e) dead behind; e = 0 is the
    circle |d| <= p. Tested without a square root as |d|^2 <= (p + e (d . n))^2 (for p > 0 and e < 1
    a negative p + e (d . n) cannot pass: |d| <= -(p + e d . n) <= -p + e |d| has no solution),
    which at e = 0 is exactly the circle's d^2 <= p^2. A target in `tracked` (E, N) bool (visible
    to this team on the tick before, `tracked_by_enemy`; author 2026-11-26) is held inside the
    tracking circle instead, |d|^2 <= (p / (1 - e))^2: the ellipse's forward reach all round, a
    superset of the ellipse (|d| <= p + e |d| gives |d| <= p / (1 - e)), equal to it at e = 0.
    None: the ellipse for every target. Sight is blocked only by fields' cores: field
    i, with core radius max(r - w / 2, 0) and outer radius r + w / 2, blocks the line when exactly
    one end is strictly inside its core and the two ends are not both inside its outer disc (core
    and band see each other), or when neither end is inside its core and the segment passes strictly
    within the core radius of the centre. Dead targets are tested like living ones (a wreck in sight
    is seen); dead observers see nothing. Bullets are not perceived."""
    c = state.constants
    x, y = state.x, state.y
    line_x = x[:, :, None] - x[:, None, :]  # (E, target, observer): target minus observer, d
    line_y = y[:, :, None] - y[:, None, :]
    length_squared = line_x * line_x + line_y * line_y

    def per_observer(value):  # a uniform number, or (E, N) per slot as (E, 1, observer)
        return value[:, None, :] if isinstance(value, Tensor) else value

    along_nose = line_x * state.heading_cos[:, None, :] + line_y * state.heading_sin[:, None, :]
    side, eccentricity = per_observer(c["vision_range"]), per_observer(c["vision_eccentricity"])
    reach = side + eccentricity * along_nose
    if tracked is not None:
        reach = torch.where(tracked[:, :, None], side / (1.0 - eccentricity), reach)
    clear = (
        (length_squared <= reach * reach)
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
    return clear


def _snapshot(state: State) -> Tensor:
    """(E, SEEN_COLUMNS, N) float32: each ship's raw values now, as a sighting records them.
    Columns on the middle axis, so each column read back is an (E, N) slice with unit stride
    along N."""
    bullet_share = ((state.bullet_moves_left > 0) & state.bullet_real).to(torch.float32).mean(-1)
    alive = state.alive.to(torch.float32)
    return torch.cat(
        [state.ships, torch.stack([bullet_share, alive, torch.ones_like(alive)], 1)], 1
    )


def perceive(state: State, revealed: Tensor, tracking: bool = True) -> State:
    """The state with each ship's sighting by its enemy team brought up to date: a ship in the
    enemy's sight or `revealed` (E, N) bool (it fired or was hit this tick) is recorded as it is
    now and its ticks since seen set to 0; any other ship keeps its last-seen row and counts up
    to the cap. Each ship has one enemy team, so one row per ship serves. tracking
    (GameConfig.fog_tracking): a ship visible on the tick before is held by the tracking circle
    (`sight_pairs`), read from the rows before this update."""
    tracked = tracked_by_enemy(state) if tracking else None
    visible = (sighted_by_enemy(state, tracked) | revealed) & state.exists
    seen = torch.where(visible[:, None], _snapshot(state), state.seen)
    since = torch.where(visible, 0, (state.ticks_since_seen + 1).clamp_max(TICKS_SINCE_SEEN_CAP))
    return dataclasses.replace(state, seen=seen, ticks_since_seen=since)


def enemy_view(state: State) -> State:
    """Every ship as its enemy team knows it: its last-seen row (a never-seen or seen-dead ship
    not alive), so that one call can read each ship's enemies for both teams at once (each
    ship has one enemy team). Bullets are not perceived (the bullet fields are None). A shot
    reveals its ship on that tick, so the true fired flag is known. Fields that are fixed and
    told at reset (exists, team, type, constants, tick) are shared. With fog off (state.seen is
    None) the state itself."""
    if state.seen is None:
        return state
    alive = (state.seen[:, SEEN_ALIVE] > 0.5) & (state.seen[:, SEEN_KNOWN] > 0.5)
    return dataclasses.replace(
        state,
        ships=state.seen[:, : len(SHIP_COLUMNS)],
        flags=torch.stack([alive, state.exists, state.fired], 1),
        bullets=None,
        bullet_counts=None,
        bullet_real=None,
        seen=None,
        ticks_since_seen=None,
    )


def team_view(state: State, team: int) -> State:
    """The game as team `team` knows it: its own ships as they are, every enemy as its
    `enemy_view` row. With fog off the state itself."""
    if state.seen is None:
        return state
    view = enemy_view(state)
    enemy = (state.team != team)[:, None]
    return dataclasses.replace(
        view,
        ships=torch.where(enemy, view.ships, state.ships),
        flags=torch.where(enemy, view.flags, state.flags),
    )


# --------------------------------------------------------------------------- observation

# The ship scalars' fixed scales, the same for every ship type, set from their ranges in scripted
# play with the training spawn (lean/game/NOTES.md "Feature scales"): symlog(x / scale) where
# the range is wide, x / scale where it is narrow.
SPEED_SCALE = 200.0  # m/s, symlog: world speed and velocity
NOSE_RATE_SCALE = 1.0  # rad/s, symlog
ANGLE_SCALE = 0.25  # rad, linear: the angle of attack, about the stall angle
SHIELD_SCALE = 100.0  # shield points, linear
SHIELD_WAIT_SCALE = 3.0  # s, linear, clipped to [0, 1]: the recharge delay
BOUNDARY_CLIP = 1000.0  # m, linear, clipped to [-1, 1]


FIELD_RADIUS_SCALE = 1000.0  # m (v2's inputs.py)
FIELD_BAND_SCALE = 40.0  # m
FIELD_LEVEL_SCALE = 2.0  # half-octaves


def observe(config: GameConfig, state: State, team: int) -> Obs:
    """Team `team`'s view of every slot: full information with fog off; with fog on, built
    from `team_view` only (enemies as last seen), so nothing in it depends on what the team
    cannot see.

    ship (E, N, SHIP_FEATURES): 18 scalars [symlog of world velocity x, y and world speed over
    200 m/s and of the nose rate over 1 rad/s; angle of attack over 0.25 rad, its cos and sin;
    store and shield as fractions of the type's full values; shield over 100; shield wait over
    3 s, clipped to 1; gun cooldown in whole ticks; distance inside the boundary over 1000 m
    clipped to [-1, 1]; the outward unit direction x, y; heading cos, sin; the share of bullet
    slots in flight], then flags [same team as the viewer, alive, exists, fired last tick], then
    the ship-type one-hot. Rows of slots that do not exist are zero. Fired last tick is the
    step's stored flag; the share of bullet slots in flight counts real bullets only (ghosts
    are not observed). Position is not a scalar (Obs.pos carries it, for the position code
    and the rotary angles).

    pending (E, N) long: the joint pending action (the one the next tick applies) of the
    team's own living ships, NO_PENDING for every other row.

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
    if state.fields.shape[-1] == 0:  # no medium (n = 1)
        speed, velocity_x, velocity_y = view.speed, view.velocity_x, view.velocity_y
    else:
        inverse_index = torch.exp(-view.log_index)
        speed = view.speed * inverse_index
        velocity_x, velocity_y = (
            view.velocity_x * inverse_index,
            view.velocity_y * inverse_index,
        )  # stored
    distance = torch.sqrt(view.x * view.x + view.y * view.y)
    inverse_distance = 1.0 / distance.clamp_min(1.0)
    fired_last = state.fired  # stored by the step (agreed fix 12), not inferred from the ring
    wide = symlog(
        torch.stack(
            [
                velocity_x / SPEED_SCALE,
                velocity_y / SPEED_SCALE,
                speed / SPEED_SCALE,
                view.nose_rate / NOSE_RATE_SCALE,
            ],
            -1,
        )
    )
    narrow = [
        view.angle / ANGLE_SCALE,
        view.angle_cos,
        view.angle_sin,
        view.energy * c["inverse_max_energy"],
        view.shield / c["max_shield"],
        view.shield / SHIELD_SCALE,
        (view.shield_wait / SHIELD_WAIT_SCALE).clamp(0.0, 1.0),
        view.cooldown,
        ((config.arena_radius - distance) / BOUNDARY_CLIP).clamp(-1.0, 1.0),
        view.x * inverse_distance,
        view.y * inverse_distance,
        view.heading_cos,
        view.heading_sin,
        bullet_share,
    ]
    flags = [same_team, alive, exists, fired_last & alive]
    one_hot = torch.nn.functional.one_hot(state.type, SHIP_TYPES)
    shown = (exists & known) if config.fog else exists
    ship = (
        torch.cat(
            [
                wide,
                torch.stack(narrow, -1),
                torch.stack(flags, -1).to(torch.float32),
                one_hot.to(torch.float32),
            ],
            -1,
        )
        * shown[..., None]
    )
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
        visible=visible,
        since_seen=since_seen,
        pending=torch.where(same_team & alive, join_action(state.pending), NO_PENDING),
    )
