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
    BULLET_SLOTS,
    CONTACTED,
    FIELD_FEATURES,
    GLOBAL_FEATURES,
    NO_PENDING,
    POWERS,
    REAL,
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
    # and on the tick it fires, is hit by a bullet or collides; a hidden enemy is its last-seen
    # row.
    # False (an ablation): full information.
    fog: bool = True
    # The tracking circle (author 2026-11-26; `sight_pairs`): an enemy visible on the tick before
    # (by sight or a reveal) stays visible while an observer has it within the circle of the
    # observer's longest sight reach (`tracking_range`), with a clear line. False (an
    # ablation): the sight egg every tick.
    fog_tracking: bool = True
    # The bullet deflector (author 2026-11-28; `circle_entry`, `circle_turn`, lean/game/NOTES.md
    # "Deflector"): a preset of card.DEFLECTOR_PRESETS ("circle", "off") setting the hit model's
    # deflector inputs at the deal. "off" (an ablation) also skips the deflector's code; under
    # "circle" every deflector number is a runtime input (`with_hit_model`).
    deflector: str = "circle"


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
INTEGER_CONSTANTS = ("fire_interval",)
# Action-indexed tables (by stick index, by power index): always rows of the per-slot table.
# The sustain table (`_sustain_fraction`) is wide too: always rows.
ACTION_TABLES = (
    "stick_table",
    "sustain_stick_table",
    "deep_stall_table",
    "nose_stiffness_table",
    "commanded_table",
    "demand_table",
    "sustain_table",
)
# The hit model (`HIT_MODEL`): always rows of the per-slot table, never Python numbers, so a
# change between calls of the compiled tick (`with_hit_model`: the workbench, the sandbox's
# sliders) needs no recompile.
RUNTIME_CONSTANTS = (
    "hit_radius",
    "core_radius",
    "hit_damage",
    "spread_speed",
    "deflector_radius",
    "deflector_turn",
    # The circle's radius as an inverse (derived, written with it): the pair test multiplies.
    "inverse_deflector_radius",
    "nose_cone",
    "nose_inner_sin",
    "inverse_nose_width",
    "hit_width",
)


def _per_type(config: GameConfig) -> dict[str, Tensor]:
    """Every ship constant for every card type: float32 (T,), the action tables (T, width), the
    whole-tick counts long; computed with the game's own float32 arithmetic."""

    def of(name: str) -> Tensor:
        return torch.tensor(card.CARD[name], dtype=torch.float32)

    lift, acceleration = of("max_lift_coefficient"), of("max_acceleration")
    # A bullet lives exactly BULLET_SLOTS ticks (its slot is rewritten then: the ring is the
    # clock), so the card's bullet life must be that (one source of truth: the slot count).
    life_ticks = torch.round(of("bullet_life") / config.tick_length).long()
    if bool((life_ticks != BULLET_SLOTS).any()):
        raise ValueError(
            f"the card's bullet life is {life_ticks.tolist()} ticks; a bullet lives exactly "
            f"contract.BULLET_SLOTS = {BULLET_SLOTS} ticks (every ship writes one bullet a tick "
            "into its ring), so bullet_life must round to that"
        )
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
    deflector = {"nose_cone": card.NOSE_CONE, **card.DEFLECTOR_PRESETS[config.deflector]}
    nose_inner_sin, inverse_nose_width = _cone_rows(
        of("nose_inner_angle"), of("nose_outer_angle"), "the nose armour"
    )
    blame_inner_sin, inverse_blame_width = _cone_rows(
        of("blame_inner_angle"), of("blame_outer_angle"), "the collision blame"
    )
    ahead, side, behind = of("vision_ahead"), of("vision_side"), of("vision_behind")
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
        "fire_interval": torch.round(1.0 / (config.tick_length * of("fire_rate"))).long(),
        "hit_damage": of("hit_damage"),
        "explosive_share": of("explosive_share"),
        "hit_radius": hit_radius,
        "core_radius": of("core_radius"),
        "nose_inner_sin": nose_inner_sin,
        "inverse_nose_width": inverse_nose_width,
        "blame_inner_sin": blame_inner_sin,
        "inverse_blame_width": inverse_blame_width,
        **{
            name: torch.full_like(hit_radius, value)
            for name, value in hit_model_constants(deflector).items()
        },
        "hit_width": hit_widths(hit_radius, deflector),
        "max_shield": of("max_shield"),
        "shield_rate": card.SHIELD_RECHARGE_RATE * of("shield_recharge_rate_scale"),
        "shield_delay": card.SHIELD_RECHARGE_DELAY * of("shield_recharge_delay_scale"),
        "spawn_energy": 0.5 * max_energy,
        "vision_ahead": ahead,
        "vision_side": side,
        "vision_behind": behind,
        "inverse_vision_ahead": 1.0 / ahead,
        "inverse_vision_side": 1.0 / side,
        "inverse_vision_behind": 1.0 / behind,
        "tracking_range": torch.maximum(torch.maximum(ahead, side), behind),
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
    # By stick index (card.STICK_LEVELS' layout): Snap's deep angle.
    level = torch.tensor(card.STICK_LEVELS).abs().expand(count, -1)
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
        "inverse_deep_stall_band": 1.0 / card_of("deep_stall_band"),
        # Snap's deep angle closes to the stall angle from where its lift is 1 g down to
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
        "deep_stall_table": deep * on[:, None],
    }


def card_of(name: str) -> Tensor:
    """A card column (one value per type) as float32."""
    return torch.tensor(card.CARD[name], dtype=torch.float32)


def stick_table(count: int) -> Tensor:
    """(count, 7) the stick's side by stick index (card.STICK_LEVELS' layout): 0 straight, +1 or
    -1 for each turn level (Sustain, Max, Snap: each all of its own line, `_setting`)."""
    return torch.sign(torch.tensor(card.STICK_LEVELS)).expand(count, -1).clone()


def sustain_stick_table(count: int) -> Tensor:
    """(count, 7) 1 at Sustain's stick indices (|level| 1), else 0."""
    level = torch.tensor(card.STICK_LEVELS).abs()
    return (level == 1.0).float().expand(count, -1).clone()


def nose_stiffness_table(count: int) -> Tensor:
    """(count, 7) the nose spring's stiffness going out, a factor of the full spring's, by stick
    index (card.NOSE_STIFFNESS by |level|; `_kick`)."""
    level = torch.tensor(card.STICK_LEVELS).abs().long()
    return torch.tensor(card.NOSE_STIFFNESS)[level].expand(count, -1).clone()


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
    - the stick tables (`stick_table`, `sustain_stick_table`, `nose_stiffness_table`);
    - Sustain's table (`_sustain_table`: the stick holding Ps = 0 over speed);
    - the middle stick speed: the scripted opponent's reference for its level cap, the speed
      above which Max is on the g limit: corner speed (out of date with the levels of
      2026-11-26; the scripted opponent is not updated yet)."""
    c = SimpleNamespace(**values)
    count = len(c.lift)  # every row of the card
    zero_stick = torch.zeros(count)
    control = steady_control(c, zero_stick, zero_stick, 0)
    low, high = torch.full((count,), 20.0), torch.full((count,), 1000.0)
    with torch.no_grad():
        for _ in range(60):  # bisection on d speed / dt = 0
            middle = 0.5 * (low + high)
            setting = _setting(c, control, middle, c.spawn_energy, 0.0)
            one, zero = torch.ones_like(middle), torch.zeros_like(middle)
            rate, _ = _rates(
                c, setting, one, middle, zero, one, 1.0 / (middle * middle), 1.0 / (c.lift * c.lift)
            )
            low, high = torch.where(rate > 0.0, middle, low), torch.where(rate > 0.0, high, middle)
    cruise = 0.5 * (low + high)
    launch = cruise + c.muzzle_speed
    life = BULLET_SLOTS * config.tick_length
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
    sustain_low, sustain_step, sustain = _sustain_table(values, stall_speed, cruise)
    return {
        "cruise_speed": cruise,
        "stall_speed": stall_speed,
        "corner_speed": corner_speed,
        "bullet_launch_speed": launch,
        # the damage's reference speed: a bullet fired at cruise, DAMAGE_REFERENCE_DISTANCE out
        "inverse_reference_bullet_speed": torch.exp(c.bullet_drag * card.DAMAGE_REFERENCE_DISTANCE)
        / launch,
        "gun_range": bullet_reach(c.bullet_drag, launch, life),
        "sustained_speed": sustained_speed,
        "sustained_rate": sustained_rate,
        "sustained_radius": sustained_speed / sustained_rate,
        "quarter_turn_time": (0.5 * math.pi) / sustained_rate,
        "corner_rate": corner_rate,
        # the scripted opponent's level-cap reference (Max is on the g limit above it)
        "middle_stick_speed": corner_speed,
        "stick_table": stick_table(count),
        "sustain_stick_table": sustain_stick_table(count),
        "nose_stiffness_table": nose_stiffness_table(count),
        "sustain_low_speed": sustain_low,
        "inverse_sustain_step": 1.0 / sustain_step,
        "sustain_table": sustain,
        "store_turn": (corner_rate - sustained_rate) * c.max_energy / corner_energy_rate.abs(),
        # a collision's reference closing speed: two ships head-on at cruise (`collisions`)
        "inverse_collision_speed": 0.5 / cruise,
    }


def steady_control(p, speed: Tensor, stick: Tensor, power: int, sustain=0.0, deep=0.0):
    """The flight functions' control for steady analysis: stick size `stick` (a fraction of the
    envelope's top line), power index `power`, `sustain` (1: Sustain's table sets the size at
    each speed, `_sustain_fraction`; a Python 0 skips the table) and `deep` (1: Snap's deep
    angle below corner), at `speed`'s shape."""
    uses_table = not (isinstance(sustain, (int, float)) and sustain == 0)
    return SimpleNamespace(
        thrust_scale=1.0,
        stick_size=stick,
        stick_sign=torch.ones_like(speed),
        sustain=torch.as_tensor(sustain, dtype=speed.dtype).expand_as(speed),
        sustain_table=p.sustain_table if uses_table else None,
        deep=torch.as_tensor(deep, dtype=speed.dtype).expand_as(speed),
        commanded=p.commanded_table[..., power],
        demand=p.demand_table[..., power],
    )


def steady_flight(
    p,
    speed: Tensor,
    stick: Tensor,
    power: int | None,
    energy: Tensor,
    sustain: Tensor | float = 0.0,
    deep: Tensor | float = 0.0,
):
    """`steady_rates` with the angle of attack flown: (turn rate (rad/s), d speed / dt, d store
    / dt, the angle of attack (rad, >= 0)), from the flight functions the tick uses. power None:
    every power at once, `speed`'s last axis the power index (its width 3; the action tables
    broadcast along it). The scripted agent's look-ahead reads the angle as the nose spring's
    target (lean.scripted.paths)."""
    control = steady_control(p, speed, stick, 0 if power is None else power, sustain, deep)
    if power is None:
        control.commanded, control.demand = p.commanded_table, p.demand_table
    inverse_lift = 1.0 / p.lift
    angle_sin = 0.0
    for _ in range(2):
        setting = _setting(p, control, speed, energy, angle_sin)
        lateral, _, vertical, thrust = setting
        angle = _commanded_angle(p, control, lateral, speed, inverse_lift, vertical)
        angle_sin = torch.sin(angle).abs()
    lift = _lift_coefficient(p, angle)
    whole = _deep_opening(p, control, speed)
    rate = (
        _path_lift(
            p, lift, angle, vertical, speed * speed, thrust, thrust * torch.sin(angle), whole
        )[0]
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
    return rate, speed_rate, store_rate, angle


def steady_rates(
    p,
    speed: Tensor,
    stick: Tensor,
    power: int,
    energy: Tensor,
    sustain: Tensor | float = 0.0,
    deep: Tensor | float = 0.0,
):
    """A ship flying steadily (no field, the nose's lag settled) at `speed` with stick size
    `stick` (0 to 1) and power index `power` (0 coast, 1 dive, 2 climb) at store `energy`: its
    turn rate (rad/s), d speed / dt and d store / dt, from the flight functions the tick uses.
    `p` holds the ship constants broadcastable against `speed` (the action tables with a last
    axis of width 3). Every quasi-steady analysis (the derived characteristics, lean.game.tuning's
    performance and envelope) goes through here. By default `stick` is a fraction of the
    envelope's top line (the wing's maximum lift against the g limit, Max); `sustain` (Sustain's
    table) and `deep` (Snap's deep angle below corner) give a stick level's own shape
    (`stick_level_shape`). The thrust's normal part counts in the 1 g hold at the angle flown
    (`_setting`): the setting is resolved twice, the second time at the first's angle
    (`steady_flight`, which also returns the angle)."""
    return steady_flight(p, speed, stick, power, energy, sustain, deep)[:3]


def current_path_rate(p, speed: Tensor, angle: Tensor, energy: Tensor) -> Tensor:
    """The flight path's turn rate (rad/s, positive left, no field) of a ship at `speed` and
    store `energy` whose angle of attack is `angle` now, coasting: the normal force at that angle
    (`_path_lift`, the 1 g held as the tick holds it) over the speed. With the nose rate it gives
    the angle's own rate (nose rate - path rate), which the observation does not carry."""
    control = steady_control(p, speed, torch.zeros_like(speed), 0)
    angle_sin = torch.sin(angle)
    _, _, vertical, thrust = _setting(p, control, speed, energy, angle_sin.abs())
    lift = _lift_coefficient(p, angle)
    normal = _path_lift(
        p, lift, angle, vertical, speed * speed, thrust, thrust * angle_sin, torch.zeros_like(speed)
    )[0]
    return normal / speed


def quasi_steady(
    p, speed: Tensor, stick: Tensor, power: int, energy: Tensor, sustain=0.0, deep=0.0
):
    """`steady_rates` with the store's rate folded into the total energy rate: turn rate (rad/s),
    d speed / dt, and speed x d speed / dt + d store / dt."""
    rate, speed_rate, store_rate = steady_rates(p, speed, stick, power, energy, sustain, deep)
    return rate, speed_rate, speed * speed_rate + store_rate


def stick_level_shape(p, index: int) -> tuple[float, float, float]:
    """A stick index's (stick size, sustain, deep) as `steady_rates` takes them (card.STICK_LEVELS'
    layout; from the constants' tables, so the post-stall switch is read): straight (0, 0, 0),
    Sustain (1, 1, 0), Max (1, 0, 0), Snap (1, 0, 1 with GameConfig.post_stall, else 0)."""
    return tuple(
        float(table.reshape(-1, len(card.STICK_LEVELS))[0, index].abs())
        for table in (p.stick_table, p.sustain_stick_table, p.deep_stall_table)
    )


SUSTAIN_GRID = 97  # Sustain's table: speeds from stall speed to cruise speed


def _sustain_table(values: dict[str, Tensor], stall_speed: Tensor, cruise: Tensor):
    """Sustain's line (author 2026-11-26): per row, on SUSTAIN_GRID speeds from stall speed to
    cruise speed, the stick (a fraction of the envelope's top line, Max) whose quasi-steady turn
    holds energy (Ps = 0) coasting at the spawn store: 1 where Max gains energy (below the
    tightest sustained speed), 0 at cruise speed (no sustained turn above it: the table's end,
    straight, holds there and beyond). Returns (first speed (T,), step (T,), table (T, grid)).
    The tick reads it by speed (`_sustain_fraction`), whatever the power: diving and climbing
    fly the coasting fraction of their own top line."""
    column = SimpleNamespace(**{name: value[:, None] for name, value in values.items()})
    energy = column.spawn_energy
    step = (cruise - stall_speed) / (SUSTAIN_GRID - 1)
    speeds = stall_speed[:, None] + step[:, None] * torch.arange(SUSTAIN_GRID)
    with torch.no_grad():
        zero, one = torch.zeros_like(speeds), torch.ones_like(speeds)
        low, high = zero, one
        for _ in range(SUSTAINED_STICK_STEPS):
            middle = 0.5 * (low + high)
            gains = quasi_steady(column, speeds, middle, 0, energy)[2] > 0.0
            low, high = torch.where(gains, middle, low), torch.where(gains, high, middle)
        max_gains = quasi_steady(column, speeds, one, 0, energy)[2] >= 0.0
        fraction = torch.where(max_gains, one, 0.5 * (low + high))
        fraction[:, -1] = 0.0
    return stall_speed, step, fraction


def _sustain_fraction(p, control, speed):
    """Sustain's stick at this speed: its table (`_sustain_table`) read linearly, clamped at
    both ends (1 below stall speed, 0 above cruise speed)."""
    table = control.sustain_table
    count = table.shape[-1]
    table = table.expand(*speed.shape, count)
    position = ((speed - p.sustain_low_speed) * p.inverse_sustain_step).clamp(0.0, count - 1.0)
    below = position.floor().clamp_max(count - 2.0)
    share = position - below
    index = below.long()[..., None]
    first = table.gather(-1, index)[..., 0]
    second = table.gather(-1, index + 1)[..., 0]
    return first + (second - first) * share


# The steady-flight table (`_steady_table`, `steady_lookup`): speeds on a grid of half Sustain's
# step, from a quarter of stall speed (STEADY_SPEED_NODES of them: past 1.6 x cruise speed, beyond
# the terminal dive), its nodes on Sustain's; STEADY_ROOMS store nodes across each edge band.
STEADY_STEP_OF_SUSTAIN = 0.5
STEADY_LOWEST = 0.25  # of stall speed
STEADY_SPEED_NODES = 400
STEADY_ROOMS = 9
STEADY_OUTPUTS = 4  # turn rate, d speed / dt, d store / dt, angle of attack
# Per-type tables (T, ...): kept by card type on the Constants (`Constants.per_type`), read by
# the ship's type, never dealt to slots (too wide for every slot's row).
PER_TYPE_TABLES = ("steady_table",)


def _steady_store_nodes(rooms: int) -> Tensor:
    """The store axis's nodes, 2 rooms - 1 of them, as places in the edge bands: the sink room
    (`store_room`) 0 to 1, then 2 less the top room, 1 to 2. Between the bands (1) the flight
    does not depend on the store."""
    return torch.linspace(0.0, 2.0, 2 * rooms - 1)


def _inverse_smootherstep(room: Tensor) -> Tensor:
    """t in [0, 1] with smootherstep(t) = room (bisection; monotonic)."""
    low, high = torch.zeros_like(room), torch.ones_like(room)
    for _ in range(40):
        middle = 0.5 * (low + high)
        above = smootherstep(middle) > room
        low, high = torch.where(above, low, middle), torch.where(above, middle, high)
    return 0.5 * (low + high)


def steady_characteristics(values: dict[str, Tensor]) -> dict[str, Tensor]:
    """The steady-flight table and its speed grid (`_steady_table`) for every type of `values`
    (`_per_type`'s). Built with the ship card (`ship_card`), not by `_per_type`: the analysis
    tools that call that often (lean.game.tuning's fits) never read it."""
    low, step, table = _steady_table(
        values, values["stall_speed"], 1.0 / values["inverse_sustain_step"]
    )
    return {"steady_low_speed": low, "inverse_steady_step": 1.0 / step, "steady_table": table}


def _steady_table(values: dict[str, Tensor], stall_speed: Tensor, sustain_step: Tensor):
    """`steady_flight` tabulated once per type (round 3 of the scripted look-ahead, author
    2026-11-27: a lookup in place of the flight functions in the agent's graph): (T, levels 4,
    powers 3, store nodes 2 STEADY_ROOMS - 1, speeds STEADY_SPEED_NODES, STEADY_OUTPUTS) of
    (turn rate, d speed / dt, d store / dt, angle) for straight, Sustain, Max and Snap (each
    level's (size, sustain, deep) from the stick tables), no field. The store enters only
    through its edge rooms (`store_room`), so its axis is the rooms: the sink room's band, then
    the top room's (`_steady_store_nodes`; between the bands both are 1). Returns (first speed
    (T,), step (T,), table)."""
    count = len(stall_speed)
    column = SimpleNamespace(
        **{name: value.reshape(count, 1, 1, *value.shape[1:]) for name, value in values.items()}
    )
    step = STEADY_STEP_OF_SUSTAIN * sustain_step
    below = torch.round((1.0 - STEADY_LOWEST) * stall_speed / step)
    low = stall_speed - below * step
    nodes = _steady_store_nodes(STEADY_ROOMS)
    speeds = low[:, None, None] + step[:, None, None] * torch.arange(STEADY_SPEED_NODES)
    speeds = speeds.expand(count, len(nodes), STEADY_SPEED_NODES)
    fade = 1.0 / column.inverse_fade[..., 0]  # (T, 1)
    max_energy = column.max_energy[..., 0]
    # A node below 1: the store whose sink room is the node; above 1: whose top room is 2 less
    # the node (`steady_lookup`); 1 itself: any store between the bands.
    energy = torch.where(
        nodes < 1.0,
        fade * _inverse_smootherstep(nodes.clamp_max(1.0)),
        torch.where(
            nodes > 1.0,
            max_energy - fade * _inverse_smootherstep((2.0 - nodes).clamp(0.0, 1.0)),
            0.5 * max_energy,
        ),
    )[..., None].expand_as(speeds)  # (T, store nodes, speeds)
    first = [int(abs(level)) for level in card.STICK_LEVELS]
    first_of_level = [first.index(level) for level in range(max(first) + 1)]
    levels = []
    with torch.no_grad():
        for index in first_of_level:

            def shape(name: str) -> Tensor:
                return values[name][:, index].abs().reshape(-1, 1, 1)

            size, sustain, deep = (
                shape(name) for name in ("stick_table", "sustain_stick_table", "deep_stall_table")
            )
            flown = (size.expand_as(speeds), sustain.expand_as(speeds), deep.expand_as(speeds))
            powers = [
                torch.stack(
                    steady_flight(column, speeds, flown[0], power, energy, flown[1], flown[2]), -1
                )
                for power in range(POWERS)
            ]
            levels.append(torch.stack(powers, 1))
    return low, step, torch.stack(levels, 1).contiguous()


def steady_lookup(p, table: Tensor, ship_type: Tensor, speed: Tensor, level, power, energy):
    """`steady_flight`'s (turn rate, d speed / dt, d store / dt, angle) read from the type's
    steady table (`_steady_table`), linear in the speed and in the store's edge rooms
    (`store_room`; clamped at the grid's ends). `ship_type`, `level` (stick level 0 to 3), `power`
    (index) long and `speed`, `energy` broadcast together; `p` the ship constants broadcastable
    against them (steady_low_speed, inverse_steady_step, inverse_fade, max_energy)."""
    _, levels, powers, stores, speeds, outputs = table.shape
    position = ((speed - p.steady_low_speed) * p.inverse_steady_step).clamp(0.0, speeds - 1.001)
    below = position.long()
    share = position - below
    sink_room, top_room = store_room(p, energy)
    rooms = 0.5 * (stores - 1)
    place = torch.where(energy < 0.5 * p.max_energy, sink_room, 2.0 - top_room)
    store = (place * rooms).clamp_max(stores - 1.001)
    store_below = store.long()
    store_share = store - store_below
    row = (((ship_type * levels + level) * powers + power) * stores + store_below) * speeds + below
    corners = table.view(-1, outputs)[row[..., None] + _corner_offsets(speeds, row.device)]
    lower, upper = corners[..., :2, :], corners[..., 2:, :]  # (..., 2 speeds, outputs) each
    along = torch.lerp(lower, upper, store_share[..., None, None])
    return torch.lerp(along[..., 0, :], along[..., 1, :], share[..., None]).unbind(-1)


@built_once
def _corner_offsets(speeds: int, device: torch.device) -> Tensor:
    """The four table rows around a point: the store's lower node at the lower and the upper
    speed, then the store's upper node at both."""
    return torch.tensor([0, 1, speeds, speeds + 1], device=device)


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
    columns, rows, per_type = {}, [], {}
    every = _per_type(config)
    every.update(steady_characteristics(every))
    for name, values in every.items():
        if name in PER_TYPE_TABLES:
            per_type[name] = values.contiguous()
            continue
        chosen = values[used]
        runtime = name in ACTION_TABLES or name in RUNTIME_CONSTANTS
        if not runtime and bool((chosen == chosen[0]).all()):
            uniform[name] = chosen[0].item()
            continue
        if name in INTEGER_CONSTANTS:
            raise ValueError(
                f"{name} differs between the lineup's types {lineup}; it must be equal"
            )
        block = values.reshape(len(card.TYPE_NAMES), -1).T  # (width, T)
        columns[name] = (sum(len(row) for row in rows), block.shape[0])
        rows.append(block)
    table = torch.cat(rows, 0).contiguous()  # (C, T): the action tables at least
    return Constants(uniform, columns, table, per_type)


# Built ship cards by (store loops, store edge band, tick length, lineup, arena radius).
_SHIP_CARDS: dict[tuple, Constants] = {}


def ship_card(config: GameConfig) -> Constants:
    """The run's ship card: its constants per ship type, before they are dealt to slots (the
    table (C, T) on the CPU, one column per card type). Built once per flight settings,
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
        config.deflector,
    )
    if key not in _SHIP_CARDS:
        _SHIP_CARDS[key] = _build_ship_card(config)
    return _SHIP_CARDS[key]


@built_once
def _card_table(ship_card_: Constants, device: torch.device) -> Tensor:
    return ship_card_.table.to(device)


@built_once
def _card_per_type(ship_card_: Constants, device: torch.device) -> dict[str, Tensor]:
    return {name: value.to(device) for name, value in ship_card_.per_type.items()}


def ship_constants(config: GameConfig, ship_type: Tensor) -> Constants:
    """The per-ship constants of slots of types ship_type (E, N) long, each a type of the lineup:
    the card's uniform numbers and its (C, T) table (shared, never written), read through
    ship_type."""
    ship_card_ = ship_card(config)
    table = _card_table(ship_card_, ship_type.device)
    per_type = _card_per_type(ship_card_, ship_type.device)
    return Constants(ship_card_.uniform, ship_card_.columns, table, per_type, ship_type)


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
    # The band is centred on the field's edge (`field_edge_distance` 0, where ln n is half
    # the field's level).
    depth = 0.5 - field_edge_distance(field_x, field_y, radius, x[..., None], y[..., None]) / band
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
    leave_one_out = slope / torch.where(in_band, complement, 1.0)  # slope is 0 outside
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


def store_room(p, energy):
    """(sink room, top room), 0 to 1: the store's distance from empty and from full in its edge
    band (`GameConfig.store_edge_band`), smootherstepped. Dive and a sinking ship need the sink
    room, a climb the top room (`_setting`)."""
    sink_room = smootherstep((energy * p.inverse_fade).clamp(0.0, 1.0))
    top_room = smootherstep(((p.max_energy - energy) * p.inverse_fade).clamp(0.0, 1.0))
    return sink_room, top_room


def _setting(p, control, speed, energy, angle_sin):
    """The controls resolved at a speed and store: signed lateral acceleration, the exchange
    between speed and store, the vertical acceleration held, and thrust.

    Weight always on the wing. The lift budget goes, in order: 1 g held (dive: its lower demand,
    which frees lift for the turn), then the turn, then a climb's extra demand gets what is left
    (the turn has precedence over the climb). Smootherstep edges of the store; at an edge, dive
    or climb fades to coast in its effect on the wing as well as in the trade (no free turn from
    diving on the deck, no lost turn from climbing at the top). The engine drives speed only
    (constant power) and never feeds the store; firing scales the thrust on the ticks it fires.

    The 1 g is held by the wing's maximum lift plus the thrust's normal part at the angle flown
    (angle_sin: |sin| of the angle of attack; author 2026-11-26), against the g limit.

    The stick levels (author 2026-11-26): a level is its size times sqrt(budget^2 - held^2),
    the budget the wing's maximum lift against the g limit (one soft minimum: the true corner):
    the envelope's top line. Max: all of it. Snap: the same, and below corner its angle of
    attack goes past the stall (`_commanded_angle`). Sustain: the fraction of it that holds
    energy at this speed (`_sustain_fraction`).
    """
    speed_squared = speed * speed
    q = p.lift_softmin_exponent
    # The corner, smooth: the wing's lift against the g limit (the only soft minimum of the turn).
    budget = softmin(p.lift * speed_squared, p.max_acceleration, q).clamp_min(1e-12)
    sink_room, top_room = store_room(p, energy)
    budget_squared = budget * budget
    climbing = control.commanded > 0.0
    room = torch.where(climbing, top_room, torch.where(control.commanded < 0.0, sink_room, 1.0))
    demand = card.GRAVITY + (control.demand - card.GRAVITY) * room
    commanded = control.commanded * room
    first = torch.where(climbing, card.GRAVITY, demand)  # held before the turn
    thrust = softmin(p.power / speed, p.static_thrust, p.thrust_softmin_exponent)
    thrust = thrust * control.thrust_scale
    # What holds it: the wing's maximum lift and the thrust's normal part, against the g limit.
    holding = softmin(p.lift * speed_squared + thrust * angle_sin, p.max_acceleration, q)
    held = softmin(first, holding, q)
    # The stick: its fraction of the most lateral acceleration the ship can give at this speed
    # (Max: all of it, the envelope's top line; lift-limited below corner, g-limited above):
    # the budget with what is held taken first.
    size = control.stick_size
    if control.sustain_table is not None:
        size = size * torch.where(control.sustain > 0.5, _sustain_fraction(p, control, speed), 1.0)
    # The wing's share of what is held, never above its budget (the same soft minimum: where
    # the thrust holds the rest the turn fades smoothly, no infinite slope where it reaches 0).
    on_wing = softmin(held, budget, q)
    lateral_size = size * torch.sqrt((budget_squared - on_wing * on_wing).clamp_min(0.0))
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


def _path_lift(p, lift, angle, vertical, speed_squared, thrust, thrust_normal, whole):
    """The lateral acceleration of the normal force: the lift plus the thrust's normal part
    (thrust_normal = thrust x sin angle; the thrust's size never depends on the angle, its part
    along the path, thrust x cos angle, drives the speed in `_rates`).

    Below the stall the angle of attack is the turn's own: lift plus thrust_normal, the held
    vertical carried apart as induced drag. Past it (GameConfig.post_stall) the angle becomes
    the wing's whole, so the held vertical comes out of the normal force first, like the lift
    (author 2026-11-26: one force budget): lateral^2 = normal^2 - share x vertical^2, share a
    smootherstep over card.POST_STALL_SHARE_WIDTH past the stall, at least what keeps the wing's
    whole load within its maximum lift against the g limit, and at least `whole` (Snap's
    deep opening, `_deep_opening`: Snap's angle is the wing's whole at any angle once open, so
    closing it toward the stall near stall speed frees no vertical); what is left turns the
    path. Where the normal force barely carries the vertical the square is floored smoothly,
    (x + sqrt(x^2 + e^2)) / 2, e = card.PATH_LIFT_SOFTNESS^2 x share x vertical^2 (0 below the
    stall, so ordinary flight is exact), so the rate has no infinite slope where it reaches 0.
    No cap at the envelope (author 2026-11-26): near stall speed Snap's thrust turns it a little
    more than Max. Returns (lateral, its share of the normal force): the share scales the
    path's response to the angle (`_kick`'s kappa)."""
    normal = lift * speed_squared + thrust_normal
    if not _switched_on(p.post_stall):
        return normal, 1.0
    share = smootherstep(((angle.abs() - p.stall) * p.inverse_share_width).clamp(0.0, 1.0))
    share = torch.maximum(share, whole)
    held = vertical * vertical
    # The wing never carries more than its maximum lift against the g limit (`_setting`'s
    # budget): what it carries apart of the vertical is at most what its lift at this angle
    # leaves (0 at Max's angle, so ordinary flight is exact; near corner Snap's angle opening
    # toward the stall would otherwise carry the whole 1 g for free on top of the maximum lift).
    most = softmin(p.lift * speed_squared, p.max_acceleration, p.lift_softmin_exponent)
    wing = lift * speed_squared
    taken = torch.maximum(share * held, wing * wing + held - most * most)
    left = normal * normal - taken
    softness = card.PATH_LIFT_SOFTNESS**2 * torch.minimum(taken.clamp_min(0.0), held)
    turn = torch.sqrt(0.5 * (left + torch.sqrt(left * left + softness * softness)))
    fraction = (turn / normal.abs().clamp_min(1e-6)).clamp_max(1.0)
    return torch.sign(normal) * turn, fraction


def _deep_opening(p, control, speed):
    """Snap's deep opening (0 to 1): a smootherstep from corner speed down over deep_stall_band
    of corner speed (GameConfig.post_stall; 0 for the other levels and with the regime off)."""
    if not _switched_on(p.post_stall):
        return torch.zeros_like(speed)
    below = ((1.0 - speed * p.inverse_corner_speed) * p.inverse_deep_stall_band).clamp(0.0, 1.0)
    return smootherstep(below) * control.deep


def _commanded_angle(p, control, lateral, speed, inverse_lift, vertical):
    """The angle of attack the stick asks for (signed): the one whose lift (below the stall)
    gives the setting's lateral acceleration; for full stick (control.deep) below corner it
    opens toward the deep angle that still holds the vertical (`_holding_angle`: coasting, the
    card's deep_stall_angle from stall speed / sqrt(POST_STALL_LIFT_FRACTION) up, shrinking
    smoothly to the stall angle at stall speed) by Snap's deep opening (`_deep_opening`: from
    corner speed, the lift's angle there, so no jump, down over deep_stall_band of corner speed;
    GameConfig.post_stall). Full stick never gives up the 1 g (author 2026-11-26). vertical:
    the setting's held vertical (`_setting`)."""
    reach = (lateral.abs() / (speed * speed) * inverse_lift).clamp(0.0, 1.0)
    angle = p.stall * (1.0 - torch.sqrt(1.0 - reach))
    if _switched_on(p.post_stall):
        opening = _deep_opening(p, control, speed)
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
    """exp(duration * A) for 2x2 A given by its entries (E, N): a degree-4 series, squared once.
    The 2x2 products are written out entry by entry: elementwise, so the compiled tick fuses
    them (as batched matrix products they were four small cuBLAS calls, about 40% of the GPU
    tick; 2026-11-27)."""
    half = duration / 2.0
    b = (a11 * half, a12 * half, a21 * half, a22 * half)

    def times_b(h, scale):  # I + (B H) * scale
        return (
            1.0 + (b[0] * h[0] + b[1] * h[2]) * scale,
            (b[0] * h[1] + b[1] * h[3]) * scale,
            (b[2] * h[0] + b[3] * h[2]) * scale,
            1.0 + (b[2] * h[1] + b[3] * h[3]) * scale,
        )

    h = (1.0 + b[0] / 4.0, b[1] / 4.0, b[2] / 4.0, 1.0 + b[3] / 4.0)
    h = times_b(h, 1.0 / 3.0)
    h = times_b(h, 1.0 / 2.0)
    h = times_b(h, 1.0)
    return (
        h[0] * h[0] + h[1] * h[2],
        h[0] * h[1] + h[1] * h[3],
        h[2] * h[0] + h[3] * h[2],
        h[2] * h[1] + h[3] * h[3],
    )


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
    angle_sin = sin0.abs()  # the thrust's normal part in the 1 g hold, at the angle flown
    first = _setting(p, control, safe0, energy, angle_sin)
    lift0, slope = _lift_and_slope(p, angle0)
    speed_rate0, energy_rate0 = _rates(
        p, first, inverse_index, speed0, lift0, cos0, 1.0 / (safe0 * safe0), inverse_lift_squared
    )
    speed = (speed0 + 0.5 * duration * speed_rate0).clamp_min(floor)
    current = _setting(p, control, speed, energy + 0.5 * duration * energy_rate0, angle_sin)
    lateral, _, vertical, thrust = current
    speed_squared = speed * speed
    inverse_speed_squared = 1.0 / speed_squared
    path_lift, fraction = _path_lift(
        p,
        lift0,
        angle0,
        vertical,
        speed_squared,
        thrust,
        thrust * sin0,
        _deep_opening(p, control, speed),
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
    # The spring per level (author 2026-11-26): going out (the target beyond the angle on its
    # side, or on the other side once past centre) its stiffness is the level's
    # (card.NOSE_STIFFNESS: Sustain 1/3, Max 2/3, Snap 1) of omega^2; going back toward the
    # flight path (a release, a smaller level on the same side, a reversal until the nose
    # crosses centre) all of it. The friction (the total damping coefficient 2 nose_damping
    # omega) is the same for every level, so the damping ratio is nose_damping / sqrt(stiffness):
    # Snap 1/sqrt(3), Max 0.707, Sustain 1. Chosen per tick from the target and the angle: no
    # state; the angle and its rate stay continuous.
    kappa = (slope * speed_squared + thrust * cos0) * fraction * over
    omega = (p.nose_frequency * speed).clamp_min(card.NOSE_FREQUENCY_FLOOR)
    if inverse_index is not None:
        omega = omega * inverse_index
    damping = 2.0 * p.nose_damping * omega - kappa
    target = _commanded_angle(p, control, lateral, speed, inverse_lift, vertical)
    step_angle = target - angle0
    going_out = (target * angle0 >= 0.0) & (target.abs() > angle0.abs())
    stiffness = torch.where(going_out, control.stiffness, 1.0) * (omega * omega)
    m11, m12, m21, m22 = _matrix_exponential(
        -kappa,
        filled((), 1.0, kappa.dtype, kappa.device).expand_as(kappa),
        damping * kappa - stiffness,
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
        sustain=p.sustain_stick_table.gather(1, index)[:, 0],
        sustain_table=p.sustain_table.movedim(1, -1),  # (E, N, grid)
        deep=p.deep_stall_table.gather(1, index)[:, 0],
        stiffness=p.nose_stiffness_table.gather(1, index)[:, 0],
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
    # Every bullet moves on every tick of its life (BULLET_SLOTS ticks: the ring); only a ghost
    # or a real bullet not yet contacted can touch a ship, and only a real one damages.
    contactable = shots.bullet_state < CONTACTED
    bullets = _move_bullets(config, state, shots.bullets)
    contact = _hit_test(config, state, flown, shots.bullets, bullets, contactable, index)
    real = (shots.bullet_state == REAL).reshape(E, N * K)
    bullet_damage = _damage(state, flown, bullets, real, contact)
    if contact.bent is not None:
        bullets = _deflect(bullets, contact)
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
    age = bullet_ages(state.tick, K)
    bullet_state, shot_outcome = _bullet_ends(
        shots.bullet_state, contactable, age, contact.hit, enemy
    )

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
            ],
            1,
        ),
        state.ships,
    )
    new_state = dataclasses.replace(
        state,
        ships=ships,
        flags=torch.stack([alive, state.exists, shots.fired], 1),
        bullets=bullets,
        bullet_state=bullet_state,
        tick=tick,
        pending=action,
    )
    bullet_matrix = (
        filled((E, N * N), 0.0, bullet_damage.dtype, device)
        .scatter_add(1, index.pair + contact.target, bullet_damage)
        .reshape(E, N, N)
    )
    collision_matrix = crash.transpose(1, 2)
    out = StepOut(
        damage_taken=total,
        died=died,
        done=done,
        result=result,
        shot_outcome=shot_outcome,
        shot_tick=(state.tick[:, None] - age)[:, None].expand(E, N, K),
        hit_damage=bullet_matrix + collision_matrix,
        bullet_damage=bullet_matrix,
        collision_damage=collision_matrix,
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
    bullet_state: Tensor  # (E, N, K) uint8 with the new bullets written
    cooldown: Tensor  # (E, N), the gun's whole ticks to wait, before this tick's count down


def _fire(
    config: GameConfig, state: State, shoot: Tensor, spread: Tensor, index: SimpleNamespace
) -> Shots:
    """Fire, from the tick-start state. A living ship with the trigger held and the gun ready
    fires a real shot (a shot costs no store). Every ship, alive or dead, writes one bullet a
    tick, into the tick's slot of its ring (tick mod K, overwriting the bullet written K ticks
    before): a real bullet if it fired, else a ghost. A new bullet starts at its ship with the
    ship's cached medium, at the ship's velocity plus the muzzle speed along the nose and the
    spread."""
    c = state.constants
    E, _, _, K = state.bullets.shape
    alive = state.alive
    # Below the band's top the store pays (part of) the shot's cost: an empty store blocks it.
    blocked = (state.speed < c["stall_speed"] * (1.0 + config.fire_store_band)) & (
        state.energy <= 0.0
    )
    fired = alive & shoot & (state.cooldown == 0) & ~blocked
    aim_x, aim_y = launch_velocity(state)
    launch_x = aim_x + c["spread_speed"] * spread[..., 0]
    launch_y = aim_y + c["spread_speed"] * spread[..., 1]
    launch_speed = torch.sqrt(launch_x * launch_x + launch_y * launch_y)
    written = (index.bullet_slot == (state.tick % K)[:, None])[:, None]  # (E, 1, K)
    launched = torch.cat(
        [
            state.ships[:, X : Y + 1],
            torch.stack([launch_x / launch_speed, launch_y / launch_speed, launch_speed], 1),
            state.ships[:, LOG_INDEX : GRADIENT_Y + 1],
        ],
        1,
    )
    return Shots(
        fired=fired,
        bullets=torch.where(written[:, None], launched[..., None], state.bullets),
        bullet_state=torch.where(written, fired[..., None].to(torch.uint8), state.bullet_state),
        cooldown=torch.where(fired, c["fire_interval"], state.cooldown),
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


def _move_bullets(config: GameConfig, state: State, start: Tensor) -> Tensor:
    """Every bullet's move (bullet_motion), whatever its state. start: the bullet block after the
    fire stage; returns the block after the move."""
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
    return torch.stack(moved, 1)


class Contact(NamedTuple):
    """The hit test's result, per bullet (E, N K)."""

    target: Tensor  # (E, N K) long: the ship of the first contact (slot 0 for a miss)
    hit: Tensor  # (E, N K) bool
    # The impact parameter of the hitting pass (metres; the distance of its line from the
    # target's centre, at most the start's distance), any value for a miss.
    impact: Tensor
    # The deflector's turn (None with the deflector off), per bullet (E, N K): turned this tick
    # (`bent`), by the rotation (turn_cos, turn_sin), its end moved by the turned rest of the
    # move (shift), and whether the hit (if any) is on the turned remainder (`bent_hit`).
    bent: Tensor | None = None
    turn_cos: Tensor | None = None
    turn_sin: Tensor | None = None
    shift_x: Tensor | None = None
    shift_y: Tensor | None = None
    bent_hit: Tensor | None = None
    # The bend ship's average map velocity over the tick (the relative frame's).
    ship_velocity_x: Tensor | None = None
    ship_velocity_y: Tensor | None = None


def _hit_test(
    config: GameConfig,
    state: State,
    flown: SimpleNamespace,
    start_bullets: Tensor,
    bullets: Tensor,
    contactable: Tensor,
    index: SimpleNamespace,
) -> Contact:
    """The swept test between the two straight motions, every bullet against every ship
    ((E, 2, N K, N), x and y on the second axis), in the relative frame (bullet minus ship). A
    contact is a closest approach within the hit radius of a ship alive at tick start (a dead
    ship is no target) that is not the bullet's owner, by a `contactable` bullet (a ghost or a
    real bullet not yet contacted: a contacted one flies on without contact). First contact: the
    smallest tau, ties within 1e-6 to the closer pass.

    The deflector (author 2026-11-28, `circle_entry`, `circle_turn`): a bullet whose straight
    path first crosses a living ship's deflector circle inward this tick (the first such ship by
    tau) is turned there, and its contact with that ship is the swept test of the turned
    remainder (the crossing to the tick's end) against the hit circle; its contacts with the
    ships whose circles it does not enter stay on the straight path (a second circle entered in
    the same tick is not seen and gives no contact: rare)."""
    E, _, N, K = bullets.shape
    c = state.constants
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
    radius = c["hit_radius"]
    candidate = index.other & contactable.reshape(E, N * K, 1) & state.alive[:, None, :]
    reach = _per_ship(radius * radius)
    contact = (miss_squared < reach) & candidate
    if config.deflector == "off":
        target, hit, _ = _first_contact(contact, tau, miss_squared)
        return Contact(target, hit, _impact(state, flown, start_bullets, bullets, target))
    entry = circle_entry(
        start[:, 0],
        start[:, 1],
        relative[:, 0],
        relative[:, 1],
        _per_ship(c["inverse_deflector_radius"]),
    )
    entry = torch.where(candidate, entry, torch.inf)
    entry_tau, bend_ship = entry.min(-1)
    bent = entry_tau < torch.inf
    # A ship whose circle the bullet enters this tick has no straight contact: the bend ship's is
    # the turned remainder's (below); a second circle entered in the same tick is not seen (rare).
    # (Excluding every entered pair, not only the bend ship's, keeps the two reductions over
    # the pairs independent, so they run as one pass.)
    target, hit, first_tau = _first_contact(contact & (entry == torch.inf), tau, miss_squared)
    # The turned remainder against the bend ship: the turn of every pair's pass (cheap
    # arithmetic in registers), read at the bend ship (`_at`), so no per-bullet gathers.
    pair_entry = torch.where(entry < torch.inf, entry, 0.0)
    turn = circle_turn(
        start[:, 0],
        start[:, 1],
        relative[:, 0],
        relative[:, 1],
        pair_entry,
        _per_ship(c["inverse_deflector_radius"]),
        _per_ship(c["deflector_turn"]),
    )
    bend = index.ship == bend_ship[..., None]
    entry_tau = torch.where(bent, entry_tau, 0.0)
    rest = 1.0 - entry_tau
    move_x, move_y = _at(relative[:, 0], bend), _at(relative[:, 1], bend)
    turn_cos, turn_sin = _at(turn.cos, bend), _at(turn.sin, bend)
    bent_hit = bent & _at_bool(turn.miss_squared < reach, bend)
    bent_tau = entry_tau + rest * _at(turn.tau, bend)
    bent_hit = bent_hit & (bent_tau < first_tau)
    straight = target[..., None] == index.ship
    impact = torch.where(
        bent_hit, _at(turn.impact, bend), _at(_line_impact(start, relative), straight)
    )
    return Contact(
        torch.where(bent_hit, bend_ship, target),
        hit | bent_hit,
        impact,
        bent,
        turn_cos,
        turn_sin,
        rest * (move_x * (turn_cos - 1.0) - move_y * turn_sin),
        rest * (move_x * turn_sin + move_y * (turn_cos - 1.0)),
        bent_hit,
        _at(moved[:, 0, None, :], bend) / config.tick_length,
        _at(moved[:, 1, None, :], bend) / config.tick_length,
    )


def _at(values: Tensor, chosen: Tensor) -> Tensor:
    """Per bullet the value of its one chosen pair: values (E, N K, N) (or broadcasting to it),
    chosen (E, N K, N) bool with one True per bullet; a masked sum over the ships, which stays in
    the pair reduction's registers (a gather after it would be its own kernel)."""
    return torch.where(chosen, values, 0.0).sum(-1)


def _at_bool(values: Tensor, chosen: Tensor) -> Tensor:
    """`_at` for a bool pair tensor."""
    return (values & chosen).any(-1)


def _line_impact(start: Tensor, relative: Tensor) -> Tensor:
    """Every pair's impact parameter on its straight pass (the distance of the relative line
    from the ship's centre, at most the start's distance): start, relative (E, 2, ...)."""
    start_x, start_y = start[:, 0], start[:, 1]
    cross = start_x * relative[:, 1] - start_y * relative[:, 0]
    relative_squared = (relative * relative).sum(1).clamp_min(1e-12)
    line = cross * cross / relative_squared
    return torch.minimum(line, start_x * start_x + start_y * start_y).sqrt()


def _first_contact(contact: Tensor, tau: Tensor, miss_squared: Tensor):
    """The first contact per bullet: (target (E, N K) long, hit bool, its tau, inf for none)."""
    contact_tau = torch.where(contact, tau, torch.inf)
    first_tau = contact_tau.min(-1).values
    tied = contact & (contact_tau <= first_tau[..., None] + 1e-6)
    target = torch.where(tied, miss_squared, torch.inf).argmin(-1)
    return target, first_tau < torch.inf, first_tau


def _impact(state, flown, start_bullets: Tensor, bullets: Tensor, target: Tensor) -> Tensor:
    """Each bullet's impact parameter against `target` on its straight in-tick pass (the
    distance of the relative line from the target's centre, at most the start's distance)."""
    start, relative, relative_squared = _pass_against(state, flown, start_bullets, bullets, target)
    start_x, start_y = start.unbind(1)
    cross = start_x * relative[:, 1] - start_y * relative[:, 0]
    line = cross * cross / relative_squared
    return torch.minimum(line, start_x * start_x + start_y * start_y).sqrt()


def _pass_against(
    state: State, flown: SimpleNamespace, start_bullets: Tensor, bullets: Tensor, ship: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    """Each bullet's pass against one ship `ship` (E, N K) long, as the hit test computes it:
    the start (bullet minus ship at tick start) and the relative move, (E, 2, N K) each, and the
    relative move's squared length (E, N K), clamped away from 0."""
    E, _, N, K = bullets.shape
    flat = (E, 2, N * K)
    ships = state.ships[:, X : Y + 1]
    at = ship[:, None, :].expand(E, 2, N * K)
    start_xy = start_bullets[:, BULLET_X : BULLET_Y + 1]
    start = start_xy.reshape(flat) - ships.gather(2, at)
    moved = torch.stack([flown.x, flown.y], 1) - ships
    relative = (bullets[:, BULLET_X : BULLET_Y + 1] - start_xy).reshape(flat)
    relative = relative - moved.gather(2, at)
    return start, relative, (relative * relative).sum(1).clamp_min(1e-12)


def _per_ship(values: Tensor | float) -> Tensor | float:
    """A per-ship value (E, N) laid along the ship axis of (E, N K, N) bullet-ship pairs."""
    return values[:, None, :] if isinstance(values, Tensor) else values


def circle_entry(start_x, start_y, move_x, move_y, inverse_radius):
    """The deflector's circle (author 2026-11-28): the tau in [0, 1) at which the straight path
    start + tau move (the bullet relative to the ship's centre) first crosses the circle of
    radius 1 / `inverse_radius` about the centre inward, inf for none: a path starting inside
    (a bullet fired from inside, or already turned) never enters. The radius comes as an
    inverse (the row inverse_deflector_radius), so the pair test multiplies. Elementwise; every
    argument broadcasts."""
    p_x, p_y = start_x * inverse_radius, start_y * inverse_radius
    d_x, d_y = move_x * inverse_radius, move_y * inverse_radius
    qa = (d_x * d_x + d_y * d_y).clamp_min(1e-12)
    qb = p_x * d_x + p_y * d_y
    qc = p_x * p_x + p_y * p_y - 1.0
    disc = qb * qb - qa * qc
    t = (-qb - disc.clamp_min(0.0).sqrt()) / qa
    valid = (qc > 0.0) & (disc > 0.0) & (t >= 0.0) & (t < 1.0)
    return torch.where(valid, t, torch.inf)


class Turn(NamedTuple):
    """`circle_turn`'s result, per path."""

    cos: Tensor  # the rotation of the path's direction
    sin: Tensor
    tau: Tensor  # the turned remainder's closest approach, its own tau in [0, 1]
    miss_squared: Tensor  # the squared distance there from the ship's centre
    impact: Tensor  # the turned remainder's impact parameter (at most the entry's distance)


def hump(x: Tensor) -> Tensor:
    """The deflector's turn curve (the "early hump", author 2026-11-27): 27/4 x (1 - x)^2 on x
    clamped to [0, 1]; 0 at both ends, its peak 1 at x = 1/3."""
    x = x.clamp(0.0, 1.0)
    return 6.75 * x * (1.0 - x) * (1.0 - x)


def circle_turn(start_x, start_y, move_x, move_y, entry_tau, inverse_radius, turn_max) -> Turn:
    """The deflector's turn at the circle entry (`circle_entry`'s tau) of the straight path
    start + tau move (relative to the ship's centre): the direction turned away from the centre
    by turn_max x hump(b / radius) (the radius given as inverse_radius), b the distance of the
    straight line from the centre (b = 0 goes straight); the speed kept; then the swept test of
    the turned remainder (the entry point to the tick's end, (1 - entry_tau) of the move turned)
    against the centre."""
    length_squared = (move_x * move_x + move_y * move_y).clamp_min(1e-12)
    inverse_length = torch.rsqrt(length_squared)
    point_x, point_y = start_x + entry_tau * move_x, start_y + entry_tau * move_y
    # Signed distance of the line from the centre: + when the centre is on the line's right.
    signed = (move_x * point_y - move_y * point_x) * inverse_length
    b = signed.abs()
    angle = turn_max * hump(b * inverse_radius)
    turn_cos = torch.cos(angle)
    turn_sin = torch.sin(angle) * signed / b.clamp_min(1e-9)
    rest = 1.0 - entry_tau
    rest_x = rest * (move_x * turn_cos - move_y * turn_sin)
    rest_y = rest * (move_x * turn_sin + move_y * turn_cos)
    rest_squared = (rest_x * rest_x + rest_y * rest_y).clamp_min(1e-12)
    tau = (-(point_x * rest_x + point_y * rest_y) / rest_squared).clamp(0.0, 1.0)
    miss_x, miss_y = point_x + tau * rest_x, point_y + tau * rest_y
    cross = point_x * rest_y - point_y * rest_x
    point_squared = point_x * point_x + point_y * point_y
    impact = torch.minimum(cross * cross / rest_squared, point_squared).sqrt()
    return Turn(turn_cos, turn_sin, tau, miss_x * miss_x + miss_y * miss_y, impact)


def _deflect(bullets: Tensor, contact: Contact) -> Tensor:
    """The turned bullets' block, in the relative frame of the bend ship: the end moved by the
    turned rest of the tick's relative move (`Contact.shift`), the map velocity relative to the
    ship (its average over the tick) rotated by the same turn, so the relative speed is kept
    (the map speed changes by the ship's part); the end's cached medium kept (the turned end is
    metres off the cached one). The bullet's speed column is its proper speed: map speed x n."""
    E, _, N, K = bullets.shape
    flat = (E, N * K)
    index = torch.exp(bullets[:, BULLET_LOG_INDEX].reshape(flat))
    map_speed = bullets[:, BULLET_SPEED].reshape(flat) / index
    relative_x = bullets[:, BULLET_DIR_X].reshape(flat) * map_speed - contact.ship_velocity_x
    relative_y = bullets[:, BULLET_DIR_Y].reshape(flat) * map_speed - contact.ship_velocity_y
    velocity_x = contact.ship_velocity_x + (
        relative_x * contact.turn_cos - relative_y * contact.turn_sin
    )
    velocity_y = contact.ship_velocity_y + (
        relative_x * contact.turn_sin + relative_y * contact.turn_cos
    )
    speed = torch.sqrt(velocity_x * velocity_x + velocity_y * velocity_y).clamp_min(1e-9)
    turned = (
        bullets[:, BULLET_X].reshape(flat) + contact.shift_x,
        bullets[:, BULLET_Y].reshape(flat) + contact.shift_y,
        velocity_x / speed,
        velocity_y / speed,
        speed * index,
    )
    bent = contact.bent.reshape(E, N, K)
    # One stack of the columns (no where over the head and cat of the tail: one write of the
    # block per tick).
    return torch.stack(
        [
            torch.where(bent, column.reshape(E, N, K), bullets[:, BULLET_X + i])
            for i, column in enumerate(turned)
        ]
        + list(bullets[:, BULLET_SPEED + 1 :].unbind(1)),
        1,
    )


def _damage(
    state: State, flown: SimpleNamespace, bullets: Tensor, real: Tensor, contact: Contact
) -> Tensor:
    """Each real bullet's damage on its target, (E, N K) (0 for a miss or a ghost): `hit_damage`
    at the hitting pass's impact parameter and its path against the target's nose (the turned
    direction for a hit on a deflector's turned remainder)."""
    c = state.constants
    E, _, N, K = bullets.shape
    target = contact.target
    flat = (E, N * K)
    direction_x, direction_y, bullet_speed = bullets[:, BULLET_DIR_X : BULLET_SPEED + 1].unbind(1)
    direction_x, direction_y = direction_x.reshape(flat), direction_y.reshape(flat)
    if contact.bent_hit is not None:
        turned = contact.bent_hit
        direction_x, direction_y = (
            torch.where(
                turned, direction_x * contact.turn_cos - direction_y * contact.turn_sin, direction_x
            ),
            torch.where(
                turned, direction_x * contact.turn_sin + direction_y * contact.turn_cos, direction_y
            ),
        )
    nose_cos = _on_target(flown.heading_cos, target)
    nose_sin = _on_target(flown.heading_sin, target)

    def per_bullet(name: str) -> Tensor | float:  # the owner's constant over its K bullets
        value = c[name]
        return (
            value[..., None].expand(E, N, K).reshape(flat) if isinstance(value, Tensor) else value
        )

    hit_cos = -(direction_x * nose_cos + direction_y * nose_sin)  # 1 head-on
    hit_sin = (direction_x * nose_sin - direction_y * nose_cos).abs()
    damage = hit_damage(
        {name: per_bullet(name) for name in SHOOTER_HIT_CONSTANTS},
        {name: _on_target(c[name], target) for name in TARGET_HIT_CONSTANTS},
        bullet_speed.reshape(flat),
        contact.impact,
        hit_cos,
        hit_sin,
    )
    return torch.where(contact.hit & real, damage, 0.0)


# The constants `hit_damage` reads: the shooter's and the target's.
SHOOTER_HIT_CONSTANTS = ("hit_damage", "explosive_share", "inverse_reference_bullet_speed")
TARGET_HIT_CONSTANTS = (
    "hit_radius",
    "core_radius",
    "nose_cone",
    "nose_inner_sin",
    "inverse_nose_width",
)


def hit_damage(shooter, target, bullet_speed, impact, hit_cos, hit_sin) -> Tensor:
    """The hit model: one bullet's damage on a ship. Kinetic (the shooter's hit damage, the
    explosive share plus the rest by the squared ratio of the bullet's speed to its reference
    speed) x closeness (`graze`: the impact parameter `impact`, the distance from the target to
    the bullet's line (its turned path's after a deflector's turn), across the band between
    the target's core and hit radii) x aspect (the target's nose armour at the bullet's path
    against its nose, `hit_cos` 1 head-on and `hit_sin` its |sine|: `aspect_factor`, weighted
    by nose_cone).
    `shooter`, `target`: the constants by name (SHOOTER_HIT_CONSTANTS,
    TARGET_HIT_CONSTANTS; numbers or tensors broadcast with the rest). The tick's damage
    (`_damage`) and the scripted agent's gun and trigger (lean.scripted.opponent) read it, so a
    change of the hit model reaches both."""
    speed_ratio = bullet_speed * shooter["inverse_reference_bullet_speed"]
    explosive = shooter["explosive_share"]
    kinetic = explosive + (1.0 - explosive) * speed_ratio * speed_ratio
    closeness = graze(impact, target["core_radius"], target["hit_radius"])
    cone = aspect_factor(hit_cos, hit_sin, target["nose_inner_sin"], target["inverse_nose_width"])
    aspect = 1.0 + target["nose_cone"] * (cone - 1.0)
    return shooter["hit_damage"] * kinetic * closeness * aspect


def graze(distance: Tensor, core: Tensor | float, radius: Tensor | float) -> Tensor:
    """The closeness of a pass `distance` from a centre: 1 inside `core`, a smoothstep down to 0
    at `radius` (a bullet: the target's core and hit radii; a collision: the two ships' summed
    core and hit radii)."""
    width = radius - core
    width = width.clamp_min(1e-6) if isinstance(width, Tensor) else max(width, 1e-6)
    band = ((distance - core) / width).clamp(0.0, 1.0)
    return 1.0 - band * band * (3.0 - 2.0 * band)


def aspect_factor(
    hit_cos: Tensor,
    hit_sin: Tensor,
    inner_sin: Tensor | float,
    inverse_width: Tensor | float,
) -> Tensor:
    """The nose cone's curve (author 2026-11-25; on the sine 2026-11-28) at an angle off the
    nose given by its cosine `hit_cos` (1 dead ahead, -1 dead astern) and the |sine| `hit_sin`
    (|cross(nose, n)|, no trig): 0 inside the inner angle, 1 beyond the outer, a smootherstep of
    t = (sin - sin inner) / (sin outer - sin inner) between; behind the beam (a negative cosine)
    t = 1. `inner_sin`, `inverse_width`: one angle pair's constants (`_cone_rows`). A bullet's
    damage factor (`hit_damage`: the nose armour, the bullet's path against the target's nose)
    and, as 1 - aspect / 2, a collision's blame (`collisions`, the blame pair). Shared with the
    scripted opponent's gun."""
    t = ((hit_sin - inner_sin) * inverse_width).clamp(0.0, 1.0)
    return smootherstep(torch.where(hit_cos < 0.0, 1.0, t))


def _cone_rows(inner: Tensor, outer: Tensor, what: str) -> tuple[Tensor, Tensor]:
    """An angle pair of the card (radians, one per type) as `aspect_factor`'s constants: the
    inner angle's sine and 1 / (sin outer - sin inner). The outer angle must be larger than the
    inner and at most 90 deg (the sine rises on [0, 90 deg])."""
    if bool((outer <= inner).any()) or bool((outer > math.pi / 2 + 1e-6).any()):
        raise ValueError(f"{what}: the outer angle must be larger than the inner, at most 90 deg")
    inner_sin = torch.sin(inner)
    return inner_sin, 1.0 / (torch.sin(outer) - inner_sin)


# The hit width's scan: parallel rays this far apart (metres) across the deflector circle.
HIT_WIDTH_STEP = 0.05


def hit_widths(hit_radius: Tensor, inputs: dict[str, float]) -> Tensor:
    """The deflector's hit width (metres): the band of aim offsets of parallel bullets on a
    still ship whose turned path passes within the hit radius, one per `hit_radius` (...),
    under the hit-model `inputs` (deflector_radius, deflector_turn). The circle has no
    orientation, so the width is the same from every aspect. The game's own `circle_entry` and
    `circle_turn` on the tick's arithmetic (2 x the hit radius with no turn). Read by the
    scripted agent's trigger and gun; the constants' `hit_width` row, rebuilt by
    `with_hit_model`."""
    radius = torch.as_tensor(hit_radius, dtype=torch.float32)
    reach = float(radius.max()) + float(inputs["deflector_radius"]) + 1.0
    offsets = torch.arange(-reach, reach, HIT_WIDTH_STEP, dtype=torch.float64)
    far = 4.0 * reach
    # Rays along +x at height `offset`, from far behind to far beyond the centre.
    start_x = torch.full_like(offsets, -far)
    move_x = torch.full_like(offsets, 2.0 * far)
    zeros = torch.zeros_like(offsets)
    inverse_radius = 1.0 / float(inputs["deflector_radius"])
    entry = circle_entry(start_x, offsets, move_x, zeros, inverse_radius)
    entered = entry < torch.inf
    turn = circle_turn(
        start_x,
        offsets,
        move_x,
        zeros,
        torch.where(entered, entry, 0.0),
        inverse_radius,
        torch.tensor(float(inputs["deflector_turn"]), dtype=torch.float64),
    )
    # Straight (no entry: a miss of the circle) or turned, against each hit radius.
    squared = torch.where(entered, turn.miss_squared, offsets * offsets)
    inside = squared[None] < (radius.reshape(-1, 1).double() ** 2)
    widths = HIT_WIDTH_STEP * inside.sum(-1).to(torch.float32)
    return widths.reshape(radius.shape)


class HitInput(NamedTuple):
    """One input of the hit model (`HIT_MODEL`): its name, label, unit and the workbench's and
    sliders' range."""

    name: str
    label: str
    unit: str
    low: float
    high: float


# The hit model's inputs (proposal 2026-11-27): every number of a bullet's hit and damage, each a
# runtime row of the constants (RUNTIME_CONSTANTS), changed by `with_hit_model` with no recompile.
# One hit model for every ship of a state. Angles in radians.
HIT_MODEL = (
    HitInput("hit_radius", "hit radius", "m", 2.0, 30.0),
    HitInput("core_radius", "core radius", "m", 0.0, 20.0),
    HitInput("hit_damage", "hit damage", "shield points", 1.0, 40.0),
    HitInput("bullet_spread", "bullet spread (velocity noise per axis)", "m/s", 0.0, 60.0),
    HitInput("deflector_radius", "deflector: circle radius", "m", 5.0, 60.0),
    HitInput("deflector_turn", "deflector: largest turn", "rad", 0.0, 1.57),
    HitInput("nose_cone", "nose armour's rule (1 on, 0 off)", "", 0.0, 1.0),
    HitInput("nose_inner_angle", "nose armour: no damage inside", "rad", 0.0, 1.4),
    HitInput("nose_outer_angle", "nose armour: full damage beyond", "rad", 0.05, 1.57),
)
_HIT_SAME = (
    "hit_radius",
    "core_radius",
    "hit_damage",
    "deflector_radius",
    "deflector_turn",
    "nose_cone",
)
# The inputs the hit width reads.
_WIDTH_INPUTS = ("deflector_radius", "deflector_turn")


def hit_model_constants(inputs: dict[str, float]) -> dict[str, float]:
    """The constants (RUNTIME_CONSTANTS names, except the hit widths) of hit-model inputs
    (`HIT_MODEL` names; the nose cone's pair only when both angles are given)."""
    out = {name: float(inputs[name]) for name in _HIT_SAME if name in inputs}
    if "deflector_radius" in inputs:
        out["inverse_deflector_radius"] = 1.0 / float(inputs["deflector_radius"])
    if "bullet_spread" in inputs:
        out["spread_speed"] = float(inputs["bullet_spread"])
    if "nose_inner_angle" in inputs and "nose_outer_angle" in inputs:
        inner_sin, inverse_width = _cone_rows(
            torch.tensor(float(inputs["nose_inner_angle"]), dtype=torch.float64),
            torch.tensor(float(inputs["nose_outer_angle"]), dtype=torch.float64),
            "the nose armour",
        )
        out["nose_inner_sin"], out["inverse_nose_width"] = float(inner_sin), float(inverse_width)
    return out


def hit_model_of(constants: Constants) -> dict[str, float]:
    """The hit model's inputs (`HIT_MODEL`) as `constants` hold them (the first slot's: one hit
    model for every ship)."""

    def first(name: str) -> float:
        return float(torch.as_tensor(constants[name]).flatten()[0])

    out = {name: first(name) for name in _HIT_SAME}
    out["bullet_spread"] = first("spread_speed")
    inner_sin = first("nose_inner_sin")
    outer_sin = inner_sin + 1.0 / first("inverse_nose_width")
    out["nose_inner_angle"] = math.asin(max(-1.0, min(1.0, inner_sin)))
    out["nose_outer_angle"] = math.asin(max(-1.0, min(1.0, outer_sin)))
    return out


def with_hit_model(constants: Constants, **inputs: float) -> Constants:
    """`constants` (a state's, or a ship card's) with hit-model inputs (`HIT_MODEL` names)
    changed for every ship, written into the table's rows (the hit widths rebuilt): the same
    shapes, so the compiled tick runs on with no recompile. Raises on an unknown name. Use as
    `state.replace(constants=with_hit_model(state.constants, hit_radius=14.0))`."""
    known = {entry.name for entry in HIT_MODEL}
    unknown = set(inputs) - known
    if unknown:
        raise KeyError(f"not hit-model inputs: {sorted(unknown)}")
    merged = {**hit_model_of(constants), **inputs}
    table = constants.table.clone()  # (C, T)
    for name, value in hit_model_constants(merged).items():
        start, _ = constants.columns[name]
        table[start] = value
    start, _ = constants.columns["hit_width"]
    radius = table[constants.columns["hit_radius"][0]]  # (T,)
    unique, inverse = radius.cpu().unique(return_inverse=True)  # built once per radius
    widths = hit_widths(unique, {name: merged[name] for name in _WIDTH_INPUTS})[inverse]
    table[start] = widths.to(table.device)
    return dataclasses.replace(constants, table=table)


class Collisions(NamedTuple):
    """The collision test per ordered pair (E, me, other)."""

    contact: Tensor  # bool: the pair first touches in this tick, both alive (symmetric)
    closing: Tensor  # m/s: |w|, the relative speed
    miss: Tensor  # m: b, the closest distance of the tick's relative line, extended
    nose_cos: Tensor  # cos(my nose, the line of centres at first touch, me to the other)
    blame: (
        Tensor  # 1 - aspect / 2 on the blame pair: 1 inside its inner angle, 0.5 beyond its outer
    )
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
    blame 1 - aspect / 2 at my nose against n (aspect_factor on the card's blame pair, the sine
    |nose x n|: 1 dead-on, 0.5 beyond the outer angle, 45 deg, and behind the beam). I take
    max shield x (|w| / tick) / (2 x cruise) x graze(b; R_core, R) x blame: linear in the
    closing speed, a full shield's worth head-on at cruise with full blame; full closeness when
    the cores would meet, a graze when only the hit radii do."""
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
    sine = (nose_cos[:, :, None] * line_y - nose_sin[:, :, None] * line_x).abs()
    blame = 1.0 - 0.5 * aspect_factor(
        cosine, sine, _per_me(c["blame_inner_sin"]), _per_me(c["inverse_blame_width"])
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
    depth = boundary_distance(config, flown.x, flown.y)
    ramp = smootherstep((depth / config.boundary_ramp_width).clamp(0.0, 1.0))
    return torch.where(alive, card.BOUNDARY_MAX_DAMAGE * ramp * config.tick_length, 0.0)


def _shields(
    config: GameConfig, state: State, damage: Tensor, stick: Tensor, power: Tensor, shoot: Tensor
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The tick's damage (E, N) against the shields. A living ship hurt with its shield already
    at 0 dies; the shield takes the damage down to 0; a ship not hurt and quiet (coast, straight,
    no trigger; author 2026-11-26: every turn level resets the wait) recharges once its wait has
    run out; being hurt or not quiet restarts the wait. Returns died (E, N) bool, the shield
    after the damage, the recharge (restored) and the wait."""
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


def bullet_ages(tick: Tensor, slots: int = BULLET_SLOTS) -> Tensor:
    """(E, K) long: each ring slot's bullet age on the tick `tick` (E,), the bullet written on
    that tick 0, K - 1 on its last tick; it left the gun on tick - age. A step passes its
    tick-start count; a state between ticks wrote its newest bullets on state.tick - 1."""
    return (tick[:, None] - torch.arange(slots, device=tick.device)) % slots


def _bullet_ends(
    bullet_state: Tensor, contactable: Tensor, age: Tensor, hit: Tensor, enemy: Tensor
) -> tuple[Tensor, Tensor]:
    """Each bullet's state after this tick (CONTACTED once it touches a ship) and its outcome,
    reported once, (E, N, K): at its first contact (1 hit an enemy, 2 hit a team-mate) or on
    its last tick (age K - 1) without one (0 missed); -1 otherwise. A contacted bullet (and
    every slot of a fresh deal) never reports. bullet_state, contactable (E, N, K); age (E, K);
    hit, enemy (E, N K) bool."""
    shape = bullet_state.shape
    hit = hit.reshape(shape)
    last = (age == shape[-1] - 1)[:, None]
    ended = hit | (last & contactable)
    outcome = torch.where(hit, torch.where(enemy.reshape(shape), 1, 2), 0)
    return torch.where(hit, CONTACTED, bullet_state), torch.where(ended, outcome, -1)


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
    """The bullet block of a fresh deal: every bullet at the origin, still, pointing along x (its
    state CONTACTED: it never touches a ship or reports, and is overwritten in its slot's tick)."""
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
        bullet_state=filled(bullet_shape, CONTACTED, torch.uint8, device),
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
    shared = ("bullets", "bullet_state", "tick", "pending")
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
        elif field.name == "constants":  # the card's table is shared: only the slots' types
            values[field.name] = dataclasses.replace(
                old, ship_type=select(new.ship_type, old.ship_type)
            )
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
    sight egg, author 2026-11-28). The observer sees inside an egg in its own frame: with d the
    offset target - observer, along = d . nose and across = d x nose, the front half (along >= 0)
    the ellipse (along / ahead)^2 + (across / side)^2 <= 1, the rear half the same with behind for
    ahead; ahead, side, behind the card's vision_ahead / vision_side / vision_behind (the
    constants' inverses: multiplies and a sign check, no square root). A target in `tracked` (E,
    N) bool (visible to this team on the tick before, `tracked_by_enemy`; author 2026-11-26) is
    held inside the tracking circle instead, |d| <= tracking_range (the longest of the three
    reaches: the circle contains the egg). None: the egg for every target. Sight is blocked only
    by fields' cores: field i, with core radius max(r - w / 2, 0) and outer radius r + w / 2,
    blocks the line when exactly one end is strictly inside its core and the two ends are not
    both inside its outer disc (core and band see each other), or when neither end is inside its
    core and the segment passes strictly within the core radius of the centre. Dead targets are
    tested like living ones (a wreck in sight is seen); dead observers see nothing. Bullets are
    not perceived."""
    c = state.constants
    x, y = state.x, state.y
    line_x = x[:, :, None] - x[:, None, :]  # (E, target, observer): target minus observer, d
    line_y = y[:, :, None] - y[:, None, :]

    def per_observer(value):  # a uniform number, or (E, N) per slot as (E, 1, observer)
        return value[:, None, :] if isinstance(value, Tensor) else value

    nose_x, nose_y = state.heading_cos[:, None, :], state.heading_sin[:, None, :]
    along = line_x * nose_x + line_y * nose_y
    across = (line_y * nose_x - line_x * nose_y) * per_observer(c["inverse_vision_side"])
    along = along * torch.where(
        along >= 0.0,
        per_observer(c["inverse_vision_ahead"]),
        per_observer(c["inverse_vision_behind"]),
    )
    inside = along * along + across * across <= 1.0
    if tracked is not None:
        reach = per_observer(c["tracking_range"])
        held = line_x * line_x + line_y * line_y <= reach * reach
        inside = torch.where(tracked[:, :, None], held, inside)
    clear = inside & state.alive[:, None, :] & (state.team[:, :, None] != state.team[:, None, :])
    if state.fields.shape[-1] > 0:
        clear = clear & ~field_blocks(
            state.fields, x[:, :, None], y[:, :, None], x[:, None, :], y[:, None, :]
        )
    return clear


def field_blocks(fields: Tensor, a_x: Tensor, a_y: Tensor, b_x: Tensor, b_y: Tensor) -> Tensor:
    """Whether the fields' cores block the line between points a and b (any shapes that
    broadcast, leading E; fields (E, 5, F), F > 0), as sight is blocked (`sight_pairs`, a the
    target and b the observer): field i, with core radius max(r - w / 2, 0) and outer radius
    r + w / 2, blocks when exactly one end is strictly inside its core and the two ends are not
    both inside its outer disc (core and band see each other), or when neither end is inside its
    core and the segment passes strictly within the core radius of the centre. Also the scripted
    agent's cover (lean.scripted.opponent)."""
    shape = (fields.shape[0],) + (1,) * (max(a_x.dim(), b_x.dim()) - 1) + (fields.shape[-1],)

    def per_field(row: int) -> Tensor:
        return fields[:, row].reshape(shape)

    radius, half_band = per_field(FIELD_RADIUS), 0.5 * per_field(FIELD_BAND)
    core = (radius - half_band).clamp_min(0.0)
    core_squared, outer = core * core, radius + half_band
    centre_x, centre_y = per_field(FIELD_X), per_field(FIELD_Y)
    a_offset_x, a_offset_y = a_x[..., None] - centre_x, a_y[..., None] - centre_y
    b_offset_x, b_offset_y = b_x[..., None] - centre_x, b_y[..., None] - centre_y
    a_squared = a_offset_x * a_offset_x + a_offset_y * a_offset_y
    b_squared = b_offset_x * b_offset_x + b_offset_y * b_offset_y
    core_a, core_b = a_squared < core_squared, b_squared < core_squared
    field_a, field_b = a_squared < outer * outer, b_squared < outer * outer
    ends = (core_a ^ core_b) & ~(field_a & field_b)
    line_x, line_y = (a_x - b_x)[..., None], (a_y - b_y)[..., None]
    # u = centre - b = -(b's offset); along = u . d, across = u x d.
    along = -(b_offset_x * line_x + b_offset_y * line_y)
    across = b_offset_x * line_y - b_offset_y * line_x
    length = line_x * line_x + line_y * line_y
    passes = (
        ~(core_a | core_b)
        & (across * across < core_squared * length)
        & (along > 0.0)
        & (along < length)
    )
    return (ends | passes).any(-1)


def _snapshot(state: State) -> Tensor:
    """(E, SEEN_COLUMNS, N) float32: each ship's raw values now, as a sighting records them.
    Columns on the middle axis, so each column read back is an (E, N) slice with unit stride
    along N."""
    bullet_share = (state.bullet_state == REAL).to(torch.float32).mean(-1)
    alive = state.alive.to(torch.float32)
    return torch.cat(
        [state.ships, torch.stack([bullet_share, alive, torch.ones_like(alive)], 1)], 1
    )


def perceive(state: State, revealed: Tensor, tracking: bool = True) -> State:
    """The state with each ship's sighting by its enemy team brought up to date: a ship in the
    enemy's sight or `revealed` (E, N) bool (it fired, was hit or collided this tick) is
    recorded as it is now and its ticks since seen set to 0; any other ship keeps its last-seen
    row and counts up to the cap. Each ship has one enemy team, so one row per ship serves. tracking
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
        bullet_state=None,
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


# --------------------------------------------------------------------------- relative geometry

# The learner's geometry heads read the game through these (lean.model's pairwise head, the
# map-edge column; one source of truth for their targets). Per-ship columns of
# `geometry_record`, besides the position and heading the observation carries.
GEOMETRY_COLUMNS = (
    "velocity_x",  # world velocity, m/s (the stored velocity over the refractive index)
    "velocity_y",
    "launch_x",  # a shot's launch velocity over the map without the spread, m/s
    "launch_y",
    "energy",  # the store
    "edge",  # signed distance beyond the arena's edge, m (negative inside: `boundary_distance`)
    "bullet_drag",  # the ship's bullet drag, 1/m
    "bullet_life",  # a bullet's life, s (BULLET_SLOTS ticks)
)
# How many fixed-point passes `pair_geometry` makes for the lead point's time of flight.
LEAD_PASSES = 3


def boundary_distance(config: GameConfig, x: Tensor, y: Tensor) -> Tensor:
    """The signed distance beyond the arena's edge, metres (negative inside): boundary damage
    rises over the ramp from 0 (`_boundary_damage`); the observation's boundary scalar is its
    negative."""
    return torch.sqrt(x * x + y * y) - config.arena_radius


def field_edge_distance(
    centre_x: Tensor, centre_y: Tensor, radius: Tensor, x: Tensor, y: Tensor
) -> Tensor:
    """The signed distance of points (x, y) to a field's edge, metres (negative inside): the
    distance to its centre less its radius, the middle of its band (`medium`). Any shapes that
    broadcast."""
    offset_x, offset_y = x - centre_x, y - centre_y
    return torch.sqrt(offset_x * offset_x + offset_y * offset_y) - radius


def world_velocity(ships) -> tuple[Tensor, Tensor]:
    """The world (map) velocity of ships with stored velocity and log index (a State or any
    view of it): the stored velocity, at proper speed, over the refractive index."""
    inverse_index = torch.exp(-ships.log_index)
    return ships.velocity_x * inverse_index, ships.velocity_y * inverse_index


def launch_velocity(state: State) -> tuple[Tensor, Tensor]:
    """A shot's launch velocity (proper speed) before the spread (`_fire`): the ship's stored
    velocity plus the muzzle speed along the nose."""
    c = state.constants
    return (
        state.velocity_x + c["muzzle_speed"] * state.heading_cos,
        state.velocity_y + c["muzzle_speed"] * state.heading_sin,
    )


def bullet_reach(drag, speed, time):
    """How far a bullet launched at `speed` (map speed) travels in `time` under the drag law
    (speed e^(-drag s) after distance s, as `bullet_motion`'s half-drag steps give): ln(1 + drag
    speed time) / drag."""
    return torch.log1p(drag * speed * time) / drag


def bullet_flight_time(drag, speed, distance):
    """The inverse of `bullet_reach`: the time a bullet launched at `speed` takes to travel
    `distance`, (e^(drag distance) - 1) / (drag speed)."""
    return torch.expm1(drag * distance) / (drag * speed)


def visible_rows(config: GameConfig, state: State, team) -> Tensor:
    """(E, N) bool, Obs.visible of team `team`'s view (an int, or (E, 1) long): its own rows and
    the enemies it sees this tick (seen on this tick and known); every row with fog off."""
    if not config.fog:
        return torch.ones_like(state.exists)
    same_team = state.team == team
    return same_team | ((state.ticks_since_seen == 0) & (state.seen[:, SEEN_KNOWN] > 0.5))


def geometry_record(config: GameConfig, state: State) -> Tensor:
    """(E, N, GEOMETRY_COLUMNS) float32: each ship's per-ship geometry now (its true state; the
    learner keeps only the rows a team sees). With the position and heading (Obs.pos,
    Obs.heading) it is what `pair_geometry` and the map-edge target read."""
    velocity_x, velocity_y = world_velocity(state)
    launch_x, launch_y = launch_velocity(state)
    inverse_index = torch.exp(-state.log_index)
    drag = state.constants["bullet_drag"]
    columns = {
        "velocity_x": velocity_x,
        "velocity_y": velocity_y,
        "launch_x": launch_x * inverse_index,
        "launch_y": launch_y * inverse_index,
        "energy": state.energy,
        "edge": boundary_distance(config, state.x, state.y),
        "bullet_drag": drag if isinstance(drag, Tensor) else torch.full_like(state.x, drag),
        "bullet_life": torch.full_like(state.x, BULLET_SLOTS * config.tick_length),
    }
    return torch.stack([columns[name].expand_as(state.x) for name in GEOMETRY_COLUMNS], -1)


def train_angle(nose_x, nose_y, line_x, line_y):
    """The antenna train angle: from a nose direction (nose_x, nose_y) to the line of sight
    (line_x, line_y), radians in (-pi, pi], counter-clockwise positive. `pair_geometry`'s
    train_angle; seen from the other ship, |train angle| over the rear cone's half-angle is a
    pursuer behind it (`lean.eval.style.TAIL_CHASE_ANGLE`)."""
    return torch.atan2(nose_x * line_y - nose_y * line_x, nose_x * line_x + nose_y * line_y)


def pair_geometry(first_position, first_heading, first, second_position, second):
    """The relative geometry of ship pairs, first on second, each (...) float32: positions
    (..., 2), headings (...) radians, `first` / `second` (..., GEOMETRY_COLUMNS) rows of
    `geometry_record` (second's heading is not needed).

    range: |d|, d = second's position - first's (m). range_rate: d . w / |d|, w = second's
    world velocity - first's (m/s, negative closing). line_of_sight_rate: (d x w) / |d|^2
    (rad/s, counter-clockwise positive). The three are symmetric in the order.
    energy_difference: first's store - second's (antisymmetric).
    train_angle: the antenna train angle, from first's nose to the line of sight to second
    (radians in (-pi, pi], counter-clockwise positive). gun_error: from the launch direction
    of a shot by first now (`launch_velocity` over the map, no spread) to the lead point,
    second's position after the bullet's time of flight to it at second's world velocity
    (`bullet_flight_time` at first's launch speed and drag, capped at the bullet's life: no
    lead beyond what a bullet can reach; LEAD_PASSES fixed-point passes from the range),
    radians in (-pi, pi]. Both depend on the order. The aspect angle and the
    heading crossing angle follow from the train angles in both orders."""
    column = {name: index for index, name in enumerate(GEOMETRY_COLUMNS)}

    def read(rows, name):
        return rows[..., column[name]]

    line_x = second_position[..., 0] - first_position[..., 0]
    line_y = second_position[..., 1] - first_position[..., 1]
    squared = (line_x * line_x + line_y * line_y).clamp_min(1e-6)
    distance = torch.sqrt(squared)
    relative_x = read(second, "velocity_x") - read(first, "velocity_x")
    relative_y = read(second, "velocity_y") - read(first, "velocity_y")
    nose_x, nose_y = torch.cos(first_heading), torch.sin(first_heading)
    launch_x, launch_y = read(first, "launch_x"), read(first, "launch_y")
    launch_speed = torch.sqrt(launch_x * launch_x + launch_y * launch_y).clamp_min(1e-6)
    drag = read(first, "bullet_drag")
    lead_x, lead_y, reach = line_x, line_y, distance
    for _ in range(LEAD_PASSES):
        flight = torch.minimum(
            bullet_flight_time(drag, launch_speed, reach), read(first, "bullet_life")
        )
        lead_x = line_x + read(second, "velocity_x") * flight
        lead_y = line_y + read(second, "velocity_y") * flight
        reach = torch.sqrt(lead_x * lead_x + lead_y * lead_y)
    return {
        "range": distance,
        "range_rate": (line_x * relative_x + line_y * relative_y) / distance,
        "line_of_sight_rate": (line_x * relative_y - line_y * relative_x) / squared,
        "energy_difference": read(first, "energy") - read(second, "energy"),
        "train_angle": train_angle(nose_x, nose_y, line_x, line_y),
        "gun_error": torch.atan2(
            launch_x * lead_y - launch_y * lead_x, launch_x * lead_x + launch_y * lead_y
        ),
    }


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
    step's stored flag; the share of bullet slots in flight counts real bullets not yet
    contacted (ghosts and contacted bullets are not observed). Position is not a scalar (Obs.pos
    carries it, for the position code and the rotary angles).

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
            (state.bullet_state == REAL).to(torch.float32).mean(-1),
            enemy_seen[:, 0],
        )
        # For the globals: own ships alive, enemies not seen dead (never-seen count as alive).
        counted = torch.where(same_team, state.alive, enemy_seen[:, SEEN_ALIVE - SEEN_SHARE] > 0.5)
    else:
        view = state
        in_flight = state.bullet_state == REAL
        bullet_share = in_flight.to(torch.float32).mean(-1)
    exists, alive = state.exists, view.alive
    # World-frame motion, as v2 observes it: the proper speed over the refractive index, so a
    # field that slows a ship shows as a slower ship (src/bnb/game/perception/observe.py).
    if state.fields.shape[-1] == 0:  # no medium (n = 1)
        speed, velocity_x, velocity_y = view.speed, view.velocity_x, view.velocity_y
    else:
        speed = view.speed * torch.exp(-view.log_index)
        velocity_x, velocity_y = world_velocity(view)  # stored
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
        (-boundary_distance(config, view.x, view.y) / BOUNDARY_CLIP).clamp(-1.0, 1.0),
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
    visible = visible_rows(config, state, team)
    if config.fog:
        hidden_ticks = torch.where(same_team, 0, state.ticks_since_seen)
        since_seen = torch.log1p(hidden_ticks.to(torch.float32) * h)
    else:
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
