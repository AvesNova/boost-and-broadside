"""The flight model's design assumptions (docs/rewrite/core-decisions.md, "Flight physics").

Each test names the rule it guards. They fly one ship with `flight_tick` (or one game step) in
`GameConfig()`; no fields.
"""

import dataclasses
import math
from types import SimpleNamespace

import pytest
import torch

from lean.game import card, core, tuning
from lean.game.tests.values import game_value

torch.set_num_threads(1)
CONFIG = core.GameConfig()
P51D = card.TYPE_NAMES.index("p51d")
COAST, DIVE, CLIMB = 0, 1, 2  # the power indices (lean.game.card: 0 coast, 1 dive, 2 climb)
# The stick indices, one side (card.STICK_LEVELS; tuning.LEVEL_INDICES names the levels).
STRAIGHT = card.STICK_LEVELS.index(0.0)
SUSTAIN, MAX, SNAP = (tuning.LEVEL_INDICES[name] for name in ("sustain", "max", "snap"))
# The flights' start speeds, as shares of the game's top level speed and corner speed.
TOP = game_value("cruise_speed", CONFIG)
CORNER = game_value("corner_speed", CONFIG)
NO_FIELDS = torch.zeros(1, 5, 0)


def constants(ship_type=P51D, slots=1):
    """One ship type's constants for (1, slots) slots, in a run of that type alone."""
    config = dataclasses.replace(CONFIG, lineup=(card.TYPE_NAMES[ship_type],))
    return core.ship_constants(config, torch.full((1, slots), ship_type))


def fly(speed, store_fraction, stick, power, seconds, thrust_scale=1.0, ship_type=P51D):
    """Fly one ship; returns per-tick speed and store fraction (lists) and the constants."""
    c = constants(ship_type)
    z = torch.zeros(1, 1)
    s = SimpleNamespace(
        x=z.clone(),
        y=z.clone(),
        heading_cos=torch.ones(1, 1),
        heading_sin=z.clone(),
        speed=torch.full((1, 1), float(speed)),
        angle=z.clone(),
        angle_cos=torch.ones(1, 1),
        angle_sin=z.clone(),
        nose_rate=z.clone(),
        energy=store_fraction * torch.full((1, 1), c["max_energy"]),
        log_index=z.clone(),
        gradient_x=z.clone(),
        gradient_y=z.clone(),
    )
    stick_t = torch.full((1, 1), stick, dtype=torch.long)
    power_t = torch.full((1, 1), power, dtype=torch.long)
    speeds, stores = [], []
    with torch.no_grad():
        for _ in range(round(seconds / CONFIG.tick_length)):
            s = core.flight_tick(
                c, NO_FIELDS, CONFIG.tick_length, s, stick_t, power_t, thrust_scale=thrust_scale
            )
            speeds.append(s.speed.item())
            stores.append((s.energy / c["max_energy"]).item())
    return speeds, stores


def control_for(c, stick_size, power, shape=(1, 1)):
    """The flight functions' control (`core._setting`) for a ship of a run's constants `c`: stick
    size `stick_size` (a fraction of the envelope's top line, Max), power index `power`, no
    Sustain table, no deep angle."""
    return SimpleNamespace(
        thrust_scale=1.0,
        stick_size=torch.full(shape, float(stick_size)),
        stick_sign=torch.ones(shape),
        sustain=torch.zeros(shape),
        sustain_table=None,
        deep=torch.zeros(shape),
        commanded=c.commanded_table[:, power],
        demand=c.demand_table[:, power],
    )


def one_g_speed(ship_type=P51D):
    """The speed below which the wing cannot carry 1 g flying straight."""
    return (card.GRAVITY / card.CARD["max_lift_coefficient"][ship_type]) ** 0.5


def test_store_is_two_loops_at_corner_speed_and_spawn_is_half():
    """The store holds `store_loops` vertical loops at corner speed; ships spawn at half."""
    c = constants()
    lift, limit = card.CARD["max_lift_coefficient"][P51D], card.CARD["max_acceleration"][P51D]
    corner_radius = (limit / lift) / (limit**2 - card.GRAVITY**2) ** 0.5
    expected = CONFIG.store_loops * card.GRAVITY * 2.0 * corner_radius
    assert abs(c["max_energy"] - expected) < 1e-3 * expected
    assert abs(c["spawn_energy"] - 0.5 * expected) < 1e-3 * expected


def test_coasting_straight_holds_altitude():
    """Coast holds altitude: flying straight above the 1 g speed, the store does not change."""
    _, stores = fly(0.54 * TOP, 0.5, STRAIGHT, COAST, 30.0)
    assert max(abs(x - 0.5) for x in stores) < 1e-4


def test_coasting_on_the_top_line_holds_altitude_above_the_one_g_speed():
    """Altitude wins over the turn: at Max (the wing's maximum lift below corner), coasting,
    above the 1 g speed, the store holds (the turn is limited instead)."""
    speeds, stores = fly(1.05 * CORNER, 0.5, MAX, COAST, 10.0)
    assert min(speeds) > 1.2 * one_g_speed()
    assert max(abs(x - 0.5) for x in stores) < 2e-3


def test_holding_snap_below_corner_bleeds_speed():
    """The post-stall regime (author 2026-11-26): Snap below corner goes to the deep angle, whose
    drag bleeds speed far faster than Max's maximum lift."""
    start = 0.8 * CORNER
    snap, _ = fly(start, 0.5, SNAP, COAST, 1.0)
    held_max, _ = fly(start, 0.5, MAX, COAST, 1.0)
    assert start - snap[-1] > 3.0 * (start - held_max[-1])


def test_engine_never_raises_the_store():
    """The engine drives speed only: coasting at any stick, the store never rises."""
    for stick in (STRAIGHT, SUSTAIN, MAX, SNAP):
        _, stores = fly(0.4 * TOP, 0.3, stick, COAST, 10.0)
        assert max(stores) <= 0.3 + 1e-6, stick


def test_below_the_one_g_speed_the_ship_sinks():
    """A wing that cannot carry 1 g sinks: store into speed."""
    speeds, stores = fly(0.7 * one_g_speed(), 0.5, STRAIGHT, COAST, 1.0)
    assert stores[-1] < 0.5 - 1e-3
    assert speeds[-1] > 0.7 * one_g_speed()


def test_dive_and_climb_are_equal_and_opposite():
    """Dive and climb trade at the same rate: straight, mid-band, the store's rates are opposite."""
    c = constants()
    speed = torch.full((1, 1), 0.67 * TOP)
    energy = torch.full((1, 1), 0.5 * c.max_energy)
    rates = []
    for power in (DIVE, CLIMB):
        setting = core._setting(c, control_for(c, 0.0, power), speed, energy, 0.0)
        _, store_rate = core._rates(
            c,
            setting,
            torch.ones(1, 1),
            speed,
            torch.zeros(1, 1),
            torch.ones(1, 1),
            1.0 / (speed * speed),
            1.0 / (c.lift * c.lift),
        )
        rates.append(store_rate.item())
    assert rates[0] < 0.0 < rates[1]
    assert abs(rates[0] + rates[1]) < 1e-3 * abs(rates[1])


def test_dive_at_an_empty_store_and_climb_at_a_full_store_do_nothing():
    """The band's edges: no dive below empty, no climb above full."""
    _, stores = fly(0.8 * TOP, 0.0, STRAIGHT, DIVE, 5.0)
    assert max(stores) < 1e-6 and min(stores) >= 0.0
    _, stores = fly(0.67 * TOP, 1.0, STRAIGHT, CLIMB, 5.0)
    assert max(stores) <= 1.0 + 1e-6


def test_the_edges_taper_smoothly():
    """Smootherstep edges: climbing into the top band, the trade's rate falls continuously to zero
    and never reverses or overshoots the top. (At full climb rate the 5% band is crossed in a few
    ticks with today's numbers: a tuning question, docs/rewrite/core-worklog.md.)"""
    _, stores = fly(0.8 * TOP, 0.9, STRAIGHT, CLIMB, 4.0)
    steps = [b - a for a, b in zip(stores, stores[1:])]
    assert all(x >= -1e-7 for x in steps)
    inside = [x for x, level in zip(steps, stores[1:]) if level > 1.0 - CONFIG.store_edge_band]
    assert all(b <= a + 1e-7 for a, b in zip(inside, inside[1:]))
    assert max(stores) <= 1.0 + 1e-6


def test_firing_costs_speed_and_no_store():
    """Firing cuts the engine: well above stall speed (above GameConfig.fire_store_band) firing
    costs speed, and the store pays nothing (one game step per tick with the gun held, against
    the same flight without firing)."""
    speeds_firing, _ = fly(
        0.8 * TOP, 0.5, STRAIGHT, COAST, 5.0, thrust_scale=CONFIG.fire_thrust_fraction
    )
    speeds_quiet, _ = fly(0.8 * TOP, 0.5, STRAIGHT, COAST, 5.0)
    assert speeds_firing[-1] < speeds_quiet[-1] - 5.0

    generator = torch.Generator().manual_seed(0)
    state = core.reset(CONFIG, 1, torch.tensor([[1, 1]]), generator, torch.device("cpu"))
    before = state.energy.clone()
    action = torch.tensor([[[0, 0, 1], [0, 0, 0]]])
    noise = core.draw_noise(CONFIG, 1, 2, generator, torch.device("cpu"))
    after, _ = core.step(
        CONFIG, state.replace(pending=action), action, noise
    )  # pending: applied now
    assert bool(after.fired[0, 0])
    assert torch.allclose(after.energy[0, 0], before[0, 0], rtol=1e-6)


def test_at_the_edges_dive_and_climb_act_as_coast_on_the_wing():
    """At the band's edges dive or climb acts as coast: diving on an empty store frees no lift
    for the turn, and climbing at a full store takes none from it."""
    coast_empty, _ = fly(0.4 * TOP, 0.0, SNAP, COAST, 2.0)
    dive_empty, _ = fly(0.4 * TOP, 0.0, SNAP, DIVE, 2.0)
    assert abs(dive_empty[-1] - coast_empty[-1]) < 0.5
    rates = {}
    for name, power, store in (("coast", COAST, 1.0), ("climb", CLIMB, 1.0)):
        c = constants()
        speed = torch.full((1, 1), 0.54 * TOP)
        store_energy = torch.full((1, 1), store * c.max_energy)
        lateral, *_ = core._setting(c, control_for(c, 1.0, power), speed, store_energy, 0.0)
        rates[name] = lateral.item()
    assert abs(rates["climb"] - rates["coast"]) < 1e-3 * abs(rates["coast"])


def test_tightest_turn_clearly_slower_than_best_rate_then_corner():
    """The author's requirement (2026-11-23, with a margin 2026-11-24): slowest to fastest, the
    minimum sustained radius at least 10% below the best sustained turn rate's speed, which is
    below the corner speed (the peak full-stick rate), for each ship type (quasi-steady,
    coasting, spawn store: lean.game.tuning.performance, which flies the game's own flight
    functions). So the one-circle (radius) and two-circle (rate) fights ask for different
    speeds."""
    for name in card.TYPE_NAMES:
        p = tuning.performance(tuning.row_of(name))
        assert p["min_radius_speed"] <= 0.9 * p["sustained_speed"], (
            name,
            p["min_radius_speed"],
            p["sustained_speed"],
        )
        assert p["sustained_speed"] < p["corner_speed_flown"], (
            name,
            p["sustained_speed"],
            p["corner_speed_flown"],
        )


def double_constants(ship_type=P51D, row=None):
    """One type's per-type constants (core.per_type_constants) in float64, shape (1,); or those
    of `row` (a card row) when given."""
    if row is None:
        c = core.per_type_constants(CONFIG, torch.tensor([ship_type]))
    else:
        with tuning._with_row(row) as index:
            c = core.per_type_constants(CONFIG, torch.tensor([index]))
    return SimpleNamespace(
        **{k: v.double() if v.is_floating_point() else v for k, v in vars(c).items()}
    )


def test_turn_rate_never_falls_with_speed_below_corner():
    """The author's rule (2026-11-24): below corner speed (the peak of Max, the envelope's top
    line) Max's turn rate never decreases as speed rises, for every card type (quasi-steady,
    coasting, the spawn store, float64, from half the 1 g speed). The peak is found on the dense
    grid itself, up to one step of tuning.performance's grid past its corner_speed_flown. Not
    Sustain (it peaks at the best sustained speed) nor Snap (its small rise toward the stall
    near stall speed, docs/rewrite/sections/post-stall/progress.md step 8, an open item)."""
    for ship_type, name in enumerate(card.TYPE_NAMES):
        row = tuning.row_of(name)
        c = double_constants(row=row)
        corner = tuning.performance(row)["corner_speed_flown"] + 0.05
        speeds = torch.linspace(0.5 * one_g_speed(ship_type), corner, 8001, dtype=torch.float64)
        full = core.quasi_steady(c, speeds, torch.ones_like(speeds), COAST, c.spawn_energy)[0]
        peak = int(full.argmax()) + 1
        rate = full[:peak]
        fall = rate[:-1] - rate[1:]
        assert float(fall.max()) <= 1e-12 * float(rate.max()), (
            name,
            float(speeds[int(fall.argmax())]),
        )


def top_line_lateral(c, speeds, stick=None):
    """The lateral acceleration `_setting` gives coasting at the spawn store (float64) at stick
    size `stick` (default 1: Max, the envelope's top line), the wing alone holding the 1 g."""
    stick = torch.ones_like(speeds) if stick is None else stick
    control = core.steady_control(c, speeds, stick, COAST)
    return core._setting(c, control, speeds, c.spawn_energy, 0.0)[0]


def level_line(c, speeds, index, power=COAST):
    """A stick level's quasi-steady (turn rate, d speed / dt, total energy rate) at the spawn
    store, its own shape from the constants' tables (core.stick_level_shape)."""
    size, sustain, deep = core.stick_level_shape(c, index)
    sticks = torch.full_like(speeds, size)
    return core.quasi_steady(c, speeds, sticks, power, c.spawn_energy, sustain, deep)


def steady_angle(c, speeds, index):
    """The angle of attack a stick level asks for, steadily coasting at the spawn store: the
    setting resolved twice, the second time at the first's angle (as core.steady_rates)."""
    size, sustain, deep = core.stick_level_shape(c, index)
    control = core.steady_control(c, speeds, torch.full_like(speeds, size), COAST, sustain, deep)
    angle_sin = 0.0
    for _ in range(2):
        lateral, _, vertical, _ = core._setting(c, control, speeds, c.spawn_energy, angle_sin)
        angle = core._commanded_angle(c, control, lateral, speeds, 1.0 / c.lift, vertical)
        angle_sin = torch.sin(angle).abs()
    return angle


def test_max_is_the_envelopes_top_line():
    """The author's design (2026-11-24, Max since 2026-11-26): Max is the most turn the wing
    gives at that speed, with no blending tax: away from the corner (whose rounding, the wing's
    lift against the g limit, is the one soft minimum kept) its lateral acceleration is the
    whole of what is left after 1 g: sqrt(a_max^2 - g^2) above corner, sqrt((C v^2)^2 - g^2)
    below (C the wing's maximum lift coefficient), to 0.1%. And no fraction of it (a sweep of
    stick sizes) turns more than Max at any speed (quasi-steady turn rate)."""
    for ship_type, name in enumerate(card.TYPE_NAMES):
        c = double_constants(ship_type)
        g, limit = card.GRAVITY, float(c.max_acceleration[0])
        corner = float(c.corner_speed[0])
        above = torch.linspace(1.25 * corner, 2.0 * corner, 200, dtype=torch.float64)
        ratio = top_line_lateral(c, above) / (limit * limit - g * g) ** 0.5
        assert float((ratio - 1.0).abs().max()) < 1e-3, (name, float(ratio.min()))
        below = torch.linspace(1.6 * one_g_speed(ship_type), 0.7 * corner, 200, dtype=torch.float64)
        lift = c.lift * below * below
        ratio = top_line_lateral(c, below) / torch.sqrt(lift * lift - g * g)
        assert float((ratio - 1.0).abs().max()) < 1e-3, (name, float(ratio.min()))
        speeds = torch.linspace(
            1.1 * one_g_speed(ship_type), 2.0 * corner, 400, dtype=torch.float64
        )
        top = level_line(c, speeds, MAX)[0]
        for stick in (0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
            sticks = torch.full_like(speeds, stick)
            rate = core.quasi_steady(c, speeds, sticks, COAST, c.spawn_energy)[0]
            assert bool((rate <= top + 1e-12).all()), (name, stick)


def test_the_levels_shapes():
    """The levels (author 2026-11-26; card.LEVEL_NAMES): each stick index's shape from the
    constants' tables (core.stick_level_shape: size, Sustain's table, Snap's deep angle), all
    of its own line; straight is no turn. Snap's deep angle follows GameConfig.post_stall."""
    c = double_constants()
    assert card.LEVEL_NAMES == ("Sustain", "Max", "Snap")
    assert core.stick_level_shape(c, STRAIGHT) == (0.0, 0.0, 0.0)
    assert core.stick_level_shape(c, SUSTAIN) == (1.0, 1.0, 0.0)
    assert core.stick_level_shape(c, MAX) == (1.0, 0.0, 0.0)
    assert core.stick_level_shape(c, SNAP) == (1.0, 0.0, 1.0)
    off = core.per_type_constants(dataclasses.replace(CONFIG, post_stall=False), torch.tensor([0]))
    assert core.stick_level_shape(off, SNAP) == (1.0, 0.0, 0.0)


def test_the_stick_tables_by_index():
    """card.STICK_LEVELS' layout in the stick tables, for every type: the side (odd indices
    right, negative; even left), Sustain's indices (|level| 1), Snap's deep angle (|level| 3)
    and the nose spring's stiffness going out (card.NOSE_STIFFNESS by |level|)."""
    levels = torch.tensor(card.STICK_LEVELS)
    for ship_type, name in enumerate(card.TYPE_NAMES):
        c = core.per_type_constants(CONFIG, torch.tensor([ship_type]))
        assert c.stick_table[0].tolist() == torch.sign(levels).tolist(), name
        assert c.sustain_stick_table[0].tolist() == (levels.abs() == 1.0).float().tolist(), name
        assert c.deep_stall_table[0].tolist() == (levels.abs() == 3.0).float().tolist(), name
        stiffness = [card.NOSE_STIFFNESS[int(abs(level))] for level in card.STICK_LEVELS]
        assert c.nose_stiffness_table[0].tolist() == pytest.approx(stiffness), name
    assert card.STICK_LEVELS[SUSTAIN] == 1.0 and card.STICK_LEVELS[SNAP] == 3.0


def test_snap_is_max_above_corner_and_the_deep_angle_below_the_band():
    """Snap (author 2026-11-26): at and above corner speed it flies Max's line (the same angle
    and turn, to rounding); below the band (deep_stall_band of corner speed) and above the
    speed where the deep angle's lift holds 1 g it asks for the card's deep_stall_angle."""
    for ship_type, name in enumerate(card.TYPE_NAMES):
        c = double_constants(ship_type)
        corner = float(c.corner_speed[0])
        above = torch.linspace(corner, 2.0 * corner, 200, dtype=torch.float64)
        snap, top = level_line(c, above, SNAP)[0], level_line(c, above, MAX)[0]
        assert torch.allclose(snap, top, rtol=1e-9), name
        held_from = float(c.holding_start_speed[0] + 1.0 / c.inverse_holding_span[0])
        band = card.CARD["deep_stall_band"][ship_type]
        below = torch.linspace(1.01 * held_from, (1.0 - band) * corner, 100, dtype=torch.float64)
        deep = card.CARD["deep_stall_angle"][ship_type]
        assert torch.allclose(steady_angle(c, below, SNAP), torch.full_like(below, deep)), name
        assert bool((steady_angle(c, below, MAX) < float(c.stall[0]) + 1e-9).all()), name


def test_sustain_holds_energy_where_a_sustained_turn_exists():
    """Sustain (author 2026-11-26): where Max loses energy, up to cruise speed, Sustain's table
    (core._sustain_table, read linearly) gives a turn whose energy rate is about 0 (within 1% of
    the largest energy Max loses there); where Max gains energy it is Max (to 1%, the table's
    linear reading at the knee); above cruise speed it flies straight."""
    for ship_type, name in enumerate(card.TYPE_NAMES):
        c = double_constants(ship_type)
        stall, cruise = float(c.stall_speed[0]), float(c.cruise_speed[0])
        speeds = torch.linspace(stall, 1.2 * cruise, 400, dtype=torch.float64)
        sustain, top = level_line(c, speeds, SUSTAIN), level_line(c, speeds, MAX)
        loses = (top[2] < 0.0) & (speeds <= cruise)
        assert bool(loses.any()), name
        largest = float(top[2][loses].abs().max())
        assert float(sustain[2][loses].abs().max()) < 0.01 * largest, name
        gains = top[2] >= 0.0
        assert torch.allclose(sustain[0][gains], top[0][gains], rtol=1e-2), name
        assert bool((sustain[0][speeds >= cruise] == 0.0).all()), name


def test_nose_damping_ratio_per_level():
    """The nose spring (author 2026-11-26): going out its stiffness is the level's
    (card.NOSE_STIFFNESS) of the full spring, the friction fixed at the card's nose_damping, so
    the damping ratio is nose_damping / sqrt(stiffness): Sustain critically damped, Max 1/sqrt(2),
    Snap 1/sqrt(3) at the design point. Flown from straight at 0.8 x corner speed, Sustain does
    not overshoot its commanded angle (0.5% slack) and the overshoot grows Sustain < Max < Snap."""
    damping = card.CARD["nose_damping"][P51D]
    ratios = [damping / math.sqrt(card.NOSE_STIFFNESS[level]) for level in (1, 2, 3)]
    assert ratios == pytest.approx([1.0, 1.0 / math.sqrt(2.0), 1.0 / math.sqrt(3.0)], rel=1e-6)
    overshoot = {index: nose_overshoot(index, 0.8 * CORNER) for index in (SUSTAIN, MAX, SNAP)}
    assert overshoot[SUSTAIN] < 0.005
    assert overshoot[SUSTAIN] < overshoot[MAX] < overshoot[SNAP]


def start_state(speed, store_fraction=0.5):
    """One P-51D flying straight along +x, the nose on the path, at `speed`."""
    c = constants()
    z = torch.zeros(1, 1)
    return SimpleNamespace(
        x=z.clone(),
        y=z.clone(),
        heading_cos=torch.ones(1, 1),
        heading_sin=z.clone(),
        speed=torch.full((1, 1), float(speed)),
        angle=z.clone(),
        angle_cos=torch.ones(1, 1),
        angle_sin=z.clone(),
        nose_rate=z.clone(),
        energy=store_fraction * torch.full((1, 1), c["max_energy"]),
        log_index=z.clone(),
        gradient_x=z.clone(),
        gradient_y=z.clone(),
    )


def nose_overshoot(index, speed, seconds=1.5):
    """A level held from straight for `seconds`, coasting: the largest angle of attack over the
    commanded angle at that tick (the steady analysis' own functions at the flown speed, store
    and angle), minus 1."""
    k, p = constants(), core.per_type_constants(CONFIG, torch.tensor([P51D]))
    size, sustain, deep = core.stick_level_shape(p, index)
    s = start_state(speed)
    stick, power = torch.full((1, 1), index), torch.zeros(1, 1, dtype=torch.long)
    peak, at_peak = -1.0, 1.0
    with torch.no_grad():
        for _ in range(round(seconds / CONFIG.tick_length)):
            s = core.flight_tick(k, NO_FIELDS, CONFIG.tick_length, s, stick, power)
            v = s.speed[0]
            control = core.steady_control(
                p, v, torch.tensor([size]), COAST, torch.tensor([sustain]), torch.tensor([deep])
            )
            lateral, _, vertical, _ = core._setting(
                p, control, v, s.energy[0], s.angle_sin[0].abs()
            )
            target = float(core._commanded_angle(p, control, lateral, v, 1.0 / p.lift, vertical))
            if float(s.angle) > peak:
                peak, at_peak = float(s.angle), target
    return peak / at_peak - 1.0


def flown_angles(k, speed, sticks):
    """The angle of attack per tick flying the stick indices `sticks` from straight at `speed`."""
    s = start_state(speed)
    power = torch.zeros(1, 1, dtype=torch.long)
    angles = []
    with torch.no_grad():
        for stick in sticks:
            s = core.flight_tick(
                k, NO_FIELDS, CONFIG.tick_length, s, torch.full((1, 1), stick), power
            )
            angles.append(float(s.angle))
    return angles


def test_the_nose_returns_on_the_full_spring():
    """The return rule (author 2026-11-26): going out the nose spring is the level's
    (card.NOSE_STIFFNESS), going back toward the flight path always the full spring. Against the
    same constants with every level's stiffness 1: Sustain held from straight flies differently
    (going out on its soft spring), while after Snap held 1 s a switch to Sustain on the same
    side (its smaller angle: going back) flies the same as long as the nose comes back."""
    k = constants()
    start, width = k.columns["nose_stiffness_table"]
    table = k.table.clone()
    table[:, start : start + width] = 1.0
    full_spring = type(k)(k.uniform, k.columns, table)
    speed = 0.8 * CORNER
    hold, settle = round(1.0 / CONFIG.tick_length), round(0.3 / CONFIG.tick_length)
    out_level = flown_angles(k, speed, [SUSTAIN] * hold)
    out_full = flown_angles(full_spring, speed, [SUSTAIN] * hold)
    assert max(abs(a - b) for a, b in zip(out_level, out_full)) > math.radians(1.0)
    sticks = [SNAP] * hold + [SUSTAIN] * settle
    back_level = flown_angles(k, speed, sticks)[hold:]
    back_full = flown_angles(full_spring, speed, sticks)[hold:]
    assert all(b < a for a, b in zip(back_level, back_level[1:]))  # coming back all along
    assert back_level == pytest.approx(back_full, abs=1e-6)


def test_holding_snap_does_not_sink_far_below_stall_speed():
    """Snap never gives up the 1 g (author 2026-11-26: the deep angle closes to the stall angle
    toward stall speed, thrust's normal part counting in the hold): held 60 s from the spawn,
    coasting, the ship never falls below 0.95 x stall speed and the store loses under 1% of
    its capacity."""
    c = double_constants()
    speeds, stores = fly(card.SPAWN_SPEED, 0.5, SNAP, COAST, 60.0)
    assert min(speeds) > 0.95 * float(c.stall_speed[0])
    assert 0.5 - min(stores) < 0.01


def first_of(c, name):
    """The first ship's value of a run constant (uniform or per slot)."""
    return float(torch.as_tensor(c[name]).reshape(-1)[0])


def fire_once(speed_over_stall, store_fraction, shoot):
    """One game step of two P-51Ds, the first flying straight at `speed_over_stall` x its stall
    speed with `store_fraction` of its store, its trigger `shoot`: (state after, before)."""
    generator = torch.Generator().manual_seed(0)
    state = core.reset(CONFIG, 1, torch.tensor([[1, 1]]), generator, torch.device("cpu"))
    c = state.constants
    state = state.replace(
        speed=c["stall_speed"] * speed_over_stall, energy=store_fraction * c["max_energy"]
    )
    action = torch.tensor([[[0, 0, shoot], [0, 0, 0]]])
    noise = core.draw_noise(CONFIG, 1, 2, generator, torch.device("cpu"))
    after, _ = core.step(CONFIG, state.replace(pending=action), action, noise)
    return after, state


def test_firing_below_stall_speed_is_paid_from_the_store():
    """Firing never takes a ship below its stall speed (author 2026-11-26): at and below stall
    speed the thrust a shot withholds (1 - fire_thrust_fraction of it) is paid from the store
    (its power over the tick), the speed as if not firing; from (1 + fire_store_band) x stall
    speed up the speed pays and the store nothing."""
    for over_stall in (0.9, 1.0):
        fired, before = fire_once(over_stall, 0.5, 1)
        quiet, _ = fire_once(over_stall, 0.5, 0)
        assert bool(fired.fired[0, 0])
        assert float(fired.speed[0, 0]) == pytest.approx(float(quiet.speed[0, 0]), rel=1e-6)
        c = before.constants
        speed = float(before.speed[0, 0])
        power, static, exponent = (
            first_of(c, name) for name in ("power", "static_thrust", "thrust_softmin_exponent")
        )
        thrust = core.softmin(torch.tensor(power / speed), static, exponent)
        cost = (1.0 - CONFIG.fire_thrust_fraction) * float(thrust) * speed * CONFIG.tick_length
        paid = float(quiet.energy[0, 0] - fired.energy[0, 0])
        assert paid == pytest.approx(cost, rel=1e-3), over_stall
    above = 1.0 + 1.5 * CONFIG.fire_store_band
    fired, _ = fire_once(above, 0.5, 1)
    quiet, _ = fire_once(above, 0.5, 0)
    assert float(fired.speed[0, 0]) < float(quiet.speed[0, 0])
    assert float(fired.energy[0, 0]) == pytest.approx(float(quiet.energy[0, 0]), rel=1e-6)


def test_an_empty_store_blocks_firing_below_the_store_band():
    """With an empty store a ship below (1 + fire_store_band) x stall speed cannot fire (the
    store would pay); above the band it can."""
    blocked, _ = fire_once(1.0 + 0.5 * CONFIG.fire_store_band, 0.0, 1)
    assert not bool(blocked.fired[0, 0])
    free, _ = fire_once(1.0 + 1.5 * CONFIG.fire_store_band, 0.0, 1)
    assert bool(free.fired[0, 0])


def test_the_turn_has_precedence_over_the_climb():
    """The author's order (2026-11-23): 1 g held, then the turn, then the climb gets what is left.
    Slow, Max and climb turns as Max coasting (the climb gains nothing it would take from the
    turn: Max takes all the lift left after 1 g, so the climb's trade is nil, at
    most the 1 g's own soft minimum against the budget below zero); straight at twice the 1 g
    speed, the climb trades at its full rate."""
    for ship_type in range(len(card.TYPE_NAMES)):
        c = constants(ship_type)
        slow = 1.3 * one_g_speed(ship_type)
        energy = torch.full((1, 1), 0.5 * c.max_energy)
        laterals, exchanges = {}, {}
        for name, power, stick, speed in (
            ("coast", COAST, 1.0, slow),
            ("climb", CLIMB, 1.0, slow),
            ("straight", CLIMB, 0.0, 2.0 * slow),
        ):
            at = torch.full((1, 1), speed)
            lateral, exchange, *_ = core._setting(c, control_for(c, stick, power), at, energy, 0.0)
            laterals[name], exchanges[name] = lateral.item(), exchange.item()
        assert abs(laterals["climb"] - laterals["coast"]) < 1e-3 * laterals["coast"], ship_type
        straight = exchanges["straight"]
        assert -1e-4 * straight <= exchanges["climb"] < 0.05 * straight, ship_type
        assert exchanges["straight"] > 0.9 * float(c.commanded_table[0, CLIMB])
