"""The core flight model's design assumptions (docs/rewrite/core-decisions.md, "Flight physics").

Each test names the rule it guards. They fly one ship with `flight_tick` (or one game step) in
`GameConfig.core()`; no fields.
"""

import dataclasses
from types import SimpleNamespace

import torch

from lean.game import card, core

torch.set_num_threads(1)
CONFIG = core.GameConfig.core()
VIZIA = 2
COAST, DIVE, CLIMB = 0, 1, 2
STRAIGHT, FULL = 0, 6
NO_FIELDS = torch.zeros(1, 5, 0)


def constants(ship_type=VIZIA, slots=1):
    """One ship type's constants for (1, slots) slots, in a run of that type alone."""
    config = dataclasses.replace(CONFIG, lineup=(card.TYPE_NAMES[ship_type],))
    return core.ship_constants(config, torch.full((1, slots), ship_type))


def fly(speed, store_fraction, stick, power, seconds, thrust_scale=1.0, ship_type=VIZIA):
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
                c, NO_FIELDS, CONFIG.tick_length, s, stick_t, power_t, model="core", thrust_scale=thrust_scale
            )
            speeds.append(s.speed.item())
            stores.append((s.energy / c["max_energy"]).item())
    return speeds, stores


def one_g_speed(ship_type=VIZIA):
    """The speed below which the wing cannot carry 1 g flying straight."""
    return (card.GRAVITY / card.CARD["max_lift_coefficient"][ship_type]) ** 0.5


def test_store_is_two_loops_at_corner_speed_and_spawn_is_half():
    """The store holds `store_loops` vertical loops at corner speed; ships spawn at half."""
    c = constants()
    lift, limit = card.CARD["max_lift_coefficient"][VIZIA], card.CARD["max_acceleration"][VIZIA]
    corner_radius = (limit / lift) / (limit**2 - card.GRAVITY**2) ** 0.5
    expected = CONFIG.store_loops * card.GRAVITY * 2.0 * corner_radius
    assert abs(c["max_energy"] - expected) < 1e-3 * expected
    assert abs(c["spawn_energy"] - 0.5 * expected) < 1e-3 * expected


def test_coasting_straight_holds_altitude():
    """Coast holds altitude: flying straight above the 1 g speed, the store does not change."""
    _, stores = fly(200.0, 0.5, STRAIGHT, COAST, 30.0)
    assert max(abs(x - 0.5) for x in stores) < 1e-4


def test_coasting_at_full_stick_holds_altitude_above_the_one_g_speed():
    """Altitude wins over the turn: at full stick, coasting, above the 1 g speed, the store holds
    (the turn is limited instead)."""
    speeds, stores = fly(268.0, 0.5, FULL, COAST, 10.0)
    assert min(speeds) > 1.2 * one_g_speed()
    assert max(abs(x - 0.5) for x in stores) < 2e-3


def test_engine_never_raises_the_store():
    """The engine drives speed only: coasting at any stick, the store never rises."""
    for stick in (STRAIGHT, 2, 4, FULL):
        _, stores = fly(150.0, 0.3, stick, COAST, 10.0)
        assert max(stores) <= 0.3 + 1e-6, stick


def test_below_the_one_g_speed_the_ship_sinks():
    """A wing that cannot carry 1 g sinks: store into speed."""
    speeds, stores = fly(0.7 * one_g_speed(), 0.5, STRAIGHT, COAST, 1.0)
    assert stores[-1] < 0.5 - 1e-3
    assert speeds[-1] > 0.7 * one_g_speed()


def test_dive_and_climb_are_equal_and_opposite():
    """Dive and climb trade at the same rate: straight, mid-band, the store's rates are opposite."""
    c = constants()
    speed = torch.full((1, 1), 250.0)
    energy = torch.full((1, 1), 0.5 * c.max_energy)
    rates = []
    for power in (DIVE, CLIMB):
        control = SimpleNamespace(
            model="core",
            thrust_scale=1.0,
            stick_size=torch.zeros(1, 1),
            stick_sign=torch.ones(1, 1),
            commanded=c.commanded_table[:, power],
            demand=c.demand_table[:, power],
        )
        control.inverse_demand = 1.0 / control.demand
        setting = core._setting(c, control, speed, energy)
        _, store_rate = core._rates(
            c,
            setting,
            torch.ones(1, 1),
            speed,
            torch.zeros(1, 1),
            torch.ones(1, 1),
            1.0 / (speed * speed),
        )
        rates.append(store_rate.item())
    assert rates[0] < 0.0 < rates[1]
    assert abs(rates[0] + rates[1]) < 1e-3 * abs(rates[1])


def test_dive_at_an_empty_store_and_climb_at_a_full_store_do_nothing():
    """The band's edges: no dive below empty, no climb above full."""
    _, stores = fly(300.0, 0.0, STRAIGHT, DIVE, 5.0)
    assert max(stores) < 1e-6 and min(stores) >= 0.0
    _, stores = fly(250.0, 1.0, STRAIGHT, CLIMB, 5.0)
    assert max(stores) <= 1.0 + 1e-6


def test_the_edges_taper_smoothly():
    """Smootherstep edges: climbing into the top band, the trade's rate falls continuously to zero
    and never reverses or overshoots the top. (At full climb rate the 5% band is crossed in a few
    ticks with today's numbers: a tuning question, docs/rewrite/core-worklog.md.)"""
    _, stores = fly(300.0, 0.9, STRAIGHT, CLIMB, 4.0)
    steps = [b - a for a, b in zip(stores, stores[1:])]
    assert all(x >= -1e-7 for x in steps)
    inside = [x for x, level in zip(steps, stores[1:]) if level > 1.0 - CONFIG.store_edge_band]
    assert all(b <= a + 1e-7 for a, b in zip(inside, inside[1:]))
    assert max(stores) <= 1.0 + 1e-6


def test_firing_costs_speed_and_no_store():
    """Firing cuts the engine: firing costs speed, and the store pays nothing (one game step per
    tick with the gun held, against the same flight without firing)."""
    speeds_firing, _ = fly(
        300.0, 0.5, STRAIGHT, COAST, 5.0, thrust_scale=CONFIG.fire_thrust_fraction
    )
    speeds_quiet, _ = fly(300.0, 0.5, STRAIGHT, COAST, 5.0)
    assert speeds_firing[-1] < speeds_quiet[-1] - 5.0

    generator = torch.Generator().manual_seed(0)
    state = core.reset(CONFIG, 1, torch.tensor([[1, 1]]), generator, torch.device("cpu"))
    before = state.energy.clone()
    action = torch.tensor([[[0, 0, 1], [0, 0, 0]]])
    noise = core.draw_noise(CONFIG, 1, 2, generator, torch.device("cpu"))
    after, out = core.step(CONFIG, state, action, noise)
    assert bool(out.fired[0, 0])
    assert torch.allclose(after.energy[0, 0], before[0, 0], rtol=1e-6)


def test_at_the_edges_dive_and_climb_act_as_coast_on_the_wing():
    """At the band's edges dive or climb acts as coast: diving on an empty store frees no lift
    for the turn, and climbing at a full store takes none from it."""
    coast_empty, _ = fly(150.0, 0.0, FULL, COAST, 2.0)
    dive_empty, _ = fly(150.0, 0.0, FULL, DIVE, 2.0)
    assert abs(dive_empty[-1] - coast_empty[-1]) < 0.5
    rates = {}
    for name, power, store in (("coast", COAST, 1.0), ("climb", CLIMB, 1.0)):
        c = constants()
        speed = torch.full((1, 1), 200.0)
        control = SimpleNamespace(
            model="core",
            thrust_scale=1.0,
            stick_size=torch.ones(1, 1),
            stick_sign=torch.ones(1, 1),
            commanded=c.commanded_table[:, power],
            demand=c.demand_table[:, power],
        )
        control.inverse_demand = 1.0 / control.demand
        lateral, *_ = core._setting(c, control, speed, torch.full((1, 1), store * c.max_energy))
        rates[name] = lateral.item()
    assert abs(rates["climb"] - rates["coast"]) < 1e-3 * abs(rates["coast"])


def test_speeds_in_order_minimum_radius_then_best_sustained_rate_then_corner():
    """The author's requirement (2026-11-23): slowest to fastest, the minimum sustained radius,
    then the best sustained turn rate, then the corner speed (quasi-steady, coasting, spawn
    store, each ship type)."""
    import numpy as np

    speeds = torch.linspace(100.0, 420.0, 321)
    sticks = torch.linspace(0.0, 1.0, 101)
    for ship_type in range(3):
        c = constants(ship_type, slots=len(speeds) * len(sticks))
        v = speeds.repeat(len(sticks))
        stick = sticks.repeat_interleave(len(speeds))
        control = SimpleNamespace(
            model="core",
            thrust_scale=1.0,
            stick_size=stick,
            stick_sign=torch.ones_like(v),
            commanded=c.commanded_table[0, COAST],
            demand=c.demand_table[0, COAST],
        )
        control.inverse_demand = 1.0 / control.demand
        setting = core._setting(c, control, v, torch.full_like(v, c.spawn_energy))
        lateral, _, _, thrust, _ = setting
        reach = (lateral.abs() / (v * v) * c.inverse_lift).clamp(0.0, 1.0)
        angle = c.stall * (1.0 - torch.sqrt(1.0 - reach))
        lift = core._lift_coefficient(c, angle)
        rate = (
            ((lift * v * v + thrust * torch.sin(angle)) / v)
            .reshape(len(sticks), len(speeds))
            .numpy()
        )
        speed_rate, store_rate = core._rates(
            c, setting, torch.ones_like(v), v, lift, torch.cos(angle), 1.0 / (v * v)
        )
        energy_rate = (v * speed_rate + store_rate).reshape(len(sticks), len(speeds)).numpy()
        sustained = np.where(energy_rate >= 0.0, rate, 0.0).max(0)  # best held rate per speed
        held = sustained > 1e-3
        radius = np.where(held, speeds.numpy() / np.maximum(sustained, 1e-9), np.inf)
        minimum_radius_speed = float(speeds[int(np.argmin(radius))])
        best_rate_speed = float(speeds[int(np.argmax(sustained))])
        corner = float(speeds[int(np.argmax(rate[-1]))])
        assert minimum_radius_speed < best_rate_speed < corner, (
            ship_type,
            minimum_radius_speed,
            best_rate_speed,
            corner,
        )


def test_the_turn_has_precedence_over_the_climb():
    """The author's order (2026-11-23): 1 g held, then the turn, then the climb gets what is left.
    Slow, full stick and climb turns as full stick coasting (the climb gains nothing it would take
    from the turn); straight at twice the 1 g speed, the climb trades at its full rate."""
    for ship_type in (VIZIA, card.TYPE_NAMES.index("p51d")):
        c = constants(ship_type)
        slow = 1.3 * one_g_speed(ship_type)
        energy = torch.full((1, 1), 0.5 * c.max_energy)
        laterals, exchanges = {}, {}
        for name, power, stick, speed in (
            ("coast", COAST, 1.0, slow), ("climb", CLIMB, 1.0, slow), ("straight", CLIMB, 0.0, 2.0 * slow)
        ):
            control = SimpleNamespace(
                model="core",
                thrust_scale=1.0,
                stick_size=torch.full((1, 1), stick),
                stick_sign=torch.ones(1, 1),
                commanded=c.commanded_table[:, power],
                demand=c.demand_table[:, power],
            )
            control.inverse_demand = 1.0 / control.demand
            lateral, exchange, *_ = core._setting(c, control, torch.full((1, 1), speed), energy)
            laterals[name], exchanges[name] = lateral.item(), exchange.item()
        assert abs(laterals["climb"] - laterals["coast"]) < 1e-3 * laterals["coast"], ship_type
        assert 0.0 <= exchanges["climb"] < 0.05 * exchanges["straight"], ship_type
        assert exchanges["straight"] > 0.9 * float(c.commanded_table[0, CLIMB])
