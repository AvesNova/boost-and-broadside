"""The core flight model's design assumptions (docs/rewrite/core-decisions.md, "Flight physics").

Each test names the rule it guards. They fly one ship with `flight_tick` (or one game step) in
`GameConfig.core()`; no fields.
"""

from types import SimpleNamespace

import torch

from lean.game import card, core

torch.set_num_threads(1)
CONFIG = core.GameConfig.core()
VIZIA = 2
COAST, DIVE, CLIMB = 0, 1, 2
STRAIGHT, FULL = 0, 6


def constants(ship_type=VIZIA):
    return core.ship_constants(CONFIG, torch.full((1, 1), ship_type))


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
        energy=store_fraction * c["max_energy"].clone(),
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
                c, CONFIG.tick_length, s, stick_t, power_t, model="core", thrust_scale=thrust_scale
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
    assert abs(c["max_energy"].item() - expected) < 1e-3 * expected
    assert abs(c["spawn_energy"].item() - 0.5 * expected) < 1e-3 * expected


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
    c = SimpleNamespace(**constants())
    speed = torch.full((1, 1), 250.0)
    energy = 0.5 * c.max_energy
    rates = []
    for power in (DIVE, CLIMB):
        control = SimpleNamespace(
            model="core",
            thrust_scale=1.0,
            stick_size=torch.zeros(1, 1),
            stick_sign=torch.ones(1, 1),
            commanded=c.commanded_table[..., power],
            demand=c.demand_table[..., power],
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
        c = SimpleNamespace(**constants())
        speed = torch.full((1, 1), 200.0)
        control = SimpleNamespace(
            model="core",
            thrust_scale=1.0,
            stick_size=torch.ones(1, 1),
            stick_sign=torch.ones(1, 1),
            commanded=c.commanded_table[..., power],
            demand=c.demand_table[..., power],
        )
        control.inverse_demand = 1.0 / control.demand
        lateral, *_ = core._setting(c, control, speed, store * c.max_energy)
        rates[name] = lateral.item()
    assert abs(rates["climb"] - rates["coast"]) < 1e-3 * abs(rates["coast"])
