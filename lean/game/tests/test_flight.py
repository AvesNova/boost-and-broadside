"""The flight model's design assumptions (docs/rewrite/core-decisions.md, "Flight physics").

Each test names the rule it guards. They fly one ship with `flight_tick` (or one game step) in
`GameConfig()`; no fields.
"""

import dataclasses
from types import SimpleNamespace

import torch

from lean.game import card, core

torch.set_num_threads(1)
CONFIG = core.GameConfig()
P51D = card.TYPE_NAMES.index("p51d")
COAST, DIVE, CLIMB = 0, 1, 2
STRAIGHT, FULL = 0, 6
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
            thrust_scale=1.0,
            stick_size=torch.zeros(1, 1),
            stick_sign=torch.ones(1, 1),
            commanded=c.commanded_table[:, power],
            demand=c.demand_table[:, power],
        )
        wing = core._wing(c, speed)
        setting = core._setting(c, control, speed, energy, wing)
        _, store_rate = core._rates(
            c,
            setting,
            torch.ones(1, 1),
            speed,
            torch.zeros(1, 1),
            torch.ones(1, 1),
            1.0 / (speed * speed),
            1.0 / (wing * wing),
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
    after, _ = core.step(CONFIG, state.replace(pending=action), action, noise)  # pending: applied now
    assert bool(after.fired[0, 0])
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
            thrust_scale=1.0,
            stick_size=torch.ones(1, 1),
            stick_sign=torch.ones(1, 1),
            commanded=c.commanded_table[:, power],
            demand=c.demand_table[:, power],
        )
        lateral, *_ = core._setting(
            c, control, speed, torch.full((1, 1), store * c.max_energy), core._wing(c, speed)
        )
        rates[name] = lateral.item()
    assert abs(rates["climb"] - rates["coast"]) < 1e-3 * abs(rates["coast"])


def test_tightest_turn_clearly_slower_than_best_rate_then_corner():
    """The author's requirement (2026-11-23, with a margin 2026-11-24): slowest to fastest, the
    minimum sustained radius at least 10% below the best sustained turn rate's speed, which is
    below the corner speed (the peak full-stick rate), for each ship type (quasi-steady,
    coasting, spawn store: lean.game.tuning.performance, which flies the game's own flight
    functions). So the one-circle (radius) and two-circle (rate) fights ask for different
    speeds."""
    from lean.game import tuning

    for name in card.TYPE_NAMES:
        p = tuning.performance(tuning.row_of(name))
        assert p["min_radius_speed"] <= 0.9 * p["sustained_speed"], (
            name, p["min_radius_speed"], p["sustained_speed"]
        )
        assert p["sustained_speed"] < p["corner_speed_flown"], (
            name, p["sustained_speed"], p["corner_speed_flown"]
        )


def test_flaps_raise_the_low_speed_lift_smoothly():
    """The automatic flaps (core._wing): the maximum lift is the clean one times 1 + flap_lift
    up to flap_full_speed_ratio x the clean stall speed, the clean one from
    flap_clear_speed_ratio x it, never rising with speed between, with no kink (the slope is
    continuous: a smootherstep)."""
    c = constants()
    row = {name: values[P51D] for name, values in card.CARD.items()}
    stall = one_g_speed()
    full, clear = row["flap_full_speed_ratio"] * stall, row["flap_clear_speed_ratio"] * stall
    lift, flapped = row["max_lift_coefficient"], row["max_lift_coefficient"] * (1.0 + row["flap_lift"])
    assert row["flap_lift"] > 0.0
    speeds = torch.linspace(0.8 * full, 1.2 * clear, 2001, dtype=torch.float64)
    wing = core._wing(c, speeds.float()).double()
    assert torch.allclose(wing[speeds <= full], torch.tensor(flapped, dtype=torch.float64), rtol=1e-6)
    assert torch.allclose(wing[speeds >= clear], torch.tensor(lift, dtype=torch.float64), rtol=1e-6)
    assert bool((wing[1:] <= wing[:-1] + 1e-12).all())
    slope = (wing[1:] - wing[:-1]) / (speeds[1:] - speeds[:-1])
    assert float((slope[1:] - slope[:-1]).abs().max()) < 0.01 * float(slope.abs().max())


def test_flaps_cost_drag_only_for_lift_beyond_the_clean_wing():
    """The flaps' drag is on the lift beyond the clean maximum: straight flight slow (1 g on the
    clean wing) pays none; full stick slow pays it (with the flaps' drag removed, the same turn
    bleeds less)."""
    c = core.per_type_constants(CONFIG, torch.tensor([P51D]))
    free = SimpleNamespace(**{**vars(c), "flap_drag": torch.zeros(1)})
    speed = torch.full((1,), 1.25 * one_g_speed())

    def speed_rate(constants_, stick):
        stick = torch.full((1,), stick)
        return float(core.quasi_steady(constants_, speed, stick, COAST, c.spawn_energy)[1])

    assert speed_rate(c, 0.0) == speed_rate(free, 0.0)
    assert speed_rate(c, 1.0) < speed_rate(free, 1.0)


def test_the_turn_has_precedence_over_the_climb():
    """The author's order (2026-11-23): 1 g held, then the turn, then the climb gets what is left.
    Slow, full stick and climb turns as full stick coasting (the climb gains nothing it would take
    from the turn); straight at twice the 1 g speed, the climb trades at its full rate."""
    for ship_type in range(len(card.TYPE_NAMES)):
        c = constants(ship_type)
        slow = 1.3 * one_g_speed(ship_type)
        energy = torch.full((1, 1), 0.5 * c.max_energy)
        laterals, exchanges = {}, {}
        for name, power, stick, speed in (
            ("coast", COAST, 1.0, slow), ("climb", CLIMB, 1.0, slow), ("straight", CLIMB, 0.0, 2.0 * slow)
        ):
            control = SimpleNamespace(
                thrust_scale=1.0,
                stick_size=torch.full((1, 1), stick),
                stick_sign=torch.ones(1, 1),
                commanded=c.commanded_table[:, power],
                demand=c.demand_table[:, power],
            )
            at = torch.full((1, 1), speed)
            lateral, exchange, *_ = core._setting(c, control, at, energy, core._wing(c, at))
            laterals[name], exchanges[name] = lateral.item(), exchange.item()
        assert abs(laterals["climb"] - laterals["coast"]) < 1e-3 * laterals["coast"], ship_type
        assert 0.0 <= exchanges["climb"] < 0.05 * exchanges["straight"], ship_type
        assert exchanges["straight"] > 0.9 * float(c.commanded_table[0, CLIMB])
