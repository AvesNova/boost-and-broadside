"""The solver's windows and error floors follow each card's own scales (time V_s / g, speed
V_s, acceleration g, the climb's E_max / (m g V_s)): the standard card flies exactly the
windows it always flew, a similar card flies its own and measures what the similarity says,
and a real-scale (P-51-like) store's climb measures."""

import dataclasses
import math

import pytest
import torch

from bnb.contracts import GameConfig
from bnb.game import api
from bnb.game.handling import FLOWN, measure
from bnb.game.handling.manoeuvres import REFERENCE, card_stretch, fly, stretch
from bnb.game.handling.solver import ERROR_FLOOR, error_floor, windows

# A similarity of the standard fighter: lengths x5.82, times so that gravity is 9.81 m/s^2
# (a P-51D's lengths and g).
LENGTH = 5.82
TIME = math.sqrt(LENGTH * 27.0 / 9.81)
SPEED, ACCELERATION = LENGTH / TIME, LENGTH / TIME**2
ENERGY, POWER = SPEED**2, SPEED**2 / TIME
CARD_UNITS = dict(
    max_lift_coefficient=1.0 / LENGTH,
    max_acceleration=ACCELERATION,
    zero_lift_drag=1.0 / LENGTH,
    engine_power=POWER,
    engine_power_full_store=POWER,
    static_thrust=ACCELERATION,
    max_energy=ENERGY,
    corner_nose_frequency=1.0 / TIME,
    muzzle_speed=SPEED,
    bullet_energy_cost=ENERGY,
    fire_rate=1.0 / TIME,
    spawn_energy=ENERGY,
    hit_radius=LENGTH,
)
# The flight constants' units; induced_drag_factor is a length at the bleed exponent 2.
FLIGHT_UNITS = dict(
    gravity=ACCELERATION,
    induced_drag_factor=LENGTH,
    coast_exchange=ACCELERATION,
    climb_exchange=ACCELERATION,
    dive_exchange=ACCELERATION,
    nose_frequency_floor=1.0 / TIME,
    speed_floor=SPEED,
)
HANDLE_UNITS = dict(
    cruise_speed=SPEED,
    climb_time=TIME,
    low_speed_acceleration=ACCELERATION,
    sustained_turn_rate=1.0 / TIME,
    nose_reversal_time=TIME,
    dive_acceleration=ACCELERATION,
    zoom_acceleration=ACCELERATION,
)


def standard_rules(tick_length: float = 1.0 / 30.0):
    rules = api.rules_of(GameConfig(1, 1, 1000.0))
    return dataclasses.replace(rules, tick_length=tick_length)


def scaled_rules(tick_length: float = 1.0 / 30.0):
    rules = standard_rules(tick_length)
    flight = rules.flight
    return dataclasses.replace(
        rules,
        flight=dataclasses.replace(
            flight, **{name: getattr(flight, name) * unit for name, unit in FLIGHT_UNITS.items()}
        ),
    )


def fighter(**units) -> api.ShipTypes:
    """The standard fighter alone, its fields times `units`."""
    card = api.ship_types(dtype=torch.float64)
    return api.ShipTypes(
        **{
            field.name: getattr(card, field.name)[:1] * units.get(field.name, 1.0)
            for field in dataclasses.fields(card)
        }
    )


def test_standard_card_flies_the_reference_windows():
    """The standard card's stretch is 1 (its heavy type sets both scales), so it flies the
    windows in seconds as written."""
    for dtype in (torch.float32, torch.float64):
        scale = card_stretch(standard_rules().flight, api.ship_types(dtype=dtype))
        assert scale.time == pytest.approx(1.0, abs=1e-6)
        assert scale.climb == pytest.approx(1.0, abs=1e-6)


def test_windows_and_floors_follow_the_card():
    """A card whose stall speed is twice the standard heavy's at the same gravity has twice
    its time scale: twice the reversal window, twice the climb window at twice the store
    (its climb scale is E_max / (m g V_s)), and floors in its own units."""
    rules = standard_rules()
    handles = measure(api.ship_types(dtype=torch.float64), rules)
    heavy = 1  # the standard card's slower type
    slow = handles.replace(
        stall_speed=handles.stall_speed * 2.0,
        ceiling=handles.ceiling * 4.0,
    )
    standard_windows, slow_windows = windows(rules, handles), windows(rules, slow)
    assert slow_windows["nose_reversal_time"] == pytest.approx(
        2.0 * standard_windows["nose_reversal_time"]
    )
    assert slow_windows["climb_time"] == pytest.approx(2.0 * standard_windows["climb_time"])
    assert standard_windows == pytest.approx(dict(nose_reversal_time=2.0, climb_time=900.0))
    standard_floor = error_floor(rules, handles)[heavy]
    slow_floor = error_floor(rules, slow)[heavy]
    for i, name in enumerate(FLOWN):
        assert float(standard_floor[i]) == pytest.approx(ERROR_FLOOR[name])
    ratios = dict(
        cruise_speed=2.0,
        climb_time=2.0,
        low_speed_acceleration=1.0,
        sustained_turn_rate=0.5,
        nose_reversal_time=2.0,
        dive_acceleration=1.0,
        zoom_acceleration=1.0,
    )
    for i, name in enumerate(FLOWN):
        assert float(slow_floor[i] / standard_floor[i]) == pytest.approx(ratios[name]), name


def test_stretch_is_the_largest_scale():
    """A batch flies its slowest type's windows."""
    constants = standard_rules().flight
    stall = torch.tensor([REFERENCE.time * 27.0, 3.0 * REFERENCE.time * 27.0])
    store = torch.tensor([1.0, 1.0])
    scale = stretch(constants, stall, store)
    assert scale.time == pytest.approx(3.0)


@pytest.mark.slow
def test_a_similar_card_flies_its_similar_handles():
    """The standard fighter scaled to P-51 lengths and g (lengths x5.82, times x4.00), flown
    at a tick scaled by the same time, is the same discrete flight: every flown handle is the
    standard's in the new units, to rounding. Fixed windows in seconds broke this (the
    settling turn and the fit window then span other numbers of ticks)."""
    standard = measure(fighter(), standard_rules())
    scaled = measure(fighter(**CARD_UNITS), scaled_rules(TIME / 30.0))
    for name, unit in HANDLE_UNITS.items():
        ratio = getattr(scaled, name) / (getattr(standard, name) * unit)
        assert float((ratio - 1.0).abs()) < 1e-9, name


@pytest.mark.slow
def test_a_real_scale_store_climbs():
    """The P-51-like fighter with a 125.3 kJ store: its climb takes longer than the 900 s the
    standard card's window allows, and measures (the window follows the store)."""
    card = dataclasses.replace(
        fighter(**CARD_UNITS), max_energy=torch.tensor([125300.0], dtype=torch.float64)
    )
    climb = fly(scaled_rules(), card).climb_time
    assert torch.isfinite(climb).all()
    assert float(climb) > 900.0
