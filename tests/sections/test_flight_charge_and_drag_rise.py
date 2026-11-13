"""The direct charge and the drag rise (`FlightConstants.direct_charge`, `critical_speed`):
straight at cruise the direct charge flies as the coast exchange does, below cruise the speed
sees only the null power, and the drag rise multiplies the zero-lift drag by
1 + (u / u_c)^q."""

import dataclasses

import pytest
import torch

from bnb.contracts import GameConfig
from bnb.game import api
from bnb.game.physics.flight import airframe_of, rates, setting, zero_lift_drag
from bnb.game.ship_types import type_names

RULES = api.rules_of(GameConfig(1, 1, 7800.0, ship_set="real-scale"))
# Yongma, its static thrust out of the way (the thrust is then the power's alone)
TYPES = dataclasses.replace(
    api.ship_types(ship_set="real-scale", dtype=torch.float64),
    static_thrust=torch.full((len(type_names("real-scale")),), 1e9, dtype=torch.float64),
)
KIND = torch.tensor([type_names("real-scale").index("Yongma")])


def _flight(**changes):
    """The real-scale flight constants with version 11's switches off (they are tested in
    test_flight_separation_and_propeller.py), and these changes."""
    v11_off = dict(separation_drag=0.0, thrust_softmin_exponent=4.0, vertical_exchange=False)
    return dataclasses.replace(RULES.flight, **{**v11_off, **changes})


def _rates(constants, speed, energy, power=0.0):
    """(speed rate, energy rate) of a Yongma flying straight, at n = 1."""
    airframe = airframe_of(TYPES, KIND)
    speed = torch.tensor([speed], dtype=torch.float64)
    energy = torch.tensor([energy], dtype=torch.float64)
    current = setting(
        constants, airframe, speed, energy, torch.zeros_like(speed), torch.full_like(speed, power)
    )
    zero = torch.zeros_like(speed)
    return rates(constants, airframe, current, 1.0, speed, zero, zero + 1.0, 1.0 / speed**2)


def test_the_direct_charge_flies_as_the_coast_exchange_at_cruise():
    airframe = airframe_of(TYPES, KIND)
    charge = float(airframe.engine_power - airframe.engine_power_full_store)
    cruise = charge / (RULES.flight.coast_exchange * float(airframe.coast_exchange_scale))
    old = _rates(_flight(direct_charge=False, critical_speed=0.0), cruise, 0.0)
    new = _rates(_flight(direct_charge=True, critical_speed=0.0), cruise, 0.0)
    for old_rate, new_rate in zip(old, new):
        torch.testing.assert_close(new_rate, old_rate)


def test_below_cruise_the_direct_charge_leaves_the_speed_only_the_null_power():
    airframe = airframe_of(TYPES, KIND)
    constants = _flight(direct_charge=True, critical_speed=0.0)
    speed = 80.0
    speed_rate, energy_rate = _rates(constants, speed, 0.0)
    torch.testing.assert_close(
        energy_rate, airframe.engine_power - airframe.engine_power_full_store
    )
    full_store_rate, _ = _rates(constants, speed, float(airframe.max_energy))
    torch.testing.assert_close(speed_rate, full_store_rate)  # the store does not change it


def test_the_drag_rise_doubles_the_zero_lift_drag_at_the_critical_speed():
    airframe = airframe_of(TYPES, KIND)
    constants = _flight(critical_speed=200.0, drag_rise_exponent=12.0)
    speeds = torch.tensor([0.0, 100.0, 200.0, 240.0], dtype=torch.float64)
    expected = airframe.zero_lift_drag * (1.0 + (speeds / 200.0) ** 12)
    torch.testing.assert_close(zero_lift_drag(constants, airframe, speeds), expected)
    off = _flight(critical_speed=0.0)
    torch.testing.assert_close(
        zero_lift_drag(off, airframe, speeds), airframe.zero_lift_drag.expand_as(expected)
    )


def test_the_stick_takes_each_types_own_fine_and_sustained_levels():
    from bnb.game.physics.flight import controls

    types = dataclasses.replace(
        TYPES,
        fine_stick=torch.full_like(TYPES.fine_stick, 0.2),
        sustained_stick=torch.full_like(TYPES.sustained_stick, 0.65),
    )
    airframe = airframe_of(types, KIND.expand(7))
    stick, power = controls(airframe, torch.zeros(7, dtype=torch.int32), torch.arange(7))
    assert stick.tolist() == pytest.approx([0.0, -0.2, 0.2, -0.65, 0.65, -1.0, 1.0])
    assert power.tolist() == [0.0] * 7
