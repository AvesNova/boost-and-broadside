"""Flow separation, the propeller's thrust and the unfaded boost (`FlightConstants.separation_drag`,
`thrust_softmin_exponent`, `vertical_exchange`; game version 11): the induced drag times
1 + s (C / C_max)^m, the thrust P / sqrt(u^2 + (P / T_0)^2) at exponent 2, and the exchange
as vertical lift demand. Together they give the real-scale ships a best sustained
turn below full stick."""

import dataclasses

import torch

from bnb.contracts import GameConfig
from bnb.game import api
from bnb.game.handling.manoeuvres import batched
from bnb.game.handling.shown import LEVEL_STORE, settled
from bnb.game.physics.flight import airframe_of, separation, setting
from bnb.game.physics.numerics import softmin
from bnb.game.ship_types import type_names

RULES = api.rules_of(GameConfig(1, 1, 7800.0, ship_set="real-scale"))
TYPES = api.ship_types(ship_set="real-scale", dtype=torch.float64)
NAMES = ("Yongma", "Jujak", "Vizia")
KINDS = torch.tensor([type_names("real-scale").index(name) for name in NAMES])


def _flight(**changes):
    return dataclasses.replace(RULES.flight, **changes)


def test_separation_is_one_plus_s_at_the_maximum_lift_and_off_by_default():
    airframe = airframe_of(TYPES, KINDS)
    constants = _flight(separation_drag=1.5, separation_exponent=8.0)
    ratio = torch.tensor([0.0, 0.5, 1.0, 1.2], dtype=torch.float64)[:, None]
    lift_squared = (ratio * airframe.max_lift_coefficient) ** 2
    expected = 1.0 + 1.5 * ratio**8
    torch.testing.assert_close(separation(constants, airframe, lift_squared), expected.expand(4, 3))
    assert separation(api.rules_of(GameConfig(1, 1)).flight, airframe, lift_squared) == 1.0


def test_the_propeller_thrust_is_power_over_the_root_of_speeds_squared():
    assert RULES.flight.thrust_softmin_exponent == 2.0
    power, static = torch.tensor(250.0, dtype=torch.float64), 5.0
    speeds = torch.tensor([1.0, 50.0, 150.0], dtype=torch.float64)
    expected = power / torch.sqrt(speeds**2 + (power / static) ** 2)
    torch.testing.assert_close(softmin(power / speeds, static, 2.0), expected)


def test_the_exchange_is_vertical_lift_demand():
    """A boost spends at its full rate in a turn and asks for about no lift; a reverse
    climbs at its full rate in a moderate turn, with gravity plus the climb's lift."""
    airframe = airframe_of(TYPES, KINDS)
    speed = torch.full((3,), 150.0, dtype=torch.float64)
    energy = 0.25 * airframe.max_energy
    stick = torch.full_like(speed, 0.3)
    for power in (-1.0, 1.0):
        turning = setting(RULES.flight, airframe, speed, energy, stick, torch.full_like(speed, power))
        straight = setting(
            RULES.flight, airframe, speed, energy, 0.0 * stick, torch.full_like(speed, power)
        )
        torch.testing.assert_close(turning.exchange, straight.exchange, rtol=1e-3, atol=0.0)
        expected = (RULES.flight.gravity + turning.exchange).clamp_min(0.0)
        torch.testing.assert_close(turning.vertical_lift, expected, rtol=0.0, atol=0.2)


def test_the_best_sustained_turn_is_below_full_stick():
    """The settled level turn's rate peaks at a partial lift fraction: full stick, held,
    settles slower and at a lower rate. The card's sustained level sits at that peak."""
    types = batched(dataclasses.replace(TYPES))
    sticks = (0.45, 0.6, 0.7, 0.8, 0.9, 1.0)
    turns = settled(RULES, types, sticks, LEVEL_STORE)
    rate = turns.turn_rate[0, KINDS]  # (3, sticks)
    speed = turns.speed[0, KINDS]
    best = rate.argmax(-1)
    assert bool(((best > 0) & (best < len(sticks) - 1)).all()), rate
    peak = rate.gather(-1, best[:, None])[:, 0]
    assert bool((rate[:, -1] < 0.95 * peak).all()), rate
    assert bool((speed[:, -1] < speed.gather(-1, best[:, None])[:, 0]).all()), speed
    sustained = TYPES.sustained_stick[KINDS]
    assert bool(((sustained > 0.5) & (sustained < 0.95)).all()), sustained
