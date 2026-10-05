"""One flight model: the field-free tick is the field tick at n = 1.

The field-free path used to apply lift as an explicit velocity kick, which
multiplies speed by ``sqrt(1 + (C_L u dt)^2)`` every tick and so pumps energy
into any turn. These pin the replacement's physical invariants.
"""

from dataclasses import replace

import pytest
import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.constants import PowerActions, TurnActions
from boost_and_broadside.env.field_physics import refresh_ship_field_cache
from boost_and_broadside.env.physics import update_ships
from tests.conftest import make_state

TURNS = (TurnActions.TURN_LEFT, TurnActions.SHARP_RIGHT)


def _actions(power: int, turn: int, ships: int = 1) -> torch.Tensor:
    actions = torch.zeros((1, ships, 3), dtype=torch.float32)  # (B, N, 3)
    actions[..., 0] = power
    actions[..., 1] = turn
    return actions


def _flying_state(config: ShipConfig, speed: float, num_fields: int = 0):
    state = make_state(num_envs=1, max_ships=1, ship_config=config, num_fields=num_fields)
    state.ship_pos[:] = complex(300.0, 400.0)
    state.ship_vel[:] = complex(speed, 0.0)
    return state


def _energy(state, config: ShipConfig) -> float:
    speed = (state.ship_local_index * state.ship_vel.abs())[0, 0]
    return float(0.5 * speed**2 + config.power_speed_constant * state.ship_power[0, 0])


@pytest.mark.parametrize("turn", TURNS)
def test_lift_alone_keeps_speed_constant_without_fields(turn):
    config = replace(
        ShipConfig(),
        base_thrust=0.0,
        no_turn_drag_coeff=0.0,
        normal_turn_drag_coeff=0.0,
        sharp_turn_drag_coeff=0.0,
    )
    state = _flying_state(config, 120.0)
    for _ in range(600):
        state = update_ships(state, _actions(PowerActions.COAST, turn), config)
    assert float(state.ship_vel.abs()[0, 0]) == pytest.approx(120.0, rel=1e-4)


@pytest.mark.parametrize("turn", TURNS)
def test_energy_never_rises_without_thrust_or_regeneration(turn):
    config = replace(ShipConfig(), base_thrust=0.0, passive_power_gain=0.0)
    state = _flying_state(config, 200.0)
    energy = _energy(state, config)
    for _ in range(300):
        state = update_ships(state, _actions(PowerActions.COAST, turn), config)
        next_energy = _energy(state, config)
        assert next_energy <= energy + 1e-3
        energy = next_energy


@pytest.mark.parametrize("turn", TURNS)
def test_turn_with_power_held_full_stays_finite(turn):
    config = ShipConfig()
    state = _flying_state(config, 100.0)
    for _ in range(int(60.0 / config.dt)):
        state = update_ships(state, _actions(PowerActions.BOOST, turn), config)
        state.ship_power = torch.full_like(state.ship_power, config.max_power)
    speed = float(state.ship_vel.abs()[0, 0])
    assert speed < 1.05 * (config.boost_thrust / config.no_turn_drag_coeff) ** 0.5


def test_field_free_tick_matches_the_field_tick_at_unit_index():
    config = ShipConfig()
    without = _flying_state(config, 110.0)
    with_field = _flying_state(config, 110.0, num_fields=1)
    with_field.field_pos[:] = complex(512.0, 512.0)
    with_field.field_radius[:] = 100.0
    with_field.field_transition_width[:] = 40.0
    with_field.field_index[:] = 1.0
    refresh_ship_field_cache(with_field, config)

    for tick in range(120):
        turn = TURNS[(tick // 30) % len(TURNS)]
        actions = _actions(PowerActions.BOOST, turn)
        without = update_ships(without, actions, config)
        with_field = update_ships(with_field, actions, config)

    assert torch.allclose(without.ship_pos, with_field.ship_pos, atol=1e-3)
