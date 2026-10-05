"""The continuous nose: slip state, lift and drag curves, the g-limit, the stall.

Runs the live Frontline configuration (30 Hz) so the transient bounds are the
ones the flight model was designed to: onset to 90% in 0.13-0.20 s.
"""

import math
from dataclasses import replace

import pytest
import torch

from boost_and_broadside.config.defaults import SHIP_CONFIG
from boost_and_broadside.constants import PowerActions, TurnActions
from boost_and_broadside.env.field_physics import refresh_ship_field_cache
from boost_and_broadside.env.frontline import frontline_ship_config
from boost_and_broadside.env.physics import slip_command, slip_lift_drag, update_ships
from tests.conftest import make_state

CONFIG = replace(frontline_ship_config(SHIP_CONFIG), world_size=(65536.0, 65536.0))
CENTRE = complex(30000.0, 30000.0)


def _state(speed: float, index: float = 1.0):
    """One ship flying +x at a proper speed, inside a uniform field of ``index``."""
    state = make_state(num_envs=1, max_ships=1, ship_config=CONFIG, num_fields=1)
    state.field_pos[:] = CENTRE
    state.field_radius[:] = 20000.0
    state.field_transition_width[:] = 40.0
    state.field_index[:] = index
    state.ship_pos[:] = CENTRE
    state.ship_vel[:] = complex(speed / index, 0.0)
    refresh_ship_field_cache(state, CONFIG)
    return state


def _step(state, power: int, turn: int):
    actions = torch.tensor([[[power, turn, 0]]], dtype=torch.float32)  # (B, N, 3)
    state = update_ships(state, actions, CONFIG)
    state.ship_pos = torch.full_like(state.ship_pos, CENTRE)
    return state


def _slip_deg(state) -> float:
    return math.degrees(state.ship_slip.item())


def _ticks_until(state, power, turn, predicate, limit=120) -> int:
    for tick in range(1, limit + 1):
        state = _step(state, power, turn)
        if predicate(state):
            return tick
    raise AssertionError("condition never reached")


def test_onset_reaches_ninety_percent_of_a_sharp_pull_in_the_design_window():
    command = math.degrees(CONFIG.stall_angle)
    ticks = _ticks_until(
        _state(60.0),
        PowerActions.COAST,
        TurnActions.SHARP_RIGHT,
        lambda s: _slip_deg(s) >= 0.9 * command,
    )
    assert 0.13 <= ticks * CONFIG.dt <= 0.20


def test_release_snaps_the_nose_back_to_the_path_in_the_design_window():
    state = _state(60.0)
    for _ in range(30):
        state = _step(state, PowerActions.COAST, TurnActions.SHARP_RIGHT)
    ticks = _ticks_until(
        state, PowerActions.COAST, TurnActions.GO_STRAIGHT, lambda s: abs(_slip_deg(s)) < 1.0
    )
    assert 0.13 <= ticks * CONFIG.dt <= 0.20


def test_a_full_reversal_is_slowed_by_the_slip_rate_limit():
    state = _state(60.0)
    for _ in range(30):
        state = _step(state, PowerActions.COAST, TurnActions.SHARP_RIGHT)
    rates = []
    for _ in range(10):
        state = _step(state, PowerActions.COAST, TurnActions.SHARP_LEFT)
        rates.append(abs(state.ship_slip_rate.item()))
    assert max(rates) == pytest.approx(CONFIG.max_slip_rate, rel=1e-5)


@pytest.mark.parametrize("speed", [20.0, 60.0, 100.0, 160.0, 300.0])
@pytest.mark.parametrize("turn", [TurnActions.TURN_LEFT, TurnActions.SHARP_RIGHT])
def test_a_held_turn_never_jitters_or_passes_the_stall_angle(speed, turn):
    """Coasting, the command only ever grows as speed settles, so rate keeps one sign."""
    state = _state(speed)
    rates = []
    for _ in range(90):
        state = _step(state, PowerActions.COAST, turn)
        assert abs(state.ship_slip.item()) <= CONFIG.stall_angle + 1e-6
        rates.append(state.ship_slip_rate.item())
    signs = {math.copysign(1.0, r) for r in rates if abs(r) > 1e-6}
    assert len(signs) <= 1


@pytest.mark.parametrize("speed", [100.0, 140.0, 215.0, 300.0])
def test_lateral_acceleration_respects_the_g_limit_above_corner(speed):
    state = _state(speed)
    for _ in range(20):
        state = _step(state, PowerActions.COAST, TurnActions.SHARP_LEFT)
    velocity = state.ship_vel[0, 0]
    before = torch.angle(velocity).item()
    state = _step(state, PowerActions.COAST, TurnActions.SHARP_LEFT)
    turn_rate = abs(torch.angle(state.ship_vel[0, 0]).item() - before) / CONFIG.dt
    lateral = turn_rate * state.ship_vel.abs().item()
    # Coasting thrust along a slipped nose adds about base_thrust * sin(stall).
    allowance = CONFIG.base_thrust * math.sin(CONFIG.stall_angle)
    assert lateral <= CONFIG.max_lateral_accel + allowance + 1.0


def test_slip_is_attitude_minus_velocity_heading_after_every_tick():
    state = _state(80.0)
    for tick in range(60):
        turn = TurnActions.SHARP_LEFT if (tick // 15) % 2 else TurnActions.TURN_RIGHT
        state = _step(state, PowerActions.BOOST, turn)
        observed = torch.angle(state.ship_attitude * torch.conj(state.ship_vel)).item()
        assert observed == pytest.approx(state.ship_slip.item(), abs=1e-5)


@pytest.mark.parametrize("index", [0.5, 2.0])
def test_nose_times_scale_with_the_field_index(index):
    """Inertia n^2 with torques on proper quantities: every nose time scales by n."""
    command = math.degrees(CONFIG.stall_angle)

    def onset(n: float) -> int:
        return _ticks_until(
            _state(60.0, n),
            PowerActions.COAST,
            TurnActions.SHARP_RIGHT,
            lambda s: _slip_deg(s) >= 0.9 * command,
        )

    assert onset(index) == pytest.approx(index * onset(1.0), abs=1.0)


def test_a_stalled_ship_holds_its_attitude():
    state = _state(0.5)
    state.ship_attitude[:] = complex(0.0, 1.0)
    state = _step(state, PowerActions.COAST, TurnActions.SHARP_RIGHT)
    assert torch.allclose(state.ship_attitude, torch.tensor([[0.0 + 1.0j]]))
    assert state.ship_ang_vel.item() == 0.0


def test_a_stall_rederives_the_slip_from_the_held_attitude():
    state = _state(0.5)
    state.ship_attitude[:] = complex(0.0, 1.0)
    state = _step(state, PowerActions.COAST, TurnActions.GO_STRAIGHT)
    held = torch.angle(state.ship_attitude * torch.conj(state.ship_vel)).item()
    assert state.ship_slip.item() == pytest.approx(held, abs=1e-5)


def test_the_slip_command_fades_below_the_fade_speed():
    side = torch.ones(1)
    sharp = torch.ones(1, dtype=torch.bool)
    half = slip_command(side, sharp, torch.tensor([0.5 * CONFIG.slip_fade_speed]), CONFIG)
    full = slip_command(side, sharp, torch.tensor([CONFIG.slip_fade_speed]), CONFIG)
    assert half.item() == pytest.approx(0.5 * full.item(), rel=1e-6)


def test_the_command_reaches_the_stall_angle_exactly_at_corner():
    side = torch.ones(1)
    sharp = torch.ones(1, dtype=torch.bool)
    command = slip_command(side, sharp, torch.tensor([CONFIG.corner_speed]), CONFIG)
    assert command.item() == pytest.approx(CONFIG.stall_angle, rel=1e-3)


def test_lift_and_drag_reproduce_the_configured_points():
    slips = torch.tensor(
        [0.0, CONFIG.normal_slip_fraction * CONFIG.stall_angle, CONFIG.stall_angle]
    )
    lift, drag = slip_lift_drag(slips, CONFIG)
    expected_lift = [0.0, CONFIG.normal_lift_coeff, CONFIG.max_lift_coeff]
    expected_drag = [
        CONFIG.zero_slip_drag_coeff,
        CONFIG.normal_slip_drag_coeff,
        CONFIG.stall_drag_coeff,
    ]
    assert torch.allclose(lift, torch.tensor(expected_lift), rtol=1e-5)
    assert torch.allclose(drag, torch.tensor(expected_drag), rtol=1e-5)


def test_lift_falls_to_zero_past_twice_the_stall_angle():
    lift, _ = slip_lift_drag(torch.tensor([2.5 * CONFIG.stall_angle]), CONFIG)
    assert lift.item() == 0.0
