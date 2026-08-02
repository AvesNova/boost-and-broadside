"""Effective-mass ship transport, energy, power, and interface damage."""

import math
from dataclasses import replace

import pytest
import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.constants import PowerActions, TurnActions
from boost_and_broadside.env.field_physics import refresh_ship_field_cache, wrap_displacement
from boost_and_broadside.env.physics import update_ships
from tests.conftest import make_state


def _passive_config(**overrides) -> ShipConfig:
    config = ShipConfig(
        base_thrust=0.0,
        boost_thrust=0.0,
        reverse_thrust=0.0,
        passive_power_gain=0.0,
        no_turn_drag_coeff=0.0,
        normal_turn_drag_coeff=0.0,
        normal_turn_lift_coeff=0.0,
        sharp_turn_drag_coeff=0.0,
        sharp_turn_lift_coeff=0.0,
        bullet_spread=0.0,
    )
    return replace(config, **overrides)


def _single_field_state(
    config: ShipConfig,
    *,
    index: float,
    damage: float = 0.0,
    radius: float = 100.0,
    width: float = 80.0,
    position: complex = 512.0 + 512.0j,
):
    state = make_state(
        num_envs=1,
        max_ships=1,
        max_bullets=0,
        ship_config=config,
        num_fields=1,
    )
    state.field_pos[:] = position
    state.field_radius[:] = radius
    state.field_transition_width[:] = width
    state.field_index[:] = index
    state.field_delta_index[:] = index - 1.0
    state.field_damage[:] = damage
    state.ship_power[:] = 50.0
    return state


def _energy(state, config: ShipConfig) -> torch.Tensor:
    kinetic = 0.5 * state.ship_local_index.square() * state.ship_vel.abs().square()
    return kinetic + config.power_speed_constant * state.ship_power


def _coast_actions():
    return torch.zeros((1, 1, 3), dtype=torch.long)


@pytest.mark.parametrize("integrator", ["two_step", "midpoint"])
def test_passive_entry_exit_restores_speed_and_preserves_long_run_energy(integrator: str):
    config = _passive_config(field_integrator=integrator)
    state = _single_field_state(config, index=config.field_index_step**2)
    state.ship_pos[:] = 350.0 + 512.0j
    state.ship_vel[:] = 100.0 + 0.0j
    state.ship_attitude[:] = 1.0 + 0.0j
    refresh_ship_field_cache(state, config)
    initial_h = 0.5 * state.ship_local_index.square() * state.ship_vel.abs().square()

    min_speed = float("inf")
    entered_core = False
    for _ in range(800):
        state = update_ships(state, _coast_actions(), config)
        min_speed = min(min_speed, state.ship_vel.abs().item())
        entered_core |= state.ship_field_alpha.item() > 0.999
        if entered_core and state.ship_local_index.item() == pytest.approx(1.0, abs=1e-5):
            break
    else:
        pytest.fail("ship did not complete a field entry and exit")

    final_h = 0.5 * state.ship_local_index.square() * state.ship_vel.abs().square()
    assert min_speed < 100.0 / config.field_index_step
    assert state.ship_local_index.item() == pytest.approx(1.0, abs=1e-5)
    assert state.ship_vel.abs().item() == pytest.approx(100.0, rel=2e-5)
    assert final_h.item() == pytest.approx(initial_h.item(), rel=2e-5)
    assert state.ship_power.item() == pytest.approx(50.0)


def test_repeated_toroidal_crossings_do_not_ratchet_passive_speed():
    config = _passive_config()
    state = _single_field_state(config, index=config.field_index_step**-2, radius=80.0, width=60.0)
    state.ship_pos[:] = 300.0 + 512.0j
    state.ship_vel[:] = 180.0 + 0.0j
    state.ship_attitude[:] = 1.0 + 0.0j
    refresh_ship_field_cache(state, config)
    initial_h = 0.5 * state.ship_local_index.square() * state.ship_vel.abs().square()

    for _ in range(1400):
        state = update_ships(state, _coast_actions(), config)

    final_h = 0.5 * state.ship_local_index.square() * state.ship_vel.abs().square()
    assert final_h.item() == pytest.approx(initial_h.item(), rel=3e-5)


def test_identical_field_trajectories_are_deterministic():
    config = _passive_config()
    original = _single_field_state(config, index=config.field_index_step**-2)
    original.ship_pos[:] = 350.0 + 480.0j
    original.ship_vel[:] = 90.0 + 35.0j
    original.ship_attitude[:] = original.ship_vel / original.ship_vel.abs()
    refresh_ship_field_cache(original, config)
    first = original.clone()
    second = original.clone()
    for _ in range(180):
        first = update_ships(first, _coast_actions(), config)
        second = update_ships(second, _coast_actions(), config)
    assert torch.equal(first.ship_pos, second.ship_pos)
    assert torch.equal(first.ship_vel, second.ship_vel)
    assert torch.equal(first.ship_health, second.ship_health)


def test_higher_index_bends_toward_interface_normal():
    config = _passive_config()
    state = _single_field_state(config, index=config.field_index_step**2, width=100.0)
    incident_angle = math.radians(35.0)
    state.ship_pos[:] = 350.0 + 470.0j
    state.ship_vel[:] = 100.0 * complex(math.cos(incident_angle), math.sin(incident_angle))
    state.ship_attitude[:] = state.ship_vel / state.ship_vel.abs()
    refresh_ship_field_cache(state, config)

    core_angle = None
    for _ in range(180):
        state = update_ships(state, _coast_actions(), config)
        if state.ship_field_alpha.item() > 0.999:
            core_angle = abs(torch.angle(state.ship_vel).item())
            break
    assert core_angle is not None
    assert core_angle < incident_angle


def test_low_index_high_incidence_reflects_smoothly_and_preserves_energy():
    config = _passive_config()
    # A large radius makes the sampled arc locally planar. 60 degrees exceeds
    # asin(n_low/n_ambient), so the inward normal velocity must reverse before
    # the flat core is reached.
    state = _single_field_state(
        config,
        index=config.field_index_step**-2,
        damage=config.field_interface_damage,
        radius=300.0,
        width=100.0,
    )
    incident_angle = math.radians(60.0)
    state.ship_pos[:] = 155.0 + 512.0j
    state.ship_vel[:] = 100.0 * complex(math.cos(incident_angle), math.sin(incident_angle))
    state.ship_attitude[:] = state.ship_vel / state.ship_vel.abs()
    refresh_ship_field_cache(state, config)
    initial_h = 0.5 * state.ship_local_index.square() * state.ship_vel.abs().square()
    max_alpha = 0.0
    inward_speeds = []

    for _ in range(240):
        state = update_ships(state, _coast_actions(), config)
        max_alpha = max(max_alpha, state.ship_field_alpha.item())
        inward = wrap_displacement(state.field_pos[:, 0] - state.ship_pos[:, 0], config.world_size)
        inward = inward / inward.abs()
        inward_speeds.append((state.ship_vel[:, 0] * torch.conj(inward)).real.item())

    final_h = 0.5 * state.ship_local_index.square() * state.ship_vel.abs().square()
    assert 0.0 < max_alpha < 1.0
    assert min(inward_speeds[-80:]) < 0.0
    assert final_h.item() == pytest.approx(initial_h.item(), rel=3e-5)
    assert 100.0 - state.ship_health.item() == pytest.approx(
        2.0 * max_alpha * config.field_interface_damage, rel=2e-3
    )


@pytest.mark.parametrize("speed", [60.0, 140.0])
@pytest.mark.parametrize("width", [30.0, 90.0])
def test_monotonic_interface_damage_is_speed_and_width_invariant(speed: float, width: float):
    config = _passive_config()
    state = _single_field_state(
        config,
        index=config.field_index_step,
        damage=config.field_interface_damage,
        width=width,
    )
    outer_left = 512.0 - 100.0 - width / 2.0
    state.ship_pos[:] = complex(outer_left - 5.0, 512.0)
    state.ship_vel[:] = speed + 0.0j
    state.ship_attitude[:] = 1.0 + 0.0j
    refresh_ship_field_cache(state, config)
    start_health = state.ship_health.item()

    for _ in range(400):
        state = update_ships(state, _coast_actions(), config)
        if state.ship_field_alpha.item() > 0.999999:
            break

    assert state.ship_field_alpha.item() > 0.999999
    assert start_health - state.ship_health.item() == pytest.approx(
        config.field_interface_damage, abs=2e-4
    )


def test_stationary_ship_in_interface_takes_no_repeated_damage():
    config = _passive_config()
    state = _single_field_state(
        config,
        index=config.field_index_step,
        damage=config.field_interface_damage,
    )
    state.ship_pos[:] = 412.0 + 512.0j
    state.ship_vel[:] = complex(1e-6, 0.0)
    state.ship_attitude[:] = 1.0 + 0.0j
    refresh_ship_field_cache(state, config)
    health = state.ship_health.clone()
    for _ in range(20):
        state = update_ships(state, _coast_actions(), config)
    assert torch.allclose(state.ship_health, health, atol=1e-6)


def test_repeated_partial_crossings_accumulate_total_variation_damage():
    config = _passive_config()
    # delta-index zero isolates damage travel from refraction. This is a direct
    # integrator test; construction correctly forbids ambient fields.
    state = _single_field_state(
        config,
        index=1.0,
        damage=config.field_interface_damage,
        width=80.0,
    )
    state.ship_pos[:] = 392.0 + 512.0j
    state.ship_vel[:] = 60.0 + 0.0j
    state.ship_attitude[:] = 1.0 + 0.0j
    refresh_ship_field_cache(state, config)
    previous_alpha = state.ship_field_alpha.item()
    expected_variation = 0.0
    turns = 0
    for _ in range(600):
        state = update_ships(state, _coast_actions(), config)
        alpha = state.ship_field_alpha.item()
        expected_variation += abs(alpha - previous_alpha)
        previous_alpha = alpha
        if state.ship_vel.real.item() > 0.0 and alpha >= 0.75:
            state.ship_vel *= -1.0
            state.ship_attitude *= -1.0
            turns += 1
        elif state.ship_vel.real.item() < 0.0 and alpha <= 0.25:
            state.ship_vel *= -1.0
            state.ship_attitude *= -1.0
            turns += 1
        if turns == 4:
            break
    assert turns == 4
    assert 100.0 - state.ship_health.item() == pytest.approx(
        config.field_interface_damage * expected_variation, abs=3e-4
    )


def test_child_crossing_does_not_reapply_parent_damage():
    config = _passive_config()
    state = make_state(
        num_envs=1,
        max_ships=1,
        max_bullets=0,
        ship_config=config,
        num_fields=2,
    )
    state.field_pos[:] = torch.tensor([[512.0 + 512.0j, 512.0 + 512.0j]])
    state.field_radius[:] = torch.tensor([[180.0, 60.0]])
    state.field_transition_width[:] = torch.tensor([[40.0, 40.0]])
    parent_n = config.field_index_step
    child_n = config.field_index_step**-2
    state.field_index[:] = torch.tensor([[parent_n, child_n]])
    state.field_delta_index[:] = torch.tensor([[parent_n - 1.0, child_n - parent_n]])
    state.field_damage[:] = torch.tensor(
        [[2.0 * config.field_interface_damage, config.field_interface_damage]]
    )
    state.ship_pos[:] = 420.0 + 512.0j  # parent core, outside child's band
    state.ship_vel[:] = 60.0 + 0.0j
    state.ship_attitude[:] = 1.0 + 0.0j
    refresh_ship_field_cache(state, config)
    assert state.ship_field_alpha[0, 0, 0].item() == 1.0

    for _ in range(180):
        state = update_ships(state, _coast_actions(), config)
        if state.ship_field_alpha[0, 0, 1].item() > 0.999999:
            break
    assert state.ship_field_alpha[0, 0, 0].item() == 1.0
    assert 100.0 - state.ship_health.item() == pytest.approx(
        config.field_interface_damage, abs=3e-4
    )


def test_drag_dissipates_generalized_kinetic_energy():
    config = _passive_config(no_turn_drag_coeff=1e-3)
    index = config.field_index_step**-2
    state = _single_field_state(config, index=index, radius=150.0, width=20.0)
    state.ship_pos[:] = 512.0 + 512.0j
    state.ship_vel[:] = (120.0 / index) + 0.0j
    state.ship_attitude[:] = 1.0 + 0.0j
    refresh_ship_field_cache(state, config)
    before = 0.5 * state.ship_local_index.square() * state.ship_vel.abs().square()
    state = update_ships(state, _coast_actions(), config)
    after = 0.5 * state.ship_local_index.square() * state.ship_vel.abs().square()
    assert after.item() < before.item()


def test_low_and_high_media_have_reciprocal_bounded_control_rates():
    config = _passive_config(normal_turn_lift_coeff=15e-3, boost_thrust=80.0)
    results = {}
    for index in (config.field_index_step**-1, config.field_index_step):
        state = _single_field_state(config, index=index, radius=150.0, width=20.0)
        state.ship_pos[:] = 512.0 + 512.0j
        state.ship_vel[:] = (100.0 / index) + 0.0j
        state.ship_attitude[:] = 1.0 + 0.0j
        refresh_ship_field_cache(state, config)
        before_proper = (state.ship_local_index * state.ship_vel.abs()).item()
        actions = _coast_actions()
        actions[..., 0] = PowerActions.BOOST
        actions[..., 1] = TurnActions.TURN_RIGHT
        state = update_ships(state, actions, config)
        results[index] = (
            (state.ship_local_index * state.ship_vel.abs()).item() - before_proper,
            abs(torch.angle(state.ship_vel).item()),
        )
    low, high = results[config.field_index_step**-1], results[config.field_index_step]
    assert 0.0 < high[0] < low[0] < 3.0
    assert 0.0 < high[1] < low[1] < math.radians(5.0)


@pytest.mark.parametrize("index", [1.12**-2, 1.12**2])
def test_stall_threshold_uses_proper_speed(index: float):
    config = _passive_config(min_speed=1.0)
    state = _single_field_state(config, index=index, radius=150.0, width=20.0)
    state.ship_pos[:] = 512.0 + 512.0j
    state.ship_vel[:] = (0.5 / index) + 0.0j
    state.ship_attitude[:] = 1.0 + 0.0j
    refresh_ship_field_cache(state, config)
    actions = _coast_actions()
    actions[..., 1] = TurnActions.TURN_RIGHT
    state = update_ships(state, actions, config)
    assert state.ship_ang_vel.item() == 0.0


@pytest.mark.parametrize("index", [1.12**-2, 1.12**-1, 1.12, 1.12**2])
def test_powered_work_matches_ship_plus_power_energy(index: float):
    config = _passive_config(boost_thrust=80.0)
    state = _single_field_state(config, index=index, radius=150.0, width=20.0)
    state.ship_pos[:] = 512.0 + 512.0j
    state.ship_vel[:] = (100.0 / index) + 0.0j
    state.ship_attitude[:] = 1.0 + 0.0j
    refresh_ship_field_cache(state, config)
    before = _energy(state, config)
    actions = _coast_actions()
    actions[..., 0] = PowerActions.BOOST
    state = update_ships(state, actions, config)
    after = _energy(state, config)
    assert after.item() == pytest.approx(before.item(), abs=2e-3)
    assert state.ship_power.item() < 50.0


def test_reverse_recovers_only_the_generalized_energy_lost():
    config = _passive_config(reverse_thrust=-80.0)
    index = config.field_index_step**2
    state = _single_field_state(config, index=index, radius=150.0, width=20.0)
    state.ship_pos[:] = 512.0 + 512.0j
    state.ship_vel[:] = (100.0 / index) + 0.0j
    state.ship_attitude[:] = 1.0 + 0.0j
    refresh_ship_field_cache(state, config)
    before = _energy(state, config)
    actions = _coast_actions()
    actions[..., 0] = PowerActions.REVERSE
    state = update_ships(state, actions, config)
    after = _energy(state, config)
    assert after.item() == pytest.approx(before.item(), abs=2e-3)
    assert state.ship_power.item() > 50.0
