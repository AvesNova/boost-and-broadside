"""Continuous refractive-field transport for fast bullets."""

import math
from dataclasses import replace

import pytest
import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.constants import ShootActions
from boost_and_broadside.env.field_physics import (
    evaluate_fields,
    refresh_ship_field_cache,
    wrap_displacement,
)
from boost_and_broadside.env.physics import advance_bullets, update_ships
from tests.conftest import activate_bullet, make_state


def _bullet_config(**overrides) -> ShipConfig:
    return replace(
        ShipConfig(
            bullet_drag_coeff=0.0,
            bullet_lifetime=10.0,
            bullet_spread=0.0,
        ),
        **overrides,
    )


def _single_field_bullet_state(
    config: ShipConfig,
    *,
    index: float,
    radius: float = 100.0,
    width: float = 80.0,
    position: complex = 350.0 + 512.0j,
    velocity: complex = 500.0 + 0.0j,
):
    state = make_state(
        num_envs=1,
        max_ships=1,
        max_bullets=1,
        ship_config=config,
        num_fields=1,
    )
    state.field_pos[:] = 512.0 + 512.0j
    state.field_radius[:] = radius
    state.field_transition_width[:] = width
    state.field_index[:] = index
    state.field_delta_index[:] = index - 1.0
    activate_bullet(state, config, position=position, velocity=velocity)
    _refresh_bullet_cache(state, config)
    return state


def _refresh_bullet_cache(state, config: ShipConfig) -> None:
    batch_size, num_ships, num_bullets = state.bullet_pos.shape
    evaluation = evaluate_fields(
        state.bullet_pos.view(batch_size, num_ships * num_bullets),
        state.field_pos,
        state.field_radius,
        state.field_transition_width,
        state.field_delta_index,
        config.world_size,
    )
    state.bullet_field_alpha = evaluation.alpha.view(
        batch_size,
        num_ships,
        num_bullets,
        state.num_fields,
    )
    state.bullet_local_index = evaluation.index.view(batch_size, num_ships, num_bullets)
    state.bullet_field_gradient = evaluation.grad_index.view(
        batch_size,
        num_ships,
        num_bullets,
    )


@pytest.mark.parametrize("integrator", ["two_step", "midpoint"])
def test_bullet_field_transport_preserves_proper_speed(integrator: str):
    config = _bullet_config(bullet_field_integrator=integrator)
    state = _single_field_bullet_state(
        config,
        index=config.field_index_step**2,
        position=350.0 + 540.0j,
    )
    initial_proper_speed = (state.bullet_local_index * state.bullet_vel.abs()).item()

    for _ in range(48):
        state, _ = advance_bullets(state, config)
        proper_speed = (state.bullet_local_index * state.bullet_vel.abs()).item()
        assert proper_speed == pytest.approx(initial_proper_speed, rel=3e-6)


def test_shooting_initializes_medium_cache_and_proper_muzzle_speed():
    config = _bullet_config()
    state = _single_field_bullet_state(
        config,
        index=config.field_index_step**2,
        position=512.0 + 512.0j,
        velocity=0.0 + 0.0j,
    )
    state.bullet_active[:] = False
    state.ship_pos[:] = 512.0 + 512.0j
    state.ship_vel[:] = 50.0 + 0.0j
    state.ship_attitude[:] = 1.0 + 0.0j
    refresh_ship_field_cache(state, config)
    actions = torch.zeros((1, 1, 3), dtype=torch.long)
    actions[..., 2] = int(ShootActions.SHOOT)

    update_ships(state, actions, config)

    assert state.bullet_active.item()
    assert state.bullet_field_alpha.item() == pytest.approx(1.0)
    assert state.bullet_local_index.item() == pytest.approx(config.field_index_step**2)
    relative_proper_speed = (
        state.bullet_local_index * (state.bullet_vel - state.ship_vel.unsqueeze(-1)).abs()
    )
    assert relative_proper_speed.item() == pytest.approx(config.bullet_speed, rel=1e-6)


@pytest.mark.parametrize("integrator", ["two_step", "midpoint"])
def test_high_index_field_bends_bullet_toward_center(integrator: str):
    config = _bullet_config(bullet_field_integrator=integrator)
    incident_angle = math.radians(25.0)
    velocity = 500.0 * complex(math.cos(incident_angle), math.sin(incident_angle))
    state = _single_field_bullet_state(
        config,
        index=config.field_index_step**2,
        position=350.0 + 512.0j,
        velocity=velocity,
    )

    core_angle = None
    for _ in range(80):
        state, _ = advance_bullets(state, config)
        if state.bullet_field_alpha.item() > 0.999:
            core_angle = abs(torch.angle(state.bullet_vel).item())
            break

    assert core_angle is not None
    assert core_angle < incident_angle


@pytest.mark.parametrize("integrator", ["two_step", "midpoint"])
def test_low_index_high_incidence_bullet_reflects(integrator: str):
    config = _bullet_config(bullet_field_integrator=integrator)
    incident_angle = math.radians(60.0)
    velocity = 500.0 * complex(math.cos(incident_angle), math.sin(incident_angle))
    state = _single_field_bullet_state(
        config,
        index=config.field_index_step**-2,
        radius=300.0,
        width=100.0,
        position=155.0 + 512.0j,
        velocity=velocity,
    )
    max_alpha = 0.0
    inward_speeds = []

    for _ in range(64):
        state, _ = advance_bullets(state, config)
        max_alpha = max(max_alpha, state.bullet_field_alpha.item())
        inward = wrap_displacement(
            state.field_pos[:, 0] - state.bullet_pos[:, 0, 0],
            config.world_size,
        )
        inward = inward / inward.abs()
        inward_speeds.append((state.bullet_vel[:, 0, 0] * torch.conj(inward)).real.item())

    assert 0.0 < max_alpha < 1.0
    assert min(inward_speeds[-20:]) < 0.0


def _trajectory(integrator: str, substeps: int) -> complex:
    config = _bullet_config(
        bullet_field_integrator=integrator,
        bullet_field_integration_substeps=substeps,
    )
    velocity = 500.0 * complex(math.cos(math.radians(28.0)), math.sin(math.radians(28.0)))
    state = _single_field_bullet_state(
        config,
        index=config.field_index_step**2,
        position=350.0 + 500.0j,
        velocity=velocity,
    )
    for _ in range(42):
        state, _ = advance_bullets(state, config)
    return complex(state.bullet_pos.item())


def test_two_step_and_midpoint_track_high_resolution_reference():
    reference = _trajectory("midpoint", 32)
    two_step_error = abs(_trajectory("two_step", 2) - reference)
    midpoint_error = abs(_trajectory("midpoint", 2) - reference)

    assert two_step_error < 8.0
    assert midpoint_error < 2.0
    assert midpoint_error < two_step_error


def _two_damage_field_state(config: ShipConfig):
    state = make_state(
        num_envs=1,
        max_ships=1,
        max_bullets=1,
        ship_config=config,
        num_fields=2,
    )
    state.field_pos[:] = torch.tensor([[300.0 + 512.0j, 700.0 + 512.0j]])
    state.field_radius[:] = 50.0
    state.field_transition_width[:] = 40.0
    # Zero delta-index isolates damage-potential loss from refraction.
    state.field_index[:] = 1.0
    state.field_delta_index[:] = 0.0
    state.field_damage[:] = torch.tensor([[10.0, 20.0]])
    activate_bullet(
        state,
        config,
        position=220.0 + 512.0j,
        velocity=500.0 + 0.0j,
    )
    _refresh_bullet_cache(state, config)
    return state


def _advance_to_field_core(state, config: ShipConfig, field_index: int) -> None:
    for _ in range(20):
        advance_bullets(state, config)
        if state.bullet_field_alpha[0, 0, 0, field_index].item() > 0.999999:
            return
    pytest.fail(f"bullet did not enter field {field_index} core")


@pytest.mark.parametrize("integrator", ["two_step", "midpoint"])
def test_barriers_reduce_bullet_damage_potential_by_ten_percent(integrator: str):
    config = _bullet_config(bullet_field_integrator=integrator)
    state = _two_damage_field_state(config)

    _advance_to_field_core(state, config, 0)
    assert state.bullet_remaining_damage.item() == pytest.approx(9.0, abs=2e-5)

    # Begin a separate crossing just outside the second field. Refreshing the
    # cache models arrival there without charging for the omitted travel.
    state.bullet_pos[:] = 620.0 + 512.0j
    _refresh_bullet_cache(state, config)
    _advance_to_field_core(state, config, 1)
    assert state.bullet_remaining_damage.item() == pytest.approx(7.0, abs=3e-5)


@pytest.mark.parametrize("integrator", ["two_step", "midpoint"])
def test_partial_reflection_charges_total_interface_variation(integrator: str):
    config = _bullet_config(bullet_field_integrator=integrator)
    incident_angle = math.radians(60.0)
    state = _single_field_bullet_state(
        config,
        index=config.field_index_step**-2,
        radius=300.0,
        width=100.0,
        position=155.0 + 512.0j,
        velocity=500.0 * complex(math.cos(incident_angle), math.sin(incident_angle)),
    )
    state.field_damage[:] = 10.0
    max_alpha = 0.0

    for _ in range(64):
        advance_bullets(state, config)
        max_alpha = max(max_alpha, state.bullet_field_alpha.item())

    expected = config.bullet_damage - 2.0 * max_alpha
    assert state.bullet_remaining_damage.item() == pytest.approx(expected, abs=2e-3)


def test_fully_depleted_bullet_deactivates():
    config = _bullet_config(bullet_damage=1.0)
    state = _two_damage_field_state(config)
    state.field_damage[:, 0] = 20.0

    for _ in range(20):
        advance_bullets(state, config)
        if not state.bullet_active.item():
            break

    assert not state.bullet_active.item()
    assert state.bullet_remaining_damage.item() == 0.0


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"bullet_drag_coeff": -1.0}, "bullet_drag_coeff"),
        ({"field_integrator": "bad"}, "field_integrator"),
        ({"bullet_field_integrator": "bad"}, "bullet_field_integrator"),
        ({"bullet_field_integration_substeps": 3}, "positive even"),
        ({"bullet_field_damage_scale": -0.1}, "bullet_field_damage_scale"),
    ],
)
def test_bullet_field_config_validation(overrides: dict, message: str):
    with pytest.raises(ValueError, match=message):
        ShipConfig(**overrides)
