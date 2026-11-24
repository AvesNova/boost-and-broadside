"""The observation's motion is in the world frame (author 2026-11-23, as v2 observes it)."""

import math

import torch

from lean.game import card, core

torch.set_num_threads(1)


def test_observed_speed_is_world_speed_inside_a_field():
    """Inside a field the observed speed and velocity are the proper speed over the refractive
    index, so a field that slows a ship shows as a slower ship."""
    config = core.GameConfig()
    generator = torch.Generator().manual_seed(0)
    state = core.reset(config, 1, torch.tensor([[1, 1]]), generator, torch.device("cpu"))
    # One strong field over the arena's centre: x, y, radius, band, target ln n (2 half-octaves).
    field = torch.tensor([[[0.0], [0.0], [2000.0], [200.0], [2.0 * card.LN_SQRT2]]])
    state = state.replace(fields=field, x=torch.zeros_like(state.x), y=torch.zeros_like(state.y))
    log_index, gradient_x, gradient_y = core.medium(state.fields, state.x, state.y)
    state = state.replace(
        log_index=log_index, gradient_x=gradient_x, gradient_y=gradient_y
    )  # moved: refresh
    assert float(log_index[0, 0]) > 0.1  # the ship really is inside the field
    observation = core.observe(config, state, 0)
    world_speed = state.speed * torch.exp(-log_index)
    unscaled = (
        torch.sign(observation.ship) * torch.expm1(observation.ship.abs()) * core.SPEED_SCALE
    )  # symlog's inverse
    observed_speed = unscaled[..., 2]
    assert torch.allclose(observed_speed[0, 0], world_speed[0, 0], rtol=1e-5)
    observed_velocity = unscaled[..., :2]
    assert math.isclose(
        float(observed_velocity[0, 0].norm()), float(world_speed[0, 0]), rel_tol=1e-4
    )
