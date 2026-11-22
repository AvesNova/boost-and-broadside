"""The observation's motion is in the world frame (author 2026-11-23, as v2 observes it)."""

import dataclasses
import math

import torch

from lean.game import core

torch.set_num_threads(1)


def test_observed_speed_is_world_speed_inside_a_field():
    """Inside a field the observed speed and velocity are the proper speed over the refractive
    index, so a field that slows a ship shows as a slower ship."""
    field = (0.0, 0.0, 2000.0, 200.0, 2.0)  # a strong field covering the arena's centre
    config = core.GameConfig(fields=(field,))
    generator = torch.Generator().manual_seed(0)
    state = core.reset(config, 1, torch.tensor([[1, 1]]), generator, torch.device("cpu"))
    state = dataclasses.replace(state, x=torch.zeros_like(state.x), y=torch.zeros_like(state.y))
    log_index, gradient_x, gradient_y = core.medium(state.constants, state.x, state.y)
    state = dataclasses.replace(
        state, medium=torch.stack([log_index, gradient_x, gradient_y], -1)
    )  # moved: refresh
    assert float(log_index[0, 0]) > 0.1  # the ship really is inside the field
    observation = core.observe(config, state, 0)
    world_speed = state.speed * torch.exp(-log_index)
    observed_speed = observation.ship[..., 2] * core.SPEED_SCALE
    assert torch.allclose(observed_speed[0, 0], world_speed[0, 0], rtol=1e-5)
    observed_velocity = observation.ship[..., :2] * core.SPEED_SCALE
    assert math.isclose(
        float(observed_velocity[0, 0].norm()), float(world_speed[0, 0]), rel_tol=1e-4
    )
