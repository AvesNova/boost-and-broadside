"""Per-environment map scale (frontline-redesign-plan.md §9).

An episode plays the reference Frontline map times a scale ``s`` drawn at reset:
zone ring, zones, playable radius and field sizes all scale together, and ship
count, physics, vision and episode length do not. Evaluation always plays s = 1.
"""

import math

import pytest
import torch

from boost_and_broadside.agents.frontline_strategy import frontline_strategy
from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.config import EnvConfig, ShipConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.frontline import (
    FRONTLINE_WORLD_SIZE,
    MAP_SCALE_UNIT_PROBABILITY,
    sample_map_scale,
    toroidal_displacement,
    validate_map_scale_cap,
)
from boost_and_broadside.env.wrapper import _scale_bucket_counts
from tests.env.test_frontline import _frontline

_SHIP = ShipConfig(world_size=FRONTLINE_WORLD_SIZE)


def _env(cap: float, *, num_envs: int = 64, num_fields: int = 4) -> TensorEnv:
    env = TensorEnv(
        num_envs,
        _SHIP,
        EnvConfig(
            num_ships=4,
            num_fields=num_fields,
            max_bullets=0,
            max_episode_steps=600,
            frontline=_frontline(),
        ),
        "cpu",
        map_scale_cap=cap,
    )
    env.reset(seed=3)
    return env


class TestSampling:
    def test_half_the_episodes_play_the_reference_map(self):
        torch.manual_seed(0)
        scale = sample_map_scale(200_000, 4.0, torch.device("cpu"))
        assert (scale == 1.0).float().mean().item() == pytest.approx(
            MAP_SCALE_UNIT_PROBABILITY, abs=0.01
        )
        assert scale.min().item() >= 1.0
        assert scale.max().item() <= 4.0

    def test_the_rest_is_a_truncated_exponential_in_log_scale(self):
        """P(s >= 2 | scaled) = (2^-1.5 - 4^-1.5) / (1 - 4^-1.5) at rate 1.5, cap 4."""
        torch.manual_seed(1)
        scale = sample_map_scale(400_000, 4.0, torch.device("cpu"))
        scaled = scale[scale > 1.0]
        expected = (2.0**-1.5 - 4.0**-1.5) / (1.0 - 4.0**-1.5)
        assert (scaled >= 2.0).float().mean().item() == pytest.approx(expected, abs=0.01)

    def test_a_cap_whose_map_outgrows_the_torus_is_refused(self):
        validate_map_scale_cap(12.0, _frontline(), FRONTLINE_WORLD_SIZE)
        with pytest.raises(ValueError, match="half-period"):
            validate_map_scale_cap(13.0, _frontline(), FRONTLINE_WORLD_SIZE)
        with pytest.raises(ValueError, match="at least 1"):
            validate_map_scale_cap(0.5, _frontline(), FRONTLINE_WORLD_SIZE)

    def test_combat_mode_has_no_map_to_scale(self):
        with pytest.raises(ValueError, match="frontline"):
            TensorEnv(
                1,
                ShipConfig(),
                EnvConfig(num_ships=2, max_bullets=0, max_episode_steps=10),
                "cpu",
                map_scale_cap=2.0,
            )


class TestGeometry:
    def test_every_map_length_follows_the_episodes_scale(self):
        env = _env(4.0)
        state = env.state
        scale = state.map_scale
        assert (scale > 1.0).any() and (scale == 1.0).any()
        frontline = _frontline()
        torch.testing.assert_close(
            state.playable_boundary_radius, frontline.playable_radius * scale
        )
        torch.testing.assert_close(
            state.zone_radius,
            frontline.zone_radius * scale.unsqueeze(1).expand_as(state.zone_radius),
        )
        ring = toroidal_displacement(
            state.zone_pos - state.map_center.unsqueeze(1), FRONTLINE_WORLD_SIZE
        ).abs()
        torch.testing.assert_close(
            ring,
            frontline.zone_ring_radius * scale.unsqueeze(1).expand_as(ring),
            rtol=1e-4,
            atol=0.05,
        )
        lower = (_SHIP.field_radius_min * scale).unsqueeze(1)
        upper = (_SHIP.field_radius_max * scale).unsqueeze(1)
        assert ((state.field_radius >= lower - 1e-3) & (state.field_radius <= upper + 1e-3)).all()

    def test_ships_spawn_inside_their_scaled_spawn_zone(self):
        env = _env(4.0)
        state = env.state
        offset = toroidal_displacement(
            state.ship_pos.unsqueeze(2) - state.zone_pos.unsqueeze(1), FRONTLINE_WORLD_SIZE
        ).abs()  # (B, N, Z)
        nearest = offset.min(-1).values
        assert (nearest <= state.zone_radius[:, :1] * 0.35 + 1e-2).all()

    def test_a_cap_of_one_is_the_reference_map(self):
        env = _env(1.0)
        assert torch.equal(env.state.map_scale, torch.ones_like(env.state.map_scale))
        torch.testing.assert_close(
            env.state.playable_boundary_radius,
            torch.full_like(env.state.playable_boundary_radius, _frontline().playable_radius),
        )

    def test_a_partial_reset_keeps_the_other_episodes_scale(self):
        env = _env(4.0)
        before = env.state.map_scale.clone()
        mask = torch.zeros(env.num_envs, dtype=torch.bool)
        mask[0] = True
        env.reset_envs(mask)
        assert torch.equal(env.state.map_scale[1:], before[1:])


class TestScriptedStrategy:
    def test_zone_preferences_are_invariant_to_the_map_scale(self):
        """A configured support radius scales with the map it describes.

        With no enemy in sight the objective preference is a function of
        distances over support radii only, so scaling the whole map about its
        centre must leave it unchanged.
        """
        env = _env(1.0, num_envs=1, num_fields=0)
        state = env.state
        # Ships spread around the map so separation never binds.
        angles = torch.arange(4, dtype=torch.float32) * (math.pi / 2)
        state.ship_pos = state.map_center + torch.polar(torch.full((1, 4), 900.0), angles)
        config = StochasticAgentConfig()
        hidden = torch.zeros((1, 2, 4), dtype=torch.bool)
        reference = frontline_strategy(state, _SHIP, config, hidden).zone_preference

        scaled = state.clone()
        s = 2.5
        scaled.map_scale = torch.full((1,), s)
        center = state.map_center.unsqueeze(1)

        def about_centre(position: torch.Tensor) -> torch.Tensor:
            offset = toroidal_displacement(position - center, FRONTLINE_WORLD_SIZE)
            return center + s * offset

        scaled.zone_pos = about_centre(state.zone_pos)
        scaled.zone_radius = state.zone_radius * s
        scaled.ship_pos = about_centre(state.ship_pos)
        result = frontline_strategy(scaled, _SHIP, config, hidden).zone_preference
        torch.testing.assert_close(result, reference, rtol=1e-4, atol=1e-5)


def test_ships_in_sight_split_by_map_scale():
    scale = torch.tensor([1.0, 1.5, 3.0])
    enemy = torch.ones((3, 2, 2), dtype=torch.bool)
    visible = torch.zeros_like(enemy)
    visible[1] = True
    counts = [int(c) for c in _scale_bucket_counts(scale, enemy, visible)]
    assert counts == [4, 0, 4, 4, 4, 0]
