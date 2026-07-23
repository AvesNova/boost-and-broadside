"""Unit tests for obstacle physics.

Covers step_obstacles_harmonic's tensor-in/tensor-out signature (AUDIT-006):
it must operate on plain obstacle tensors without requiring a full TensorState.
"""

import pytest
import torch

from boost_and_broadside.config import EnvConfig, ShipConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.obstacle_physics import step_obstacles_harmonic


@pytest.fixture
def cfg() -> ShipConfig:
    return ShipConfig(obstacle_gravity_harmonic=0.2)


class TestStepObstaclesHarmonicSignature:
    def test_takes_plain_tensors_and_returns_pos_vel_tuple(self, cfg):
        """The function must accept individual obstacle tensors, not a TensorState."""
        pos = torch.tensor([[100.0 + 100.0j]], dtype=torch.complex64)
        vel = torch.tensor([[0.0 + 0j]], dtype=torch.complex64)
        radius = torch.tensor([[5.0]], dtype=torch.float32)
        gcenter = torch.tensor([[100.0 + 100.0j]], dtype=torch.complex64)

        result = step_obstacles_harmonic(pos, vel, radius, gcenter, cfg, enable_pbd=False)

        assert isinstance(result, tuple) and len(result) == 2
        new_pos, new_vel = result
        assert new_pos.shape == pos.shape
        assert new_vel.shape == vel.shape

    def test_no_extra_tensors_beyond_pos_and_vel(self, cfg):
        """Only (pos, vel) are returned — radius/gcenter are read-only inputs."""
        pos = torch.tensor([[100.0 + 100.0j]], dtype=torch.complex64)
        vel = torch.tensor([[0.0 + 0j]], dtype=torch.complex64)
        radius = torch.tensor([[5.0]], dtype=torch.float32)
        gcenter = torch.tensor([[100.0 + 100.0j]], dtype=torch.complex64)
        radius_before = radius.clone()
        gcenter_before = gcenter.clone()

        step_obstacles_harmonic(pos, vel, radius, gcenter, cfg, enable_pbd=False)

        assert torch.equal(radius, radius_before)
        assert torch.equal(gcenter, gcenter_before)


class TestHarmonicGravityPhysics:
    def test_obstacle_accelerates_toward_gravity_center(self, cfg):
        """An obstacle offset from its gravity center must gain velocity toward it."""
        gcenter = torch.tensor([[0.0 + 0j]], dtype=torch.complex64)
        pos = torch.tensor([[50.0 + 0j]], dtype=torch.complex64)  # offset east of gc
        vel = torch.tensor([[0.0 + 0j]], dtype=torch.complex64)
        radius = torch.tensor([[5.0]], dtype=torch.float32)

        new_pos, new_vel = step_obstacles_harmonic(pos, vel, radius, gcenter, cfg, enable_pbd=False)

        # Force points from pos toward gcenter (west, i.e. negative real direction)
        assert new_vel[0, 0].real < 0
        assert abs(new_vel[0, 0].imag) < 1e-4

    def test_obstacle_at_gravity_center_feels_no_force(self, cfg):
        """An obstacle exactly at its gravity center should not accelerate."""
        gcenter = torch.tensor([[100.0 + 100.0j]], dtype=torch.complex64)
        pos = torch.tensor([[100.0 + 100.0j]], dtype=torch.complex64)
        vel = torch.tensor([[0.0 + 0j]], dtype=torch.complex64)
        radius = torch.tensor([[5.0]], dtype=torch.float32)

        _, new_vel = step_obstacles_harmonic(pos, vel, radius, gcenter, cfg, enable_pbd=False)

        assert abs(new_vel[0, 0]) < 1e-5


class TestEnvStepMovesObstacles:
    def test_obstacle_position_changes_after_env_step(self):
        """TensorEnv.step()'s call site must still advance obstacles under gravity."""
        ship_cfg = ShipConfig(obstacle_gravity_harmonic=0.2)
        env_cfg = EnvConfig(num_ships=2, max_bullets=0, max_episode_steps=100, num_obstacles=1)
        env = TensorEnv(num_envs=1, ship_config=ship_cfg, env_config=env_cfg, device="cpu")
        env.reset()

        # No obstacle cache is configured, so obstacles start at (0, 0) — offset
        # them from their gravity center to give harmonic gravity something to do.
        env.state.obstacle_pos[0, 0] = 50.0 + 0j
        env.state.obstacle_vel[0, 0] = 0.0 + 0j
        env.state.obstacle_gcenter[0, 0] = 0.0 + 0j
        env.state.obstacle_radius[0, 0] = 5.0
        pos_before = env.state.obstacle_pos.clone()

        actions = torch.zeros((1, env_cfg.num_ships, 3), dtype=torch.float32)
        env.step(actions)

        assert not torch.equal(env.state.obstacle_pos, pos_before)
