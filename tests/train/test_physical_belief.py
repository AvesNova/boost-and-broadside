"""The physical belief representation: its moment layout and its deltas."""

import math

import pytest
import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.train.rl.physical_belief import (
    ANGULAR_VELOCITY_SIGMA,
    ATTITUDE,
    ATTITUDE_SIGMA,
    BELIEF_MOMENT_DIM,
    COOLDOWN_RESIDUAL,
    PHYSICAL_MEAN_DIM,
    PHYSICAL_MEAN_NAMES,
    PHYSICAL_UNCERTAINTY_DIM,
    POSITION_SIGMA,
    POSITION_X,
    SCALAR_SIGMAS,
    UNCERTAINTY_NAMES,
    VELOCITY_COVARIANCE,
    PhysicalNextState,
    physical_mean_deltas,
    physical_means_from_state,
)
from boost_and_broadside.train.rl.physical_deltas import (
    PhysicalShipSnapshot,
    physical_ship_deltas,
)


@pytest.fixture
def spec() -> PhysicalNextState:
    return PhysicalNextState.from_ship_config(ShipConfig())


class TestLayout:
    def test_widths_match_the_decided_semantics(self):
        """Eleven means and fourteen spreads, 25 floats per ship (§8.3)."""
        assert PHYSICAL_MEAN_DIM == 11
        assert PHYSICAL_UNCERTAINTY_DIM == 14
        assert BELIEF_MOMENT_DIM == 25

    def test_every_spread_column_is_used_exactly_once(self):
        columns = [POSITION_SIGMA, ATTITUDE_SIGMA, ANGULAR_VELOCITY_SIGMA]
        columns += list(range(PHYSICAL_UNCERTAINTY_DIM))[VELOCITY_COVARIANCE]
        columns += [column for _, column in SCALAR_SIGMAS]
        columns += list(range(PHYSICAL_UNCERTAINTY_DIM))[COOLDOWN_RESIDUAL]
        assert sorted(columns) == list(range(PHYSICAL_UNCERTAINTY_DIM))

    def test_names_are_ordered_and_distinct(self):
        assert len(set(UNCERTAINTY_NAMES)) == PHYSICAL_UNCERTAINTY_DIM
        assert PHYSICAL_MEAN_NAMES[POSITION_X] == "position_x"
        assert PHYSICAL_MEAN_NAMES[ATTITUDE] == "attitude"
        assert UNCERTAINTY_NAMES[POSITION_SIGMA] == "position_sigma"


class TestDeltas:
    def test_mean_deltas_agree_with_the_calibration_definition(self, spec):
        """One source of truth: the state-space and mean-space deltas must match."""
        torch.manual_seed(0)
        world = spec.world_size

        def snapshot():
            return PhysicalShipSnapshot(
                position=torch.complex(torch.rand(2, 3) * world[0], torch.rand(2, 3) * world[1]),
                velocity=torch.complex(torch.randn(2, 3), torch.randn(2, 3)) * 50.0,
                attitude=torch.exp(1j * (torch.rand(2, 3) * 2 * math.pi - math.pi)),
                angular_velocity=torch.randn(2, 3),
                shield_delay=torch.rand(2, 3) * 5.0,
                health=torch.rand(2, 3) * 100.0,
                power=torch.rand(2, 3) * 100.0,
                cooldown=torch.rand(2, 3) * 0.1,
                local_index=torch.rand(2, 3) + 0.5,
            )

        a, b = snapshot(), snapshot()
        reference = physical_ship_deltas(a, b, world)

        def means(s):
            return torch.stack(
                (
                    s.position.real,
                    s.position.imag,
                    s.velocity.real,
                    s.velocity.imag,
                    torch.angle(s.attitude),
                    s.angular_velocity,
                    s.shield_delay,
                    s.health,
                    s.power,
                    s.cooldown,
                    torch.log(s.local_index),
                ),
                dim=-1,
            )

        measured = physical_mean_deltas(means(a), means(b), world)
        assert torch.allclose(measured, reference, atol=1e-5)

    def test_zero_delta_normalizes_to_exact_zero(self, spec):
        means = torch.randn(4, PHYSICAL_MEAN_DIM)
        labels = spec.labels(means, means)
        assert torch.equal(labels, torch.zeros_like(labels))

    def test_position_delta_takes_the_short_way_round_the_torus(self, spec):
        width, height = spec.world_size
        current = torch.zeros(1, PHYSICAL_MEAN_DIM)
        current[0, POSITION_X] = 2.0
        next_ = torch.zeros(1, PHYSICAL_MEAN_DIM)
        next_[0, POSITION_X] = width - 3.0
        delta = physical_mean_deltas(current, next_, (width, height))
        assert delta[0, POSITION_X].item() == pytest.approx(-5.0)

    def test_means_from_state_are_physical(self):
        from boost_and_broadside.config import EnvConfig
        from boost_and_broadside.env.env import TensorEnv

        env = TensorEnv(
            2,
            ShipConfig(),
            EnvConfig(num_ships=4, max_bullets=0, max_episode_steps=100),
            "cpu",
        )
        env.reset(seed=3)
        means = physical_means_from_state(env.state)
        assert means.shape == (2, 4, PHYSICAL_MEAN_DIM)
        assert torch.allclose(means[..., POSITION_X], env.state.ship_pos.real)
        assert torch.allclose(means[..., 7], env.state.ship_health)
        # Natural log, not the observation's normalized log-index.
        assert torch.allclose(means[..., 10], torch.log(env.state.ship_local_index))
