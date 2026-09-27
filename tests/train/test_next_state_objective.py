"""What the physical next-state model's uncertainty output buys.

The label steps from the believed state to the true next one, so its spread is
set by how wrong the belief currently is -- which depends on the policy's own
head, on how long ships stay unseen, and so on how well the policy plays.
Measured over one run, velocity labels sat about 33x their calibrated width and
position's scale fell by a third *within* the run. No constant fits that. A
Gaussian likelihood does not need one: the mean's gradient is weighted by
``1/sigma^2``, so a token whose label is mostly unpredictable belief error earns
a wide spread instead of dominating the sum.
"""

import math

import pytest
import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.models.yemong.policy import NextStateHead
from boost_and_broadside.train.rl.physical_belief import (
    CORRELATION_COLUMNS,
    HEALTH,
    LOG_SIGMA_COLUMNS,
    LOG_SIGMA_MAX,
    LOG_SIGMA_MIN,
    NEXT_STATE_OUTPUT_DIM,
    PHYSICAL_MEAN_DIM,
    POSITION_RHO,
    POSITION_X,
    POSITION_Y,
    VELOCITY_X,
    PhysicalNextState,
)


@pytest.fixture
def spec() -> PhysicalNextState:
    return PhysicalNextState.from_ship_config(ShipConfig())


class TestHead:
    def test_the_head_clamps_its_spreads_and_leaves_the_means_alone(self):
        """The likelihood is unbounded below as sigma falls; nothing else bounds it."""
        head = NextStateHead(8)
        with torch.no_grad():
            head.net[-1].bias.fill_(1e4)
            high = head(torch.randn(4, 8))
            head.net[-1].bias.fill_(-1e4)
            low = head(torch.randn(4, 8))

        P = PHYSICAL_MEAN_DIM
        assert high[..., P:][:, list(LOG_SIGMA_COLUMNS)].max() <= LOG_SIGMA_MAX + 1e-5
        assert low[..., P:][:, list(LOG_SIGMA_COLUMNS)].min() >= LOG_SIGMA_MIN - 1e-5
        assert high[..., :P].max() > LOG_SIGMA_MAX, "means are deliberately not clamped"

    def test_the_correlation_latents_stay_strictly_inside_the_unit_interval(self, spec):
        """``tanh`` of a large latent is exactly 1.0 in float32, which is a pole."""
        head = NextStateHead(8)
        with torch.no_grad():
            head.net[-1].bias.fill_(1e4)
            out = head(torch.randn(4, 8))
        latents = out[..., PHYSICAL_MEAN_DIM:][:, list(CORRELATION_COLUMNS)]
        rho = torch.tanh(latents)
        assert (rho.abs() < 1.0).all()
        assert (1.0 - rho * rho > 0.0).all()
        assert torch.isfinite(spec.loss(out, torch.zeros(4, PHYSICAL_MEAN_DIM))).all()

    def test_the_clamp_bounds_are_not_saved_in_a_checkpoint(self):
        """They are a property of the code, not of trained weights."""
        head = NextStateHead(8)
        assert "uncertainty_min" not in head.state_dict()
        assert "uncertainty_max" not in head.state_dict()

    def test_the_head_is_exactly_as_wide_as_the_physical_layout(self):
        head = NextStateHead(8)
        assert head(torch.randn(3, 8)).shape[-1] == NEXT_STATE_OUTPUT_DIM


class TestWhatTheLikelihoodBuys:
    def test_a_wide_spread_costs_precision_but_survives_a_huge_residual(self, spec):
        """Squared error on a 33x label would swamp every other channel."""
        labels = torch.zeros(1, PHYSICAL_MEAN_DIM)
        labels[0, VELOCITY_X] = 33.0

        tight = torch.zeros(1, NEXT_STATE_OUTPUT_DIM)
        loose = torch.zeros(1, NEXT_STATE_OUTPUT_DIM)
        loose[0, PHYSICAL_MEAN_DIM + 3] = math.log(33.0)  # velocity log sigma x

        assert spec.loss(loose, labels)[0, VELOCITY_X] < spec.loss(tight, labels)[0, VELOCITY_X]
        # And the optimum is the residual's own magnitude, not the floor.
        best = min(
            (
                spec.loss(
                    torch.zeros(1, NEXT_STATE_OUTPUT_DIM).index_put_(
                        (torch.tensor([0]), torch.tensor([PHYSICAL_MEAN_DIM + 3])),
                        torch.tensor([float(candidate)]),
                    ),
                    labels,
                )[0, VELOCITY_X].item(),
                candidate,
            )
            for candidate in torch.linspace(0.0, 6.0, 61).tolist()
        )[1]
        assert best == pytest.approx(math.log(33.0), abs=0.15)

    def test_the_precision_weighting_reaches_the_mean_gradient(self, spec):
        """``eps/sigma^2``: a confident channel pulls harder on its mean."""

        def mean_gradient(log_sigma: float) -> float:
            prediction = torch.zeros(1, NEXT_STATE_OUTPUT_DIM, requires_grad=True)
            with torch.no_grad():
                prediction[0, PHYSICAL_MEAN_DIM + 9] = log_sigma  # health log sigma
            labels = torch.zeros(1, PHYSICAL_MEAN_DIM)
            labels[0, HEALTH] = 1.0
            spec.loss(prediction, labels)[0, HEALTH].backward()
            return prediction.grad[0, HEALTH].abs().item()

        assert mean_gradient(-1.0) > mean_gradient(0.0) > mean_gradient(1.0)

    def test_every_channel_is_differentiable_through_both_blocks(self, spec):
        prediction = torch.randn(4, NEXT_STATE_OUTPUT_DIM, requires_grad=True)
        labels = torch.randn(4, PHYSICAL_MEAN_DIM)
        spec.loss(prediction, labels).sum().backward()
        assert torch.isfinite(prediction.grad).all()
        assert (prediction.grad.abs().sum(0) > 0).all(), "no output is unused"

    def test_the_paired_channels_share_one_correlation(self, spec):
        """Two log sigmas and one rho, not four independent variances."""
        prediction = torch.zeros(1, NEXT_STATE_OUTPUT_DIM, requires_grad=True)
        labels = torch.zeros(1, PHYSICAL_MEAN_DIM)
        labels[0, POSITION_X] = 1.0
        labels[0, POSITION_Y] = 1.0
        spec.loss(prediction, labels)[0, POSITION_X].backward()
        assert prediction.grad[0, PHYSICAL_MEAN_DIM + POSITION_RHO] != 0.0
