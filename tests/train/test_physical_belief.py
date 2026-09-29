"""The physical belief representation and its Gaussian next-state objective."""

import math

import pytest
import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.train.rl.physical_belief import (
    ATTITUDE,
    CORRELATION_COLUMNS,
    LOG_SIGMA_MAX,
    LOG_SIGMA_MIN,
    NEXT_STATE_OUTPUT_DIM,
    PHYSICAL_MEAN_DIM,
    PHYSICAL_MEAN_NAMES,
    PHYSICAL_UNCERTAINTY_DIM,
    POSITION_RHO,
    POSITION_X,
    POSITION_Y,
    SCALAR_UNCERTAINTY,
    UNCERTAINTY_NAMES,
    VELOCITY_RHO,
    VELOCITY_X,
    PhysicalNextState,
    certain_uncertainty,
    physical_mean_deltas,
    physical_means_from_state,
    uncertainty_clamp_bounds,
    unknown_uncertainty,
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
        assert PHYSICAL_MEAN_DIM == 11
        assert PHYSICAL_UNCERTAINTY_DIM == 13
        assert NEXT_STATE_OUTPUT_DIM == 24

    def test_every_channel_has_exactly_one_spread(self):
        """Eleven log sigmas plus two correlation latents, no column unused."""
        paired = {*range(2), *range(3, 5)}
        scalar = {column for _, column in SCALAR_UNCERTAINTY}
        assert paired | scalar | set(CORRELATION_COLUMNS) == set(range(PHYSICAL_UNCERTAINTY_DIM))
        assert len(scalar) == len(SCALAR_UNCERTAINTY) == 7
        assert not paired & scalar

    def test_names_are_ordered_and_distinct(self):
        assert len(set(UNCERTAINTY_NAMES)) == PHYSICAL_UNCERTAINTY_DIM
        assert PHYSICAL_MEAN_NAMES[POSITION_X] == "position_x"
        assert PHYSICAL_MEAN_NAMES[ATTITUDE] == "attitude"

    def test_correlation_columns_clamp_tighter_than_the_sigmas(self):
        lower, upper = uncertainty_clamp_bounds()
        for column in range(PHYSICAL_UNCERTAINTY_DIM):
            if column in CORRELATION_COLUMNS:
                assert abs(math.tanh(upper[column])) < 1.0
            else:
                assert (lower[column], upper[column]) == (LOG_SIGMA_MIN, LOG_SIGMA_MAX)

    def test_certain_is_tighter_than_unknown_and_both_zero_the_correlation(self):
        certain, unknown = certain_uncertainty(), unknown_uncertainty()
        for column in range(PHYSICAL_UNCERTAINTY_DIM):
            if column in CORRELATION_COLUMNS:
                assert certain[column] == unknown[column] == 0.0
            else:
                assert certain[column] < unknown[column]
                assert math.isfinite(certain[column])


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


class TestRecursion:
    def test_position_wraps_and_bounded_channels_clamp(self, spec):
        means = torch.zeros(1, PHYSICAL_MEAN_DIM)
        means[0, 7] = 100.0  # health at its cap
        prediction = torch.zeros(1, NEXT_STATE_OUTPUT_DIM)
        prediction[0, POSITION_X] = -10.0  # 10 * 2.5 px backwards past zero
        prediction[0, 7] = 50.0  # a large positive health delta
        advanced = spec.apply_means(means, prediction)
        assert advanced[0, POSITION_X].item() == pytest.approx(spec.world_size[0] - 25.0)
        assert advanced[0, 7].item() == pytest.approx(100.0)

    def test_a_runaway_forecast_cannot_leave_the_bounded_set(self, spec):
        means = torch.zeros(3, PHYSICAL_MEAN_DIM)
        prediction = torch.full((3, NEXT_STATE_OUTPUT_DIM), 1e6)
        for _ in range(20):
            means = spec.apply_means(means, prediction)
        assert torch.isfinite(means).all()
        lower = spec.lower_vector(means.device)
        upper = spec.upper_vector(means.device)
        finite = torch.isfinite(lower) & torch.isfinite(upper)
        assert (means[..., finite] <= upper[finite] + 1e-4).all()
        assert (means[..., finite] >= lower[finite] - 1e-4).all()
        assert (means[..., POSITION_X].abs() <= spec.world_size[0]).all()
        assert (means[..., ATTITUDE].abs() <= math.pi + 1e-5).all()

    def test_attitude_wraps_rather_than_saturating(self, spec):
        means = torch.zeros(1, PHYSICAL_MEAN_DIM)
        means[0, ATTITUDE] = 3.0
        prediction = torch.zeros(1, NEXT_STATE_OUTPUT_DIM)
        prediction[0, ATTITUDE] = 10.0  # 1.0 rad
        advanced = spec.apply_means(means, prediction)
        assert advanced[0, ATTITUDE].item() == pytest.approx(4.0 - 2 * math.pi)


class TestObjective:
    def test_per_channel_terms_sum_to_the_joint_likelihood(self, spec):
        """The paired split must be exact, not merely plausible."""
        torch.manual_seed(1)
        prediction = torch.randn(5, NEXT_STATE_OUTPUT_DIM) * 0.5
        labels = torch.randn(5, PHYSICAL_MEAN_DIM)
        per_channel = spec.loss(prediction, labels)

        residual = spec.residual(prediction, labels)
        uncertainty = prediction[..., PHYSICAL_MEAN_DIM:]
        total = torch.zeros(5)
        for mean_x, sigma_x, sigma_y, rho_column in (
            (POSITION_X, 0, 1, POSITION_RHO),
            (2, 3, 4, VELOCITY_RHO),
        ):
            sx = uncertainty[:, sigma_x]
            sy = uncertainty[:, sigma_y]
            rho = torch.tanh(uncertainty[:, rho_column])
            a = residual[:, mean_x] / sx.exp()
            b = residual[:, mean_x + 1] / sy.exp()
            one_minus = 1.0 - rho * rho
            total = total + (
                math.log(2 * math.pi)
                + sx
                + sy
                + 0.5 * torch.log(one_minus)
                + (a * a - 2 * rho * a * b + b * b) / (2 * one_minus)
            )
        for channel, column in SCALAR_UNCERTAINTY:
            s = uncertainty[:, column]
            z = residual[:, channel] / s.exp()
            total = total + 0.5 * math.log(2 * math.pi) + s + 0.5 * z * z
        assert torch.allclose(per_channel.sum(-1), total, atol=1e-5)

    def test_zero_correlation_reduces_to_two_independent_scalars(self, spec):
        prediction = torch.zeros(1, NEXT_STATE_OUTPUT_DIM)
        labels = torch.zeros(1, PHYSICAL_MEAN_DIM)
        labels[0, POSITION_X] = 1.0
        labels[0, POSITION_Y] = -2.0
        loss = spec.loss(prediction, labels)
        expected_x = 0.5 * math.log(2 * math.pi) + 0.5 * 1.0
        expected_y = 0.5 * math.log(2 * math.pi) + 0.5 * 4.0
        assert loss[0, POSITION_X].item() == pytest.approx(expected_x, abs=1e-6)
        assert loss[0, POSITION_Y].item() == pytest.approx(expected_y, abs=1e-6)

    def test_correlation_explains_a_correlated_residual(self, spec):
        """A positive rho must make a same-sign residual pair cheaper."""
        labels = torch.zeros(1, PHYSICAL_MEAN_DIM)
        labels[0, POSITION_X] = 1.0
        labels[0, POSITION_Y] = 1.0

        def cost(latent: float) -> float:
            prediction = torch.zeros(1, NEXT_STATE_OUTPUT_DIM)
            prediction[0, PHYSICAL_MEAN_DIM + POSITION_RHO] = latent
            return spec.loss(prediction, labels)[0, :2].sum().item()

        assert cost(1.0) < cost(0.0) < cost(-1.0)

    def test_scale_free_in_the_label(self, spec):
        """A channel whose labels are a thousand times larger learns a sigma, not a weight."""
        labels = torch.zeros(1, PHYSICAL_MEAN_DIM)
        labels[0, 9] = 1.0
        tight = torch.zeros(1, NEXT_STATE_OUTPUT_DIM)
        loose = torch.zeros(1, NEXT_STATE_OUTPUT_DIM)
        loose[0, PHYSICAL_MEAN_DIM + 11] = math.log(1000.0)
        big = labels * 1000.0
        assert spec.loss(tight, labels)[0, 9].item() == pytest.approx(
            spec.loss(loose, big)[0, 9].item() - math.log(1000.0), abs=1e-4
        )

    def test_attitude_residual_wraps_on_the_circle(self, spec):
        """+pi and -pi are the same rotation, so that residual must cost nothing."""
        labels = torch.zeros(1, PHYSICAL_MEAN_DIM)
        labels[0, ATTITUDE] = math.pi / spec.scales[ATTITUDE]
        prediction = torch.zeros(1, NEXT_STATE_OUTPUT_DIM)
        prediction[0, ATTITUDE] = -math.pi / spec.scales[ATTITUDE]
        residual = spec.residual(prediction, labels)
        assert residual[0, ATTITUDE].item() == pytest.approx(0.0, abs=1e-4)
        assert spec.loss(prediction, labels)[0, ATTITUDE].item() == pytest.approx(
            0.5 * math.log(2 * math.pi), abs=1e-4
        )

    def test_only_the_attitude_residual_is_wrapped(self, spec):
        prediction = torch.zeros(1, NEXT_STATE_OUTPUT_DIM)
        labels = torch.full((1, PHYSICAL_MEAN_DIM), 500.0)
        residual = spec.residual(prediction, labels)
        assert (residual[0, [c for c in range(PHYSICAL_MEAN_DIM) if c != ATTITUDE]] == -500.0).all()

    def test_gradients_are_finite_at_the_clamp_bounds(self, spec):
        prediction = torch.zeros(2, NEXT_STATE_OUTPUT_DIM, requires_grad=True)
        with torch.no_grad():
            prediction[:, PHYSICAL_MEAN_DIM:] = LOG_SIGMA_MIN
            prediction[0, PHYSICAL_MEAN_DIM + POSITION_RHO] = 5.0
        labels = torch.randn(2, PHYSICAL_MEAN_DIM)
        spec.loss(prediction, labels).sum().backward()
        assert torch.isfinite(prediction.grad).all()

    def test_variance_reads_each_channels_own_spread(self, spec):
        prediction = torch.zeros(1, NEXT_STATE_OUTPUT_DIM)
        prediction[0, PHYSICAL_MEAN_DIM + 9] = math.log(3.0)  # health log sigma
        variance = spec.variance(prediction)
        assert variance[0, 7].item() == pytest.approx(9.0)
        assert variance[0, 0].item() == pytest.approx(1.0)


class TestBetaNLL:
    """The beta-NLL weighting, which decides *which tokens* the term learns from.

    A plain Gaussian likelihood weights a token by its Fisher information, so
    ``d/dmu`` is ``r / sigma**2`` and, where sigma is calibrated, the gradient
    goes as ``1 / r`` -- the head is pulled hardest by whatever it already
    predicts best. Beta dials that exponent; what has to hold is that it dials
    exactly the exponent and nothing else.
    """

    @staticmethod
    def _model():
        from boost_and_broadside.config.defaults import SHIP_CONFIG

        return PhysicalNextState.from_ship_config(SHIP_CONFIG)

    @staticmethod
    def _sample(seed: int = 0):
        torch.manual_seed(seed)
        return (
            torch.randn(128, NEXT_STATE_OUTPUT_DIM),
            torch.randn(128, PHYSICAL_MEAN_DIM),
        )

    def _grad(self, model, prediction, labels, beta):
        leaf = prediction.clone().requires_grad_(True)
        model.loss(leaf, labels, beta=beta).sum().backward()
        return leaf.grad[:, :PHYSICAL_MEAN_DIM].clone()

    def test_beta_zero_is_the_plain_likelihood(self):
        model = self._model()
        prediction, labels = self._sample()
        assert torch.equal(model.loss(prediction, labels, beta=0.0), model.loss(prediction, labels))

    @pytest.mark.parametrize("beta", [0.5, 1.0])
    def test_the_mean_gradient_is_the_residual_over_sigma_to_the_right_power(self, beta):
        """``r / sigma**(2 - 2*beta)`` on the seven scalar channels: 1/sigma^2 at
        beta 0, 1/sigma at 0.5, and plain ``r`` -- the MSE gradient -- at 1."""
        model = self._model()
        prediction, labels = self._sample()
        scalars = [channel for channel, _ in SCALAR_UNCERTAINTY]
        leaf = prediction.clone().requires_grad_(True)
        expected = model.residual(leaf, labels) / model.variance(leaf).pow(1.0 - beta)
        actual = self._grad(model, prediction, labels, beta)
        assert torch.allclose(actual[:, scalars], expected[:, scalars], atol=1e-5)

    @pytest.mark.parametrize("beta", [0.25, 0.5, 1.0])
    def test_beta_only_rescales_the_gradient_it_never_redirects_it(self, beta):
        """Every channel, bivariate blocks included: the beta gradient is the
        plain one times the detached weight, so nothing about the direction of
        the fit has changed -- only how much each token is listened to."""
        model = self._model()
        prediction, labels = self._sample(1)
        leaf = prediction.clone().requires_grad_(True)
        weight = model._beta_weight(leaf, beta)
        plain = self._grad(model, prediction, labels, 0.0)
        actual = self._grad(model, prediction, labels, beta)
        assert torch.allclose(actual, weight * plain, atol=1e-5)

    def test_the_two_axes_of_a_bivariate_block_share_one_weight(self):
        """Position and velocity are single joint terms split across two
        columns that share a cross term. Weighting the columns separately would
        scale the two halves of that term differently and stop the pair summing
        to any likelihood."""
        model = self._model()
        prediction, _ = self._sample(2)
        weight = model._beta_weight(prediction, 0.5)
        assert torch.allclose(weight[..., POSITION_X], weight[..., POSITION_X + 1])
        assert torch.allclose(weight[..., VELOCITY_X], weight[..., VELOCITY_X + 1])

    def test_the_weight_is_detached_so_sigma_still_trains(self):
        """The whole point of beta-NLL over plain MSE here: the belief plane
        reads sigma, so it has to keep being learned. Only the weighting is
        frozen."""
        model = self._model()
        prediction, labels = self._sample(3)
        leaf = prediction.clone().requires_grad_(True)
        model.loss(leaf, labels, beta=0.5).sum().backward()
        uncertainty_gradient = leaf.grad[:, PHYSICAL_MEAN_DIM:]
        assert uncertainty_gradient.abs().sum() > 0.0

    def test_beta_half_equalizes_a_confident_and_an_uncertain_token(self):
        """The measured failure it exists to fix. At beta 0 two tokens with the
        same standardized error pull in inverse proportion to their sigma; at
        0.5 they pull equally, whatever their confidence."""
        model = self._model()
        labels = torch.zeros(2, PHYSICAL_MEAN_DIM)
        prediction = torch.zeros(2, NEXT_STATE_OUTPUT_DIM)
        cooldown, column = SCALAR_UNCERTAINTY[-2]
        sigma_index = PHYSICAL_MEAN_DIM + column  # uncertainty follows the means
        # Same standardized residual of one, sigmas two orders of magnitude apart.
        prediction[0, sigma_index], prediction[1, sigma_index] = -5.0, -0.4
        prediction[:, cooldown] = prediction[:, sigma_index].exp()

        confident, uncertain = self._grad(model, prediction, labels, 0.0)[:, cooldown]
        # Exactly the ratio of the two sigmas, exp(5 - 0.4).
        assert float(confident / uncertain) == pytest.approx(math.exp(4.6), rel=1e-4)

        confident, uncertain = self._grad(model, prediction, labels, 0.5)[:, cooldown]
        assert float(confident / uncertain) == pytest.approx(1.0, rel=1e-4)
