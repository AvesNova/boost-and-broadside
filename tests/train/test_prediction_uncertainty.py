"""Heteroscedastic next-state predictions: what the uncertainty output buys.

The auxiliary label steps from the believed state to the true next one, so its
spread is set by how wrong the belief currently is -- which depends on the
policy's own head, on how long ships stay unseen, and so on how well the policy
plays. Measured over one run, velocity labels sat about 33x their calibrated
width and position's scale fell by a third *within* the run. No constant fits
that. A Gaussian likelihood does not need one.
"""

import math

import pytest
import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.models.yemong.policy import LOG_VAR_MAX, LOG_VAR_MIN, NextStateHead
from boost_and_broadside.train.rl.features import build_standard_coordinator


@pytest.fixture
def coordinator():
    return build_standard_coordinator(ShipConfig())


def _split(coordinator, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    return torch.cat([mean, log_var], dim=-1)


def _gaussian_dims(coordinator) -> list[int]:
    return [
        spec.p_offset + offset
        for spec in coordinator._predictor_specs
        if spec.predictor.uncertainty_kind == "gaussian"
        for offset in range(spec.p_dim)
    ]


def test_rescaling_a_label_shifts_the_loss_by_a_constant(coordinator) -> None:
    """The property the whole change rests on: the objective is scale-free.

    Scaling a label by k and the predicted sigma with it moves the likelihood by
    exactly log k, the same for every residual. So a feature whose labels grow
    tenfold mid-run does not thereby take over the objective -- which is what
    ``label_scale`` was failing to guarantee.
    """

    P, U = coordinator.total_prediction_dimension, coordinator.total_uncertainty_dimension
    assert U, "no predictor reports uncertainty"
    torch.manual_seed(0)
    # Gaussian channels only: a circular quantity has no scale ambiguity to
    # remove -- an angle is already measured against a fixed 2*pi period -- so
    # von Mises is deliberately not invariant here.
    mean = torch.randn(64, P)
    labels = torch.randn(64, P)
    log_var = torch.zeros(64, U)

    base = coordinator.prediction_loss(_split(coordinator, mean, log_var), labels)

    k = 10.0
    scaled = coordinator.prediction_loss(
        _split(coordinator, mean * k, log_var + 2 * math.log(k)), labels * k
    )

    for dim in _gaussian_dims(coordinator):
        shift = (scaled[:, dim] - base[:, dim]).abs()
        assert torch.allclose(shift, torch.full_like(shift, math.log(k)), atol=1e-4), (
            f"dim {dim} did not shift by a constant under rescaling"
        )


def test_a_wider_sigma_earns_less_gradient_on_the_mean(coordinator) -> None:
    """Weighting is 1/sigma^2, which is what down-weights unpredictable labels.

    A long-hidden token's label is mostly belief error nobody could have
    predicted. The head learns a wide sigma there, and the learning signal
    concentrates on tokens whose labels are real dynamics.
    """

    P, U = coordinator.total_prediction_dimension, coordinator.total_uncertainty_dimension
    labels = torch.ones(1, P)

    def gradient(log_var_value: float) -> torch.Tensor:
        mean = torch.zeros(1, P, requires_grad=True)
        log_var = torch.full((1, U), log_var_value)
        coordinator.prediction_loss(_split(coordinator, mean, log_var), labels).sum().backward()
        return mean.grad[0].clone()

    tight = gradient(0.0)  # sigma^2 = 1
    wide = gradient(math.log(4.0))  # sigma^2 = 4

    for dim in _gaussian_dims(coordinator):
        assert wide[dim] == pytest.approx(tight[dim] / 4.0, rel=1e-5)


def test_a_predictor_without_uncertainty_keeps_plain_squared_error() -> None:
    """The fallback that let these channels convert one at a time.

    Every shipped predictor now reports an uncertainty, so this builds one that
    does not rather than asserting against whichever feature happens to lag.
    """

    from boost_and_broadside.env.observation import ObsKey
    from boost_and_broadside.train.rl.features import (
        Accessor,
        AbsolutePredictor,
        Feature,
        FeatureCoordinator,
        Identity,
    )

    class Plain(AbsolutePredictor):
        uncertainty_kind = None

        def uncertainty_dim(self, in_channels: int) -> int:
            return 0

    lone = FeatureCoordinator(
        [Feature("ang", Accessor(ObsKey.ANG_VEL), Identity(), Identity(), Plain())]
    )
    assert lone.total_uncertainty_dimension == 0
    loss = lone.prediction_loss(torch.zeros(1, 1), torch.full((1, 1), 3.0))
    assert loss[0, 0] == pytest.approx(9.0), "squared error, with no spread involved"


def test_the_head_clamps_its_log_variance(coordinator) -> None:
    """The likelihood is unbounded below as sigma falls; nothing else bounds it."""

    head = NextStateHead(
        8,
        pred_dim=coordinator.total_prediction_dimension,
        uncertainty_dim=coordinator.total_uncertainty_dimension,
    )
    with torch.no_grad():
        # Drive the output layer hard in both directions.
        head.net[-1].bias.fill_(1e4)
        high = head(torch.randn(4, 8))
        head.net[-1].bias.fill_(-1e4)
        low = head(torch.randn(4, 8))

    P = coordinator.total_prediction_dimension
    assert high[..., P:].max() <= LOG_VAR_MAX + 1e-5
    assert low[..., P:].min() >= LOG_VAR_MIN - 1e-5
    # Means are deliberately not clamped.
    assert high[..., :P].max() > LOG_VAR_MAX


def test_resources_predict_as_scalars_that_invert_to_physical_units(coordinator) -> None:
    """health, power and cooldown are bounded scalars, not circular quantities."""

    ship = ShipConfig()
    specs = {spec.name: spec for spec in coordinator._predictor_specs}
    for name, scale in (
        ("health", ship.max_health),
        ("power", ship.max_power),
        ("cooldown", ship.firing_cooldown),
    ):
        spec = specs[name]
        assert spec.t_dim == 1, f"{name} should have a scalar target, not a (sin, cos) pair"
        assert spec.u_dim == 1, f"{name} should carry an uncertainty"
        half = torch.full((2, 1), 0.5)
        assert spec.target_encoder.invert(half)[0, 0] == pytest.approx(scale / 2.0)


def _circular_dims(coordinator) -> list[int]:
    return [
        spec.p_offset
        for spec in coordinator._predictor_specs
        if spec.predictor.uncertainty_kind == "von_mises"
    ]


def test_the_circular_loss_is_periodic_in_two_pi(coordinator) -> None:
    """The reason a Gaussian cannot be used on these channels.

    A phase residual of d and one of d + 2*pi describe the same place on the
    circle. Squared error calls the second enormous; the likelihood has to call
    them equal, or the head is penalised for being right the long way round.
    """

    P, U = coordinator.total_prediction_dimension, coordinator.total_uncertainty_dimension
    scale = coordinator.label_scale_vector(torch.device("cpu"))
    circular = _circular_dims(coordinator)
    assert circular, "no predictor reports a von Mises uncertainty"

    def loss_at(dim: int, residual_radians: float) -> float:
        mean = torch.zeros(1, P)
        mean[0, dim] = residual_radians * scale[dim]
        full = torch.cat([mean, torch.full((1, U), 2.0)], dim=-1)
        return coordinator.prediction_loss(full, torch.zeros(1, P))[0, dim].item()

    for dim in circular:
        assert loss_at(dim, 0.3) == pytest.approx(loss_at(dim, 0.3 + 2 * math.pi), rel=1e-5)
        # Symmetric across the wrap rather than discontinuous at it.
        assert loss_at(dim, math.pi - 0.01) == pytest.approx(
            loss_at(dim, -math.pi + 0.01), rel=1e-5
        )
        # And still minimised by being right.
        assert loss_at(dim, 0.0) < loss_at(dim, 0.3) < loss_at(dim, 1.2)


def test_a_confident_circular_belief_matches_the_gaussian_limit(coordinator) -> None:
    """von Mises becomes Gaussian as the belief tightens, with kappa as 1/sigma^2.

    This is what says the two branches are one objective in different geometry
    rather than two unrelated losses whose magnitudes cannot be compared.
    """

    P, U = coordinator.total_prediction_dimension, coordinator.total_uncertainty_dimension
    scale = coordinator.label_scale_vector(torch.device("cpu"))
    dim = _circular_dims(coordinator)[0]

    log_kappa, residual = 6.0, 0.01
    mean = torch.zeros(1, P)
    mean[0, dim] = residual * scale[dim]
    full = torch.cat([mean, torch.full((1, U), log_kappa)], dim=-1)
    measured = coordinator.prediction_loss(full, torch.zeros(1, P))[0, dim].item()

    kappa = math.exp(log_kappa)
    gaussian = 0.5 * (kappa * residual**2 - math.log(kappa) + math.log(2 * math.pi))
    assert measured == pytest.approx(gaussian, rel=1e-3)


def test_the_circular_loss_undoes_label_scale_before_taking_a_cosine(coordinator) -> None:
    """label_scale conditions the mean; it is not part of the geometry.

    Position's scale is ~177, so a residual left in scaled space would be tens of
    radians and the cosine would be measuring noise.
    """

    P, U = coordinator.total_prediction_dimension, coordinator.total_uncertainty_dimension
    scale = coordinator.label_scale_vector(torch.device("cpu"))
    dim = _circular_dims(coordinator)[0]
    assert scale[dim].item() > 10.0, "this test is only meaningful for a scaled channel"

    def loss_at(residual_radians: float) -> float:
        mean = torch.zeros(1, P)
        mean[0, dim] = residual_radians * scale[dim]
        full = torch.cat([mean, torch.full((1, U), 1.0)], dim=-1)
        return coordinator.prediction_loss(full, torch.zeros(1, P))[0, dim].item()

    # A full turn in *radians* must be indistinguishable from no error at all.
    assert loss_at(2 * math.pi) == pytest.approx(loss_at(0.0), rel=1e-5)
