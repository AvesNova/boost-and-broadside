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
from boost_and_broadside.env.observation import ObsKey
from boost_and_broadside.models.yemong.policy import LOG_VAR_MAX, LOG_VAR_MIN, NextStateHead
from boost_and_broadside.train.rl.features import (
    Accessor,
    Feature,
    FeatureCoordinator,
    Fourier,
    UnitCirclePredictor,
    build_standard_coordinator,
)


@pytest.fixture
def coordinator():
    return build_standard_coordinator(ShipConfig())


@pytest.fixture
def circular_coordinator():
    """A coordinator whose one predictor is circular.

    Nothing in the shipped table reports a von Mises uncertainty any more:
    position and attitude are predicted as Fourier moments, which shrink toward
    the origin under squared error and so represent an unknown angle without
    needing a concentration. The circular likelihood stays in the library for
    channels that do want a phase, and these pin its geometry -- which is what
    would otherwise rot unnoticed the moment nothing exercised it.

    ``label_scale`` is deliberately large, because one of the properties under
    test is that the loss divides it out before taking a cosine.
    """

    world = 1024.0
    return FeatureCoordinator(
        [
            Feature(
                name="phase_x",
                accessor=Accessor(ObsKey.POS, channels=[0]),
                input_encoder=Fourier(n_freqs=1, periods=world),
                target_encoder=Fourier(n_freqs=1, periods=world),
                predictor=UnitCirclePredictor(cosine_first=False),
                label_scale=177.4,
            )
        ]
    )


def _split(coordinator, mean: torch.Tensor, log_var: torch.Tensor) -> torch.Tensor:
    return torch.cat([mean, log_var], dim=-1)


def _gaussian_dims(coordinator) -> list[int]:
    """Dimensions that actually carry a Gaussian spread.

    Read from the live mask rather than from ``uncertainty_kind``: a predictor
    can report a spread for *some* of its dimensions and leave the rest on plain
    squared error, which is exactly what the circular features do -- one sigma on
    the finest harmonic, nothing on the other nine.
    """

    gaussian, _von_mises, _gather = coordinator._uncertainty_layout(torch.device("cpu"))
    return [i for i, on in enumerate(gaussian.tolist()) if on]


def _plain_mse_dims(coordinator) -> list[int]:
    """Dimensions with no spread at all, which fall through to squared error."""

    gaussian, von_mises, _gather = coordinator._uncertainty_layout(torch.device("cpu"))
    return [
        i for i, (g, v) in enumerate(zip(gaussian.tolist(), von_mises.tolist())) if not g and not v
    ]


def test_the_circular_features_are_a_hybrid(coordinator) -> None:
    """Nine harmonics on squared error, the finest on a likelihood.

    Confidence for a harmonic pair rides on its magnitude, so no spread is
    needed. The finest one carries a sigma anyway, for gradient share: squared
    error's gradient shrinks as a channel becomes accurate while a likelihood's
    grows, so a purely-MSE position would be starved against the scalar channels
    on exactly the visible ships whose labels are learnable dynamics.
    """

    specs = {s.name: s for s in coordinator._predictor_specs}
    with_sigma = set(_gaussian_dims(coordinator))
    for name in ("position_x", "position_y", "attitude"):
        spec = specs[name]
        harmonics = spec.p_dim // 2
        # Blocked layout: the finest harmonic is index n-1 of each half, so the
        # sigma-bearing columns are n-1 and 2n-1 -- not the final two.
        expected = {spec.p_offset + harmonics - 1, spec.p_offset + spec.p_dim - 1}
        got = {d for d in with_sigma if spec.p_offset <= d < spec.p_offset + spec.p_dim}
        assert got == expected, name
        assert spec.u_dim == 1, name

    # Velocity keeps one spread per axis: its magnitude is speed, not confidence,
    # so it has no spare dimension to carry one.
    assert specs["velocity"].u_dim == 2


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
        AbsolutePredictor,
        Accessor,
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


def test_the_circular_loss_is_periodic_in_two_pi(circular_coordinator) -> None:
    """The reason a Gaussian cannot be used on these channels.

    A phase residual of d and one of d + 2*pi describe the same place on the
    circle. Squared error calls the second enormous; the likelihood has to call
    them equal, or the head is penalised for being right the long way round.
    """

    P = circular_coordinator.total_prediction_dimension
    U = circular_coordinator.total_uncertainty_dimension
    scale = circular_coordinator.label_scale_vector(torch.device("cpu"))
    circular = _circular_dims(circular_coordinator)
    assert circular, "no predictor reports a von Mises uncertainty"

    def loss_at(dim: int, residual_radians: float) -> float:
        mean = torch.zeros(1, P)
        mean[0, dim] = residual_radians * scale[dim]
        full = torch.cat([mean, torch.full((1, U), 2.0)], dim=-1)
        return circular_coordinator.prediction_loss(full, torch.zeros(1, P))[0, dim].item()

    for dim in circular:
        assert loss_at(dim, 0.3) == pytest.approx(loss_at(dim, 0.3 + 2 * math.pi), rel=1e-5)
        # Symmetric across the wrap rather than discontinuous at it.
        assert loss_at(dim, math.pi - 0.01) == pytest.approx(
            loss_at(dim, -math.pi + 0.01), rel=1e-5
        )
        # And still minimised by being right.
        assert loss_at(dim, 0.0) < loss_at(dim, 0.3) < loss_at(dim, 1.2)


def test_a_confident_circular_belief_matches_the_gaussian_limit(circular_coordinator) -> None:
    """von Mises becomes Gaussian as the belief tightens, with kappa as 1/sigma^2.

    This is what says the two branches are one objective in different geometry
    rather than two unrelated losses whose magnitudes cannot be compared.
    """

    P = circular_coordinator.total_prediction_dimension
    U = circular_coordinator.total_uncertainty_dimension
    scale = circular_coordinator.label_scale_vector(torch.device("cpu"))
    dim = _circular_dims(circular_coordinator)[0]

    log_var, residual = -6.0, 0.01
    mean = torch.zeros(1, P)
    mean[0, dim] = residual * scale[dim]
    full = torch.cat([mean, torch.full((1, U), log_var)], dim=-1)
    measured = circular_coordinator.prediction_loss(full, torch.zeros(1, P))[0, dim].item()

    kappa = math.exp(-log_var)
    gaussian = 0.5 * (kappa * residual**2 - math.log(kappa) + math.log(2 * math.pi))
    assert measured == pytest.approx(gaussian, rel=1e-3)


def test_the_circular_loss_undoes_label_scale_before_taking_a_cosine(circular_coordinator) -> None:
    """label_scale conditions the mean; it is not part of the geometry.

    Position's scale is ~177, so a residual left in scaled space would be tens of
    radians and the cosine would be measuring noise.
    """

    P = circular_coordinator.total_prediction_dimension
    U = circular_coordinator.total_uncertainty_dimension
    scale = circular_coordinator.label_scale_vector(torch.device("cpu"))
    dim = _circular_dims(circular_coordinator)[0]
    assert scale[dim].item() > 10.0, "this test is only meaningful for a scaled channel"

    def loss_at(residual_radians: float) -> float:
        mean = torch.zeros(1, P)
        mean[0, dim] = residual_radians * scale[dim]
        full = torch.cat([mean, torch.full((1, U), 1.0)], dim=-1)
        return circular_coordinator.prediction_loss(full, torch.zeros(1, P))[0, dim].item()

    # A full turn in *radians* must be indistinguishable from no error at all.
    assert loss_at(2 * math.pi) == pytest.approx(loss_at(0.0), rel=1e-5)


def test_both_kinds_read_their_uncertainty_the_same_way_round(coordinator) -> None:
    """One block, one meaning: larger is always less certain.

    Concentration is the natural von Mises parameter and the inverse of a
    spread, so the circular loss inverts it internally rather than letting the
    head's output mean opposite things in neighbouring channels.

    The residual is set per kind in the units each loss actually sees -- radians
    for circular, scaled label units for Gaussian -- and chosen well outside the
    confident spread. A residual *inside* it makes confidence cheaper, correctly
    so, which is a different property from the one under test here.
    """

    P, U = coordinator.total_prediction_dimension, coordinator.total_uncertainty_dimension
    scale = coordinator.label_scale_vector(torch.device("cpu"))
    circular = set(_circular_dims(coordinator))
    reported = sorted(set(_gaussian_dims(coordinator)) | circular)
    assert reported

    mean = torch.zeros(1, P)
    for dim in reported:
        # 2 radians for a circular channel; 2 scaled units otherwise. Against a
        # confident sigma^2 of e^-2 that is several sigma out either way.
        mean[0, dim] = 2.0 * scale[dim] if dim in circular else 2.0

    def loss_at(log_var: float) -> torch.Tensor:
        full = torch.cat([mean, torch.full((1, U), log_var)], dim=-1)
        return coordinator.prediction_loss(full, torch.zeros(1, P))[0]

    confident, vague = loss_at(-2.0), loss_at(2.0)
    for dim in reported:
        assert confident[dim] > vague[dim], (
            f"dim {dim}: claiming certainty while badly wrong must cost more than doubt"
        )

    # And the same direction once turned into a variance for the tracker.
    def variance_at(log_var: float) -> torch.Tensor:
        full = torch.cat([torch.zeros(1, P), torch.full((1, U), log_var)], dim=-1)
        return coordinator.prediction_variance(full)[0]

    high, low = variance_at(2.0), variance_at(-2.0)
    for dim in reported:
        assert high[dim] > low[dim], f"dim {dim}: variance must rise with reported log variance"
