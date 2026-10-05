"""Numerical invariants of the categorical belief codes."""

import math

import pytest
import torch

from boost_and_broadside.train.rl.categorical_codes import (
    CircularCode,
    PositionCode,
    ScalarCode,
    VelocityCode,
    compress_velocity,
    expand_velocity,
    packed_inverse_velocity_jacobian,
    packed_velocity_jacobian,
    sandwich,
    velocity_jacobian,
)


@pytest.mark.parametrize("sigma", [0.0, 0.01, 0.2, 1.0])
def test_attitude_round_trip(sigma):
    code = CircularCode(2.0 * math.pi, 4, 4)
    means = torch.linspace(0.0, 2.0 * math.pi, 137, dtype=torch.float64)[:-1]
    spread = torch.full_like(means, sigma)
    decoded, decoded_spread = code.decode(code.encode(means, spread))
    error = (decoded - means + math.pi).remainder(2.0 * math.pi) - math.pi
    torch.testing.assert_close(error, torch.zeros_like(error), atol=1e-10, rtol=0.0)
    torch.testing.assert_close(decoded_spread, spread, atol=1e-9, rtol=1e-6)


@pytest.mark.parametrize("sigma", [0.0, 1.0, 100.0, 2000.0])
def test_position_round_trip_including_torus_seam(sigma):
    code = PositionCode((65536.0, 65536.0))
    means = torch.tensor(
        [[0.0, 0.0], [65535.9, 0.1], [1234.56, 23456.78], [40000.0, 19000.0]],
        dtype=torch.float64,
    )
    spread = torch.full(means.shape[:-1], sigma, dtype=means.dtype)
    decoded, decoded_spread = code.decode(code.encode(means, spread))
    error = (decoded - means + 32768.0).remainder(65536.0) - 32768.0
    torch.testing.assert_close(error, torch.zeros_like(error), atol=1e-8, rtol=0.0)
    torch.testing.assert_close(decoded_spread, spread, atol=1e-8, rtol=1e-6)


def test_sharp_position_has_at_most_four_colours_per_level():
    code = PositionCode((65536.0, 65536.0))
    mean = torch.tensor([[1234.56, 23456.78]])
    probabilities = code.encode(mean, torch.zeros(1))
    assert ((probabilities > 0).sum(-1) <= 4).all()
    torch.testing.assert_close(probabilities.sum(-1), torch.ones(1, code.levels))


def test_float32_position_round_trip_is_within_coordinate_precision():
    generator = torch.Generator().manual_seed(8)
    code = PositionCode((65536.0, 65536.0))
    mean = torch.rand(1000, 2, generator=generator) * 65536.0
    spread = torch.logspace(0, 3.7, 1000)
    decoded, decoded_spread = code.decode(code.encode(mean, spread))
    error = (decoded - mean + 32768.0).remainder(65536.0) - 32768.0
    torch.testing.assert_close(error, torch.zeros_like(error), atol=0.008, rtol=0.0)
    torch.testing.assert_close(decoded_spread, spread, atol=1e-5, rtol=2e-6)


def test_sharp_position_stays_certain_in_float32():
    generator = torch.Generator().manual_seed(9)
    code = PositionCode((65536.0, 65536.0))
    mean = torch.rand(1000, 2, generator=generator) * 65536.0
    _, spread = code.decode(code.encode(mean, torch.zeros(1000)))
    torch.testing.assert_close(spread, torch.zeros_like(spread), atol=0.0, rtol=0.0)


def test_uniform_circular_code_remains_uncertain():
    code = CircularCode(2.0 * math.pi, 4, 4)
    probabilities = torch.full((2, code.levels, code.colours), 1.0 / code.colours)
    mean, spread = code.decode(probabilities)
    assert (spread >= code.period).all()
    torch.testing.assert_close(code.encode(mean, spread), probabilities)


def test_position_codec_supports_full_graph_compilation():
    code = PositionCode((65536.0, 65536.0))

    def round_trip(mean, spread):
        return code.decode(code.encode(mean, spread))

    compiled = torch.compile(round_trip, backend="eager", fullgraph=True)
    mean = torch.tensor([[65535.9, 0.1], [1234.56, 23456.78]])
    spread = torch.tensor([0.0, 100.0])
    expected = round_trip(mean, spread)
    actual = compiled(mean, spread)
    for measured, reference in zip(actual, expected):
        torch.testing.assert_close(measured, reference)


def test_velocity_compression_round_trip_at_zero_and_high_speed():
    velocity = torch.tensor([[0.0, 0.0], [0.001, -0.002], [215.0, 0.0], [-600.0, 195.0]])
    torch.testing.assert_close(expand_velocity(compress_velocity(velocity)), velocity)


@pytest.mark.parametrize("velocity", [[0.0, 0.0], [20.0, -35.0], [600.0, 195.0]])
def test_velocity_jacobian_matches_autograd(velocity):
    mean = torch.tensor(velocity, dtype=torch.float64, requires_grad=True)
    measured = torch.autograd.functional.jacobian(compress_velocity, mean)
    torch.testing.assert_close(velocity_jacobian(mean), measured)


def test_velocity_covariance_jacobian_round_trip():
    mean = torch.tensor([[0.0, 0.0], [100.0, 215.0], [600.0, -195.0]], dtype=torch.float64)
    covariance = torch.tensor([[40.0, -10.0], [-10.0, 20.0]], dtype=torch.float64)
    jacobian = velocity_jacobian(mean)
    compressed = jacobian @ covariance @ jacobian.transpose(-1, -2)
    inverse = torch.linalg.inv(jacobian)
    decoded = inverse @ compressed @ inverse.transpose(-1, -2)
    torch.testing.assert_close(decoded, covariance.expand_as(decoded))


# ----------------------------------------------------------------------
# Scalar and velocity codes
# ----------------------------------------------------------------------

_HEALTH = ScalarCode(0.0, 100.0, 21)


@pytest.mark.parametrize("sigma", [0.0, 0.3, 1.0, 2.5, 8.0])
def test_scalar_round_trip_is_exact_in_the_interior(sigma):
    means = torch.linspace(40.0, 60.0, 41, dtype=torch.float64)
    spread = torch.full_like(means, sigma)
    decoded, decoded_spread = _HEALTH.decode(_HEALTH.encode(means, spread))
    torch.testing.assert_close(decoded, means, atol=1e-9, rtol=0.0)
    torch.testing.assert_close(decoded_spread, spread, atol=1e-5, rtol=1e-6)


def test_sigma_zero_is_the_exact_two_hot():
    means = torch.tensor([0.0, 3.0, 52.5, 73.3, 99.0, 100.0], dtype=torch.float64)
    encoded = _HEALTH.encode(means, torch.zeros_like(means))
    assert torch.equal(encoded, _HEALTH.sharp(means))
    assert ((encoded > 0).sum(-1) <= 2).all()
    _, spread = _HEALTH.decode(encoded)
    assert torch.equal(spread, torch.zeros_like(spread))


def test_the_two_hot_term_is_subtracted_on_decode():
    """A two-hot halfway between bins has variance h^2/4 and no physical spread."""
    probabilities = torch.zeros(21, dtype=torch.float64)
    probabilities[10] = probabilities[11] = 0.5
    mean, spread = _HEALTH.decode(probabilities)
    assert mean.item() == pytest.approx(52.5)
    assert spread.item() == 0.0


def test_a_decoded_spread_is_never_negative_for_any_histogram():
    generator = torch.Generator().manual_seed(4)
    probabilities = torch.rand(500, 21, generator=generator).pow(8)
    probabilities = probabilities / probabilities.sum(-1, keepdim=True)
    mean, spread = _HEALTH.decode(probabilities)
    assert (spread >= 0).all()
    assert ((mean >= 0.0) & (mean <= 100.0)).all()


@pytest.mark.parametrize(
    "mean, sigma",
    [(3.0, 4.0), (10.0, 8.0), (99.0, 2.0), (96.0, 1.0), (1.0, 0.5), (85.0, 5.0)],
)
def test_the_moment_correction_is_exact_near_the_edges(mean, sigma):
    """Folding the spill into an end bin moves both moments; one step restores them."""
    means = torch.tensor([mean], dtype=torch.float64)
    spread = torch.tensor([sigma], dtype=torch.float64)
    decoded, decoded_spread = _HEALTH.decode(_HEALTH.encode(means, spread))
    torch.testing.assert_close(decoded, means, atol=1e-9, rtol=0.0)
    torch.testing.assert_close(decoded_spread, spread, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("mean, sigma", [(97.0, 5.0), (90.0, 10.0), (80.0, 15.0)])
def test_edge_drift_settles_instead_of_compounding(mean, sigma):
    """Where the end bin holds too much mass to correct in one step, the moments
    are projected once and then held: repeated zero-residual round trips settle
    within a few units rather than drifting further each step.
    """
    m = torch.tensor([mean], dtype=torch.float32)
    s = torch.tensor([sigma], dtype=torch.float32)
    history = []
    for _ in range(30):
        m, s = _HEALTH.decode(_HEALTH.encode(m, s))
        history.append((m.item(), s.item()))
    assert abs(history[-1][0] - mean) < 3.0
    assert abs(history[-1][1] - sigma) < 3.0
    assert abs(history[-1][0] - history[-10][0]) < 0.5, "the mean has stopped moving"


def test_a_spread_beyond_what_the_range_admits_is_projected():
    """A mean at the edge cannot carry a spread: the histogram has no room."""
    means = torch.tensor([100.0, 0.0], dtype=torch.float64)
    encoded = _HEALTH.encode(means, torch.tensor([10.0, 10.0], dtype=torch.float64))
    assert torch.equal(encoded, _HEALTH.sharp(means))


def test_float32_scalar_round_trip_is_within_precision():
    generator = torch.Generator().manual_seed(6)
    means = 20.0 + torch.rand(2000, generator=generator) * 60.0
    spread = torch.rand(2000, generator=generator) * 6.0
    decoded, decoded_spread = _HEALTH.decode(_HEALTH.encode(means, spread))
    torch.testing.assert_close(decoded, means, atol=2e-4, rtol=0.0)
    torch.testing.assert_close(decoded_spread, spread, atol=2e-3, rtol=0.0)


@pytest.mark.parametrize(
    "velocity",
    [[0.0, 0.0], [100.0, 215.0], [-300.0, 50.0], [5.0, -3.0], [0.001, 0.0]],
)
def test_velocity_round_trip_with_covariance(velocity):
    code = VelocityCode()
    mean = torch.tensor([velocity], dtype=torch.float64)
    covariance = torch.tensor([[40.0, -10.0, 20.0]], dtype=torch.float64)
    decoded, decoded_covariance = code.decode(code.encode(mean, covariance))
    torch.testing.assert_close(decoded, mean, atol=1e-8, rtol=0.0)
    torch.testing.assert_close(decoded_covariance, covariance, atol=1e-4, rtol=1e-6)


def test_a_sharp_velocity_is_three_two_hots_and_decodes_to_zero_covariance():
    code = VelocityCode()
    mean = torch.tensor([[120.0, -35.0]], dtype=torch.float64)
    encoded = code.encode(mean, torch.zeros(1, 3, dtype=torch.float64))
    assert torch.equal(encoded, code.sharp(mean))
    assert ((encoded > 0).sum(-1) <= 2).all()
    decoded, covariance = code.decode(encoded)
    torch.testing.assert_close(decoded, mean, atol=1e-9, rtol=0.0)
    assert torch.equal(covariance, torch.zeros_like(covariance))


def test_the_packed_jacobians_match_the_matrix_form():
    velocity = torch.tensor([[0.0, 0.0], [20.0, -35.0], [600.0, 195.0]], dtype=torch.float64)
    matrix = velocity_jacobian(velocity)
    packed = packed_velocity_jacobian(velocity)
    torch.testing.assert_close(packed[..., 0], matrix[..., 0, 0])
    torch.testing.assert_close(packed[..., 1], matrix[..., 0, 1])
    torch.testing.assert_close(packed[..., 2], matrix[..., 1, 1])
    inverse = packed_inverse_velocity_jacobian(velocity)
    identity = sandwich(inverse, sandwich(packed, torch.tensor([1.0, 0.0, 1.0])))
    # J^-1 (J I J) J^-1 = I
    torch.testing.assert_close(identity, torch.tensor([1.0, 0.0, 1.0]).expand_as(identity).double())
