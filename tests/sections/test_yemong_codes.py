"""Yemong's acceptance item 5 (spec 11): the codes round-trip at the v2 ranges, and their
layout and constants are the spec's."""

import math

import pytest
import torch

from bnb.contracts import GameConfig
from bnb.game import api
from bnb.models.yemong import codes, critic, density, shots
from bnb.models.yemong.model import Yemong, YemongConfig, yemong_build
from bnb.models.yemong.state_code import (
    MEANS,
    POSITION_PERIOD,
    ShipStateCode,
    state_code_ranges,
    velocity_reach,
)

# The code under test draws from torch's default generator (model initialisation, the
# rollout, the scripted opponents): each test runs with it seeded and restored.
pytestmark = pytest.mark.usefixtures("global_generator")


RULES = api.rules_of(GameConfig(5, 10, 2600.0))
TYPES = api.ship_types()
COUNT = 4000


@pytest.fixture(scope="module")
def code():
    return ShipStateCode(state_code_ranges(RULES, TYPES)).double()


def _uniform(low, high, generator):
    return low + (high - low) * torch.rand(COUNT, dtype=torch.float64, generator=generator)


def _states(generator, margin: float):
    """Random states over the v2 ranges; scalars kept `margin` (a fraction of the range) in
    from the edges, where a spread is representable."""
    ship_type = torch.randint(0, 2, (COUNT,), generator=generator)
    shield_max = torch.where(ship_type == 0, 100.0, 160.0).double()
    energy_max = torch.where(ship_type == 0, 10_000.0, 14_000.0).double()
    interval = torch.where(ship_type == 0, 2, 3)
    attack = 2.0 * RULES.flight.stall_angle

    def inside(low, high):
        pad = margin * (high - low)
        return _uniform(low + pad, high - pad, generator)

    speed = _uniform(0.0, velocity_reach(), generator)
    angle = _uniform(-math.pi, math.pi, generator)
    means = torch.stack(
        [
            _uniform(-13_000.0, 13_000.0, generator),
            _uniform(-13_000.0, 13_000.0, generator),
            speed * torch.cos(angle),
            speed * torch.sin(angle),
            _uniform(-math.pi, math.pi, generator),
            inside(-8.0, 8.0),
            inside(-attack, attack),
            inside(0.0, RULES.shield_recharge_delay),
            inside(0.0, 1.0) * shield_max,
            inside(0.0, 1.0) * energy_max,
            (torch.rand(COUNT, generator=generator) * (interval + 1)).floor().double()
            * RULES.tick_length,
            inside(-math.log(2.0), math.log(2.0)),
        ],
        -1,
    )
    return means, ship_type


def _periodic_error(actual, expected, period):
    return torch.remainder(actual - expected + 0.5 * period, period) - 0.5 * period


def _angle_error(actual, expected):
    return _periodic_error(actual, expected, 2.0 * math.pi)


def _errors(decoded, means):
    """|decoded - means|, positions modulo the code's period, the heading modulo 2 pi."""
    error = (decoded - means).abs()
    error[:, 0:2] = _periodic_error(decoded[:, 0:2], means[:, 0:2], POSITION_PERIOD).abs()
    error[:, MEANS.index("heading")] = _angle_error(decoded[:, 4], means[:, 4]).abs()
    return error


def test_layout(code):
    assert code.width == 490 and code.groups == 23
    assert code.runs[0] == (9, 9) and code.runs[1] == (3, 81) and code.runs[-1] == (1, 4)
    assert len(MEANS) == 12 and code.spread_count == 15
    assert velocity_reach() == pytest.approx(50.0 * (math.e**2 - 1.0))


def test_sharp_round_trip(code):
    generator = torch.Generator().manual_seed(0)
    means, ship_type = _states(generator, margin=0.0)
    decoded, spreads = code.decode(code.sharp(means, ship_type), ship_type)
    error = _errors(decoded, means)
    scale = means.abs().amax(0).clamp_min(1.0)
    assert (error.amax(0) / scale).max() < 1e-12
    assert spreads.abs().max() < 1e-9


def test_round_trip_with_spreads(code):
    generator = torch.Generator().manual_seed(1)
    means, ship_type = _states(generator, margin=0.25)
    means[:, 2:4] = means[:, 2:4].clamp(-150.0, 150.0)
    spreads = torch.tensor(
        [30.0, 25.0, 5.0, 16.0, 0.3, 0.5, 0.05, 0.5, 5.0, 300.0, 0.05, 0.0, 0.0, 0.0, 0.0],
        dtype=torch.float64,
    ).expand(COUNT, -1)
    decoded, decoded_spreads = code.decode(code.encode(means, spreads, ship_type), ship_type)
    error = _errors(decoded, means)
    assert (error.amax(0) / means.abs().amax(0).clamp_min(1.0)).max() < 1e-12
    torch.testing.assert_close(decoded_spreads, spreads, rtol=1e-4, atol=1e-9)


def test_decoded_positions_lie_in_one_period(code):
    means = torch.zeros(3, 12, dtype=torch.float64)
    means[:, 0] = torch.tensor([-12_000.0, -1.0, 30_000.0], dtype=torch.float64)
    ship_type = torch.zeros(3, dtype=torch.long)
    decoded, _ = code.decode(code.sharp(means, ship_type), ship_type)
    assert ((decoded[:, 0] >= 0) & (decoded[:, 0] < POSITION_PERIOD)).all()
    torch.testing.assert_close(
        torch.remainder(means[:, 0], POSITION_PERIOD), decoded[:, 0], rtol=0, atol=1e-9
    )


def test_v2_shot_constants():
    model = Yemong(yemong_build(YemongConfig(), RULES, TYPES))
    parameters = model.shot_code
    top = 2.0 * (500.0 + velocity_reach())
    assert parameters.top_speed == pytest.approx(top)
    assert parameters.displacement_period == pytest.approx(2.5 * top * 12 / 30)
    assert parameters.collision_radius == 8.0
    assert parameters.trajectory_width == 154 and parameters.outcome_width == 19
    snapshots = torch.tensor([[10.0, -5.0, 700.0, 20.0, 1.3]])
    assert shots.trajectory_code(parameters, snapshots).shape == (1, 154)
    outcome = shots.outcome_code(
        parameters, torch.tensor([0, 1, 0]), torch.tensor([20.0, 0.0, 1e4]), torch.tensor([2, 1, 1])
    )
    torch.testing.assert_close(outcome.sum(-1), torch.ones(3))
    assert outcome[1, 0] == 1.0 and outcome[2, -1] == 1.0  # an enemy hit; clear
    assert outcome[0, 2:10].sum() == 0.0 and outcome[0, 10:18].sum() > 0.99  # an ally pass


def test_density_target_and_rates():
    model = Yemong(yemong_build(YemongConfig(), RULES, TYPES))
    cells = density.cell_count(10)
    assert cells == 331 and model.density.lasts[0].out_features == 2 * cells
    bias = model.density.lasts[0].bias
    assert torch.allclose(bias, torch.full_like(bias, math.log(5.0 / cells)))
    x = torch.tensor([[0.0, 100.0, 9000.0]])
    counts = density.HexGrid(10).counts(
        x,
        torch.zeros_like(x),
        torch.tensor([[True, False, True]]),
        torch.ones_like(x, dtype=torch.bool),
        torch.tensor([[2600.0]]),
    )
    assert counts.shape == (1, 662)
    assert counts[0, :cells].sum() == pytest.approx(1.0)  # the far ally is off the grid
    assert counts[0, cells:].sum() == pytest.approx(1.0)


def test_a_ship_on_a_grid_triangle_s_diagonal_counts_no_negative_weight():
    """Where the barycentric u + v rounds to 1 from above, 1 - u - v was -6e-8: a negative
    count, whose y ln y made the density's explained deviance NaN."""
    generator = torch.Generator().manual_seed(0)
    ships = 200_000
    along_r = torch.rand(ships, generator=generator)
    corner = torch.randint(-5, 5, (2, ships), generator=generator).float()
    axis_r, axis_q = corner[0] + along_r, corner[1] + (1.0 - along_r)  # u + v = 1
    scale = 10 / 2600.0
    y = axis_r * density.SQRT3_HALF / scale
    x = (axis_q + 0.5 * axis_r) / scale
    counts = density.HexGrid(10).counts(
        x[:, None],
        y[:, None],
        torch.ones((len(x), 1), dtype=torch.bool),
        torch.ones((len(x), 1), dtype=torch.bool),
        torch.tensor(2600.0),
    )
    assert counts.min() >= 0.0
    assert not torch.special.xlogy(counts, counts).isnan().any()


def test_value_bins_and_two_hot():
    bins = critic.value_bins(51, 2.0, dtype=torch.float64)
    values = torch.linspace(-6.0, 6.0, 101, dtype=torch.float64)
    target = critic.two_hot(values, bins)
    torch.testing.assert_close((target * bins).sum(-1), values)
    logits = torch.log(target.clamp_min(1e-300))
    torch.testing.assert_close(critic.expectation(logits, bins), values)


def test_floor_removal_inverts_the_baseline(code):
    """decode_logits of the next-state baseline is the decode of the code itself."""
    generator = torch.Generator().manual_seed(2)
    means, ship_type = _states(generator, margin=0.25)
    probabilities = code.sharp(means, ship_type)
    recovered = codes.remove_baseline_floor(codes.baseline(probabilities), code.runs)
    torch.testing.assert_close(recovered, probabilities, rtol=0, atol=1e-12)


# A calibrated forecaster's code is the average of the sharp codes of the outcomes it
# expects (the cross-entropy optimum against sharp targets). Its fine levels are flat up to
# sampling noise, and it need not be Gaussian; the decoded spread must still cover those
# outcomes at the Gaussian rate. One code's spread depends on where its mean falls between
# colours (by up to about 40%), so these average over random centres.
POSITION_LEVELS = 9
POSITION_PERIODS = (POSITION_PERIOD, POSITION_PERIOD)
WITHIN_ONE_SIGMA_2D = 1.0 - math.exp(-1.0)  # |e|^2 / (2 sigma^2) <= 1 for a 2-D Gaussian
WITHIN_ONE_SIGMA_1D = math.erf(1.0 / math.sqrt(2.0))
CENTRES, SAMPLES = 64, 500


def _averaged_position_code(samples):
    """(centres, samples, 2) outcomes to the (centres, levels, 9) averaged sharp code."""
    zero = torch.zeros(samples.shape[:-1], dtype=samples.dtype)
    return codes.position_encode(samples, zero, POSITION_PERIODS, POSITION_LEVELS).mean(1)


def _normalised_square(decoded_mean, spread, samples):
    """|e|^2 / (2 sigma^2) per sample, (centres, samples)."""
    error = _periodic_error(decoded_mean[:, None, :], samples, POSITION_PERIOD)
    return error.square().sum(-1) / (2.0 * spread[:, None] ** 2)


def _centres(generator):
    return 3000.0 + 20_000.0 * torch.rand(CENTRES, 1, 2, dtype=torch.float64, generator=generator)


@pytest.mark.parametrize("sigma", [3.0, 30.0, 150.0, 600.0])
def test_an_averaged_gaussian_code_covers_its_samples(sigma):
    generator = torch.Generator().manual_seed(3)
    noise = torch.randn(CENTRES, SAMPLES, 2, dtype=torch.float64, generator=generator)
    samples = _centres(generator) + sigma * noise
    mean, spread = codes.position_decode(
        _averaged_position_code(samples), POSITION_PERIODS, POSITION_LEVELS
    )
    assert (spread / sigma).median() == pytest.approx(1.0, abs=0.15)
    covered = (_normalised_square(mean, spread, samples) <= 1.0).double().mean()
    assert covered == pytest.approx(WITHIN_ONE_SIGMA_2D, abs=0.06)


def test_a_bimodal_code_reports_the_spread_of_both_modes():
    """Two clusters 500 m apart: the spread is the mixture's, not a cluster's."""
    generator = torch.Generator().manual_seed(4)
    side = torch.where(torch.rand(CENTRES, SAMPLES, generator=generator) < 0.5, -1.0, 1.0)
    noise = torch.randn(CENTRES, SAMPLES, 2, dtype=torch.float64, generator=generator)
    samples = _centres(generator) + 20.0 * noise
    samples[..., 0] += 250.0 * side.double()
    mean, spread = codes.position_decode(
        _averaged_position_code(samples), POSITION_PERIODS, POSITION_LEVELS
    )
    centred = samples - samples.mean(1, keepdim=True)
    rms = torch.sqrt(0.5 * centred.square().sum(-1).mean(1))  # per axis
    assert (spread / rms).median() == pytest.approx(1.0, abs=0.2)
    within_two = (_normalised_square(mean, spread, samples) <= 4.0).double().mean()
    assert within_two > 0.9


@pytest.mark.parametrize("flat_from", [3, 5, 7])
def test_flat_fine_levels_are_a_wide_spread_not_certainty(flat_from):
    """Sharp coarse levels over flat fine ones: the position is known only to about the
    first flat level's period, so the spread is a sizeable part of it."""
    mean = torch.tensor([[5000.0, 7000.0]], dtype=torch.float64)
    zero = torch.zeros(1, dtype=torch.float64)
    code = codes.position_encode(mean, zero, POSITION_PERIODS, POSITION_LEVELS)
    code[..., flat_from:, :] = 1.0 / 9.0
    _, spread = codes.position_decode(code, POSITION_PERIODS, POSITION_LEVELS)
    period = POSITION_PERIOD / 3.0**flat_from
    assert 0.5 * period <= spread.item() <= period


@pytest.mark.parametrize("sigma", [0.05, 0.3, 0.8])
def test_an_averaged_heading_code_covers_its_samples(sigma):
    generator = torch.Generator().manual_seed(5)
    centres = 2.0 * math.pi * torch.rand(CENTRES, 1, dtype=torch.float64, generator=generator)
    noise = torch.randn(CENTRES, SAMPLES, dtype=torch.float64, generator=generator)
    samples = centres + sigma * noise
    sharp = codes.circular_encode(samples, torch.zeros_like(samples), 2.0 * math.pi, 4, 4)
    mean, spread = codes.circular_decode(sharp.mean(1), 2.0 * math.pi, 4, 4)
    assert (spread / sigma).median() == pytest.approx(1.0, abs=0.15)
    covered = (_angle_error(mean[:, None], samples).abs() <= spread[:, None]).double().mean()
    assert covered == pytest.approx(WITHIN_ONE_SIGMA_1D, abs=0.06)
