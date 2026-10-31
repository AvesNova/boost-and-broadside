"""Yemong's acceptance item 5 (spec 11): the codes round-trip at the v2 ranges, and their
layout and constants are the spec's."""

import math

import pytest
import torch

from bnb.contracts import GameConfig
from bnb.game import api
from bnb.models.yemong import codes, critic, density, shots
from bnb.models.yemong.model import Yemong, YemongConfig
from bnb.models.yemong.state_code import MEANS, POSITION_PERIOD, ShipStateCode, velocity_reach

RULES = api.rules_of(GameConfig(5, 10, 2600.0))
TYPES = api.ship_types()
COUNT = 4000


@pytest.fixture(scope="module")
def code():
    return ShipStateCode(RULES, TYPES).double()


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
    means = torch.stack([
        _uniform(-13_000.0, 13_000.0, generator), _uniform(-13_000.0, 13_000.0, generator),
        speed * torch.cos(angle), speed * torch.sin(angle), _uniform(-math.pi, math.pi, generator),
        inside(-8.0, 8.0), inside(-attack, attack), inside(0.0, RULES.shield_recharge_delay),
        inside(0.0, 1.0) * shield_max, inside(0.0, 1.0) * energy_max,
        (torch.rand(COUNT, generator=generator) * (interval + 1)).floor().double()
        * RULES.tick_length,
        inside(-math.log(2.0), math.log(2.0))], -1)
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
    spreads = torch.tensor([30.0, 25.0, 5.0, 16.0, 0.3, 0.5, 0.05, 0.5, 5.0, 300.0, 0.05,
                            0.0, 0.0, 0.0, 0.0], dtype=torch.float64).expand(COUNT, -1)
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
    torch.testing.assert_close(torch.remainder(means[:, 0], POSITION_PERIOD), decoded[:, 0],
                               rtol=0, atol=1e-9)


def test_v2_shot_constants():
    model = Yemong(YemongConfig(), RULES, TYPES)
    parameters = model.shot_code
    top = 2.0 * (500.0 + velocity_reach())
    assert parameters.top_speed == pytest.approx(top)
    assert parameters.displacement_period == pytest.approx(2.5 * top * 12 / 30)
    assert parameters.collision_radius == 8.0
    assert parameters.trajectory_width == 154 and parameters.outcome_width == 19
    snapshots = torch.tensor([[10.0, -5.0, 700.0, 20.0, 1.3]])
    assert shots.trajectory_code(parameters, snapshots).shape == (1, 154)
    outcome = shots.outcome_code(parameters, torch.tensor([0, 1, 0]),
                                 torch.tensor([20.0, 0.0, 1e4]), torch.tensor([2, 1, 1]))
    torch.testing.assert_close(outcome.sum(-1), torch.ones(3))
    assert outcome[1, 0] == 1.0 and outcome[2, -1] == 1.0  # an enemy hit; clear
    assert outcome[0, 2:10].sum() == 0.0 and outcome[0, 10:18].sum() > 0.99  # an ally pass


def test_density_target_and_rates():
    model = Yemong(YemongConfig(), RULES, TYPES)
    cells = density.cell_count(10)
    assert cells == 331 and model.density.lasts[0].out_features == 2 * cells
    bias = model.density.lasts[0].bias
    assert torch.allclose(bias, torch.full_like(bias, math.log(5.0 / cells)))
    x = torch.tensor([[0.0, 100.0, 9000.0]])
    counts = density.hex_counts(x, torch.zeros_like(x), torch.tensor([[True, False, True]]),
                                torch.ones_like(x, dtype=torch.bool), torch.tensor([[2600.0]]), 10)
    assert counts.shape == (1, 662)
    assert counts[0, :cells].sum() == pytest.approx(1.0)  # the far ally is off the grid
    assert counts[0, cells:].sum() == pytest.approx(1.0)


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
