"""The ship-state code: layout, round trip, and what a decoded prediction can be.

These are the §10.2 gates for the belief: with a zero residual the belief must
decode to itself, or a hidden ship drifts on its own every decision.
"""

import math

import pytest
import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.env.frontline import frontline_ship_config
from boost_and_broadside.train.rl.physical_belief import (
    BELIEF_MOMENT_DIM,
    COOLDOWN,
    COOLDOWN_RESIDUAL,
    HEALTH,
    PHYSICAL_MEAN_DIM,
    PHYSICAL_UNCERTAINTY_DIM,
    POSITION_X,
)
from boost_and_broadside.train.rl.ship_codes import (
    CODE_GROUP_DIM,
    CODE_GROUP_NAMES,
    CODE_RUN,
    SHIP_CODE_DIM,
    ShipStateCodec,
)

_CONFIG = frontline_ship_config(ShipConfig())


@pytest.fixture
def codec() -> ShipStateCodec:
    return ShipStateCodec.from_ship_config(_CONFIG)


def _moments(batch: int, dtype=torch.float64) -> tuple[torch.Tensor, torch.Tensor]:
    """Interior means with spreads well inside every channel's range."""
    generator = torch.Generator().manual_seed(5)
    u = torch.rand(batch, PHYSICAL_MEAN_DIM, generator=generator, dtype=dtype)
    means = torch.stack(
        [
            u[:, 0] * 65536.0,
            u[:, 1] * 65536.0,
            (u[:, 2] - 0.5) * 300.0,
            (u[:, 3] - 0.5) * 300.0,
            (u[:, 4] - 0.5) * 2.0 * math.pi,
            (u[:, 5] - 0.5) * 2.0,
            1.5 + u[:, 6] * 2.0,
            30.0 + u[:, 7] * 40.0,
            30.0 + u[:, 8] * 40.0,
            torch.zeros(batch, dtype=dtype),
            (u[:, 10] - 0.5) * 0.4,
        ],
        dim=-1,
    )
    v = torch.rand(batch, 10, generator=generator, dtype=dtype)
    spreads = torch.zeros(batch, PHYSICAL_UNCERTAINTY_DIM, dtype=dtype)
    spreads[:, 0] = v[:, 0] * 500.0  # position, px
    spreads[:, 1] = 5.0 + v[:, 1] * 40.0  # velocity xx
    spreads[:, 3] = 5.0 + v[:, 2] * 40.0  # velocity yy
    spreads[:, 2] = (v[:, 3] - 0.5) * (spreads[:, 1] * spreads[:, 3]).sqrt()  # xy
    spreads[:, 4] = v[:, 4] * 0.3  # attitude, rad
    spreads[:, 5] = v[:, 5] * 0.3  # angular velocity, rad/s
    spreads[:, 6] = v[:, 6] * 0.2  # shield delay, s
    spreads[:, 7] = v[:, 7] * 6.0  # health
    spreads[:, 8] = v[:, 8] * 6.0  # power
    spreads[:, 9] = v[:, 9] * 0.05  # local log index
    # Cooldown 0.3 ticks: the two-hot (0.7, 0.3) widened by a residual with
    # zero mass and zero mean, to (0.8, 0.1, 0.1, 0).
    means[:, COOLDOWN] = 0.3 * _CONFIG.firing_cooldown / 3.0
    spreads[:, 10:14] = torch.tensor([0.1, -0.2, 0.1, 0.0], dtype=dtype)
    return means, spreads


def _wrapped_error(decoded: torch.Tensor, means: torch.Tensor) -> torch.Tensor:
    error = decoded - means
    error[:, :2] = (error[:, :2] + 32768.0).remainder(65536.0) - 32768.0
    error[:, 4] = (error[:, 4] + math.pi).remainder(2.0 * math.pi) - math.pi
    return error


class TestLayout:
    def test_the_code_is_469_wide_in_22_groups(self):
        assert SHIP_CODE_DIM == 81 + 243 + 16 + 41 + 4 * 21 + 4
        assert CODE_GROUP_DIM == len(CODE_GROUP_NAMES) == 9 + 3 + 4 + 1 + 4 + 1

    def test_every_group_of_an_encoded_code_sums_to_one(self, codec):
        means, spreads = _moments(16)
        code = codec.encode(means, spreads)
        assert code.shape == (16, SHIP_CODE_DIM)
        assert (code >= 0).all()
        for run in CODE_RUN.values():
            groups = code[:, run.start : run.stop].unflatten(-1, (run.groups, run.size))
            torch.testing.assert_close(groups.sum(-1), torch.ones(16, run.groups).double())

    def test_angular_velocity_spans_the_flight_model(self, codec):
        """Path rate plus slip-rate limit at the lowest index: about 610 deg/s."""
        assert math.degrees(codec.angular_velocity.high) == pytest.approx(609.4, abs=0.5)


class TestRoundTrip:
    def test_moments_decode_to_themselves(self, codec):
        means, spreads = _moments(64)
        decoded = codec.decode(codec.encode(means, spreads))
        assert decoded.shape == (64, BELIEF_MOMENT_DIM)
        error = _wrapped_error(decoded[:, :PHYSICAL_MEAN_DIM], means)
        torch.testing.assert_close(error, torch.zeros_like(error), atol=1e-6, rtol=0.0)
        torch.testing.assert_close(decoded[:, PHYSICAL_MEAN_DIM:], spreads, atol=1e-4, rtol=1e-5)

    def test_sharp_truth_decodes_to_zero_spread(self, codec):
        means, _ = _moments(64)
        means[:, COOLDOWN] = torch.linspace(0.0, _CONFIG.firing_cooldown, 64)
        decoded = codec.decode(codec.sharp(means))
        error = _wrapped_error(decoded[:, :PHYSICAL_MEAN_DIM], means)
        torch.testing.assert_close(error, torch.zeros_like(error), atol=1e-6, rtol=0.0)
        assert decoded[:, PHYSICAL_MEAN_DIM:].abs().max() < 1e-6

    def test_float32_round_trip_is_within_coordinate_precision(self, codec):
        means, spreads = _moments(256, torch.float32)
        decoded = codec.decode(codec.encode(means, spreads))
        error = _wrapped_error(decoded[:, :PHYSICAL_MEAN_DIM], means)
        # Position carries fp32 resolution at 65536 px.
        assert error[:, :2].abs().max() < 0.01
        assert error[:, 2:].abs().max() < 1e-3
        spread_error = (decoded[:, PHYSICAL_MEAN_DIM:] - spreads).abs()
        assert spread_error[:, 0].max() < 0.05  # position sigma, px
        assert spread_error[:, 1:4].max() < 0.02  # velocity covariance, px^2/s^2
        assert spread_error[:, 4:].max() < 1e-3

    def test_a_zero_residual_head_output_decodes_to_the_input(self, codec):
        """The floor under the log is removed before decoding."""
        means, spreads = _moments(32, torch.float32)
        code = codec.encode(means, spreads)
        decoded = codec.decode_logits(codec.baseline(code))
        torch.testing.assert_close(decoded, codec.decode(code), atol=5e-3, rtol=1e-4)

    def test_repeated_zero_residual_steps_hold_the_belief(self, codec):
        """What a hidden ship's belief does on its own, for one second of decisions."""
        means, spreads = _moments(32, torch.float32)
        state = torch.cat([means, spreads], dim=-1)
        for _ in range(30):
            code = codec.encode(state[:, :PHYSICAL_MEAN_DIM], state[:, PHYSICAL_MEAN_DIM:])
            state = codec.decode_logits(codec.baseline(code))
        error = _wrapped_error(state[:, :PHYSICAL_MEAN_DIM], means)
        assert error[:, :2].abs().max() < 0.05
        assert error[:, 2:].abs().max() < 2e-3
        assert (state[:, PHYSICAL_MEAN_DIM:] - spreads).abs().max() < 0.1


class TestDecodedPredictions:
    def test_any_head_output_decodes_inside_every_range(self, codec):
        logits = torch.randn(512, SHIP_CODE_DIM) * 5.0
        decoded = codec.decode_logits(logits)
        assert torch.isfinite(decoded).all()
        means = decoded[:, :PHYSICAL_MEAN_DIM]
        assert ((means[:, POSITION_X] >= 0) & (means[:, POSITION_X] < 65536.0)).all()
        assert ((means[:, HEALTH] >= 0) & (means[:, HEALTH] <= _CONFIG.max_health)).all()
        assert (means[:, 4].abs() <= math.pi + 1e-6).all()
        assert (decoded[:, PHYSICAL_MEAN_DIM : PHYSICAL_MEAN_DIM + 10][:, [0, 4, 5, 6]] >= 0).all()
        covariance = decoded[:, PHYSICAL_MEAN_DIM + 1 : PHYSICAL_MEAN_DIM + 4]
        determinant = covariance[:, 0] * covariance[:, 2] - covariance[:, 1].square()
        assert (determinant >= -1e-3 * covariance[:, 0] * covariance[:, 2] - 1e-6).all()

    def test_a_decoded_cooldown_re_encodes_exactly(self, codec):
        """The residual has zero mass and zero mean, so mean plus residual is the
        whole distribution and re-encoding reproduces it."""
        decoded = codec.decode_logits(torch.randn(64, SHIP_CODE_DIM))
        residual = decoded[:, PHYSICAL_MEAN_DIM:][:, COOLDOWN_RESIDUAL]
        torch.testing.assert_close(residual.sum(-1), torch.zeros(64), atol=1e-6, rtol=0.0)
        again = codec.decode(
            codec.encode(decoded[:, :PHYSICAL_MEAN_DIM], decoded[:, PHYSICAL_MEAN_DIM:])
        )
        torch.testing.assert_close(again[:, COOLDOWN], decoded[:, COOLDOWN], atol=1e-6, rtol=0)
        torch.testing.assert_close(
            again[:, PHYSICAL_MEAN_DIM:][:, COOLDOWN_RESIDUAL], residual, atol=1e-6, rtol=0
        )


class TestCompilation:
    def test_the_codec_compiles_in_one_graph(self, codec):
        def round_trip(means, spreads):
            return codec.decode_logits(codec.baseline(codec.encode(means, spreads)))

        means, spreads = _moments(8, torch.float32)
        compiled = torch.compile(round_trip, backend="eager", fullgraph=True)
        torch.testing.assert_close(compiled(means, spreads), round_trip(means, spreads))

    def test_bf16_autocast_leaves_the_code_in_float32(self, codec):
        means, spreads = _moments(8, torch.float32)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            code = codec.encode(means, spreads)
        assert code.dtype == torch.float32
        torch.testing.assert_close(code, codec.encode(means, spreads))
