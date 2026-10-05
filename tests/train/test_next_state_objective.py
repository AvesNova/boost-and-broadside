"""The next-state head: residual logits on the code it read, cross-entropy on truth.

The head starts as "nothing changes" (§8.2): its last layer is zero, so its
logits are the floored log of the input code, and decoding them returns the
moments the code was built from. Training is per-group cross-entropy against
the exact code of the true next state, which is proper: it is minimised by
predicting the target itself.
"""

import math

import pytest
import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.env.frontline import frontline_ship_config
from boost_and_broadside.models.yemong.policy import NextStateHead
from boost_and_broadside.train.rl.physical_belief import (
    BELIEF_MOMENT_DIM,
    PHYSICAL_MEAN_DIM,
    PHYSICAL_UNCERTAINTY_DIM,
)
from boost_and_broadside.train.rl.ship_codes import (
    CODE_GROUP_DIM,
    CODE_RUNS,
    SHIP_CODE_DIM,
    ShipStateCodec,
)

_CONFIG = frontline_ship_config(ShipConfig())


@pytest.fixture
def codec() -> ShipStateCodec:
    return ShipStateCodec.from_ship_config(_CONFIG)


def _means(batch: int) -> torch.Tensor:
    generator = torch.Generator().manual_seed(3)
    uniform = torch.rand(batch, PHYSICAL_MEAN_DIM, generator=generator)
    return torch.stack(
        [
            uniform[:, 0] * 65536.0,
            uniform[:, 1] * 65536.0,
            (uniform[:, 2] - 0.5) * 300.0,
            (uniform[:, 3] - 0.5) * 300.0,
            (uniform[:, 4] - 0.5) * 2.0 * math.pi,
            (uniform[:, 5] - 0.5) * 4.0,
            uniform[:, 6] * 5.0,
            10.0 + uniform[:, 7] * 80.0,
            10.0 + uniform[:, 8] * 80.0,
            uniform[:, 9] * 0.1,
            (uniform[:, 10] - 0.5) * 1.0,
        ],
        dim=-1,
    )


def _head(d_model: int = 16) -> NextStateHead:
    head = NextStateHead(d_model)
    torch.nn.init.zeros_([m for m in head.net if isinstance(m, torch.nn.Linear)][-1].weight)
    torch.nn.init.zeros_([m for m in head.net if isinstance(m, torch.nn.Linear)][-1].bias)
    return head


class TestHead:
    def test_the_head_is_exactly_as_wide_as_the_code(self):
        assert NextStateHead(16).net[-1].out_features == SHIP_CODE_DIM

    def test_a_zero_residual_states_the_input_back(self, codec):
        """Decoding the untrained head's output returns the belief it read."""
        means = _means(32)
        spreads = torch.zeros(32, PHYSICAL_UNCERTAINTY_DIM)
        spreads[:, 0] = torch.linspace(0.0, 400.0, 32)  # position sigma, px
        spreads[:, 7] = torch.linspace(0.0, 10.0, 32)  # health sigma
        code = codec.encode(means, spreads)
        logits = _head()(torch.randn(32, 16), code)
        decoded = codec.decode_logits(logits)
        assert decoded.shape[-1] == BELIEF_MOMENT_DIM
        torch.testing.assert_close(decoded, codec.decode(code), atol=5e-3, rtol=1e-4)

    def test_the_policy_starts_the_residual_at_zero(self):
        from boost_and_broadside.config import ModelConfig
        from boost_and_broadside.models.yemong.policy import YemongPolicy
        from boost_and_broadside.train.rl.features import build_standard_coordinator

        policy = YemongPolicy(
            ModelConfig(d_model=16, n_heads=2, n_yemong_blocks=1),
            build_standard_coordinator(_CONFIG),
            num_value_components=5,
            num_ships=2,
            global_value_k=(),
            ship_config=_CONFIG,
        )
        final = [m for m in policy.next_state_head.net if isinstance(m, torch.nn.Linear)][-1]
        assert final.weight.abs().max() == 0.0
        assert final.bias.abs().max() == 0.0


class TestObjective:
    def test_cross_entropy_is_one_value_per_group(self, codec):
        means = _means(4)
        target = codec.sharp(means)
        losses = codec.cross_entropy(torch.zeros(4, SHIP_CODE_DIM), target)
        assert losses.shape == (4, CODE_GROUP_DIM)
        # Uniform logits cost log(group size) in every group.
        expected = torch.tensor(
            [math.log(run.size) for run in CODE_RUNS for _ in range(run.groups)]
        )
        torch.testing.assert_close(losses, expected.expand(4, -1), rtol=1e-5, atol=1e-5)

    def test_it_is_minimised_by_predicting_the_target(self, codec):
        """Proper scoring: the gradient at the target's own log is zero."""
        target = codec.sharp(_means(8))
        logits = torch.log(target + 1e-12).requires_grad_(True)
        codec.cross_entropy(logits, target).sum().backward()
        assert logits.grad.abs().max() < 1e-5

    def test_every_group_is_differentiable_through_the_head(self, codec):
        head = NextStateHead(16)
        means = _means(6)
        code = codec.sharp(means)
        logits = head(torch.randn(6, 16), code)
        target = codec.sharp(means + 1.0)
        codec.cross_entropy(logits, target).sum().backward()
        final = [m for m in head.net if isinstance(m, torch.nn.Linear)][-1]
        assert (final.weight.grad.abs().sum(-1) > 0).all(), "every output unit gets a gradient"
