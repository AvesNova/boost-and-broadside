"""The spatial attention's output gate (`YemongConfig.gate`, `spatial.SpatialBlock`): off by
default and then the model before it; elementwise and headwise gates train."""

import arm_goldens
import pytest
import torch
import yemong_cases as cases

from bnb.contracts import GameConfig
from bnb.models.yemong.model import YemongConfig
from bnb.models.yemong.spatial import GATE_BIAS

pytestmark = pytest.mark.usefixtures("global_generator")

OWN = 5
GATED = ("elementwise", "headwise")


@pytest.fixture(scope="module")
def segment(module_global_generator):
    _, trajectory = cases.rollout(cases.model(), length=12)
    return trajectory


def _gate_parameters(net) -> dict:
    return {name: p for name, p in net.named_parameters() if ".gate." in name}


def test_none_is_the_default_and_the_model_before_the_gate():
    """No gate parameters, the same parameters as an explicit "none", and C-fixed's step loop
    and unroll as recorded before the gate existed (`arm_goldens`)."""
    assert YemongConfig().gate == "none"
    default, explicit = cases.model(), cases.model(gate="none")
    assert not _gate_parameters(default)
    for (name, a), (_, b) in zip(
        default.state_dict().items(), explicit.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b), name
    arm_goldens.reproduces("c_fixed")


def test_an_unknown_gate_fails():
    with pytest.raises(ValueError, match="gate"):
        YemongConfig(gate="sideways")


@pytest.mark.parametrize("gate", GATED)
def test_the_gate_parameters_start_open(gate):
    net = cases.model(gate=gate)
    config = net.config
    layers = config.blocks * config.spatial_per_block
    outputs = config.width if gate == "elementwise" else config.heads
    gates = _gate_parameters(net)
    assert sum(p.numel() for p in gates.values()) == layers * outputs * (config.width + 1)
    for name, p in gates.items():
        assert torch.equal(p, torch.full_like(p, GATE_BIAS if name.endswith("bias") else 0.0))


@pytest.mark.parametrize("gate", GATED)
def test_the_gated_unroll_equals_a_loop_of_steps(segment, gate):
    net = cases.double(cases.model(gate=gate))
    with torch.no_grad():
        for parameters in _gate_parameters(net).values():  # gates that differ per input
            parameters.normal_(0.0, 0.3)
    obs = cases.cast(segment.obs, torch.float64)
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    with torch.no_grad():
        decisions, _ = cases.step_loop(net, net.step, carry, obs, cases.joint_actions(segment))
        unrolled = net.unroll(carry, obs, decisions.replay, cases.resets(obs))
    for name in ("policy_logits", "values"):
        error = cases.scale_error(getattr(unrolled, name), getattr(decisions, name))
        assert error <= 1e-9, (name, error)


@pytest.mark.parametrize("gate", GATED)
def test_a_gated_model_trains_a_few_steps(segment, gate):
    net = cases.model(gate=gate)
    obs = segment.obs
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    with torch.no_grad():
        decisions, _ = cases.step_loop(net, net.step, carry, obs, cases.joint_actions(segment))
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    gates = _gate_parameters(net)
    losses = []
    for _ in range(4):
        optimizer.zero_grad()
        unrolled = net.unroll(carry, obs, decisions.replay, cases.resets(obs))
        loss = unrolled.policy_logits.square().mean() + unrolled.values.square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        for name, parameters in gates.items():
            assert torch.isfinite(parameters.grad).all(), name
            assert parameters.grad.abs().sum() > 0, name
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]
    assert all(torch.isfinite(p).all() for p in net.parameters())


def test_the_set_line_builds_the_gate():
    from bnb.cli import parse_setting, with_settings
    from bnb.runtime.train import TrainConfig, model_build

    config = with_settings(
        TrainConfig(game=GameConfig(5, 10, 2600.0)), [parse_setting("yemong.gate=elementwise")]
    )
    assert model_build(config).config.gate == "elementwise"
