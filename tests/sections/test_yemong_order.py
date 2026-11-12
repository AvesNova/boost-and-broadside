"""The trunk's order switch (temporal-first brief): spatial_first is today's model,
temporal_first runs each block's temporal sublayer first and still unrolls as a loop of
steps."""

import pytest
import torch
import yemong_cases as cases

from bnb.models.yemong import network
from bnb.models.yemong.model import YemongConfig
from bnb.models.yemong.spatial import SpatialBlock
from bnb.models.yemong.temporal import TemporalBlock

pytestmark = pytest.mark.usefixtures("global_generator")

OWN = 5


@pytest.fixture(scope="module")
def segment(module_global_generator):
    """A real fogged rollout, played by today's model (the arms below replay its
    observations; only their decisions differ)."""
    _, trajectory = cases.rollout(cases.model(), length=48)
    return trajectory


def _loop_and_unroll(net, trajectory):
    obs = cases.cast(trajectory.obs, torch.float64)
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    with torch.no_grad():
        decisions, _ = cases.step_loop(net, net.step, carry, obs, cases.joint_actions(trajectory))
        unrolled = net.unroll(carry, obs, decisions.replay, cases.resets(obs))
    return decisions, unrolled


def _compare(unrolled, decisions, tolerance=1e-9):
    pairs = (
        ("policy_logits", unrolled.policy_logits),
        ("enemy_action_logits", unrolled.enemy_action_logits[..., OWN:, :]),
        ("next_state_logits", unrolled.next_state_logits[..., OWN:, :]),
        ("values", unrolled.values),
    )
    for name, actual in pairs:
        error = cases.scale_error(actual, getattr(decisions, name))
        assert error <= tolerance, (name, error)


def test_the_defaults_are_spatial_first():
    """spatial_first is today's model: the same parameters, and (checked once against the
    commit before the switch, b11e1765) a rollout, its step loop, final carry and unroll
    identical bit for bit."""
    assert YemongConfig().order == "spatial_first"
    default = cases.model()
    explicit = cases.model(order="spatial_first")
    for (name, a), (_, b) in zip(
        default.state_dict().items(), explicit.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b), name


def test_an_unknown_order_fails():
    with pytest.raises(ValueError, match="order"):
        YemongConfig(order="sideways")


@pytest.mark.parametrize(
    "settings",
    [
        dict(order="spatial_first"),
        dict(order="temporal_first"),
        dict(order="temporal_first", forecast_feedback=False),
        dict(order="spatial_only"),
    ],
    ids=["spatial_first", "C", "C_no_feedback", "spatial_only"],
)
def test_unroll_equals_a_loop_of_steps(segment, settings):
    net = cases.double(cases.model(**settings))
    decisions, unrolled = _loop_and_unroll(net, segment)
    _compare(unrolled, decisions)


def _sublayer_log(monkeypatch):
    """Patch the sublayers to log their calls (step, sequence and the spatial forward)."""
    log = []
    for cls, names in ((TemporalBlock, ("step", "sequence")), (SpatialBlock, ("forward",))):
        for name in names:
            original = getattr(cls, name)

            def logged(self, *args, original=original, **kwargs):
                log.append(self)
                return original(self, *args, **kwargs)

            monkeypatch.setattr(cls, name, logged)
    return log


@pytest.mark.parametrize("order", network.ORDERS)
def test_the_order_is_the_blocks_sublayer_order(segment, monkeypatch, order):
    net = cases.double(cases.model(order=order))
    log = _sublayer_log(monkeypatch)
    _loop_and_unroll(net, segment)
    block = [*net.trunk.blocks]
    spatial_first = [*block[0].spatial, block[0].temporal, *block[1].spatial, block[1].temporal]
    temporal_first = [block[0].temporal, *block[0].spatial, block[1].temporal, *block[1].spatial]
    spatial_only = [*block[0].spatial, *block[1].spatial]
    expected = dict(
        spatial_first=spatial_first, temporal_first=temporal_first, spatial_only=spatial_only
    )[order]
    sublayers = len(expected)
    assert len(log) % sublayers == 0 and len(log) > sublayers
    for start in range(0, len(log), sublayers):  # every step and the unroll
        assert all(a is b for a, b in zip(log[start : start + sublayers], expected, strict=True))


def test_spatial_only_has_no_temporal_layers_and_no_trunk_carry():
    """The ablation without temporal layers: the spatial sublayers and every other
    parameter as the default build's, none of the temporal ones, and a trunk carry of no
    blocks (the beliefs remain)."""
    net = cases.model(order="spatial_only")
    default = cases.model()
    names = set(net.state_dict())
    assert names == {name for name in default.state_dict() if ".temporal." not in name}
    assert all(block.temporal is None for block in net.trunk.blocks)
    carry = net.initial_carry(3, OWN, 10)
    assert carry.state.shape == (3, 0, 11, net.config.width)
    assert carry.convolution_buffer.shape == (3, 0, 11, 3, net.config.width)


def test_a_policy_file_keeps_the_order(tmp_path):
    from bnb.runs import checkpoint

    net = cases.model(order="temporal_first", forecast_feedback=False)
    checkpoint.save_policy(tmp_path / "d", net)
    loaded = checkpoint.load_policy(tmp_path / "d", torch.device("cpu"))
    assert loaded.config == net.config
    assert loaded.config.order == "temporal_first"
