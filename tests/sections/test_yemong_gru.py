"""The GRU temporal sublayer (`YemongConfig.temporal_block`, `temporal.GRUBlock`, arm A16):
Griffin's block by default and then the model before the option; the GRU is torch's GRU
cell, clears its state at every episode start, unrolls as a loop of steps, trains, saves
and loads."""

import arm_goldens
import pytest
import torch
import yemong_cases as cases

from bnb.contracts import GameConfig
from bnb.models.yemong.model import YemongConfig
from bnb.models.yemong.temporal import GRUBlock, TemporalBlock

pytestmark = pytest.mark.usefixtures("global_generator")

OWN = 5
WIDTH = 16


@pytest.fixture(scope="module")
def segment(module_global_generator):
    _, trajectory = cases.rollout(cases.model(), length=24)
    return trajectory


def _block(seed: int = 0) -> GRUBlock:
    torch.manual_seed(seed)
    block = GRUBlock(WIDTH).double()
    with torch.no_grad():
        block.input_norm.weight.normal_(1.0, 0.2)
    return block


def test_griffin_is_the_default_and_the_model_before_the_option():
    """The same parameters as an explicit "griffin", a Griffin carry, and C-fixed's step loop
    and unroll as recorded before the option existed (`arm_goldens`)."""
    assert YemongConfig().temporal_block == "griffin"
    default, explicit = cases.model(), cases.model(temporal_block="griffin")
    for (name, a), (_, b) in zip(
        default.state_dict().items(), explicit.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b), name
    assert all(isinstance(block.temporal, TemporalBlock) for block in default.trunk.blocks)
    carry = default.initial_carry(3, OWN, 10)
    assert carry.convolution_buffer.shape == (3, 2, 11, 3, default.config.width)
    arm_goldens.reproduces("c_fixed")


def test_an_unknown_temporal_block_fails():
    with pytest.raises(ValueError, match="temporal_block"):
        YemongConfig(temporal_block="lstm")


def test_the_gru_build_has_the_gru_and_an_empty_buffer():
    """Every temporal sublayer a GRUBlock, every other parameter as the default build's, and
    a carry buffer of no inputs."""
    net = cases.model(temporal_block="gru")
    default = cases.model()
    assert all(isinstance(block.temporal, GRUBlock) for block in net.trunk.blocks)
    others = {name for name in net.state_dict() if ".temporal." not in name}
    assert others == {name for name in default.state_dict() if ".temporal." not in name}
    carry = net.initial_carry(3, OWN, 10)
    assert carry.state.shape == (3, 2, 11, net.config.width)
    assert carry.convolution_buffer.shape == (3, 2, 11, 0, net.config.width)


def test_the_parameter_counts():
    """At width D: Griffin's sublayer 17 D^2 + 10 D, the GRU's 19 D^2 + 8 D (the gated MLP's
    12 D^2 and the norms' 2 D in both)."""
    width = 128
    griffin = sum(p.numel() for p in TemporalBlock(width).parameters())
    gru = sum(p.numel() for p in GRUBlock(width).parameters())
    assert griffin == 17 * width**2 + 10 * width
    assert gru == 19 * width**2 + 8 * width


def test_the_step_is_torchs_gru_cell():
    block = _block()
    cell = torch.nn.GRUCell(WIDTH, WIDTH).double()
    with torch.no_grad():
        cell.weight_ih.copy_(block.input_gates.weight)
        cell.bias_ih.copy_(block.input_gates.bias)
        cell.weight_hh.copy_(block.hidden_gates.weight)
        cell.bias_hh.copy_(block.hidden_gates.bias)
    tokens = torch.randn(7, WIDTH, dtype=torch.float64)
    state = torch.randn(7, WIDTH, dtype=torch.float64).tanh()
    buffer = torch.zeros(7, 0, WIDTH, dtype=torch.float64)
    with torch.no_grad():
        _, new_state, new_buffer = block.step(tokens, state, buffer)
        expected = cell(block.input_norm(tokens), state)
    torch.testing.assert_close(new_state, expected, rtol=1e-12, atol=1e-12)
    assert new_buffer is buffer


def test_the_sequence_is_a_loop_of_steps_cleared_at_every_start():
    """Starts inside the segment clear the state; the first start is the caller's."""
    block = _block()
    length, rows = 12, 6
    tokens = torch.randn(length, rows, WIDTH, dtype=torch.float64)
    state = torch.randn(rows, WIDTH, dtype=torch.float64).tanh()
    buffer = torch.zeros(rows, 0, WIDTH, dtype=torch.float64)
    starts = torch.zeros(length, rows, dtype=torch.bool)
    starts[0] = True  # ignored
    starts[3, 1] = starts[7, 1] = starts[5, 4] = starts[11, 0] = True
    with torch.no_grad():
        outputs, final, _ = block.sequence(tokens, state, buffer, starts)
        looped, carried = [], state
        for time in range(length):
            if time:
                carried = torch.where(starts[time][:, None], 0.0, carried)
            out, carried, _ = block.step(tokens[time], carried, buffer)
            looped.append(out)
    torch.testing.assert_close(outputs, torch.stack(looped), rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(final, carried, rtol=1e-12, atol=1e-12)


def test_a_start_forgets_everything_before_it():
    block = _block()
    length, rows = 8, 2
    tokens = torch.randn(length, rows, WIDTH, dtype=torch.float64)
    changed = tokens.clone()
    changed[:4, 0] = torch.randn(4, WIDTH, dtype=torch.float64)
    starts = torch.zeros(length, rows, dtype=torch.bool)
    starts[4] = True
    state = torch.randn(rows, WIDTH, dtype=torch.float64).tanh()
    buffer = torch.zeros(rows, 0, WIDTH, dtype=torch.float64)
    with torch.no_grad():
        before, _, _ = block.sequence(tokens, state, buffer, starts)
        after, _, _ = block.sequence(changed, state, buffer, starts)
    assert not torch.allclose(before[:4, 0], after[:4, 0])
    torch.testing.assert_close(before[4:], after[4:], rtol=0.0, atol=0.0)


@pytest.mark.parametrize("order", ["spatial_first", "temporal_first"])
def test_the_gru_unroll_equals_a_loop_of_steps(segment, order):
    net = cases.double(cases.model(temporal_block="gru", order=order))
    obs = cases.cast(segment.obs, torch.float64)
    reset = cases.resets(obs)
    assert reset[1:].any(), "the segment should hold an episode start"
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    with torch.no_grad():
        decisions, _ = cases.step_loop(net, net.step, carry, obs, cases.joint_actions(segment))
        unrolled = net.unroll(carry, obs, decisions.replay, reset)
    for name in ("policy_logits", "values"):
        error = cases.scale_error(getattr(unrolled, name), getattr(decisions, name))
        assert error <= 1e-9, (name, error)


def test_the_gru_model_trains_a_few_steps(segment):
    net = cases.model(temporal_block="gru", order="temporal_first")
    obs = segment.obs
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    with torch.no_grad():
        decisions, _ = cases.step_loop(net, net.step, carry, obs, cases.joint_actions(segment))
    optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
    recurrent = {n: p for n, p in net.named_parameters() if "_gates." in n}
    assert recurrent
    losses = []
    for _ in range(4):
        optimizer.zero_grad()
        unrolled = net.unroll(carry, obs, decisions.replay, cases.resets(obs))
        loss = unrolled.policy_logits.square().mean() + unrolled.values.square().mean()
        loss.backward()
        assert torch.isfinite(loss)
        for name, parameters in recurrent.items():
            assert torch.isfinite(parameters.grad).all(), name
            assert parameters.grad.abs().sum() > 0, name
        optimizer.step()
        losses.append(loss.item())
    assert losses[-1] < losses[0]


def test_the_gru_runs_under_bfloat16_autocast(segment):
    """The model's 16-bit matmuls (`YemongConfig.bfloat16_matmuls`): the recurrent state
    stays in the model's dtype, and the unroll stays close to the float32 one."""
    plain = cases.model(temporal_block="gru")
    mixed = cases.model(temporal_block="gru", bfloat16_matmuls=True)
    mixed.load_state_dict(plain.state_dict())
    obs = segment.obs
    carry = plain.initial_carry(obs.ships.shape[1], OWN, 10)
    with torch.no_grad():
        decisions, final = cases.step_loop(
            mixed, mixed.step, carry, obs, cases.joint_actions(segment)
        )
        assert final.state.dtype == torch.float32
        want = plain.unroll(carry, obs, decisions.replay, cases.resets(obs))
        got = mixed.unroll(carry, obs, decisions.replay, cases.resets(obs))
    assert cases.scale_error(got.policy_logits, want.policy_logits) < 0.1


def test_a_policy_file_keeps_the_gru(tmp_path, segment):
    from bnb.runs import checkpoint

    net = cases.model(temporal_block="gru", order="temporal_first")
    checkpoint.save_policy(tmp_path / "d", net)
    loaded = checkpoint.load_policy(tmp_path / "d", torch.device("cpu"))
    assert loaded.config.temporal_block == "gru"
    for (name, a), (_, b) in zip(
        net.state_dict().items(), loaded.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b), name


def test_the_set_line_builds_the_gru():
    from bnb.cli import parse_setting, with_settings
    from bnb.runtime.train import TrainConfig, model_build

    config = with_settings(
        TrainConfig(game=GameConfig(5, 10, 2600.0)), [parse_setting("yemong.temporal_block=gru")]
    )
    assert model_build(config).config.temporal_block == "gru"
