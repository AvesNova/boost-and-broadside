"""The trunk's order switch and the rotary read-out (temporal-first and per-block read-out
briefs): spatial_first is today's model, temporal_first runs each block's temporal sublayer
first and still unrolls as a loop of steps, and each block's read-out drives the hidden
ships' rotary tables in that block's spatial sublayers."""

import dataclasses

import pytest
import torch
import yemong_cases as cases

from bnb.contracts.obs import SHIP
from bnb.game import api
from bnb.models.yemong import network
from bnb.models.yemong.inputs import truth
from bnb.models.yemong.model import Yemong, YemongConfig, yemong_build
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


def test_the_defaults_are_spatial_first_without_a_read_out():
    """spatial_first is today's model: the same parameters, and (checked once against the
    commit before the switch, b11e1765) a rollout, its step loop, final carry and unroll
    identical bit for bit."""
    config = YemongConfig()
    assert config.order == "spatial_first" and not config.rotary_readout
    default = cases.model()
    assert default.rotary_readout is None
    explicit = cases.model(order="spatial_first", rotary_readout=False)
    for (name, a), (_, b) in zip(
        default.state_dict().items(), explicit.state_dict().items(), strict=True
    ):
        assert torch.equal(a, b), name


def test_the_read_out_needs_temporal_first_and_rotary():
    with pytest.raises(ValueError, match="rotary_readout"):
        YemongConfig(rotary_readout=True)
    with pytest.raises(ValueError, match="rotary_readout"):
        YemongConfig(order="temporal_first", rotary_readout=True, rotary=False)
    with pytest.raises(ValueError, match="order"):
        YemongConfig(order="sideways")


def _initialised(**settings) -> dict:
    """The state dict of a freshly constructed Yemong, its constructor's draws seeded."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        build = yemong_build(YemongConfig(**settings), api.rules_of(cases.SQUAD), api.ship_types())
        return Yemong(build).state_dict()


def test_the_read_out_adds_only_one_head_per_block_and_leaves_the_rest_initialised_as_before():
    plain = _initialised(order="temporal_first")
    readout = _initialised(order="temporal_first", rotary_readout=True)
    added = set(readout) - set(plain)
    blocks = YemongConfig().blocks
    assert added == {
        f"rotary_readout.{block}.{name}" for block in range(blocks) for name in ("weight", "bias")
    }
    for block in range(blocks):
        weight = readout[f"rotary_readout.{block}.weight"]
        assert weight.shape == (2 * YemongConfig().rotary_pairs, 128)
    assert not torch.equal(readout["rotary_readout.0.weight"], readout["rotary_readout.1.weight"])
    assert all(torch.equal(plain[name], readout[name]) for name in plain)


@pytest.mark.parametrize(
    "settings",
    [
        dict(order="spatial_first"),
        dict(order="temporal_first"),
        dict(order="temporal_first", rotary_readout=True),
        dict(order="temporal_first", rotary_readout=True, forecast_feedback=False),
    ],
    ids=["spatial_first", "C", "temporal_readout", "D"],
)
def test_unroll_equals_a_loop_of_steps(segment, settings):
    net = cases.double(cases.model(**settings))
    decisions, unrolled = _loop_and_unroll(net, segment)
    _compare(unrolled, decisions)
    assert (unrolled.rotary_readout is not None) == net.config.rotary_readout
    if net.config.rotary_readout:  # every block's read-out, as the steps' inspection has it
        obs = cases.cast(segment.obs, torch.float64)
        carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
        joint, reset = cases.joint_actions(segment), cases.resets(obs)
        state, inspected, replays = carry, [], []
        with torch.no_grad():
            for t in range(obs.ships.shape[0]):
                view = cases.Obs(
                    ships=obs.ships[t],
                    fields=obs.fields[t],
                    globals=obs.globals[t],
                    zones=obs.zones[t],
                )
                decision, outputs = net.inspect(state, view, reset[t])
                state = dataclasses.replace(decision.carry, previous_action=joint[t])
                inspected.append(outputs.rotary_readout[0])
                replays.append(decision.replay)
            replay = torch.utils._pytree.tree_map(lambda *x: torch.stack(x), *replays)
            unrolled = net.unroll(carry, obs, replay, reset)
        assert unrolled.rotary_readout.shape[-3] == net.config.blocks
        error = cases.scale_error(unrolled.rotary_readout, torch.stack(inspected))
        assert error <= 1e-9


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
    expected = spatial_first if order == "spatial_first" else temporal_first
    sublayers = len(expected)
    assert len(log) % sublayers == 0 and len(log) > sublayers
    for start in range(0, len(log), sublayers):  # every step and the unroll
        assert all(a is b for a, b in zip(log[start : start + sublayers], expected, strict=True))


def test_the_read_out_sets_the_hidden_ships_tables_and_visible_ships_keep_theirs(
    segment, monkeypatch
):
    """Every spatial sublayer's token tables: a visible ship's are its true tables, a hidden
    ship's its block's read-out pairs, the global token's the map centre's; the map tables
    are the exact ones."""
    net = cases.double(cases.model(order="temporal_first", rotary_readout=True))
    seen = []
    original = SpatialBlock.forward

    def recording(self, tokens, map_memory, token_tables, map_tables):
        seen.append((token_tables, map_tables))
        return original(self, tokens, map_memory, token_tables, map_tables)

    monkeypatch.setattr(SpatialBlock, "forward", recording)
    obs = cases.cast(segment.obs, torch.float64)
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    with torch.no_grad():
        decisions, _ = cases.step_loop(net, net.step, carry, obs, cases.joint_actions(segment))
        seen.clear()
        unrolled = net.unroll(carry, obs, decisions.replay, cases.resets(obs))
        tokens = net.tokens(obs, decisions.replay)
        _, _, belief_tables, exact_map = net._encode(tokens)
    assert len(seen) == net.config.blocks * net.config.spatial_per_block
    ships = obs.ships
    hidden = ships[..., SHIP["visible"]] < 0.5
    assert hidden.any() and (~hidden).any()
    offset = decisions.replay.offset[:, :, None, :]
    observed = truth(ships, decisions.replay.offset)
    true = net.phasors(observed[..., 0:2], observed[..., 4])  # (L, R, N, 2, pairs)
    readouts = unrolled.rotary_readout  # (L, R, N, blocks, 2, pairs)
    blocks, per_block = net.config.blocks, net.config.spatial_per_block
    assert readouts.shape == (*true.shape[:-2], blocks, *true.shape[-2:])
    assert not torch.equal(readouts[..., 0, :, :], readouts[..., 1, :, :])
    for index, (token_tables, map_tables) in enumerate(seen):
        readout = readouts[..., index // per_block, :, :]
        assert all(torch.equal(a, b) for a, b in zip(map_tables, exact_map, strict=True))
        cos, sin = (table[..., 0, :] for table in token_tables)  # (L, R, Q, pairs)
        pairs = torch.stack([cos[..., :-1, :], sin[..., :-1, :]], -2)
        torch.testing.assert_close(pairs[~hidden], true[~hidden], rtol=0, atol=1e-12)
        assert torch.equal(pairs[hidden], readout[hidden])
        centre = net.phasors(offset[..., 0, :][..., None, :], torch.zeros_like(offset[..., 0]))
        torch.testing.assert_close(
            torch.stack([cos[..., -1:, :], sin[..., -1:, :]], -2), centre, rtol=0, atol=1e-12
        )
        # the belief's tables are what the hidden rows replaced
        assert torch.equal(belief_tables[0][..., -1:, :, :], token_tables[0][..., -1:, :, :])


def test_the_read_out_enters_the_tables_detached(segment):
    """The read-out trains on its own loss only: no gradient reaches it through attention."""
    net = cases.model(order="temporal_first", rotary_readout=True).train()
    obs = segment.obs
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    with torch.no_grad():
        decisions, _ = cases.step_loop(net, net.step, carry, obs, cases.joint_actions(segment))
    unrolled = net.unroll(carry, obs, decisions.replay, cases.resets(obs))
    unrolled.policy_logits.sum().backward()
    for head in net.rotary_readout:
        assert head.weight.grad is None or not head.weight.grad.any()
    assert net.trunk.blocks[0].temporal.output.weight.grad.abs().sum() > 0


def test_the_ablated_views_read_the_identity_rotation_with_the_read_out(segment):
    """`Ablations.rotary` makes every table the identity on its views, the read-out's too:
    an ablated view decides as the model with rotary off would, but with the read-out's
    head unused."""
    from bnb.models.yemong.model import Ablations

    net = cases.double(cases.model(order="temporal_first", rotary_readout=True))
    obs = cases.cast(segment.obs, torch.float64)
    views = obs.ships.shape[1]
    carry = net.initial_carry(views, OWN, 10)
    ablate = torch.zeros(views, dtype=torch.bool)
    ablate[0] = True
    ablations = Ablations(rotary=ablate, forecast=torch.zeros_like(ablate))
    seen = []
    original = SpatialBlock.forward

    def recording(self, tokens, map_memory, token_tables, map_tables):
        seen.append(token_tables)
        return original(self, tokens, map_memory, token_tables, map_tables)

    with torch.no_grad(), pytest.MonkeyPatch.context() as patch:
        patch.setattr(SpatialBlock, "forward", recording)
        view = cases.Obs(
            ships=obs.ships[0], fields=obs.fields[0], globals=obs.globals[0], zones=obs.zones[0]
        )
        net.step(carry, view, cases.resets(obs)[0], ablations)
    for cos, sin in seen:
        assert (cos[0] == 1).all() and (sin[0] == 0).all()
        assert not (cos[1:] == 1).all()


def test_a_policy_file_keeps_the_order_and_the_read_out(tmp_path):
    from bnb.runs import checkpoint

    net = cases.model(order="temporal_first", rotary_readout=True, forecast_feedback=False)
    checkpoint.save_policy(tmp_path / "d", net)
    loaded = checkpoint.load_policy(tmp_path / "d", torch.device("cpu"))
    assert loaded.config == net.config
    assert loaded.config.order == "temporal_first" and loaded.config.rotary_readout
    for loaded_head, head in zip(loaded.rotary_readout, net.rotary_readout, strict=True):
        assert torch.equal(loaded_head.weight, head.weight)
