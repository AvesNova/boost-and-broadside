"""Yemong's inference-time ablations (`Ablations`): per view, the rotary rotation as the
identity and the forecast not fed back. Off, a step is today's step bit for bit; on, the
ablated views compute what a model with the feature off computes, and the other views what
they computed before."""

import pytest
import torch
import yemong_cases as cases
from torch.utils import _pytree

import bnb.models.yemong.model as model_module
from bnb.models.yemong.model import Ablations

pytestmark = pytest.mark.usefixtures("global_generator")

OWN = 5  # squad: own ships per view


@pytest.fixture(scope="module")
def segment(module_global_generator):
    """A real squad-sized rollout, half the environments fogged, with several resets."""
    net = cases.model()
    _, trajectory = cases.rollout(net)
    return net, trajectory


def _ablations(views: int, rotary=(), forecast=()) -> Ablations:
    def mask(rows):
        out = torch.zeros(views, dtype=torch.bool)
        out[list(rows)] = True
        return out

    return Ablations(rotary=mask(rotary), forecast=mask(forecast))


def _loop(net, trajectory, ablations=None):
    """The step loop over the segment (decisions stacked over time, the final carry), from
    a fixed offset draw."""
    obs = trajectory.obs
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    with torch.no_grad():
        return cases.step_loop(
            net,
            lambda c, v, r, k: net.step(c, v, r, k, ablations),
            carry,
            obs,
            cases.joint_actions(trajectory),
        )


def _assert_rows_equal(first, second, rows, axis):
    """Every leaf equal on `rows` of its view axis (the stacked decisions' second, the
    carry's first)."""
    for one, two in zip(_pytree.tree_leaves(first), _pytree.tree_leaves(second), strict=True):
        index = (slice(None),) * axis + (rows,)
        assert torch.equal(one[index], two[index])


def _assert_trees_equal(first, second):
    for one, two in zip(_pytree.tree_leaves(first), _pytree.tree_leaves(second), strict=True):
        assert torch.equal(one, two)


def test_no_ablation_is_todays_step_bit_for_bit(segment):
    net, trajectory = segment
    views = trajectory.obs.ships.shape[1]
    _assert_trees_equal(_loop(net, trajectory), _loop(net, trajectory, _ablations(views)))


def test_rotary_ablated_views_see_zero_angles_and_the_others_are_untouched(segment, monkeypatch):
    net, trajectory = segment
    views = trajectory.obs.ships.shape[1]
    ablated, kept = [0, 3, 4], [1, 2, 5, 6, 7]
    plain = _loop(net, trajectory)
    mixed = _loop(net, trajectory, _ablations(views, rotary=ablated))

    real = model_module.rotary_tables

    def zero_angles(position, heading, *frequencies):
        cos, sin = real(position, heading, *frequencies)
        return torch.ones_like(cos), torch.zeros_like(sin)

    monkeypatch.setattr(model_module, "rotary_tables", zero_angles)
    identity = _loop(net, trajectory)
    _assert_rows_equal(mixed[0], identity[0], ablated, 1)
    _assert_rows_equal(mixed[1], identity[1], ablated, 0)
    _assert_rows_equal(mixed[0], plain[0], kept, 1)
    _assert_rows_equal(mixed[1], plain[1], kept, 0)
    assert not torch.equal(plain[0].policy_logits[:, ablated], mixed[0].policy_logits[:, ablated])


def test_forecast_ablated_views_play_as_the_switch_off_and_the_others_are_untouched(segment):
    net, trajectory = segment
    views = trajectory.obs.ships.shape[1]
    ablated, kept = [1, 2, 6], [0, 3, 4, 5, 7]
    switched = cases.model(forecast_feedback=False)
    switched.load_state_dict(net.state_dict())
    plain = _loop(net, trajectory)
    mixed = _loop(net, trajectory, _ablations(views, forecast=ablated))
    off = _loop(switched, trajectory)
    _assert_rows_equal(mixed[0], off[0], ablated, 1)
    _assert_rows_equal(mixed[1], off[1], ablated, 0)
    _assert_rows_equal(mixed[0], plain[0], kept, 1)
    _assert_rows_equal(mixed[1], plain[1], kept, 0)
    replay = mixed[0].replay
    assert not torch.equal(plain[0].replay.spreads[:, ablated], replay.spreads[:, ablated])
    assert (replay.spreads[:, ablated] == 0).all()


def test_both_ablations_on_one_view_compose(segment, monkeypatch):
    net, trajectory = segment
    views = trajectory.obs.ships.shape[1]
    switched = cases.model(forecast_feedback=False)
    switched.load_state_dict(net.state_dict())
    both = _loop(net, trajectory, _ablations(views, rotary=[2], forecast=[2]))
    real = model_module.rotary_tables
    monkeypatch.setattr(
        model_module,
        "rotary_tables",
        lambda *a: tuple(f(t) for f, t in zip((torch.ones_like, torch.zeros_like), real(*a))),
    )
    reference = _loop(switched, trajectory)
    _assert_rows_equal(both[0], reference[0], [2], 1)
    _assert_rows_equal(both[1], reference[1], [2], 0)


# ------------------------------------------------------------------ the training switch


def _switched(net, **settings):
    """The same weights under other YemongConfig settings."""
    other = cases.model(**settings)
    other.load_state_dict(net.state_dict())
    return other


def _zero_angles(monkeypatch):
    real = model_module.rotary_tables

    def zero_angles(position, heading, *frequencies):
        cos, sin = real(position, heading, *frequencies)
        return torch.ones_like(cos), torch.zeros_like(sin)

    monkeypatch.setattr(model_module, "rotary_tables", zero_angles)


def _unrolled(net, trajectory, decisions):
    obs = trajectory.obs
    torch.manual_seed(11)
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    with torch.no_grad():
        return net.unroll(carry, obs, decisions.replay, cases.resets(obs))


def test_rotary_on_is_the_default_and_todays_model(segment):
    net, trajectory = segment
    assert model_module.YemongConfig().rotary
    on = _switched(net, rotary=True)
    _assert_trees_equal(_loop(net, trajectory), _loop(on, trajectory))
    off = _loop(_switched(net, rotary=False), trajectory)
    assert not torch.equal(off[0].policy_logits, _loop(net, trajectory)[0].policy_logits)


def test_rotary_off_is_zero_angles_in_step_and_unroll(segment, monkeypatch):
    net, trajectory = segment
    views = trajectory.obs.ships.shape[1]
    off = _switched(net, rotary=False)
    switched = _loop(off, trajectory)
    switched_unroll = _unrolled(off, trajectory, switched[0])
    # every view ablated per row is the switch off
    _assert_trees_equal(switched, _loop(net, trajectory, _ablations(views, rotary=range(views))))
    _zero_angles(monkeypatch)
    reference = _loop(net, trajectory)
    _assert_trees_equal(switched, reference)
    _assert_trees_equal(switched_unroll, _unrolled(net, trajectory, reference[0]))


def test_rotary_off_is_recorded_in_the_policy_file(segment, tmp_path):
    from bnb.runs import checkpoint

    net, _ = segment
    checkpoint.save_policy(tmp_path / "off", _switched(net, rotary=False))
    assert not checkpoint.load_policy(tmp_path / "off", torch.device("cpu")).config.rotary


# ------------------------------------------------------------------ the absent zone slots


def _absent_zone_ablations(views: int, masked=()) -> Ablations:
    on = torch.zeros(views, dtype=torch.bool)
    on[list(masked)] = True
    off = torch.zeros(views, dtype=torch.bool)
    return Ablations(rotary=off, forecast=off, absent_zones=on)


def _assert_rows_close(first, second, rows, axis):
    for one, two in zip(_pytree.tree_leaves(first), _pytree.tree_leaves(second), strict=True):
        index = (slice(None),) * axis + (rows,)
        if one.is_floating_point():
            torch.testing.assert_close(one[index], two[index], rtol=1e-5, atol=1e-5)
        else:
            assert torch.equal(one[index], two[index])


def test_the_segment_has_no_zones(segment):
    """The squad game has no zones: every view's zone slots are the absent token."""
    from bnb.contracts.obs import ZONE

    _, trajectory = segment
    assert (trajectory.obs.zones[..., ZONE["present"]] < 0.5).all()


def test_absent_zones_unmasked_is_the_unablated_step(segment):
    """The ablation carried but on no view: the plain attention path with a full mask, which
    computes what the unmasked step does, within rounding."""
    net, trajectory = segment
    views = trajectory.obs.ships.shape[1]
    plain = _loop(net, trajectory)
    carried = _loop(net, trajectory, _absent_zone_ablations(views))
    _assert_rows_close(plain[0], carried[0], slice(None), 1)
    _assert_rows_close(plain[1], carried[1], slice(None), 0)


def test_absent_zones_masked_views_do_not_read_the_absent_token(segment):
    """On a masked view the absent zone token takes no attention: its outputs do not change
    with the token's value; an unmasked view's do."""
    net, trajectory = segment
    views = trajectory.obs.ships.shape[1]
    masked, kept = [0, 3, 6], [1, 2, 4, 5, 7]
    other = _switched(net)
    with torch.no_grad():
        other.encoder.absent_zone.add_(torch.randn_like(other.encoder.absent_zone))
    ablations = _absent_zone_ablations(views, masked)
    first = _loop(net, trajectory, ablations)
    second = _loop(other, trajectory, ablations)
    _assert_rows_close(first[0], second[0], masked, 1)
    _assert_rows_close(first[1], second[1], masked, 0)
    plain = _loop(net, trajectory)
    assert not torch.allclose(
        first[0].policy_logits[:, masked], plain[0].policy_logits[:, masked], atol=1e-4
    )
    _assert_rows_close(first[0], plain[0], kept, 1)
    assert not torch.allclose(
        second[0].policy_logits[:, kept], plain[0].policy_logits[:, kept], atol=1e-4
    )


def test_absent_zones_is_a_player_suffix():
    from bnb.league.matches import ablations_of

    assert ablations_of("runs/x/policy:absent_zones") == ("runs/x/policy", ("absent_zones",))
    assert ablations_of("p:rotary+absent_zones") == ("p", ("absent_zones", "rotary"))
