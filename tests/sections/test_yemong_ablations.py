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
    torch.manual_seed(11)  # the offsets the initial carry and each reset draw
    carry = net.initial_carry(obs.ships.shape[1], OWN, 10)
    with torch.no_grad():
        return cases.step_loop(
            net,
            lambda c, v, r: net.step(c, v, r, ablations),
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
