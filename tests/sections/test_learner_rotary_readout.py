"""The rotary read-out's loss term (temporal-first and per-block read-out briefs): its target
is the true phasor of every living ship at t, it is the mean over the blocks' read-outs of
their squared pair error, zero when every read-out equals that target, and it appears in
PPO's and BC's terms, the update's series (per block too) and the gradient decomposition."""

import dataclasses
import math

import learner_cases as lc
import pytest
import torch
import yemong_cases as cases
from scripted_cases import CHARACTERISTICS

from bnb.contracts import Pool
from bnb.contracts.obs import SHIP
from bnb.controllers.scripted import Scripted
from bnb.controllers.scripted_defaults import MODE_DEFAULTS
from bnb.game import api
from bnb.learn import bc, labels
from bnb.learn.ppo import COUNTS, TERMS, LearnerConfig
from bnb.learn.rewards import RewardConfig
from bnb.models.yemong.inputs import truth
from bnb.runtime import metrics
from bnb.runtime.rollout import Rollout

pytestmark = pytest.mark.usefixtures("global_generator")

OWN = lc.OWN
ARM_D = dict(order="temporal_first", rotary_readout=True, forecast_feedback=False)


@pytest.fixture(scope="module")
def segment(module_global_generator):
    net = cases.model(**ARM_D)
    runner = lc.rollout(net)
    learner = lc.learner(net, runner)
    prepared = learner.prepare(runner.store, runner.bootstrap())
    return net, runner, learner, prepared


def _terms(learner, runner, prepared, label_change=None):
    views = torch.arange(runner.store.views)
    trajectory, shot_labels = next(learner.staged(runner.store, [views]))
    label = learner.labels(trajectory, shot_labels)
    if label_change is not None:
        label = label_change(label, trajectory)
    denominator = dict(zip(COUNTS, prepared["counts"].sum(0).clamp_min(1).unbind()))
    with torch.no_grad():
        _, terms, _, _, _, _ = learner.loss(
            trajectory, label, prepared["derived"], denominator, prepared["scale"]
        )
        unrolled = learner.unroll(trajectory)
    return dict(zip(TERMS, terms.tolist())), trajectory, label, unrolled


def test_the_term_is_the_blocks_mean_squared_pair_error_over_living_ships(segment):
    net, runner, learner, prepared = segment
    got, trajectory, label, unrolled = _terms(learner, runner, prepared)
    ships = torch.cat([trajectory.obs.ships[..., :OWN, :], trajectory.truth], -2)
    means = truth(ships, trajectory.replay.offset)
    angles = torch.cat(
        [
            2 * math.pi * means[..., 0:1] / net.position_periods,
            2 * math.pi * means[..., 1:2] / net.position_periods,
            means[..., 4:5] * net.heading_frequencies,
        ],
        -1,
    )
    target = torch.stack([angles.cos(), angles.sin()], -2)
    torch.testing.assert_close(label["rotary_readout"], target, rtol=0, atol=2e-4)
    alive = ships[..., SHIP["alive"]] > 0.5
    readouts = unrolled.rotary_readout  # (L, R, N, blocks, 2, pairs)
    assert readouts.shape[-3] == net.config.blocks
    per_block = [
        (readouts[..., block, :, :] - target).square().sum(-2).mean(-1)[alive].mean().item()
        for block in range(net.config.blocks)
    ]
    expected = sum(per_block) / len(per_block)
    assert math.isclose(got["rotary_readout"], expected, rel_tol=1e-4)
    assert got["rotary_readout"] > 0
    assert not math.isclose(per_block[0], per_block[1], rel_tol=1e-3)  # each its own head


def test_the_term_is_zero_when_every_read_out_is_the_truth(segment, monkeypatch):
    net, runner, learner, prepared = segment
    _, trajectory, label, _ = _terms(learner, runner, prepared)
    original = net.unroll
    target = label["rotary_readout"][..., None, :, :]
    truth_per_block = target.expand(*target.shape[:-3], net.config.blocks, *target.shape[-2:])

    def truthful(*args, **kwargs):
        return dataclasses.replace(original(*args, **kwargs), rotary_readout=truth_per_block)

    monkeypatch.setattr(net, "unroll", truthful)
    got, *_ = _terms(learner, runner, prepared)
    assert got["rotary_readout"] == 0.0


def test_the_update_reports_the_term_its_slices_and_its_gradient(segment):
    net, runner, learner, prepared = segment
    import copy

    model = copy.deepcopy(net)
    fresh = lc.learner(model, runner)
    out = metrics.read(fresh.update(runner.store, runner.bootstrap(), cadence=True))
    assert out["rotary_readout/loss"] > 0
    for name in ("ally", "enemy_visible", "enemy_hidden"):
        assert math.isfinite(out[f"rotary_readout/error/{name}"])
    blocks = [out[f"rotary_readout/error/block_{k}"] for k in range(net.config.blocks)]
    assert all(math.isfinite(value) and value > 0 for value in blocks)
    assert f"rotary_readout/error/block_{net.config.blocks}" not in out
    lengths = [
        value for key, value in out.items() if key.startswith("rotary_readout/hidden_length/")
    ]
    assert any(math.isfinite(value) and value > 0 for value in lengths)
    assert out["gradient/norm/trunk/rotary_readout"] > 0  # it reaches the trunk
    assert math.isfinite(out["gradient/share/trunk/rotary_readout"])


def test_without_the_read_out_the_term_is_absent():
    net = cases.model()
    runner = lc.rollout(net, environments=4, length=8, runs=1)
    learner = lc.learner(net, runner)
    out = metrics.read(learner.update(runner.store, runner.bootstrap(), cadence=True))
    assert out["rotary_readout/loss"] == 0.0
    assert not any(key.startswith("rotary_readout/error") for key in out)
    assert not any("rotary_readout" in key for key in out if key.startswith("gradient/"))


def test_the_term_is_in_bcs_terms(module_global_generator):
    net = cases.model(**ARM_D)
    pools = (Pool("scripted", 1.0, ("learner", "scripted"), 0.5),)
    controllers = {
        "learner": cases.controller(net, cases.SHORT_SQUAD),
        "scripted": Scripted(CHARACTERISTICS, OWN, MODE_DEFAULTS),
    }
    runner = Rollout(
        cases.SHORT_SQUAD,
        api.ship_types(lc.CPU),
        pools,
        controllers,
        "learner",
        4,
        16,
        lc.CPU,
        compiled=False,
        extra_reads=Scripted.reads,
    )
    runner.run()
    cloning = bc.BehaviourCloning(
        net,
        bc.config(minibatches=1, micro_batch_rows=4 * runner.rules.num_ships),
        runner.rules,
        runner.carry["types"],
        RewardConfig().outcome,
        lc.CPU,
        Scripted(CHARACTERISTICS, OWN, MODE_DEFAULTS),
        bc.TeacherConfig(),
        tuple(pool.name for pool in pools),
        runner.lineup.pool_of["learner"],
    )
    assert cloning.config.rotary_readout == LearnerConfig().rotary_readout
    out = metrics.read(cloning.update(runner.store, runner.bootstrap()))
    assert out["rotary_readout/loss"] > 0


def test_the_targets_need_the_alive_rows_only(segment):
    """The labels read every row's truth; the term masks it by `alive` (a dead row's stored
    truth is not a position)."""
    net, runner, learner, _ = segment
    trajectory = lc.whole(runner)
    rows = labels.privileged_rows(trajectory)
    target = labels.rotary_targets(net, trajectory, rows)
    assert target.shape == (*trajectory.alive.shape, 2, net.config.rotary_pairs)
    assert torch.isfinite(target).all()
