"""The geometry head's loss term (position-codes brief): nats per target over the pairs the
view sees, its own denominator (the pairs), in PPO's and BC's terms, the update's series
and the gradient decomposition; absent without the head."""

import copy
import math

import learner_cases as lc
import pytest
import torch
import yemong_cases as cases
from scripted_cases import CHARACTERISTICS

from bnb.contracts import Pool
from bnb.controllers.scripted import Scripted
from bnb.controllers.scripted_defaults import MODE_DEFAULTS
from bnb.game import api
from bnb.learn import bc, labels
from bnb.learn.ppo import COUNTS, TERMS, LearnerConfig
from bnb.learn.rewards import RewardConfig
from bnb.models.yemong import codes, geometry
from bnb.runtime import metrics
from bnb.runtime.rollout import Rollout

pytestmark = pytest.mark.usefixtures("global_generator")

OWN = lc.OWN


@pytest.fixture(scope="module")
def segment(module_global_generator):
    net = cases.model(order="temporal_first", geometry_head=True)
    runner = lc.rollout(net)
    learner = lc.learner(net, runner)
    prepared = learner.prepare(runner.store, runner.bootstrap())
    return net, runner, learner, prepared


def test_the_term_is_the_pairs_mean_cross_entropy_per_target(segment):
    net, runner, learner, prepared = segment
    views = torch.arange(runner.store.views)
    trajectory, shot_labels = next(learner.staged(runner.store, [views]))
    label = learner.labels(trajectory, shot_labels)
    denominator = dict(zip(COUNTS, prepared["counts"].sum(0).clamp_min(1).unbind()))
    with torch.no_grad():
        _, terms, _, _, _, _ = learner.loss(
            trajectory, label, prepared["derived"], denominator, prepared["scale"]
        )
        unrolled = learner.unroll(trajectory)
    got = dict(zip(TERMS, terms.tolist()))["geometry"]
    partners, mask, values = labels.geometry_pairs(trajectory)
    assert torch.equal(label["geometry_partners"], partners)
    assert torch.equal(label["geometry_mask"], mask)
    assert prepared["counts"][:, COUNTS.index("pairs")].sum() == mask.sum()
    logits = net.geometry(unrolled.geometry_latents, partners).float()
    log_probabilities = codes.grouped_log_softmax(logits, geometry.RUNS)
    target = geometry.encode(values.float())
    per_pair = -(target * log_probabilities).sum(-1) / len(geometry.TARGETS)
    expected = per_pair[mask].mean().item()
    assert math.isclose(got, expected, rel_tol=1e-4)
    assert got == pytest.approx(math.log(geometry.ANGLE_BINS), rel=0.1)  # untrained: near uniform


def test_the_update_reports_the_term_its_targets_and_its_gradient(segment):
    net, runner, _, _ = segment
    fresh = lc.learner(copy.deepcopy(net), runner)
    out = metrics.read(fresh.update(runner.store, runner.bootstrap(), cadence=True))
    assert out["geometry/loss"] > 0
    for name in geometry.TARGETS:
        assert math.isfinite(out[f"geometry/cross_entropy/{name}"])
    assert out["gradient/norm/trunk/geometry"] > 0  # it reaches the trunk
    assert math.isfinite(out["gradient/share/trunk/geometry"])


def test_without_the_head_the_term_is_absent():
    net = cases.model()
    assert net.geometry_head is None
    runner = lc.rollout(net, environments=4, length=8, runs=1)
    learner = lc.learner(net, runner)
    out = metrics.read(learner.update(runner.store, runner.bootstrap(), cadence=True))
    assert out["geometry/loss"] == 0.0
    assert not any(key.startswith("geometry/cross_entropy") for key in out)
    assert not any("geometry" in key for key in out if key.startswith("gradient/"))


def test_the_term_is_in_bcs_terms(module_global_generator):
    net = cases.model(geometry_head=True)
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
    assert cloning.config.geometry == LearnerConfig().geometry
    out = metrics.read(cloning.update(runner.store, runner.bootstrap()))
    assert out["geometry/loss"] > 0
