"""The compile layout: `TrainConfig.compile_split` is the same arithmetic as the whole loss
(eager exactly, the plumbing; compiled within float rounding, slow), and the micro-batches
are of one shape where they can be."""

import learner_cases as lc
import pytest
import torch
import yemong_cases as cases

from bnb.learn.ppo import Learner
from bnb.runtime.train import compiler

pytestmark = pytest.mark.usefixtures("global_generator")


@pytest.fixture(scope="module")
def segment(module_global_generator):
    net = cases.model()
    runner = lc.rollout(net)
    return net, runner


def _gradient(net, runner, compiled: bool, split: bool):
    """(terms, every parameter's gradient) of one minibatch of every view, the learner
    built with the switch."""
    learner = lc.learner(net, runner)
    learner = Learner(
        net,
        learner.config,
        runner.rules,
        runner.carry["types"],
        learner.outcome_weight,
        learner.device,
        stage=compiler(compiled),
        split=split,
    )
    prepared = learner.prepare(runner.store, runner.bootstrap())
    terms, *_ = learner.gradient(runner.store, prepared, torch.arange(runner.store.views))
    gradients = [parameter.grad.clone() for parameter in net.parameters()]
    return terms, gradients


def _check(case, reference, relative: float):
    terms, gradients = case
    reference_terms, reference_gradients = reference
    torch.testing.assert_close(terms, reference_terms, rtol=relative, atol=relative * 1e-2)
    for a, b in zip(gradients, reference_gradients, strict=True):
        scale = max(1e-12, b.abs().max().item())
        assert (a - b).abs().max().item() <= relative * scale + 1e-8


def test_the_split_changes_nothing_eager(segment):
    net, runner = segment
    _check(_gradient(net, runner, False, True), _gradient(net, runner, False, False), 0.0)


@pytest.mark.slow
@pytest.mark.usefixtures("fresh_compiler")
def test_the_split_compiles_to_the_same_arithmetic(segment):
    net, runner = segment
    _check(_gradient(net, runner, True, True), _gradient(net, runner, True, False), 1e-4)


@pytest.mark.parametrize(
    ("views", "most", "sizes"),
    [(255, 64, [51] * 5), (255, 256, [255]), (256, 64, [64] * 4), (25, 16, [16, 9])],
)
def test_micro_batches_are_of_one_size_where_they_can_be(segment, views, most, sizes):
    """One shape of micro-batch where up to twice the fewest divide the views (a smaller
    last one is a second compile of every learner stage), never above micro_views."""
    net, runner = segment
    learner = lc.with_micro_batch(lc.learner(net, runner), most)
    batches = learner._micro_batches(torch.arange(views))
    assert [len(batch) for batch in batches] == sizes
    assert torch.equal(torch.cat(batches), torch.arange(views))
