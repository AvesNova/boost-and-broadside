"""The compile layout: `TrainConfig.compile_split` is the same arithmetic as the whole loss
(eager exactly, the plumbing; compiled within float rounding, slow), and the micro-batches
are of one shape where they can be."""

import learner_cases as lc
import pytest
import torch
import yemong_cases as cases

from bnb.learn.ppo import Learner, micro_batch_sizes
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
    [(255, 64, [51] * 5), (255, 256, [255]), (256, 64, [64] * 4), (25, 16, [13, 12])],
)
def test_micro_batches_are_of_one_size_where_they_can_be(segment, views, most, sizes):
    """One shape of micro-batch where up to twice the fewest divide the views (a smaller
    last one is a second compile of every learner stage), otherwise the fewest as equal as
    they can be; never above micro_views."""
    net, runner = segment
    learner = lc.with_micro_batch(lc.learner(net, runner), most)
    batches = learner._micro_batches(torch.arange(views))
    assert [len(batch) for batch in batches] == sizes
    assert torch.equal(torch.cat(batches), torch.arange(views))


@pytest.mark.parametrize(
    ("counts", "most", "shapes"),
    [
        # 5090 RL (12 240 views in 32 minibatches) at 640 rows: six micro-batches of 64 or 63
        ((383, 382), 64, {64, 63}),
        ((383, 382), 32, {32, 31}),
        # 5090 BC (256 minibatches): one micro-batch a minibatch, one size per minibatch
        ((48, 47), 256, {48, 47}),
        ((765,), 64, {51}),
        ((96, 64), 64, {32}),  # a size common to both
    ],
)
def test_minibatches_of_two_sizes_share_their_micro_batch_sizes(counts, most, shapes):
    """The update's minibatches of two sizes (the minibatches do not divide the views) take
    two micro-batch sizes together, not two each: each size is a compile of every stage."""
    seen = set()
    for count in counts:
        sizes = micro_batch_sizes(count, most, counts)
        assert sum(sizes) == count and max(sizes) <= most
        seen |= set(sizes)
    assert seen == shapes


def test_an_update_runs_its_loss_on_the_shapes_it_names(segment):
    """Every micro-batch the update's loss and labels see is of a size
    `micro_batch_shapes` names (the trainer prints them; each is one compile of every
    compiled stage): a new source of a second shape fails here. 12 views in 5 minibatches
    (3 3 2 2 2) at 3 views a micro-batch: sizes 3 and 2."""
    net, runner = segment
    views = runner.store.views
    learner = lc.with_micro_batch(lc.learner(net, runner, epochs=1, minibatches=5), 3)
    expected = learner.micro_batch_shapes(views)
    assert views == 12 and expected == (2, 3)
    seen = []
    loss, codes = learner.loss, learner.codes

    def recorded(function):
        def call(trajectory, *rest):
            seen.append(trajectory.obs.ships.shape[1])
            return function(trajectory, *rest)

        return call

    learner.loss, learner.codes = recorded(loss), recorded(codes)
    learner.update(runner.store, runner.bootstrap())
    assert set(seen) == set(expected)


def test_minibatches_that_divide_the_views_compile_once(segment):
    """With the minibatches dividing the views every micro-batch is of one size."""
    net, runner = segment
    learner = lc.with_micro_batch(lc.learner(net, runner, minibatches=4), 3)
    assert learner.micro_batch_shapes(12) == (3,)
    # 2040 environments' views (12 240) over 256 minibatches do not divide (256 x 47 + 208);
    # the launch's 2048 (12 288 views) divide BC's 256 and 128 and RL's 32 and 16
    learner = lc.with_micro_batch(lc.learner(net, runner, minibatches=256), 256)
    assert learner.micro_batch_shapes(12240) == (47, 48)
    assert learner.micro_batch_shapes(12288) == (48,)
    for minibatches in (128, 32, 16):
        learner = lc.with_micro_batch(lc.learner(net, runner, minibatches=minibatches), 128)
        assert len(learner.micro_batch_shapes(12288)) == 1
