"""Streaming (learner spec 2, acceptance item 6): the update through pinned host memory
gives exactly the losses and gradients of the all-device path, and the rollout leaves only a
two-block staging ring of records on the device."""

import copy

import learner_cases as lc
import pytest
import torch
import yemong_cases as cases
from torch.utils import _pytree

from bnb.controllers.scripted import Chaser
from bnb.learn.ppo import Learner, LearnerConfig
from bnb.learn.rewards import RewardConfig
from bnb.runtime.rollout import Rollout
from bnb.runtime.store import SegmentStore

pytestmark = [pytest.mark.gpu, pytest.mark.slow,
              pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")]
CUDA = torch.device("cuda") if torch.cuda.is_available() else None
LENGTH, BLOCK = 64, 16


def _device_copy(store: SegmentStore) -> SegmentStore:
    """The same segments in a store on the device."""
    first = lambda tree: _pytree.tree_map_only(torch.Tensor, lambda x: x[0], tree)  # noqa: E731
    ticks = {name: first(_pytree.tree_unflatten(store.data[name], store.specs[name]))
             for name in store.per_tick}
    segments = {name: _pytree.tree_unflatten(store.data[name], store.specs[name])
                for name in store.names if name not in store.per_tick}
    copied = SegmentStore({name: _pytree.tree_map_only(torch.Tensor, lambda x: x.to(CUDA), tree)
                           for name, tree in ticks.items()},
                          {name: _pytree.tree_map_only(torch.Tensor, lambda x: x.to(CUDA), tree)
                           for name, tree in segments.items()},
                          store.shards, store.length, CUDA, host=False)
    for name in store.names:
        for target, source in zip(copied.data[name], store.data[name], strict=True):
            target.copy_(source)
    return copied


def test_the_update_through_pinned_memory_equals_the_device_path():
    net = cases.model().to(CUDA)
    config = cases.SHORT_SQUAD
    controllers = {"learner": cases.controller(net, config),
                   "chaser": Chaser(config.ships_per_team)}
    runner = Rollout(config, lc.POOLS, controllers, "learner", 16, LENGTH, CUDA, compile=False,
                     graph_ticks=BLOCK, host_store=True)
    runner.run()
    runner.run()
    bootstrap = runner.bootstrap()
    host = runner.store
    assert host.pinned and host.data["obs"][0].device.type == "cpu"
    device = _device_copy(host)
    views = host.views
    results = []
    for store in (host, device):
        model = copy.deepcopy(net)
        learner = Learner(model, LearnerConfig(minibatches=1, micro_batch_rows=views // 4 * 10),
                          runner.rules, runner.carry["types"], RewardConfig().outcome, CUDA)
        prepared = learner.prepare(store, [bootstrap])
        chosen = torch.randperm(views, generator=torch.Generator().manual_seed(0))
        terms, levels, statistics = learner.gradient(store, prepared, chosen)
        results.append((terms, levels, [p.grad.clone() for p in model.parameters()],
                        prepared["counts"]))
    (terms_a, levels_a, grads_a, counts_a), (terms_b, levels_b, grads_b, counts_b) = results
    assert torch.equal(counts_a, counts_b)
    assert torch.equal(terms_a, terms_b) and torch.equal(levels_a, levels_b)
    for a, b in zip(grads_a, grads_b, strict=True):
        assert torch.equal(a, b)


def test_the_rollout_keeps_two_blocks_of_records_on_the_device():
    net = cases.model().to(CUDA)
    config = cases.SHORT_SQUAD
    controllers = {"learner": cases.controller(net, config),
                   "chaser": Chaser(config.ships_per_team)}
    runner = Rollout(config, lc.POOLS, controllers, "learner", 16, LENGTH, CUDA, compile=False,
                     graph_ticks=BLOCK, host_store=True)
    trajectory = runner.run()
    ring = runner.carry["buf"]["obs"].ships
    assert ring.shape[0] == 2 * BLOCK and ring.device.type == "cuda"
    assert trajectory.obs.ships.shape[0] == LENGTH and trajectory.obs.ships.device.type == "cpu"
    # The stored segment is what the captured graph wrote: unroll reproduces its decisions.
    on_device = _pytree.tree_map_only(torch.Tensor, lambda x: x.to(CUDA), trajectory)
    from bnb.models.yemong import policy
    with torch.no_grad():
        unrolled = net.unroll(on_device.initial_carry, on_device.obs, on_device.replay,
                              cases.resets(on_device.obs))
    logp, _ = policy.evaluate(unrolled.policy_logits, cases.joint_actions(on_device))
    assert cases.scale_error(logp, on_device.logp) <= 1e-5
    assert cases.scale_error(unrolled.values, on_device.values) <= 1e-5
