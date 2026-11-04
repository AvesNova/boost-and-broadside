"""Streaming (learner spec 2, acceptance item 6): the update through pinned host memory
gives exactly the losses and gradients of the all-device path, and the rollout leaves only a
two-block staging ring of records on the device."""

import warnings

import learner_cases as lc
import pytest
import torch
import yemong_cases as cases
from torch.utils import _pytree

from bnb.controllers.scripted import Chaser
from bnb.game import api
from bnb.learn.ppo import Learner, LearnerConfig
from bnb.learn.rewards import RewardConfig
from bnb.runtime import metrics
from bnb.runtime.rollout import Rollout

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.usefixtures("global_generator"),
    pytest.mark.slow,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]
CUDA = torch.device("cuda") if torch.cuda.is_available() else None
LENGTH, BLOCK = 64, 16


def _rollout(host_store: bool, shards: int = 1, seed: int = 0):
    """A Yemong rollout on the GPU, the same from the same seed whatever the store: every
    shard collected, with its bootstrap."""
    net = cases.model().to(CUDA)
    torch.manual_seed(seed)  # the rollout's draws come from the global generator
    config = cases.SHORT_SQUAD
    controllers = {
        "learner": cases.controller(net, config),
        "chaser": Chaser(config.ships_per_team),
    }
    runner = Rollout(
        config,
        api.ship_types(CUDA),
        lc.POOLS,
        controllers,
        "learner",
        16,
        LENGTH,
        CUDA,
        compiled=False,
        graph_ticks=BLOCK,
        shards=shards,
        host_store=host_store,
    )
    runner.run(0)  # history for the stored initial carry
    bootstraps = []
    for shard in range(shards):
        runner.run(shard)
        bootstraps.append(runner.bootstrap())
    return net, runner, bootstraps


def _gradient(net, runner, bootstraps):
    """One minibatch of every view, in micro-batches of a quarter: the counts, the terms,
    the value losses and the gradient."""
    store = runner.store
    views = store.views
    learner = Learner(
        net,
        LearnerConfig(minibatches=1, micro_batch_rows=views // 4 * runner.rules.num_ships),
        runner.rules,
        runner.carry["types"],
        RewardConfig().outcome,
        CUDA,
    )
    prepared = learner.prepare(store, bootstraps)
    chosen = torch.randperm(views, generator=torch.Generator().manual_seed(0))
    terms, levels, _, _, _ = learner.gradient(store, prepared, chosen)
    return prepared["counts"], terms, levels, [p.grad.clone() for p in net.parameters()]


@pytest.mark.parametrize("shards", [1, 2])
def test_the_update_through_pinned_memory_equals_the_device_path(shards):
    """The same rollout written into pinned host memory and into a device store gives the
    same records, and the update the same counts, losses and gradients, bit for bit."""
    host = _rollout(host_store=True, shards=shards)
    device = _rollout(host_store=False, shards=shards)
    assert host[1].store.pinned and host[1].store.data["obs"][0].device.type == "cpu"
    assert device[1].store.data["obs"][0].device.type == "cuda"
    for name in host[1].store.names:
        for a, b in zip(host[1].store.data[name], device[1].store.data[name], strict=True):
            assert torch.equal(a.to(CUDA), b), name
    for a, b in zip(_gradient(*host), _gradient(*device), strict=True):
        if isinstance(a, list):
            assert all(torch.equal(x, y) for x, y in zip(a, b, strict=True))
        else:
            assert torch.equal(a, b)


def test_an_update_synchronises_only_to_read_its_kl_and_its_results():
    """No host sync per minibatch or per staged batch: the update reads the epoch's mean KL
    once per epoch but the last, and its results once."""
    net, runner, bootstraps = _rollout(host_store=True)
    config = LearnerConfig(
        epochs=3, minibatches=4, target_kl=float("inf"), micro_batch_rows=4 * runner.rules.num_ships
    )
    learner = Learner(
        net, config, runner.rules, runner.carry["types"], RewardConfig().outcome, CUDA
    )
    learner.update(runner.store, bootstraps)  # the first: the staging buffers
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            out = metrics.read(learner.update(runner.store, bootstraps))
        finally:
            torch.cuda.set_sync_debug_mode("default")
    synchronising = [w for w in caught if "called a synchronizing" in str(w.message)]
    assert out["policy/minibatch_steps"] == 12
    assert len(synchronising) == config.epochs, [(w.filename, w.lineno) for w in synchronising]


def test_the_rollout_keeps_two_blocks_of_records_on_the_device():
    net = cases.model().to(CUDA)
    config = cases.SHORT_SQUAD
    controllers = {
        "learner": cases.controller(net, config),
        "chaser": Chaser(config.ships_per_team),
    }
    runner = Rollout(
        config,
        api.ship_types(CUDA),
        lc.POOLS,
        controllers,
        "learner",
        16,
        LENGTH,
        CUDA,
        compiled=False,
        graph_ticks=BLOCK,
        host_store=True,
    )
    trajectory = runner.run()
    ring = runner.carry["records"]["obs"].ships
    assert ring.shape[0] == 2 * BLOCK and ring.device.type == "cuda"
    assert trajectory.obs.ships.shape[0] == LENGTH and trajectory.obs.ships.device.type == "cpu"
    # The stored segment is what the captured graph wrote: unroll reproduces its decisions.
    on_device = _pytree.tree_map_only(torch.Tensor, lambda x: x.to(CUDA), trajectory)
    from bnb.models.yemong import policy

    with torch.no_grad():
        unrolled = net.unroll(
            on_device.initial_carry, on_device.obs, on_device.replay, cases.resets(on_device.obs)
        )
    log_probability, _ = policy.evaluate(unrolled.policy_logits, cases.joint_actions(on_device))
    assert cases.scale_error(log_probability, on_device.log_probability) <= 1e-5
    assert cases.scale_error(unrolled.values, on_device.values) <= 1e-5


def test_the_cadence_groups_add_no_synchronisation():
    """The cadence groups (the gradient decomposition, the representation) and the heads'
    series stay on the device: an update that measures them synchronises as often as one
    that does not."""
    net, runner, bootstraps = _rollout(host_store=True)
    config = LearnerConfig(
        epochs=2, minibatches=2, target_kl=float("inf"), micro_batch_rows=4 * runner.rules.num_ships
    )
    learner = Learner(
        net, config, runner.rules, runner.carry["types"], RewardConfig().outcome, CUDA
    )
    learner.update(runner.store, bootstraps, cadence=True)  # the first: the staging buffers
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            series = learner.update(runner.store, bootstraps, cadence=True)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    synchronising = [w for w in caught if "called a synchronizing" in str(w.message)]
    assert len(synchronising) == config.epochs - 1, [(w.filename, w.lineno) for w in synchronising]
    out = metrics.read(series)
    assert "gradient/actor_share" in out and "representation/effective_rank/trunk_block_0" in out
    assert learner.cadence_stopwatch.seconds() > 0


def test_the_rollouts_game_sums_are_taken_without_a_synchronisation():
    """The games' sums and the outcome ledger live in the captured graph's buffers; taking
    them copies and resets on the device."""
    net, runner, _ = _rollout(host_store=True)
    torch.cuda.synchronize()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        torch.cuda.set_sync_debug_mode("warn")
        try:
            sums, tables = runner.take_games()
        finally:
            torch.cuda.set_sync_debug_mode("default")
    assert not [w for w in caught if "called a synchronizing" in str(w.message)]
    assert sums["resolve"].sum() > 0 and tables["calibration"].sum() > 0
    again, _ = runner.take_games()
    assert again["resolve"].abs().sum() == 0  # reset in place
