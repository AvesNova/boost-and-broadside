"""Shared cases for the learner section tests: a real stored Yemong segment on the CPU, and a
learner over it."""

import dataclasses

import torch
import yemong_cases as cases

from bnb.contracts import Pool
from bnb.controllers.scripted import Chaser
from bnb.game import api
from bnb.learn.ppo import Learner, LearnerConfig
from bnb.learn.rewards import RewardConfig
from bnb.runtime.rollout import Rollout

CPU = torch.device("cpu")
OWN = 5
# Self-play and the chaser, half fogged, short clocks: resets, hidden ships and fights.
POOLS = (
    Pool("self_play", 0.5, ("learner", "learner"), 0.5),
    Pool("chaser", 0.5, ("learner", "chaser"), 0.5),
)


def rollout(
    net,
    environments: int = 8,
    length: int = 32,
    runs: int = 2,
    config=cases.SHORT_SQUAD,
    pools=POOLS,
    controllers=None,
):
    """A Rollout run `runs` times (so the stored initial carry has history), with Yemong as
    the learner."""
    controllers = controllers or {
        "learner": cases.controller(net, config),
        "chaser": Chaser(config.ships_per_team),
    }
    runner = Rollout(
        config,
        api.ship_types(CPU),
        pools,
        controllers,
        "learner",
        environments,
        length,
        CPU,
        compiled=False,
    )
    for _ in range(runs):
        runner.run()
    return runner


def learner(net, runner: Rollout, **config) -> Learner:
    """Every view in one minibatch by default."""
    views = runner.store.views
    rows = views * runner.rules.num_ships
    config = dict(dict(minibatches=1, micro_batch_rows=rows), **config)
    return Learner(
        net,
        LearnerConfig(**config),
        runner.rules,
        runner.carry["types"],
        RewardConfig().outcome,
        CPU,
    )


def whole(runner: Rollout):
    """Every view of the store as one micro-batch Trajectory."""
    from bnb.contracts import Trajectory

    store = runner.store
    views = torch.arange(store.views)
    return Trajectory(**next(iter(store.batches(store.names, [views]))))


def with_micro_batch(learner: Learner, views: int) -> Learner:
    """Micro-batches of `views` views."""
    rows = views * learner.rules.num_ships
    learner.config = dataclasses.replace(learner.config, micro_batch_rows=rows)
    return learner
