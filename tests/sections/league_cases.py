"""Small league runs for the league's tests: a 2v2 game with short episodes and the toy
relational learner, so slots drain within a few hundred ticks on the CPU."""

import dataclasses
import pathlib

import torch

from bnb.contracts import GameConfig, Pool
from bnb.league.league import League
from bnb.learn.ppo import LearnerConfig
from bnb.models.relational import RelationalConfig
from bnb.runtime.lanes import LaneConfig
from bnb.runtime.opponents import LeagueConfig
from bnb.runtime.train import LEARNER, TrainConfig, Trainer

GAME = GameConfig(2, 2, 1500.0, episode_seconds=2.0, respawn_seconds=1.0)
SELF_PLAY = (Pool("self_play", 1.0, (LEARNER, LEARNER)),)  # a lineup without the league


def config(device: str = "cpu", **changes) -> TrainConfig:
    """16 environments (8 opponent rows, 2 per slot), one 16-tick segment per update, no
    lookahead and no rating lanes."""
    values = dict(
        game=GAME,
        environments=16,
        updates=0,
        length=16,
        shards=1,
        lookahead=0,
        lanes=LaneConfig(environments=0),
        model="relational",
        relational=RelationalConfig(hidden=16),
        learner=LearnerConfig(epochs=1, minibatches=2, warmup_steps=1000, micro_batch_rows=64),
        compiled=device == "cuda",
        league=LeagueConfig(
            snapshot_every=16 * 16, cheese=()
        ),  # the base pool; the cheese is its own test
    )
    values.update(changes)
    return TrainConfig(**values)


def league(directory: pathlib.Path, device: str = "cpu", **changes) -> tuple[Trainer, League]:
    """A trainer with its league attached, its snapshots under `directory`."""
    trainer = Trainer(config(device, **changes), torch.device(device))
    return trainer, League(trainer, directory)


def with_league(base: TrainConfig, **changes) -> TrainConfig:
    """`base` with its LeagueConfig changed."""
    return dataclasses.replace(base, league=dataclasses.replace(base.league, **changes))
