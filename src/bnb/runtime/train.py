"""The training loop: rollout, PPO update, league refresh, logging.

Skeleton lineup: self-play, one league slot (a lagged copy of the learner), the chaser and
random, in fixed shares of the batch.
"""

import copy
import dataclasses
import time
from collections.abc import Callable

import torch

from bnb.contracts import GameConfig, Pool
from bnb.controllers.model import ModelController
from bnb.controllers.scripted import Chaser, Random
from bnb.learn.ppo import PPO, PPOConfig
from bnb.models.mlp import MLP

from .rollout import Rollout

LEARNER = "learner"


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    game: GameConfig
    num_envs: int = 1024
    updates: int = 100
    length: int = 128
    hidden: int = 256
    league_every: int = 10  # updates between copies of the learner into the league slot
    pools: tuple[Pool, ...] = (
        Pool("self_play", 0.5, (LEARNER, LEARNER)),
        Pool("league_0", 0.25, (LEARNER, "league_0")),
        Pool("chaser", 0.125, (LEARNER, "chaser")),
        Pool("random", 0.125, (LEARNER, "random")),
    )
    ppo: PPOConfig = PPOConfig()
    seed: int = 0
    compile: bool = True


class Trainer:
    def __init__(self, cfg: TrainConfig, device: torch.device) -> None:
        g = cfg.game
        self.cfg = cfg
        self.model = MLP(g.ships_per_team, g.num_ships, g.num_fields, cfg.hidden).to(device)
        self.league = copy.deepcopy(self.model).requires_grad_(False)
        controllers = {LEARNER: ModelController(self.model),
                       "league_0": ModelController(self.league),
                       "chaser": Chaser(g.ships_per_team), "random": Random(g.ships_per_team)}
        self.rollout = Rollout(g, cfg.pools, controllers, LEARNER, cfg.num_envs, cfg.length,
                               device, compile=cfg.compile)
        self.ppo = PPO(self.model, cfg.ppo, compile=cfg.compile)
        lineup = self.rollout.lineup
        self.pool_names = tuple(pool.name for pool in lineup.pools)
        self.learner_pool = lineup.pool_of[LEARNER]
        self.updates_done = 0

    def collect(self):
        """One rollout and its value bootstrap (no learning)."""
        traj = self.rollout.run()
        with torch.no_grad():
            _, last_value = self.model(self.rollout.learner_view())
        return traj, last_value

    def pool_rewards(self, traj) -> dict[str, float]:
        """Learner reward per ship per second, by pool: the skeleton's progress signal."""
        per_row = traj.reward.mean((0, 2))  # (R,)
        count = torch.zeros(len(self.pool_names), device=per_row.device)
        total = torch.zeros_like(count)
        count.index_add_(0, self.learner_pool, torch.ones_like(per_row))
        total.index_add_(0, self.learner_pool, per_row)
        rate = (total / count.clamp_min(1) / self.rollout.p.tick).tolist()
        return dict(zip(self.pool_names, rate))

    def update(self) -> dict:
        t0 = time.perf_counter()
        traj, last_value = self.collect()
        rewards = self.pool_rewards(traj)
        t1 = time.perf_counter()
        stats = self.ppo.update(traj, last_value)  # ends in one host sync
        t2 = time.perf_counter()
        self.updates_done += 1
        if self.updates_done % self.cfg.league_every == 0:
            with torch.no_grad():
                for dst, src in zip(self.league.parameters(), self.model.parameters()):
                    dst.copy_(src)
        return dict(update=self.updates_done, rollout_s=t1 - t0, learn_s=t2 - t1,
                    reward_per_s=rewards, **stats)


def train(cfg: TrainConfig, device: torch.device, log: Callable[[dict], None] = print) -> Trainer:
    torch.manual_seed(cfg.seed)  # the entry point owns process-wide settings
    trainer = Trainer(cfg, device)
    for _ in range(cfg.updates):
        log(trainer.update())
    return trainer
