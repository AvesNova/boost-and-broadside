"""The training loop: rollout segments into the store, the learner's update, league refresh,
progress metrics.

Lineup: self-play, one league slot (a lagged copy of the learner), the chaser and random, in
fixed shares of the batch. A mixed batch (mixed spec 5) crosses the lineup with the games of
a training mix. The learner is Yemong, or the toy relational model (learner spec 9).
"""

import copy
import dataclasses
import time
from collections.abc import Callable

import torch

from bnb.contracts import REWARD_LEVELS, GameConfig, Pool
from bnb.controllers.relational import RelationalController
from bnb.controllers.scripted import Chaser, Random
from bnb.controllers.yemong import YemongController
from bnb.game import api
from bnb.game.modes.rules import PRESETS
from bnb.learn.ppo import Learner, LearnerConfig
from bnb.learn.rewards import RewardConfig
from bnb.models.relational import Relational, RelationalConfig
from bnb.models.yemong.model import Yemong, YemongConfig

from .evaluate import first_results, shares
from .rollout import Rollout

LEARNER = "learner"
# The default training mix (mixed spec 1, a proposal): each game preset's share of the batch.
# The battle map is 25% of team deathmatch.
TRAINING_MIX = (("frontline", 0.5), ("deathmatch", 0.375), ("deathmatch_wide", 0.125))
LINEUP = (
    Pool("self_play", 0.5, (LEARNER, LEARNER)),
    Pool("league_0", 0.25, (LEARNER, "league_0")),
    Pool("chaser", 0.125, (LEARNER, "chaser")),
    Pool("random", 0.125, (LEARNER, "random")),
)
MODELS = ("yemong", "relational")
PRECISIONS = ("float32", "tf32", "bfloat16")


def mixed(lineup: tuple[Pool, ...] = LINEUP,
          mix: tuple[tuple[str, float], ...] = TRAINING_MIX) -> tuple[GameConfig, tuple[Pool, ...]]:
    """A mixed batch: the shared config (the first game's) and every game of the mix (a
    preset name and its share) crossed with every pool of the lineup, game by game, so each
    game's environments are one contiguous slice. Pool "frontline/chaser" takes the
    frontline share of the chaser share."""
    base = PRESETS[mix[0][0]]
    pools = []
    for name, share in mix:
        game = PRESETS[name]
        if (game.ships_per_team, game.num_fields) != (base.ships_per_team, base.num_fields):
            raise ValueError(f"{name} has other sizes than {mix[0][0]}: one batch, one size")
        pools.extend(Pool(f"{name}/{pool.name}", share * pool.share, pool.seats, pool.fog_share,
                          mode=game.mode, arena_radius=game.arena_radius,
                          spawn_groups=game.spawn_groups, episode_s=game.episode_s,
                          respawn_s=game.respawn_s) for pool in lineup)
    return base, tuple(pools)


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    """model: "yemong" or "relational" (the toy learner; its sizes are fixed at build).
    length: ticks per segment; shards: segments per update. precision: "float32", "tf32" (a
    process-wide setting, which the entry point applies with `set_precision`) or
    "bfloat16" (the model's matmuls). host_store: stream the segments through pinned host
    memory (on a GPU) rather than keep them on the device."""

    game: GameConfig
    num_envs: int = 1024
    updates: int = 100
    length: int = 128
    shards: int = 1
    model: str = "yemong"
    relational_hidden: int = 256
    league_every: int = 10  # updates between copies of the learner into the league slot
    pools: tuple[Pool, ...] = LINEUP
    learner: LearnerConfig = LearnerConfig()
    precision: str = "float32"
    host_store: bool = True
    seed: int = 0  # the entry point seeds the process with it
    compile: bool = True

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise ValueError(f"model {self.model!r} is not one of {MODELS}")
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision {self.precision!r} is not one of {PRECISIONS}")


def set_precision(precision: str) -> None:
    """The process-wide matmul precision of a run (an entry point's setting): TF32 for
    "tf32", full float32 otherwise (bf16 is the model's own autocast)."""
    torch.set_float32_matmul_precision("high" if precision == "tf32" else "highest")


def build_model(cfg: TrainConfig, device: torch.device) -> torch.nn.Module:
    game, bfloat16 = cfg.game, cfg.precision == "bfloat16"
    if cfg.model == "relational":
        config = RelationalConfig(hidden=cfg.relational_hidden, bfloat16_matmuls=bfloat16)
        return Relational(config, game.ships_per_team, game.num_ships).to(device)
    return Yemong(YemongConfig(bfloat16_matmuls=bfloat16), api.rules_of(game),
                  api.ship_types()).to(device)


def model_controller(model: torch.nn.Module, game: GameConfig):
    if isinstance(model, Relational):
        return RelationalController(model)
    return YemongController(model, game.ships_per_team, game.num_ships)


def compiler(enabled: bool) -> Callable:
    """The runtime's stage cut for the learner: the micro-batch loss, compiled whole."""
    if not enabled:
        return lambda function: function
    return lambda function: torch.compile(function, fullgraph=True, dynamic=False)


class Trainer:
    def __init__(self, cfg: TrainConfig, device: torch.device) -> None:
        game = cfg.game
        self.cfg = cfg
        self.model = build_model(cfg, device)
        self.league = copy.deepcopy(self.model).requires_grad_(False)
        self.controllers = {LEARNER: model_controller(self.model, game),
                            "league_0": model_controller(self.league, game),
                            "chaser": Chaser(game.ships_per_team),
                            "random": Random(game.ships_per_team)}
        reward_config = RewardConfig()
        self.rollout = Rollout(game, cfg.pools, self.controllers, LEARNER, cfg.num_envs,
                               cfg.length, device, compile=cfg.compile,
                               reward_config=reward_config, shards=cfg.shards,
                               host_store=cfg.host_store)
        self.ppo = Learner(self.model, cfg.learner, self.rollout.rules, api.ship_types(device),
                           reward_config.outcome, device, stage=compiler(cfg.compile))
        lineup = self.rollout.lineup
        self.pool_names = tuple(pool.name for pool in lineup.pools)
        self.learner_pool = lineup.pool_of[LEARNER]
        self.updates_done = 0

    def collect(self):
        """Every shard of one update and its bootstrap (no learning): (the store, the
        bootstraps)."""
        bootstraps = []
        for shard in range(self.cfg.shards):
            self.rollout.run(shard)
            bootstraps.append(self.rollout.bootstrap())
        return self.rollout.store, bootstraps

    def pool_metrics(self, store) -> dict:
        """Per pool: the learner's reward per level, per ship and second, and its episode
        results (wins, draws, losses) in the stored segments."""
        pools = len(self.pool_names)
        device = self.learner_pool.device
        steps = torch.zeros(pools, device=device)
        totals = torch.zeros((pools, len(REWARD_LEVELS)), device=device)
        results = torch.zeros((pools, 3), device=device)
        for shard in range(store.shards):
            small = store.shard(("reward", "done", "result"), shard)
            reward, ended, result = small["reward"], small["done"], small["result"]
            steps.index_add_(0, self.learner_pool,
                             torch.full_like(reward[0, :, 0, 0], reward.shape[0]))
            totals.index_add_(0, self.learner_pool, reward.mean(2).sum(0))
            per_view = torch.stack([(ended & (result > 0)).sum(0),
                                    (ended & (result == 0)).sum(0),
                                    (ended & (result < 0)).sum(0)], -1).float()
            results.index_add_(0, self.learner_pool, per_view)
        rate = (totals / steps.clamp_min(1)[:, None] / self.rollout.rules.tick_length).tolist()
        results = results.tolist()
        return {name: dict(reward_per_s=dict(zip(REWARD_LEVELS, rate[index])),
                           results=dict(zip(("wins", "draws", "losses"), results[index])))
                for index, name in enumerate(self.pool_names)}

    def update(self) -> dict:
        t0 = time.perf_counter()
        store, bootstraps = self.collect()
        pools = self.pool_metrics(store)
        t1 = time.perf_counter()
        env_steps = self.cfg.num_envs * self.cfg.length * self.cfg.shards
        stats = self.ppo.update(store, bootstraps, env_steps)  # ends in host syncs
        t2 = time.perf_counter()
        self.updates_done += 1
        if self.updates_done % self.cfg.league_every == 0:
            with torch.no_grad():
                for dst, src in zip(self.league.parameters(), self.model.parameters()):
                    dst.copy_(src)
        return dict(update=self.updates_done, rollout_s=t1 - t0, learn_s=t2 - t1, pools=pools,
                    **stats)


def win_rate(trainer: Trainer, opponent: str = "chaser", games: int = 256) -> dict:
    """The learner against a scripted opponent over `games` fresh games (one per
    environment, every clock from zero, the seat drawn per environment): the shares won,
    drawn and lost."""
    cfg = trainer.cfg
    device = next(trainer.model.parameters()).device
    controllers = {LEARNER: trainer.controllers[LEARNER], opponent: trainer.controllers[opponent]}
    pool = Pool("evaluation", 1.0, (LEARNER, opponent), cfg.pools[0].fog_share)
    return shares(first_results(cfg.game, (pool,), controllers, LEARNER, games, device,
                                cfg.length, cfg.compile))


def train(cfg: TrainConfig, device: torch.device, log: Callable[[dict], None] = print) -> Trainer:
    """The loop. Process-wide settings (the seed, `set_precision`) are the entry point's."""
    trainer = Trainer(cfg, device)
    for _ in range(cfg.updates):
        log(trainer.update())
    return trainer
