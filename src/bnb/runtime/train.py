"""The training loop: rollout segments into the store, the learner's update, league refresh,
progress metrics.

Lineup: self-play, one league slot (a lagged copy of the learner), the chaser and random, in
fixed shares of the batch. A mixed batch (mixed spec 5) crosses the lineup with the games of
a training mix. The learner is Yemong, or the toy relational model (learner spec 9). The
objective is PPO, or behaviour cloning of the scripted agent (`bnb.learn.bc`, with its own
lineup and stop rule in `bnb.runtime.cloning`).
"""

import copy
import dataclasses
import functools
import time
from collections.abc import Callable

import torch

from bnb.contracts import REWARD_LEVELS, GameConfig, Pool
from bnb.controllers.relational import RelationalController
from bnb.controllers.scripted import Chaser, Random, Scripted, distribution
from bnb.controllers.yemong import YemongController
from bnb.game import api
from bnb.game.handling.characteristics import characteristics
from bnb.game.modes.rules import PRESETS
from bnb.learn.bc import BehaviourCloning, TeacherConfig, TeacherOpponent
from bnb.learn.ppo import Learner, LearnerConfig
from bnb.learn.rewards import RewardConfig
from bnb.models.relational import Relational, RelationalBuild, RelationalConfig
from bnb.models.yemong.model import Yemong, YemongBuild, YemongConfig, yemong_build

from .evaluate import first_results, shares
from .rollout import Rollout

LEARNER = "learner"
SCRIPTED = "scripted"
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
OBJECTIVES = ("ppo", "bc")
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
    length: ticks per segment; shards: segments per update. precision: "bfloat16" (the
    model's matmuls under autocast; the default), "float32" or "tf32" (a process-wide
    setting, which the entry point applies with `set_precision`). reward: the reward
    levels' weights. host_store: stream the segments through pinned host memory (on a GPU)
    rather than keep them on the device.
    seed: the learner's random streams; the entry point seeds the process with it too.
    objective: "ppo", or "bc" (clone the scripted agent; `learner` should then be
    `bnb.learn.bc.config()`, and `teacher` sets its labels and how the scripted opponent
    plays, `bnb.learn.bc.TeacherOpponent`)."""

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
    reward: RewardConfig = RewardConfig()
    precision: str = "bfloat16"
    host_store: bool = True
    seed: int = 0
    compile: bool = True
    objective: str = "ppo"
    teacher: TeacherConfig = TeacherConfig()

    def __post_init__(self) -> None:
        if self.objective not in OBJECTIVES:
            raise ValueError(f"objective {self.objective!r} is not one of {OBJECTIVES}")
        if self.model not in MODELS:
            raise ValueError(f"model {self.model!r} is not one of {MODELS}")
        if self.precision not in PRECISIONS:
            raise ValueError(f"precision {self.precision!r} is not one of {PRECISIONS}")


def set_precision(precision: str) -> None:
    """The process-wide matmul precision of a run (an entry point's setting): TF32 for
    "tf32", full float32 otherwise (bf16 is the model's own autocast)."""
    torch.set_float32_matmul_precision("high" if precision == "tf32" else "highest")


def model_build(config: TrainConfig, architecture: YemongConfig | RelationalConfig | None = None):
    """The build record of the run's model: `architecture` (a saved model's config; the
    code's defaults without one) with the run's precision (and the relational model's
    width), and the numbers the run's game and ship card derive."""
    game, bfloat16 = config.game, config.precision == "bfloat16"
    if config.model == "relational":
        architecture = dataclasses.replace(architecture or RelationalConfig(),
                                           hidden=config.relational_hidden,
                                           bfloat16_matmuls=bfloat16)
        return RelationalBuild(architecture, game.ships_per_team, game.num_ships)
    architecture = dataclasses.replace(architecture or YemongConfig(), bfloat16_matmuls=bfloat16)
    return yemong_build(architecture, api.rules_of(game), api.ship_types())


def build_model(build: YemongBuild | RelationalBuild, device: torch.device) -> torch.nn.Module:
    """A fresh model of the build record, on `device`."""
    return (Relational(build) if isinstance(build, RelationalBuild) else Yemong(build)).to(device)


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
    """The run's model, league slot, opponents, rollout and learner. build: the model's build
    record (a checkpoint's); without one, `model_build(config)`."""

    def __init__(self, config: TrainConfig, device: torch.device,
                 build: YemongBuild | RelationalBuild | None = None) -> None:
        game = config.game
        self.config = config
        self.model = build_model(build or model_build(config), device)
        self.league = copy.deepcopy(self.model).requires_grad_(False)
        cloning = config.objective == "bc"
        seated = {name for pool in config.pools for name in pool.seats}
        types = api.ship_types(device)
        # The scripted agent's characteristics, when it plays or teaches (measured once).
        self.characteristics = characteristics(types, api.rules_of(game)) \
            if cloning or SCRIPTED in seated else None
        builders = {LEARNER: lambda: model_controller(self.model, game),
                    "league_0": lambda: model_controller(self.league, game),
                    "chaser": lambda: Chaser(game.ships_per_team),
                    "random": lambda: Random(game.ships_per_team),
                    # BC scores the learner against the teacher its labels describe
                    SCRIPTED: lambda: TeacherOpponent(Scripted(self.characteristics),
                                                      config.teacher)
                    if cloning else Scripted(self.characteristics)}
        self.controllers = {name: build() for name, build in builders.items()
                            if name in seated or name in (LEARNER, "league_0")}
        self.rollout = Rollout(game, config.pools, self.controllers, LEARNER, config.num_envs,
                               config.length, device, compile=config.compile,
                               reward_config=config.reward, shards=config.shards,
                               host_store=config.host_store,
                               extra_reads=Scripted.reads if cloning else ())
        lineup = self.rollout.lineup
        self.pool_names = tuple(pool.name for pool in lineup.pools)
        self.learner_pool = lineup.pool_of[LEARNER]
        rules, stage = self.rollout.rules, compiler(config.compile)
        if cloning:  # the labels are the scripted agent's distribution, transformed
            agent = Scripted(self.characteristics)
            teacher = functools.partial(distribution, characteristics=agent.characteristics,
                                        parameters=agent.parameters)
            self.ppo = BehaviourCloning(self.model, config.learner, rules, types,
                                        config.reward.outcome, device, teacher, config.teacher,
                                        self.pool_names, self.learner_pool.repeat(config.shards),
                                        stage=stage, seed=config.seed)
        else:
            self.ppo = Learner(self.model, config.learner, rules, types, config.reward.outcome,
                               device, stage=stage, seed=config.seed)
        self.updates_done = 0

    @property
    def steps_per_update(self) -> int:
        """Environment steps one update collects."""
        return self.config.num_envs * self.config.length * self.config.shards

    def collect(self):
        """Every shard of one update and its bootstrap (no learning): (the store, the
        bootstraps)."""
        bootstraps = []
        for shard in range(self.config.shards):
            self.rollout.run(shard)
            bootstraps.append(self.rollout.bootstrap())
        return self.rollout.store, bootstraps

    def pool_metrics(self, store) -> dict:
        """Per pool: the learner's reward per level, per ship and second, and its episode
        results (wins, draws, losses) in the stored segments. The per-pool sums are masked
        reductions, not atomic adds, so the metrics are the same on every run of the same
        data (an exact resume logs what a run straight through logs)."""
        pools = len(self.pool_names)
        device = self.learner_pool.device
        member = (self.learner_pool[None, :] == torch.arange(pools, device=device)[:, None])
        member = member.float()[..., None]  # (pools, views, 1)
        steps = torch.zeros(pools, device=device)
        totals = torch.zeros((pools, len(REWARD_LEVELS)), device=device)
        results = torch.zeros((pools, 3), device=device)
        for shard in range(store.shards):
            small = store.shard(("reward", "done", "result"), shard)
            reward, ended, result = small["reward"], small["done"], small["result"]
            steps += member[..., 0].sum(1) * reward.shape[0]
            totals += (member * reward.mean(2).sum(0)).sum(1)
            per_view = torch.stack([(ended & (result > 0)).sum(0),
                                    (ended & (result == 0)).sum(0),
                                    (ended & (result < 0)).sum(0)], -1).float()
            results += (member * per_view).sum(1)
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
        stats = self.ppo.update(store, bootstraps, self.steps_per_update)  # ends in host syncs
        t2 = time.perf_counter()
        self.updates_done += 1
        if self.updates_done % self.config.league_every == 0:
            with torch.no_grad():
                for dst, src in zip(self.league.parameters(), self.model.parameters()):
                    dst.copy_(src)
        return dict(update=self.updates_done, rollout_s=t1 - t0, learn_s=t2 - t1, pools=pools,
                    nonfinite_beliefs=self.nonfinite_beliefs(), **stats)

    def nonfinite_beliefs(self) -> int:
        """Non-finite belief entries the learner's carry has counted since its rollouts began
        (Yemong's `Carry.nonfinite`; 0 for a model without one)."""
        counts = getattr(self.rollout.carry["ctrl"][LEARNER], "nonfinite", None)
        return 0 if counts is None else int(counts.sum())


def win_rate(trainer: Trainer, opponent: str = "chaser", games: int = 256) -> dict:
    """The learner against a scripted opponent over `games` fresh games (one per
    environment, every clock from zero, the seat drawn per environment): the shares won,
    drawn and lost."""
    config = trainer.config
    device = next(trainer.model.parameters()).device
    controllers = {LEARNER: trainer.controllers[LEARNER], opponent: trainer.controllers[opponent]}
    pool = Pool("evaluation", 1.0, (LEARNER, opponent), config.pools[0].fog_share)
    return shares(first_results(config.game, (pool,), controllers, LEARNER, games, device,
                                config.length, config.compile))


def train(config: TrainConfig, device: torch.device,
          log: Callable[[dict], None] = print) -> Trainer:
    """The loop. Process-wide settings (the seed, `set_precision`) are the entry point's."""
    trainer = Trainer(config, device)
    for _ in range(config.updates):
        log(trainer.update())
    return trainer
