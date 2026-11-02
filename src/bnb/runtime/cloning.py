"""Behaviour cloning's loop (scripted spec B2, B4): the student flies the learner views
against itself and the scripted agent, and BC stops on the first of its rules.

The live scores come from the training games themselves (no evaluation rollouts). The
teacher's temperature is calibrated offline on the scripted agent's own games
(`teacher_distributions`, then `bnb.learn.bc.calibrate`).
"""

from collections.abc import Callable

import torch

from bnb.contracts import GameConfig, Pool
from bnb.contracts.obs import SHIP
from bnb.controllers.scripted import Scripted, distribution
from bnb.game import api
from bnb.game.handling.characteristics import characteristics
from bnb.learn.bc import StopConfig, StopRule, flat_views

from .rollout import Rollout
from .train import LEARNER, SCRIPTED, Trainer

# BC's opponents (proposal): self-play and the scripted agent at its natural distribution,
# half each, no rungs. Section 8's pool replaces this when it exists.
LINEUP = (Pool("self_play", 0.5, (LEARNER, LEARNER)),
          Pool("scripted", 0.5, (LEARNER, SCRIPTED)))


def opponent_kind(pool: Pool) -> str:
    """The pool's opponent: the other seat's controller ("learner" for self-play)."""
    others = [name for name in pool.seats if name != LEARNER]
    return others[0] if others else LEARNER


def scores(pools: tuple[Pool, ...], metrics: dict) -> dict:
    """Per opponent kind, from one update's pool metrics: games finished and the score
    (wins plus half the draws; None without a finished game)."""
    totals: dict[str, list[float]] = {}
    for pool in pools:
        results = metrics[pool.name]["results"]
        total = totals.setdefault(opponent_kind(pool), [0.0, 0.0, 0.0])
        for index, name in enumerate(("wins", "draws", "losses")):
            total[index] += results[name]
    out = {}
    for kind, (wins, draws, losses) in totals.items():
        games = wins + draws + losses
        out[kind] = dict(wins=wins, draws=draws, losses=losses, games=games,
                         score=(wins + 0.5 * draws) / games if games else None)
    return out


def clone(trainer: Trainer, stop: StopConfig = StopConfig(),
          log: Callable[[dict], None] = print,
          interrupted: Callable[[], bool] = lambda: False) -> str:
    """BC with `trainer` (objective "bc") until a stop rule fires or `interrupted()` is true
    after an update: the rule's name ("score", "kl_plateau" or "budget"), or
    "interrupted"."""
    if trainer.config.objective != "bc":
        raise ValueError("clone trains with objective 'bc'")
    rule = StopRule(stop)
    while True:
        row = trainer.update()
        live = scores(trainer.rollout.lineup.pools, row["pools"])
        against = live.get(SCRIPTED, dict(wins=0, draws=0, losses=0))
        reason = rule.observe(trainer.ppo.env_steps, against["wins"], against["draws"],
                              against["losses"], row["bc_kl"])
        if reason is None and interrupted():
            reason = "interrupted"
        row.update(scores=live, window_score=rule.score, kl_average=rule.kl_average,
                   env_steps=trainer.ppo.env_steps, stopped_by=reason)
        log(row)
        if reason is not None:
            return reason


@torch.no_grad()
def warm_start(trainer: Trainer, student: Trainer) -> None:
    """RL from BC (scripted spec B4): the student's weights (into the learner and its league
    copy) and its return scale. The optimizer stays fresh and nothing clones; the copy is in
    place, so the rollout's captured graphs keep reading the weights."""
    if trainer.config.objective != "ppo":
        raise ValueError("warm_start starts reinforcement learning (objective 'ppo')")
    state = student.model.state_dict()
    trainer.model.load_state_dict(state)
    trainer.league.load_state_dict(state)
    trainer.ppo.return_scale.low.copy_(student.ppo.return_scale.low)
    trainer.ppo.return_scale.high.copy_(student.ppo.return_scale.high)


@torch.no_grad()
def teacher_distributions(game: GameConfig, pools: tuple[Pool, ...], num_envs: int,
                          segments: int, device: torch.device, length: int = 128,
                          compile: bool = True) -> torch.Tensor:
    """The scripted agent's natural distributions (tokens, 42) over the living own ships of
    its own games: it plays every seat of these pools (each pool's seats renamed to it),
    from staggered clocks, for `segments` segments of `length` ticks."""
    agent = Scripted(characteristics(api.ship_types(device), api.rules_of(game)))
    own_pools = tuple(Pool(pool.name, pool.share, (SCRIPTED,) * len(pool.seats), pool.fog_share,
                           pool.mode, pool.arena_radius, pool.spawn_groups, pool.episode_s,
                           pool.respawn_s) for pool in pools)
    rollout = Rollout(game, own_pools, {SCRIPTED: agent}, SCRIPTED, num_envs, length, device,
                      compile=compile)
    rows = []
    for _ in range(segments):
        obs = rollout.run().obs
        views = flat_views(obs)
        probabilities = distribution(views, agent.characteristics, agent.parameters)
        alive = views.ships[:, :probabilities.shape[1], SHIP["alive"]] > 0.5
        rows.append(probabilities[alive])
    return torch.cat(rows)
