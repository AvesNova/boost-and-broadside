"""Behaviour cloning's loop (scripted spec B2, B4): the student flies the learner views
against itself and the teacher, and BC stops on the first of its rules.

The score stop reads the rating lanes (`bnb.runtime.lanes`): the live policy against the raw
scripted agent, in the same rollout as the training games. The training games' scores, by
opponent, are series too. The teacher's temperature is calibrated offline on the scripted
agent's own games (`teacher_distributions`, then `bnb.learn.bc.calibrate`).
"""

from collections.abc import Callable

import torch

from bnb.contracts import GameConfig, Pool
from bnb.contracts.obs import SHIP
from bnb.controllers.scripted import Scripted
from bnb.game import api
from bnb.game.handling.characteristics import characteristics
from bnb.learn.bc import flat_views, teacher_carries, teacher_distribution

from .lanes import BC_MATCHUPS, LIVE, SCRIPTED_PLAYER, LaneConfig
from .rollout import Rollout
from .train import LEARNER, SCRIPTED, Trainer

# BC's opponents (proposal): self-play and the scripted agent, half each, no rungs. The
# scripted seat plays the labels' distribution (`bnb.learn.bc.TeacherOpponent`), so the
# live score is against the teacher the clone copies. Section 8's pool replaces this when
# it exists.
LINEUP = (Pool("self_play", 0.5, (LEARNER, LEARNER)), Pool("scripted", 0.5, (LEARNER, SCRIPTED)))
# BC's rating lanes: the live policy against the raw scripted agent, which the score stop reads.
LANES = LaneConfig(matchups=BC_MATCHUPS)


def opponent_kind(pool: Pool) -> str:
    """The pool's opponent: the other seat's controller ("learner" for self-play)."""
    others = [name for name in pool.seats if name != LEARNER]
    return others[0] if others else LEARNER


def scores(pools: tuple[Pool, ...], metrics: dict) -> dict:
    """Per opponent kind, from one update's pool metrics: games finished and the score
    (wins plus half the draws; None without a finished game)."""
    totals: dict[str, list[float]] = {}
    for pool in pools:
        total = totals.setdefault(opponent_kind(pool), [0, 0, 0])
        for index, name in enumerate(("wins", "draws", "losses")):
            total[index] += metrics[f"episode/{name}/{pool.name}"]
    out = {}
    for kind, (wins, draws, losses) in totals.items():
        games = wins + draws + losses
        out[kind] = dict(
            wins=wins,
            draws=draws,
            losses=losses,
            games=games,
            score=(wins + 0.5 * draws) / games if games else None,
        )
    return out


def lane_results(trainer: Trainer) -> tuple[int, int, int]:
    """The last update's lane games of the live policy against the raw scripted agent, over
    every mode: (wins, draws, losses) from the live policy's side."""
    totals = [0, 0, 0]
    for lane, games in trainer.lane_record.items():
        players = (lane.matchup.first, lane.matchup.second)
        if players == (LIVE, SCRIPTED_PLAYER):
            ordered = games
        elif players == (SCRIPTED_PLAYER, LIVE):
            ordered = games[::-1]
        else:
            continue
        totals = [total + count for total, count in zip(totals, ordered, strict=True)]
    return tuple(totals)


def update(trainer: Trainer) -> tuple[dict, str | None]:
    """One BC update of `trainer` (objective "bc"), fed to its stop rule
    (`trainer.stop_rule`): (the update's row with the training games' scores and the rule's
    window over the lanes' games against the scripted agent, the rule that fired: "score",
    "kl_plateau" or "budget", or None)."""
    if trainer.config.objective != "bc":
        raise ValueError("cloning trains with objective 'bc'")
    rule = trainer.stop_rule
    row = trainer.update()
    live = scores(trainer.config.pools, row)
    wins, draws, losses = lane_results(trainer)
    reason = rule.observe(trainer.ppo.environment_steps, wins, draws, losses, row["bc/kl"])
    for kind, value in live.items():
        row[f"bc/games/{kind}"] = value["games"]
        if value["score"] is not None:
            row[f"bc/score/{kind}"] = value["score"]
    for name, value in (
        ("window_score", rule.score),
        ("window_stderr", rule.standard_error),
        ("kl_average", rule.kl_average),
    ):
        if value is not None:
            row[f"bc/stop/{name}"] = value
    return row, reason


def clone(
    trainer: Trainer,
    log: Callable[[dict], None] = print,
    interrupted: Callable[[], bool] = lambda: False,
) -> str:
    """BC with `trainer` (objective "bc") until its stop rule (`TrainConfig.stop`) fires or
    `interrupted()` is true after an update: the rule's name ("score", "kl_plateau" or
    "budget"), or "interrupted". A run's loop, with checkpoints, is
    `bnb.runs.session.train`."""
    while True:
        row, reason = update(trainer)
        log(row)
        if reason is not None:
            return reason
        if interrupted():
            return "interrupted"


@torch.no_grad()
def warm_start(trainer: Trainer, student: Trainer) -> None:
    """RL from BC (scripted spec B4): the student's weights (into the learner and its
    opponent slots) and its return scale. The optimizer stays fresh and nothing clones; the
    copy is in place, so the rollout's captured graphs keep reading the weights."""
    if trainer.config.objective != "ppo":
        raise ValueError("warm_start starts reinforcement learning (objective 'ppo')")
    trainer.model.load_state_dict(student.model.state_dict())
    trainer.reset_opponents()
    trainer.ppo.return_scale.low.copy_(student.ppo.return_scale.low)
    trainer.ppo.return_scale.high.copy_(student.ppo.return_scale.high)


@torch.no_grad()
def teacher_distributions(
    game: GameConfig,
    types: api.ShipTypes,
    pools: tuple[Pool, ...],
    environments: int,
    segments: int,
    device: torch.device,
    length: int = 128,
    compiled: bool = True,
) -> Callable[[float], torch.Tensor]:
    """The scripted agent's distributions over the living own ships of its own games with
    the ship card `types` (on `device`), as a function of its temperature: T -> (tokens,
    42). It plays every seat of these pools (each pool's seats renamed to it) at T = 1,
    from staggered clocks, for `segments` segments of `length` ticks; the observations are
    kept and relabelled at each T asked, by the agent at T with its own carry run over
    them."""
    agent = Scripted(characteristics(types, api.rules_of(game)), game.ships_per_team)
    own_pools = tuple(
        Pool(
            pool.name,
            pool.share,
            (SCRIPTED,) * len(pool.seats),
            pool.fog_share,
            pool.mode,
            pool.arena_radius,
            pool.spawn_groups,
            pool.episode_seconds,
            pool.respawn_seconds,
        )
        for pool in pools
    )
    rollout = Rollout(
        game,
        types,
        own_pools,
        {SCRIPTED: agent},
        SCRIPTED,
        environments,
        length,
        device,
        compiled=compiled,
    )
    observed = [rollout.run().obs for _ in range(segments)]

    def at(temperature: float) -> torch.Tensor:
        teacher = Scripted(
            agent.characteristics, game.ships_per_team, agent.parameters, temperature
        )
        carry = teacher.init_carry(observed[0].ships.shape[1], device)
        rows = []
        for obs in observed:  # the teacher's carry runs on through the segments
            carries, carry = teacher_carries(teacher, obs, carry)
            probabilities = teacher_distribution(teacher, obs, carries).flatten(0, 1)
            views = flat_views(obs)
            alive = views.ships[:, : probabilities.shape[1], SHIP["alive"]] > 0.5
            rows.append(probabilities[alive])
        return torch.cat(rows)

    return at
