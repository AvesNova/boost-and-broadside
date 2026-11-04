"""The ratings from the rating lanes (`bnb.runtime.lanes`): per mode, one Bradley-Terry fit of
every lane player with the scripted agent pinned at 1000.

Games are recorded by mode and by who played them. A lane names its players; a snapshot
slot (`snapshot_<i>`) is resolved to the snapshot it holds, `lane_snapshot_<steps>`, so a
snapshot keeps its games after its slot takes another. A game between two stationary
players (the ladder's rungs, the snapshots) is kept for the whole run; one with a moving
player (the live policy, an average: their weights change every update) for its last
`LaneConfig.window` updates. A player is rated once its games connect it to the scripted
agent, so none is reported before its matchups have played, and a snapshot joins the fit as
soon as its chain does. The fit gives each player one prior game against the scripted agent,
half won (`bnb.league.ratings.fit_ratings`), which keeps a sweep finite.

The snapshots: every `LaneConfig.snapshot_every` updates the live policy's weights are frozen
into the next slot, round robin; the first slot holds the starting weights. A game in
progress when a slot's weights change was played partly by the old ones, so a slot's games
count only once every game since the swap began after it: `settle` updates, the longest
episode over the ticks an update plays.

The live rating and the averaged agent's of each mode are the series `rating/live/<mode>` and
`rating/average/<mode>`; every rated player's is `lanes/rating/<mode>/<player>`. The ladder's
rungs, rated here from their own games, are the league's calibrated ladder.
"""

import collections
import math

import torch

from bnb.runtime.lanes import (
    EXPONENTIAL_AVERAGE,
    LIVE,
    SCRIPTED_PLAYER,
    SNAPSHOT_PREFIX,
    Lane,
    ladder_probability,
    snapshot_slot,
)
from bnb.runtime.train import Trainer

from .ratings import fit_ratings

ANCHOR_RATING = 1000.0  # the scripted agent's
PRIOR_GAMES = 1.0  # each player's virtual games against the scripted agent, half won
AVERAGE = EXPONENTIAL_AVERAGE  # the averaged agent whose rating is `rating/average`
FROZEN_PREFIX = (
    "lane_snapshot_"  # a frozen snapshot's name, by the environment steps it was taken at
)


def stationary(player: str) -> bool:
    """A ladder player or a frozen snapshot: its play never changes."""
    return ladder_probability(player) is not None or player.startswith(FROZEN_PREFIX)


class LaneRatings:
    """The lanes' ratings of one run, and its snapshot slots. trainer: its lanes, their seat
    and their games (`Trainer.lane_record`, taken after each collection); state: `state()`
    of a checkpoint to continue from, and tensors: `tensors()` of it (the slots' weights).
    A league calls `update` and `freeze`; without one, `LaneHooks` does."""

    def __init__(
        self,
        trainer: Trainer,
        state: dict | None = None,
        tensors: dict[str, torch.Tensor] | None = None,
    ) -> None:
        config = trainer.config
        self.lanes = trainer.lanes
        self.seat = trainer.lane_seat
        self.window, self.every = config.lanes.window, config.lanes.snapshot_every
        players = {p for lane in self.lanes for p in (lane.matchup.first, lane.matchup.second)}
        self.slots = sum(1 for player in players if player.startswith(SNAPSHOT_PREFIX))
        self.modes = tuple(dict.fromkeys(lane.mode for lane in self.lanes))
        played = config.shards * config.length + config.lookahead
        longest = int(trainer.rollout.episode_ticks.max()) if self.lanes else 0
        self.settle = math.ceil(longest / played) if played else 0
        self.whole: dict[str, list[float]] = {}  # stationary players' games, by "mode|a|b"
        self.recent: collections.deque = collections.deque(maxlen=self.window)
        self.ratings: dict[str, dict[str, tuple[float, float]]] = {mode: {} for mode in self.modes}
        self.holders: list[str | None] = [None] * self.slots  # each slot's snapshot
        self.since = [0] * self.slots  # the update after which each slot's weights changed
        self.next_slot = 0
        self.weights: dict[int, dict[str, torch.Tensor]] = {}
        if state is not None:
            self.whole = {name: list(games) for name, games in state["whole"].items()}
            self.recent.extend(state["recent"])
            self.ratings = {
                mode: {player: tuple(value) for player, value in rated.items()}
                for mode, rated in state["ratings"].items()
            }
            self.holders, self.since = list(state["holders"]), list(state["since"])
            self.next_slot = state["next_slot"]
            for slot, holder in enumerate(self.holders):
                if holder is not None:
                    prefix = f"slot{slot}."
                    self.weights[slot] = {
                        name[len(prefix) :]: value.to(trainer.rollout.device)
                        for name, value in (tensors or {}).items()
                        if name.startswith(prefix)
                    }
                    self._load(slot)
        elif self.slots:  # the first slot: the starting weights, before any game
            self._freeze_into(trainer, settled=True)

    # ---- the snapshot slots

    def _load(self, slot: int) -> None:
        if self.seat is not None:
            self.seat.load(snapshot_slot(slot), self.weights[slot])

    @torch.no_grad()
    def _freeze_into(self, trainer: Trainer, settled: bool = False) -> None:
        slot = self.next_slot
        self.next_slot = (slot + 1) % self.slots
        self.weights[slot] = {
            name: value.detach().clone() for name, value in trainer.model.state_dict().items()
        }
        self.holders[slot] = f"{FROZEN_PREFIX}{trainer.ppo.environment_steps}"
        # an unsettled slot's games count from `settle` updates after this one
        self.since[slot] = trainer.updates_done - (self.settle if settled else 0)
        self._load(slot)

    def freeze(self, trainer: Trainer) -> None:
        """After an update: when one is due, freeze the live weights into the next slot."""
        if self.slots and trainer.updates_done % self.every == 0:
            self._freeze_into(trainer)

    def resolved(self, player: str, update: int) -> str | None:
        """Who `player` was in update `update`'s games: a slot's settled snapshot (None while
        empty or settling), any other player itself."""
        if not player.startswith(SNAPSHOT_PREFIX):
            return player
        slot = int(player[len(SNAPSHOT_PREFIX) :])
        holder = self.holders[slot] if slot < self.slots else None
        if holder is None or update - self.since[slot] <= self.settle:
            return None
        return holder

    # ---- the update

    def update(self, record: dict[Lane, tuple[int, int, int]], update: int | None = None) -> None:
        """Fold update `update`'s lane games in (by the players that played them) and refit
        every mode. update: the trainer's update count after it (None: every slot settled)."""
        moving: dict[str, list[float]] = {}
        for lane, games in record.items():
            first = self.resolved(lane.matchup.first, math.inf if update is None else update)
            second = self.resolved(lane.matchup.second, math.inf if update is None else update)
            if first is None or second is None:
                continue
            key = f"{lane.mode}|{first}|{second}"
            target = (
                self.whole.setdefault(key, [0.0, 0.0, 0.0])
                if stationary(first) and stationary(second)
                else moving.setdefault(key, [0.0, 0.0, 0.0])
            )
            for index, count in enumerate(games):
                target[index] += count
        self.recent.append(moving)
        for mode in self.modes:
            self.ratings[mode] = self._fit(mode)

    def _games(self, mode: str) -> dict[tuple[str, str], list[float]]:
        """{(first, second): (wins, ties, losses)} of `mode`'s record."""
        out: dict[tuple[str, str], list[float]] = {}
        for record in (self.whole, *self.recent):
            for key, games in record.items():
                key_mode, first, second = key.split("|")
                if key_mode != mode:
                    continue
                total = out.setdefault((first, second), [0.0, 0.0, 0.0])
                for index, count in enumerate(games):
                    total[index] += count
        return out

    def _fit(self, mode: str) -> dict[str, tuple[float, float]]:
        """{player: (rating, standard error)} of the players connected to the scripted agent
        by games in `mode`'s record."""
        games = {pair: value for pair, value in self._games(mode).items() if sum(value) > 0}
        edges: dict[str, set[str]] = collections.defaultdict(set)
        for first, second in games:
            edges[first].add(second)
            edges[second].add(first)
        if SCRIPTED_PLAYER not in edges:
            return {}
        players, frontier = [SCRIPTED_PLAYER], [SCRIPTED_PLAYER]
        while frontier:
            for neighbour in sorted(edges[frontier.pop()]):
                if neighbour not in players:
                    players.append(neighbour)
                    frontier.append(neighbour)
        index = {player: position for position, player in enumerate(players)}
        wins = torch.zeros((len(players), len(players)), dtype=torch.float64)
        for (first, second), (won, tied, lost) in games.items():
            if first not in index:  # games that do not reach the scripted agent
                continue
            wins[index[first], index[second]] += won + tied / 2
            wins[index[second], index[first]] += lost + tied / 2
        ratings, errors = fit_ratings(wins, index[SCRIPTED_PLAYER], PRIOR_GAMES)
        return {
            player: (float(ratings[position]) + ANCHOR_RATING, float(errors[position]))
            for player, position in index.items()
        }

    def rating(self, mode: str, player: str) -> tuple[float, float] | None:
        """(rating, standard error) of `player` in `mode`, or None while it is unrated."""
        return self.ratings.get(mode, {}).get(player)

    def series(self) -> dict:
        """Per mode: `rating/live` and `rating/average` (the averaged agent) with their
        standard errors, and every rated player's `lanes/rating` and `lanes/rating_stderr`;
        a player is absent until its games rate it."""
        out = {}
        for mode, rated in self.ratings.items():
            for player, (rating, error) in rated.items():
                if player == SCRIPTED_PLAYER:
                    continue
                out[f"lanes/rating/{mode}/{player}"] = rating
                out[f"lanes/rating_stderr/{mode}/{player}"] = error
            for player, name in ((LIVE, "live"), (AVERAGE, "average")):
                if player in rated:
                    out[f"rating/{name}/{mode}"], out[f"rating/{name}_stderr/{mode}"] = rated[
                        player
                    ]
        return out

    def live(self) -> dict[str, float]:
        """The live policy's rating per mode, where it is rated (a policy file keeps it, for a
        warm start's starting rating)."""
        return {
            mode: rated[LIVE][0]
            for mode, rated in self.ratings.items()
            if LIVE in rated and math.isfinite(rated[LIVE][0])
        }

    def state(self) -> dict:
        """The records, the last fit and the slots' holders, as JSON."""
        return dict(
            whole=self.whole,
            recent=list(self.recent),
            ratings={
                mode: {player: list(value) for player, value in rated.items()}
                for mode, rated in self.ratings.items()
            },
            holders=list(self.holders),
            since=list(self.since),
            next_slot=self.next_slot,
        )

    def tensors(self) -> dict[str, torch.Tensor]:
        """The snapshot slots' weights, by `slot<i>.<parameter>`."""
        return {
            f"slot{slot}.{name}": value
            for slot, weights in self.weights.items()
            for name, value in weights.items()
        }


class LaneHooks:
    """The lanes' ratings as a trainer's hooks (`bnb.runtime.train.LeagueHooks`), for a run
    without a league (behaviour cloning). Attaching it sets the trainer's hooks; state: a
    training checkpoint's league state (`bnb.runs.checkpoint.league_state`)."""

    def __init__(
        self, trainer: Trainer, state: tuple[dict, dict[str, torch.Tensor]] | None = None
    ) -> None:
        if state is None:
            self.ratings = LaneRatings(trainer)
        else:
            prefix = "lanes."
            tensors = {k[len(prefix) :]: v for k, v in state[1].items() if k.startswith(prefix)}
            self.ratings = LaneRatings(trainer, state[0]["lanes"], tensors)
        trainer.league = self

    def after_shard(self, trainer: Trainer, shard: int) -> None:
        """Nothing: the lanes' games are taken once per update."""

    def collected(self, trainer: Trainer) -> None:
        """Nothing before learning."""

    def updated(self, trainer: Trainer) -> dict:
        """Fold the update's lane games in; the ratings' series."""
        self.ratings.update(trainer.lane_record, trainer.updates_done)
        self.ratings.freeze(trainer)
        return self.ratings.series()

    def state(self) -> tuple[dict, dict[str, torch.Tensor]]:
        """(the records as JSON under "lanes", the snapshot slots' weights under "lanes.")."""
        tensors = {f"lanes.{name}": value for name, value in self.ratings.tensors().items()}
        return dict(lanes=self.ratings.state()), tensors
