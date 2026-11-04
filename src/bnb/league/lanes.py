"""The ratings from the rating lanes (`bnb.runtime.lanes`): per mode, one Bradley-Terry fit of
every lane player with the scripted agent pinned at 1000.

Games are recorded by mode and by who played them. A lane names its players; a league slot
(`league_<k>`) is resolved to the pool entry the lanes hold for it (`assign`: the league
copies a slot's entry into the lanes when the slot takes it), so an entry keeps its games
after it cycles out, and older entries, which played lower ones, chain the newer to the
scripted agent. A game between two stationary players (the ladder's rungs, the league's
snapshots) is kept for the whole run; one with a moving player (the live policy, an
average) for its last `LaneConfig.window` updates. A player is rated once its games connect
it to the scripted agent, so none is reported before its matchups have played, and an entry
joins the fit as soon as its games connect it. The fit gives each player one prior game
against the scripted agent, half won (`bnb.league.ratings.fit_ratings`), which keeps a sweep
finite. A game in progress when a slot's weights change is left out
(`bnb.runtime.lanes.LaneSeat.tally`).

The live rating and the averaged agent's of each mode are the series `rating/live/<mode>` and
`rating/average/<mode>`; every rated player's is `lanes/rating/<mode>/<player>`. The ladder's
rungs, rated here from their own games, are the league's calibrated ladder.
"""

import collections
import math

import torch

from bnb.runtime.lanes import (
    EXPONENTIAL_AVERAGE,
    LEAGUE_PREFIX,
    LIVE,
    SCRIPTED_PLAYER,
    Lane,
    ladder_probability,
)
from bnb.runtime.train import Trainer

from .ratings import fit_ratings

ANCHOR_RATING = 1000.0  # the scripted agent's
PRIOR_GAMES = 1.0  # each player's virtual games against the scripted agent, half won
AVERAGE = EXPONENTIAL_AVERAGE  # the averaged agent whose rating is `rating/average`
SNAPSHOT_PREFIX = "snapshot_"  # a league snapshot's name (`bnb.league.pool.snapshot_name`)


def stationary(player: str) -> bool:
    """A ladder player or a league snapshot: its play never changes."""
    return ladder_probability(player) is not None or player.startswith(SNAPSHOT_PREFIX)


class LaneRatings:
    """The lanes' ratings of one run. trainer: its lanes, their seat and their games
    (`Trainer.lane_record`, taken after each collection); state: `state()` of a checkpoint
    to continue from. A league calls `assign` and `update`; without one, `LaneHooks` calls
    `update`."""

    def __init__(self, trainer: Trainer, state: dict | None = None) -> None:
        self.lanes = trainer.lanes
        self.seat = trainer.lane_seat
        self.window = trainer.config.lanes.window
        self.modes = tuple(dict.fromkeys(lane.mode for lane in self.lanes))
        self.whole: dict[str, list[float]] = {}  # stationary players' games, by "mode|a|b"
        self.recent: collections.deque = collections.deque(maxlen=self.window)
        self.ratings: dict[str, dict[str, tuple[float, float]]] = {mode: {} for mode in self.modes}
        self.holders: dict[str, str | None] = {}  # each league slot's entry in the lanes
        if state is not None:
            self.whole = {name: list(games) for name, games in state["whole"].items()}
            self.recent.extend(state["recent"])
            self.ratings = {
                mode: {player: tuple(value) for player, value in rated.items()}
                for mode, rated in state["ratings"].items()
            }
            self.holders = dict(state["holders"])
            if self.seat is not None:
                self.seat.generation.copy_(torch.tensor(state["generation"]))

    def assign(
        self,
        player: str,
        entry: str | None,
        weights: dict[str, torch.Tensor] | None,
        fresh: bool = True,
    ) -> None:
        """The lanes' `player` (a league slot) now plays `entry` with `weights` (None: an
        entry the lanes do not rate, such as a ladder entry or a moving average; its games
        are left out). Nothing changes when it already does."""
        if self.holders.get(player) == entry and entry is not None:
            return
        self.holders[player] = entry
        if weights is not None and self.seat is not None:
            self.seat.load(player, weights, fresh=fresh)

    def resolved(self, player: str) -> str | None:
        """Who `player` is now: a league slot's entry (None: not rated), any other player
        itself."""
        if not player.startswith(LEAGUE_PREFIX):
            return player
        return self.holders.get(player)

    # ---- the update

    def update(self, record: dict[Lane, tuple[int, int, int]]) -> None:
        """Fold an update's lane games in (by the players that played them: the league
        slots' entries as the lanes hold them now) and refit every mode."""
        moving: dict[str, list[float]] = {}
        for lane, games in record.items():
            first, second = self.resolved(lane.matchup.first), self.resolved(lane.matchup.second)
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
        """The records, the last fit, the league slots' entries and the lanes' weight
        generations, as JSON."""
        return dict(
            whole=self.whole,
            recent=list(self.recent),
            ratings={
                mode: {player: list(value) for player, value in rated.items()}
                for mode, rated in self.ratings.items()
            },
            holders=dict(self.holders),
            generation=[] if self.seat is None else self.seat.generation.tolist(),
        )


class LaneHooks:
    """The lanes' ratings as a trainer's hooks (`bnb.runtime.train.LeagueHooks`), for a run
    without a league (behaviour cloning). Attaching it sets the trainer's hooks; state: a
    training checkpoint's league state (`bnb.runs.checkpoint.league_state`)."""

    def __init__(
        self, trainer: Trainer, state: tuple[dict, dict[str, torch.Tensor]] | None = None
    ) -> None:
        self.ratings = LaneRatings(trainer, None if state is None else state[0]["lanes"])
        trainer.league = self

    def after_shard(self, trainer: Trainer, shard: int) -> None:
        """Nothing: the lanes' games are taken once per update."""

    def collected(self, trainer: Trainer) -> None:
        """Nothing before learning."""

    def updated(self, trainer: Trainer) -> dict:
        """Fold the update's lane games in; the ratings' series."""
        self.ratings.update(trainer.lane_record)
        return self.ratings.series()

    def state(self) -> tuple[dict, dict[str, torch.Tensor]]:
        """(the records as JSON under "lanes", no tensors)."""
        return dict(lanes=self.ratings.state()), {}
