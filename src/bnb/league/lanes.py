"""The ratings from the rating lanes (`bnb.runtime.lanes`): per mode, one Bradley-Terry fit of
every lane player with the scripted agent pinned at 1000.

The record of each matchup is kept by mode. A matchup of two ladder players (both
stationary) keeps every game of the run; one with a network player (the live policy, an
average: their weights move) keeps the games of its last `LaneConfig.window` updates. A
player is rated only once it has games that connect it to the scripted agent, so no rating
is reported before its matchups have played. The fit gives each player one prior game
against the scripted agent, half won (`bnb.league.ratings.fit_ratings`), which keeps a sweep
finite.

The live rating and the averaged agent's of each mode are the series `rating/live/<mode>` and
`rating/average/<mode>`; every rated player's is `lanes/rating/<mode>/<player>`. The ladder's
rungs, rated here from their own games, are the league's calibrated ladder.
"""

import collections
import math

import torch

from bnb.runtime.lanes import EXPONENTIAL_AVERAGE, LIVE, SCRIPTED_PLAYER, Lane, ladder_probability
from bnb.runtime.train import Trainer

from .ratings import fit_ratings

ANCHOR_RATING = 1000.0  # the scripted agent's
PRIOR_GAMES = 1.0  # each player's virtual games against the scripted agent, half won
AVERAGE = EXPONENTIAL_AVERAGE  # the averaged agent whose rating is `rating/average`


def stationary(lane: Lane) -> bool:
    """Both players of the lane are ladder players."""
    matchup = lane.matchup
    return ladder_probability(matchup.first) is not None and (
        ladder_probability(matchup.second) is not None
    )


class LaneRatings:
    """The lanes' ratings of one run. trainer: its lanes and their games
    (`Trainer.lane_record`, taken after each collection); state: `state()` of a checkpoint
    to continue from. It is the trainer's hooks when no league is attached (behaviour
    cloning); a league calls `update` itself."""

    def __init__(self, trainer: Trainer, state: dict | None = None) -> None:
        self.lanes = trainer.lanes
        self.window = trainer.config.lanes.window
        self.modes = tuple(dict.fromkeys(lane.mode for lane in self.lanes))
        self.whole: dict[str, list[float]] = {}  # stationary lanes: every game
        self.recent: collections.deque = collections.deque(maxlen=self.window)
        self.ratings: dict[str, dict[str, tuple[float, float]]] = {mode: {} for mode in self.modes}
        if state is not None:
            self.whole = {name: list(games) for name, games in state["whole"].items()}
            self.recent.extend(state["recent"])
            self.ratings = {
                mode: {player: tuple(value) for player, value in rated.items()}
                for mode, rated in state["ratings"].items()
            }

    # ---- the update

    def update(self, record: dict[Lane, tuple[int, int, int]]) -> None:
        """Fold one update's lane games in and refit every mode."""
        moving = {}
        for lane, games in record.items():
            if stationary(lane):
                total = self.whole.setdefault(lane.name, [0.0, 0.0, 0.0])
                for index, count in enumerate(games):
                    total[index] += count
            else:
                moving[lane.name] = list(games)
        self.recent.append(moving)
        for mode in self.modes:
            self.ratings[mode] = self._fit(mode)

    def _games(self, mode: str) -> dict[Lane, list[float]]:
        """Each lane of `mode` with its games in the record (wins, ties, losses)."""
        out = {}
        for lane in self.lanes:
            if lane.mode != mode:
                continue
            if stationary(lane):
                games = self.whole.get(lane.name, [0.0, 0.0, 0.0])
            else:
                games = [0.0, 0.0, 0.0]
                for update in self.recent:
                    for index, count in enumerate(update.get(lane.name, (0, 0, 0))):
                        games[index] += count
            out[lane] = games
        return out

    def _fit(self, mode: str) -> dict[str, tuple[float, float]]:
        """{player: (rating, standard error)} of the players connected to the scripted agent
        by games in `mode`'s record."""
        games = {lane: value for lane, value in self._games(mode).items() if sum(value) > 0}
        edges: dict[str, set[str]] = collections.defaultdict(set)
        for lane in games:
            edges[lane.matchup.first].add(lane.matchup.second)
            edges[lane.matchup.second].add(lane.matchup.first)
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
        for lane, (won, tied, lost) in games.items():
            first, second = index[lane.matchup.first], index[lane.matchup.second]
            wins[first, second] += won + tied / 2
            wins[second, first] += lost + tied / 2
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
        """The records and the last fit, as JSON."""
        return dict(
            whole=self.whole,
            recent=list(self.recent),
            ratings={
                mode: {player: list(value) for player, value in rated.items()}
                for mode, rated in self.ratings.items()
            },
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
