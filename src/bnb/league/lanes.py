"""The run's ratings: per mode, one Bradley-Terry fit of every game a rated player played,
with the scripted agent pinned at 1000. The games are the rating lanes' (`bnb.runtime.lanes`)
and, in a league, the training games of the live policy against the league's entries
(`bnb.league.league.League`); a league without lanes is rated from its own games alone.

Games are recorded by mode and by who played them, under the league's entry names. A lane
names its players; a league slot (`league_<k>`) is resolved to the entry the lanes hold for
it (`assign`: the league copies a slot's network entry into the lanes when the slot takes
it), so an entry keeps its games after it cycles out, and older entries, which played lower
ones, chain the newer to the scripted agent. The league's games are the live policy's
against each entry it played (`League.record`; self-play is not an entry), and a snapshot
taken at an update is credited with that update's games too, since its weights played them.
A game between two stationary players (the ladder's rungs, the league's snapshots) is kept
for the whole run; one with a moving player (the live policy, an average) for its last
`LaneConfig.window` updates. A player is rated once its games connect it to the scripted
agent, so none is reported before its matchups have played, and an entry joins the fit as
soon as its games connect it. The fit gives each player one prior game against the scripted
agent, half won (`bnb.league.ratings.fit_ratings`), which keeps a sweep finite. A lane game
in progress when a slot's weights change is left out (`bnb.runtime.lanes.LaneSeat.tally`).

The live rating and the averaged agent's (the exponential average) of each mode are the
series `rating/live/<mode>` and `rating/average/<mode>`, the uniform average's
`rating/uniform_average/<mode>`; every rated player's is `lanes/rating/<mode>/<player>`.
The league's gauges (`bnb.league.gauge`) read every entry's rating from here.
"""

import collections
import math

import torch

from bnb.runtime.lanes import (
    EXPONENTIAL_AVERAGE,
    LEAGUE_PREFIX,
    LIVE,
    SCRIPTED_PLAYER,
    UNIFORM_AVERAGE,
    Lane,
)
from bnb.runtime.opponents import ladder_rung
from bnb.runtime.train import Trainer

from .gauge import Record
from .pool import Averages
from .ratings import fit_ratings

ANCHOR_RATING = 1000.0  # the scripted agent's
PRIOR_GAMES = 1.0  # each player's virtual games against the scripted agent, half won
AVERAGE = EXPONENTIAL_AVERAGE  # the averaged agent whose rating is `rating/average`
# each rated player's headline series, `rating/<name>/<mode>`
HEADLINES = ((LIVE, "live"), (AVERAGE, "average"), (UNIFORM_AVERAGE, UNIFORM_AVERAGE))
SNAPSHOT_PREFIX = "snapshot_"  # a league snapshot's name (`bnb.league.pool.snapshot_name`)


def stationary(player: str) -> bool:
    """A ladder player or a league snapshot: its play never changes."""
    return ladder_rung(player) is not None or player.startswith(SNAPSHOT_PREFIX)


class RunRatings:
    """The ratings of one run (see the module docstring). trainer: its lanes, their seat and
    their games (`Trainer.lane_record`, taken after each collection); state: `state()` of a
    checkpoint to continue from; modes: the league's modes, rated from its games with or
    without lanes (the lanes' modes are rated in any case). A league calls `assign` and
    `update` with its games; without one, `LaneHooks` calls `update`."""

    def __init__(
        self, trainer: Trainer, state: dict | None = None, modes: tuple[str, ...] = ()
    ) -> None:
        self.lanes = trainer.lanes
        self.seat = trainer.lane_seat
        self.window = trainer.config.lanes.window
        self.modes = tuple(dict.fromkeys((*(lane.mode for lane in self.lanes), *modes)))
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
        entry the lanes do not play, a ladder entry; its games are left out). A moving entry
        (an average) it already plays takes its new weights as the same player's, the games
        in progress tallied; a stationary one it already plays is left as it is. fresh: for
        another entry, the games in progress were played by the previous one."""
        same = self.holders.get(player) == entry and entry is not None
        self.holders[player] = entry
        if weights is None or self.seat is None or (same and stationary(entry)):
            return
        self.seat.load(player, weights, fresh=fresh and not same)

    def resolved(self, player: str) -> str | None:
        """The fit's name for `player`, the one mapping of the lanes' players and the
        league's entries onto it: a league slot is the entry the lanes hold for it (None:
        not rated); any other lane player, and every league entry, is its own name."""
        if not player.startswith(LEAGUE_PREFIX):
            return player
        return self.holders.get(player)

    # ---- the update

    def update(
        self,
        record: dict[Lane, tuple[int, int, int]],
        league: dict[str, dict[str, Record]] | None = None,
        snapshot: str | None = None,
    ) -> None:
        """Fold an update's games in and refit every mode. record: the lanes' games, by the
        players that played them (the league slots' entries as the lanes hold them now);
        league: the live policy's training games against each league entry, by mode and
        entry, from its side; snapshot: a snapshot taken at this update, whose weights
        played the league's games as the live policy."""
        moving: dict[str, list[float]] = {}

        def add(mode: str, first: str | None, second: str | None, games) -> None:
            if first is None or second is None or first == second:
                return  # an unrated slot, or a player against itself
            key = f"{mode}|{first}|{second}"
            target = (
                self.whole.setdefault(key, [0.0, 0.0, 0.0])
                if stationary(first) and stationary(second)
                else moving.setdefault(key, [0.0, 0.0, 0.0])
            )
            for index, count in enumerate(games):
                target[index] += count

        for lane, games in record.items():
            first, second = self.resolved(lane.matchup.first), self.resolved(lane.matchup.second)
            add(lane.mode, first, second, games)
        for mode, entries in (league or {}).items():
            for entry, games in entries.items():
                outcomes = (games.wins, games.ties, games.losses)
                for player in (LIVE, snapshot):
                    add(mode, player, self.resolved(entry), outcomes)
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
        """Per mode: `rating/live`, `rating/average` (the averaged agent) and
        `rating/uniform_average` with their standard errors, and every rated player's
        `lanes/rating` and `lanes/rating_stderr`; a player is absent until its games rate
        it."""
        out = {}
        for mode, rated in self.ratings.items():
            for player, (rating, error) in rated.items():
                if player == SCRIPTED_PLAYER:
                    continue
                out[f"lanes/rating/{mode}/{player}"] = rating
                out[f"lanes/rating_stderr/{mode}/{player}"] = error
            for player, name in HEADLINES:
                if player in rated:
                    out[f"rating/{name}/{mode}"], out[f"rating/{name}_stderr/{mode}"] = rated[
                        player
                    ]
        return out

    def of(self, player: str) -> dict[str, float]:
        """`player`'s rating per mode, where it is rated (a policy file keeps it, for a warm
        start's starting rating)."""
        return {
            mode: rated[player][0]
            for mode, rated in self.ratings.items()
            if player in rated and math.isfinite(rated[player][0])
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
    """The lanes' ratings and the averages of the live weights as a trainer's hooks
    (`bnb.runtime.train.LeagueHooks`), for a run without a league (behaviour cloning). The
    averages are the league's (`bnb.league.pool.Averages`, with the run's
    `TrainConfig.league` settings), folded in after every update and played by the lanes
    that name them. Attaching it sets the trainer's hooks; state: a training checkpoint's
    league state (`bnb.runs.checkpoint.league_state`)."""

    def __init__(
        self, trainer: Trainer, state: tuple[dict, dict[str, torch.Tensor]] | None = None
    ) -> None:
        self.ratings = RunRatings(trainer, None if state is None else state[0]["ratings"])
        weights, config = trainer.model.state_dict(), trainer.config.league
        start = trainer.config.uniform_start()
        if state is None:
            self.averages = Averages(weights, config, start)
        else:
            self.averages = Averages(weights, config, start, state[0]["averaged"], state[1])
        self._load_lanes()
        trainer.league = self

    def _load_lanes(self) -> None:
        """The averages' weights into the lanes that play them (the same players' weights
        moving on: the games in progress are tallied)."""
        if self.ratings.seat is None:
            return
        for name, weights in self.averages.weights.items():
            self.ratings.seat.load(name, weights)

    def after_shard(self, trainer: Trainer, shard: int) -> None:
        """Nothing: the lanes' games are taken once per update."""

    def collected(self, trainer: Trainer) -> None:
        """Nothing before learning."""

    def updated(self, trainer: Trainer) -> dict:
        """Fold the update's lane games in, and its weights into the averages; the ratings'
        series."""
        self.ratings.update(trainer.lane_record)
        self.averages.fold(trainer.model.state_dict(), trainer.ppo.environment_steps)
        self._load_lanes()
        return self.ratings.series()

    def state(self) -> tuple[dict, dict[str, torch.Tensor]]:
        """(the records as JSON under "ratings" and the uniform average's count under
        "averaged", the averages' weights as tensors)."""
        return dict(ratings=self.ratings.state(), averaged=self.averages.count), (
            self.averages.tensors()
        )
