"""The live gauge (league spec 4): the ratings that steer sampling and the dashboard during
training, one gauge per mode and map.

The stationary ladder holds ratings the fits never move: the scripted agent 1000 and every
other rung the rating lanes' calibration (`calibrate`, from `bnb.league.lanes`); before the
lanes have rated a rung it sits at the prior `1000 p` (random 0). Once per update:
  1. the ladder: the checkpoints' ratings, fitted to the accumulated record among
     stationary players with the gauge held (`fit_ladder`);
  2. the live policy: the lanes' rating when given, else rated from this update's training
     record alone against the entries' current ratings (`rate_live`);
  3. the averages: the lanes' rating of an average they rate, else each rated against the
     per-game live ratings of its last `AVERAGE_WINDOW` games (`rate_against`).
A snapshot taken at an update starts at that update's live rating, and its games of that
update (the snapshot's weights are the ones that played them) enter the record.

A rating that comes out infinite (a sweep) is reported as such, and the gauge holds its last
finite value for everything that uses it. Nothing averages it.
"""

import collections
import dataclasses
import math

import torch

from .ratings import fit_ladder, rate_against, rate_live, win_probability

GAUGE_TOP = 1000.0  # the scripted agent's fixed rating; rung p's prior is GAUGE_TOP * p
LADDER_PRIOR_GAMES = 1.0  # virtual games per checkpoint at the strongest fixed rating
AVERAGE_WINDOW = 200  # an average's last games that rate it
KINDS = ("checkpoint", "average", "ladder")


def gauge_rating(scripted_probability: float) -> float:
    """A ladder entry's prior rating, until the lanes calibrate it: random (0) at 0, the
    scripted agent (1) at GAUGE_TOP."""
    return GAUGE_TOP * scripted_probability


@dataclasses.dataclass
class Record:
    """Games from the live policy's side against one entry: wins, losses, ties."""

    wins: float = 0.0
    losses: float = 0.0
    ties: float = 0.0

    @property
    def games(self) -> float:
        """Every game in the record."""
        return self.wins + self.losses + self.ties


@dataclasses.dataclass
class Rated:
    """One entry's rating in a gauge: `rating` the last finite value (what sampling and the
    fits use), `reported` the latest estimate (+-inf after a sweep), `stderr` its error."""

    kind: str
    rating: float
    reported: float
    stderr: float = math.inf


class Gauge:
    """One mode's live gauge: every entry's rating, the record among stationary players, the
    averages' windows and the live policy's rating."""

    def __init__(self, initial_live: float = 0.0) -> None:
        self.entries: dict[str, Rated] = {}
        self.wins: dict[tuple[str, str], float] = collections.defaultdict(float)
        self.ties: dict[tuple[str, str], float] = collections.defaultdict(float)
        self.windows: dict[str, collections.deque] = {}
        self.live = Rated("live", initial_live, initial_live)

    # ---- entries

    def add(self, name: str, kind: str, rating: float | None = None) -> None:
        """Add an entry: a ladder entry at its fixed `rating`; a checkpoint or an average at
        `rating`, the live rating by default."""
        if kind not in KINDS:
            raise ValueError(f"entry kind {kind!r} is not one of {KINDS}")
        if name in self.entries:
            raise ValueError(f"entry {name!r} is already in the gauge")
        start = self.live.rating if rating is None else float(rating)
        self.entries[name] = Rated(kind, start, start, 0.0 if kind == "ladder" else math.inf)
        if kind == "average":
            self.windows[name] = collections.deque(maxlen=AVERAGE_WINDOW)

    def remove(self, name: str) -> None:
        """Drop an entry and its games (retention removed it from the pool)."""
        del self.entries[name]
        self.windows.pop(name, None)
        for table in (self.wins, self.ties):
            for pair in [pair for pair in table if name in pair]:
                del table[pair]

    def credit(self, snapshot: str, record: dict[str, Record]) -> None:
        """Enter a new snapshot's games (the update it was taken at) into the record, against
        the stationary entries among `record`'s opponents."""
        for opponent, games in record.items():
            if self.entries[opponent].kind == "average":
                continue  # the averages move; a pooled record would mix their versions
            self.wins[(snapshot, opponent)] += games.wins
            self.wins[(opponent, snapshot)] += games.losses
            self.ties[(snapshot, opponent)] += games.ties
            self.ties[(opponent, snapshot)] += games.ties

    # ---- the update

    def calibrate(self, ratings: dict[str, tuple[float, float]]) -> None:
        """The ladder entries' ratings from the lanes, {name: (rating, standard error)};
        the entries it does not name keep theirs."""
        for name, (rating, stderr) in ratings.items():
            rated = self.entries.get(name)
            if rated is not None and rated.kind == "ladder" and math.isfinite(rating):
                rated.rating = rated.reported = rating
                rated.stderr = stderr

    def _fit_ladder(self) -> None:
        stationary = [
            name for name, rated in self.entries.items() if rated.kind in ("checkpoint", "ladder")
        ]
        index = {name: position for position, name in enumerate(stationary)}
        count = len(stationary)
        wins = torch.zeros((count, count), dtype=torch.float64)
        ties = torch.zeros((count, count), dtype=torch.float64)
        for table, matrix in ((self.wins, wins), (self.ties, ties)):
            for (first, second), value in table.items():
                if first in index and second in index:
                    matrix[index[first], index[second]] = value
        played = (wins + wins.T + ties).sum(1) > 0
        kinds = [self.entries[name].kind for name in stationary]
        fixed = torch.tensor([kind == "ladder" for kind in kinds])
        free = played & ~fixed
        if not bool(free.any()) or not bool(fixed.any()):
            return
        # only players with games, so every fitted player has at least one
        keep = played | fixed
        ratings = torch.tensor(
            [self.entries[name].rating for name in stationary], dtype=torch.float64
        )
        fitted = fit_ladder(
            wins[keep][:, keep], ties[keep][:, keep], fixed[keep], ratings[keep], LADDER_PRIOR_GAMES
        )
        kept = [name for name, used in zip(stationary, keep.tolist()) if used]
        for name, value in zip(kept, fitted.tolist()):
            rated = self.entries[name]
            if rated.kind == "checkpoint":
                rated.rating = rated.reported = value

    def update(
        self,
        record: dict[str, Record],
        live: tuple[float, float] | None = None,
        rated: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        """One update's gauge from this update's live games against each opponent entry
        (`record`, by entry name): the ladder fit, the live rating, the averages' ratings.
        live: the lanes' (rating, standard error) of the live policy; rated: the lanes'
        ratings of averages, by name."""
        self._fit_ladder()
        rated = rated or {}
        names = [name for name in record if record[name].games > 0]
        if live is not None:
            self.live.reported, self.live.stderr = live
            if math.isfinite(live[0]):
                self.live.rating = live[0]
        elif names:
            rating, stderr = rate_live(
                [self.entries[name].rating for name in names],
                [record[name].wins for name in names],
                [record[name].losses for name in names],
                [record[name].ties for name in names],
            )
            self.live.reported, self.live.stderr = rating, stderr
            if math.isfinite(rating):
                self.live.rating = rating
        for name, window in self.windows.items():
            games = record.get(name)
            if games is not None:
                # from the average's side: its wins are the live policy's losses
                window.extend([(self.live.rating, 1.0)] * round(games.losses))
                window.extend([(self.live.rating, 0.0)] * round(games.wins))
                window.extend([(self.live.rating, 0.5)] * round(games.ties))
            entry = self.entries[name]
            if name in rated:
                rating, stderr = rated[name]
            elif not window:
                entry.rating = entry.reported = self.live.rating
                entry.stderr = math.inf
                continue
            else:
                opponents = torch.tensor([value for value, _ in window], dtype=torch.float64)
                score = torch.tensor([result for _, result in window], dtype=torch.float64)
                rating, stderr = rate_against(opponents, score, 1.0 - score)
            entry.reported, entry.stderr = rating, stderr
            if math.isfinite(rating):
                entry.rating = rating

    # ---- what the update reads

    def names(self) -> list[str]:
        """The entries, in the order they were added."""
        return list(self.entries)

    def win_probabilities(self) -> torch.Tensor:
        """(K,) float64: the predicted probability that the live policy beats each entry."""
        ratings = torch.tensor(
            [rated.rating for rated in self.entries.values()], dtype=torch.float64
        )
        return win_probability(torch.tensor(self.live.rating, dtype=torch.float64), ratings)

    # ---- checkpoints

    def state(self) -> dict:
        """Everything the gauge holds, as plain values (a resume restores it exactly)."""
        return dict(
            live=dataclasses.asdict(self.live),
            entries={name: dataclasses.asdict(rated) for name, rated in self.entries.items()},
            wins=[[first, second, value] for (first, second), value in self.wins.items()],
            ties=[[first, second, value] for (first, second), value in self.ties.items()],
            windows={
                name: [list(game) for game in window] for name, window in self.windows.items()
            },
        )

    @classmethod
    def from_state(cls, state: dict) -> Gauge:
        """The gauge `state` describes."""
        gauge = cls()
        gauge.live = Rated(**state["live"])
        gauge.entries = {name: Rated(**rated) for name, rated in state["entries"].items()}
        for key, table in (("wins", gauge.wins), ("ties", gauge.ties)):
            for first, second, value in state[key]:
                table[(first, second)] = value
        gauge.windows = {
            name: collections.deque([tuple(game) for game in window], maxlen=AVERAGE_WINDOW)
            for name, window in state["windows"].items()
        }
        return gauge
