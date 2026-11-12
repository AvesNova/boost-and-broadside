"""The live gauge (league spec 4): the ratings that steer sampling and the dashboard during
training, one gauge per mode and map.

Every rating comes from the run's one fit (`bnb.league.lanes.RunRatings`: the lanes' games
and the league's), read once per update (`update`). An entry the fit does not rate this
update holds its last rating: the scripted agent its fixed 1000 (GAUGE_TOP) until the fit
first rates it; a snapshot the live rating of the update it was taken at, until its games
rate it; an average the live rating until the fit first rates it (it is an average of the
live weights). The live rating starts at a warm start's parent's (0 without one) and holds
its last fitted value.
"""

import dataclasses
import math

import torch

from bnb.runtime.lanes import LIVE

from .ratings import win_probability

GAUGE_TOP = 1000.0  # the scripted agent's fixed rating, its start in the gauge
KINDS = ("checkpoint", "average", "scripted")


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
    """One player's rating in a gauge: `rating` what sampling uses (the last fitted value,
    or its start), `stderr` its error (infinite until the fit first rates it)."""

    kind: str
    rating: float
    stderr: float = math.inf


class Gauge:
    """One mode's live gauge: every entry's rating and the live policy's."""

    def __init__(self, initial_live: float = 0.0) -> None:
        self.entries: dict[str, Rated] = {}
        self.live = Rated("live", initial_live)

    # ---- entries

    def add(self, name: str, kind: str, rating: float | None = None) -> None:
        """Add an entry at `rating`, the live rating by default."""
        if kind not in KINDS:
            raise ValueError(f"entry kind {kind!r} is not one of {KINDS}")
        if name in self.entries:
            raise ValueError(f"entry {name!r} is already in the gauge")
        self.entries[name] = Rated(kind, self.live.rating if rating is None else float(rating))

    def remove(self, name: str) -> None:
        """Drop an entry (retention removed it from the pool)."""
        del self.entries[name]

    # ---- the update

    def update(self, fitted: dict[str, tuple[float, float]]) -> None:
        """Take this update's fit, {player: (rating, standard error)}: the live policy's and
        every entry's it rates (the others as the module docstring says)."""
        for player, rated in ((LIVE, self.live), *self.entries.items()):
            value = fitted.get(player)
            if value is not None and math.isfinite(value[0]):
                rated.rating, rated.stderr = value
        for rated in self.entries.values():
            if rated.kind == "average" and not math.isfinite(rated.stderr):
                rated.rating = self.live.rating

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
        )

    @classmethod
    def from_state(cls, state: dict) -> Gauge:
        """The gauge `state` describes."""
        gauge = cls()
        gauge.live = Rated(**state["live"])
        gauge.entries = {name: Rated(**rated) for name, rated in state["entries"].items()}
        return gauge
