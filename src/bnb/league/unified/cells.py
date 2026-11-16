"""The unified evaluator's units of work: a condition (the game a cell is played in) and a
cell (a condition and an ordered pair of players), with the size buckets that decide which
compiled engine plays a cell.

A condition is everything about a game that the engine holds as data, plus its mode
(`preset`): the two teams' ship counts (the first player's, the second's; uneven teams are
benched slots) and the arena scale. A cell's games are tallied from its first player's side.
"""

import dataclasses
import math

# The default size buckets (ships per team of each compiled engine): a cell plays in the
# smallest bucket that holds its larger team, its other slots benched.
BUCKETS = (5, 10, 20)


@dataclasses.dataclass(frozen=True, order=True)
class Condition:
    """preset: the game preset (its mode, arena and clocks, `bnb.game.PRESETS`); first,
    second: the first player's and the second player's ships; arena_scale: the map's scale
    against the preset's radius (1: the preset's own map)."""

    preset: str
    first: int
    second: int
    arena_scale: float = 1.0

    def __post_init__(self) -> None:
        if self.first < 1 or self.second < 1:
            raise ValueError(f"{self}: each side fields at least one ship")
        if not self.arena_scale > 0.0:
            raise ValueError(f"{self}: the arena scale must be positive")

    @property
    def largest(self) -> int:
        """The larger team's ships."""
        return max(self.first, self.second)

    @property
    def key(self) -> str:
        """The condition's record name: the preset, then `_<first>v<second>` unless 5v5,
        then `_scale<s>` unless 1 (the rating pool's preset key for an Elo condition)."""
        name = self.preset
        if (self.first, self.second) != (5, 5):
            name += f"_{self.first}v{self.second}"
        if self.arena_scale != 1.0:
            name += f"_scale{self.arena_scale:g}"
        return name


@dataclasses.dataclass(frozen=True, order=True)
class Cell:
    """A condition and its players, the first's side tallied."""

    condition: Condition
    first: str
    second: str

    def __post_init__(self) -> None:
        if self.first == self.second:
            raise ValueError(f"{self}: a cell needs two players")

    @property
    def key(self) -> str:
        return f"{self.condition.key}/{self.first}_vs_{self.second}"


@dataclasses.dataclass(frozen=True)
class Outcome:
    """One finished game, from the cell's first player's side: score (1 a win, 0.5 a draw,
    0 a loss), kills (the first player's, the second's), seconds (game time)."""

    score: float
    first_kills: int = 0
    second_kills: int = 0
    seconds: float = 0.0

    @property
    def kill_share(self) -> float:
        """The first player's kills over both sides' (0.5 with none), team deathmatch's
        kill-share score."""
        both = self.first_kills + self.second_kills
        return self.first_kills / both if both else 0.5


def bucket_of(condition: Condition, buckets: tuple[int, ...] = BUCKETS) -> int:
    """The smallest bucket holding the condition's larger team. Raises ValueError when none
    does."""
    for size in sorted(buckets):
        if condition.largest <= size:
            return size
    raise ValueError(f"{condition}: {condition.largest} ships, past the largest bucket {buckets}")


def arena_uniform(scale: float, sigma: float, low: float, high: float) -> float:
    """The map-size draw's uniform that gives `scale` (`bnb.game.layout.draw_arena_scale`'s
    inverse): scale = exp(sigma ndtri(cdf_low + u (cdf_high - cdf_low)))."""
    if not low <= scale <= high:
        raise ValueError(f"arena scale {scale} outside the sampled range [{low}, {high}]")

    def cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

    first, last = cdf(math.log(low) / sigma), cdf(math.log(high) / sigma)
    return (cdf(math.log(scale) / sigma) - first) / (last - first)
