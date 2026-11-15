"""Lineups and games: which controller plays which role, in which game, for what share of the
environments."""

import dataclasses
import math
from fractions import Fraction

from .config import ELIMINATION, ActiveFields, GameConfig


@dataclasses.dataclass(frozen=True)
class ArenaScale:
    """Map-size sampling: each environment's arena scale, drawn at every reset, is
    log-normal about 1 (ln scale ~ Normal(0, sigma)) truncated to [low, high], by the
    inverse CDF of one uniform (`bnb.game.layout.draw_arena_scale`). The episode's arena
    radius is then the pool's radius times the scale, at least the mode's minimum
    (`bnb.game.api.arena_radius_for`)."""

    sigma: float = 0.35
    low: float = 0.5
    high: float = 2.0

    def __post_init__(self) -> None:
        if not (self.sigma > 0.0 and 0.0 < self.low <= 1.0 <= self.high):
            raise ValueError(f"{self}: sigma > 0 and 0 < low <= 1 <= high")


@dataclasses.dataclass(frozen=True)
class TeamSizes:
    """Team-size sampling (arm A2): each episode's team size N, the same for both teams
    (an N-against-N game), drawn at every reset from the sizes low to high (None: the
    game's ships per team), each with its weight (None: every size alike), without the
    sizes below the mode's smallest team (`bnb.game.api.MINIMUM_SHIPS_PER_TEAM`: Frontline
    3, the others 1), the weights renormalised over the sizes left. A team's slots past N
    are benched (`bnb.game.api`: held as dead ships, out of play, shown with the benched
    flag), so the batch keeps its one static size. The draw is the runtime's
    (`bnb.runtime.rollout`), keyed by (seed, environment, episode)."""

    low: int = 1
    high: int | None = None
    weights: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.low < 1 or (self.high is not None and self.high < self.low):
            raise ValueError(f"{self}: need 1 <= low <= high")
        if self.weights is not None:
            if self.high is None:
                raise ValueError(f"{self}: weights need high, one weight per size low to high")
            if len(self.weights) != self.high - self.low + 1:
                raise ValueError(
                    f"{self}: {len(self.weights)} weights for the sizes {self.low} to {self.high}"
                )
            if any(weight < 0.0 for weight in self.weights) or not sum(self.weights) > 0.0:
                raise ValueError(f"{self}: weights must be non-negative, and not all 0")

    def probabilities(self, ships_per_team: int, minimum: int = 1) -> tuple[float, ...]:
        """Each size's probability, sizes 1 to ships_per_team in order, for a game of
        ships_per_team slots a side whose mode's smallest team is `minimum`. Raises
        ValueError for a range past the game's slots or with no size left."""
        high = ships_per_team if self.high is None else self.high
        if high > ships_per_team:
            raise ValueError(f"{self}: at most the game's {ships_per_team} ships per team")
        weights = (1.0,) * (high - self.low + 1) if self.weights is None else self.weights
        by_size = [0.0] * ships_per_team
        for size, weight in zip(range(self.low, high + 1), weights, strict=True):
            if size >= minimum:
                by_size[size - 1] = float(weight)
        total = sum(by_size)
        if not total > 0.0:
            raise ValueError(
                f"{self}: no size of {self.low} to {high} with weight is at least the mode's "
                f"smallest team, {minimum}"
            )
        return tuple(weight / total for weight in by_size)


@dataclasses.dataclass(frozen=True)
class Pool:
    """A contiguous slice of the batch with a fixed lineup and one game.

    seats: the controller name for each role (role r plays team (r + seat) mod T, with the
        seat drawn per environment at every reset, so every controller plays both sides).
    fog_share: the fraction of the pool's environments with fog of war; fog_share times the
        pool's size must be a whole number.
    mode, arena_radius, spawn_groups, episode_seconds, respawn_seconds: the pool's game
        (mixed spec 1); None takes the GameConfig's value, and for the clocks then the
        mode's default. Every environment of the pool shares them. The sizes (ships, fields) are the
        GameConfig's, shared by the whole batch.
    arena_scale: map-size sampling (`ArenaScale`): each episode's arena radius drawn about
        the pool's; None plays the pool's radius every episode (evaluation, ratings, play).
    active_fields: field-count sampling (`bnb.contracts.ActiveFields`): each episode plays a
        drawn number of the fields, the rest absent; None takes the GameConfig's.
    team_sizes: team-size sampling (`TeamSizes`): each episode plays a drawn N-against-N
        game, the rest of each team's slots benched; None plays every slot (the
        GameConfig's size). The runtime's draw, not the game's (`bnb.runtime.rollout`).
    """

    name: str
    share: float
    seats: tuple[str, ...]
    fog_share: float = 1.0
    mode: str | None = None
    arena_radius: float | None = None  # m
    spawn_groups: tuple[int, ...] | None = None
    episode_seconds: float | None = None
    respawn_seconds: float | None = None
    arena_scale: ArenaScale | None = None
    active_fields: ActiveFields | None = None
    team_sizes: TeamSizes | None = None


GAME_FIELDS = (
    "mode",
    "arena_radius",
    "spawn_groups",
    "episode_seconds",
    "respawn_seconds",
    "active_fields",
)


def pool_game(config: GameConfig, pool: Pool) -> GameConfig:
    """The game a pool plays: the config with the pool's settings where it gives them. The
    config validates the result (spawn groups only in team deathmatch, and so on). An
    elimination pool inherits no respawn wait (it has none)."""
    given = {name: getattr(pool, name) for name in GAME_FIELDS if getattr(pool, name) is not None}
    if given.get("mode", config.mode) == ELIMINATION and pool.respawn_seconds is None:
        given["respawn_seconds"] = None
    return dataclasses.replace(config, **given)


def environment_multiple(pools: tuple[Pool, ...]) -> int:
    """The smallest batch the pools split into whole environments (with whole fogged
    shares); every batch size they accept is a multiple of it."""
    multiple = 1
    for pool in pools:
        share = Fraction(pool.share).limit_denominator(1 << 20)
        fogged = share * Fraction(pool.fog_share).limit_denominator(1 << 20)
        multiple = math.lcm(multiple, share.denominator, fogged.denominator)
    return multiple


def environments_per_pool(pools: tuple[Pool, ...], environments: int) -> tuple[int, ...]:
    """Environments per pool. Each share times environments must be a whole number and the
    shares must sum to 1, or this fails."""
    sizes = []
    for pool in pools:
        size = pool.share * environments
        if abs(size - round(size)) > 1e-6:
            raise ValueError(
                f"pool {pool.name}: {pool.share} x {environments} environments is not whole"
            )
        sizes.append(round(size))
        fogged = pool.fog_share * round(size)
        if not 0.0 <= pool.fog_share <= 1.0 or abs(fogged - round(fogged)) > 1e-6:
            raise ValueError(
                f"pool {pool.name}: fog share {pool.fog_share} x {round(size)} environments "
                "is not whole"
            )
    if sum(sizes) != environments:
        raise ValueError(f"pool shares cover {sum(sizes)} of {environments} environments")
    return tuple(sizes)
