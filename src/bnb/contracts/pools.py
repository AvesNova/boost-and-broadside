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


# Team-size sampling's ship rules (`TeamSizes.rule`) and field rules (`TeamSizes.fields`).
SIZE_RULES = ("uniform_pairs", "independent_teams", "near_full", "table")
FIELD_RULES = ("per_ship", "game")


@dataclasses.dataclass(frozen=True)
class TeamSizes:
    """Team-size sampling (arm A2): each episode's two team sizes and its active field count,
    drawn together at every reset from a table of (team 0's ships, team 1's ships, fields)
    with their probabilities (`table`). A team's slots past its size are benched
    (`bnb.game.api`: held as dead ships, out of play, shown with the benched flag), and the
    fields past the count are absent, so the batch keeps its one static size. The draw is
    the runtime's (`bnb.runtime.rollout`), keyed by (seed, environment, episode). The
    equations are a proposal to the author (docs/rewrite/a2-sampling-proposal.md).

    The ships, over the games of S slots a side whose teams are at least `low` and the
    mode's smallest team (`bnb.game.api.MINIMUM_SHIPS_PER_TEAM`: Frontline 3, the others
    1), as unordered pairs (larger a, smaller b), each uneven pair then split evenly between
    the two orientations:
    - "uniform_pairs": S against S with `full_share`, the rest spread evenly over the other
      pairs (with `uneven` False, over the N against N games only);
    - "independent_teams": each team's size drawn on its own, S with chance
      sqrt(full_share) (so S against S has full_share), the other sizes alike;
    - "near_full": S against S with `full_share`, the rest in proportion to
      decay^(2S - a - b), the missing ships' count (with `uneven` False, N against N only);
    - "table": `pairs`, (a, b, weight) each, renormalised (full_share unused).
    The fields, of the game's M slots (point-symmetric pairs, `ActiveFields`: an even M plays
    even counts only):
    - "per_ship": the count M (a + b) / (2S) to the nearest playable count, ties up (all M at
      S against S), or with `field_spread` k > 0 drawn evenly from the k playable counts
      either side of it, within 0 to M;
    - "game": the game's own (every field, or the pool's `active_fields` draw), drawn
      apart from the ships.
    series: the training log's series of the drawn sizes per mode
    (`bnb.runtime.analytics.team_size_series`)."""

    rule: str = "uniform_pairs"
    full_share: float = 0.5
    low: int = 1
    uneven: bool = True
    decay: float = 0.5
    pairs: tuple[tuple[int, int, float], ...] | None = None
    fields: str = "per_ship"
    field_spread: int = 0
    series: bool = True

    def __post_init__(self) -> None:
        if self.rule not in SIZE_RULES:
            raise ValueError(f"{self}: rule {self.rule!r} is not one of {SIZE_RULES}")
        if self.fields not in FIELD_RULES:
            raise ValueError(f"{self}: fields {self.fields!r} is not one of {FIELD_RULES}")
        if not 0.0 < self.full_share <= 1.0:
            raise ValueError(f"{self}: need 0 < full_share <= 1")
        if self.low < 1 or self.field_spread < 0 or not 0.0 < self.decay:
            raise ValueError(f"{self}: need low >= 1, field_spread >= 0 and decay > 0")
        if (self.rule == "table") != (self.pairs is not None):
            raise ValueError(f"{self}: pairs go with the rule 'table', and only with it")
        if self.rule == "independent_teams" and not self.uneven:
            raise ValueError(f"{self}: independent teams are uneven by nature")
        if self.pairs is not None:
            for larger, smaller, weight in self.pairs:
                if not 1 <= smaller <= larger or weight < 0.0:
                    raise ValueError(f"{self}: a pair needs 1 <= smaller <= larger, weight >= 0")
            if not sum(weight for _, _, weight in self.pairs) > 0.0:
                raise ValueError(f"{self}: the pairs' weights are all 0")

    def pair_probabilities(
        self, ships_per_team: int, minimum: int = 1
    ) -> dict[tuple[int, int], float]:
        """{(larger, smaller): probability} for a game of ships_per_team slots a side whose
        mode's smallest team is `minimum`. Raises ValueError where no pair is left."""
        S, smallest = ships_per_team, max(self.low, minimum)
        if smallest > S:
            raise ValueError(f"{self}: the smallest team {smallest} is past the game's {S} ships")
        allowed = [
            (a, b)
            for a in range(smallest, S + 1)
            for b in range(smallest, a + 1)
            if self.uneven or a == b
        ]
        if self.rule == "table":
            weights = {}
            for a, b, weight in self.pairs:
                if a > S:
                    raise ValueError(f"{self}: {a} ships is past the game's {S} ships per team")
                if b >= smallest and weight > 0.0:
                    weights[(a, b)] = weights.get((a, b), 0.0) + weight
        elif self.rule == "independent_teams":
            sizes = range(smallest, S + 1)
            full = math.sqrt(self.full_share)
            rest = (1.0 - full) / (S - smallest) if S > smallest else 0.0
            team = {n: full if n == S or S == smallest else rest for n in sizes}
            weights = {}
            for first in sizes:
                for second in sizes:
                    pair = (max(first, second), min(first, second))
                    weights[pair] = weights.get(pair, 0.0) + team[first] * team[second]
        else:
            others = [pair for pair in allowed if pair != (S, S)]
            if self.rule == "uniform_pairs":
                spread = {pair: 1.0 for pair in others}
            else:  # near_full
                spread = {(a, b): self.decay ** (2 * S - a - b) for a, b in others}
            total = sum(spread.values())
            full = self.full_share if others else 1.0
            weights = {(S, S): full}
            weights.update({pair: (1.0 - full) * w / total for pair, w in spread.items()})
        weights = {pair: w for pair, w in weights.items() if w > 0.0}
        total = sum(weights.values())
        if not total > 0.0:
            raise ValueError(f"{self}: no team size of at least {smallest} has weight")
        return {pair: w / total for pair, w in sorted(weights.items(), reverse=True)}

    def field_counts(self, ships: int, ships_per_team: int, num_fields: int) -> dict[int, float]:
        """{field count: probability} of a game of `ships` ships in all (both teams), of
        ships_per_team slots a side and num_fields field slots, by the "per_ship" rule."""
        step = 1 if num_fields % 2 else 2
        target = num_fields * ships / (2 * ships_per_team)
        nearest = min(num_fields, step * math.floor(target / step + 0.5))
        counts = [
            nearest + step * offset
            for offset in range(-self.field_spread, self.field_spread + 1)
            if 0 <= nearest + step * offset <= num_fields
        ]
        return {count: 1.0 / len(counts) for count in counts}

    def table(
        self, ships_per_team: int, num_fields: int, minimum: int = 1
    ) -> tuple[tuple[int, int, int, float], ...]:
        """The draw's table: (team 0's ships, team 1's ships, fields, probability), the
        probabilities summing to 1; fields -1 where the game draws its own ("game")."""
        rows = []
        for (a, b), chance in self.pair_probabilities(ships_per_team, minimum).items():
            sides = ((a, b),) if a == b else ((a, b), (b, a))
            if self.fields == "game":
                fields = {-1: 1.0}
            else:
                fields = self.field_counts(a + b, ships_per_team, num_fields)
            for first, second in sides:
                for count, share in fields.items():
                    rows.append((first, second, count, chance * share / len(sides)))
        return tuple(rows)


@dataclasses.dataclass(frozen=True)
class Pool:
    """A contiguous slice of the batch with a fixed lineup and one game.

    seats: the controller name for each role (role r plays team (r + seat) mod T, with the
        seat drawn per environment at every reset, so every controller plays both sides).
    fog_share: the fraction of the pool's environments with fog of war where its game has
        fog (`pool_fog_share`: times the game's `GameConfig.fogged_share`, 0 in team
        deathmatch); fog_share times the pool's size must be a whole number.
    mode, arena_radius, spawn_groups, episode_seconds, respawn_seconds: the pool's game
        (mixed spec 1); None takes the GameConfig's value, and for the clocks then the
        mode's default. Every environment of the pool shares them. The sizes (ships, fields) are the
        GameConfig's, shared by the whole batch.
    kills_per_ship, ticket_exponent, ticket_rounding: the pool's team deathmatch tickets
        (`GameConfig`); None takes the GameConfig's. A mixed batch's GameConfig is its first
        game's, so its deathmatch pools carry the deathmatch preset's own ticket rule.
    arena_scale: map-size sampling (`ArenaScale`): each episode's arena radius drawn about
        the pool's; None plays the pool's radius every episode (evaluation, ratings, play).
    active_fields: field-count sampling (`bnb.contracts.ActiveFields`): each episode plays a
        drawn number of the fields, the rest absent; None takes the GameConfig's.
    team_sizes: team-size sampling (`TeamSizes`): each episode plays drawn team sizes (equal
        or not) and, by default, a field count that follows them, the rest of each team's
        slots benched and the other fields absent; None plays every slot (the GameConfig's
        size). The runtime's draw, not the game's (`bnb.runtime.rollout`).
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
    kills_per_ship: int | None = None
    ticket_exponent: float | None = None
    ticket_rounding: int | None = None
    arena_scale: ArenaScale | None = None
    active_fields: ActiveFields | None = None
    team_sizes: TeamSizes | None = None


GAME_FIELDS = (
    "mode",
    "arena_radius",
    "spawn_groups",
    "episode_seconds",
    "respawn_seconds",
    "kills_per_ship",
    "ticket_exponent",
    "ticket_rounding",
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


def pool_fog_share(config: GameConfig, pool: Pool) -> float:
    """The share of the pool's environments with fog of war: the pool's fog_share times its
    game's (`pool_game`, `GameConfig.fogged_share`: 0 in team deathmatch, 1 in the other
    modes unless the config sets it)."""
    return pool.fog_share * pool_game(config, pool).fogged_share


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
