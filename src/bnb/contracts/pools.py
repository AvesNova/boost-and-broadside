"""Lineups and games: which controller plays which role, in which game, for what share of the
environments."""

import dataclasses
import math
from fractions import Fraction

from .config import ELIMINATION, GameConfig


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


GAME_FIELDS = ("mode", "arena_radius", "spawn_groups", "episode_seconds", "respawn_seconds")


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
