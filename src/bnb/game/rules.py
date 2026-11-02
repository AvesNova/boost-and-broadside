"""The fixed numerics of the game (game spec 1, 3.2): plain Python numbers.

Compiled graphs specialize on these, so they cost nothing at run time. The ship-card
parameters that differ by ship type are tensors instead (`ship_types.ShipTypes`), so one
compiled graph serves any values of them. A game has exactly two teams (author decision).

A batch may hold several games (mixed spec): `Rules.pools` lists them as static, contiguous
slices of the batch, each with its mode, arena and clocks. The sizes are shared.
"""

import dataclasses
import math

from bnb.contracts import GameConfig, Pool, ticks_of
from bnb.contracts.pools import environments_per_pool, pool_game

from .modes.rules import CLOCKS, ZONE_SLOTS, ModeRules, mode_rules_of

TEAMS = 2
STICK_LEVELS = (0.0, -0.4, 0.4, -0.75, 0.75, -1.0, 1.0)  # by stick index; + turns left
POWER_MODES = (0.0, -1.0, 1.0)  # by power index: coast, boost, reverse
FIELD_LEVELS = (-2, -1, 1, 2)
LN_SQRT2 = 0.5 * math.log(2.0)
MAP_SCALE_RADIUS = 2600.0  # the map scale is arena_radius / 2600
BULLET_LIFE = 0.4  # s, every bullet's: 12 moves at 30 Hz


@dataclasses.dataclass(frozen=True)
class FlightConstants:
    """Game spec 3.2, the flight constants every ship type shares (spec symbol in brackets)."""

    mass: float = 1.0  # kg [m]
    gravity: float = 27.0  # m/s^2, the lift that holds level flight [g]
    stall_angle: float = math.radians(15.0)  # [alpha_s]
    induced_drag_factor: float = 11.4  # [k]
    engine_knee: float = 0.5  # store fraction where engine power starts to fall [e_k]
    store_fade_band: float = 0.02  # fraction of the maximum energy [beta]
    coast_exchange: float = 3.0  # m/s^2 [a_0]
    climb_exchange: float = 27.0  # m/s^2, reverse [a_c]
    dive_exchange: float = 27.0  # m/s^2, boost [a_d]
    lift_softmin_exponent: float = 16.0  # [p]
    nose_damping: float = 0.7  # damping ratio [zeta]
    nose_frequency_floor: float = 1.0  # rad/s [omega_min]
    speed_floor: float = 1e-6  # m/s [epsilon]


@dataclasses.dataclass(frozen=True)
class PoolRules:
    """One game of the batch (mixed spec 1): `environments` contiguous environments (None:
    the whole batch, at any size) playing `mode` in an arena of this radius, with these
    clocks. Adjacent pools of the same game are one PoolRules."""

    environments: int | None
    mode: ModeRules
    arena_radius: float  # m
    episode_ticks: int
    respawn_ticks: int

    @property
    def game(self) -> tuple:
        """Everything but the size: pools that agree on it play the same game."""
        return self.mode, self.arena_radius, self.episode_ticks, self.respawn_ticks


@dataclasses.dataclass(frozen=True)
class Rules:
    """Sizes and every shared constant of a batch of games (game spec 1), and the games, as
    static slices of the batch (`pools`). Static: a different value is a different compiled
    graph. Built by `rules_of`."""

    ships_per_team: int
    num_fields: int
    pools: tuple[PoolRules, ...]
    team_types: tuple[int, ...] = ()  # ship type per slot, the same for both teams
    bullet_slots: int = 6  # per ship [K]
    tick_length: float = 1.0 / 30.0  # s [h]
    bullet_moves: int = 12  # a bullet's lifetime, BULLET_LIFE in ticks (rules_of converts)
    bullet_drag: float = 8e-4  # 1/m [k_b]
    bullet_spread: float = 12.0  # m/s per axis, proper [sigma_b]
    head_on_fraction: float = 0.3  # [f_0]
    shield_recharge_delay: float = 6.0  # s
    shield_recharge_rate: float = 12.0  # per s
    boundary_damage_base: float = 5.0  # per s
    boundary_damage_per_metre: float = 0.05  # per s per m of depth
    spawn_speed: float = 30.0  # m/s, proper
    spawn_offset: float = 0.8  # spawn centres at +-0.8 arena radii along the spawn axis
    spawn_radius: float = 120.0  # m, before map scaling [rho_s]
    field_band_width: float = 40.0  # m [b]
    vision_range: float = 1024.0  # m [V]
    flight: FlightConstants = FlightConstants()

    def __post_init__(self) -> None:
        if not self.team_types:  # within a team, slots alternate starting with a fighter
            object.__setattr__(self, "team_types", tuple(i % 2 for i in range(self.ships_per_team)))
        if len(self.team_types) != self.ships_per_team:
            raise ValueError("team_types needs one ship type per slot")

    @property
    def num_ships(self) -> int:
        return TEAMS * self.ships_per_team

    @property
    def slot_types(self) -> tuple[int, ...]:
        """Ship type of every sim index (team-major)."""
        return self.team_types * TEAMS

    @property
    def modes(self) -> tuple[ModeRules, ...]:
        return tuple(pool.mode for pool in self.pools)

    @property
    def num_zones(self) -> int:
        """Zone slots in every environment, whatever its mode (a mode without zones leaves
        them absent)."""
        return ZONE_SLOTS

    @property
    def map_uniform_count(self) -> int:
        """Uniforms per environment for a map: 4 per field plus the spawn axis."""
        return 4 * self.num_fields + 1

    @property
    def game_uniform_count(self) -> int:
        """Uniforms per environment per tick for the game (game spec 2): the map, a
        formation rotation per team, a formation key per ship."""
        return self.map_uniform_count + TEAMS + self.num_ships

    @property
    def tick_uniform_count(self) -> int:
        """All uniforms per environment per tick: the game's, then the mode's spawn draws
        (modes spec 2), as wide as the widest mode needs; each environment reads its own
        mode's prefix (mixed spec 2)."""
        return self.game_uniform_count + max(
            mode.uniform_count(self.num_ships) for mode in self.modes
        )


def _pool_rules(environments: int | None, game: GameConfig, tick_length: float) -> PoolRules:
    """A game's mode and its time-native rules (the mode's clock where it leaves them out)."""
    episode_s, respawn_s = CLOCKS[game.mode]
    episode_s = episode_s if game.episode_s is None else game.episode_s
    respawn_s = respawn_s if game.respawn_s is None else game.respawn_s
    return PoolRules(
        environments=environments,
        mode=mode_rules_of(game),
        arena_radius=game.arena_radius,
        episode_ticks=ticks_of(episode_s, tick_length),
        respawn_ticks=ticks_of(respawn_s, tick_length),
    )


def rules_of(
    config: GameConfig, pools: tuple[Pool, ...] = (), num_envs: int | None = None, **overrides
) -> Rules:
    """The rules for a config's sizes and games; `overrides` replace defaults.

    Without pools the batch plays the config's game, at any batch size. With pools (and
    num_envs, their batch size) each pool plays its own (`pool_game`), in pool order;
    adjacent pools of the same game merge, and if one game remains it covers the whole
    batch as without pools.
    """
    tick_length = overrides.pop("tick_length", Rules.tick_length)
    if not pools:
        games = (_pool_rules(None, config, tick_length),)
    else:
        sizes = environments_per_pool(pools, num_envs)
        merged: list[PoolRules] = []
        for size, pool in zip(sizes, pools):
            game = _pool_rules(size, pool_game(config, pool), tick_length)
            if merged and merged[-1].game == game.game:
                game = dataclasses.replace(game, environments=merged.pop().environments + size)
            if game.environments:
                merged.append(game)
        if len(merged) == 1:
            merged = [dataclasses.replace(merged[0], environments=None)]
        games = tuple(merged)
    overrides.setdefault("bullet_moves", ticks_of(BULLET_LIFE, tick_length))
    return Rules(
        ships_per_team=config.ships_per_team,
        num_fields=config.num_fields,
        pools=games,
        tick_length=tick_length,
        **overrides,
    )
