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
from bnb.contracts.config import TEAMS
from bnb.contracts.obs import ZONE_SLOTS
from bnb.contracts.pools import environments_per_pool, pool_game

from . import ship_types, ships
from .modes.rules import CLOCKS, ModeRules, mode_rules_of

STICK_LEVELS = (0.0, -0.4, 0.4, -0.75, 0.75, -1.0, 1.0)  # by stick index; + turns left
POWER_MODES = (0.0, -1.0, 1.0)  # by power index: coast, boost, reverse
COAST = POWER_MODES.index(0.0)  # the power index that neither boosts nor reverses
SLIGHT_TURN = min(abs(level) for level in STICK_LEVELS if level != 0.0)  # the lightest level
# The quiet inputs, which leave the shield recharging (game spec 7.2): coasting, the stick
# straight or at a slight turn to either side, and not shooting. Any other input restarts
# its delay.
QUIET_STICKS = tuple(index for index, level in enumerate(STICK_LEVELS) if abs(level) <= SLIGHT_TURN)
FIELD_LEVELS = (-2, -1, 1, 2)  # a field's level: -2 and -1 low, 1 and 2 high; 2 strong
LN_SQRT2 = 0.5 * math.log(2.0)
MAP_SCALE_RADIUS = 2600.0  # m: the arena radius of map scale 1 (map_scale = R_a / this)
# rad: the angle off the nose at which armour is at half strength (the aspect halfway from
# the head-on fraction to 1): 42.27 degrees, the exp(-4 theta^2 / pi) curve's
HEAD_ON_ANGLE = math.sqrt(math.pi * math.log(2.0) / 4.0)
# s: the longest bullet life a card may have (a type's life follows from its gun range,
# `handling.weapons.bullet_life`); what a match that may be tuned sizes its bullet slots by
LONGEST_BULLET_LIFE = 2.0


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
    # The longest bullet life on the card, in moves (rules_of reads the card; 12 is the
    # standard card's): the static size of what follows a shot (its labels, the lookahead).
    longest_bullet_moves: int = 12
    bullet_drag: float = (
        8e-4  # 1/m [k_b], the floor of the card's (bullet_drag_scale >= 1 by design)
    )
    # m: the distance at which a card's hit damage holds (its reference bullet speed's, from a
    # shot at cruise; ship-card spec 3): 300 real metres, the real-scale sets' value, over the
    # standard metre's 5.82. Only the card's solve reads it (the game reads the card's speed).
    damage_reference_distance: float = 300.0 / 5.82
    bullet_spread: float = 12.0  # m/s per axis, proper [sigma_b]
    head_on_fraction: float = 0.3  # [f_0]
    head_on_angle: float = HEAD_ON_ANGLE  # rad, armour at half strength [theta_h]
    shield_recharge_delay: float = 6.0  # s without a hit, counted on quiet ticks (QUIET_STICKS)
    shield_recharge_rate: float = 12.0  # per s, on quiet ticks
    boundary_damage_base: float = 5.0  # per s
    boundary_damage_per_metre: float = 0.05  # per s per m of depth
    spawn_speed: float = 30.0  # m/s, proper
    spawn_offset: float = 0.8  # spawn centres at +-0.8 arena radii along the spawn axis
    spawn_radius: float = 120.0  # m, before map scaling [rho_s]
    # m: a random spawn heading's room to turn away from the edge (`modes.spawns._heading`):
    # within this distance of it a heading points outward by a cosine of at most the
    # distance over this one. The standard set's largest full-stick excursion from spawn
    # speed, straight out, is 65 m (the heavy, reversing), rounded up.
    spawn_turn_distance: float = 75.0
    field_band_width: float = 40.0  # m [b]
    # The random map's fields (`physics.spawning.new_map`). Radius: uniform between these
    # two, in metres at map scale 1 (scaled by the arena's map scale like the other map
    # lengths). Centres: uniform in a disc of this fraction of the arena radius. Levels:
    # the sign and the strength are independent draws; a field is high (+) with probability
    # `field_high_share` and strong (+-2 rather than +-1) with `field_strong_share`. A pair
    # of point-symmetric fields shares its level, so any shares leave the map fair.
    field_radius_min: float = 30.0  # m
    field_radius_max: float = 490.0  # m
    field_disc_fraction: float = 0.85
    field_high_share: float = 0.5
    field_strong_share: float = 0.5
    vision_range: float = 1024.0  # m [V]
    flight: FlightConstants = FlightConstants()

    def __post_init__(self) -> None:
        if not self.team_types:  # within a team, slots alternate starting with a fighter
            object.__setattr__(self, "team_types", tuple(i % 2 for i in range(self.ships_per_team)))
        if len(self.team_types) != self.ships_per_team:
            raise ValueError("team_types needs one ship type per slot")

    @property
    def num_ships(self) -> int:
        """N: TEAMS times ships_per_team."""
        return TEAMS * self.ships_per_team

    @property
    def slot_types(self) -> tuple[int, ...]:
        """Ship type of every sim index (team-major)."""
        return self.team_types * TEAMS

    @property
    def modes(self) -> tuple[ModeRules, ...]:
        """Each pool's ModeRules, in pool order."""
        return tuple(pool.mode for pool in self.pools)

    @property
    def num_zones(self) -> int:
        """Zone slots in every environment, whatever its mode (a mode without zones leaves
        them absent)."""
        return ZONE_SLOTS

    @property
    def map_uniform_count(self) -> int:
        """Uniforms per environment for a map: 4 per field plus the spawn axis (a fixed
        layout reads a level draw per pair of fields and the axis as its rotation)."""
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
    """A game's mode and its time-native rules (the mode's clock where it leaves them out).
    A mode without respawn (elimination: None in its clock) waits a dead ship out for the
    episode's whole length, so it never comes back before the episode ends."""
    episode_seconds, respawn_seconds = CLOCKS[game.mode]
    episode_seconds = episode_seconds if game.episode_seconds is None else game.episode_seconds
    respawn_seconds = respawn_seconds if game.respawn_seconds is None else game.respawn_seconds
    if respawn_seconds is None:
        respawn_seconds = episode_seconds
    for field in () if game.layout is None else game.layout.fields:
        if not set(field.levels) <= set(FIELD_LEVELS):
            raise ValueError(f"field levels {field.levels}: each one of {FIELD_LEVELS}")
    return PoolRules(
        environments=environments,
        mode=mode_rules_of(game),
        arena_radius=game.arena_radius,
        episode_ticks=ticks_of(episode_seconds, tick_length),
        respawn_ticks=ticks_of(respawn_seconds, tick_length),
    )


def life_moves(bullet_life, tick_length: float) -> list[int]:
    """The card's bullet lives (s, one per type, or (B, types)) as whole moves at this tick."""
    if hasattr(bullet_life, "flatten"):
        bullet_life = bullet_life.flatten().tolist()
    return [round(life / tick_length) for life in bullet_life]


def bullet_slots_for(fire_rate, bullet_life, tick_length: float) -> int:
    """The bullet slots a ship needs so that none of its bullets is replaced in flight: the
    most of any type's life in moves over its ticks between shots (fire_rate: shots per
    second, bullet_life: seconds, the card's, one per type)."""
    rates = fire_rate.flatten().tolist() if hasattr(fire_rate, "flatten") else list(fire_rate)
    return max(
        math.ceil(moves / max(round(1.0 / (tick_length * rate)), 1))
        for moves, rate in zip(life_moves(bullet_life, tick_length), rates, strict=True)
    )


def set_overrides(ship_set: str, tick_length: float) -> dict:
    """The `Rules` fields a ship set's rules.toml sets (`ships.rules`): its [rules] values
    and its [flight] constants (as `flight`). Empty for the standard set."""
    found = ships.rules(ship_set)
    shared = {
        key.split(".", 1)[1]: value for key, value in found.items() if key.startswith("rules.")
    }
    flight = {
        key.split(".", 1)[1]: value for key, value in found.items() if key.startswith("flight.")
    }
    if flight:
        shared["flight"] = dataclasses.replace(FlightConstants(), **flight)
    return shared


def rules_of(
    config: GameConfig,
    pools: tuple[Pool, ...] = (),
    environments: int | None = None,
    types=None,
    **overrides,
) -> Rules:
    """The rules for a config's sizes and games; `overrides` replace defaults.

    Without pools the batch plays the config's game, at any batch size. With pools (and
    environments, their batch size) each pool plays its own (`pool_game`), in pool order;
    adjacent pools of the same game merge, and if one game remains it covers the whole
    batch as without pools. types: the ship card (None: the config's ship set's default
    card), whose bullet lives set `longest_bullet_moves`, and which sets the bullet slots
    (`bullet_slots_for`; 6 without a card) unless `bullet_slots` is given. The config's
    ship set gives its own rule values (`set_overrides`) where `overrides` leave them out.
    """
    tick_length = overrides.pop("tick_length", Rules.tick_length)
    for name, value in set_overrides(config.ship_set, tick_length).items():
        overrides.setdefault(name, value)
    if not pools:
        games = (_pool_rules(None, config, tick_length),)
    else:
        sizes = environments_per_pool(pools, environments)
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
    lives = (
        ship_types.defaults(config.ship_set)["bullet_life"] if types is None else types.bullet_life
    )
    overrides.setdefault("longest_bullet_moves", max(life_moves(lives, tick_length)))
    if types is not None:
        overrides.setdefault(
            "bullet_slots", bullet_slots_for(types.fire_rate, types.bullet_life, tick_length)
        )
    return Rules(
        ships_per_team=config.ships_per_team,
        num_fields=config.num_fields,
        pools=games,
        tick_length=tick_length,
        **overrides,
    )
