"""Live tuning's view of the rules: the numbers a running match can change, and the rules
with changed values.

A setting is named by a key, `<owner>.<field>`:

- `rules.<field>`: a shared constant of `Rules` (bullet drag, shield recharge, ...);
- `flight.<field>`: a flight constant (`FlightConstants`) every ship type shares;
- `mode.<field>`: a mode's rule (`ModeRules`), for every pool of that mode;
- `pool.arena_radius`, `pool.episode_seconds`, `pool.respawn_seconds`: each game's arena
  and clocks, the clocks in seconds (whole ticks).

Sizes (ships, fields, lineup, bullet slots, the tick, the mode's spawn draws) are not
settings: they change the state's shapes, so they take a new match. The ship card's values
are tuned through the handles (`bnb.game.handling`), not here; so are the shared values the
card scales per type (bullet spread, shield recharge, the head-on fraction, vision, the
stall angle, the induced drag and the exchanges), which are left out to keep one control
per quantity. The bullet drag is the exception: the game's drag, which sets how long a
bullet takes to cover a gun range (each type's bullet life, `handling.weapons`), while the
type's drag scale only trims its range.

A setting applies from the next tick, except where `Setting.when` is "episode": the map and
the spawns are drawn at an episode's start, so those apply from the next episode.
"""

import dataclasses
import math

import torch

from bnb.contracts import ticks_of
from bnb.contracts.config import ELIMINATION, FRONTLINE, SANDBOX, TEAM_DEATHMATCH

from .rules import LONGEST_BULLET_LIFE, FlightConstants, PoolRules, Rules

NOW, EPISODE = "now", "episode"
TEAM_DEATHMATCH_GROUP, FRONTLINE_GROUP = "Team deathmatch", "Frontline"


@dataclasses.dataclass(frozen=True)
class Setting:
    """One tunable number: its key, the panel group and label, its unit, its bounds (in its
    unit), whether it is a whole number, and when it applies (NOW or EPISODE). `ticks`: it
    is a time in seconds held to whole ticks. `modes`: the modes it belongs to (empty for
    every mode). `log`: its slider is logarithmic (a value spanning decades)."""

    key: str
    group: str
    label: str
    unit: str
    low: float
    high: float
    integer: bool = False
    when: str = NOW
    ticks: bool = False
    modes: tuple[str, ...] = ()
    log: bool = False


def _s(key, group, label, unit, low, high, **kw) -> Setting:
    return Setting(key, group, label, unit, low, high, **kw)


ARENA, COMBAT, FLIGHT, SPAWNS, MAP = (
    "Arena and clock",
    "Bullets",
    "Flight constants (every type)",
    "Spawns",
    "Map",
)
SETTINGS = (
    _s("pool.arena_radius", ARENA, "arena radius", "m", 800.0, 30_000.0),
    _s("pool.episode_seconds", ARENA, "episode length", "s", 10.0, 900.0, ticks=True),
    _s(
        "pool.respawn_seconds",
        ARENA,
        "respawn wait",
        "s",
        0.0,
        30.0,
        ticks=True,
        modes=(TEAM_DEATHMATCH, FRONTLINE, SANDBOX),  # elimination has none
    ),
    _s("rules.boundary_damage_base", ARENA, "boundary damage", "/s", 0.0, 50.0),
    _s("rules.boundary_damage_per_metre", ARENA, "boundary damage per metre", "/s/m", 0.0, 1.0),
    _s("rules.bullet_drag", COMBAT, "bullet drag", "1/m", 1e-5, 5e-3, log=True),
    _s("flight.gravity", FLIGHT, "gravity", "m/s^2", 5.0, 80.0),
    _s("flight.engine_knee", FLIGHT, "engine knee (store fraction)", "", 0.05, 0.95),
    _s("flight.store_fade_band", FLIGHT, "store fade band", "", 0.005, 0.2),
    _s("flight.nose_damping", FLIGHT, "nose damping", "", 0.2, 1.5),
    _s("flight.nose_frequency_floor", FLIGHT, "nose frequency floor", "rad/s", 0.1, 10.0),
    _s("rules.spawn_speed", SPAWNS, "spawn speed", "m/s", 0.0, 200.0, when=EPISODE),
    _s("rules.spawn_offset", SPAWNS, "spawn offset", "arena radii", 0.0, 0.95, when=EPISODE),
    _s("rules.spawn_radius", SPAWNS, "spawn radius", "m", 10.0, 600.0, when=EPISODE),
    _s("rules.spawn_turn_distance", SPAWNS, "spawn turn distance", "m", 1.0, 2000.0),
    _s("rules.field_band_width", MAP, "field band width", "m", 5.0, 200.0, when=EPISODE),
    _s(
        "mode.kills_per_ship",
        TEAM_DEATHMATCH_GROUP,
        "kills per ship to win",
        "",
        1,
        20,
        integer=True,
        modes=("team_deathmatch",),
    ),
    *(
        _s(
            f"mode.{name}",
            TEAM_DEATHMATCH_GROUP,
            label,
            unit,
            low,
            high,
            when=EPISODE,
            modes=("team_deathmatch",),
        )
        for name, label, unit, low, high in (
            ("group_centre_fraction", "group centres (arena fraction)", "", 0.1, 0.95),
            ("group_radius", "group radius", "m", 20.0, 1500.0),
            ("anchor_distance_min", "respawn distance, nearest", "m", 0.0, 2000.0),
            ("anchor_distance_max", "respawn distance, farthest", "m", 0.0, 3000.0),
            ("spawn_margin", "spawn margin", "m", 0.0, 500.0),
        )
    ),
    _s(
        "mode.zone_radius",
        FRONTLINE_GROUP,
        "zone radius (map scale 1)",
        "m",
        50.0,
        1200.0,
        modes=("frontline",),
    ),
    _s(
        "mode.zone_ring_fraction",
        FRONTLINE_GROUP,
        "zone ring (arena fraction)",
        "",
        0.1,
        0.95,
        modes=("frontline",),
    ),
    _s(
        "mode.capture_seconds",
        FRONTLINE_GROUP,
        "capture time",
        "s",
        1.0,
        60.0,
        modes=("frontline",),
    ),
    _s(
        "mode.front_threshold",
        FRONTLINE_GROUP,
        "captures ahead to win",
        "",
        1,
        10,
        integer=True,
        modes=("frontline",),
    ),
    _s(
        "mode.spawn_zone_fraction",
        FRONTLINE_GROUP,
        "spawn disc (zone fraction)",
        "",
        0.05,
        1.0,
        when=EPISODE,
        modes=("frontline",),
    ),
)
BY_KEY = {setting.key: setting for setting in SETTINGS}


def most_bullet_slots(tick_length: float) -> int:
    """The bullet slots a ship needs at the fastest fire rate (a shot every tick) and the
    longest bullet life a card may have: what a match that may be tuned allocates."""
    return longest_moves(tick_length)


def longest_moves(tick_length: float) -> int:
    """LONGEST_BULLET_LIFE in moves at this tick."""
    return math.ceil(round(LONGEST_BULLET_LIFE / tick_length, 9))


def settings_for(rules: Rules) -> tuple[Setting, ...]:
    """The settings these rules' games use: every shared one, and each mode's own for the
    modes they play."""
    names = {pool.mode.name for pool in rules.pools}
    return tuple(s for s in SETTINGS if not s.modes or names & set(s.modes))


def value(rules: Rules, key: str) -> float:
    """A setting's value in these rules (a pool's or a mode's from the first pool that has
    it)."""
    owner, name = key.split(".", 1)
    if owner == "rules":
        return float(getattr(rules, name))
    if owner == "flight":
        return float(getattr(rules.flight, name))
    modes = BY_KEY[key].modes
    pools = [pool for pool in rules.pools if not modes or pool.mode.name in modes]
    if not pools:
        raise KeyError(f"{key}: no game of these rules plays a mode it belongs to")
    pool = pools[0]
    if owner == "pool":
        if name == "arena_radius":
            return pool.arena_radius
        ticks = pool.episode_ticks if name == "episode_seconds" else pool.respawn_ticks
        return ticks * rules.tick_length
    return float(getattr(pool.mode, name))


def snap(setting: Setting, number: float, tick_length: float, bounded: bool = True) -> float:
    """A number as the setting holds it: within its bounds (unless not `bounded`), whole
    where it is an integer or a time in whole ticks."""
    if bounded:
        number = min(max(number, setting.low), setting.high)
    if setting.integer:
        return float(round(number))
    if setting.ticks:
        return max(round(number / tick_length), 0 if setting.low <= 0.0 else 1) * tick_length
    return number


def tuned(rules: Rules, values: dict[str, float], bounded: bool = True) -> Rules:
    """These rules with the settings' values replaced (keys of SETTINGS; times snapped to
    whole ticks, and every value held within its setting's bounds unless not `bounded`: a
    time scale may carry values past a slider's ends). Raises KeyError for an unknown key."""
    shared, flight, modes, pools = {}, {}, {}, {}
    for key, number in values.items():
        setting = BY_KEY[key]
        number = snap(setting, number, rules.tick_length, bounded)
        owner, name = key.split(".", 1)
        if setting.integer:
            number = int(number)
        if owner == "rules":
            shared[name] = number
        elif owner == "flight":
            flight[name] = number
        elif owner == "mode":
            modes[name] = number
        else:
            pools[name] = number

    def pool_rules(pool: PoolRules) -> PoolRules:
        mode = pool.mode
        own = {
            name: number
            for name, number in modes.items()
            if not BY_KEY[f"mode.{name}"].modes or mode.name in BY_KEY[f"mode.{name}"].modes
        }
        changes = {"mode": dataclasses.replace(mode, **own)} if own else {}
        if "arena_radius" in pools:
            changes["arena_radius"] = pools["arena_radius"]
        if "episode_seconds" in pools:
            changes["episode_ticks"] = ticks_of(pools["episode_seconds"], rules.tick_length)
            if mode.name == ELIMINATION:  # no respawn: the wait is the episode's length
                changes["respawn_ticks"] = changes["episode_ticks"]
        if "respawn_seconds" in pools and mode.name in BY_KEY["pool.respawn_seconds"].modes:
            changes["respawn_ticks"] = ticks_of(pools["respawn_seconds"], rules.tick_length)
        return dataclasses.replace(pool, **changes) if changes else pool

    flight_constants: FlightConstants = dataclasses.replace(rules.flight, **flight)
    return dataclasses.replace(
        rules,
        pools=tuple(pool_rules(pool) for pool in rules.pools),
        flight=flight_constants,
        **shared,
    )


def changed(rules: Rules, base: Rules) -> dict[str, float]:
    """The settings whose values differ between two rules of the same games (a preset's
    contents)."""
    return {
        s.key: value(rules, s.key)
        for s in settings_for(rules)
        if abs(value(rules, s.key) - value(base, s.key)) > 1e-12 * max(1.0, abs(value(base, s.key)))
    }


# ---------------------------------------------------------------- live values
# A compiled tick that takes the settings' values as tensors serves every value of them: a
# setting changes data, not the graph. `live_of` reads the values a running match passes
# in, `static_of` the rules with them blanked (what the compiled tick is keyed on), and
# `with_live` the rules rebuilt from the two (inside the tick).

LIVE_RULES = (
    "boundary_damage_base",
    "boundary_damage_per_metre",
    "bullet_drag",
    "spawn_speed",
    "spawn_offset",
    "spawn_radius",
    "spawn_turn_distance",
    "field_band_width",
)
LIVE_FLIGHT = ("gravity", "engine_knee", "store_fade_band", "nose_damping", "nose_frequency_floor")
LIVE_MODE = (
    "kills_per_ship",
    "group_centre_fraction",
    "group_radius",
    "anchor_distance_min",
    "anchor_distance_max",
    "spawn_margin",
    "zone_radius",
    "zone_ring_fraction",
    "capture_seconds",
    "front_threshold",
    "spawn_zone_fraction",
)
LIVE_POOL = ("arena_radius", "episode_ticks", "respawn_ticks")
_INTEGERS = {"kills_per_ship", "front_threshold", "episode_ticks", "respawn_ticks"}


def _tensor(name: str, number) -> torch.Tensor:
    return torch.tensor(number, dtype=torch.int32 if name in _INTEGERS else torch.float32)


def live_of(rules: Rules) -> dict[str, torch.Tensor]:
    """The settings' values of these rules as 0-d tensors, by path ("rules.<field>",
    "flight.<field>", "pool<i>.<field>", "pool<i>.mode.<field>")."""
    live = {f"rules.{name}": _tensor(name, getattr(rules, name)) for name in LIVE_RULES}
    live |= {f"flight.{name}": _tensor(name, getattr(rules.flight, name)) for name in LIVE_FLIGHT}
    for index, pool in enumerate(rules.pools):
        live |= {f"pool{index}.{name}": _tensor(name, getattr(pool, name)) for name in LIVE_POOL}
        live |= {
            f"pool{index}.mode.{name}": _tensor(name, getattr(pool.mode, name))
            for name in LIVE_MODE
        }
    return live


def static_of(rules: Rules) -> Rules:
    """The rules with every live value blanked: rules that differ only in live values have
    the same static rules (one compiled tick serves them all)."""
    return with_live(rules, {path: 0 for path in live_of(rules)})


def with_live(rules: Rules, live: dict) -> Rules:
    """These rules with the live values in `live` (tensors in a tick, or numbers)."""

    def part(prefix: str) -> dict:
        return {
            path[len(prefix) :]: value
            for path, value in live.items()
            if path.startswith(prefix) and "." not in path[len(prefix) :]
        }

    pools = tuple(
        dataclasses.replace(
            pool,
            mode=dataclasses.replace(pool.mode, **part(f"pool{index}.mode.")),
            **part(f"pool{index}."),
        )
        for index, pool in enumerate(rules.pools)
    )
    flight = dataclasses.replace(rules.flight, **part("flight."))
    return dataclasses.replace(rules, pools=pools, flight=flight, **part("rules."))
