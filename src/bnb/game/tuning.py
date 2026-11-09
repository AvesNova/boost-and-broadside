"""Live tuning's view of the rules: the numbers a running match can change, and the rules
with changed values.

A setting is named by a key, `<owner>.<field>`:

- `rules.<field>`: a shared constant of `Rules` (bullet drag, shield recharge, ...);
- `flight.<field>`: a flight constant (`FlightConstants`) every ship type shares;
- `mode.<field>`: a mode's rule (`ModeRules`), for every pool of that mode;
- `pool.arena_radius`, `pool.episode_seconds`, `pool.respawn_seconds`: each game's arena
  and clocks, the clocks in seconds (whole ticks);
- `rules.bullet_life`: every bullet's life in seconds (whole ticks: `Rules.bullet_moves`).

Sizes (ships, fields, lineup, bullet slots, the tick, the mode's spawn draws) are not
settings: they change the state's shapes, so they take a new match. The ship card's values
are tuned through the handles (`bnb.game.handling`), not here; so are the shared values the
card scales per type (bullet drag and spread, shield recharge, the head-on fraction, vision,
the stall angle, the induced drag and the exchanges), which are left out to keep one
control per quantity.

A setting applies from the next tick, except where `Setting.when` is "episode": the map and
the spawns are drawn at an episode's start, so those apply from the next episode.
"""

import dataclasses

from bnb.contracts import ticks_of

from .modes.rules import ModeRules
from .rules import FlightConstants, PoolRules, Rules

NOW, EPISODE = "now", "episode"
TEAM_DEATHMATCH_GROUP, FRONTLINE_GROUP = "Team deathmatch", "Frontline"


@dataclasses.dataclass(frozen=True)
class Setting:
    """One tunable number: its key, the panel group and label, its unit, its bounds (in its
    unit), whether it is a whole number, and when it applies (NOW or EPISODE). `ticks`: it
    is a time in seconds held to whole ticks. `modes`: the modes it belongs to (empty for
    every mode)."""

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
    _s("pool.arena_radius", ARENA, "arena radius", "m", 800.0, 12_000.0),
    _s("pool.episode_seconds", ARENA, "episode length", "s", 10.0, 900.0, ticks=True),
    _s("pool.respawn_seconds", ARENA, "respawn wait", "s", 0.0, 30.0, ticks=True),
    _s("rules.boundary_damage_base", ARENA, "boundary damage", "/s", 0.0, 50.0),
    _s("rules.boundary_damage_per_metre", ARENA, "boundary damage per metre", "/s/m", 0.0, 1.0),
    _s("rules.bullet_life", COMBAT, "bullet life", "s", 0.1, 1.5, ticks=True),
    _s("flight.gravity", FLIGHT, "gravity", "m/s^2", 5.0, 80.0),
    _s("flight.engine_knee", FLIGHT, "engine knee (store fraction)", "", 0.05, 0.95),
    _s("flight.store_fade_band", FLIGHT, "store fade band", "", 0.005, 0.2),
    _s("flight.nose_damping", FLIGHT, "nose damping", "", 0.2, 1.5),
    _s("flight.nose_frequency_floor", FLIGHT, "nose frequency floor", "rad/s", 0.1, 10.0),
    _s("rules.spawn_speed", SPAWNS, "spawn speed", "m/s", 0.0, 200.0, when=EPISODE),
    _s("rules.spawn_offset", SPAWNS, "spawn offset", "arena radii", 0.0, 0.95, when=EPISODE),
    _s("rules.spawn_radius", SPAWNS, "spawn radius", "m", 10.0, 600.0, when=EPISODE),
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
        if name == "bullet_life":
            return rules.bullet_moves * rules.tick_length
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


def snap(setting: Setting, number: float, tick_length: float) -> float:
    """A number as the setting holds it: within its bounds, whole where it is an integer or
    a time in whole ticks."""
    number = min(max(number, setting.low), setting.high)
    if setting.integer:
        return float(round(number))
    if setting.ticks:
        return max(round(number / tick_length), 0 if setting.low <= 0.0 else 1) * tick_length
    return number


def tuned(rules: Rules, values: dict[str, float]) -> Rules:
    """These rules with the settings' values replaced (keys of SETTINGS; times snapped to
    whole ticks). Raises KeyError for an unknown key."""
    shared, flight, modes, pools = {}, {}, {}, {}
    for key, number in values.items():
        setting = BY_KEY[key]
        number = snap(setting, number, rules.tick_length)
        owner, name = key.split(".", 1)
        if setting.integer:
            number = int(number)
        if owner == "rules" and name == "bullet_life":
            shared["bullet_moves"] = ticks_of(number, rules.tick_length)
        elif owner == "rules":
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
        if "respawn_seconds" in pools:
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


__all__ = [
    "BY_KEY",
    "EPISODE",
    "NOW",
    "SETTINGS",
    "ModeRules",
    "Setting",
    "changed",
    "settings_for",
    "snap",
    "tuned",
    "value",
]
