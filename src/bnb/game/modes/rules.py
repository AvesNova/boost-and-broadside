"""The static rules of each mode (modes spec 1): plain Python numbers, like `Rules`.

A mode is a composition of rule components: its clock (episode and respawn lengths), its
win rule, its zones and capture, its spawn placement and its spawn protection. A batch
holds one or several games (one per pool), all at one size.
"""

import dataclasses
import math

from bnb.contracts import GameConfig
from bnb.contracts.config import (
    ELIMINATION,
    FRONTLINE,
    SANDBOX,
    TEAM_DEATHMATCH,
    FixedField,
    FixedSpawn,
    MapLayout,
)

from .. import ships

# The mode's clock when the config leaves it out: (episode, respawn) seconds. Sandbox keeps
# the physics game's (game spec 1), so the race tests run in it unchanged. Elimination has
# no respawn (None): a dead ship's wait is the episode's whole length, so it is still
# waiting when the episode ends (`bnb.game.rules`). Its episode is short (the author,
# 2026-11-12): 90 s here, 180 s at real scale. The respawn waits (the author, 2026-11-13):
# 8 s at real scale in team deathmatch (was 10 s) and Frontline's minimum 8 s (was a fixed
# 16 s), after which a Frontline ship waits for its team's next respawn wave
# (`ModeRules.wave_seconds`); the standard game's clock is half the real-scale one.
CLOCKS = {
    TEAM_DEATHMATCH: (300.0, 4.0),
    FRONTLINE: (300.0, 4.0),
    ELIMINATION: (90.0, None),
    SANDBOX: (60.0, 8.0),
}

# Training presets, all 5v5 at 10 fields (modes spec 1; mixed spec 1: the battle map keeps
# 10 fields, zoomed with the arena). The real-scale ones play the real-scale ship set
# (docs/explanation/game/real-scale.md): the map 3x the standard one in metres (map scale 3)
# and the mode's clock doubled, as crossing between fields takes about twice as long.
REAL_SCALE = "real-scale"
REAL_SCALE_ARENA = 3.0 * 2600.0  # m
REAL_SCALE_CLOCK = 2.0  # rule times against the standard game's
PRESETS = {
    "deathmatch": GameConfig(5, 10, 2600.0, mode=TEAM_DEATHMATCH),
    "deathmatch_wide": GameConfig(5, 10, 8200.0, mode=TEAM_DEATHMATCH, spawn_groups=(2, 2, 1)),
    "frontline": GameConfig(5, 10, 2600.0, mode=FRONTLINE),
    # Elimination on the standard map (game v12): a rating game, not in the training mix.
    "elimination": GameConfig(5, 10, 2600.0, mode=ELIMINATION),
    **{
        f"real_scale_{name}": GameConfig(
            5,
            10,
            REAL_SCALE_ARENA,
            episode_seconds=REAL_SCALE_CLOCK * CLOCKS[mode][0],
            respawn_seconds=REAL_SCALE_CLOCK * CLOCKS[mode][1],
            mode=mode,
            ship_set=REAL_SCALE,
        )
        for name, mode in (("deathmatch", TEAM_DEATHMATCH), ("frontline", FRONTLINE))
    },
    # Elimination at the training size (5v5, 10 random fields, the real-scale map): the
    # mode's own rules, ready for the training mix (its share there is the author's call).
    "real_scale_elimination": GameConfig(
        5,
        10,
        REAL_SCALE_ARENA,
        episode_seconds=REAL_SCALE_CLOCK * CLOCKS[ELIMINATION][0],
        mode=ELIMINATION,
        ship_set=REAL_SCALE,
    ),
}

# The arena: a 2v2 team deathmatch of the real-scale set on a small map with no fields (and
# no zones: they are frontline's), a ship killed spawning again on the next tick (one tick
# of the game's 30 Hz). Not a training preset. Its first radius, 1300 m, came from the two
# ships' numbers: a cruise near 162 m/s, a gun range near 600 m and a full-stick turn radius
# of 500-600 m (cruise over turn rate), so the map was about two turn circles across and a
# turning fight met the boundary. The author widened it by half to 1950 m (2026-11-11),
# about three turn circles across. The scripted agent's 2v2s on it (3 seeds, 40 s each, the
# shield's recharge pause in play) come within gun range 0-12 s after the start and first
# kill 9-23 s in, with 3-10% of the ships' time beyond the boundary; at 1300 m the same
# matches take 0-7.5 s and 6-14 s with 10-16% outside.
ARENA_RADIUS = 1950.0  # m
PRESETS["arena"] = GameConfig(
    2,
    0,
    ARENA_RADIUS,
    episode_seconds=REAL_SCALE_CLOCK * CLOCKS[TEAM_DEATHMATCH][0],
    respawn_seconds=1.0 / 30.0,  # one tick (`Rules.tick_length`)
    mode=TEAM_DEATHMATCH,
    ship_set=REAL_SCALE,
)

# Clover (the author's map, 2026-11-12): a 2v2 elimination of the real-scale set on a fixed
# map, a showcase rather than a training preset. The arena is the 2v2 size of the
# real-scale map by the fleet presets' density rule (`<base>_<n>v<n>_density`: the radius
# grows as sqrt(n / 5)), 7800 sqrt(2 / 5) = 4933.2 m. Four fields in a square around the
# centre, each of radius a quarter of the arena radius (1233 m), their centres at 0.45
# arena radii on the diagonals: upper right and lower left high index (level +1 or +2, the
# pair drawn together each match), upper left and lower right low (-1 or -2), so four
# combinations. Each team's two spawns lie at 0.875 arena radii between a low and a high
# field, at 1 and 4 o'clock for team 0 and at 7 and 10 for team 1 (their 180-degree
# images), each ship nose to the centre with up to 150 m and 5 degrees of jitter. The
# whole layout turns by a random angle each match, as the random maps' fields and spawn
# axis do.
CLOVER_RADIUS = round(REAL_SCALE_ARENA * (2 / 5) ** 0.5, 1)  # m
CLOVER_FIELD_RADIUS = 0.25 * CLOVER_RADIUS
CLOVER_FIELD_DISTANCE = 0.45 * CLOVER_RADIUS  # the centres' distance from the map centre
CLOVER_SPAWN_DISTANCE = 0.875 * CLOVER_RADIUS
HIGH_LEVELS, LOW_LEVELS = (1, 2), (-1, -2)


def _polar(distance: float, degrees: float) -> tuple[float, float]:
    angle = math.radians(degrees)
    return distance * math.cos(angle), distance * math.sin(angle)


CLOVER = MapLayout(
    fields=(
        FixedField(*_polar(CLOVER_FIELD_DISTANCE, 45.0), CLOVER_FIELD_RADIUS, HIGH_LEVELS),
        FixedField(*_polar(CLOVER_FIELD_DISTANCE, 135.0), CLOVER_FIELD_RADIUS, LOW_LEVELS),
    ),
    spawns=tuple(
        FixedSpawn(
            *_polar(CLOVER_SPAWN_DISTANCE, degrees),
            position_jitter=150.0,
            heading_jitter=math.radians(5.0),
        )
        for degrees in (60.0, -30.0)  # 1 and 4 o'clock
    ),
)
PRESETS["clover"] = GameConfig(
    2,
    4,
    CLOVER_RADIUS,
    episode_seconds=REAL_SCALE_CLOCK * CLOCKS[ELIMINATION][0],
    mode=ELIMINATION,
    ship_set=REAL_SCALE,
    layout=CLOVER,
)

# The presets whose lineup is not the game's default (`Rules.team_types`): each slot's ship
# type name in the preset's set, the same for both teams (`bnb.game.api.preset_lineup`
# gives the type indices). Kept beside the config, not in it: a policy file records its
# game's every config field, and one without a field is another game.
LINEUPS = {"arena": ("P-51D", "Spitfire XIV")}

# Fleet-size evaluation presets (zero-shot: training is 5v5 only), of each base game:
# `<base>_<n>v<n>_arena` keeps the base's arena, so n ships crowd it n/5 times as densely;
# `<base>_<n>v<n>_density` keeps its crowding, the arena radius growing as sqrt(n / 5) (its
# 10 fields zoom with the arena, as every map scale's do). Not training presets, and not maps
# of the play shell (`FLEET_PRESETS`).
FLEET_SIZES = (10, 20, 50)  # the largest is 10x the trained 5v5 (the zero-shot claim)
FLEET_BASES = ("deathmatch", "frontline", "real_scale_deathmatch", "real_scale_frontline")
FLEET_PRESETS: tuple[str, ...] = tuple(
    f"{base}_{n}v{n}_{kind}"
    for base in FLEET_BASES
    for n in FLEET_SIZES
    for kind in ("arena", "density")
)
for _base in FLEET_BASES:
    _game = PRESETS[_base]
    for _n in FLEET_SIZES:
        PRESETS[f"{_base}_{_n}v{_n}_arena"] = dataclasses.replace(_game, ships_per_team=_n)
        PRESETS[f"{_base}_{_n}v{_n}_density"] = dataclasses.replace(
            _game,
            ships_per_team=_n,
            arena_radius=round(_game.arena_radius * (_n / _game.ships_per_team) ** 0.5, 1),
        )
del _base, _game, _n


@dataclasses.dataclass(frozen=True)
class ModeRules:
    """name: the mode. Team deathmatch (and elimination, its variant: spawns by team side,
    and one kill per enemy ship wins): kills_per_ship, spawn_groups (ships per team per
    group; empty for uniform spawns), spawn_candidates (C, the best of C), the group centres'
    disc (group_centre_fraction of the arena radius), the members' disc (group_radius), the
    respawn ring around an anchor (anchor_distance_min to _max), and spawn_margin, how far
    inside the arena edge a grouped candidate is clipped to. Elimination spawns by team
    side (`modes.spawns.side_placement`): each team's area a disc of side_area_fraction
    arena radii around its centre at `Rules.spawn_offset` arena radii on the spawn axis,
    the noses toward the enemy area's centre within side_heading_jitter (radians).
    Frontline: the zone ring (zone_ring_fraction of the arena radius), zone_radius (m
    before map scaling), capture_seconds (a lead of one), front_threshold (captures ahead
    that win), the spawn disc (spawn_zone_fraction of the zone radius), the capture lock
    (after a capture the team that lost the zone gains no progress on it for
    lock_seconds_per_root_km x sqrt(d / 1 km) s, d the distance between adjacent zone
    centres; 0 none) and the respawn waves (each team's every wave_seconds, from a random
    phase per team at the match's start: a dead ship respawns at its team's first wave
    after its respawn wait; 0 no waves). layout: the
    game's fixed map (`MapLayout`), or None for a random one; part of the mode's rules, so
    games of one mode with different layouts are different slices of the batch
    (`layout.per_mode`)."""

    name: str
    kills_per_ship: int = 5
    spawn_groups: tuple[int, ...] = ()
    spawn_candidates: int = 8
    group_centre_fraction: float = 0.85
    group_radius: float = 300.0  # m
    anchor_distance_min: float = 250.0  # m
    anchor_distance_max: float = 600.0  # m
    spawn_margin: float = 50.0  # m: grouped candidates are kept this far inside the arena
    zone_ring_fraction: float = 0.615
    zone_radius: float = 330.0  # m, before map scaling
    capture_seconds: float = 10.0
    front_threshold: int = 3
    spawn_zone_fraction: float = 0.35
    # The author, 2026-11-13: about 45 s at 5v5 on the real-scale map (d = 5.64 km), 54 s
    # at 10v10 and 64 s at 20v20 (bench/frontline_design.py: the hard lock that gives an
    # even P(same team captures next)); to be calibrated again on the final ship card.
    lock_seconds_per_root_km: float = 19.0  # s / sqrt(km)
    wave_seconds: float = 10.0  # s: 20 s at real scale (the set's [mode])
    side_area_fraction: float = 0.15
    side_heading_jitter: float = math.radians(15.0)
    layout: MapLayout | None = None

    def uniform_count(self, num_ships: int) -> int:
        """The mode's spawn uniforms per environment per tick, after the game's (spec 2).

        Team deathmatch: per ship C radii, C angles and a heading, plus an anchor when
        grouped; per environment C centre candidates (2 each) per group. Frontline: the
        ring's rotation and handedness, each team's first respawn wave, and per ship a point
        in the spawn disc."""
        candidates, groups = self.spawn_candidates, len(self.spawn_groups)
        if self.name in (TEAM_DEATHMATCH, ELIMINATION):
            return 2 * candidates * groups + num_ships * (2 * candidates + 1 + (1 if groups else 0))
        if self.name == FRONTLINE:
            return 4 + 2 * num_ships
        return 0


def mode_rules_of(config: GameConfig) -> ModeRules:
    """The ModeRules of the config's mode: its name, kills_per_ship (1 in elimination: each
    enemy ship once, a wipe), spawn_groups and layout from the config, the fields its ship
    set's rules give ([mode] in `ships.rules`), the others their defaults."""
    given = {
        key.split(".", 1)[1]: value
        for key, value in ships.rules(config.ship_set).items()
        if key.startswith("mode.")
    }
    return ModeRules(
        name=config.mode,
        kills_per_ship=1 if config.mode == ELIMINATION else config.kills_per_ship,
        spawn_groups=config.spawn_groups,
        layout=config.layout,
        **given,
    )
