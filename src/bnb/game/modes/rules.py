"""The static rules of each mode (modes spec 1): plain Python numbers, like `Rules`.

A mode is a composition of rule components: its clock (episode and respawn lengths), its
win rule, its zones and capture, its spawn placement and its spawn protection. A batch
holds one or several games (one per pool), all at one size.
"""

import dataclasses

from bnb.contracts import GameConfig
from bnb.contracts.config import FRONTLINE, SANDBOX, TEAM_DEATHMATCH

from .. import ships

# The mode's clock when the config leaves it out: (episode, respawn) seconds. Sandbox keeps
# the physics game's (game spec 1), so the race tests run in it unchanged.
CLOCKS = {TEAM_DEATHMATCH: (300.0, 5.0), FRONTLINE: (300.0, 8.0), SANDBOX: (60.0, 8.0)}

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
}

# The arena: a 2v2 team deathmatch of the real-scale set on a small map with no fields (and
# no zones: they are frontline's), a ship killed spawning again on the next tick (one tick
# of the game's 30 Hz). Not a training preset. Its radius from the two ships' numbers: a
# cruise near 162 m/s, a gun range near 600 m and a full-stick turn radius of 500-600 m
# (cruise over turn rate), so the map is about two turn circles across and a turning fight
# meets the boundary. The scripted agent's 2v2s on it (3 seeds, 40 s each) come within gun
# range 0-7.5 s after the start and first kill 6-14 s in, with 7-18% of the ships' time
# beyond the boundary; at 1600 m gun range takes up to 10 s, at 1000 m a fifth of the time
# is spent outside.
ARENA_RADIUS = 1300.0  # m
PRESETS["arena"] = GameConfig(
    2,
    0,
    ARENA_RADIUS,
    episode_seconds=REAL_SCALE_CLOCK * CLOCKS[TEAM_DEATHMATCH][0],
    respawn_seconds=1.0 / 30.0,  # one tick (`Rules.tick_length`)
    mode=TEAM_DEATHMATCH,
    ship_set=REAL_SCALE,
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
FLEET_SIZES = (10, 20, 40)
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
    """name: the mode. Team deathmatch: kills_per_ship, spawn_groups (ships per team per
    group; empty for uniform spawns), spawn_candidates (C, the best of C), the group centres'
    disc (group_centre_fraction of the arena radius), the members' disc (group_radius), the
    respawn ring around an anchor (anchor_distance_min to _max), and spawn_margin, how far
    inside the arena edge a grouped candidate is clipped to. Frontline: the zone
    ring (zone_ring_fraction of the arena radius), zone_radius (m before map scaling),
    capture_seconds (a lead of one), front_threshold (captures ahead that win) and the
    spawn disc (spawn_zone_fraction of the zone radius)."""

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

    def uniform_count(self, num_ships: int) -> int:
        """The mode's spawn uniforms per environment per tick, after the game's (spec 2).

        Team deathmatch: per ship C radii, C angles and a heading, plus an anchor when
        grouped; per environment C centre candidates (2 each) per group. Frontline: the
        ring's rotation and handedness, and per ship a point in the spawn disc."""
        candidates, groups = self.spawn_candidates, len(self.spawn_groups)
        if self.name == TEAM_DEATHMATCH:
            return 2 * candidates * groups + num_ships * (2 * candidates + 1 + (1 if groups else 0))
        if self.name == FRONTLINE:
            return 2 + 2 * num_ships
        return 0


def mode_rules_of(config: GameConfig) -> ModeRules:
    """The ModeRules of the config's mode: its name, kills_per_ship and spawn_groups from the
    config, the fields its ship set's rules give ([mode] in `ships.rules`), the others their
    defaults."""
    given = {
        key.split(".", 1)[1]: value
        for key, value in ships.rules(config.ship_set).items()
        if key.startswith("mode.")
    }
    return ModeRules(
        name=config.mode,
        kills_per_ship=config.kills_per_ship,
        spawn_groups=config.spawn_groups,
        **given,
    )
