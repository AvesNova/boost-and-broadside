"""The static rules of each mode (modes spec 1): plain Python numbers, like `Rules`.

A mode is a composition of rule components: its clock (episode and respawn lengths), its
win rule, its zones and capture, its spawn placement and its spawn protection. A batch
holds one or several games (one per pool), all at one size.
"""

import dataclasses

from bnb.contracts import GameConfig

TEAM_DEATHMATCH, FRONTLINE, SANDBOX = "team_deathmatch", "frontline", "sandbox"
# The mode's clock when the config leaves it out: (episode, respawn) seconds. Sandbox keeps
# the physics game's (game spec 1), so the race tests run in it unchanged.
CLOCKS = {TEAM_DEATHMATCH: (300.0, 5.0), FRONTLINE: (300.0, 8.0), SANDBOX: (60.0, 8.0)}
FRONTLINE_ZONES = 5
# Every environment of every mode holds this many zone slots, a fixed shape of the game like
# the field count, so a model sees the same tokens in any batch; a mode without zones leaves
# them absent.
ZONE_SLOTS = FRONTLINE_ZONES

# Training presets, all 5v5 at 10 fields (modes spec 1; mixed spec 1: the battle map keeps
# 10 fields, zoomed with the arena).
PRESETS = {
    "deathmatch": GameConfig(5, 10, 2600.0, mode=TEAM_DEATHMATCH),
    "deathmatch_wide": GameConfig(5, 10, 8200.0, mode=TEAM_DEATHMATCH, spawn_groups=(2, 2, 1)),
    "frontline": GameConfig(5, 10, 2600.0, mode=FRONTLINE),
}


@dataclasses.dataclass(frozen=True)
class ModeRules:
    """name: the mode. Team deathmatch: kills_per_ship, spawn_groups (ships per team per
    group; empty for uniform spawns), spawn_candidates (K, best of K), the group centres'
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

        Team deathmatch: per ship K radii, K angles and a heading, plus an anchor when
        grouped; per environment K centre candidates (2 each) per group. Frontline: the
        ring's rotation and handedness, and per ship a point in the spawn disc."""
        K, groups = self.spawn_candidates, len(self.spawn_groups)
        if self.name == TEAM_DEATHMATCH:
            return 2 * K * groups + num_ships * (2 * K + 1 + (1 if groups else 0))
        if self.name == FRONTLINE:
            return 2 + 2 * num_ships
        return 0


def mode_rules_of(config: GameConfig) -> ModeRules:
    return ModeRules(
        name=config.mode, kills_per_ship=config.kills_per_ship, spawn_groups=config.spawn_groups
    )
