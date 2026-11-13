"""What a team sees (`Obs`) and the truth only learners may use (`Privileged`).

An observation is raw and in world coordinates, with no rotation or mirroring. The only
canonicalization is the team: each ship row carries `team_relative = (team - viewer) mod T`,
and the ship rows are ordered by it (own ships first); zones and the mode's globals are in
the viewer's terms. So nothing reveals which seat a team has.

Columns are read by name, and a column's meaning never changes under the same name: a new
meaning is a new name. A policy file records the column names it was trained with
(`COLUMNS`, runs spec 3), so adding columns locks out no policy and does not raise
`Obs.SCHEMA_VERSION`; that rises only for a real break (a part's shape or a removed or
renamed column a policy reads).
"""

from typing import ClassVar

import torch

from .tree import contract

SHIP_COLUMNS = (
    # zero when the team cannot see the ship
    "x",
    "y",
    "heading_cos",
    "heading_sin",
    "velocity_x",
    "velocity_y",
    "proper_speed",
    "angle_of_attack",
    "nose_rate",
    "energy",
    "shield",
    "shield_recharge_wait",
    "gun_cooldown",
    "log_refractive_index",
    "log_index_gradient_x",
    "log_index_gradient_y",
    # The change over the last tick, per second, world frame: of the velocity
    # (acceleration_x, acceleration_y; m/s^2) and of the nose rate (nose_acceleration;
    # rad/s^2). acceleration_valid 1 when it is a real tick of motion the team saw: the ship
    # alive at both ticks, not spawned at this one, not the game's first tick, and seen by
    # the team at both (so a ship just spotted shows none, and nothing of the tick before
    # leaks); 0, with the three zero, otherwise.
    "acceleration_x",
    "acceleration_y",
    "nose_acceleration",
    "acceleration_valid",
    # always shown; fired: fired this tick; benched: a slot out of the match (an uneven
    # game's, `bnb.game.state.benched_of`: a team's last S - active slots), held as a dead
    # ship with respawn_wait frozen at the full respawn length; shown to both teams, fog or
    # not (both know the roster), 0 in a game with every ship active
    "alive",
    "visible",
    "fired",
    "respawn_wait",
    "ship_type",
    "team_relative",
    "time_since_seen",
    "spawned",
    "benched",
)
SHIP_HIDDEN_ZERO = 20  # the first this-many columns are zero for a ship the team cannot see
FIELD_COLUMNS = ("x", "y", "radius", "band_width", "level")
# Zones (Frontline's), for the viewer: role 0 own spawn, 1 own defense, 2 neutral, 3 enemy
# defense, 4 enemy spawn; progress, the meter in [0, 1]; direction +1 when the meter moves
# in the viewer's favour, -1 in the enemy's, 0 still; the distances: captures to the
# viewer's win and to its loss; present, 1 for a zone of the environment's mode and 0 for an
# absent slot (a mode without zones in a batch that has them), whose other columns are 0;
# lock_own and lock_enemy, the seconds left of the capture lock on the viewer's team and on
# the enemy's (the team that lost the zone gains no progress on it until 0; public, as
# every zone column is).
ZONE_COLUMNS = (
    "x",
    "y",
    "radius",
    "role",
    "progress",
    "direction",
    "offensive_distance",
    "defensive_distance",
    "present",
    "lock_own",
    "lock_enemy",
)
# The mode block: a mode one-hot (all zero in sandbox); the scores by team_relative and the
# score that wins (team deathmatch; in elimination the enemy's ships, a wipe); the front,
# positive in the viewer's favour, and the front that wins (Frontline); next_wave_own, the
# seconds to the viewer's team's next respawn wave (Frontline; the enemy's wave clock is
# not shown). Absent quantities are 0. Before the mode block, the boundary:
# boundary_fraction, its radius now over the arena radius (1 but where it shrinks, in
# elimination: positions and map lengths stay in metres, and arena_radius stays the
# arena's; 0 once it has passed radius 0, the game's boundary going on to a ramp width
# inside the centre), and boundary_time_to_zero, the seconds until it reaches radius 0
# (0 after; 0 where it does not shrink).
GLOBAL_COLUMNS = (
    "episode_progress",
    "arena_radius",
    "fogged",
    "reset",
    "boundary_fraction",
    "boundary_time_to_zero",
    "mode_team_deathmatch",
    "mode_frontline",
    "mode_elimination",
    "score_own",
    "score_enemy",
    "score_target",
    "front",
    "front_threshold",
    "next_wave_own",
)
# Times (shield_recharge_wait, gun_cooldown, respawn_wait, time_since_seen, lock_own,
# lock_enemy, next_wave_own, boundary_time_to_zero) are in seconds. respawn_wait is the
# ship's respawn wait left (its minimum, in Frontline): a Frontline ship then respawns at
# its team's next wave, so
# an own ship's time to respawn is the first of next_wave_own + k x the wave interval at or
# after its respawn_wait.

# Z: the zone slots of every observation in every mode, a fixed shape like the field count,
# so a model sees the same tokens in any batch. Frontline fills them all; a mode without
# zones leaves them absent.
ZONE_SLOTS = 5

# Every column list by observation part: what a policy file records (runs spec 3).
COLUMNS = dict(ships=SHIP_COLUMNS, fields=FIELD_COLUMNS, globals=GLOBAL_COLUMNS, zones=ZONE_COLUMNS)
# The column lists as they were at v2 before policy files recorded them (2026-11-08, Obs
# schema 3): a policy file without a record of its columns was trained on these. Frozen as
# data: never edit it.
COLUMNS_BEFORE_RECORDS = dict(
    ships=(
        "x", "y", "heading_cos", "heading_sin", "velocity_x", "velocity_y", "proper_speed",
        "angle_of_attack", "nose_rate", "energy", "shield", "shield_recharge_wait",
        "gun_cooldown", "log_refractive_index", "log_index_gradient_x", "log_index_gradient_y",
        "alive", "visible", "fired", "respawn_wait", "ship_type", "team_relative",
        "time_since_seen", "spawned",
    ),
    fields=("x", "y", "radius", "band_width", "level"),
    globals=(
        "episode_progress", "arena_radius", "fogged", "reset", "mode_team_deathmatch",
        "mode_frontline", "score_own", "score_enemy", "score_target", "front",
        "front_threshold",
    ),
    zones=(
        "x", "y", "radius", "role", "progress", "direction", "offensive_distance",
        "defensive_distance", "present",
    ),
)  # fmt: skip

SHIP = {name: i for i, name in enumerate(SHIP_COLUMNS)}
FIELD = {name: i for i, name in enumerate(FIELD_COLUMNS)}
ZONE = {name: i for i, name in enumerate(ZONE_COLUMNS)}
GLOBAL = {name: i for i, name in enumerate(GLOBAL_COLUMNS)}


@contract
class Obs:
    """One or more team views, float32. Leading axes are (batch..., ) then:

    ships (..., N, len(SHIP_COLUMNS)), own team's S ships first, then by team_relative and slot;
    fields (..., M, len(FIELD_COLUMNS)); globals (..., len(GLOBAL_COLUMNS)); zones (..., Z,
    len(ZONE_COLUMNS)), Z = ZONE_SLOTS in every mode (a mode without zones shows them
    absent).

    A stored batch may leave out (None) a part its model does not read.

    There are no bullets (author, 2026-10-31): models do not read them, a firing ship is
    always visible, and shot labels are rebuilt learner-side from stored truth.
    """

    ships: torch.Tensor | None
    fields: torch.Tensor | None
    globals: torch.Tensor | None
    zones: torch.Tensor | None = None
    SCHEMA_VERSION: ClassVar[int] = 3  # 2: zones and the mode's global columns; 3: present


@contract
class Privileged:
    """Truth for labels and diagnostics. Only learners may read it; it never reaches a
    controller's `step`.

    team: (B, N) int32, the sim team of each ship (sim order).
    view_order: (B, T, N) int64, the sim index of each observation row of each view.
    ships: (B, T, N, len(SHIP_COLUMNS)), every row as if its ship were seen: `visible` 1,
        `time_since_seen` 0 and the true hidden columns for every ship, the dead included
        (unlike a fog-off observation, where a dead enemy is not visible).
    """

    team: torch.Tensor
    view_order: torch.Tensor
    ships: torch.Tensor
