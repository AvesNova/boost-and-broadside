"""What a team sees (`Obs`) and the truth only learners may use (`Privileged`).

An observation is raw and in world coordinates, with no rotation or mirroring. The only
canonicalization is the team: every entity carries `team_relative = (team - viewer) mod T`, and
rows are ordered by `team_relative` (own ships first), so nothing reveals which seat a team has.
"""

from typing import ClassVar

import torch

from .tree import contract

SHIP_COLUMNS = (
    # zero when the team cannot see the ship
    "x", "y", "heading_cos", "heading_sin", "velocity_x", "velocity_y", "proper_speed",
    "angle_of_attack", "nose_rate", "energy", "shield", "shield_recharge_wait", "gun_cooldown",
    "log_refractive_index", "log_index_gradient_x", "log_index_gradient_y",
    # always shown; fired: fired this tick
    "alive", "visible", "fired", "respawn_wait", "ship_type", "team_relative", "time_since_seen",
    "spawned",
)
SHIP_HIDDEN_ZERO = 16  # the first this-many columns are zero for a ship the team cannot see
FIELD_COLUMNS = ("x", "y", "radius", "band_width", "level")
# Zones (Frontline's), for the viewer: role 0 own spawn, 1 own defense, 2 neutral, 3 enemy
# defense, 4 enemy spawn; progress, the meter in [0, 1]; direction +1 when the meter moves
# in the viewer's favour, -1 in the enemy's, 0 still; the distances: captures to the
# viewer's win and to its loss.
ZONE_COLUMNS = ("x", "y", "radius", "role", "progress", "direction", "offensive_distance",
                "defensive_distance")
# The mode block: a mode one-hot (all zero in sandbox); the scores by team_relative and the
# score that wins (team deathmatch); the front, positive in the viewer's favour, and the
# front that wins (Frontline). Absent quantities are 0.
GLOBAL_COLUMNS = ("episode_progress", "arena_radius", "fogged", "reset", "mode_team_deathmatch",
                  "mode_frontline", "score_own", "score_enemy", "score_target", "front",
                  "front_threshold")
# Times (shield_recharge_wait, gun_cooldown, respawn_wait, time_since_seen) are in seconds.

SHIP = {name: i for i, name in enumerate(SHIP_COLUMNS)}
FIELD = {name: i for i, name in enumerate(FIELD_COLUMNS)}
ZONE = {name: i for i, name in enumerate(ZONE_COLUMNS)}
GLOBAL = {name: i for i, name in enumerate(GLOBAL_COLUMNS)}


@contract
class Obs:
    """One or more team views, float32. Leading axes are (batch..., ) then:

    ships (..., N, len(SHIP_COLUMNS)), own team's S ships first, then by team_relative and slot;
    fields (..., M, len(FIELD_COLUMNS)); globals (..., len(GLOBAL_COLUMNS)); zones (..., Z,
    len(ZONE_COLUMNS)), Z = 0 in modes without zones.

    A stored batch may leave out (None) a part its model does not read.

    There are no bullets (author, 2026-10-31): models do not read them, a firing ship is
    always visible, and shot labels are rebuilt learner-side from stored truth.
    """

    ships: torch.Tensor | None
    fields: torch.Tensor | None
    globals: torch.Tensor | None
    zones: torch.Tensor | None = None
    SCHEMA_VERSION: ClassVar[int] = 1


@contract
class Privileged:
    """Truth for labels and diagnostics. Only learners may read it; it never reaches a
    controller's `step`.

    team: (B, N) int32, the sim team of each ship (sim order).
    view_order: (B, T, N) int64, the sim index of each observation row of each view.
    ships: (B, T, N, len(SHIP_COLUMNS)), the observation with fog off.
    """

    team: torch.Tensor
    view_order: torch.Tensor
    ships: torch.Tensor
