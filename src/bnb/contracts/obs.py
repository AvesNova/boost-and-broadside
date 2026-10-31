"""What a team sees (`Obs`) and the truth only learners may use (`Privileged`).

An observation is raw and in world coordinates, with no rotation or mirroring. The only
canonicalization is the team: every entity carries `team_rel = (team - viewer) mod T`, and
rows are ordered by `team_rel` (own ships first), so nothing reveals which seat a team has.
"""

from typing import ClassVar

import torch

from .tree import contract

SHIP_COLUMNS = (
    # zero when the team cannot see the ship
    "x", "y", "c", "s", "vx", "vy", "w", "alpha", "r", "E", "shield", "gun_wait",
    "alive", "visible", "respawn", "type", "team_rel", "since_seen", "spawned",  # always shown
)
SHIP_HIDDEN_ZERO = 12  # the first this-many columns are zero for a ship the team cannot see
FIELD_COLUMNS = ("x", "y", "radius", "level")
GLOBAL_COLUMNS = ("t_frac", "arena_radius", "fog", "reset")

SHIP = {name: i for i, name in enumerate(SHIP_COLUMNS)}
FIELD = {name: i for i, name in enumerate(FIELD_COLUMNS)}
GLOBAL = {name: i for i, name in enumerate(GLOBAL_COLUMNS)}


@contract
class Obs:
    """One or more team views, float32. Leading axes are (batch..., ) then:

    ships (..., N, len(SHIP_COLUMNS)), own team's S ships first, then by team_rel and slot;
    fields (..., M, len(FIELD_COLUMNS)); globals (..., len(GLOBAL_COLUMNS)).

    A stored batch may leave out (None) a part its model does not read.

    There are no bullets (author, 2026-10-31): models do not read them, a firing ship is
    always visible, and shot labels are rebuilt learner-side from stored truth.
    """

    ships: torch.Tensor | None
    fields: torch.Tensor | None
    globals: torch.Tensor | None
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
