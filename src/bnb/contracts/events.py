"""What happened in one tick. Read by rewards and diagnostics, never by controllers.

Indices are sim indices (team-major), not observation rows.
"""

import math
from typing import ClassVar

import torch

from .tree import contract

# A hit's aspect class, for the statistics (the damage itself follows the smooth aspect of
# `bnb.game.physics.combat.hits`): theta, the angle between the bullet's reversed direction
# and the target's nose (0: the shooter dead ahead of the target's guns). Under 10 degrees
# a mutual head-on (both ships can fire), 10 to 45 an offset head-on (only the shooter can),
# 45 to 135 the flank, over 135 the tail. ASPECT_BOUNDS are the classes' upper edges but the
# last, each class taking theta above the one before and up to its own (the first from 0).
ASPECTS = ("mutual_head_on", "offset_head_on", "flank", "tail")
ASPECT_BOUNDS = (math.radians(10.0), math.pi / 4, 3 * math.pi / 4)


@contract
class Events:
    """Per tick; (B, N) unless noted.

    damage: (B, N, N) float hit damage from shooter to target. A per-tick transient: reward
        functions consume it and trajectories never store it.
    boundary_damage: float. died, fired, spawned: bool.
    killer: int32 sim index of the shooter whose hit killed the ship, -1 if none.
    done: (B,) bool, the environment reset after this tick.
    shield_lost: the shield a ship alive at tick start actually lost to this tick's damage
        (overkill ignored). shield_restored: the recharge applied this tick.
    objective_progress, objective: (B, N, 2) float, the mode's two objective amounts per
        ship as their gain (>= 0) and their loss (<= 0), each event in its own column (a
        ship credited in one zone and charged in another on one tick keeps both); each
        amount, gain plus loss, sums to zero over the environment (Frontline: meter
        fractions, captures).
    result: (B,) int32, the match result for team 0 (+1 win, -1 loss, 0 draw), nonzero
        only where done (team deathmatch: by the kill lead, for the tallies).
    win_loss: (B, 2) float, team 0's weights of a win and of a loss where done, zero
        elsewhere; a draw is the rest, 1 - win - loss (team 1's: flipped). The learner's
        outcome: the result one-hot (`bnb.game.modes.phase.win_loss`), or in team
        deathmatch with `ModeRules.deathmatch_outcome` "kill_share" its kill share
        (share, 1 - share).
    enemy_damage_by_aspect: (B, N, len(ASPECTS)) float, the hit damage each ship took from
        the enemy team this tick by the hit's aspect class (friendly fire left out).
    kill_aspect: int32, the aspect class (an index of ASPECTS) of the hit that killed the
        ship (the killer's earliest hit), -1 if none.
    """

    damage: torch.Tensor
    boundary_damage: torch.Tensor
    died: torch.Tensor
    fired: torch.Tensor
    spawned: torch.Tensor
    killer: torch.Tensor
    done: torch.Tensor
    shield_lost: torch.Tensor
    shield_restored: torch.Tensor
    objective_progress: torch.Tensor
    objective: torch.Tensor
    result: torch.Tensor
    win_loss: torch.Tensor
    enemy_damage_by_aspect: torch.Tensor
    kill_aspect: torch.Tensor
    # 3: the aspect statistics; 4: four classes; 5: win_loss; 6: objective gain and loss
    SCHEMA_VERSION: ClassVar[int] = 6
