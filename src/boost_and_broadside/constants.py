"""Central constants for Boost and Broadside.

Single source of truth for action definitions, derived sizes, and shared
numerical guards.
"""

import math
from enum import IntEnum

EPS: float = 1e-6  # division safety guard for direction/speed normalization

# One-second projectiles fired at the default 0.1-second cooldown need at most
# nine simultaneously live slots at 60 Hz. Ten retains one safety slot while
# avoiding the dense physics/collision cost of the previous 20-slot pools.
DEFAULT_MAX_BULLETS_PER_SHIP: int = 10


class PowerActions(IntEnum):
    COAST = 0
    BOOST = 1
    REVERSE = 2


class TurnActions(IntEnum):
    GO_STRAIGHT = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    SHARP_LEFT = 3
    SHARP_RIGHT = 4
    AIR_BRAKE = 5
    SHARP_AIR_BRAKE = 6


class ShootActions(IntEnum):
    NO_SHOOT = 0
    SHOOT = 1


NUM_POWER_ACTIONS: int = len(PowerActions)  # 3
NUM_TURN_ACTIONS: int = len(TurnActions)  # 7
NUM_SHOOT_ACTIONS: int = len(ShootActions)  # 2

# Sum of logits for all action heads (used to size the action output layer)
TOTAL_ACTION_LOGITS: int = NUM_POWER_ACTIONS + NUM_TURN_ACTIONS + NUM_SHOOT_ACTIONS  # 12

# Slices into the flat logit vector
POWER_SLICE: slice = slice(0, NUM_POWER_ACTIONS)
TURN_SLICE: slice = slice(NUM_POWER_ACTIONS, NUM_POWER_ACTIONS + NUM_TURN_ACTIONS)
SHOOT_SLICE: slice = slice(NUM_POWER_ACTIONS + NUM_TURN_ACTIONS, TOTAL_ACTION_LOGITS)

# The three factors in one order, with the entropy of a uniform distribution over
# each. The factors have different cardinalities, so a plain sum of their
# cross-entropies weights them by how many options they happen to offer rather
# than by how much there is to learn: turn alone would be 52% of an untrained
# total and shoot 19%. Dividing each by its own maximum puts them on one scale,
# where 1 means "no better than a coin toss over that factor" and 0 means exact.
ACTION_FACTOR_SLICES: tuple[slice, ...] = (POWER_SLICE, TURN_SLICE, SHOOT_SLICE)
ACTION_FACTOR_MAX_ENTROPY: tuple[float, ...] = (
    math.log(NUM_POWER_ACTIONS),
    math.log(NUM_TURN_ACTIONS),
    math.log(NUM_SHOOT_ACTIONS),
)
