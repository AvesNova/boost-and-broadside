"""Central action-space constants for Boost and Broadside.

Single source of truth for action definitions and derived sizes.
"""

from enum import IntEnum


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
TOTAL_ACTION_LOGITS: int = (
    NUM_POWER_ACTIONS + NUM_TURN_ACTIONS + NUM_SHOOT_ACTIONS
)  # 12

# Slices into the flat logit vector
POWER_SLICE: slice = slice(0, NUM_POWER_ACTIONS)
TURN_SLICE: slice = slice(NUM_POWER_ACTIONS, NUM_POWER_ACTIONS + NUM_TURN_ACTIONS)
SHOOT_SLICE: slice = slice(NUM_POWER_ACTIONS + NUM_TURN_ACTIONS, TOTAL_ACTION_LOGITS)
