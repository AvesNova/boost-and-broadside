"""
Constants and enumerations for the game environment.

Defines action indices and reward calculation constants.
"""

from enum import IntEnum, auto

from matplotlib.pyplot import cla


class HumanActions(IntEnum):
    """Action indices for ship control."""

    forward = 0
    backward = auto()
    left = auto()
    right = auto()
    sharp_turn = auto()
    shoot = auto()


class PowerActions(IntEnum):
    COAST = 0
    BOOST = auto()
    BRAKE = auto()


class TurnActions(IntEnum):
    GO_STRAIGHT = 0
    TURN_LEFT = 1
    TURN_RIGHT = 2
    SHARP_LEFT = 3
    SHARP_RIGHT = 4


class ShootActions(IntEnum):
    NO_SHOOT = 0
    SHOOT = 1


class RewardConstants:
    """Constants for reward calculation."""

    VICTORY = 1.0
    DEFEAT = -1.0
    DRAW = 0.0
    ENEMY_DEATH = 0.1
    ALLY_DEATH = -0.1
    ENEMY_DAMAGE = 0.001
    ALLY_DAMAGE = -0.001
