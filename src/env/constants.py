"""
Constants and enumerations for the game environment.

Defines action indices and reward calculation constants.
"""
from enum import IntEnum, auto


class Actions(IntEnum):
    """Action indices for ship control."""

    forward = 0
    backward = auto()
    left = auto()
    right = auto()
    sharp_turn = auto()
    shoot = auto()


class RewardConstants:
    """Constants for reward calculation."""

    VICTORY = 1.0
    DEFEAT = -1.0
    DRAW = 0.0
    ENEMY_DEATH = 0.1
    ALLY_DEATH = -0.1
    ENEMY_DAMAGE = 0.001
    ALLY_DAMAGE = -0.001
