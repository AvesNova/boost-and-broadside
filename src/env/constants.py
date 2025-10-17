from enum import IntEnum, StrEnum, auto


class Actions(IntEnum):
    """Action indices for ship control."""

    forward = 0
    backward = auto()
    left = auto()
    right = auto()
    sharp_turn = auto()
    shoot = auto()


class RewardConstants:
    """Constants for reward calculation"""

    VICTORY_REWARD = 1.0
    DEFEAT_REWARD = -1.0
    DRAW_REWARD = 0.0
    ENEMY_DEATH_BONUS = 0.1
    ALLY_DEATH_PENALTY = -0.1
    DAMAGE_REWARD_SCALE = 0.001
