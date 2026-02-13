"""
Central constants and enumerations for the Boost and Broadside project.

This file serves as the single source of truth for global dimensions,
normalization limits, and action definitions.
"""

from enum import IntEnum

# =============================================================================
# Global Dimensions & Configuration Limits
# =============================================================================
# These values represent the architectural maximums or defaults.
# Specific runtime config might use fewer ships, but models are often 
# built to support up to these limits.

class StateFeature(IntEnum):
    HEALTH = 0
    POWER = 1
    VX = 2
    VY = 3
    ANG_VEL = 4


class TargetFeature(IntEnum):
    DX = 0
    DY = 1
    DVX = 2
    DVY = 3
    DHEALTH = 4
    DPOWER = 5
    DANG_VEL = 6


MAX_SHIPS = 8       # Maximum number of ships per team (for fixed-size buffers)
STATE_DIM = len(StateFeature)
TARGET_DIM = len(TargetFeature)
ACTION_DIM = 3      # Number of discrete action components (Power, Turn, Shoot)

# Normalization Constants (for Tokenizer)
# Used to normalize raw observation values into [0, 1] or [-1, 1] range.
# DEPRECATED: Use FeatureNormalizer with raw_stats.csv instead.

# =============================================================================
# Action Definitions
# =============================================================================

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

# Derived Action Statistics
NUM_POWER_ACTIONS = len(PowerActions)
NUM_TURN_ACTIONS = len(TurnActions)
NUM_SHOOT_ACTIONS = len(ShootActions)
TOTAL_ACTION_LOGITS = NUM_POWER_ACTIONS + NUM_TURN_ACTIONS + NUM_SHOOT_ACTIONS  # 3 + 7 + 2 = 12

# =============================================================================
# Reward Constants
# =============================================================================

class RewardConstants:
    """Constants for reward calculation."""
    VICTORY = 1.0
    DEFEAT = -1.0
    DRAW = 0.0
    ENEMY_DEATH = 0.1
    ALLY_DEATH = -0.1
    ENEMY_DAMAGE = 0.001
    ALLY_DAMAGE = -0.001
