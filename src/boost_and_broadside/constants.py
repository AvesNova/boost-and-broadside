"""Central constants for Boost and Broadside.

Single source of truth for action definitions, derived sizes, and shared
numerical guards.
"""

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


class ShootActions(IntEnum):
    NO_SHOOT = 0
    SHOOT = 1


NUM_POWER_ACTIONS: int = len(PowerActions)  # 3
NUM_TURN_ACTIONS: int = len(TurnActions)  # 5
NUM_SHOOT_ACTIONS: int = len(ShootActions)  # 2

# The policy emits one categorical over the physical Cartesian product. Runtime
# commands stay as [power, turn, shoot] triples; only the policy boundary uses
# the joint ID.
NUM_JOINT_ACTIONS: int = NUM_POWER_ACTIONS * NUM_TURN_ACTIONS * NUM_SHOOT_ACTIONS  # 30
TOTAL_ACTION_LOGITS: int = NUM_JOINT_ACTIONS

# Slices into the scripted teacher's compact independent marginals. These do
# not index policy logits now that the actor is joint.
POWER_SLICE: slice = slice(0, NUM_POWER_ACTIONS)
# Match outcome as a classification target, ego-relative: index 0 loss, 1 tie,
# 2 win. Ordered so the index is monotone in the result, which lets
# ``(probabilities * OUTCOME_VALUES).sum(-1)`` recover the signed expectation
# the scalar ``outcome`` component regresses directly.
NUM_OUTCOME_CLASSES: int = 3
OUTCOME_LOSS_INDEX: int = 0
OUTCOME_TIE_INDEX: int = 1
OUTCOME_WIN_INDEX: int = 2

# Width of the scripted teacher's independent marginals, power then turn then shoot.
NUM_EXPERT_MARGINALS: int = NUM_POWER_ACTIONS + NUM_TURN_ACTIONS + NUM_SHOOT_ACTIONS

TURN_SLICE: slice = slice(NUM_POWER_ACTIONS, NUM_POWER_ACTIONS + NUM_TURN_ACTIONS)
SHOOT_SLICE: slice = slice(
    NUM_POWER_ACTIONS + NUM_TURN_ACTIONS,
    NUM_POWER_ACTIONS + NUM_TURN_ACTIONS + NUM_SHOOT_ACTIONS,
)
