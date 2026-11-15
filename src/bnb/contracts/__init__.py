"""Types shared between sections: shapes, dtypes and meanings only, no game logic.

See docs/rewrite/contracts.md for the reasoning behind each field.
"""

from .action import Action
from .config import ActiveFields, GameConfig, ticks_of
from .controller import Controller, Outputs
from .events import ASPECT_BOUNDS, ASPECTS, Events
from .obs import Obs, Privileged
from .pools import ArenaScale, Pool, TeamSizes
from .trajectory import REWARD_LEVELS, REWARD_PARTS, VALUE_LEVELS, Trajectory, level_part

__all__ = [
    "ASPECTS",
    "ActiveFields",
    "ASPECT_BOUNDS",
    "Action",
    "Controller",
    "Events",
    "GameConfig",
    "Obs",
    "Outputs",
    "ArenaScale",
    "Pool",
    "Privileged",
    "TeamSizes",
    "REWARD_LEVELS",
    "REWARD_PARTS",
    "Trajectory",
    "VALUE_LEVELS",
    "level_part",
    "ticks_of",
]
