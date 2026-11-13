"""Types shared between sections: shapes, dtypes and meanings only, no game logic.

See docs/rewrite/contracts.md for the reasoning behind each field.
"""

from .action import Action
from .config import GameConfig, ticks_of
from .controller import Controller, Outputs
from .events import ASPECT_BOUNDS, ASPECTS, Events
from .obs import Obs, Privileged
from .pools import ArenaScale, Pool
from .trajectory import REWARD_LEVELS, SUMMED_LEVEL, Trajectory

__all__ = [
    "ASPECTS",
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
    "REWARD_LEVELS",
    "SUMMED_LEVEL",
    "Trajectory",
    "ticks_of",
]
