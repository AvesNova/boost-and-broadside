"""Types shared between sections: shapes, dtypes and meanings only, no game logic.

See docs/rewrite/contracts.md for the reasoning behind each field.
"""

from .action import Action
from .config import GameConfig, ticks_of
from .controller import Controller, Outputs
from .events import Events
from .obs import Obs, Privileged
from .pools import Pool

__all__ = [
    "Action",
    "Controller",
    "Events",
    "GameConfig",
    "Obs",
    "Outputs",
    "Pool",
    "Privileged",
    "ticks_of",
]
