"""The ship card's handles and solver (ship-card spec): measure a card's handles by flying
the game's own code, and solve for the card that meets a set of handles.

    handles = measure(types, rules)        # ShipTypes -> Handles
    solution = solve(handles, rules)       # Handles -> Solution(types, error, ...)
"""

from .errors import Infeasible, RedundantHandles
from .handles import COMBAT, FLIGHT, FLOWN, NAMES, Handles
from .solver import Solution, measure, solve

__all__ = [
    "COMBAT",
    "FLIGHT",
    "FLOWN",
    "NAMES",
    "Handles",
    "Infeasible",
    "RedundantHandles",
    "Solution",
    "measure",
    "solve",
]
