import numpy as np
from .bullets import Bullets
from .ship import Ship


class State:
    """Represents a state of the game state at a specific time"""

    def __init__(self, ships: dict[int, Ship], time: float = 0.0) -> None:
        self.ships = ships
        self.time = time

        max_bullets = np.sum(ship.max_bullets for ship in ships.values())
        self.bullets = Bullets(max_bullets=max_bullets)
