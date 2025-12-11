"""
Game state representation.

Defines the State class which encapsulates the current state of all ships and bullets.
"""

import numpy as np

from env.bullets import Bullets
from env.ship import Ship


class State:
    """
    Represents the complete game state at a specific time.

    Attributes:
        ships: Dictionary mapping ship_id to Ship instances.
        time: Current simulation time.
        bullets: Bullet manager for all active projectiles.
    """

    def __init__(self, ships: dict[int, Ship], time: float = 0.0) -> None:
        """
        Initialize game state.

        Args:
            ships: Dictionary of ships in the game.
            time: Initial simulation time.
        """
        self.ships = ships
        self.time = time

        max_bullets = np.sum(ship.max_bullets for ship in ships.values())
        self.bullets = Bullets(max_bullets=max_bullets)
