from dataclasses import dataclass
from typing import Dict
import numpy as np

@dataclass
class RenderShip:
    """Snapshot of a ship's state for rendering purposes."""
    id: int
    team_id: int
    position: complex
    attitude: complex
    health: float
    max_health: float
    power: float
    alive: bool

@dataclass
class RenderState:
    """Snapshot of the entire game state for rendering purposes."""
    ships: Dict[int, RenderShip]
    
    # Bullet data as contiguous arrays for efficient rendering
    bullet_x: np.ndarray # (N_bullets,)
    bullet_y: np.ndarray # (N_bullets,)
    bullet_owner_id: np.ndarray # (N_bullets,) - used for coloring (team)
    
    time: float
