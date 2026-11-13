"""The drawn state between two ticks: positions linear, headings by the shortest angle,
shield and energy linear, everything else from the later tick.

A ship that spawned or respawned draws at its new state (no interpolation across a
teleport; an episode reset spawns every ship), and a bullet fired on the later tick draws
from its launch point, the shooter's position at the earlier tick, which is where the game
fires it from (a reset clears every bullet). A bullet flies straight within a tick (a
field bends it only between moves), so the drawn bullet is exact at both ticks and on its
path between them; its speed is that move's length over the tick, and its age counts back
from the later tick.
"""

import dataclasses
import math

import numpy as np

from .snapshot import Bullets, Ships, Snapshot


def lerp(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    """alpha of the way from start to end."""
    return start + (end - start) * alpha


def angle_lerp(start: np.ndarray, end: np.ndarray, alpha: float) -> np.ndarray:
    """From start toward end by the shorter way round, wrapped to [-pi, pi)."""
    turn = np.remainder(end - start + math.pi, 2.0 * math.pi) - math.pi
    return np.remainder(start + turn * alpha + math.pi, 2.0 * math.pi) - math.pi


def _ships(previous: Ships, current: Ships, alpha: float) -> Ships:
    moving = current.alive & previous.alive & ~current.spawned

    def blend(name: str, how) -> np.ndarray:
        before, after = getattr(previous, name), getattr(current, name)
        return np.where(moving, how(before, after, alpha), after).astype(after.dtype)

    return dataclasses.replace(
        current,
        x=blend("x", lerp),
        y=blend("y", lerp),
        heading=blend("heading", angle_lerp),
        flight_path=blend("flight_path", angle_lerp),
        shield=blend("shield", lerp),
        energy=blend("energy", lerp),
    )


def _bullets(previous: Snapshot, current: Snapshot, alpha: float) -> Bullets:
    before, after = previous.bullets, current.bullets
    # A slot holds a new bullet when it was idle, or its count of moves went up (refired).
    fresh = after.active & (~before.active | (after.moves_left >= before.moves_left))
    start_x = np.where(fresh, previous.ships.x[after.owner], before.x)
    start_y = np.where(fresh, previous.ships.y[after.owner], before.y)
    tick = current.tick_length
    moved = np.hypot(after.x - start_x, after.y - start_y)
    return dataclasses.replace(
        after,
        x=lerp(start_x, after.x, alpha).astype(after.x.dtype),
        y=lerp(start_y, after.y, alpha).astype(after.y.dtype),
        age=np.where(after.active, after.age - (1.0 - alpha) * tick, 0.0).astype(np.float32),
        speed=np.where(after.active, moved / tick, 0.0).astype(np.float32),
    )


def interpolate(previous: Snapshot, current: Snapshot, alpha: float) -> Snapshot:
    """The state drawn at alpha in [0, 1] of the way from previous to current."""
    return dataclasses.replace(
        current,
        ships=_ships(previous.ships, current.ships, alpha),
        bullets=_bullets(previous, current, alpha),
        boundary_radius=float(lerp(previous.boundary_radius, current.boundary_radius, alpha))
        if previous.tick < current.tick
        else current.boundary_radius,
    )
