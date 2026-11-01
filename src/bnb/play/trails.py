"""Flight trails: each ship's last TRAIL_SECONDS of drawn positions, kept in game time.

A sample is taken whenever game time has moved on by SAMPLE_SECONDS since the last one,
so a trail covers the same stretch of flight at any time scale and stops while paused. A
ship that teleports (spawns, respawns or resets) starts a new trail; a dead ship's trail
fades out where it ended. Each sample records which teams saw the ship then, so a team's
view draws only the stretches of an enemy's path that team saw.
"""

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:  # the renderer reads trails and never imports torch
    from .snapshot import Snapshot

TRAIL_SECONDS = 1.5
SAMPLE_SECONDS = 1.0 / 60.0
SAMPLES = math.ceil(TRAIL_SECONDS / SAMPLE_SECONDS) + 2


class Trails:
    """Samples (N, SAMPLES), newest first: position, game time, heading, and the angle of
    attack as a fraction of the stall angle (signed, + turning left); seen (2, N, SAMPLES),
    team T saw the ship at that sample (its observation's `visible`). time is -inf for a
    slot with no sample yet."""

    def __init__(self, num_ships: int, stall_angle: float) -> None:
        shape = (num_ships, SAMPLES)
        self.x, self.y = np.zeros(shape, np.float32), np.zeros(shape, np.float32)
        self.time = np.full(shape, -np.inf)
        self.heading = np.zeros(shape, np.float32)
        self.turn = np.zeros(shape, np.float32)
        self.seen = np.zeros((2, *shape), bool)
        self.stall_angle = stall_angle
        self.seconds = TRAIL_SECONDS
        self.last = -np.inf

    def clear(self) -> None:
        self.time[:] = -np.inf
        self.last = -np.inf

    def record(self, frame: Snapshot, now: float, restart: np.ndarray) -> None:
        """Add the frame's living ships at game time `now` (seconds), if a sample is due.
        restart (N,) bool: these ships start a new trail."""
        self.time[restart] = -np.inf
        if now < self.last + SAMPLE_SECONDS and not restart.any():
            return
        self.last = now
        ships = frame.ships
        for name in ("x", "y", "time", "heading", "turn", "seen"):
            values = getattr(self, name)
            values[..., 1:] = values[..., :-1]
        self.x[:, 0], self.y[:, 0] = ships.x, ships.y
        self.seen[..., 0] = ships.visible
        self.time[:, 0] = np.where(ships.alive, now, -np.inf)
        self.heading[:, 0] = ships.heading
        self.turn[:, 0] = ships.angle_of_attack / self.stall_angle
