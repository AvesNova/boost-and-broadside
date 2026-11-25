"""Pass 3: Frontline's zones. A disc tinted with its owner's colour (the neutral colour
for the neutral zone), stronger on the defense zones, an outline, and on each defense
zone the capture meter: an arc in the attacker's colour, growing clockwise from the top.
A zone under a capture lock shows it inside its edge: a thinner arc in the held team's
colour, from the top, as long as the share of the lock still to run."""

import math
from typing import TYPE_CHECKING

import numpy as np

from ..themes import Theme
from .arcs import CIRCLE, arcs

if TYPE_CHECKING:  # rendering never imports torch, which the snapshot module does
    from ..snapshot import Zones

SPAWN_TINT = 0.10  # fill opacity
DEFENSE_TINT = 0.20
OUTLINE_PIXELS = 2.0
OUTLINE_OPACITY = 0.8
METER_PIXELS = 5.0
METER_GAP_PIXELS = 7.0  # between the zone's edge and the meter
METER_TRACK_OPACITY = 0.2
LOCK_PIXELS = 3.0
LOCK_OPACITY = 0.75


def zone_table(theme: Theme, zones: Zones, metres_per_pixel: float) -> np.ndarray:
    """The arcs table of Frontline's zones: each disc and outline, and each defense zone's meter."""
    shown = np.flatnonzero(zones.present)
    team = [colour.floats() for colour in theme.team]
    rows = []
    for zone in shown:
        owner, attacker = int(zones.owner[zone]), int(zones.attacker[zone])
        colour = np.array(team[owner] if owner >= 0 else theme.neutral.floats(), np.float32)
        disc = arcs(1)[0]
        disc[0:3] = zones.x[zone], zones.y[zone], zones.radius[zone]
        disc[4] = OUTLINE_PIXELS
        disc[5:9] = colour
        disc[8] = DEFENSE_TINT if attacker >= 0 else SPAWN_TINT
        disc[9:13] = colour
        disc[12] = OUTLINE_OPACITY
        disc[14] = CIRCLE
        rows.append(disc)
        for side in range(2):
            left = float(zones.lock[side, zone])
            length = float(zones.lock_length[zone])
            if left <= 0.0 or length <= 0.0:
                continue
            lock = arcs(1)[0]
            radius = zones.radius[zone] - METER_GAP_PIXELS * metres_per_pixel
            lock[0:3] = zones.x[zone], zones.y[zone], max(float(radius), 0.0)
            lock[4] = LOCK_PIXELS
            lock[9:13] = np.array(team[side], np.float32)
            lock[12] = LOCK_OPACITY
            half = math.pi * min(left / length, 1.0)
            lock[13], lock[14] = 0.5 * math.pi - half, half
            rows.append(lock)
        if attacker < 0:
            continue
        meter_colour = np.array(team[attacker], np.float32)
        radius = zones.radius[zone] + METER_GAP_PIXELS * metres_per_pixel
        for opacity, fraction in ((METER_TRACK_OPACITY, 1.0), (1.0, float(zones.meter[zone]))):
            if fraction <= 0.0:
                continue
            arc = arcs(1)[0]
            arc[0:3] = zones.x[zone], zones.y[zone], radius
            arc[4] = METER_PIXELS
            arc[9:13] = meter_colour
            arc[12] = opacity
            half = math.pi * min(fraction, 1.0)
            arc[13], arc[14] = 0.5 * math.pi - half, half
            rows.append(arc)
    return np.array(rows, np.float32).reshape(-1, arcs(0).shape[1])
