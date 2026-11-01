"""Pass 6: one ring at the hitbox per ship (docs/rewrite/visual-mockups/shields3.png, row
3). The shield on a 240 degree arc centred on the nose, the energy on a 120 degree arc
centred on the tail; each shrinks toward its centre as it empties, over a faint track of
its full span. An empty shield draws no shield arc."""

import math

import numpy as np

from ..themes import Theme
from .arcs import arcs

SHIELD_HALF_SPAN = math.radians(120.0) - 0.1  # rad; 0.1 rad leaves a gap for the round ends
ENERGY_HALF_SPAN = math.radians(60.0) - 0.1
TRACK_OPACITY = 0.15
WIDTH_FRACTION = 0.12  # of the ring radius
MINIMUM_WIDTH_PIXELS = 1.6


def ring_table(theme: Theme, x: np.ndarray, y: np.ndarray, heading: np.ndarray,
               size: np.ndarray, team: np.ndarray, shield: np.ndarray,
               energy: np.ndarray) -> np.ndarray:
    """The arcs of the listed ships (arrays of one length; size the ring radius in m,
    shield and energy fractions): shield track, shield, energy track, energy."""
    count = len(x)
    table = arcs(4 * count).reshape(count, 4, -1)
    table[:, :, 0], table[:, :, 1] = x[:, None], y[:, None]
    table[:, :, 2] = size[:, None]
    table[:, :, 3] = WIDTH_FRACTION * size[:, None]
    table[:, :, 4] = MINIMUM_WIDTH_PIXELS
    shield_colour = np.array([colour.floats() for colour in theme.shield], np.float32)[team]
    energy_colour = np.array(theme.energy.floats(), np.float32)
    for arc, colour, opacity in ((0, shield_colour, TRACK_OPACITY), (1, shield_colour, 1.0),
                                 (2, energy_colour, TRACK_OPACITY), (3, energy_colour, 1.0)):
        table[:, arc, 9:13] = colour
        table[:, arc, 12] *= opacity
    tail = np.remainder(heading + math.pi, 2.0 * math.pi)
    table[:, 0, 13], table[:, 0, 14] = heading, SHIELD_HALF_SPAN
    table[:, 1, 13], table[:, 1, 14] = heading, SHIELD_HALF_SPAN * np.clip(shield, 0.0, 1.0)
    table[:, 2, 13], table[:, 2, 14] = tail, ENERGY_HALF_SPAN
    table[:, 3, 13], table[:, 3, 14] = tail, ENERGY_HALF_SPAN * np.clip(energy, 0.0, 1.0)
    table[:, 1, 12] *= shield > 0.0  # an empty shield draws no arc
    table[:, 3, 12] *= energy > 0.0
    return table.reshape(4 * count, -1)
