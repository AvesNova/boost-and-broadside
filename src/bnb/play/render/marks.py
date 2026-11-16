"""Death marks: a team-coloured X where each ship died, kept on the battlefield for the
rest of the session (the figure videos, `bnb.figures.video`), so a ship is seen to die
rather than vanish. Drawn by the line program (`lines.py`) under the living ships."""

import numpy as np

from ..themes import Theme
from .lines import segments
from .ships import ring_radius

MARK_PIXELS = 2.5  # the X's stroke width on screen
ARM_FRACTION = 0.6  # each arm's end, along both axes, as a fraction of the ship's ring
# radius: the X's corners sit at 0.85 of the ring, inside the ship's drawn size


class DeathMarks:
    """The deaths so far: per death, the position (m), the dead ship's team and hitbox
    radius (m, which sizes its X like the ship's ring)."""

    def __init__(self) -> None:
        self.rows: list[tuple[float, float, int, float]] = []

    def add(self, x: float, y: float, team: int, hit_radius: float) -> None:
        """One death at (x, y)."""
        self.rows.append((float(x), float(y), int(team), float(hit_radius)))

    def __len__(self) -> int:
        return len(self.rows)

    def segments(self, theme: Theme, metres_per_pixel: float, true_size: bool) -> np.ndarray:
        """The marks as a line instance table (`lines.segments`): two strokes per death,
        in its team's colour, sized like the ship's ring (`ships.ring_radius`)."""
        table = segments(2 * len(self.rows))
        if not self.rows:
            return table
        rows = np.array(self.rows, np.float64)
        x, y, team = rows[:, 0], rows[:, 1], rows[:, 2].astype(np.int64)
        arm = ARM_FRACTION * ring_radius(rows[:, 3], metres_per_pixel, true_size)
        colour = np.array([colour.floats() for colour in theme.team], np.float32)[team]
        for stroke, sign in enumerate((1.0, -1.0)):  # "\" then "/"
            rows_of = table[stroke::2]
            rows_of[:, 0], rows_of[:, 1] = x - arm, y + sign * arm
            rows_of[:, 2], rows_of[:, 3] = x + arm, y - sign * arm
            rows_of[:, 4] = MARK_PIXELS
            rows_of[:, 5:9] = colour
        return table
