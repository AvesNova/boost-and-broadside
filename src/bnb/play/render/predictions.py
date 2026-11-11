"""The prediction overlays as drawing tables (`bnb.play.overlays`): line segments, circles,
ghost ships and the density's hex cells, built in numpy from one frame's predictions.

What each overlay draws:

- true bullet paths: each live bullet's flight ahead, a thin line in its team's colour
  that fades with the moves ahead;
- the predicted shot path: the selected ship's next shot, the head's mean at each age
  joined from the launch point, in the theme's prediction colour, a one-sigma circle at
  each age;
- next-state ghosts: each ship's predicted pose one tick on, a translucent silhouette with
  its one-sigma circle;
- belief ghosts: each hidden enemy's believed pose, with one- and two-sigma circles in the
  prediction colour; with the truth, a ring in the truth colour at its true position and
  a line from the ghost to it;
- dead-reckoning ghosts: each remembered enemy's recalled pose, its opacity the agent's
  weight on it, a line back to its last sighting; no circle (the memory states no
  sigma); with the truth, as the beliefs;
- density: the hex cells, shaded by the expected count (one side's cells, in that side's
  team colour), the true count, or the signed error (over in the prediction colour, under
  in the truth colour).
"""

import dataclasses

import numpy as np

from ..overlays import BulletPaths, Density, Ghosts, OverlaySettings, Reckoning, ShotPaths
from ..themes import Theme
from .arcs import CIRCLE, arcs
from .lines import segments
from .ships import ring_radius
from .triangles import FLOATS as TRIANGLE_FLOATS

PATH_PIXELS = 1.4
PATH_OPACITY = 0.7
SHOT_PIXELS = 1.8
SIGMA_PIXELS = 1.1
DOT_PIXELS = 2.2  # the radius of a predicted shot's dot at each age
GHOST_OPACITY = {"next_state": 0.35, "belief": 0.5}
SIGMA_OPACITY = 0.75
TWO_SIGMA_OPACITY = 0.35
TRUTH_PIXELS = 1.6
TRUTH_GAP_PIXELS = 3.0  # the truth's ring sits this far outside the ship's ring
CONNECTOR_OPACITY = 0.55
SIGHTING_OPACITY = 0.45
DENSITY_OPACITY = 0.55  # a cell's opacity at one expected ship (counts above saturate)
ERROR_OPACITY = 0.8  # the error map's at an error of one ship
DENSITY_FLOOR = 0.02  # cells fainter than this are not drawn
CELL_FILL = 0.94  # of a cell's size: a hairline gap between cells


@dataclasses.dataclass
class OverlayDrawing:
    """One frame's overlay tables: density (vertices, TRIANGLE_FLOATS), drawn under the
    bullets; segments and circles (`arcs`) and the ghost ships (per ship: x, y, heading,
    size, team, ship_type, opacity), drawn over them."""

    density: np.ndarray
    segments: np.ndarray
    circles: np.ndarray
    ghosts: dict[str, np.ndarray]


def _colour(colour, opacity: float = 1.0) -> np.ndarray:
    floats = np.array(colour.floats(), np.float32)
    floats[3] *= opacity
    return floats


def path_segments(
    x: np.ndarray, y: np.ndarray, valid: np.ndarray, colour: np.ndarray, pixels: float
) -> np.ndarray:
    """Segments joining consecutive valid points of each path, (P, A) points: colour
    (P, 4) or (P, A - 1, 4), one per segment (a fading path)."""
    joined = valid[:, 1:] & valid[:, :-1]
    table = segments(int(joined.sum()))
    table[:, 0], table[:, 1] = x[:, :-1][joined], y[:, :-1][joined]
    table[:, 2], table[:, 3] = x[:, 1:][joined], y[:, 1:][joined]
    table[:, 4] = pixels
    per_segment = np.broadcast_to(
        colour if colour.ndim == 3 else colour[:, None, :], joined.shape + (4,)
    )
    table[:, 5:9] = per_segment[joined]
    return table


def bullet_path_segments(theme: Theme, paths: BulletPaths) -> np.ndarray:
    """The true bullet paths: each bullet's flight in its team's colour, fading to a third
    of PATH_OPACITY at its last move."""
    count, points = paths.x.shape
    team = np.array([colour.floats() for colour in theme.team], np.float32)[paths.team]
    fade = np.linspace(1.0, 1.0 / 3.0, max(points - 1, 1), dtype=np.float32)
    colour = np.repeat(team[:, None, :], max(points - 1, 1), 1)
    colour[..., 3] *= PATH_OPACITY * fade
    return path_segments(paths.x, paths.y, paths.valid, colour[:, : points - 1], PATH_PIXELS)


def circle_table(
    x: np.ndarray, y: np.ndarray, radius: np.ndarray, colour: np.ndarray, pixels: float
) -> np.ndarray:
    """Outline circles (arrays of one length), colour (4,) or (count, 4)."""
    table = arcs(len(x))
    table[:, 0], table[:, 1], table[:, 2] = x, y, radius
    table[:, 4] = pixels
    table[:, 9:13] = colour
    table[:, 14] = CIRCLE
    return table


def dot_table(x: np.ndarray, y: np.ndarray, radius: float, colour: np.ndarray) -> np.ndarray:
    """Filled dots of `radius` m."""
    table = arcs(len(x))
    table[:, 0], table[:, 1], table[:, 2] = x, y, radius
    table[:, 5:9] = colour
    table[:, 14] = CIRCLE
    return table


def shot_path_tables(
    theme: Theme, shot: ShotPaths, ship: int, metres_per_pixel: float
) -> tuple[np.ndarray, np.ndarray]:
    """The selected ship's predicted shot: (segments, circles) from its launch point
    through the mean at each age, a dot and a one-sigma circle at each age."""
    x = np.concatenate([[shot.launch_x[ship]], shot.x[ship]])[None]
    y = np.concatenate([[shot.launch_y[ship]], shot.y[ship]])[None]
    colour = _colour(theme.prediction)
    lines = path_segments(x, y, np.ones_like(x, bool), colour[None], SHOT_PIXELS)
    dots = dot_table(shot.x[ship], shot.y[ship], DOT_PIXELS * metres_per_pixel, colour)
    circles = circle_table(
        shot.x[ship],
        shot.y[ship],
        shot.sigma[ship],
        _colour(theme.prediction, SIGMA_OPACITY),
        SIGMA_PIXELS,
    )
    return lines, np.concatenate([circles, dots])


def ghost_ships(ghosts: Ghosts, team: np.ndarray, ship_type: np.ndarray, size: np.ndarray, opacity):
    """The ghost silhouettes: per ghost its pose, its ship's size, team and type, and
    opacity (a float, or per ghost)."""
    ship = ghosts.ship
    return dict(
        x=ghosts.x.astype(np.float32),
        y=ghosts.y.astype(np.float32),
        heading=ghosts.heading.astype(np.float32),
        size=size[ship],
        team=team[ship],
        ship_type=ship_type[ship],
        opacity=np.broadcast_to(np.asarray(opacity, np.float32), ship.shape).copy(),
    )


def truth_tables(
    theme: Theme, ghosts: Ghosts, size: np.ndarray, metres_per_pixel: float
) -> tuple[np.ndarray, np.ndarray]:
    """The truth beside ghosts: (segments from each ghost to its true position, a ring at
    the true position just outside the ship's ring), for the living ones."""
    living = ghosts.true_alive.astype(bool)
    table = segments(int(living.sum()))
    table[:, 0], table[:, 1] = ghosts.x[living], ghosts.y[living]
    table[:, 2], table[:, 3] = ghosts.true_x[living], ghosts.true_y[living]
    table[:, 4] = SIGMA_PIXELS
    table[:, 5:9] = _colour(theme.truth, CONNECTOR_OPACITY)
    radius = size[ghosts.ship[living]] + TRUTH_GAP_PIXELS * metres_per_pixel
    rings = circle_table(
        ghosts.true_x[living], ghosts.true_y[living], radius, _colour(theme.truth), TRUTH_PIXELS
    )
    return table, rings


def hexagon_vertices(centres: np.ndarray, radius: float, colour: np.ndarray) -> np.ndarray:
    """Filled pointy-top hexagons (the density grid's cells: neighbours along x) of
    centre-to-corner `radius` m: six triangles each, (C * 18, TRIANGLE_FLOATS)."""
    angles = np.radians(30.0 + 60.0 * np.arange(7))
    corners = radius * np.stack([np.cos(angles), np.sin(angles)], -1)  # (7, 2)
    count = len(centres)
    table = np.zeros((count, 6, 3, TRIANGLE_FLOATS), np.float32)
    table[:, :, 0, 0:2] = centres[:, None, :]
    table[:, :, 1, 0:2] = centres[:, None, :] + corners[None, :6]
    table[:, :, 2, 0:2] = centres[:, None, :] + corners[None, 1:]
    table[..., 2:6] = colour[:, None, None, :]
    return table.reshape(count * 18, TRIANGLE_FLOATS)


def density_vertices(theme: Theme, density: Density, team: int, settings: OverlaySettings):
    """The density overlay's cells for the predicting `team`: its settings' side and what
    it shows (the true and error maps need the truth on; without it, the prediction)."""
    side = settings.density_side
    shown = settings.density_shown if settings.truth else "predicted"
    if shown == "error":
        error = density.predicted[side] - density.true[side]
        over, under = _colour(theme.prediction), _colour(theme.truth)
        colour = np.where((error >= 0.0)[:, None], over, under)
        amount, opacity = np.abs(error), ERROR_OPACITY
    else:
        owner = team if side == "ally" else 1 - team
        amount = density.predicted[side] if shown == "predicted" else density.true[side]
        colour = np.repeat(_colour(theme.team[owner])[None], len(amount), 0)
        opacity = DENSITY_OPACITY
    colour[:, 3] *= opacity * np.clip(amount, 0.0, 1.0)
    drawn = colour[:, 3] >= DENSITY_FLOOR
    return hexagon_vertices(density.centres[drawn], CELL_FILL * density.cell_radius, colour[drawn])


def overlay_drawing(
    theme: Theme,
    settings: OverlaySettings,
    drawn: dict,
    frame,
    selected: int | None,
    metres_per_pixel: float,
    true_size: bool = False,
) -> OverlayDrawing:
    """The tables of the overlays to draw this frame. drawn: overlay name to (team, its
    arrays) (`BulletPaths` with team -1, or the providing team's `Predictions` field);
    frame: the snapshot drawn (each ship's team, type and hit radius); selected: the
    selected ship (the shot path's), or None; true_size: the ships are drawn at true size
    (`ships`), the ghosts' rings with them."""
    ships = frame.ships
    size = ring_radius(ships.hit_radius, metres_per_pixel, true_size)
    lines, circles, ghosts, density = [segments(0)], [arcs(0)], [], np.zeros((0, 6), np.float32)
    if "bullet_paths" in drawn:
        lines.append(bullet_path_segments(theme, drawn["bullet_paths"][1]))
    if "shot_path" in drawn and selected is not None:
        shot_lines, shot_circles = shot_path_tables(
            theme, drawn["shot_path"][1], selected, metres_per_pixel
        )
        lines.append(shot_lines)
        circles.append(shot_circles)
    if "next_state" in drawn:
        forecast: Ghosts = drawn["next_state"][1]
        ghosts.append(
            ghost_ships(forecast, ships.team, ships.ship_type, size, GHOST_OPACITY["next_state"])
        )
        team = np.array([colour.floats() for colour in theme.team], np.float32)
        colour = team[ships.team[forecast.ship]]
        colour[:, 3] *= SIGMA_OPACITY
        circles.append(circle_table(forecast.x, forecast.y, forecast.sigma, colour, SIGMA_PIXELS))
    if "belief" in drawn:
        belief: Ghosts = drawn["belief"][1]
        opacity = GHOST_OPACITY["belief"]
        ghosts.append(ghost_ships(belief, ships.team, ships.ship_type, size, opacity))
        for multiple, opacity in ((1.0, SIGMA_OPACITY), (2.0, TWO_SIGMA_OPACITY)):
            circles.append(
                circle_table(
                    belief.x,
                    belief.y,
                    multiple * belief.sigma,
                    _colour(theme.prediction, opacity),
                    SIGMA_PIXELS,
                )
            )
        if settings.truth:
            truth_lines, truth_rings = truth_tables(theme, belief, size, metres_per_pixel)
            lines.append(truth_lines)
            circles.append(truth_rings)
    if "dead_reckoning" in drawn:
        reckoning: Reckoning = drawn["dead_reckoning"][1]
        recalled = reckoning.ghosts
        ghosts.append(ghost_ships(recalled, ships.team, ships.ship_type, size, recalled.opacity))
        sighting = segments(len(recalled.ship))
        sighting[:, 0], sighting[:, 1] = reckoning.last_x, reckoning.last_y
        sighting[:, 2], sighting[:, 3] = recalled.x, recalled.y
        sighting[:, 4] = SIGMA_PIXELS
        team = np.array([colour.floats() for colour in theme.team], np.float32)
        sighting[:, 5:9] = team[ships.team[recalled.ship]]
        sighting[:, 8] *= SIGHTING_OPACITY * recalled.opacity
        lines.append(sighting)
        if settings.truth:
            truth_lines, truth_rings = truth_tables(theme, recalled, size, metres_per_pixel)
            lines.append(truth_lines)
            circles.append(truth_rings)
    if "density" in drawn:
        team, field = drawn["density"]
        density = density_vertices(theme, field, team, settings)
    ghost_table = {
        name: np.concatenate([ghost[name] for ghost in ghosts]) if ghosts else np.zeros(0)
        for name in ("x", "y", "heading", "size", "team", "ship_type", "opacity")
    }
    return OverlayDrawing(
        density=density,
        segments=np.concatenate(lines),
        circles=np.concatenate(circles),
        ghosts=ghost_table,
    )
