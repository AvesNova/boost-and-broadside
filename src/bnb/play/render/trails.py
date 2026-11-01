"""Pass 4: trails. Each ship's trail is a strip that tapers and fades along its length; in
hard turns (the angle of attack beyond the theme's vortex threshold, as a fraction of the
stall angle) a faint vortex line trails from the inside wingtip."""

from typing import TYPE_CHECKING

import moderngl
import numpy as np

from ..themes import Theme
from .programs import program, set_uniforms
from .silhouettes import HALF_OUTLINES, MOCKUP_RING

if TYPE_CHECKING:
    from ..trails import Trails

NEWEST_PIXELS = 2.5  # the trail's width at the ship
OLDEST_PIXELS = 0.5
VORTEX_PIXELS = 1.2
VORTEX_RAMP = 4.0  # the vortex reaches full strength this many thresholds' worth past it
WINGTIP = {kind: max(abs(side) for _, side in outline) / MOCKUP_RING
           for kind, outline in HALF_OUTLINES.items()}  # half span, in ring radii
VERTEX_FLOATS = 10


def _strip(x, y, width, colour, valid) -> tuple[np.ndarray, np.ndarray]:
    """Vertices (2 N P, VERTEX_FLOATS) and triangle indices for strips through points
    (N, P); a segment is drawn where both its ends are valid."""
    N, P = x.shape
    dx, dy = np.gradient(x, axis=1), np.gradient(y, axis=1)
    length = np.maximum(np.hypot(dx, dy), 1e-9)
    normal_x, normal_y = -dy / length, dx / length
    vertices = np.zeros((N, P, 2, VERTEX_FLOATS), np.float32)
    for side_index, side in enumerate((-1.0, 1.0)):
        v = vertices[:, :, side_index]
        v[..., 0], v[..., 1], v[..., 2], v[..., 3] = x, y, normal_x, normal_y
        v[..., 4], v[..., 5] = side, width
        v[..., 6:10] = colour
    point = np.arange(N * P).reshape(N, P)
    segments = (valid[:, :-1] & valid[:, 1:]).nonzero()
    first = 2 * point[:, :-1][segments]
    second = 2 * point[:, 1:][segments]
    indices = np.stack([first, first + 1, second, second, first + 1, second + 1], -1)
    return vertices.reshape(-1, VERTEX_FLOATS), indices.astype(np.int32).reshape(-1)


def trail_geometry(trails: Trails, theme: Theme, now: float, shown: np.ndarray,
                   team: np.ndarray, ship_type: np.ndarray, size: np.ndarray):
    """(vertices, indices) for the trails and wingtip vortices of the shown ships (N,)."""
    age = now - trails.time
    fresh = np.clip(1.0 - age / trails.seconds, 0.0, 1.0)  # 1 at the ship, 0 at the end
    valid = shown[:, None] & np.isfinite(trails.time) & (age <= trails.seconds)
    colours = np.array([c.floats() for c in theme.trail], np.float32)[team][:, None, :]
    colour = np.repeat(colours, trails.x.shape[1], 1)
    colour[..., 3] *= fresh
    width = OLDEST_PIXELS + (NEWEST_PIXELS - OLDEST_PIXELS) * fresh
    trail = _strip(trails.x, trails.y, width, colour, valid)

    threshold = theme.vortex_threshold
    hard = np.clip((np.abs(trails.turn) - threshold) / max(threshold, 1e-6) * VORTEX_RAMP,
                   0.0, 1.0)
    wingtip = np.array([WINGTIP[int(k)] for k in ship_type], np.float32) * size
    offset = np.sign(trails.turn) * wingtip[:, None]  # + is the left wingtip
    vortex_x = trails.x - np.sin(trails.heading) * offset
    vortex_y = trails.y + np.cos(trails.heading) * offset
    vortex_colour = np.broadcast_to(np.array(theme.vortex.floats(), np.float32),
                                    colour.shape).copy()
    vortex_colour[..., 3] *= hard * fresh
    vortex_valid = valid & (hard > 0.0)
    vortex = _strip(vortex_x, vortex_y, np.full_like(width, VORTEX_PIXELS), vortex_colour,
                    vortex_valid)
    vertices = np.concatenate([trail[0], vortex[0]])
    indices = np.concatenate([trail[1], vortex[1] + len(trail[0])])
    return vertices, indices


class TrailsPass:
    def __init__(self, context: moderngl.Context) -> None:
        self.context = context
        self.program = program(context, "strips")
        self.vertices = context.buffer(reserve=4, dynamic=True)
        self.indices = context.buffer(reserve=4, dynamic=True)
        self.vertex_array = context.vertex_array(
            self.program, [(self.vertices, "2f 2f 1f 1f 4f", "position", "normal", "side",
                            "width", "tint")], index_buffer=self.indices, index_element_size=4)

    def draw(self, camera: dict, vertices: np.ndarray, indices: np.ndarray) -> None:
        if len(indices) == 0:
            return
        for buffer, data in ((self.vertices, vertices), (self.indices, indices)):
            if buffer.size < data.nbytes:
                buffer.orphan(data.nbytes)
            buffer.write(data.tobytes())
        set_uniforms(self.program, **camera)
        self.vertex_array.render(moderngl.TRIANGLES, vertices=len(indices))
