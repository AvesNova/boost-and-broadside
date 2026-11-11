"""Pass 5: the ships, instanced silhouettes filled with the team colour, every one
outlined, never smaller on screen than MINIMUM_RING_PIXELS unless drawn at true size.

At true size (the top bar's True size) a ship is drawn at its size in metres at any zoom:
its ring at its hitbox, whose radius is half the type's wingspan (the ship cards'
`hit_radius`), and its silhouette grown to fill the ring from wingtip to wingtip, so the
drawn span is the real one. Otherwise the ring is the hitbox but never smaller on screen
than MINIMUM_RING_PIXELS, and the silhouette sits inside it."""

import moderngl
import numpy as np

from ..themes import Theme
from .programs import program, set_uniforms
from .silhouettes import HALF_OUTLINES, WINGTIPS, mesh, silhouette_of

MINIMUM_RING_PIXELS = 14.0  # the ring radius (and the silhouette with it) at the farthest zoom
OUTLINE_PIXELS = 1.3
OUTLINE_FRACTION = 0.035  # of the ring radius, when the ship is drawn large


def ring_radius(
    hit_radius: np.ndarray, metres_per_pixel: float, true_size: bool = False
) -> np.ndarray:
    """The drawn ring radius (m): the hitbox, or the minimum on-screen size where that is
    larger and the ships are not drawn at true size."""
    if true_size:
        return np.asarray(hit_radius, np.float32)
    return np.maximum(hit_radius, MINIMUM_RING_PIXELS * metres_per_pixel).astype(np.float32)


def silhouette_size(ring: np.ndarray, ship_type: np.ndarray, true_size: bool) -> np.ndarray:
    """The silhouette's scale (m per silhouette unit) for ships of this ring radius (m) and
    type: the ring radius, which keeps the silhouette inside its ring, or at true size the
    ring over the silhouette's half span, so the wingtips reach the ring."""
    if not true_size:
        return ring
    half_span = np.array([WINGTIPS[key] for key in silhouette_of(ship_type)], np.float32)
    return (ring / half_span).astype(np.float32)


class ShipsPass:
    """Pass 5: the ships, an instanced outlined silhouette per ship type."""

    def __init__(self, context: moderngl.Context, capacity: int = 256) -> None:
        self.context = context
        self.program = program(context, "ships")
        self.meshes = {}
        for silhouette in HALF_OUTLINES:
            vertices, indices = mesh(silhouette)
            self.meshes[silhouette] = (
                context.buffer(vertices.tobytes()),
                context.buffer(indices.tobytes()),
                indices.size,
            )
        self.capacity = 0
        self._reserve(capacity)

    def _reserve(self, count: int) -> None:
        if count <= self.capacity:
            return
        if self.capacity:
            for vertex_array in self.vertex_arrays.values():
                vertex_array.release()
            self.instances.release()
        self.capacity = max(count, 2 * self.capacity)
        self.instances = self.context.buffer(reserve=self.capacity * 8 * 4, dynamic=True)
        self.vertex_arrays = {
            silhouette: self.context.vertex_array(
                self.program,
                [
                    (vertices, "2f 2f 1f", "position", "miter", "outline"),
                    (self.instances, "2f 1f 1f 4f/i", "centre", "heading", "size", "fill"),
                ],
                index_buffer=indices,
                index_element_size=4,
            )
            for silhouette, (vertices, indices, _) in self.meshes.items()
        }

    def draw(
        self,
        camera: dict,
        theme: Theme,
        x: np.ndarray,
        y: np.ndarray,
        heading: np.ndarray,
        size: np.ndarray,
        team: np.ndarray,
        ship_type: np.ndarray,
        opacity: np.ndarray | None = None,
    ) -> None:
        """The listed ships (arrays of one length; size the silhouette's scale in m,
        `silhouette_size`), each
        silhouette in one instanced draw. opacity: each ship's, fill and outline (a
        prediction overlay's ghosts); None draws them opaque."""
        if len(x) == 0:
            return
        self._reserve(len(x))
        silhouette = silhouette_of(ship_type)
        colours = np.array([colour.floats() for colour in theme.team], np.float32)[team]
        if opacity is not None:
            colours[:, 3] *= opacity
        set_uniforms(
            self.program,
            **camera,
            outline_colour=theme.outline.floats(),
            outline_pixels=OUTLINE_PIXELS,
            outline_fraction=OUTLINE_FRACTION,
        )
        for key, vertex_array in self.vertex_arrays.items():
            chosen = silhouette == key
            count = int(chosen.sum())
            if count == 0:
                continue
            table = np.concatenate(
                [
                    np.stack([x[chosen], y[chosen], heading[chosen], size[chosen]], -1),
                    colours[chosen],
                ],
                -1,
            )
            self.instances.orphan()  # fresh storage: no wait on the last frame's draw
            self.instances.write(table.astype(np.float32).tobytes())
            vertex_array.render(moderngl.TRIANGLES, vertices=self.meshes[key][2], instances=count)
