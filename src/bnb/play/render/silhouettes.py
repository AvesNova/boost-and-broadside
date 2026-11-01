"""Ship silhouettes from the Komet family (docs/rewrite/visual-mockups/ships3.png), as
triangle meshes with an outline.

Coordinates have the nose along +x and are in units of the ring radius (the hitbox), so a
silhouette sits inside its shield and energy ring. Each outline lists one side, nose to
tail; the other side mirrors it. A ship type without a silhouette of its own draws the
generic dart (`silhouette_of`).
"""

import numpy as np

# Half outlines (forward, sideways) in the mock-up's units (ring radius 73.3), by ship type.
MOCKUP_RING = 73.3
GENERIC = -1  # the key of the silhouette for a ship type with none of its own
HALF_OUTLINES = {
    0: ((50, 0), (32, 7), (8, 12), (-22, 42), (-30, 40), (-20, 10), (-34, 7), (-36, 0)),  # Komet
    1: ((46, 0), (30, 11), (6, 17), (-20, 44), (-30, 42), (-20, 14), (-34, 10), (-36, 0)),
    GENERIC: ((46, 0), (-28, 34), (-20, 0)),  # a plain dart
}
MITER_LIMIT = 2.0
WINGTIPS = {key: max(abs(side) for _, side in outline) / MOCKUP_RING
            for key, outline in HALF_OUTLINES.items()}  # half span, in ring radii


def silhouette_of(ship_type: np.ndarray) -> np.ndarray:
    """The key in HALF_OUTLINES each ship type draws with: its own, or GENERIC."""
    own = np.array([key for key in HALF_OUTLINES if key != GENERIC])
    return np.where(np.isin(ship_type, own), ship_type, GENERIC)


def polygon(silhouette: int) -> np.ndarray:
    """(V, 2) the closed outline of a HALF_OUTLINES key, counter-clockwise, in ring radii."""
    half = np.array(HALF_OUTLINES[silhouette], np.float64) / MOCKUP_RING
    points = np.concatenate([half, (half[1:-1] * (1.0, -1.0))[::-1]])
    return points if _area(points) > 0.0 else points[::-1].copy()


def _area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def triangulate(points: np.ndarray) -> np.ndarray:
    """(V - 2, 3) vertex indices of a simple counter-clockwise polygon (ear clipping)."""
    remaining = list(range(len(points)))
    triangles = []

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    while len(remaining) > 3:
        for k in range(len(remaining)):
            i, j, n = remaining[k - 1], remaining[k], remaining[(k + 1) % len(remaining)]
            a, b, c = points[i], points[j], points[n]
            if cross(a, b, c) <= 0.0:
                continue  # a reflex corner is not an ear
            inside = any(cross(a, b, points[m]) >= 0 and cross(b, c, points[m]) >= 0
                         and cross(c, a, points[m]) >= 0
                         for m in remaining if m not in (i, j, n))
            if not inside:
                triangles.append((i, j, n))
                remaining.pop(k)
                break
        else:
            raise ValueError("not a simple counter-clockwise polygon")
    triangles.append(tuple(remaining))
    return np.array(triangles, np.int32)


def miters(points: np.ndarray) -> np.ndarray:
    """(V, 2) each vertex's outward offset per unit of outline width (capped)."""
    edges = np.roll(points, -1, 0) - points
    normals = np.stack([edges[:, 1], -edges[:, 0]], -1)  # outward for counter-clockwise
    normals /= np.linalg.norm(normals, axis=-1, keepdims=True)
    previous = np.roll(normals, 1, 0)
    bisector = previous + normals
    bisector /= np.linalg.norm(bisector, axis=-1, keepdims=True)
    scale = 1.0 / np.maximum((bisector * normals).sum(-1), 1.0 / MITER_LIMIT)
    return bisector * scale[:, None]


def mesh(silhouette: int) -> tuple[np.ndarray, np.ndarray]:
    """(vertices (2V, 5): position, miter, outline flag; indices (2 (V - 2), 3)): the
    outline copy first, then the fill."""
    points = polygon(silhouette)
    triangles = triangulate(points)
    offsets = miters(points)
    outline = np.concatenate([points, offsets, np.ones((len(points), 1))], -1)
    fill = np.concatenate([points, np.zeros_like(offsets), np.zeros((len(points), 1))], -1)
    vertices = np.concatenate([outline, fill]).astype(np.float32)
    indices = np.concatenate([triangles, triangles + len(points)]).astype(np.int32)
    return vertices, indices
