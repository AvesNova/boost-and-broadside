"""The camera: a world point at the middle of the view and a zoom in metres per pixel.

World y points up, so a mouse drag (pixels, y down) pans the other way in y.
"""

import dataclasses
import math

FIT_MARGIN = 1.08  # the fitted view shows the arena with this much room around it
CLOSEST = 0.05  # m per pixel at the closest zoom
FARTHEST_FACTOR = 3.0  # the farthest zoom, as a multiple of the fitted one


@dataclasses.dataclass
class Camera:
    width: int
    height: int
    centre_x: float = 0.0
    centre_y: float = 0.0
    metres_per_pixel: float = 1.0
    farthest: float = math.inf  # the largest metres per pixel; fit sets it

    def fit(self, radius: float) -> None:
        """Centre on the origin with a disc of this radius in view."""
        self.centre_x = self.centre_y = 0.0
        self.metres_per_pixel = 2.0 * radius * FIT_MARGIN / min(self.width, self.height)
        self.farthest = self.metres_per_pixel * FARTHEST_FACTOR

    def resize(self, width: int, height: int) -> None:
        self.width, self.height = max(width, 1), max(height, 1)

    def pan(self, dx: float, dy: float) -> None:
        """Move the view with a drag of (dx, dy) pixels."""
        self.centre_x -= dx * self.metres_per_pixel
        self.centre_y += dy * self.metres_per_pixel

    def world(self, px: float, py: float) -> tuple[float, float]:
        """The world point under pixel (px, py) (y down)."""
        return (
            self.centre_x + (px - 0.5 * self.width) * self.metres_per_pixel,
            self.centre_y - (py - 0.5 * self.height) * self.metres_per_pixel,
        )

    def zoom(self, factor: float, px: float, py: float) -> None:
        """Zoom by `factor` (above 1 zooms in) keeping the point under pixel (px, py)."""
        before = self.world(px, py)
        self.metres_per_pixel = min(max(self.metres_per_pixel / factor, CLOSEST), self.farthest)
        after = self.world(px, py)
        self.centre_x += before[0] - after[0]
        self.centre_y += before[1] - after[1]

    def follow(self, x: float, y: float) -> None:
        self.centre_x, self.centre_y = x, y

    def uniforms(self) -> dict:
        """The shaders' camera, pixel and viewport uniforms."""
        return dict(
            camera=(
                self.centre_x,
                self.centre_y,
                2.0 / (self.width * self.metres_per_pixel),
                2.0 / (self.height * self.metres_per_pixel),
            ),
            pixel=self.metres_per_pixel,
            viewport=(float(self.width), float(self.height)),
        )
