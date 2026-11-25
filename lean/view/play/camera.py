"""The camera: a world point at the middle of the view and a zoom in metres per pixel, and
the look-ahead that leads a followed ship's view in its direction of travel.

World y points up, so a mouse drag (pixels, y down) pans the other way in y.
"""

import dataclasses
import math

FIT_MARGIN = 1.08  # the fitted view shows the arena with this much room around it
CLOSEST = 0.05  # m per pixel at the closest zoom
FARTHEST_FACTOR = 3.0  # the farthest zoom, as a multiple of the fitted one
LEAD_SECONDS = 4.0  # the lead is where the ship will be this many seconds on, up to the cap
LEAD_CAP = 0.35  # the longest lead, as a fraction of half the view's smaller side
LEAD_FREQUENCY = 2.0  # rad/s: the spring's natural frequency (settles in about 2.4 s)
LEAD_STEP_LIMIT = 0.5  # s of game time: a longer step between frames (a seek) is no velocity


@dataclasses.dataclass
class Camera:
    """The view: the world point at its middle, the zoom in metres per pixel, and the
    target's size in pixels."""

    width: int
    height: int
    centre_x: float = 0.0
    centre_y: float = 0.0
    metres_per_pixel: float = 1.0
    farthest: float = math.inf  # the largest metres per pixel; fit sets it
    fitted: float = math.inf  # the fitted view's metres per pixel; fit sets it

    def fit(self, radius: float) -> None:
        """Centre on the origin with a disc of this radius in view."""
        self.centre_x = self.centre_y = 0.0
        self.metres_per_pixel = 2.0 * radius * FIT_MARGIN / min(self.width, self.height)
        self.fitted = self.metres_per_pixel
        self.farthest = self.fitted * FARTHEST_FACTOR

    def zoom_in_to(self, factor: float) -> None:
        """Zoom in to `factor` times the fitted view's, unless the view is already closer."""
        self.metres_per_pixel = min(self.metres_per_pixel, self.fitted / factor)

    def resize(self, width: int, height: int) -> None:
        """Take the target's new size in pixels (at least 1 by 1)."""
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

    def pixel(self, x: float, y: float) -> tuple[float, float]:
        """The pixel (y down) a world point (x, y) is drawn at: the inverse of `world`."""
        return (
            0.5 * self.width + (x - self.centre_x) / self.metres_per_pixel,
            0.5 * self.height - (y - self.centre_y) / self.metres_per_pixel,
        )

    def zoom(self, factor: float, px: float, py: float) -> None:
        """Zoom by `factor` (above 1 zooms in) keeping the point under pixel (px, py)."""
        before = self.world(px, py)
        self.metres_per_pixel = min(max(self.metres_per_pixel / factor, CLOSEST), self.farthest)
        after = self.world(px, py)
        self.centre_x += before[0] - after[0]
        self.centre_y += before[1] - after[1]

    def follow(self, x: float, y: float) -> None:
        """Centre the view on the world point (x, y)."""
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


def lead_target(
    velocity_x: float, velocity_y: float, metres_per_pixel: float, width: int, height: int
) -> tuple[float, float]:
    """The lead (pixels, world axes: y up) for a ship at this velocity (m/s): along the
    velocity, LEAD_SECONDS of travel on screen, saturating smoothly (tanh) at LEAD_CAP of
    half the view's smaller side."""
    speed = math.hypot(velocity_x, velocity_y)
    if speed == 0.0:
        return 0.0, 0.0
    cap = LEAD_CAP * 0.5 * min(width, height)
    length = cap * math.tanh(speed * LEAD_SECONDS / metres_per_pixel / cap)
    return velocity_x / speed * length, velocity_y / speed * length


def spring_step(
    position: float, rate: float, target: float, elapsed: float, frequency: float = LEAD_FREQUENCY
) -> tuple[float, float]:
    """A critically damped spring toward `target` after `elapsed` seconds: the exact
    solution for a target held over the step, so the path does not depend on how the time
    is split into frames. From rest it never overshoots. Returns (position, rate)."""
    offset = position - target
    decay = math.exp(-frequency * elapsed)
    slope = rate + frequency * offset
    return (
        target + (offset + slope * elapsed) * decay,
        (rate - frequency * slope * elapsed) * decay,
    )


@dataclasses.dataclass
class LookAhead:
    """The followed ship's lead: an offset of the view from the ship along its velocity
    (`lead_target`), eased by a critically damped spring (`spring_step`) in display time.
    The offset is held in pixels, so a zoom keeps the ship where it is on screen. The
    velocity is the drawn ship's: its move between frames over the game time between them,
    so it is the motion the player sees (a frame with no game time between, paused or
    after a seek, keeps the last). A new ship, or a spawn, starts from no lead."""

    ship: int | None = None
    x: float = 0.0  # the ship's drawn position at the last frame (m)
    y: float = 0.0
    time: float = 0.0  # the game time of the last frame (s)
    velocity_x: float = 0.0  # m/s
    velocity_y: float = 0.0
    offset_x: float = 0.0  # pixels, world axes (y up)
    offset_y: float = 0.0
    rate_x: float = 0.0  # pixels per display second
    rate_y: float = 0.0

    def update(
        self,
        ship: int,
        x: float,
        y: float,
        time: float,
        elapsed: float,
        camera: Camera,
        on: bool,
        spawned: bool = False,
    ) -> tuple[float, float]:
        """Take the ship's drawn position at game time `time`, `elapsed` display seconds
        after the last frame, and return the view's offset from it in metres. on: lead
        (off: the offset eases back to the ship)."""
        if ship != self.ship or spawned:
            self.ship, self.x, self.y, self.time = ship, x, y, time
            self.velocity_x = self.velocity_y = self.offset_x = self.offset_y = 0.0
            self.rate_x = self.rate_y = 0.0
            return 0.0, 0.0
        step = time - self.time
        if 0.0 < step <= LEAD_STEP_LIMIT:
            self.velocity_x, self.velocity_y = (x - self.x) / step, (y - self.y) / step
        self.x, self.y, self.time = x, y, time
        target_x, target_y = (
            lead_target(
                self.velocity_x,
                self.velocity_y,
                camera.metres_per_pixel,
                camera.width,
                camera.height,
            )
            if on
            else (0.0, 0.0)
        )
        self.offset_x, self.rate_x = spring_step(self.offset_x, self.rate_x, target_x, elapsed)
        self.offset_y, self.rate_y = spring_step(self.offset_y, self.rate_y, target_y, elapsed)
        return self.offset_x * camera.metres_per_pixel, self.offset_y * camera.metres_per_pixel
