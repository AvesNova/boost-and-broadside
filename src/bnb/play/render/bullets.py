"""Pass 8: bullets, as tracers: a round head in the shooter's team colour and a short trail
behind it along its direction of flight, in the team's trail colour, leaving the head as
wide as the head (its diameter) and tapering and fading to a thin tail as a ship's trail
does (`trails`). Every bullet draws in every view.

The trail spans the last trail seconds of the bullet's flight as the screen shows it (that
many seconds of play times the playback speed), never past the point it was fired from,
so a fast bullet reads as one continuous streak at any time scale or playback speed. A
bullet has no size in the game (hits are tested against the ship's hit radius), so the
tracer is a fixed few pixels wide at every zoom, the true-size view included.

The look is an option (`BulletLook`, the Settings menu): the style (a tracer, or the trail
alone with no head), the head's size, the trail's width at its two ends and its length in
seconds; the defaults are the tracer above."""

import dataclasses

import moderngl
import numpy as np

from ..themes import Theme
from .programs import program, set_uniforms
from .trails import OLDEST_PIXELS

STYLES = {"tracer": "Tracer", "trail": "Trail only"}  # by style: its label
FLOATS = 13


@dataclasses.dataclass
class BulletLook:
    """How bullets draw: style (STYLES: "tracer", a round head and its trail, or "trail",
    the trail alone); head_pixels, the head's radius; trail_head_pixels and
    trail_tail_pixels, the trail's width at the head's end and at its tail (screen pixels,
    as every width in the scene); trail_seconds, the trail's length in seconds of the
    screen's time."""

    style: str = "tracer"
    head_pixels: float = 2.5
    trail_head_pixels: float = 5.0  # the head's diameter: the trail leaves it as wide
    trail_tail_pixels: float = OLDEST_PIXELS
    trail_seconds: float = 0.04

    def __post_init__(self) -> None:
        if self.style not in STYLES:
            raise ValueError(f"bullet style {self.style!r} is not one of {list(STYLES)}")


def tracer_lengths(speed: np.ndarray, age: np.ndarray, seconds: float) -> np.ndarray:
    """Each tracer's trail (m): the distance the bullet flew in the last `seconds` of play,
    at its speed, up to where it was fired (its `age` ago)."""
    return (speed * np.clip(np.minimum(seconds, age), 0.0, None)).astype(np.float32)


class BulletsPass:
    """Pass 8: the bullets, one instanced tracer per active bullet."""

    def __init__(self, context: moderngl.Context, capacity: int = 1024) -> None:
        self.context = context
        self.program = program(context, "bullets")
        corners = np.array([0, -1, 1, -1, 0, 1, 1, 1], np.float32)
        self.corners = context.buffer(corners.tobytes())
        self.capacity = 0
        self._reserve(capacity)

    def _reserve(self, count: int) -> None:
        if count <= self.capacity:
            return
        if self.capacity:
            self.vertex_array.release()
            self.instances.release()
        self.capacity = max(count, 2 * self.capacity)
        self.instances = self.context.buffer(reserve=self.capacity * FLOATS * 4, dynamic=True)
        self.vertex_array = self.context.vertex_array(
            self.program,
            [
                (self.corners, "2f", "corner"),
                (
                    self.instances,
                    "2f 2f 1f 4f 4f/i",
                    "head",
                    "direction",
                    "trail_metres",
                    "head_tint",
                    "trail_tint",
                ),
            ],
        )

    def draw(
        self,
        camera: dict,
        theme: Theme,
        x: np.ndarray,
        y: np.ndarray,
        direction_x: np.ndarray,
        direction_y: np.ndarray,
        trail: np.ndarray,
        team: np.ndarray,
        look: BulletLook,
    ) -> None:
        """The listed bullets (arrays of one length): head position, unit direction of
        flight, trail length (m, `tracer_lengths`) and team, drawn in this look (a head
        radius of 0: no head)."""
        if len(x) == 0:
            return
        self._reserve(len(x))
        heads = np.array([c.floats() for c in theme.team], np.float32)[team]
        trails = np.array([c.floats() for c in theme.trail], np.float32)[team]
        table = np.concatenate(
            [np.stack([x, y, direction_x, direction_y, trail], -1), heads, trails], -1
        ).astype(np.float32)
        self.instances.orphan()  # fresh storage: no wait on the last frame's draw
        self.instances.write(table.tobytes())
        set_uniforms(
            self.program,
            **camera,
            head_pixels=look.head_pixels if look.style == "tracer" else 0.0,
            trail_pixels=(look.trail_tail_pixels, look.trail_head_pixels),
        )
        self.vertex_array.render(moderngl.TRIANGLE_STRIP, vertices=4, instances=len(x))
