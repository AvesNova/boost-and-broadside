"""The frame-rate readout's numbers: the frame time smoothed over about half a second
(frames per second is its inverse) and the longest frame of the last whole second, so a
stall shows for a second after it even when the smoothed rate hides it. A frame adds a few
float operations; nothing is stored per frame."""

import math

SMOOTHING_SECONDS = 0.5  # the smoothed frame time's time constant
WORST_WINDOW_SECONDS = 1.0  # the longest frame is over windows of this long


class FrameRate:
    """The frame rate from each frame's display seconds (`add`)."""

    def __init__(self) -> None:
        self.frame_seconds = 0.0  # smoothed; 0 until the first frame
        self.worst_seconds = 0.0  # the longest frame of the last whole window
        self._window_worst = 0.0
        self._window_seconds = 0.0

    def add(self, elapsed: float) -> None:
        """Count a frame that took `elapsed` seconds (a non-positive one is skipped)."""
        if elapsed <= 0.0:
            return
        if self.frame_seconds == 0.0:
            self.frame_seconds = elapsed
        else:
            weight = 1.0 - math.exp(-elapsed / SMOOTHING_SECONDS)
            self.frame_seconds += weight * (elapsed - self.frame_seconds)
        self._window_worst = max(self._window_worst, elapsed)
        self._window_seconds += elapsed
        if self._window_seconds >= WORST_WINDOW_SECONDS:
            self.worst_seconds = self._window_worst
            self._window_worst, self._window_seconds = 0.0, 0.0

    @property
    def per_second(self) -> float:
        """Frames per second (0 before the first frame)."""
        return 1.0 / self.frame_seconds if self.frame_seconds > 0.0 else 0.0

    def text(self) -> str:
        """The readout: "60 fps  16.7 ms  max 18.0 ms" (the max once a second has passed)."""
        line = f"{self.per_second:.0f} fps  {1e3 * self.frame_seconds:.1f} ms"
        if self.worst_seconds > 0.0:
            line += f"  max {1e3 * self.worst_seconds:.0f} ms"
        return line
