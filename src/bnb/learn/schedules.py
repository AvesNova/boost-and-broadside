"""Settings that change over a run: a value piecewise linear in the run's environment
steps between keypoints (`Keypoints`)."""

import dataclasses


@dataclasses.dataclass(frozen=True)
class Keypoints:
    """A value at each of `steps` (environment steps, ascending, the first at least 0),
    linear between them and constant before the first and after the last. One keypoint is
    a constant. `Keypoints((0, 20_000_000, 60_000_000), (2.0, 2.0, 1.0))` holds 2 for the
    first 20M steps, then falls linearly to 1 at 60M and stays there."""

    steps: tuple[int, ...] = (0,)
    values: tuple[float, ...] = (1.0,)

    def __post_init__(self) -> None:
        if not self.steps or len(self.steps) != len(self.values):
            raise ValueError(
                f"keypoints need one value per step and some: steps {list(self.steps)}, "
                f"values {list(self.values)}"
            )
        if self.steps[0] < 0 or any(
            later <= earlier for earlier, later in zip(self.steps, self.steps[1:])
        ):
            raise ValueError(f"keypoint steps {list(self.steps)} must be ascending from 0 or later")

    def value(self, step: int) -> float:
        """The value at environment step `step`."""
        if step <= self.steps[0]:
            return self.values[0]
        for (start, low), (end, high) in zip(
            zip(self.steps, self.values), zip(self.steps[1:], self.values[1:])
        ):
            if step <= end:
                return low + (high - low) * (step - start) / (end - start)
        return self.values[-1]
