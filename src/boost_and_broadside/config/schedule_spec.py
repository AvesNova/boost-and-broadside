"""Schedules as data: a keypoint table per parameter, compiled to callables.

Every time-varying parameter is a table of ``(step, value, interp)`` rows. Each
row states a value and how to get from it to the next one, and the last row
holds forever. That is the whole language::

    learning_rate = (
        (0, 1e-7, "linear"),            # warm up to ...
        (5_000_000, 4.5e-4, "hold"),    # ... and sit there until ...
        (100_000_000, 4.5e-4, "exponential"),  # ... decaying to ...
        (500_000_000, 1.5e-4, "hold"),  # ... and staying.
    )

This replaced a tree of ``ScheduleSpec`` nodes -- ``join`` of ``linear``,
``constant`` and ``exponential`` sub-specs -- that expressed the same four
numbers as eleven nested objects, and stated every interior step twice because a
segment's activation and its first keypoint had to agree by hand. The flat form
cannot disagree with itself, and it is JSON: a run can record the schedule it is
running under, which is what ``checkpoints/<run>/config.json`` needs.

A schedule is a pure function of ``global_step``, so nothing about the current
position is stored -- resuming evaluates the table at the restored step.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, fields
from typing import Any, Literal

from boost_and_broadside.config.schedule import Schedule, TrainingSchedule

Interp = Literal["hold", "linear", "exponential"]
Keypoint = tuple[int, Any, Interp]
Keypoints = tuple[Keypoint, ...]

_INTERPOLATIONS = ("hold", "linear", "exponential")


def hold(value: Any) -> Keypoints:
    """A table of one row: ``value`` at every step."""

    return ((0, value, "hold"),)


def validate_keypoints(table: Sequence[Keypoint], *, name: str = "schedule") -> Keypoints:
    """Check one table and return it normalized, or raise naming the problem."""

    if not table:
        raise ValueError(f"{name} must have at least one keypoint")
    rows = tuple((int(step), value, interp) for step, value, interp in table)
    for step, value, interp in rows:
        if interp not in _INTERPOLATIONS:
            raise ValueError(f"{name}: unknown interpolation {interp!r} at step {step}")
        if interp != "hold" and not isinstance(value, int | float):
            raise ValueError(f"{name}: {interp} interpolation needs a number at step {step}")
        if interp == "exponential" and value <= 0:
            raise ValueError(f"{name}: exponential interpolation needs a positive value")
    for previous, current in zip(rows, rows[1:], strict=False):
        if current[0] <= previous[0]:
            raise ValueError(f"{name}: keypoint steps must strictly increase")
        if previous[2] == "exponential" and current[1] <= 0:
            raise ValueError(f"{name}: exponential interpolation needs a positive target")
    return rows


def compile_keypoints(table: Sequence[Keypoint], *, name: str = "schedule") -> Schedule:
    """Compile one keypoint table into a ``(step) -> value`` callable."""

    rows = validate_keypoints(table, name=name)

    def _schedule(step: int) -> Any:
        if step <= rows[0][0]:
            return rows[0][1]
        for current, following in zip(rows, rows[1:], strict=False):
            start, value, interp = current
            end, target, _ = following
            if step >= end:
                continue
            if interp == "hold" or step == start:
                # Landing exactly on a keypoint returns the value written there.
                # Without this an exponential row would answer
                # exp(log(v)) != v -- 4.5e-4 read back as 0.0004499999999999998
                # at the step where the decay starts.
                return value
            fraction = (step - start) / (end - start)
            if interp == "linear":
                return value + fraction * (target - value)
            # Ratio-and-power, not exp-of-log-lerp. The two are equal in real
            # arithmetic and differ in the last bits in floating point, and this
            # is the spelling the schedules were tuned and measured under.
            return value * (target / value) ** fraction
        return rows[-1][1]

    return _schedule


@dataclass(frozen=True)
class TrainingScheduleSpec:
    """One keypoint table per field of :class:`TrainingSchedule`."""

    learning_rate: Keypoints
    policy_gradient_coef: Keypoints
    entropy_coef: Keypoints
    behavior_cloning_coef: Keypoints
    value_function_coef: Keypoints
    sigreg_coef: Keypoints
    true_reward_scale: Keypoints
    global_scale: Keypoints
    local_scale: Keypoints
    league_fraction: Keypoints
    checkpoint_interval: Keypoints
    num_epochs: Keypoints
    target_kl: Keypoints
    high_winrate_threshold: Keypoints
    high_winrate_target_kl: Keypoints

    def __post_init__(self) -> None:
        for field in fields(self):
            object.__setattr__(
                self,
                field.name,
                validate_keypoints(getattr(self, field.name), name=field.name),
            )

    def compile(self) -> TrainingSchedule:
        return TrainingSchedule(
            **{
                field.name: compile_keypoints(getattr(self, field.name), name=field.name)
                for field in fields(self)
            }
        )
