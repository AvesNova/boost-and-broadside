"""Declarative, serializable schedule intent compiled to runtime callables."""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Literal

from boost_and_broadside.config.schedule import (
    Schedule,
    TrainingSchedule,
    constant,
    exponential,
    join,
    linear,
    stepped,
)

ScheduleKind = Literal["constant", "linear", "stepped", "exponential", "join"]


@dataclass(frozen=True)
class ScheduleSpec:
    """One schedule expression with no executable or process-local state."""

    kind: ScheduleKind
    value: Any = None
    keypoints: tuple[tuple[int, Any], ...] = ()
    segments: tuple[tuple[int, ScheduleSpec], ...] = ()

    def __post_init__(self) -> None:
        if self.kind not in {"constant", "linear", "stepped", "exponential", "join"}:
            raise ValueError(f"unknown schedule kind: {self.kind!r}")
        populated = int(self.value is not None) + bool(self.keypoints) + bool(self.segments)
        if self.kind == "constant":
            if self.keypoints or self.segments:
                raise ValueError("constant schedule cannot contain keypoints or segments")
            return
        if populated != 1:
            raise ValueError(f"{self.kind} schedule must define exactly one payload")
        if self.kind in {"linear", "exponential"} and len(self.keypoints) < 2:
            raise ValueError(f"{self.kind} schedule requires at least two keypoints")
        if self.kind == "stepped" and not self.keypoints:
            raise ValueError("stepped schedule requires at least one keypoint")
        if self.kind == "join" and not self.segments:
            raise ValueError("join schedule requires at least one segment")
        ordered = self.segments if self.kind == "join" else self.keypoints
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if current[0] <= previous[0]:
                raise ValueError(f"{self.kind} schedule steps must be strictly increasing")
        if self.kind == "exponential" and any(value <= 0 for _, value in self.keypoints):
            raise ValueError("exponential schedule values must be positive")


def constant_spec(value: Any) -> ScheduleSpec:
    return ScheduleSpec(kind="constant", value=value)


def linear_spec(*keypoints: tuple[int, float]) -> ScheduleSpec:
    return ScheduleSpec(kind="linear", keypoints=keypoints)


def stepped_spec(*keypoints: tuple[int, Any]) -> ScheduleSpec:
    return ScheduleSpec(kind="stepped", keypoints=keypoints)


def exponential_spec(*keypoints: tuple[int, float]) -> ScheduleSpec:
    return ScheduleSpec(kind="exponential", keypoints=keypoints)


def join_spec(*segments: tuple[int, ScheduleSpec]) -> ScheduleSpec:
    return ScheduleSpec(kind="join", segments=segments)


def compile_schedule(spec: ScheduleSpec) -> Schedule:
    """Compile declarative intent through the established runtime primitives."""

    match spec.kind:
        case "constant":
            return constant(spec.value)
        case "linear":
            return linear(*spec.keypoints)
        case "stepped":
            return stepped(*spec.keypoints)
        case "exponential":
            return exponential(*spec.keypoints)
        case "join":
            return join(*((step, compile_schedule(child)) for step, child in spec.segments))
    raise AssertionError(f"unreachable schedule kind: {spec.kind}")


@dataclass(frozen=True)
class TrainingScheduleSpec:
    """Serializable counterpart to every field in ``TrainingSchedule``."""

    learning_rate: ScheduleSpec
    policy_gradient_coef: ScheduleSpec
    entropy_coef: ScheduleSpec
    behavior_cloning_coef: ScheduleSpec
    value_function_coef: ScheduleSpec
    sigreg_coef: ScheduleSpec
    true_reward_scale: ScheduleSpec
    global_scale: ScheduleSpec
    local_scale: ScheduleSpec
    league_fraction: ScheduleSpec
    checkpoint_interval: ScheduleSpec
    num_epochs: ScheduleSpec
    target_kl: ScheduleSpec
    high_winrate_threshold: ScheduleSpec
    high_winrate_target_kl: ScheduleSpec

    def compile(self) -> TrainingSchedule:
        return TrainingSchedule(
            **{
                field.name: compile_schedule(getattr(self, field.name))
                for field in fields(self)
            }
        )
