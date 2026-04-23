"""Schedule primitives for time-varying training parameters.

A Schedule is a callable (int) -> T that maps a global training step to a value.
All primitives return named inner functions (no lambdas) for clarity in tracebacks.

Primitives
----------
constant(value)
    Returns ``value`` for all steps.

linear(*keypoints)
    Linearly interpolates between (step, value) keypoints.
    Clamps to the first value before the first keypoint and the last value after
    the last keypoint. Requires ≥ 2 keypoints.

stepped(*keypoints)
    Step function: holds each value until the next keypoint's step is reached.
    Works for any value type (float, int, bool, frozenset, ...).
    Requires ≥ 1 keypoint.

exponential(*keypoints)
    Exponential (log-space linear) interpolation between keypoints.
    All values must be strictly positive. Clamps at both ends.
    Requires ≥ 2 keypoints.

join(*segments)
    Compose multiple schedules. Each segment is (activation_step, schedule_fn).
    At any step, the last segment whose activation_step ≤ step is used.
    Requires ≥ 1 segment. Segments must be in ascending activation_step order.

Compound example
----------------
    learning_rate = join(
        (0,           linear((0, 1e-7), (5_000_000, 3e-4))),   # warmup
        (5_000_000,   constant(3e-4)),                           # hold
        (50_000_000,  exponential((50_000_000, 3e-4), (150_000_000, 1e-5))),
    )
"""

import math
from dataclasses import dataclass
from typing import Any, Callable

# A schedule maps a global step (int) to a value of any type.
# The concrete return type is encoded per-field in TrainingSchedule.
Schedule = Callable[[int], Any]


def constant(value: Any) -> Schedule:
    """Return ``value`` for every step."""

    def _schedule(step: int) -> Any:
        return value

    return _schedule


def linear(*keypoints: tuple[int, float]) -> Schedule:
    """Linearly interpolate between (step, value) keypoints.

    Clamps to the first value before the first keypoint and to the last
    value after the last keypoint. Requires ≥ 2 keypoints.
    """
    if len(keypoints) < 2:
        raise ValueError(
            f"linear() requires at least 2 keypoints, got {len(keypoints)}"
        )
    steps = [kp[0] for kp in keypoints]
    values = [kp[1] for kp in keypoints]

    def _schedule(step: int) -> float:
        if step <= steps[0]:
            return values[0]
        if step >= steps[-1]:
            return values[-1]
        for i in range(len(steps) - 1):
            if steps[i] <= step <= steps[i + 1]:
                t = (step - steps[i]) / (steps[i + 1] - steps[i])
                return values[i] + t * (values[i + 1] - values[i])
        return values[-1]  # unreachable

    return _schedule


def stepped(*keypoints: tuple[int, Any]) -> Schedule:
    """Step function: hold each value until the next keypoint step is reached.

    Works for any value type: float, int, bool, frozenset, etc.
    Requires ≥ 1 keypoint.
    """
    if len(keypoints) < 1:
        raise ValueError(
            f"stepped() requires at least 1 keypoint, got {len(keypoints)}"
        )

    def _schedule(step: int) -> Any:
        value = keypoints[0][1]
        for keypoint_step, keypoint_value in keypoints:
            if step >= keypoint_step:
                value = keypoint_value
        return value

    return _schedule


def exponential(*keypoints: tuple[int, float]) -> Schedule:
    """Exponential (log-space linear) interpolation between keypoints.

    All values must be strictly positive. Clamps at both ends.
    Requires ≥ 2 keypoints.
    """
    if len(keypoints) < 2:
        raise ValueError(
            f"exponential() requires at least 2 keypoints, got {len(keypoints)}"
        )
    for step, value in keypoints:
        if value <= 0:
            raise ValueError(
                f"exponential() requires all values > 0, got value={value} at step={step}"
            )
    steps = [kp[0] for kp in keypoints]
    values = [kp[1] for kp in keypoints]

    def _schedule(step: int) -> float:
        if step <= steps[0]:
            return values[0]
        if step >= steps[-1]:
            return values[-1]
        for i in range(len(steps) - 1):
            if steps[i] <= step <= steps[i + 1]:
                t = (step - steps[i]) / (steps[i + 1] - steps[i])
                return values[i] * (values[i + 1] / values[i]) ** t
        return values[-1]  # unreachable

    return _schedule


def join(*segments: tuple[int, Schedule]) -> Schedule:
    """Compose multiple schedules, each active from a given step.

    At any step, the last segment whose activation step ≤ current step is used.
    Segments must be provided in ascending activation-step order.
    Requires ≥ 1 segment.

    Example:
        join(
            (0,           linear((0, 1e-7), (5_000_000, 3e-4))),
            (5_000_000,   constant(3e-4)),
            (50_000_000,  exponential((50_000_000, 3e-4), (150_000_000, 1e-5))),
        )
    """
    if len(segments) < 1:
        raise ValueError(f"join() requires at least 1 segment, got {len(segments)}")
    for i in range(1, len(segments)):
        if segments[i][0] <= segments[i - 1][0]:
            raise ValueError(
                f"join() segments must be in ascending activation-step order: "
                f"segment {i} step {segments[i][0]} <= segment {i - 1} step {segments[i - 1][0]}"
            )

    def _schedule(step: int) -> Any:
        active_fn = segments[0][1]
        for activation_step, schedule_fn in segments:
            if step >= activation_step:
                active_fn = schedule_fn
        return active_fn(step)

    return _schedule


@dataclass(frozen=True)
class TrainingSchedule:
    """All time-varying training parameters as per-field schedule functions.

    Each field is a callable (int) -> T that maps a global training step to its
    current value. Use the schedule primitives (constant, linear, stepped,
    exponential, join) to construct each field.

    Static reward weights live in RewardConfig. Only the group scales
    (true_reward_scale, global_scale, local_scale) are here because they
    realistically vary between BC and RL phases.
    """

    # --- Optimization ---
    learning_rate: Callable[[int], float]
    policy_gradient_coef: Callable[[int], float]  # 0.0 = BC only; 1.0 = full RL
    entropy_coef: Callable[[int], float]
    behavior_cloning_coef: Callable[[int], float]
    value_function_coef: Callable[[int], float]
    sigreg_coef: Callable[[int], float]  # weight for SIGReg encoder regularization loss

    # --- Reward group scales ---
    true_reward_scale: Callable[[int], float]  # multiplier for: ally_win, enemy_win
    global_scale: Callable[
        [int], float
    ]  # multiplier for: global outcome + shaping (ally/enemy damage, death, facing, closing_speed, shoot_quality)
    local_scale: Callable[
        [int], float
    ]  # multiplier for: self-only per-ship rewards (kill_shot, kill_assist, damage_taken, damage_dealt, death)

    # --- Opponents ---
    scripted_fraction: Callable[[int], float]
    avg_model_fraction: Callable[[int], float]
    league_fraction: Callable[[int], float]
    allow_avg_model_updates: Callable[[int], bool]
    allow_scripted_in_roster: Callable[[int], bool]

    # --- Checkpointing / eval ---
    elo_eval_games: Callable[[int], int]
    elo_eval_interval: Callable[[int], int]  # 0 = disabled
    checkpoint_interval: Callable[[int], int]  # 0 = disabled
