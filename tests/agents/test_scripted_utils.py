"""Unit tests for shared scripted-agent geometry helpers.

Boundary coverage for turn_toward (AUDIT-024): the 5deg/15deg thresholds use
>= on the low end and < on the high end for the "normal" band, so a mechanical
extraction could easily flip that semantics without any test noticing.
"""

import math

import pytest
import torch

from boost_and_broadside.agents.scripted_utils import turn_toward
from boost_and_broadside.constants import TurnActions

_EPS_DEG = 0.01


@pytest.mark.parametrize(
    "angle_deg,expected",
    [
        # --- 5 degree threshold, positive (right) ---
        (5 - _EPS_DEG, TurnActions.GO_STRAIGHT),
        (5, TurnActions.TURN_RIGHT),
        (5 + _EPS_DEG, TurnActions.TURN_RIGHT),
        # --- 5 degree threshold, negative (left) ---
        (-(5 - _EPS_DEG), TurnActions.GO_STRAIGHT),
        (-5, TurnActions.TURN_LEFT),
        (-(5 + _EPS_DEG), TurnActions.TURN_LEFT),
        # --- 15 degree threshold, positive (right) ---
        (15 - _EPS_DEG, TurnActions.TURN_RIGHT),
        (15, TurnActions.SHARP_RIGHT),
        (15 + _EPS_DEG, TurnActions.SHARP_RIGHT),
        # --- 15 degree threshold, negative (left) ---
        (-(15 - _EPS_DEG), TurnActions.TURN_LEFT),
        (-15, TurnActions.SHARP_LEFT),
        (-(15 + _EPS_DEG), TurnActions.SHARP_LEFT),
    ],
)
def test_turn_toward_angle_boundaries(angle_deg: float, expected: int) -> None:
    """turn_toward must switch bands using >= on the low edge, < on the high edge."""
    rel_angle = torch.tensor([math.radians(angle_deg)], dtype=torch.float32)

    turn = turn_toward(rel_angle)

    assert turn.item() == expected


def test_turn_toward_zero_angle_goes_straight() -> None:
    """No heading error must never trigger a turn."""
    rel_angle = torch.zeros(1, dtype=torch.float32)

    turn = turn_toward(rel_angle)

    assert turn.item() == TurnActions.GO_STRAIGHT


def test_turn_toward_preserves_shape_and_dtype() -> None:
    """turn_toward must return an int32 tensor matching the input's shape."""
    rel_angle = torch.zeros((2, 3), dtype=torch.float32)

    turn = turn_toward(rel_angle)

    assert turn.shape == (2, 3)
    assert turn.dtype == torch.int32
