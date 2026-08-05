"""Unit tests for shared scripted-agent geometry helpers.

Boundary coverage for turn_toward (AUDIT-024): the 5deg/15deg thresholds use
>= on the low end and < on the high end for the "normal" band, so a mechanical
extraction could easily flip that semantics without any test noticing.
"""

import math

import pytest
import torch

from boost_and_broadside.agents.scripted_utils import turn_toward
from boost_and_broadside.agents.stochastic_config import StochasticAgentConfig
from boost_and_broadside.agents.stochastic_scripted import StochasticScriptedAgent
from boost_and_broadside.config import ShipConfig
from boost_and_broadside.constants import TurnActions
from tests.conftest import make_state

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


def _field_steering_state(*, num_fields: int):
    config = ShipConfig()
    state = make_state(
        num_envs=1,
        max_ships=2,
        max_bullets=0,
        ship_config=config,
        num_fields=num_fields,
    )
    state.ship_team_id[0] = torch.tensor([0, 1], dtype=torch.int32)
    state.ship_pos[0] = torch.tensor([100.0 + 100.0j, 300.0 + 100.0j])
    state.ship_attitude[0] = 1.0 + 0.0j
    if num_fields:
        state.field_pos[0, 0] = 100.0 + 55.0j
        state.field_radius[0, 0] = 40.0
        state.field_transition_width[0, 0] = 40.0
        state.field_index[0, 0] = 2.0
        state.field_delta_index[0, 0] = 1.0
        state.field_damage[0, 0] = 20.0
    return config, state


def test_scripted_turn_targets_ignore_nearby_fields() -> None:
    """The scripted controller aims and manoeuvres as if the medium were uniform.

    Its former material-aware steering measured net-negative: against a
    uniform-random agent it produced *more* interface crossings, so behaviour
    cloning imprinted extra crossing damage that the field_damage_taken penalty
    then had to unteach. Field representation comes from the auxiliary
    local_log_index prediction head, which never decays, rather than from BC.
    """
    config, ambient = _field_steering_state(num_fields=0)
    _, fielded = _field_steering_state(num_fields=1)
    agent = StochasticScriptedAgent(config, StochasticAgentConfig())

    torch.manual_seed(0)
    _, ambient_probs = agent.get_actions_and_probs(ambient)
    torch.manual_seed(0)
    _, field_probs = agent.get_actions_and_probs(fielded)

    assert torch.allclose(ambient_probs, field_probs)
