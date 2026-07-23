"""Watch-mode renderer invariants."""

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.train.rl.features import build_standard_coordinator
from boost_and_broadside.ui.renderer import (
    _GHOST_DPHI_ATT,
    _GHOST_DPHI_X,
    _GHOST_DPHI_Y,
)


def test_ghost_prediction_indices_match_coordinator_layout():
    """The ghost-decode channel constants must track the coordinator prediction layout.

    _draw_ghost_ships indexes the raw prediction tensor by fixed channel numbers;
    if the coordinator's prediction ordering ever changes, these must move with it.
    Pinning them against get_feature_names() turns a silent desync — exactly the
    stale-AUX_PRED_DIM drift AUDIT-022's addendum flagged — into a failing test.
    """
    names = build_standard_coordinator(ShipConfig()).get_feature_names()
    assert names[_GHOST_DPHI_X] == "position_x_0"
    assert names[_GHOST_DPHI_Y] == "position_y_0"
    assert names[_GHOST_DPHI_ATT] == "attitude_0"
