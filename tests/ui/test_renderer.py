"""Watch-mode renderer invariants."""

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.train.rl.features import build_standard_coordinator
from boost_and_broadside.ui.renderer import (
    _GHOST_DPHI_ATT,
    _GHOST_DPHI_X,
    _GHOST_DPHI_Y,
    GameRenderer,
    field_border_pattern,
    field_color,
    wrapped_field_centers,
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


def test_field_colors_separate_fast_and_slow_and_strengthen_with_magnitude():
    low = field_color(-1)
    very_low = field_color(-2)
    high = field_color(1)
    very_high = field_color(2)
    assert low[2] > low[0] and very_low[2] > very_low[0]  # cyan/blue family
    assert high[0] > high[1] and high[2] > high[1]  # violet family
    assert sum(very_low) > sum(low)
    assert sum(very_high) > sum(high)


def test_field_damage_levels_map_to_dotted_dashed_and_solid():
    assert field_border_pattern(0) == ("dotted", 1)
    assert field_border_pattern(1) == ("dashed", 2)
    assert field_border_pattern(2) == ("solid", 3)


def test_wrapped_field_centers_include_visible_edge_copies():
    copies = wrapped_field_centers(2.0 + 2.0j, 10.0, (100.0, 100.0))
    assert set(copies) == {2.0 + 2.0j, 102.0 + 2.0j, 2.0 + 102.0j, 102.0 + 102.0j}


def test_field_outline_patterns_leave_the_interior_unfilled():
    import pygame

    background = (10, 10, 20)
    color = (40, 225, 255)
    for pattern, width in (("dotted", 1), ("dashed", 2), ("solid", 3)):
        surface = pygame.Surface((80, 80))
        surface.fill(background)
        GameRenderer._draw_field_outline(surface, (40, 40), 25, color, pattern, width)
        assert surface.get_at((40, 40))[:3] == background
