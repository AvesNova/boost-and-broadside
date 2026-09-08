"""Watch-mode renderer invariants."""

from types import SimpleNamespace

import pygame
import pytest
import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.env.env import TensorEnv
from boost_and_broadside.env.frontline import FRONTLINE_WORLD_SIZE
from boost_and_broadside.modes.interactive import PLAY_ENV_CONFIG
from boost_and_broadside.train.rl.features import build_standard_coordinator
from boost_and_broadside.ui.renderer import (
    _GHOST_DPHI_ATT,
    _GHOST_DPHI_X,
    _GHOST_DPHI_Y,
    Camera,
    GameRenderer,
    RenderConfig,
    field_border_pattern,
    field_color,
    wrapped_field_centers,
)


def test_camera_roundtrips_through_the_nearest_toroidal_image():
    camera = Camera((100.0, 80.0), (200, 160), center=95.0 + 40.0j)

    screen = camera.world_to_screen(5.0 + 40.0j)

    assert screen == pytest.approx((120.0, 80.0))
    assert camera.screen_to_world(screen) == pytest.approx(5.0 + 40.0j)


def test_camera_fit_uses_both_world_dimensions_and_reset_recenters():
    camera = Camera((200.0, 100.0), (300, 300), center=10.0 + 20.0j, zoom=4.0)

    camera.fit()
    assert camera.center == 10.0 + 20.0j
    assert camera.scale == pytest.approx(1.5)

    camera.pan(15.0 + 10.0j)
    camera.reset()
    assert camera.center == 100.0 + 50.0j
    assert camera.zoom == camera.min_zoom


def test_camera_can_fit_the_smaller_playable_region():
    camera = Camera(FRONTLINE_WORLD_SIZE, (900, 900))

    camera.fit_region(100.0 + 200.0j, 2600.0)

    assert camera.center == 100.0 + 200.0j
    assert camera.zoom > camera.min_zoom
    assert 2 * 2600.0 * camera.scale < 900


def test_camera_cursor_anchored_zoom_preserves_anchor_and_clamps():
    camera = Camera((100.0, 100.0), (200, 200), max_zoom=8.0)
    cursor = (150.0, 75.0)
    anchor = camera.screen_to_world(cursor)

    camera.zoom_at(4.0, cursor)

    assert camera.zoom == 4.0
    assert camera.screen_to_world(cursor) == pytest.approx(anchor)
    camera.zoom_at(100.0, cursor)
    assert camera.zoom == 8.0
    camera.zoom_at(1e-6, cursor)
    assert camera.zoom == camera.min_zoom


def test_camera_follow_updates_until_manual_pan_releases_it():
    camera = Camera((100.0, 100.0), (200, 200))
    camera.follow(95.0 + 5.0j)
    camera.update_follow(3.0 + 7.0j)
    assert camera.is_following
    assert camera.center == 3.0 + 7.0j

    camera.pan_screen((20.0, 0.0))
    released_center = camera.center
    camera.update_follow(40.0 + 40.0j)
    assert not camera.is_following
    assert camera.center == released_center


def test_camera_visible_images_preserve_field_wrap_copies():
    camera = Camera((100.0, 100.0), (100, 100))
    images = camera.visible_images(2.0 + 2.0j, extent=10.0)
    assert set(images) == {2.0 + 2.0j, 102.0 + 2.0j, 2.0 + 102.0j, 102.0 + 102.0j}


def test_renderer_events_zoom_pan_release_follow_and_reset(monkeypatch):
    monkeypatch.setenv("HEADLESS", "1")
    renderer = GameRenderer(ShipConfig(), RenderConfig(window_size=200, show_ui=False))
    try:
        monkeypatch.setattr(pygame.mouse, "get_pos", lambda: (150, 100))
        renderer.follow_position(10.0 + 20.0j)
        renderer._handle_event(pygame.event.Event(pygame.MOUSEWHEEL, y=1))
        assert renderer.camera.zoom == pytest.approx(1.2)
        assert not renderer.camera.is_following

        center_before_drag = renderer.camera.center
        renderer._handle_event(pygame.event.Event(pygame.MOUSEBUTTONDOWN, button=3, pos=(100, 100)))
        renderer._handle_event(pygame.event.Event(pygame.MOUSEMOTION, pos=(120, 100), rel=(20, 0)))
        renderer._handle_event(pygame.event.Event(pygame.MOUSEBUTTONUP, button=3, pos=(120, 100)))
        assert renderer.camera.center != center_before_drag
        assert not renderer.camera_dragging

        renderer._handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_f))
        assert renderer.camera.zoom == renderer.camera.min_zoom
        renderer._handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_r))
        assert renderer.camera.center == complex(renderer._world_w / 2.0, renderer._world_h / 2.0)
    finally:
        renderer.close()


def test_ship_geometry_and_bars_stay_screen_pixels_at_large_world_scale(monkeypatch):
    monkeypatch.setenv("HEADLESS", "1")
    renderer = GameRenderer(
        ShipConfig(world_size=(16384.0, 16384.0)),
        RenderConfig(window_size=256, show_ui=False, ship_size=10),
    )
    polygons = []
    rectangles = []

    def capture_polygon(surface, color, points, width=0):
        del surface, color, width
        polygons.append(points)

    def capture_rect(surface, color, rect, width=0):
        del surface, color, width
        rectangles.append(rect)

    monkeypatch.setattr(pygame.draw, "polygon", capture_polygon)
    monkeypatch.setattr(pygame.draw, "rect", capture_rect)
    state = SimpleNamespace(
        ship_pos=torch.tensor([[8192.0 + 8192.0j]], dtype=torch.complex64),
        ship_attitude=torch.tensor([[1.0 + 0.0j]], dtype=torch.complex64),
        ship_health=torch.tensor([[100.0]]),
        ship_power=torch.tensor([[100.0]]),
        ship_alive=torch.tensor([[True]]),
        ship_team_id=torch.tensor([[0]], dtype=torch.int32),
    )
    try:
        renderer._draw_ships(state, renderer._screen)
        assert polygons == [[(138, 128), (122, 134), (122, 122)]]
        assert rectangles[0] == (118, 112, 20, 4)
    finally:
        renderer.close()


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
    background = (10, 10, 20)
    color = (40, 225, 255)
    for pattern, width in (("dotted", 1), ("dashed", 2), ("solid", 3)):
        surface = pygame.Surface((80, 80))
        surface.fill(background)
        GameRenderer._draw_field_outline(surface, (40, 40), 25, color, pattern, width)
        assert surface.get_at((40, 40))[:3] == background


def test_play_resource_button_toggles_unlimited_health_and_power(monkeypatch):
    monkeypatch.setenv("HEADLESS", "1")
    renderer = GameRenderer(
        ShipConfig(),
        RenderConfig(show_unlimited_button=True),
    )
    try:
        assert not renderer.unlimited_resources
        renderer._handle_left_click(renderer._unlimited_rect.center)
        assert renderer.unlimited_resources
        renderer._handle_left_click(renderer._unlimited_rect.center)
        assert not renderer.unlimited_resources
    finally:
        renderer.close()


def test_headless_frontline_frame_draws_boundary_zones_hud_and_selection(monkeypatch):
    monkeypatch.setenv("HEADLESS", "1")
    ship_config = ShipConfig(world_size=FRONTLINE_WORLD_SIZE)
    env = TensorEnv(1, ship_config, PLAY_ENV_CONFIG, "cpu")
    env.reset(seed=11)
    renderer = GameRenderer(ship_config, RenderConfig(window_size=320))
    renderer.set_selectable_ships((0, 2))
    try:
        background = renderer._render_config.background_color

        surface = renderer.draw_frame(env.state)

        pixels = pygame.surfarray.array3d(surface)
        non_background = (pixels != background).any(axis=2)
        assert non_background.sum() > 500
        assert renderer.selected_ship == 0
        assert renderer.camera.zoom > renderer.camera.min_zoom
        renderer._handle_event(pygame.event.Event(pygame.KEYDOWN, key=pygame.K_TAB))
        assert renderer.selected_ship == 2
    finally:
        renderer.close()
