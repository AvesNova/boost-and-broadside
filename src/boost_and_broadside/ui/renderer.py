"""Pygame renderer for a single-environment game state.

Reads env index 0 from TensorState and draws ships, bullets, and health
bars through a toroidal camera at a fixed frame rate. All tensor reads call
.cpu() after slicing — acceptable overhead at 60fps on a single interactive
environment.
"""

import math
import os
from dataclasses import dataclass
from enum import StrEnum

import pygame
import torch

from boost_and_broadside.config import InterfaceDamageLevel, ShipConfig, ZoneRole
from boost_and_broadside.env.perception import TeamVisibility
from boost_and_broadside.env.state import TensorState

# Prediction-vector channel indices used to decode ghost trajectories: the
# position-x, position-y, and attitude phase-delta channels of the standard
# FeatureCoordinator prediction layout (the order of get_feature_names()). Named
# rather than inlined so the dependency on that layout is explicit; a regression
# test (tests/ui/test_renderer.py) pins them against the live coordinator so a
# reordering of the prediction vector can't silently desync these ghosts.
_GHOST_DPHI_X = 0
_GHOST_DPHI_Y = 1
_GHOST_DPHI_ATT = 4


class VisionMode(StrEnum):
    FULL = "FULL"
    TEAM_0 = "TEAM_0"
    TEAM_1 = "TEAM_1"


def field_color(index_level: int) -> tuple[int, int, int]:
    """Map log-index level to a magnitude-aware fast/slow color."""

    colors = {
        -2: (40, 225, 255),  # very low: bright saturated cyan
        -1: (65, 155, 195),  # low: dimmer blue-cyan
        1: (150, 95, 195),  # high: dim violet
        2: (220, 105, 255),  # very high: bright saturated violet
    }
    try:
        return colors[int(index_level)]
    except KeyError as error:
        raise ValueError(f"invalid non-ambient field index level {index_level}") from error


def field_border_pattern(damage_level: int) -> tuple[str, int]:
    """Return orthogonal border pattern and line width for interface damage."""

    patterns = {
        int(InterfaceDamageLevel.NONE): ("dotted", 1),
        int(InterfaceDamageLevel.STANDARD): ("dashed", 2),
        int(InterfaceDamageLevel.SEVERE): ("solid", 3),
    }
    try:
        return patterns[int(damage_level)]
    except KeyError as error:
        raise ValueError(f"invalid field damage level {damage_level}") from error


def wrapped_field_centers(
    center: complex,
    outer_radius: float,
    world_size: tuple[float, float],
) -> list[complex]:
    """Return visible toroidal copies for a circle near world boundaries."""

    world_w, world_h = world_size
    result = []
    for offset_x in (-world_w, 0.0, world_w):
        for offset_y in (-world_h, 0.0, world_h):
            candidate = center + complex(offset_x, offset_y)
            if (
                -outer_radius <= candidate.real <= world_w + outer_radius
                and -outer_radius <= candidate.imag <= world_h + outer_radius
            ):
                result.append(candidate)
    return result


@dataclass
class Camera:
    """Pure toroidal world/view transform.

    ``zoom == min_zoom`` fits the complete world inside the viewport. World
    points are projected from their nearest toroidal image relative to
    ``center``; the inverse always returns the canonical point in the physical
    world. The class deliberately knows nothing about pygame so transform and
    interaction semantics can be tested without a display.
    """

    world_size: tuple[float, float]
    viewport_size: tuple[int, int]
    center: complex | None = None
    zoom: float = 1.0
    min_zoom: float = 1.0
    max_zoom: float = 64.0

    def __post_init__(self) -> None:
        world_w, world_h = self.world_size
        view_w, view_h = self.viewport_size
        if world_w <= 0.0 or world_h <= 0.0:
            raise ValueError("camera world dimensions must be positive")
        if view_w <= 0 or view_h <= 0:
            raise ValueError("camera viewport dimensions must be positive")
        if not 0.0 < self.min_zoom <= self.max_zoom:
            raise ValueError("camera zoom bounds must satisfy 0 < min_zoom <= max_zoom")
        self.zoom = max(self.min_zoom, min(self.max_zoom, self.zoom))
        if self.center is None:
            self.center = complex(world_w / 2.0, world_h / 2.0)
        self.center = self._wrap(self.center)
        self._following = False
        self._follow_position: complex | None = None

    @property
    def scale(self) -> float:
        """Screen pixels per world unit at the current zoom."""
        world_w, world_h = self.world_size
        view_w, view_h = self.viewport_size
        return min(view_w / world_w, view_h / world_h) * self.zoom

    @property
    def is_following(self) -> bool:
        return self._following

    def _wrap(self, position: complex) -> complex:
        world_w, world_h = self.world_size
        return complex(position.real % world_w, position.imag % world_h)

    def shortest_displacement(self, position: complex) -> complex:
        """Displacement from camera center to the nearest image of ``position``."""
        assert self.center is not None
        world_w, world_h = self.world_size
        dx = (position.real - self.center.real + world_w / 2.0) % world_w - world_w / 2.0
        dy = (position.imag - self.center.imag + world_h / 2.0) % world_h - world_h / 2.0
        return complex(dx, dy)

    def nearest_image(self, position: complex) -> complex:
        """Unwrapped image of ``position`` nearest to the camera center."""
        assert self.center is not None
        return self.center + self.shortest_displacement(position)

    def world_to_screen(self, position: complex) -> tuple[float, float]:
        """Project a world point using its shortest toroidal displacement."""
        delta = self.shortest_displacement(position)
        view_w, view_h = self.viewport_size
        return (view_w / 2.0 + delta.real * self.scale, view_h / 2.0 + delta.imag * self.scale)

    def unwrapped_world_to_screen(self, position: complex) -> tuple[float, float]:
        """Project a chosen unwrapped image, used for repeated edge geometry."""
        assert self.center is not None
        delta = position - self.center
        view_w, view_h = self.viewport_size
        return (view_w / 2.0 + delta.real * self.scale, view_h / 2.0 + delta.imag * self.scale)

    def screen_to_world(self, position: tuple[float, float]) -> complex:
        """Invert a screen point to its canonical toroidal world coordinate."""
        assert self.center is not None
        view_w, view_h = self.viewport_size
        world = self.center + complex(
            (position[0] - view_w / 2.0) / self.scale,
            (position[1] - view_h / 2.0) / self.scale,
        )
        return self._wrap(world)

    def visible_images(self, position: complex, extent: float = 0.0) -> list[complex]:
        """Return unwrapped toroidal images whose extent intersects the viewport."""
        assert self.center is not None
        world_w, world_h = self.world_size
        view_w, view_h = self.viewport_size
        half_w = view_w / (2.0 * self.scale)
        half_h = view_h / (2.0 * self.scale)
        nearest = self.nearest_image(position)
        result = []
        for offset_x in (-world_w, 0.0, world_w):
            for offset_y in (-world_h, 0.0, world_h):
                candidate = nearest + complex(offset_x, offset_y)
                if (
                    abs(candidate.real - self.center.real) <= half_w + extent
                    and abs(candidate.imag - self.center.imag) <= half_h + extent
                ):
                    result.append(candidate)
        return result

    def fit(self) -> None:
        """Fit the complete world while retaining the current toroidal center."""
        self.release_follow()
        self.zoom = self.min_zoom

    def fit_region(self, center: complex, radius: float, padding: float = 1.15) -> None:
        """Fit a circular playable region inside the viewport."""

        if radius <= 0.0 or padding < 1.0:
            raise ValueError("camera fit radius must be positive and padding at least one")
        world_w, world_h = self.world_size
        view_w, view_h = self.viewport_size
        base_scale = min(view_w / world_w, view_h / world_h)
        desired_scale = min(view_w, view_h) / (2.0 * radius * padding)
        self.release_follow()
        self.center = self._wrap(center)
        self.zoom = max(self.min_zoom, min(self.max_zoom, desired_scale / base_scale))

    def reset(self) -> None:
        """Return to the centered full-world view."""
        world_w, world_h = self.world_size
        self.release_follow()
        self.center = complex(world_w / 2.0, world_h / 2.0)
        self.zoom = self.min_zoom

    def pan(self, delta: complex) -> None:
        """Move the camera by a world-space displacement and leave follow mode."""
        assert self.center is not None
        self.release_follow()
        self.center = self._wrap(self.center + delta)

    def pan_screen(self, delta: tuple[float, float]) -> None:
        """Drag the world by a screen-space displacement."""
        self.pan(complex(-delta[0] / self.scale, -delta[1] / self.scale))

    def zoom_at(self, factor: float, cursor: tuple[float, float]) -> None:
        """Zoom around ``cursor``, retaining the world point beneath it."""
        if factor <= 0.0:
            raise ValueError("camera zoom factor must be positive")
        assert self.center is not None
        anchor = self.screen_to_world(cursor)
        new_zoom = max(self.min_zoom, min(self.max_zoom, self.zoom * factor))
        if new_zoom == self.zoom:
            return
        self.release_follow()
        self.zoom = new_zoom
        view_w, view_h = self.viewport_size
        offset = complex(
            (cursor[0] - view_w / 2.0) / self.scale,
            (cursor[1] - view_h / 2.0) / self.scale,
        )
        self.center = self._wrap(anchor - offset)

    def follow(self, position: complex) -> None:
        """Enter follow mode at ``position``."""
        self._following = True
        self._follow_position = self._wrap(position)
        self.center = self._follow_position

    def update_follow(self, position: complex | None = None) -> None:
        """Move an active follow target; no-op after manual release."""
        if not self._following:
            return
        if position is not None:
            self._follow_position = self._wrap(position)
        if self._follow_position is not None:
            self.center = self._follow_position

    def release_follow(self) -> None:
        """Return cleanly to free-camera mode without moving the view."""
        self._following = False
        self._follow_position = None


@dataclass(frozen=True)
class RenderConfig:
    """Display settings for the pygame renderer.

    All fields have defaults — these are operational settings, not hyperparameters.
    """

    window_size: int = 900
    fps: int = 60
    show_ui: bool = True  # pause button + FPS slider; off for clean video capture
    show_unlimited_button: bool = False  # play-only health/power toggle
    vision_mode: VisionMode = VisionMode.FULL
    team_colors: tuple[tuple[int, int, int], tuple[int, int, int]] = (
        (100, 180, 255),  # team 0: blue
        (255, 120, 80),  # team 1: red
    )
    bullet_color: tuple[int, int, int] = (255, 255, 100)
    background_color: tuple[int, int, int] = (10, 10, 20)
    fog_color: tuple[int, int, int] = (105, 105, 112)
    fog_alpha: int = 55
    fog_mask_scale: float = 0.25
    fog_update_interval: int = 8
    ship_size: int = 10  # pixels from center to tip
    health_bar_height: int = 4
    power_bar_height: int = 4


class GameRenderer:
    """Pygame renderer for a single TensorState environment (env index 0).

    Args:
        ship_config:   Physics constants (world_size for coordinate mapping).
        render_config: Display settings.
    """

    def __init__(self, ship_config: ShipConfig, render_config: RenderConfig) -> None:
        self._ship_config = ship_config
        self._render_config = render_config
        self._world_w, self._world_h = ship_config.world_size

        if os.environ.get("HEADLESS"):
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        s = render_config.window_size
        self._screen = pygame.display.set_mode((s, s))
        self.camera = Camera(ship_config.world_size, (s, s))
        pygame.display.set_caption("Boost and Broadside")
        self._clock = pygame.time.Clock()

        # UI state
        self.paused = False
        self.unlimited_resources = False
        self.target_fps = render_config.fps
        self.vision_mode = VisionMode(render_config.vision_mode)
        self.slider_dragging = False
        self.camera_dragging = False
        self._camera_drag_button: int | None = None
        self.selected_ship: int | None = None
        self._selectable_ships: tuple[int, ...] = ()
        self._selection_initialized = False
        self._selected_position: complex | None = None
        self._frontline_fit: tuple[complex, float] | None = None
        self._did_initial_frontline_fit = False

        W = s
        H = s
        self._pause_rect = pygame.Rect(W - 200, H - 40, 60, 30)
        self._slider_track_rect = pygame.Rect(W - 120, H - 30, 100, 10)
        self._unlimited_rect = pygame.Rect(W - 220, 10, 200, 30)
        fog_size = max(1, round(s * render_config.fog_mask_scale))
        self._fog_team_mask = pygame.Surface((fog_size, fog_size))
        self._fog_observer_mask = pygame.Surface((fog_size, fog_size))
        self._fog_overlay = pygame.Surface((fog_size, fog_size), pygame.SRCALPHA)
        self._fog_overlay_scaled = pygame.Surface((s, s), pygame.SRCALPHA)
        self._fog_last_step = -render_config.fog_update_interval
        self._fog_last_view: tuple[VisionMode, complex | None, float, float] | None = None

    def render(
        self,
        state: TensorState,
        pred_nexts: list[torch.Tensor] | torch.Tensor | None = None,
        visibility: TeamVisibility | None = None,
    ) -> bool:
        """Draw one frame from env 0 of state.

        Args:
            state: Live TensorState — only env index 0 is read.
            pred_nexts: Optional list of (B, N, pred_dim) tensors, one per imagined
                step, where pred_dim is the coordinator's total prediction width.
                A single tensor is also accepted for backward compatibility.

        Returns:
            True to keep running, False if the user closed the window.
        """
        for event in pygame.event.get():
            if not self._handle_event(event):
                return False

        if isinstance(pred_nexts, torch.Tensor):
            pred_nexts = [pred_nexts]
        self._draw_frame(state, pred_nexts, visibility)
        pygame.display.flip()
        return True

    def render_with_label(
        self,
        state: TensorState | None,
        text: str,
        color: tuple[int, int, int] = (220, 220, 220),
        visibility: TeamVisibility | None = None,
    ) -> bool:
        """Draw one frame then overlay a centered text label before flipping.

        Pass state=None to show a blank background (e.g. during loading screens).

        Returns:
            True to keep running, False if the user closed the window.
        """
        for event in pygame.event.get():
            if not self._handle_event(event):
                return False

        if state is not None:
            self._draw_frame(state, visibility=visibility)
        else:
            self._screen.fill(self._render_config.background_color)
        self._blit_label(text, color)
        pygame.display.flip()
        return True

    def tick(self) -> None:
        """Cap frame rate to target_fps."""
        self._clock.tick(self.target_fps)

    def _update_slider(self, mouse_x: int) -> None:
        rel_x = mouse_x - self._slider_track_rect.x
        frac = max(0.0, min(1.0, rel_x / self._slider_track_rect.width))
        # Map frac to FPS (e.g. 1 to 120)
        self.target_fps = int(1 + frac * 119)

    @property
    def game_speed(self) -> float:
        """Simulation speed implied by the frame-coupled interactive loop."""

        return self.target_fps / self._render_config.fps

    def _adjust_game_speed(self, direction: int) -> None:
        """Step among explicit 1x/2x/4x/8x interactive simulation rates."""

        levels = (1.0, 2.0, 4.0, 8.0)
        current = self.game_speed
        if direction > 0:
            selected = next((level for level in levels if level > current + 1e-6), levels[-1])
        else:
            selected = next(
                (level for level in reversed(levels) if level < current - 1e-6),
                levels[0],
            )
        self.target_fps = round(self._render_config.fps * selected)

    def _handle_event(self, event: pygame.event.Event) -> bool:
        """Apply one renderer event; return false only for window close."""
        if event.type == pygame.QUIT:
            return False
        if event.type == pygame.MOUSEWHEEL:
            self.camera.zoom_at(1.2**event.y, pygame.mouse.get_pos())
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                self._handle_left_click(event.pos)
            elif event.button in (2, 3):
                self.camera.release_follow()
                self.camera_dragging = True
                self._camera_drag_button = event.button
            elif event.button in (4, 5):
                factor = 1.2 if event.button == 4 else 1.0 / 1.2
                self.camera.zoom_at(factor, event.pos)
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.slider_dragging = False
            if event.button == self._camera_drag_button:
                self.camera_dragging = False
                self._camera_drag_button = None
        elif event.type == pygame.MOUSEMOTION:
            if self.slider_dragging:
                self._update_slider(event.pos[0])
            elif self.camera_dragging:
                self.camera.pan_screen(event.rel)
        elif event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_r, pygame.K_HOME):
                self.camera.reset()
            elif event.key == pygame.K_f:
                if self._frontline_fit is None:
                    self.camera.fit()
                else:
                    self.camera.fit_region(*self._frontline_fit)
            elif event.key == pygame.K_TAB:
                self._cycle_selected_ship()
            elif event.key == pygame.K_v:
                modes = tuple(VisionMode)
                self.vision_mode = modes[(modes.index(self.vision_mode) + 1) % len(modes)]
            elif event.key in (pygame.K_EQUALS, pygame.K_RIGHTBRACKET):
                self._adjust_game_speed(1)
            elif event.key in (pygame.K_MINUS, pygame.K_LEFTBRACKET):
                self._adjust_game_speed(-1)
            elif event.key == pygame.K_c:
                if self.camera.is_following:
                    self.camera.release_follow()
                elif self._selected_position is not None:
                    self.camera.follow(self._selected_position)
        return True

    def _handle_left_click(self, position: tuple[int, int]) -> None:
        """Apply one UI click, including the play-only unlimited toggle."""
        if (
            self._render_config.show_ui
            and self._render_config.show_unlimited_button
            and self._unlimited_rect.collidepoint(position)
        ):
            self.unlimited_resources = not self.unlimited_resources
        elif self._pause_rect.collidepoint(position):
            self.paused = not self.paused
        elif self._slider_track_rect.inflate(10, 20).collidepoint(position):
            self.slider_dragging = True
            self._update_slider(position[0])

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _draw_frame(
        self,
        state: TensorState,
        pred_nexts: list[torch.Tensor] | None = None,
        visibility: TeamVisibility | None = None,
    ) -> None:
        surf = self._screen
        surf.fill(self._render_config.background_color)
        ship_visible, bullet_visible = self._perspective_masks(state, visibility)
        if self.selected_ship is not None and self.selected_ship < state.max_ships:
            if bool(state.ship_alive[0, self.selected_ship].item()):
                self._selected_position = complex(state.ship_pos[0, self.selected_ship].item())
                self.camera.update_follow(self._selected_position)
        if state.num_zones > 0:
            center = complex(state.map_center[0].item())
            radius = float(state.playable_boundary_radius[0].item())
            self._frontline_fit = (center, radius)
            if not self._did_initial_frontline_fit:
                self.camera.fit_region(center, radius)
                self._did_initial_frontline_fit = True
            self._draw_boundary(state, surf)
            self._draw_zones(state, surf)
        self._draw_fields(state, surf)
        self._draw_fog_overlay(state, surf, visibility)
        self._draw_bullets(state, surf, bullet_visible)
        if pred_nexts is not None:
            self._draw_ghost_ships(state, pred_nexts, surf, ship_visible)
        self._draw_ships(state, surf, ship_visible)
        if state.num_zones > 0:
            self._draw_minimap(state, surf, ship_visible)
        if self._render_config.show_ui:
            self._draw_ui(state, surf)

    def draw_frame(
        self,
        state: TensorState,
        pred_nexts: list[torch.Tensor] | None = None,
        visibility: TeamVisibility | None = None,
    ) -> pygame.Surface:
        """Supported offscreen frame API used by capture and smoke tests."""

        self._draw_frame(state, pred_nexts, visibility)
        return self._screen

    def _perspective_masks(
        self,
        state: TensorState,
        visibility: TeamVisibility | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Resolve draw masks from the renderer's selected perception contract."""

        if self.vision_mode is VisionMode.FULL:
            return state.ship_alive[0], state.bullet_active[0]
        if visibility is None:
            raise ValueError("team vision rendering requires authoritative visibility masks")
        team = 0 if self.vision_mode is VisionMode.TEAM_0 else 1
        return visibility.ship[0, team] & state.ship_alive[0], visibility.bullet[0, team]

    def _blit_label(self, text: str, color: tuple[int, int, int]) -> None:
        if not hasattr(self, "_font_large"):
            self._font_large = pygame.font.SysFont("monospace", 24, bold=True)
        surf = self._screen
        label = self._font_large.render(text, True, color)
        x = (surf.get_width() - label.get_width()) // 2
        y = surf.get_height() - label.get_height() - 16
        surf.blit(label, (x, y))

    def _draw_ui(self, state: TensorState, surf: pygame.Surface) -> None:
        if not hasattr(self, "_font"):
            self._font = pygame.font.SysFont("monospace", 16, bold=True)

        # Draw pause button
        color = (150, 150, 150) if not self.paused else (255, 100, 100)
        pygame.draw.rect(surf, color, self._pause_rect)
        label = self._font.render("Play" if self.paused else "Pause", True, (0, 0, 0))
        surf.blit(
            label,
            (
                self._pause_rect.centerx - label.get_width() // 2,
                self._pause_rect.centery - label.get_height() // 2,
            ),
        )

        # Draw FPS slider track
        pygame.draw.rect(surf, (100, 100, 100), self._slider_track_rect)

        # Draw slider handle
        frac = max(0.0, min(1.0, (self.target_fps - 1) / 119.0))
        handle_x = self._slider_track_rect.x + int(frac * self._slider_track_rect.width)
        handle_rect = pygame.Rect(handle_x - 5, self._slider_track_rect.y - 5, 10, 20)
        pygame.draw.rect(surf, (200, 200, 200), handle_rect)

        # Draw FPS text
        fps_label = self._font.render(f"GAME {self.game_speed:g}x", True, (200, 200, 200))
        surf.blit(fps_label, (self._slider_track_rect.x, self._slider_track_rect.y - 20))

        if self._render_config.show_unlimited_button:
            resource_color = (80, 220, 120) if self.unlimited_resources else (110, 110, 125)
            pygame.draw.rect(surf, resource_color, self._unlimited_rect)
            resource_label = self._font.render(
                f"Unlimited HP/PW: {'ON' if self.unlimited_resources else 'OFF'}",
                True,
                (0, 0, 0),
            )
            surf.blit(
                resource_label,
                (
                    self._unlimited_rect.centerx - resource_label.get_width() // 2,
                    self._unlimited_rect.centery - resource_label.get_height() // 2,
                ),
            )

        # Interface legend: color carries index, pattern carries damage. In
        # particular, a solid outline means severe damage—not impermeability.
        legend = self._font.render(
            "Fields: cyan fast | violet slow | · none  -- standard  — severe (traversable)",
            True,
            (175, 175, 190),
        )
        surf.blit(legend, (12, surf.get_height() - legend.get_height() - 10))

        if state.num_zones > 0:
            front = int(state.front_position[0].item())
            threshold = int(state.front_win_threshold[0].item())
            max_steps = int(state.match_max_steps[0].item())
            steps = int(state.step_count[0].item())
            remaining = max(0.0, (max_steps - steps) * self._ship_config.dt)
            lines = (
                f"TEAM 0  FRONT {front:+d}/{threshold}  TEAM 1",
                (
                    f"TIME {remaining:05.1f}s   VIEW {self.vision_mode.value}   "
                    f"SPEED {self.game_speed:g}x"
                ),
                "V view  F fit  R world  wheel zoom  drag pan  C follow  TAB select  -/+ speed",
            )
            for row, text in enumerate(lines):
                label = self._font.render(text, True, (225, 225, 235))
                surf.blit(label, (12, 10 + row * 20))

    def close(self) -> None:
        """Tear down the pygame window."""
        pygame.quit()

    def follow_position(self, position: complex) -> None:
        """Start following a world position; callers may update it each frame."""
        self.camera.follow(position)

    def update_follow_position(self, position: complex) -> None:
        """Update the followed position without re-entering released follow mode."""
        self.camera.update_follow(position)

    def release_follow(self) -> None:
        """Leave follow mode at the current view."""
        self.camera.release_follow()

    def set_selectable_ships(self, ship_indices: tuple[int, ...]) -> None:
        """Set controllable slots while preserving an explicit spectator state."""

        self._selectable_ships = ship_indices
        if not ship_indices:
            self.selected_ship = None
        elif not self._selection_initialized:
            self.selected_ship = ship_indices[0] if ship_indices else None
            self._selection_initialized = True
        elif self.selected_ship is not None and self.selected_ship not in ship_indices:
            self.selected_ship = ship_indices[0]

    def _cycle_selected_ship(self) -> None:
        if not self._selectable_ships:
            self.selected_ship = None
            return
        if self.selected_ship not in self._selectable_ships:
            self.selected_ship = self._selectable_ships[0]
            return
        index = self._selectable_ships.index(self.selected_ship)
        if index == len(self._selectable_ships) - 1:
            self.selected_ship = None
            self._selected_position = None
            self.camera.release_follow()
        else:
            self.selected_ship = self._selectable_ships[index + 1]

    # ------------------------------------------------------------------
    # Private drawing helpers
    # ------------------------------------------------------------------

    def _world_to_screen(self, c: complex) -> tuple[int, int]:
        """Convert a world-space complex position to screen pixel coords."""
        x, y = self.camera.world_to_screen(c)
        return round(x), round(y)

    def _unwrapped_world_to_screen(self, c: complex) -> tuple[int, int]:
        x, y = self.camera.unwrapped_world_to_screen(c)
        return round(x), round(y)

    def _draw_toroidal_line(
        self,
        surf: pygame.Surface,
        from_pos: complex,
        to_pos: complex,
        color: tuple[int, int, int],
    ) -> None:
        """Draw a line between two world positions, splitting at wrap boundaries."""
        world_w, world_h = self._world_w, self._world_h
        dx = to_pos.real - from_pos.real
        dy = to_pos.imag - from_pos.imag
        if dx > world_w / 2:
            dx -= world_w
        elif dx < -world_w / 2:
            dx += world_w
        if dy > world_h / 2:
            dy -= world_h
        elif dy < -world_h / 2:
            dy += world_h
        start = self.camera.nearest_image(from_pos)
        target = start + complex(dx, dy)
        width, height = surf.get_size()
        for offset_x in (-world_w, 0.0, world_w):
            for offset_y in (-world_h, 0.0, world_h):
                offset = complex(offset_x, offset_y)
                screen_start = self._unwrapped_world_to_screen(start + offset)
                screen_target = self._unwrapped_world_to_screen(target + offset)
                if (
                    max(screen_start[0], screen_target[0]) < 0
                    or min(screen_start[0], screen_target[0]) >= width
                    or max(screen_start[1], screen_target[1]) < 0
                    or min(screen_start[1], screen_target[1]) >= height
                ):
                    continue
                pygame.draw.line(surf, color, screen_start, screen_target, 1)

    def _draw_ghost_ships(
        self,
        state: TensorState,
        pred_nexts: list[torch.Tensor],
        surf: pygame.Surface,
        visible: torch.Tensor | None = None,
    ) -> None:
        """Draw autoregressive predicted positions as fading hollow triangles.

        pred_nexts: list of (B, N, pred_dim) tensors, one per imagined step, in the
          coordinator's prediction layout. Channels _GHOST_DPHI_X and _GHOST_DPHI_Y
          are position phase shifts relative to the previous ghost position;
          _GHOST_DPHI_ATT is the attitude phase shift relative to the previous
          attitude. Ghost brightness fades linearly from 1.0 (step 0) to 0.5 (last).
        """
        import math

        cfg = self._render_config

        visible = state.ship_alive[0] if visible is None else visible
        alive = (state.ship_alive[0] & visible).cpu()  # (N,) bool
        team_id = state.ship_team_id[0].cpu()  # (N,) int32
        real_pos = state.ship_pos[0].cpu()  # (N,) complex64
        real_att = state.ship_attitude[0].cpu()  # (N,) complex64

        sz = cfg.ship_size
        world_w, world_h = self._world_w, self._world_h
        _2pi = 2.0 * math.pi
        n_steps = len(pred_nexts)

        pn_cpu = [pn[0].cpu() for pn in pred_nexts]  # list of (N, pred_dim)

        for n in range(pn_cpu[0].shape[0]):
            if not alive[n].item():
                continue

            color = cfg.team_colors[int(team_id[n].item()) % 2]
            dim_color = (max(0, color[0] - 50), max(0, color[1] - 50), max(0, color[2] - 50))

            prev_p = complex(real_pos[n].item())
            prev_att_angle = math.atan2(real_att[n].imag.item(), real_att[n].real.item())

            for k, pn in enumerate(pn_cpu):
                # Skip ships where this agent produced no prediction (zeros = null/scripted).
                if pn[n].abs().sum().item() == 0:
                    break

                alpha = 1.0 - 0.5 * k / max(n_steps - 1, 1)
                fade = tuple(int(c * alpha) for c in dim_color)

                # Decode ghost position: phase shift applied to prev ghost position.
                phi_x = _2pi * prev_p.real / world_w
                phi_y = _2pi * prev_p.imag / world_h
                ghost_x = ((phi_x + pn[n, _GHOST_DPHI_X].item()) % _2pi) / _2pi * world_w
                ghost_y = ((phi_y + pn[n, _GHOST_DPHI_Y].item()) % _2pi) / _2pi * world_h
                ghost_p = complex(ghost_x, ghost_y)

                # Decode ghost attitude: phase shift applied to prev attitude.
                att_angle = prev_att_angle + pn[n, _GHOST_DPHI_ATT].item()
                ghost_a = complex(math.cos(att_angle), math.sin(att_angle))

                center = complex(*self._world_to_screen(ghost_p))
                tip = center + ghost_a * sz
                left = center + ghost_a * (-sz * 0.6) + ghost_a * 1j * (sz * 0.6)
                right = center + ghost_a * (-sz * 0.6) - ghost_a * 1j * (sz * 0.6)
                verts = [(round(v.real), round(v.imag)) for v in (tip, left, right)]
                pygame.draw.polygon(surf, fade, verts, width=1)

                self._draw_toroidal_line(surf, prev_p, ghost_p, fade)

                prev_p = ghost_p
                prev_att_angle = att_angle

    def _draw_ships(
        self,
        state: TensorState,
        surf: pygame.Surface,
        visible: torch.Tensor | None = None,
    ) -> None:
        """Draw all alive ships in env 0 as colored triangles with health bars."""
        cfg = self._render_config
        sc = self._ship_config

        pos = state.ship_pos[0].cpu()  # (N,) complex64
        att = state.ship_attitude[0].cpu()  # (N,) complex64
        health = state.ship_health[0].cpu()  # (N,) float32
        power = state.ship_power[0].cpu()  # (N,) float32
        visible = state.ship_alive[0] if visible is None else visible
        alive = (state.ship_alive[0] & visible).cpu()  # (N,) bool
        team_id = state.ship_team_id[0].cpu()  # (N,) int32

        sz = cfg.ship_size
        for n in range(pos.shape[0]):
            if not alive[n].item():
                continue

            p = complex(pos[n].item())
            a = complex(att[n].item())
            color = cfg.team_colors[int(team_id[n].item()) % 2]

            center = complex(*self._world_to_screen(p))
            if n == self.selected_ship:
                pygame.draw.circle(
                    surf,
                    (255, 255, 255),
                    (round(center.real), round(center.imag)),
                    sz + 6,
                    width=2,
                )
            tip = center + a * sz
            left = center + a * (-sz * 0.6) + a * 1j * (sz * 0.6)
            right = center + a * (-sz * 0.6) - a * 1j * (sz * 0.6)
            verts = [(round(v.real), round(v.imag)) for v in (tip, left, right)]
            pygame.draw.polygon(surf, color, verts)

            # Health bar above ship
            hp_frac = float(health[n].item()) / sc.max_health
            bar_w = sz * 2
            bar_x = round(center.real) - sz
            bar_y = round(center.imag) - sz - cfg.health_bar_height - 2
            pygame.draw.rect(surf, (60, 0, 0), (bar_x, bar_y, bar_w, cfg.health_bar_height))
            pygame.draw.rect(
                surf,
                (0, 200, 0),
                (bar_x, bar_y, int(bar_w * hp_frac), cfg.health_bar_height),
            )

            # Power bar above health bar
            pw_frac = float(power[n].item()) / sc.max_power
            pw_bar_y = bar_y - cfg.power_bar_height - 2
            pygame.draw.rect(surf, (0, 0, 60), (bar_x, pw_bar_y, bar_w, cfg.power_bar_height))
            pygame.draw.rect(
                surf,
                (30, 100, 255),
                (bar_x, pw_bar_y, int(bar_w * pw_frac), cfg.power_bar_height),
            )

    def _draw_boundary(self, state: TensorState, surf: pygame.Surface) -> None:
        center = complex(state.map_center[0].item())
        radius = float(state.playable_boundary_radius[0].item())
        radius_px = max(1, round(radius * self.camera.scale))
        for image in self.camera.visible_images(center, radius):
            screen = self._unwrapped_world_to_screen(image)
            pygame.draw.circle(surf, (170, 70, 70), screen, radius_px, width=3)
            pygame.draw.circle(surf, (90, 45, 55), screen, max(1, radius_px - 6), width=1)

    def _draw_zones(self, state: TensorState, surf: pygame.Surface) -> None:
        """Draw role, affiliation, hazard meaning, and capture progress."""

        if not hasattr(self, "_font_small"):
            self._font_small = pygame.font.SysFont("monospace", 13, bold=True)
        positions = state.zone_pos[0].cpu()
        radii = state.zone_radius[0].cpu()
        roles = state.zone_roles[0].cpu()
        progress = state.zone_capture_progress[0].cpu()
        role_style = {
            int(ZoneRole.TEAM0_SPAWN): ((100, 180, 255), "S0 HEAL"),
            int(ZoneRole.TEAM0_DEFENSE): ((100, 180, 255), "D0 DMG"),
            int(ZoneRole.NEUTRAL): ((180, 180, 180), "NEUTRAL"),
            int(ZoneRole.TEAM1_DEFENSE): ((255, 120, 80), "D1 DMG"),
            int(ZoneRole.TEAM1_SPAWN): ((255, 120, 80), "S1 HEAL"),
        }
        for index in range(positions.shape[0]):
            position = complex(positions[index].item())
            radius = float(radii[index].item())
            role = int(roles[index].item())
            color, text = role_style[role]
            radius_px = max(2, round(radius * self.camera.scale))
            for image in self.camera.visible_images(position, radius):
                center = self._unwrapped_world_to_screen(image)
                pygame.draw.circle(surf, color, center, radius_px, width=3)
                if role in (int(ZoneRole.TEAM0_DEFENSE), int(ZoneRole.TEAM1_DEFENSE)):
                    attacker = (
                        self._render_config.team_colors[1]
                        if role == int(ZoneRole.TEAM0_DEFENSE)
                        else self._render_config.team_colors[0]
                    )
                    rect = pygame.Rect(
                        center[0] - radius_px - 5,
                        center[1] - radius_px - 5,
                        2 * (radius_px + 5),
                        2 * (radius_px + 5),
                    )
                    pygame.draw.arc(
                        surf,
                        attacker,
                        rect,
                        -math.pi / 2,
                        -math.pi / 2 + 2 * math.pi * float(progress[index].item()),
                        width=5,
                    )
                label = self._font_small.render(text, True, color)
                surf.blit(
                    label,
                    (center[0] - label.get_width() // 2, center[1] - label.get_height() // 2),
                )

    def _draw_fields(self, state: TensorState, surf: pygame.Surface) -> None:
        """Draw each overlapping field's transition band and damage outline."""
        if state.num_fields == 0:
            return
        positions = state.field_pos[0].cpu()
        radii = state.field_radius[0].cpu()
        widths = state.field_transition_width[0].cpu()
        index_levels = state.field_index_level[0].cpu()
        damage_levels = state.field_damage_level[0].cpu()

        # Large contours first keeps small/coincident field outlines legible.
        order = sorted(range(positions.shape[0]), key=lambda i: radii[i].item(), reverse=True)
        for field_idx in order:
            center = complex(positions[field_idx].item())
            radius_world = float(radii[field_idx].item())
            outer = radius_world + 0.5 * float(widths[field_idx].item())
            radius_px = max(1, int(round(radius_world * self.camera.scale)))
            width_px = max(1, int(round(float(widths[field_idx].item()) * self.camera.scale)))
            color = field_color(int(index_levels[field_idx].item()))
            pattern, line_width = field_border_pattern(int(damage_levels[field_idx].item()))
            for wrapped_center in self.camera.visible_images(center, outer):
                screen_center = self._unwrapped_world_to_screen(wrapped_center)
                self._draw_field_band(surf, screen_center, radius_px, width_px, color)
                self._draw_field_outline(
                    surf,
                    screen_center,
                    radius_px,
                    color,
                    pattern,
                    line_width,
                )

    def _draw_fog_overlay(
        self,
        state: TensorState,
        surf: pygame.Surface,
        visibility: TeamVisibility | None,
    ) -> None:
        """Gray unseen world space using allied sight circles and field shadows.

        Static geometry is drawn first and therefore desaturates with unseen
        empty space. Visible ships, bullets, and prediction ghosts are drawn
        afterwards at full contrast. The mask is a renderer representation of
        the same range/core-LOS rule used by policy perception; firing reveals
        the ship marker but does not illuminate the surrounding terrain.
        """

        if self.vision_mode is VisionMode.FULL:
            return
        if visibility is None:
            raise ValueError("team vision rendering requires authoritative visibility masks")
        if visibility.vision_range is None:
            return

        team = 0 if self.vision_mode is VisionMode.TEAM_0 else 1
        vision_range = float(visibility.vision_range)
        step = int(state.step_count[0].item())
        view = (self.vision_mode, self.camera.center, self.camera.zoom, vision_range)
        interval = max(1, self._render_config.fog_update_interval)
        if (
            self._fog_last_view == view
            and step >= self._fog_last_step
            and step - self._fog_last_step < interval
        ):
            surf.blit(self._fog_overlay_scaled, (0, 0))
            return

        mask = self._fog_team_mask
        observer_mask = self._fog_observer_mask
        mask.fill((0, 0, 0))
        mask_scale_x = mask.get_width() / surf.get_width()
        mask_scale_y = mask.get_height() / surf.get_height()

        def mask_point(screen: tuple[int, int]) -> tuple[int, int]:
            return round(screen[0] * mask_scale_x), round(screen[1] * mask_scale_y)

        positions = state.ship_pos[0].cpu()
        teams = state.ship_team_id[0].cpu()
        alive = state.ship_alive[0].cpu()
        field_positions = state.field_pos[0].cpu()
        field_radii = state.field_radius[0].cpu()
        field_widths = state.field_transition_width[0].cpu()
        vision_px = max(1, round(vision_range * self.camera.scale * mask_scale_x))

        for index in range(state.max_ships):
            if not bool(alive[index].item()) or int(teams[index].item()) != team:
                continue
            observer = complex(positions[index].item())
            for observer_image in self.camera.visible_images(observer, vision_range):
                observer_mask.fill((0, 0, 0))
                observer_screen = mask_point(self._unwrapped_world_to_screen(observer_image))
                pygame.draw.circle(
                    observer_mask,
                    (255, 255, 255),
                    observer_screen,
                    vision_px,
                )
                for field_pos, field_radius, field_width in zip(
                    field_positions,
                    field_radii,
                    field_widths,
                    strict=True,
                ):
                    core_radius = max(
                        0.0,
                        float(field_radius.item()) - 0.5 * float(field_width.item()),
                    )
                    if core_radius <= 0.0:
                        continue
                    field = complex(field_pos.item())
                    dx = (field.real - observer.real + self._world_w / 2.0) % self._world_w
                    dy = (field.imag - observer.imag + self._world_h / 2.0) % self._world_h
                    delta = complex(dx - self._world_w / 2.0, dy - self._world_h / 2.0)
                    distance = abs(delta)
                    if distance <= core_radius or distance - core_radius >= vision_range:
                        continue

                    center_angle = math.atan2(delta.imag, delta.real)
                    half_angle = math.asin(min(1.0, core_radius / distance))
                    tangent_distance = math.sqrt(
                        max(0.0, distance * distance - core_radius * core_radius)
                    )
                    ray_angles = (
                        center_angle - half_angle,
                        center_angle + half_angle,
                    )
                    rays = [complex(math.cos(angle), math.sin(angle)) for angle in ray_angles]
                    tangent = [observer_image + ray * tangent_distance for ray in rays]
                    # Extend beyond the sight circle so the polygon covers the
                    # complete curved cap at maximum range; the circle already
                    # clips all irrelevant pixels outside the sensor footprint.
                    far_distance = vision_range * 4.0 + distance
                    far = [observer_image + ray * far_distance for ray in rays]
                    polygon = [
                        mask_point(self._unwrapped_world_to_screen(tangent[0])),
                        mask_point(self._unwrapped_world_to_screen(far[0])),
                        mask_point(self._unwrapped_world_to_screen(far[1])),
                        mask_point(self._unwrapped_world_to_screen(tangent[1])),
                    ]
                    pygame.draw.polygon(observer_mask, (0, 0, 0), polygon)
                mask.blit(observer_mask, (0, 0), special_flags=pygame.BLEND_RGB_MAX)

        self._fog_overlay.fill((*self._render_config.fog_color, self._render_config.fog_alpha))
        alpha = pygame.surfarray.pixels_alpha(self._fog_overlay)
        visible_pixels = pygame.surfarray.pixels3d(mask)
        alpha[visible_pixels[:, :, 0] > 0] = 0
        del visible_pixels
        del alpha
        pygame.transform.smoothscale(
            self._fog_overlay,
            surf.get_size(),
            self._fog_overlay_scaled,
        )
        self._fog_last_step = step
        self._fog_last_view = view
        surf.blit(self._fog_overlay_scaled, (0, 0))

    @staticmethod
    def _draw_field_band(
        surf: pygame.Surface,
        center: tuple[int, int],
        radius: int,
        transition_width: int,
        color: tuple[int, int, int],
    ) -> None:
        """Tint one interface annulus without filling the field core.

        Independent alpha blits make partial and coincident overlaps visible as
        stronger/mixed bands while every nominal contour remains separately
        outlined by its material and damage pattern.
        """

        half_width = max(1, transition_width // 2)
        outer = radius + half_width
        inner = max(0, radius - half_width)
        size = 2 * outer + 1
        band = pygame.Surface((size, size), pygame.SRCALPHA)
        local_center = (outer, outer)
        pygame.draw.circle(band, (*color, 34), local_center, outer)
        if inner > 0:
            pygame.draw.circle(band, (0, 0, 0, 0), local_center, inner)
        surf.blit(band, (center[0] - outer, center[1] - outer))

    @staticmethod
    def _draw_field_outline(
        surf: pygame.Surface,
        center: tuple[int, int],
        radius: int,
        color: tuple[int, int, int],
        pattern: str,
        line_width: int,
    ) -> None:
        """Draw one unfilled patterned circle copy."""

        if pattern == "solid":
            pygame.draw.circle(surf, color, center, radius, width=line_width)
            return
        if pattern == "dotted":
            circumference = max(1.0, 2.0 * math.pi * radius)
            num_dots = max(12, int(circumference / 9.0))
            dot_radius = max(1, line_width)
            for dot_idx in range(num_dots):
                angle = 2.0 * math.pi * dot_idx / num_dots
                point = (
                    round(center[0] + radius * math.cos(angle)),
                    round(center[1] + radius * math.sin(angle)),
                )
                pygame.draw.circle(surf, color, point, dot_radius)
            return
        if pattern != "dashed":
            raise ValueError(f"unknown field border pattern {pattern!r}")
        rect = pygame.Rect(center[0] - radius, center[1] - radius, 2 * radius, 2 * radius)
        num_dashes = max(12, int(2.0 * math.pi * radius / 24.0))
        step = 2.0 * math.pi / num_dashes
        for dash_idx in range(num_dashes):
            start = dash_idx * step
            pygame.draw.arc(surf, color, rect, start, start + 0.55 * step, line_width)

    def _draw_minimap(
        self,
        state: TensorState,
        surf: pygame.Surface,
        ship_visible: torch.Tensor,
    ) -> None:
        """Draw static map geometry and only dynamically visible ships."""

        size = 180
        margin = 12
        left = surf.get_width() - size - margin
        top = 52
        panel = pygame.Rect(left, top, size, size)
        pygame.draw.rect(surf, (16, 18, 30), panel)
        pygame.draw.rect(surf, (105, 110, 130), panel, width=1)

        map_center = complex(state.map_center[0].item())
        playable = float(state.playable_boundary_radius[0].item())
        scale = (size / 2.0 - 8.0) / max(playable, 1.0)
        center_px = complex(panel.centerx, panel.centery)

        def point(position: complex) -> tuple[int, int]:
            dx = (position.real - map_center.real + self._world_w / 2.0) % self._world_w
            dy = (position.imag - map_center.imag + self._world_h / 2.0) % self._world_h
            delta = complex(dx - self._world_w / 2.0, dy - self._world_h / 2.0)
            mapped = center_px + delta * scale
            return round(mapped.real), round(mapped.imag)

        pygame.draw.circle(surf, (130, 65, 75), panel.center, round(playable * scale), width=2)
        for position, radius, level in zip(
            state.field_pos[0].cpu(),
            state.field_radius[0].cpu(),
            state.field_index_level[0].cpu(),
        ):
            pygame.draw.circle(
                surf,
                field_color(int(level.item())),
                point(complex(position.item())),
                max(1, round(float(radius.item()) * scale)),
                width=1,
            )
        for position, role in zip(state.zone_pos[0].cpu(), state.zone_roles[0].cpu()):
            team = 0 if int(role.item()) <= 1 else 1 if int(role.item()) >= 3 else None
            color = (175, 175, 175) if team is None else self._render_config.team_colors[team]
            pygame.draw.circle(surf, color, point(complex(position.item())), 4, width=2)

        positions = state.ship_pos[0].cpu()
        teams = state.ship_team_id[0].cpu()
        alive_visible = (state.ship_alive[0] & ship_visible).cpu()
        for index in range(state.max_ships):
            if not bool(alive_visible[index].item()):
                continue
            p = point(complex(positions[index].item()))
            color = self._render_config.team_colors[int(teams[index].item())]
            pygame.draw.circle(surf, color, p, 3)
            if index == self.selected_ship:
                pygame.draw.circle(surf, (255, 255, 255), p, 5, width=1)

        assert self.camera.center is not None
        camera_center = point(self.camera.center)
        view_w = surf.get_width() / self.camera.scale * scale
        view_h = surf.get_height() / self.camera.scale * scale
        viewport = pygame.Rect(0, 0, max(2, round(view_w)), max(2, round(view_h)))
        viewport.center = camera_center
        pygame.draw.rect(surf, (220, 220, 230), viewport, width=1)

    def _draw_bullets(
        self,
        state: TensorState,
        surf: pygame.Surface,
        visible: torch.Tensor | None = None,
    ) -> None:
        """Draw all active bullets in env 0 as small rectangles."""
        cfg = self._render_config
        bpos = state.bullet_pos[0].cpu()  # (N, K) complex64
        visible = state.bullet_active[0] if visible is None else visible
        bact = (state.bullet_active[0] & visible).cpu()  # (N, K) bool
        team_id = state.ship_team_id[0].cpu()  # (N,) int32

        N, K = bpos.shape
        for n in range(N):
            color = cfg.team_colors[int(team_id[n].item()) % 2]
            for k in range(K):
                if not bact[n, k].item():
                    continue
                p = complex(bpos[n, k].item())
                sx, sy = self._world_to_screen(p)
                pygame.draw.rect(surf, color, (sx - 1, sy - 1, 3, 3))
