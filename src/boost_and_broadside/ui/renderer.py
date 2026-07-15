"""Pygame renderer for a single-environment game state.

Reads env index 0 from TensorState and draws ships, bullets, and health
bars at a fixed frame rate. All tensor reads call .cpu() after slicing —
acceptable overhead at 60fps on a single interactive environment.
"""

import os
from dataclasses import dataclass

import pygame
import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.env.state import TensorState


@dataclass(frozen=True)
class RenderConfig:
    """Display settings for the pygame renderer.

    All fields have defaults — these are operational settings, not hyperparameters.
    """

    window_size: int = 900
    fps: int = 60
    team_colors: tuple[tuple[int, int, int], tuple[int, int, int]] = (
        (100, 180, 255),  # team 0: blue
        (255, 120, 80),  # team 1: red
    )
    bullet_color: tuple[int, int, int] = (255, 255, 100)
    background_color: tuple[int, int, int] = (10, 10, 20)
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
        self._scale = render_config.window_size / self._world_w

        if os.environ.get("HEADLESS"):
            os.environ["SDL_VIDEODRIVER"] = "dummy"
        pygame.init()
        s = render_config.window_size
        self._screen = pygame.display.set_mode((s, s))
        pygame.display.set_caption("Boost and Broadside")
        self._clock = pygame.time.Clock()

        # UI state
        self.paused = False
        self.target_fps = render_config.fps
        self.slider_dragging = False

        W = s
        H = s
        self._pause_rect = pygame.Rect(W - 200, H - 40, 60, 30)
        self._slider_track_rect = pygame.Rect(W - 120, H - 30, 100, 10)

    def render(
        self,
        state: TensorState,
        pred_nexts: list[torch.Tensor] | torch.Tensor | None = None,
    ) -> bool:
        """Draw one frame from env 0 of state.

        Args:
            state: Live TensorState — only env index 0 is read.
            pred_nexts: Optional list of (B, N, AUX_PRED_DIM) tensors, one per imagined
                step. A single tensor is also accepted for backward compatibility.

        Returns:
            True to keep running, False if the user closed the window.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:
                    if self._pause_rect.collidepoint(event.pos):
                        self.paused = not self.paused
                    elif self._slider_track_rect.inflate(10, 20).collidepoint(event.pos):
                        self.slider_dragging = True
                        self._update_slider(event.pos[0])
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    self.slider_dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if self.slider_dragging:
                    self._update_slider(event.pos[0])

        if isinstance(pred_nexts, torch.Tensor):
            pred_nexts = [pred_nexts]
        self._draw_frame(state, pred_nexts)
        pygame.display.flip()
        return True

    def render_with_label(
        self,
        state: TensorState | None,
        text: str,
        color: tuple[int, int, int] = (220, 220, 220),
    ) -> bool:
        """Draw one frame then overlay a centered text label before flipping.

        Pass state=None to show a blank background (e.g. during loading screens).

        Returns:
            True to keep running, False if the user closed the window.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        if state is not None:
            self._draw_frame(state)
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

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _draw_frame(self, state: TensorState, pred_nexts: list[torch.Tensor] | None = None) -> None:
        surf = self._screen
        surf.fill(self._render_config.background_color)
        self._draw_obstacles(state, surf)
        self._draw_bullets(state, surf)
        if pred_nexts is not None:
            self._draw_ghost_ships(state, pred_nexts, surf)
        self._draw_ships(state, surf)
        self._draw_ui(surf)

    def _blit_label(self, text: str, color: tuple[int, int, int]) -> None:
        if not hasattr(self, "_font_large"):
            self._font_large = pygame.font.SysFont("monospace", 24, bold=True)
        surf = self._screen
        label = self._font_large.render(text, True, color)
        x = (surf.get_width() - label.get_width()) // 2
        y = surf.get_height() - label.get_height() - 16
        surf.blit(label, (x, y))

    def _draw_ui(self, surf: pygame.Surface) -> None:
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
        frac = (self.target_fps - 1) / 119.0
        handle_x = self._slider_track_rect.x + int(frac * self._slider_track_rect.width)
        handle_rect = pygame.Rect(handle_x - 5, self._slider_track_rect.y - 5, 10, 20)
        pygame.draw.rect(surf, (200, 200, 200), handle_rect)

        # Draw FPS text
        fps_label = self._font.render(f"{self.target_fps} FPS", True, (200, 200, 200))
        surf.blit(fps_label, (self._slider_track_rect.x, self._slider_track_rect.y - 20))

    def close(self) -> None:
        """Tear down the pygame window."""
        pygame.quit()

    # ------------------------------------------------------------------
    # Private drawing helpers
    # ------------------------------------------------------------------

    def _world_to_screen(self, c: complex) -> tuple[int, int]:
        """Convert a world-space complex position to screen pixel coords."""
        return (int(c.real * self._scale), int(c.imag * self._scale))

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
        wrapped = False
        if dx > world_w / 2:
            dx -= world_w
            wrapped = True
        elif dx < -world_w / 2:
            dx += world_w
            wrapped = True
        if dy > world_h / 2:
            dy -= world_h
            wrapped = True
        elif dy < -world_h / 2:
            dy += world_h
            wrapped = True
        sx0, sy0 = self._world_to_screen(from_pos)
        target = complex(from_pos.real + dx, from_pos.imag + dy)
        sx1, sy1 = self._world_to_screen(target)
        pygame.draw.line(surf, color, (sx0, sy0), (sx1, sy1), 1)
        if wrapped:
            src = complex(to_pos.real - dx, to_pos.imag - dy)
            pygame.draw.line(
                surf, color, self._world_to_screen(src), self._world_to_screen(to_pos), 1
            )

    def _draw_ghost_ships(
        self, state: TensorState, pred_nexts: list[torch.Tensor], surf: pygame.Surface
    ) -> None:
        """Draw autoregressive predicted positions as fading hollow triangles.

        pred_nexts: list of (B, N, AUX_PRED_DIM=10) tensors, one per imagined step.
          Each tensor's [0] Δφ_x and [1] Δφ_y are phase shifts relative to the
          previous ghost position; [4] Δφ_att is relative to the previous attitude.
        Ghost brightness fades linearly from 1.0 (step 0) to 0.5 (last step).
        """
        import math

        cfg = self._render_config

        alive = state.ship_alive[0].cpu()  # (N,) bool
        team_id = state.ship_team_id[0].cpu()  # (N,) int32
        real_pos = state.ship_pos[0].cpu()  # (N,) complex64
        real_att = state.ship_attitude[0].cpu()  # (N,) complex64

        sz = cfg.ship_size
        world_w, world_h = self._world_w, self._world_h
        _2pi = 2.0 * math.pi
        n_steps = len(pred_nexts)

        pn_cpu = [pn[0].cpu() for pn in pred_nexts]  # list of (N, 10)

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
                ghost_x = ((phi_x + pn[n, 0].item()) % _2pi) / _2pi * world_w
                ghost_y = ((phi_y + pn[n, 1].item()) % _2pi) / _2pi * world_h
                ghost_p = complex(ghost_x, ghost_y)

                # Decode ghost attitude: phase shift applied to prev attitude.
                att_angle = prev_att_angle + pn[n, 4].item()
                ghost_a = complex(math.cos(att_angle), math.sin(att_angle))

                tip = ghost_p + ghost_a * sz
                left = ghost_p + ghost_a * (-sz * 0.6) + ghost_a * 1j * (sz * 0.6)
                right = ghost_p + ghost_a * (-sz * 0.6) - ghost_a * 1j * (sz * 0.6)
                verts = [self._world_to_screen(v) for v in (tip, left, right)]
                pygame.draw.polygon(surf, fade, verts, width=1)

                self._draw_toroidal_line(surf, prev_p, ghost_p, fade)

                prev_p = ghost_p
                prev_att_angle = att_angle

    def _draw_ships(self, state: TensorState, surf: pygame.Surface) -> None:
        """Draw all alive ships in env 0 as colored triangles with health bars."""
        cfg = self._render_config
        sc = self._ship_config

        pos = state.ship_pos[0].cpu()  # (N,) complex64
        att = state.ship_attitude[0].cpu()  # (N,) complex64
        health = state.ship_health[0].cpu()  # (N,) float32
        power = state.ship_power[0].cpu()  # (N,) float32
        alive = state.ship_alive[0].cpu()  # (N,) bool
        team_id = state.ship_team_id[0].cpu()  # (N,) int32

        sz = cfg.ship_size
        for n in range(pos.shape[0]):
            if not alive[n].item():
                continue

            p = complex(pos[n].item())
            a = complex(att[n].item())
            color = cfg.team_colors[int(team_id[n].item()) % 2]

            tip = p + a * sz
            left = p + a * (-sz * 0.6) + a * 1j * (sz * 0.6)
            right = p + a * (-sz * 0.6) - a * 1j * (sz * 0.6)
            verts = [self._world_to_screen(v) for v in (tip, left, right)]
            pygame.draw.polygon(surf, color, verts)

            # Health bar above ship
            hp_frac = float(health[n].item()) / sc.max_health
            bar_w = sz * 2
            bar_x = int(p.real * self._scale) - sz
            bar_y = int(p.imag * self._scale) - sz - cfg.health_bar_height - 2
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

    def _draw_obstacles(self, state: TensorState, surf: pygame.Surface) -> None:
        """Draw all obstacles in env 0 as filled white circles."""
        if state.num_obstacles == 0:
            return
        obs_pos = state.obstacle_pos[0].cpu()  # (M,) complex64
        obs_rad = state.obstacle_radius[0].cpu()  # (M,) float32
        for m in range(obs_pos.shape[0]):
            cx, cy = self._world_to_screen(complex(obs_pos[m].item()))
            r = max(1, int(obs_rad[m].item() * self._scale))
            pygame.draw.circle(surf, (255, 255, 255), (cx, cy), r)

    def _draw_bullets(self, state: TensorState, surf: pygame.Surface) -> None:
        """Draw all active bullets in env 0 as small rectangles."""
        cfg = self._render_config
        bpos = state.bullet_pos[0].cpu()  # (N, K) complex64
        bact = state.bullet_active[0].cpu()  # (N, K) bool
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
