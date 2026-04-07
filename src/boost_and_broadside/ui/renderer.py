"""Pygame renderer for a single-environment game state.

Reads env index 0 from TensorState and draws ships, bullets, and health
bars at a fixed frame rate. All tensor reads call .cpu() after slicing —
acceptable overhead at 60fps on a single interactive environment.
"""

import os
import sys
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

    def render(self, state: TensorState) -> bool:
        """Draw one frame from env 0 of state.

        Args:
            state: Live TensorState — only env index 0 is read.

        Returns:
            True to keep running, False if the user closed the window.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False

        surf = self._screen
        surf.fill(self._render_config.background_color)
        self._draw_bullets(state, surf)
        self._draw_ships(state, surf)
        pygame.display.flip()
        return True

    def tick(self) -> None:
        """Cap frame rate to render_config.fps."""
        self._clock.tick(self._render_config.fps)

    def close(self) -> None:
        """Tear down the pygame window."""
        pygame.quit()

    # ------------------------------------------------------------------
    # Private drawing helpers
    # ------------------------------------------------------------------

    def _world_to_screen(self, c: complex) -> tuple[int, int]:
        """Convert a world-space complex position to screen pixel coords."""
        return (int(c.real * self._scale), int(c.imag * self._scale))

    def _draw_ships(self, state: TensorState, surf: pygame.Surface) -> None:
        """Draw all alive ships in env 0 as colored triangles with health bars."""
        cfg = self._render_config
        sc = self._ship_config

        pos = state.ship_pos[0].cpu()  # (N,) complex64
        att = state.ship_attitude[0].cpu()  # (N,) complex64
        health = state.ship_health[0].cpu()  # (N,) float32
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
            pygame.draw.rect(
                surf, (60, 0, 0), (bar_x, bar_y, bar_w, cfg.health_bar_height)
            )
            pygame.draw.rect(
                surf,
                (0, 200, 0),
                (bar_x, bar_y, int(bar_w * hp_frac), cfg.health_bar_height),
            )

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
