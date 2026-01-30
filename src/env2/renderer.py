import os
import pygame
import math
import numpy as np
from typing import Dict, Tuple

from core.types import RenderState, RenderShip
from core.config import ShipConfig

# Constants for Rendering
# Constants for Rendering
NATIVE_GAME_WIDTH = 1024
NATIVE_HEIGHT = 1024
NATIVE_SIDEBAR_WIDTH = 250
NATIVE_WIDTH = NATIVE_GAME_WIDTH + NATIVE_SIDEBAR_WIDTH
BACKGROUND_COLOR = (10, 10, 20)
GRID_COLOR = (30, 30, 50)
TEXT_COLOR = (255, 255, 255)
SIDEBAR_COLOR = (20, 20, 30)

TEAM_COLORS = {
    0: (50, 150, 255),   # Blue Team
    1: (255, 80, 80),    # Red Team
}

HP_BAR_COLOR = (0, 255, 0)
HP_BAR_BG = (50, 0, 0)

class GameRenderer:
    """
    Renders the game state using Pygame.
    
    This renderer is backend-agnostic, relying on `RenderState` and `RenderShip` 
    data structures from `src.core.types`.
    """
    def __init__(self, config: ShipConfig, target_fps: int = 60):
        self.config = config
        self.target_fps = target_fps
        self.world_size = config.world_size
        
        # Initialize Pygame
        os.environ["SDL_VIDEODRIVER"] = "dummy" if os.environ.get("HEADLESS") else ""
        pygame.init()
        pygame.font.init()
        
        # Core Dimensions
        self.game_width = NATIVE_GAME_WIDTH
        self.game_height = NATIVE_HEIGHT
        self.sidebar_width = NATIVE_SIDEBAR_WIDTH
        self.native_width = NATIVE_WIDTH
        self.native_height = NATIVE_HEIGHT
        
        # Surfaces
        # Native surface for high-res rendering
        self.display_surface = pygame.Surface((self.native_width, self.native_height))
        # Actual Window - Resizable, starting at a reasonable default
        self.window_width = 1000
        self.window_height = 800
        self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)
        pygame.display.set_caption("Boost and Broadside (Env2)")
        
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("monospace", 14)
        
        # Scaling (World -> Native Surface)
        self.scale_x = self.game_width / self.world_size[0]
        self.scale_y = self.game_height / self.world_size[1]
        
        # Human Input
        self.human_players = set()
        self.human_actions = {} # Map ship_id -> action dict
        
        # Precompute ship shape (triangle) - Adjusted scale
        self.ship_shape = [
            (12, 0),   # Nose
            (-8, 8),   # Rear Left
            (-4, 0),   # Engine
            (-8, -8)   # Rear Right
        ]

    def add_human_player(self, ship_id: int):
        self.human_players.add(ship_id)
        
    def remove_human_player(self, ship_id: int):
        self.human_players.discard(ship_id)

    def close(self):
        pygame.quit()

    def handle_events(self):
        """Process Pygame events and update human actions."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                return
            elif event.type == pygame.VIDEORESIZE:
                self.window_width = event.w
                self.window_height = event.h
                # The screen surface is automatically resized by pygame on VIDEORESIZE? 
                # Usually we need to call set_mode again.
                self.screen = pygame.display.set_mode((self.window_width, self.window_height), pygame.RESIZABLE)

        # Keyboard Input
        keys = pygame.key.get_pressed()
        
        # Default Action: Coast, No Turn, Hold Fire
        # We need to map keys to actions for each human player
        # For now, support single player WASD + Space
        
        # TODO: Support multiple local players or key mapping
        
        if not self.human_players:
            return

        # Determine actions
        power = 0 # 0=Coast
        turn = 0  # 0=No Turn
        shoot = 0 # 0=Hold
        
        # W/S: Boost/Reverse
        if keys[pygame.K_w]:
            power = 1 # Boost
        elif keys[pygame.K_s]:
            power = 2 # Reverse
            
        # A/D: Turn
        if keys[pygame.K_a]:
            turn = 1 # Left
            if keys[pygame.K_LSHIFT]:
               turn = 3 # Sharp Left
        elif keys[pygame.K_d]:
            turn = 2 # Right
            if keys[pygame.K_LSHIFT]:
               turn = 4 # Sharp Right
               
        # Space: Shoot
        if keys[pygame.K_SPACE]:
            shoot = 1
            
        # Assign to all human players (or just the first one)
        for ship_id in self.human_players:
            self.human_actions[ship_id] = {
                "power": power,
                "turn": turn,
                "shoot": shoot
            }

    def get_human_actions(self) -> Dict[int, Dict[str, int]]:
        return self.human_actions

    def render(self, state: RenderState):
        """Render the given frame."""
        # Draw everything to the native high-res surface
        self.display_surface.fill(BACKGROUND_COLOR)
        
        # Draw Grid
        self._draw_grid()
        
        # Draw Ships
        for ship_id, ship in state.ships.items():
            if ship.alive:
                self._draw_ship(ship)
                
        # Draw Bullets
        self._draw_bullets(state)
        
        # Draw UI
        self._draw_ui(state)
        
        # Scale and Blit to Window
        # Calculate aspect ratio preserving scale
        scale = min(self.window_width / self.native_width, self.window_height / self.native_height)
        new_width = int(self.native_width * scale)
        new_height = int(self.native_height * scale)
        
        scaled_surface = pygame.transform.smoothscale(self.display_surface, (new_width, new_height))
        
        # Center the scaled surface
        x_pos = (self.window_width - new_width) // 2
        y_pos = (self.window_height - new_height) // 2
        
        self.screen.fill((0, 0, 0)) # Fill black bars
        self.screen.blit(scaled_surface, (x_pos, y_pos))
        
        pygame.display.flip()
        self.clock.tick(self.target_fps)

    def _world_to_screen(self, x: float, y: float) -> Tuple[int, int]:
        sx = int(x * self.scale_x)
        sy = int(y * self.scale_y)
        # Flip Y axis? Usually simulation (0,0) is bottom-left or top-left.
        # Assuming top-left origin for both sim and screen for simplicity unless stated otherwise.
        # If sim is cartesian (y up), we need `game_height - sy`.
        # Standard pygame is y-down.
        # If `env` coordinates are standard math (y-up), we convert.
        # Assuming consistent coordinate system for now (y-down matches screen).
        return sx, sy

    def _draw_grid(self):
        # Draw vertical lines
        for x in range(0, int(self.world_size[0]) + 1, 100):
            sx, _ = self._world_to_screen(x, 0)
            pygame.draw.line(self.display_surface, GRID_COLOR, (sx, 0), (sx, self.game_height))
            
        # Draw horizontal lines
        for y in range(0, int(self.world_size[1]) + 1, 100):
            _, sy = self._world_to_screen(0, y)
            pygame.draw.line(self.display_surface, GRID_COLOR, (0, sy), (self.game_width, sy))

    def _draw_ship(self, ship: RenderShip):
        # Position
        x, y = ship.position.real, ship.position.imag
        sx, sy = self._world_to_screen(x, y)
        
        # Color
        color = TEAM_COLORS.get(ship.team_id, (255, 255, 255))
        if ship.id in self.human_players:
            color = (255, 255, 0) # Highlight human
            
        # Rotation
        # Attitude is complex number (cos, sin)
        # Angle in radians = atan2(sin, cos)
        # Pygame rotation is degrees, counter-clockwise?
        angle_rad = math.atan2(ship.attitude.imag, ship.attitude.real)
        angle_deg = -math.degrees(angle_rad) # Pygame rotates CCW, but 0 is right?
        
        # Rotate logic
        # 0 rad = Right (1, 0)
        # Pygame 0 deg = Up? No, usually Right?
        # Let's rotate the points manually or use transform
        
        transformed_points = []
        cos_a = math.cos(angle_rad)
        sin_a = math.sin(angle_rad)
        
        for px, py in self.ship_shape:
            # Rotate
            rx = px * cos_a - py * sin_a
            ry = px * sin_a + py * cos_a
            # Translate
            tx = rx * self.scale_x + sx
            ty = ry * self.scale_y + sy
            transformed_points.append((tx, ty))
            
        pygame.draw.polygon(self.display_surface, color, transformed_points)
        
        # Health Bar
        bar_width = 30
        bar_height = 4
        health_pct = max(0, ship.health / ship.max_health)
        
        bar_x = sx - bar_width // 2
        bar_y = sy - 20
        
        pygame.draw.rect(self.display_surface, HP_BAR_BG, (bar_x, bar_y, bar_width, bar_height))
        pygame.draw.rect(self.display_surface, HP_BAR_COLOR, (bar_x, bar_y, int(bar_width * health_pct), bar_height))

    def _draw_bullets(self, state: RenderState):
        if len(state.bullet_x) == 0:
            return
            
        for i in range(len(state.bullet_x)):
            x = state.bullet_x[i]
            y = state.bullet_y[i]
            team_id = 0 # Default (TODO: pass owner team via RenderState if needed)
            # We have owner_id, so we can look up team
            owner_id = state.bullet_owner_id[i]
            if owner_id in state.ships:
                team_id = state.ships[owner_id].team_id
            
            color = TEAM_COLORS.get(team_id, (255, 255, 255))
            
            sx, sy = self._world_to_screen(x, y)
            pygame.draw.circle(self.display_surface, color, (sx, sy), 2)

    def _draw_ui(self, state: RenderState):
        """Draw sidebar with game telemetry."""
        # Draw Sidebar background
        sidebar_rect = pygame.Rect(self.game_width, 0, self.sidebar_width, self.native_height)
        pygame.draw.rect(self.display_surface, SIDEBAR_COLOR, sidebar_rect)
        pygame.draw.line(self.display_surface, (100, 100, 100), (self.game_width, 0), (self.game_width, self.native_height))
        
        # Display Stats for Human Player (if any)
        x_offset = self.game_width + 15
        y_offset = 20
        
        for ship_id in sorted(state.ships.keys()):
            ship = state.ships[ship_id]
            is_human = ship_id in self.human_players
            
            color = TEXT_COLOR
            if is_human:
                color = (255, 255, 0)
                
            stats = [
                f"S{ship_id} T{ship.team_id} {'[HUMAN]' if is_human else ''}",
                f"HP: {ship.health:.0f}/{ship.max_health:.0f}",
                f"PWR: {ship.power:.0f}/100",
                f"Status: {'ALIVE' if ship.alive else 'DEAD'}"
            ]
            
            for line in stats:
                text = self.font.render(line, True, color if line == stats[0] else TEXT_COLOR)
                self.display_surface.blit(text, (x_offset, y_offset))
                y_offset += 16
                
            y_offset += 10 # Spacer between ships
            
            if y_offset > self.native_height - 50:
                break # Avoid overflow
