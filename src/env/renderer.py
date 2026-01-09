"""
Pygame-based rendering for the ship combat environment.

Handles visualization of ships, bullets, and UI elements, as well as
human player input via keyboard controls.
"""

import numpy as np
import torch
import pygame

from env.constants import PowerActions, TurnActions, ShootActions
from env.ship import Ship
from env.bullets import Bullets
from env.state import State


class GameRenderer:
    """Handles pygame rendering and human input for the environment."""

    def __init__(self, world_size: tuple[int, int], target_fps: int = 60):
        """
        Initialize the game renderer.

        Args:
            world_size: Width and height of the game world in pixels.
            target_fps: Target frames per second for rendering.
        """
        self.world_size = world_size
        self.target_fps = target_fps

        # Pygame components
        self.screen: pygame.Surface | None = None
        self.clock: pygame.time.Clock | None = None
        self.font: pygame.font.Font | None = None
        self.initialized = False

        # Input handling
        self.human_ship_ids: set[int] = set()
        self.human_actions: dict[int, torch.Tensor] = {}

    def initialize(self) -> None:
        """Initialize pygame components."""
        if self.initialized:
            return

        try:
            pygame.init()
            # Ensure video system is properly initialized
            if not pygame.display.get_init():
                pygame.display.init()

            self.screen = pygame.display.set_mode(self.world_size)
            pygame.display.set_caption("Ship Combat Environment")
            self.clock = pygame.time.Clock()
            self.font = pygame.font.Font(None, 36)
            self.initialized = True
        except pygame.error as e:
            raise RuntimeError(
                f"Failed to initialize pygame: {e}. Make sure you have a display available."
            ) from e

    def add_human_player(self, ship_id: int) -> None:
        """
        Register a ship to be controlled by human input.

        Args:
            ship_id: ID of the ship to control.
        """
        self.human_ship_ids.add(ship_id)
        # Initialize with 3 indices: Power, Turn, Shoot
        self.human_actions[ship_id] = torch.zeros(3, dtype=torch.int)

    def remove_human_player(self, ship_id: int) -> None:
        """
        Remove human control from a ship.

        Args:
            ship_id: ID of the ship to remove control from.
        """
        self.human_ship_ids.discard(ship_id)
        self.human_actions.pop(ship_id, None)

    def handle_events(self) -> bool:
        """
        Handle pygame events.

        Returns:
            True if the game should continue running, False if quit was requested.
        """
        if not self.initialized:
            self.initialize()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def update_human_actions(self) -> None:
        """Update human actions based on current keyboard state."""
        if not self.human_ship_ids:
            return

        if not self.initialized:
            self.initialize()

        keys = pygame.key.get_pressed()

        for ship_id in self.human_ship_ids:
            # Action vector: [Power, Turn, Shoot]
            power_idx = PowerActions.COAST
            turn_idx = TurnActions.GO_STRAIGHT
            shoot_idx = ShootActions.NO_SHOOT

            # Key mappings
            if ship_id == 0:
                k_up = keys[pygame.K_UP] or keys[pygame.K_w]
                k_down = keys[pygame.K_DOWN] or keys[pygame.K_s]
                k_left = keys[pygame.K_LEFT] or keys[pygame.K_a]
                k_right = keys[pygame.K_RIGHT] or keys[pygame.K_d]
                k_shift = keys[pygame.K_LSHIFT] or keys[pygame.K_RSHIFT]
                k_shoot = keys[pygame.K_SPACE]
            elif ship_id == 1:
                k_up = keys[pygame.K_i]
                k_down = keys[pygame.K_k]
                k_left = keys[pygame.K_j]
                k_right = keys[pygame.K_l]
                k_shift = keys[pygame.K_u]  # Using 'u' as shift equivalent for P2
                k_shoot = keys[pygame.K_o]
            else:
                continue

            # Power Logic
            if k_up and k_down:
                power_idx = PowerActions.COAST
            elif k_up:
                power_idx = PowerActions.BOOST
            elif k_down:
                power_idx = PowerActions.REVERSE
            else:
                power_idx = PowerActions.COAST

            # Turn Logic
            if k_left and k_right:
                if k_shift:
                    turn_idx = TurnActions.SHARP_AIR_BRAKE
                else:
                    turn_idx = TurnActions.AIR_BRAKE
            elif k_left:
                if k_shift:
                    turn_idx = TurnActions.SHARP_LEFT
                else:
                    turn_idx = TurnActions.TURN_LEFT
            elif k_right:
                if k_shift:
                    turn_idx = TurnActions.SHARP_RIGHT
                else:
                    turn_idx = TurnActions.TURN_RIGHT
            else:
                turn_idx = TurnActions.GO_STRAIGHT

            # Shoot Logic
            if k_shoot:
                shoot_idx = ShootActions.SHOOT
            else:
                shoot_idx = ShootActions.NO_SHOOT

            self.human_actions[ship_id] = torch.tensor(
                [power_idx, turn_idx, shoot_idx], dtype=torch.float32
            )

    def get_human_actions(self) -> dict[int, torch.Tensor]:
        """
        Get current human actions for all human-controlled ships.

        Returns:
            Dictionary mapping ship_id to action tensor.
        """
        return self.human_actions.copy()

    def _render_ship(self, ship: Ship) -> None:
        """
        Render a single ship.

        Args:
            ship: Ship instance to render.
        """
        if not ship.alive:
            return

        x, y = ship.position.real, ship.position.imag

        # Ship color based on team and health
        health_ratio = ship.health / ship.config.max_health
        if ship.team_id == 0:
            base_color = (0, 100, 255)  # Blue team
        else:
            base_color = (255, 100, 0)  # Red team

        # Dim color based on health
        color = tuple(int(c * health_ratio) for c in base_color)

        # Draw ship as triangle pointing in attitude direction
        size = ship.config.collision_radius
        attitude_angle = np.angle(ship.attitude)

        # Triangle points (pointing right initially)
        points = np.array([[size, 0], [-size / 2, -size / 2], [-size / 2, size / 2]])

        # Rotate by attitude
        cos_a, sin_a = np.cos(attitude_angle), np.sin(attitude_angle)
        rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
        rotated_points = points @ rotation_matrix.T

        # Translate to ship position
        screen_points = rotated_points + np.array([x, y])

        pygame.draw.polygon(self.screen, color, screen_points)

        # Draw health bar
        bar_width = int(size * 2)
        bar_height = 4
        bar_x = int(x - bar_width // 2)
        bar_y = int(y - size - 10)

        # Background (red)
        pygame.draw.rect(
            self.screen, (100, 0, 0), (bar_x, bar_y, bar_width, bar_height)
        )
        # Health (green)
        pygame.draw.rect(
            self.screen,
            (0, 200, 0),
            (bar_x, bar_y, int(bar_width * health_ratio), bar_height),
        )

    def _render_bullets(self, bullets: Bullets) -> None:
        """
        Render all active bullets.

        Args:
            bullets: Bullets instance containing active projectiles.
        """
        if bullets.num_active == 0:
            return

        for i in range(bullets.num_active):
            x, y = bullets.x[i], bullets.y[i]
            ship_id = bullets.ship_id[i]

            # Bullet color matches ship team
            if ship_id == 0:
                color = (100, 150, 255)  # Light blue
            else:
                color = (255, 150, 100)  # Light red

            pygame.draw.circle(self.screen, color, (int(x), int(y)), 2)

    def _render_ui(self, state: State) -> None:
        """
        Render UI information.

        Args:
            state: Current game state.
        """
        y_offset = 10

        for ship_id, ship in state.ships.items():
            if not ship.alive:
                continue

            # Ship info
            health_text = f"Ship {ship_id} Health: {ship.health:.1f}"
            power_text = f"Power: {ship.power:.1f}"

            health_surface = self.font.render(health_text, True, (255, 255, 255))
            power_surface = self.font.render(power_text, True, (255, 255, 255))

            self.screen.blit(health_surface, (10, y_offset))
            self.screen.blit(power_surface, (10, y_offset + 25))

            # Mark human-controlled ships
            if ship_id in self.human_ship_ids:
                human_text = "(Human)"
                human_surface = self.font.render(human_text, True, (0, 255, 0))
                self.screen.blit(human_surface, (200, y_offset))

            y_offset += 70

    def render(self, state: State) -> None:
        """
        Render the current game state.

        Args:
            state: Current game state to render.
        """
        if not self.initialized:
            self.initialize()

        # Clear screen
        self.screen.fill((0, 0, 20))  # Dark blue background

        # Render game objects
        for ship in state.ships.values():
            self._render_ship(ship)

        self._render_bullets(state.bullets)
        self._render_ui(state)

        # Update display
        pygame.display.flip()
        self.clock.tick(self.target_fps)

    def close(self) -> None:
        """Clean up pygame resources."""
        if self.initialized:
            pygame.quit()
            self.initialized = False
            self.screen = None
            self.clock = None
            self.font = None


def create_renderer(
    world_size: tuple[int, int], target_fps: int = 60
) -> GameRenderer | None:
    """
    Create a game renderer instance.

    Args:
        world_size: Width and height of the game world in pixels.
        target_fps: Target frames per second for rendering.

    Returns:
        GameRenderer instance.
    """
    return GameRenderer(world_size, target_fps)
