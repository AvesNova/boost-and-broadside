import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from env.event import EventType, GameEvent

from .bullets import Bullets
from .ship import Ship, default_ship_config
from .renderer import create_renderer
from .constants import RewardConstants
from .state import State


class Environment(gym.Env):
    """
    Gymnasium-compatible environment for the space battle game.

    This class manages the game state, physics updates, and rendering.
    It supports multiple game modes (1v1, NvN, etc.) and handles
    the interaction between agents and the physics engine.
    """

    def __init__(
        self,
        render_mode: str,
        world_size: tuple[int, int],
        memory_size: int,
        max_ships: int,
        agent_dt: float,
        physics_dt: float,
        random_positioning: bool,
        random_speed: bool,
        random_initialization: bool = False,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        """
        Initialize the environment.

        Args:
            render_mode: "human" for visualization, "none" for headless.
            world_size: (width, height) of the game world.
            memory_size: Size of the replay buffer (if applicable).
            max_ships: Maximum number of ships allowed in the game.
            agent_dt: Time step for agent decisions (seconds).
            physics_dt: Time step for physics updates (seconds).
            random_positioning: Whether to use completely random positioning instead of fractal.
            random_speed: Whether to use random speeds (0-180) instead of fixed speed (100).
            random_initialization: Whether to randomize health (10-100%) and power (0-100%).
            rng: Random number generator.
        """
        super().__init__()

        self.render_mode = render_mode
        self.world_size = world_size
        self.memory_size = memory_size
        self.max_ships = max_ships
        self.agent_dt = agent_dt
        self.physics_dt = physics_dt
        self.target_fps = 1 / physics_dt
        self.random_positioning = random_positioning
        self.random_speed = random_speed
        self.random_initialization = random_initialization
        self.rng = rng

        assert (
            agent_dt / physics_dt
        ) % 1 == 0, "agent_dt must be multiple of physics_dt"
        self.physics_substeps = int(agent_dt / physics_dt)

        # Lazy-loaded renderer
        self._renderer = None

        # Initialize state
        self.current_time = 0.0
        self.state: State | None = None
        self.events: list[GameEvent] = []

    @property
    def renderer(self):
        """Lazy-load the renderer only when needed"""
        if self._renderer is None and self.render_mode == "human":
            self._renderer = create_renderer(self.world_size, self.target_fps)
            if self._renderer is None:
                raise ImportError("pygame is required for human rendering mode")
        return self._renderer

    def add_human_player(self, ship_id: int) -> None:
        """Register a ship to be controlled by human input"""
        if self.render_mode == "human":
            self.renderer.add_human_player(ship_id)

    def remove_human_player(self, ship_id: int) -> None:
        """Remove human control from a ship"""
        if self.render_mode == "human":
            self.renderer.remove_human_player(ship_id)

    def one_vs_one_reset(self) -> State:
        """Reset to a standard 1v1 configuration."""
        ship_0 = Ship(
            ship_id=0,
            team_id=0,
            ship_config=default_ship_config,
            initial_x=0.25 * self.world_size[0],
            initial_y=0.40 * self.world_size[1],
            initial_vx=100.0,
            initial_vy=0.0,
            world_size=self.world_size,
        )

        ship_1 = Ship(
            ship_id=1,
            team_id=1,
            ship_config=default_ship_config,
            initial_x=0.75 * self.world_size[0],
            initial_y=0.60 * self.world_size[1],
            initial_vx=-100.0,
            initial_vy=0.0,
            world_size=self.world_size,
        )

        ships = {0: ship_0, 1: ship_1}
        return State(ships=ships)

    @staticmethod
    def fractal_ship_positions(level: int) -> np.ndarray:
        """Generate fractal positions for ship squads."""
        # Base shape (a1) as complex numbers
        base = np.array([0j, 1 - 1j, -1 - 1j, -2 - 2j], dtype=np.complex128)
        if level == 0:
            return np.array([0j], dtype=np.complex128)
        if level == 1:
            return base

        prev = Environment.fractal_ship_positions(level - 1)
        size = 4 ** (level - 1)
        shift = int(np.sqrt(size) * 2)

        offsets = np.array(
            [0, shift - shift * 1j, -shift - shift * 1j, 2 * shift - 2 * shift * 1j],
            dtype=np.complex128,
        )
        result = np.concatenate([prev + off for off in offsets])
        return result

    @staticmethod
    def get_ship_positions(n: int) -> np.ndarray:
        """Get n positions from the fractal sequence."""
        if n <= 1:
            return np.array([0j], dtype=np.complex128)
        # find minimal level with enough points
        level = int(np.ceil(np.log(n) / np.log(4)))
        seq = Environment.fractal_ship_positions(level)
        return seq[:n]

    def create_squad(
        self,
        team_id: int,
        n_ships: int,
        ship_id_offset: int,
        random_positioning: bool = False,
        random_speed: bool = False,
        random_initialization: bool = False,
    ) -> dict:
        """Create a squad of ships with either fractal or random positioning."""
        ships = {}

        for i in range(n_ships):
            # Calculate health and power
            initial_health = None
            initial_power = None

            if random_initialization:
                # Health: 10% to 100%
                initial_health = self.rng.uniform(0.1, 1.0) * default_ship_config.max_health
                # Power: 0% to 100%
                initial_power = self.rng.uniform(0.0, 1.0) * default_ship_config.max_power
            
            # ... Positioning logic ...
            pass # We'll replace the loop body below with distinct blocks to avoid repeating logic 

        # Let's rewrite the method body cleanly
        
        # Helper to get position/velocity
        def get_pos_vel(idx, n, rand_pos, rand_spd):
             if rand_pos:
                pos = self.rng.uniform(0, self.world_size[0]) + 1j * self.rng.uniform(0, self.world_size[1])
                att = np.exp(1j * self.rng.uniform(0, 2 * np.pi))
                speed = self.rng.uniform(0.1, 180) if rand_spd else 100.0
                vel = att * speed
                return pos, vel
             else:
                # Fractal logic requires computing all positions first...
                # So we can't easily helper-ize per ship without recomputing
                return None, None

        if random_positioning:
            for i in range(n_ships):
                position = self.rng.uniform(
                    0, self.world_size[0]
                ) + 1j * self.rng.uniform(0, self.world_size[1])
                attitude = np.exp(1j * self.rng.uniform(0, 2 * np.pi))
                speed = self.rng.uniform(0.1, 180) if random_speed else 100.0
                velocity = attitude * speed

                initial_health = None
                initial_power = None
                if random_initialization:
                    initial_health = self.rng.uniform(0.1, 1.0) * default_ship_config.max_health
                    initial_power = self.rng.uniform(0.0, 1.0) * default_ship_config.max_power

                ships[i + ship_id_offset] = Ship(
                    ship_id=i + ship_id_offset,
                    team_id=team_id,
                    ship_config=default_ship_config,
                    initial_x=position.real,
                    initial_y=position.imag,
                    initial_vx=velocity.real,
                    initial_vy=velocity.imag,
                    world_size=self.world_size,
                    initial_health=initial_health,
                    initial_power=initial_power,
                )
        else:
            # Original fractal positioning
            origin = self.rng.uniform(0, self.world_size[0]) + 1j * self.rng.uniform(
                0, self.world_size[1]
            )
            attitude = np.exp(1j * self.rng.uniform(0, 2 * np.pi))
            speed = self.rng.uniform(0, 180) if random_speed else 100.0
            velocity = attitude * speed

            offsets = (
                self.get_ship_positions(n_ships)
                * attitude
                * default_ship_config.collision_radius
                * 4
            )

            positions = offsets + origin

            for i, position in enumerate(positions):
                initial_health = None
                initial_power = None
                if random_initialization:
                    initial_health = self.rng.uniform(0.1, 1.0) * default_ship_config.max_health
                    initial_power = self.rng.uniform(0.0, 1.0) * default_ship_config.max_power

                ships[i + ship_id_offset] = Ship(
                    ship_id=i + ship_id_offset,
                    team_id=team_id,
                    ship_config=default_ship_config,
                    initial_x=position.real,
                    initial_y=position.imag,
                    initial_vx=velocity.real,
                    initial_vy=velocity.imag,
                    world_size=self.world_size,
                    initial_health=initial_health,
                    initial_power=initial_power,
                )

        return ships

    def n_vs_n_reset(
        self,
        ships_per_team: int | None,
        random_positioning: bool = False,
        random_speed: bool = False,
        random_initialization: bool = False,
    ) -> State:
        """Reset to an NvN configuration."""
        if ships_per_team is None:
            ships_per_team = self.rng.integers(
                1, int(self.max_ships / 2), endpoint=True
            )

        team_0 = self.create_squad(
            team_id=0,
            n_ships=ships_per_team,
            ship_id_offset=0,
            random_positioning=random_positioning,
            random_speed=random_speed,
            random_initialization=random_initialization,
        )
        team_1 = self.create_squad(
            team_id=1,
            n_ships=ships_per_team,
            ship_id_offset=ships_per_team,
            random_positioning=random_positioning,
            random_speed=random_speed,
            random_initialization=random_initialization,
        )
        ships = team_0 | team_1
        return State(ships=ships)

    def reset(
        self,
        seed: int | None = None,
        game_mode: str = "1v1",
        initial_obs: dict | None = None,
    ) -> tuple[dict, dict]:
        """
        Reset the environment.

        Args:
            seed: Random seed for reproducibility.
            game_mode: The game mode to initialize.
            initial_obs: Optional observation dict to restore state from.

        Returns:
            Tuple of (observation, info).
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_time = 0.0
        self.state = None
        self.events = []

        match game_mode:
            case "1v1_old":
                self.state = self.one_vs_one_reset()
            case "1v1":
                self.state = self.n_vs_n_reset(
                    ships_per_team=1,
                    random_positioning=self.random_positioning,
                    random_speed=self.random_speed,
                    random_initialization=self.random_initialization,
                )
            case "2v2":
                self.state = self.n_vs_n_reset(
                    ships_per_team=2,
                    random_positioning=self.random_positioning,
                    random_speed=self.random_speed,
                    random_initialization=self.random_initialization,
                )
            case "3v3":
                self.state = self.n_vs_n_reset(
                    ships_per_team=3,
                    random_positioning=self.random_positioning,
                    random_speed=self.random_speed,
                    random_initialization=self.random_initialization,
                )
            case "4v4":
                self.state = self.n_vs_n_reset(
                    ships_per_team=4,
                    random_positioning=self.random_positioning,
                    random_speed=self.random_speed,
                    random_initialization=self.random_initialization,
                )
            case "nvn":
                self.state = self.n_vs_n_reset(
                    ships_per_team=None,
                    random_positioning=self.random_positioning,
                    random_speed=self.random_speed,
                    random_initialization=self.random_initialization,
                )
            case "reset_from_observation":
                if initial_obs is None:
                    raise ValueError(
                        "initial_obs must be provided for reset_from_observation"
                    )
                self.state = self.reset_from_observation(initial_obs=initial_obs)
            case _:
                raise ValueError(f"Unknown game mode: {game_mode}")

        return self.get_observation(), {}

    def reset_from_observation(self, initial_obs: dict) -> State:
        """Reset environment to match the state described by an observation dict"""
        self.current_time = 0.0
        self.state = None

        ships = {}

        num_ships = initial_obs["ship_id"].shape[0]

        for i in range(num_ships):
            alive = bool(initial_obs["alive"][i, 0].item())
            if not alive:
                continue

            # Extract values for this ship (slice i)
            ship_id = int(initial_obs["ship_id"][i, 0].item())
            team_id = int(initial_obs["team_id"][i, 0].item())
            health = float(initial_obs["health"][i, 0].item())
            power = float(initial_obs["power"][i, 0].item())
            position = initial_obs["position"][i, 0].item()  # complex64
            velocity = initial_obs["velocity"][i, 0].item()  # complex64
            speed = float(initial_obs["speed"][i, 0].item())
            attitude = initial_obs["attitude"][i, 0].item()  # complex64
            is_shooting = bool(initial_obs["is_shooting"][i, 0].item())

            # Extract position and velocity components
            initial_x = position.real
            initial_y = position.imag
            initial_vx = velocity.real
            initial_vy = velocity.imag

            # Create ship with default config and initial position/velocity
            ship = Ship(
                ship_id=ship_id,
                team_id=team_id,
                ship_config=default_ship_config,
                initial_x=initial_x,
                initial_y=initial_y,
                initial_vx=initial_vx,
                initial_vy=initial_vy,
                world_size=self.world_size,
            )

            # Override ship state with values from observation
            ship.alive = alive
            ship.health = health
            ship.power = power
            ship.speed = speed
            ship.attitude = attitude
            ship.is_shooting = is_shooting

            ships[ship_id] = ship

        return State(ships=ships)

    def render(self, state: State) -> None:
        """Render current game state"""
        if self.render_mode == "human" and self.state is not None:
            self.renderer.render(state)

    def _wrap_ship_position(self, position: complex) -> complex:
        """Wrap ship position to toroidal world boundaries"""
        wrapped_real = position.real % self.world_size[0]
        wrapped_imag = position.imag % self.world_size[1]
        return wrapped_real + 1j * wrapped_imag

    def _wrap_bullet_positions(self, bullets: Bullets) -> None:
        """Wrap bullet positions to toroidal world boundaries"""
        if bullets.num_active == 0:
            return

        active_slice = slice(0, bullets.num_active)
        bullets.x[active_slice] %= self.world_size[0]
        bullets.y[active_slice] %= self.world_size[1]

    def _ship_actions(self, actions: dict[int, torch.Tensor], state: State) -> None:
        for ship_id, ship in state.ships.items():
            if ship.alive:
                ship.forward(
                    actions[ship_id],
                    state.bullets,
                    self.current_time,
                    self.physics_dt,
                )
                ship.position = self._wrap_ship_position(ship.position)

    def _bullet_actions(self, bullets: Bullets) -> None:
        bullets.update_all(self.physics_dt)
        self._wrap_bullet_positions(bullets)

    def _ship_bullet_collisions(self, ships: dict[int, Ship], bullets: Bullets):
        if bullets.num_active == 0:
            return

        bx, by, bullet_ship_ids = bullets.get_active_positions()

        for ship in ships.values():
            if not ship.alive:
                continue

            dx = bx - ship.position.real
            dy = by - ship.position.imag
            distances_sq = dx * dx + dy * dy

            hit_mask = (distances_sq < ship.collision_radius_squared) & (
                bullet_ship_ids != ship.ship_id
            )

            if np.any(hit_mask):
                damage = np.sum(hit_mask) * ship.config.bullet_damage
                ship.damage_ship(damage)

                self.events.append(
                    GameEvent(
                        event_type=EventType.DAMAGE,
                        team_id=ship.team_id,
                        ship_id=ship.ship_id,
                        amount=damage,
                    )
                )

                if not ship.alive:
                    self.events.append(
                        GameEvent(
                            event_type=EventType.DEATH,
                            team_id=ship.team_id,
                            ship_id=ship.ship_id,
                            amount=1,
                        )
                    )

                hit_indices = np.where(hit_mask)[0]
                for idx in reversed(sorted(hit_indices)):
                    bullets.remove_bullet(idx)

    def _calculate_team_reward(self, team_id: int) -> float:
        reward = 0

        for event in self.events:
            match event.event_type:
                case EventType.DAMAGE:
                    if event.team_id == team_id:
                        reward += event.amount * RewardConstants.ALLY_DAMAGE
                    else:
                        reward += event.amount * RewardConstants.ENEMY_DAMAGE
                case EventType.DEATH:
                    if event.team_id == team_id:
                        reward += event.amount * RewardConstants.ALLY_DEATH
                    else:
                        reward += event.amount * RewardConstants.ENEMY_DEATH
                case EventType.WIN:
                    if event.team_id == team_id:
                        reward += RewardConstants.VICTORY
                    else:
                        reward += RewardConstants.DEFEAT
                case EventType.TIE:
                    reward += RewardConstants.DRAW

        return reward

    def _check_termination(self, state: State) -> bool:
        """Check if episode should terminate and which agents are done"""
        alive_ships = [ship for ship in state.ships.values() if ship.alive]
        alive_teams = set(ship.team_id for ship in alive_ships)

        len_alive_teams = len(alive_teams)

        if len_alive_teams == 0:
            self.events.append(GameEvent(event_type=EventType.TIE))
            return True

        if len_alive_teams == 1:
            (alive_team,) = alive_teams
            self.events.append(GameEvent(event_type=EventType.WIN, team_id=alive_team))
            return True

        return False

    def step(
        self, actions: dict[int, torch.Tensor]
    ) -> tuple[dict, dict[int, float], bool, bool, dict]:

        self.events = []

        # Handle events if in human mode
        if self.render_mode == "human":
            if not self.renderer.handle_events():
                # User closed window - could handle this gracefully
                pass

        self.state.time += self.agent_dt

        # Run physics substeps with rendering
        for substep in range(self.physics_substeps):
            # Update human input at physics rate
            if self.render_mode == "human":
                self.renderer.update_human_actions()
                human_actions = self.renderer.get_human_actions()
                # Merge AI and human actions
                merged_actions = {**actions, **human_actions}
            else:
                merged_actions = actions

            # Physics step
            self._ship_actions(merged_actions, self.state)
            self._bullet_actions(self.state.bullets)
            self._ship_bullet_collisions(self.state.ships, self.state.bullets)
            self.current_time += self.physics_dt

            if self.render_mode == "human":
                self.render(self.state)

        # Calculate termination
        terminated = self._check_termination(self.state)

        rewards = {
            team_id: self._calculate_team_reward(team_id=team_id) for team_id in (0, 1)
        }

        info = {
            "current_time": self.current_time,
            "active_bullets": self.state.bullets.num_active,
        }

        # Add human control info if in human mode
        if self.render_mode == "human":
            info["human_controlled"] = list(self.renderer.human_ship_ids)

        return self.get_observation(), rewards, terminated, False, info

    def close(self):
        """Clean up resources"""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None

    def get_observation(self) -> dict:
        """Extract observations for each ship from current state"""
        observations = self._get_empty_observation()

        if not self.state:
            return observations

        for ship_id, ship in self.state.ships.items():
            local_obs = ship.get_state()
            for key, value in local_obs.items():
                observations[key][ship_id] = torch.tensor(value)

        return observations

    def _get_empty_observation(self) -> dict:
        """Empty observation for reset state"""
        return {
            "ship_id": torch.zeros((self.max_ships), dtype=torch.int64),
            "team_id": torch.zeros((self.max_ships), dtype=torch.int64),
            "alive": torch.zeros((self.max_ships), dtype=torch.int64),
            "health": torch.zeros((self.max_ships), dtype=torch.int64),
            "power": torch.zeros((self.max_ships), dtype=torch.float32),
            "position": torch.zeros((self.max_ships), dtype=torch.complex64),
            "velocity": torch.zeros((self.max_ships), dtype=torch.complex64),
            "speed": torch.zeros((self.max_ships), dtype=torch.float32),
            "attitude": torch.zeros((self.max_ships), dtype=torch.complex64),
            "is_shooting": torch.zeros((self.max_ships), dtype=torch.int64),
        }

    @property
    def action_space(self) -> spaces.Space:
        """Define the action space for each ship"""
        # Multi-discrete for categorical actions: [Power(3), Turn(7), Shoot(2)]
        return spaces.MultiDiscrete([3, 7, 2])

    @property
    def observation_space(self) -> spaces.Space:
        """Define the observation space for each ship"""
        return spaces.Dict(
            {
                "tokens": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.max_ships, 10),
                    dtype=np.float32,
                )
            }
        )
