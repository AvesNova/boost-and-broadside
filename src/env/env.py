from collections import deque
from copy import deepcopy
import gymnasium as gym
import numpy as np
import torch
from gymnasium import spaces

from .bullets import Bullets
from .ship import Ship, default_ship_config
from .renderer import create_renderer
from .constants import Actions, RewardConstants
from .state import State


class Environment(gym.Env):
    def __init__(
        self,
        render_mode: str,
        world_size: tuple[int, int],
        memory_size: int,
        max_ships: int,
        agent_dt: float,
        physics_dt: float,
        rng: np.random.Generator = np.random.default_rng(),
    ):
        super().__init__()

        self.render_mode = render_mode
        self.world_size = world_size
        self.memory_size = memory_size
        self.max_ships = max_ships
        self.agent_dt = agent_dt
        self.physics_dt = physics_dt
        self.target_fps = 1 / physics_dt
        self.rng = rng

        assert (
            agent_dt / physics_dt
        ) % 1 == 0, "agent_dt must be multiple of physics_dt"
        self.physics_substeps = int(agent_dt / physics_dt)

        # Lazy-loaded renderer
        self._renderer = None

        # Initialize state
        self.current_time = 0.0
        self.state: deque[State] = deque(maxlen=memory_size)

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
    def fractal_ship_positions(level):
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
    def get_ship_positions(n):
        if n <= 1:
            return np.array([0j], dtype=np.complex128)
        # find minimal level with enough points
        level = int(np.ceil(np.log(n) / np.log(4)))
        seq = Environment.fractal_ship_positions(level)
        return seq[:n]

    def create_squad(self, team_id: int, n_ships: int, ship_id_offset: int) -> dict:
        origin = self.rng.uniform(0, self.world_size[0]) + 1j * self.rng.uniform(
            0, self.world_size[1]
        )
        attitude = np.exp(1j * self.rng.uniform(0, 2 * np.pi))
        velocity = attitude * 100.0

        offsets = (
            self.get_ship_positions(n_ships)
            * attitude
            * default_ship_config.collision_radius
            * 4
        )

        positions = offsets + origin

        ships = {}
        for i, position in enumerate(positions):
            ships[i + ship_id_offset] = Ship(
                ship_id=i + ship_id_offset,
                team_id=team_id,
                ship_config=default_ship_config,
                initial_x=position.real,
                initial_y=position.imag,
                initial_vx=velocity.real,
                initial_vy=velocity.imag,
                world_size=self.world_size,
            )

        return ships

    def n_vs_n_reset(self, ships_per_team: int | None) -> State:
        if ships_per_team is None:
            ships_per_team = self.rng.integers(
                1, int(self.max_ships / 2), endpoint=True
            )

        team_0 = self.create_squad(team_id=0, n_ships=ships_per_team, ship_id_offset=0)
        team_1 = self.create_squad(
            team_id=1, n_ships=ships_per_team, ship_id_offset=ships_per_team
        )
        ships = team_0 | team_1
        return State(ships=ships)

    def reset(
        self, game_mode: str = "1v1", initial_obs: dict | None = None
    ) -> tuple[dict, dict]:
        self.current_time = 0.0
        self.state.clear()

        if game_mode == "1v1_old":
            self.state.append(self.one_vs_one_reset())
        elif game_mode == "1v1":
            self.state.append(self.n_vs_n_reset(ships_per_team=1))
        elif game_mode == "2v2":
            self.state.append(self.n_vs_n_reset(ships_per_team=2))
        elif game_mode == "3v3":
            self.state.append(self.n_vs_n_reset(ships_per_team=3))
        elif game_mode == "4v4":
            self.state.append(self.n_vs_n_reset(ships_per_team=4))
        elif game_mode == "nvn":
            self.state.append(self.n_vs_n_reset(ships_per_team=None))
        elif game_mode == "reset_from_observation":
            self.state.append(self.reset_from_observation(initial_obs=initial_obs))
        else:
            raise ValueError(f"Unknown game mode: {game_mode}")

        return self.get_observation(), {}

    def reset_from_observation(self, initial_obs: dict) -> tuple[dict, dict]:
        """Reset environment to match the state described by an observation dict"""
        self.current_time = 0.0
        self.state.clear()

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

        initial_state = State(ships=ships)
        self.state.append(initial_state)

        return self.get_observation(), {}

    def render(self, state: State) -> None:
        """Render current game state"""
        if self.render_mode == "human" and len(self.state) > 0:
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
                ship.damage_ship(np.sum(hit_mask) * ship.config.bullet_damage)

                hit_indices = np.where(hit_mask)[0]
                for idx in reversed(sorted(hit_indices)):
                    bullets.remove_bullet(idx)

    def _calculate_team_reward(
        self, current_state: State, team_id: int, episode_ended: bool = False
    ) -> float:
        """
        Calculate reward for a specific team using the new reward structure:
        - Victory/Defeat: ±1.0 (episode end only)
        - Ship Death: ±0.1 (ally death = -0.1, enemy death = +0.1)
        - Damage: ±0.001 per damage point (dealt = +0.001, taken = -0.001)
        """
        team_reward = 0.0

        # Calculate tactical rewards (damage/death) from state transitions
        if len(self.state) >= 2:
            previous_state = self.state[-2]
            team_reward += self._calculate_tactical_rewards(
                current_state, previous_state, team_id
            )

        # Add episode outcome rewards if the episode has ended
        if episode_ended:
            team_reward += self._calculate_outcome_rewards(current_state, team_id)

        return team_reward

    def _calculate_tactical_rewards(
        self, current_state: State, previous_state: State, team_id: int
    ) -> float:
        """Calculate damage and death rewards from state transition"""
        reward = 0.0

        for ship_id, current_ship in current_state.ships.items():
            if ship_id not in previous_state.ships:
                continue

            previous_ship = previous_state.ships[ship_id]

            # Death events (±0.1)
            if previous_ship.alive and not current_ship.alive:
                if current_ship.team_id == team_id:
                    reward += RewardConstants.ALLY_DEATH_PENALTY  # Our ship died
                else:
                    reward += RewardConstants.ENEMY_DEATH_BONUS  # Enemy ship died

            # Damage events (±0.001 per damage point)
            elif previous_ship.alive and current_ship.alive:
                damage_taken = previous_ship.health - current_ship.health
                if damage_taken > 0:
                    if current_ship.team_id == team_id:
                        reward -= (
                            RewardConstants.DAMAGE_REWARD_SCALE * damage_taken
                        )  # Our ship took damage
                    else:
                        reward += (
                            RewardConstants.DAMAGE_REWARD_SCALE * damage_taken
                        )  # Enemy took damage

        return reward

    def _calculate_outcome_rewards(self, final_state: State, team_id: int) -> float:
        """Calculate episode outcome rewards (±1.0)"""

        # Count alive ships per team
        team_ships_alive = {}
        for ship in final_state.ships.values():
            if ship.alive:
                team_ships_alive[ship.team_id] = (
                    team_ships_alive.get(ship.team_id, 0) + 1
                )

        our_ships_alive = team_ships_alive.get(team_id, 0)
        enemy_ships_alive = sum(
            count for tid, count in team_ships_alive.items() if tid != team_id
        )

        # Determine outcome
        if our_ships_alive > 0 and enemy_ships_alive == 0:
            return RewardConstants.VICTORY_REWARD  # Victory
        elif our_ships_alive == 0 and enemy_ships_alive > 0:
            return RewardConstants.DEFEAT_REWARD  # Defeat
        else:
            return (
                RewardConstants.DRAW_REWARD
            )  # Draw (both dead or timeout with survivors on both sides)

    def _check_termination(self, state: State) -> tuple[bool, dict[int, bool]]:
        """Check if episode should terminate and which agents are done"""
        alive_ships = [ship for ship in state.ships.values() if ship.alive]
        alive_teams = set(ship.team_id for ship in alive_ships)

        terminated = len(alive_teams) <= 1
        done = {ship_id: not ship.alive for ship_id, ship in state.ships.items()}

        return terminated, done

    def step(
        self, actions: dict[int, torch.Tensor]
    ) -> tuple[dict, dict[int, float], bool, bool, dict]:

        # Handle events if in human mode
        if self.render_mode == "human":
            if not self.renderer.handle_events():
                # User closed window - could handle this gracefully
                pass

        current_state = deepcopy(self.state[-1])
        current_state.time += self.agent_dt

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
            self._ship_actions(merged_actions, current_state)
            self._bullet_actions(current_state.bullets)
            self._ship_bullet_collisions(current_state.ships, current_state.bullets)
            self.current_time += self.physics_dt

            if self.render_mode == "human":
                self.render(current_state)

        # Save final state
        self.state.append(current_state)

        # Calculate termination
        terminated, done = self._check_termination(current_state)

        info = {
            "current_time": self.current_time,
            "active_bullets": current_state.bullets.num_active,
            "ship_states": {
                ship_id: ship.get_state()
                for ship_id, ship in current_state.ships.items()
            },
            "individual_done": done,
        }

        # Add human control info if in human mode
        if self.render_mode == "human":
            info["human_controlled"] = list(self.renderer.human_ship_ids)

        return self.get_observation(), {}, terminated, False, info

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

        current_state = self.state[-1]

        for ship_id, ship in current_state.ships.items():
            local_obs = ship.get_state()
            for key, value in local_obs.items():
                if key == "token":
                    # Token is already a tensor, store directly
                    observations[key][ship_id, :] = value
                else:
                    observations[key][ship_id, :] = torch.tensor(value)

        # Create tokens matrix for transformer model
        observations["tokens"] = observations["token"]

        return observations

    def _get_empty_observation(self) -> dict:
        """Empty observation for reset state"""
        return {
            "ship_id": torch.zeros((self.max_ships, 1), dtype=torch.int64),
            "team_id": torch.zeros((self.max_ships, 1), dtype=torch.int64),
            "alive": torch.zeros((self.max_ships, 1), dtype=torch.int64),
            "health": torch.zeros((self.max_ships, 1), dtype=torch.int64),
            "power": torch.zeros((self.max_ships, 1), dtype=torch.float32),
            "position": torch.zeros((self.max_ships, 1), dtype=torch.complex64),
            "velocity": torch.zeros((self.max_ships, 1), dtype=torch.complex64),
            "speed": torch.zeros((self.max_ships, 1), dtype=torch.float32),
            "attitude": torch.zeros((self.max_ships, 1), dtype=torch.complex64),
            "is_shooting": torch.zeros((self.max_ships, 1), dtype=torch.int64),
            "token": torch.zeros((self.max_ships, 10), dtype=torch.float32),
        }

    @property
    def action_space(self) -> spaces.Space:
        """Define the action space for each ship"""
        # Multi-discrete for binary actions: [forward, backward, left, right, sharp_turn, shoot]
        return spaces.MultiBinary(len(Actions))

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
