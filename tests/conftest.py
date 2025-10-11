"""
Shared test fixtures and configuration for the ship game test suite.
"""

import pytest
import numpy as np
import torch

from src.ship import Ship, ShipConfig
from src.bullets import Bullets
from src.env import Environment
from src.state import State
from src.constants import Actions
from src.derive_ship_parameters import derive_ship_parameters


# --- Ship Derived Parameters Fixtures ---


@pytest.fixture
def derived_ship_parameters() -> dict:
    """Provides the default ship configuration."""
    return derive_ship_parameters()


# --- Ship Configuration Fixtures ---


@pytest.fixture
def default_ship_config() -> ShipConfig:
    """Provides the default ship configuration."""
    return ShipConfig()


@pytest.fixture
def minimal_ship_config() -> ShipConfig:
    """Provides a minimal ship config for faster testing."""
    default_config = ShipConfig()
    return ShipConfig(
        collision_radius=default_config.collision_radius / 2,
        max_health=default_config.max_health / 2,
        max_power=default_config.max_power / 2,
        base_thrust=default_config.base_thrust * 0.625,  # 5.0 / 8.0
        boost_thrust=default_config.boost_thrust * 0.5,
        reverse_thrust=default_config.reverse_thrust * 0.5,
        bullet_lifetime=default_config.bullet_lifetime * 0.5,
        firing_cooldown=default_config.firing_cooldown,
    )


@pytest.fixture
def zero_drag_ship_config() -> ShipConfig:
    """Ship config with no drag for predictable physics tests."""
    return ShipConfig(
        no_turn_drag_coeff=0.0,
        normal_turn_drag_coeff=0.0,
        sharp_turn_drag_coeff=0.0,
        normal_turn_lift_coeff=0.0,
        sharp_turn_lift_coeff=0.0,
    )


# --- Ship Fixtures ---


@pytest.fixture
def basic_ship(default_ship_config) -> Ship:
    """Creates a basic ship at origin moving right."""
    # Use relative positions based on a standard world size
    world_width, world_height = 800, 600
    return Ship(
        ship_id=0,
        team_id=0,
        ship_config=default_ship_config,
        initial_x=world_width * 0.5,  # Center x
        initial_y=world_height * 0.5,  # Center y
        initial_vx=100.0,  # Standard test velocity
        initial_vy=0.0,
        world_size=(world_width, world_height),
        rng=np.random.default_rng(42),
    )


@pytest.fixture
def stationary_ship_attempt():
    """Returns a function that attempts to create a stationary ship (should fail)."""

    def _create():
        world_width, world_height = 800, 600
        return Ship(
            ship_id=0,
            team_id=0,
            ship_config=ShipConfig(),
            initial_x=world_width * 0.5,
            initial_y=world_height * 0.5,
            initial_vx=0.0,
            initial_vy=0.0,
        )

    return _create


@pytest.fixture
def two_ships(default_ship_config) -> tuple[Ship, Ship]:
    """Creates two opposing ships."""
    # Use relative positions based on a standard world size
    world_width, world_height = 800, 600
    ship1 = Ship(
        ship_id=0,
        team_id=0,
        ship_config=default_ship_config,
        initial_x=world_width * 0.25,  # Left quarter
        initial_y=world_height * 0.5,  # Center y
        initial_vx=100.0,
        initial_vy=0.0,
        world_size=(world_width, world_height),
        rng=np.random.default_rng(42),
    )
    ship2 = Ship(
        ship_id=1,
        team_id=1,
        ship_config=default_ship_config,
        initial_x=world_width * 0.75,  # Right quarter
        initial_y=world_height * 0.5,  # Center y
        initial_vx=-100.0,
        initial_vy=0.0,
        world_size=(world_width, world_height),
        rng=np.random.default_rng(43),
    )
    return ship1, ship2


# --- Bullet System Fixtures ---


@pytest.fixture
def empty_bullets() -> Bullets:
    """Creates an empty bullet container."""
    return Bullets(max_bullets=10)


@pytest.fixture
def bullets_with_some() -> Bullets:
    """Creates a bullet container with some bullets."""
    bullets = Bullets(max_bullets=10)
    bullets.add_bullet(0, 100.0, 100.0, 50.0, 0.0, 1.0)
    bullets.add_bullet(1, 200.0, 200.0, -50.0, 50.0, 0.5)
    bullets.add_bullet(0, 300.0, 100.0, 0.0, -50.0, 0.75)
    return bullets


# --- Environment Fixtures ---


@pytest.fixture
def basic_env() -> Environment:
    """Creates a basic environment for testing."""
    return Environment(
        render_mode=None,
        world_size=(800, 600),
        memory_size=1,
        max_ships=2,
        agent_dt=0.02,
        physics_dt=0.02,
    )


@pytest.fixture
def env_with_substeps() -> Environment:
    """Environment with multiple physics substeps per agent step."""
    return Environment(
        render_mode=None,
        world_size=(800, 600),
        memory_size=1,
        max_ships=2,
        agent_dt=0.04,  # 2x physics_dt
        physics_dt=0.02,
    )


# --- Action Fixtures ---


@pytest.fixture
def no_action() -> torch.Tensor:
    """Returns a zero action tensor."""
    return torch.zeros(len(Actions))


@pytest.fixture
def forward_action() -> torch.Tensor:
    """Returns an action tensor for moving forward."""
    action = torch.zeros(len(Actions))
    action[Actions.forward] = 1
    return action


@pytest.fixture
def all_actions() -> dict[str, torch.Tensor]:
    """Returns a dictionary of all individual actions."""
    actions = {}
    for action_name in Actions:
        action = torch.zeros(len(Actions))
        action[action_name] = 1
        actions[action_name.name] = action
    return actions


@pytest.fixture
def action_combinations() -> dict[str, torch.Tensor]:
    """Returns common action combinations."""
    combinations = {}

    # Single actions
    for action_name in Actions:
        action = torch.zeros(len(Actions))
        action[action_name] = 1
        combinations[action_name.name] = action

    # Common combinations
    action = torch.zeros(len(Actions))
    action[Actions.forward] = 1
    action[Actions.left] = 1
    combinations["forward_left"] = action

    action = torch.zeros(len(Actions))
    action[Actions.forward] = 1
    action[Actions.right] = 1
    combinations["forward_right"] = action

    action = torch.zeros(len(Actions))
    action[Actions.left] = 1
    action[Actions.right] = 1
    combinations["left_right"] = action

    action = torch.zeros(len(Actions))
    action[Actions.sharp_turn] = 1
    action[Actions.left] = 1
    combinations["sharp_left"] = action

    action = torch.zeros(len(Actions))
    action[Actions.forward] = 1
    action[Actions.shoot] = 1
    combinations["forward_shoot"] = action

    return combinations


# --- State Fixtures ---


@pytest.fixture
def empty_state() -> State:
    """Creates an empty state."""
    return State(ships={})


@pytest.fixture
def combat_state(two_ships) -> State:
    """Creates a state with two ships ready for combat."""
    ship1, ship2 = two_ships
    return State(ships={0: ship1, 1: ship2})


# --- Random Number Generator Fixtures ---


@pytest.fixture
def fixed_rng() -> np.random.Generator:
    """Provides a fixed-seed RNG for deterministic tests."""
    return np.random.default_rng(12345)


# --- Tolerance Fixtures ---


@pytest.fixture
def physics_tolerance() -> float:
    """Tolerance for physics calculations."""
    return 1e-5


@pytest.fixture
def loose_tolerance() -> float:
    """Looser tolerance for aggregate calculations."""
    return 1e-3


# --- Helper Function Fixtures ---


@pytest.fixture
def step_environment():
    """Returns a function to step environment multiple times."""

    def _step(env: Environment, actions: dict, n_steps: int = 1):
        observations = []
        rewards = []
        infos = []

        for _ in range(n_steps):
            obs, rew, term, trunc, info = env.step(actions)
            observations.append(obs)
            rewards.append(rew)
            infos.append(info)

            if term or trunc:
                break

        return observations, rewards, infos

    return _step


@pytest.fixture
def assert_complex_close():
    """Returns a function to assert complex numbers are close."""

    def _assert(actual: complex, expected: complex, tolerance: float = 1e-5):
        assert (
            abs(actual.real - expected.real) < tolerance
        ), f"Real part mismatch: {actual.real} != {expected.real}"
        assert (
            abs(actual.imag - expected.imag) < tolerance
        ), f"Imaginary part mismatch: {actual.imag} != {expected.imag}"

    return _assert


@pytest.fixture
def assert_vector_close():
    """Returns a function to assert numpy arrays are close."""

    def _assert(actual: np.ndarray, expected: np.ndarray, tolerance: float = 1e-5):
        np.testing.assert_allclose(actual, expected, rtol=tolerance, atol=tolerance)

    return _assert
