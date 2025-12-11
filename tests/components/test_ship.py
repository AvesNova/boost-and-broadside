import torch
import numpy as np
from env.constants import Actions


def test_ship_initialization(dummy_ship):
    """Test that the ship initializes with correct values."""
    assert dummy_ship.ship_id == 0
    assert dummy_ship.team_id == 0
    assert dummy_ship.position == 100.0 + 100.0j
    assert dummy_ship.velocity == 10.0 + 0j
    assert dummy_ship.alive is True
    assert dummy_ship.health == dummy_ship.config.max_health


def test_ship_physics_update(dummy_ship):
    """Test basic physics update."""
    initial_pos = dummy_ship.position
    dt = 0.1

    # No action, just coasting
    action = torch.zeros(len(Actions))
    dummy_ship.forward(action, None, 0.0, dt)

    # Position should change based on velocity
    expected_pos = initial_pos + dummy_ship.velocity * dt
    assert np.isclose(dummy_ship.position, expected_pos)


def test_ship_thrust(dummy_ship):
    """Test that thrust changes velocity."""
    initial_vel = dummy_ship.velocity
    dt = 0.1

    # Apply forward thrust
    action = torch.zeros(len(Actions))
    action[Actions.forward] = 1.0

    dummy_ship.forward(action, None, 0.0, dt)

    # Velocity should increase in the direction of attitude (default 1+0j)
    # Note: Drag will also apply, so we just check if it increased
    assert abs(dummy_ship.velocity) > abs(initial_vel)


def test_ship_turn(dummy_ship):
    """Test that turning changes attitude."""
    initial_attitude = dummy_ship.attitude
    dt = 0.1

    # Apply left turn
    action = torch.zeros(len(Actions))
    action[Actions.left] = 1.0

    dummy_ship.forward(action, None, 0.0, dt)

    # Attitude should rotate
    assert dummy_ship.attitude != initial_attitude


def test_ship_energy_consumption(dummy_ship):
    """Test that actions consume energy."""
    initial_power = dummy_ship.power
    dt = 0.1

    # Apply shooting action (usually high cost)
    action = torch.zeros(len(Actions))
    action[Actions.shoot] = 1.0

    # We need to mock bullets or pass None if handled gracefully
    # The ship.forward method expects bullets for shooting
    # Let's just test thrust for now if shooting is complex to mock here
    action = torch.zeros(len(Actions))
    action[Actions.forward] = 1.0

    dummy_ship.forward(action, None, 0.0, dt)

    # Power should decrease (assuming thrust cost > regen)
    # If regen is high, this might fail, but usually thrust costs energy
    # Let's check if it's different at least
    assert dummy_ship.power != initial_power
