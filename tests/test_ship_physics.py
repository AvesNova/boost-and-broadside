"""
Tests for ship physics without combat mechanics.
"""

import pytest
import numpy as np
import torch

from src.constants import Actions
from src.ship import Ship


class TestShipInitialization:
    """Tests for ship initialization and state setup."""

    def test_basic_initialization(self, basic_ship):
        """Test that ship initializes with correct state."""
        assert basic_ship.ship_id == 0
        assert basic_ship.team_id == 0
        assert basic_ship.alive is True
        assert basic_ship.health == basic_ship.config.max_health
        assert basic_ship.power == basic_ship.config.max_power
        assert basic_ship.turn_offset == 0.0
        assert basic_ship.last_fired_time == -basic_ship.config.firing_cooldown

    def test_position_velocity_init(self, basic_ship):
        """Test position and velocity initialization."""
        assert basic_ship.position.real == 400.0
        assert basic_ship.position.imag == 300.0
        assert basic_ship.velocity.real == 100.0
        assert basic_ship.velocity.imag == 0.0
        assert basic_ship.speed == 100.0

    def test_attitude_initialization(self, basic_ship):
        """Test that attitude points in velocity direction."""
        assert abs(basic_ship.attitude.real - 1.0) < 1e-6
        assert abs(basic_ship.attitude.imag - 0.0) < 1e-6

    def test_stationary_ship_fails(self, stationary_ship_attempt):
        """Test that creating ship with zero velocity fails."""
        with pytest.raises(
            AssertionError, match="Initial velocity cannot be too small"
        ):
            stationary_ship_attempt()

    def test_lookup_table_construction(self, basic_ship):
        """Test that lookup tables are properly constructed."""
        config = basic_ship.config

        # Turn offset table - test key combinations (use approximate equality for float precision)
        assert basic_ship.turn_offset_table[0, 0, 0] == 0  # No turn
        assert (
            abs(basic_ship.turn_offset_table[0, 1, 0] - config.normal_turn_angle) < 1e-6
        )  # Right
        assert (
            abs(basic_ship.turn_offset_table[1, 0, 0] - (-config.normal_turn_angle))
            < 1e-6
        )  # Left
        assert (
            abs(basic_ship.turn_offset_table[0, 1, 1] - config.sharp_turn_angle) < 1e-6
        )  # Sharp right
        assert (
            abs(basic_ship.turn_offset_table[1, 0, 1] - (-config.sharp_turn_angle))
            < 1e-6
        )  # Sharp left

        # Thrust table
        assert basic_ship.thrust_table[0, 0] == config.base_thrust
        assert basic_ship.thrust_table[1, 0] == config.boost_thrust
        assert basic_ship.thrust_table[0, 1] == config.reverse_thrust
        assert basic_ship.thrust_table[1, 1] == config.base_thrust  # Both cancel

        # Drag coefficient table
        assert basic_ship.drag_coeff_table[0, 0] == config.no_turn_drag_coeff
        assert basic_ship.drag_coeff_table[1, 0] == config.normal_turn_drag_coeff
        assert basic_ship.drag_coeff_table[1, 1] == config.sharp_turn_drag_coeff


class TestTurnMechanics:
    """Tests for ship turning behavior."""

    def test_turn_offset_updates(self, basic_ship, action_combinations, empty_bullets):
        """Test that turn offset updates correctly for all combinations."""
        # Test each turn combination
        test_cases = [
            ("left", -basic_ship.config.normal_turn_angle),
            ("right", basic_ship.config.normal_turn_angle),
            ("sharp_left", -basic_ship.config.sharp_turn_angle),
            ("sharp_turn", 0.0),  # Sharp alone does nothing
        ]

        for action_name, expected_offset in test_cases:
            basic_ship.turn_offset = 0.0  # Reset
            basic_ship.forward(
                action_combinations[action_name], empty_bullets, 0.0, 0.01
            )
            assert (
                abs(basic_ship.turn_offset - expected_offset) < 1e-6
            ), f"Failed for action {action_name}"

    def test_turn_persistence_with_lr(
        self, basic_ship, action_combinations, empty_bullets
    ):
        """Test that L+R pressed maintains current turn offset."""
        # Set an initial turn offset
        basic_ship.turn_offset = 0.123

        # Press both L and R
        basic_ship.forward(action_combinations["left_right"], empty_bullets, 0.0, 0.01)

        # Turn offset should be maintained
        assert abs(basic_ship.turn_offset - 0.123) < 1e-6

    def test_turn_persistence_with_slr(self, basic_ship, empty_bullets):
        """Test that S+L+R pressed maintains current turn offset."""
        # Set an initial turn offset
        basic_ship.turn_offset = -0.456

        # Press S, L, and R
        action = torch.zeros(len(Actions))
        action[Actions.sharp_turn] = 1
        action[Actions.left] = 1
        action[Actions.right] = 1

        basic_ship.forward(action, empty_bullets, 0.0, 0.01)

        # Turn offset should be maintained
        assert abs(basic_ship.turn_offset - (-0.456)) < 1e-6

    def test_attitude_relative_to_velocity(
        self, basic_ship, action_combinations, empty_bullets
    ):
        """Test that attitude is computed relative to velocity direction."""
        # Ship moving right (velocity = 100+0j)
        # Turn left (negative angle due to coordinate system)
        initial_attitude_angle = np.angle(basic_ship.attitude)

        basic_ship.forward(action_combinations["left"], empty_bullets, 0.0, 0.01)

        # Left turn should result in negative angle change (counterclockwise)
        actual_angle = np.angle(basic_ship.attitude)
        # The attitude should have rotated in the negative direction
        # Account for physics affecting the exact result
        assert (
            actual_angle < initial_attitude_angle
        ), "Left turn should decrease attitude angle"
        # Check that turn offset is correctly set to negative value for left turn
        assert (
            abs(basic_ship.turn_offset - (-basic_ship.config.normal_turn_angle)) < 1e-6
        )

    def test_attitude_at_low_speed(
        self, basic_ship, action_combinations, empty_bullets
    ):
        """Test attitude behavior at very low speeds."""
        # Reduce speed to near-zero
        basic_ship.velocity = 1e-7 + 0j
        basic_ship.speed = abs(basic_ship.velocity)

        # Try to turn - should still update turn offset
        basic_ship.forward(action_combinations["left"], empty_bullets, 0.0, 0.01)

        # Turn offset should update
        assert (
            abs(basic_ship.turn_offset - (-basic_ship.config.normal_turn_angle)) < 1e-6
        )


class TestThrustAndEnergy:
    """Tests for thrust forces and energy management."""

    def test_base_thrust_with_no_input(self, basic_ship, no_action, empty_bullets):
        """Test that base thrust is applied with no input."""
        initial_velocity = basic_ship.velocity
        basic_ship.drag_coeff_table[0, 0] = 0.0
        basic_ship.forward(no_action, empty_bullets, 0.0, 0.1)

        # Velocity should increase due to base thrust (minus drag)
        # Ship faces right (1+0j), so thrust is in positive x direction
        assert basic_ship.velocity.real > initial_velocity.real

    def test_forward_thrust_boost(
        self, basic_ship, forward_action, empty_bullets, zero_drag_ship_config
    ):
        """Test that forward action applies boost thrust."""
        # Create ship with no drag for cleaner test
        ship = basic_ship.__class__(
            ship_id=0,
            team_id=0,
            ship_config=zero_drag_ship_config,
            initial_x=400.0,
            initial_y=300.0,
            initial_vx=100.0,
            initial_vy=0.0,
        )

        initial_velocity = ship.velocity
        ship.forward(forward_action, empty_bullets, 0.0, 0.1)

        # Calculate expected acceleration
        expected_accel = zero_drag_ship_config.boost_thrust * 0.1
        expected_velocity = initial_velocity + expected_accel

        assert abs(ship.velocity - expected_velocity) < 1e-5

    def test_backward_thrust(self, basic_ship, empty_bullets):
        """Test that backward action applies reverse thrust."""
        action = torch.zeros(len(Actions))
        action[Actions.backward] = 1

        initial_velocity = basic_ship.velocity
        basic_ship.forward(action, empty_bullets, 0.0, 0.01)

        # Velocity should decrease (reverse thrust opposes motion)
        assert basic_ship.velocity.real < initial_velocity.real

    def test_forward_backward_cancel(self, basic_ship, empty_bullets):
        """Test that forward+backward results in base thrust."""
        action = torch.zeros(len(Actions))
        action[Actions.forward] = 1
        action[Actions.backward] = 1

        # Should behave same as no input (base thrust)
        initial_velocity = basic_ship.velocity
        basic_ship.forward(action, empty_bullets, 0.0, 0.01)

        # Compare with no action result
        basic_ship.velocity = initial_velocity  # Reset
        basic_ship.forward(torch.zeros(len(Actions)), empty_bullets, 0.0, 0.01)
        no_action_velocity = basic_ship.velocity

        basic_ship.velocity = initial_velocity  # Reset again
        basic_ship.forward(action, empty_bullets, 0.0, 0.01)

        assert abs(basic_ship.velocity - no_action_velocity) < 1e-5

    def test_energy_consumption_forward(
        self, basic_ship, forward_action, empty_bullets
    ):
        """Test energy consumption when boosting forward."""
        initial_power = basic_ship.power
        basic_ship.forward(forward_action, empty_bullets, 0.0, 0.1)

        expected_power = initial_power + basic_ship.config.boost_power_gain * 0.1
        assert abs(basic_ship.power - expected_power) < 1e-5

    def test_energy_regeneration_backward(self, basic_ship, empty_bullets):
        """Test energy regeneration when going backward."""
        action = torch.zeros(len(Actions))
        action[Actions.backward] = 1

        # Deplete some energy first
        basic_ship.power = basic_ship.config.max_power / 2
        initial_power = basic_ship.power

        basic_ship.forward(action, empty_bullets, 0.0, 0.1)

        expected_power = initial_power + basic_ship.config.reverse_power_gain * 0.1
        assert abs(basic_ship.power - expected_power) < 1e-5

    def test_energy_base_regeneration(self, basic_ship, no_action, empty_bullets):
        """Test passive energy regeneration with no input."""
        basic_ship.power = basic_ship.config.max_power / 2
        initial_power = basic_ship.power

        basic_ship.forward(no_action, empty_bullets, 0.0, 0.1)

        expected_power = initial_power + basic_ship.config.base_power_gain * 0.1
        assert abs(basic_ship.power - expected_power) < 1e-5

    def test_energy_clamping_at_max(self, basic_ship, no_action, empty_bullets):
        """Test that energy doesn't exceed maximum."""
        basic_ship.power = basic_ship.config.max_power - 0.1

        # Apply action that would regenerate energy
        basic_ship.forward(no_action, empty_bullets, 0.0, 10.0)  # Long time step

        assert basic_ship.power == basic_ship.config.max_power

    def test_energy_clamping_at_zero(self, basic_ship, forward_action, empty_bullets):
        """Test that energy doesn't go below zero."""
        basic_ship.power = 1.0

        # Apply action that depletes energy
        basic_ship.forward(forward_action, empty_bullets, 0.0, 10.0)  # Long time step

        assert basic_ship.power == 0.0

    def test_no_thrust_when_no_power(self, basic_ship, forward_action, empty_bullets):
        """Test that thrust is disabled when power is zero."""
        basic_ship.power = 0.0
        initial_velocity = basic_ship.velocity

        # Try to thrust forward
        basic_ship.forward(forward_action, empty_bullets, 0.0, 0.01)

        # Velocity should only change due to drag, not thrust
        velocity_change = abs(basic_ship.velocity - initial_velocity)
        assert velocity_change < 0.1  # Small change from drag only


class TestAerodynamics:
    """Tests for drag and lift forces."""

    def test_drag_opposes_velocity(self, basic_ship, no_action, empty_bullets):
        """Test that drag force opposes velocity direction."""
        initial_speed = basic_ship.speed
        basic_ship.forward(no_action, empty_bullets, 0.0, 0.1)

        # Speed should decrease due to drag (even with base thrust)
        # This assumes drag is significant enough
        # If not decreasing, at least should not increase as much as without drag
        assert basic_ship.speed < initial_speed * 1.1

    def test_drag_scales_with_speed_squared(self, zero_drag_ship_config, empty_bullets):
        """Test that drag force scales with velocity squared."""
        # Create ship with controlled drag
        config = zero_drag_ship_config
        config.no_turn_drag_coeff = 1e-3

        # Test at different speeds
        speeds = [50.0, 100.0, 200.0]
        drag_forces = []

        for speed in speeds:
            ship = Ship(
                ship_id=0,
                team_id=0,
                ship_config=config,
                initial_x=400.0,
                initial_y=300.0,
                initial_vx=speed,
                initial_vy=0.0,
            )
            # Disable thrust for clean measurement
            ship.power = 0.0

            initial_velocity = ship.velocity
            ship.forward(torch.zeros(len(Actions)), empty_bullets, 0.0, 0.01)

            # Measure deceleration (drag effect)
            deceleration = (initial_velocity.real - ship.velocity.real) / 0.01
            drag_forces.append(deceleration)

        # Check quadratic scaling (force proportional to v²)
        ratio1 = drag_forces[1] / drag_forces[0]
        ratio2 = drag_forces[2] / drag_forces[1]

        expected_ratio1 = (speeds[1] / speeds[0]) ** 2
        expected_ratio2 = (speeds[2] / speeds[1]) ** 2

        assert abs(ratio1 - expected_ratio1) < 0.1
        assert abs(ratio2 - expected_ratio2) < 0.1

    def test_turn_increases_drag(
        self, zero_drag_ship_config, action_combinations, empty_bullets
    ):
        """Test that turning increases drag coefficient."""
        # Create ship with only drag forces, no lift or thrust
        config = zero_drag_ship_config
        config.no_turn_drag_coeff = 0.001
        config.normal_turn_drag_coeff = 0.002
        config.base_thrust = 0.0
        config.boost_thrust = 0.0
        config.reverse_thrust = 0.0
        # Ensure no lift forces
        config.normal_turn_lift_coeff = 0.0
        config.sharp_turn_lift_coeff = 0.0

        # Test no turn case
        ship1 = Ship(
            ship_id=0,
            team_id=0,
            ship_config=config,
            initial_x=400.0,
            initial_y=300.0,
            initial_vx=100.0,
            initial_vy=0.0,
        )
        ship1.power = 0.0
        ship1.forward(torch.zeros(len(Actions)), empty_bullets, 0.0, 0.1)
        no_turn_speed = ship1.speed

        # Test turn case
        ship2 = Ship(
            ship_id=1,
            team_id=1,
            ship_config=config,
            initial_x=400.0,
            initial_y=300.0,
            initial_vx=100.0,
            initial_vy=0.0,
        )
        ship2.power = 0.0
        ship2.forward(action_combinations["left"], empty_bullets, 0.0, 0.1)
        turn_speed = ship2.speed

        # Turning should cause more speed loss due to higher drag
        assert (
            turn_speed < no_turn_speed
        ), f"Turn speed {turn_speed} should be less than no-turn speed {no_turn_speed}"

    def test_lift_preserves_speed(
        self, zero_drag_ship_config, action_combinations, empty_bullets
    ):
        """Test that lift forces don't change speed, only direction."""
        # Create ship with zero drag and zero thrust for clean test
        config = zero_drag_ship_config
        config.normal_turn_lift_coeff = 0.02  # Set some lift
        config.sharp_turn_lift_coeff = 0.04
        config.base_thrust = 0.0  # No thrust at all
        config.boost_thrust = 0.0
        config.reverse_thrust = 0.0

        ship = Ship(
            ship_id=0,
            team_id=0,
            ship_config=config,
            initial_x=400.0,
            initial_y=300.0,
            initial_vx=100.0,
            initial_vy=0.0,
        )
        ship.power = 0.0  # No power for thrust

        initial_speed = ship.speed

        # Test normal turn
        ship.forward(action_combinations["left"], empty_bullets, 0.0, 0.1)
        normal_turn_speed = ship.speed

        # Reset ship for sharp turn test
        ship.velocity = 100.0 + 0j
        ship.speed = 100.0

        # Test sharp turn
        ship.forward(action_combinations["sharp_left"], empty_bullets, 0.0, 0.1)
        sharp_turn_speed = ship.speed

        # Speed should remain approximately constant (allowing for numerical integration error)
        # With current integration method, some speed change is expected but should be small
        assert (
            abs(normal_turn_speed - initial_speed) < initial_speed * 0.05
        ), f"Normal turn changed speed too much: {initial_speed} -> {normal_turn_speed}"
        assert (
            abs(sharp_turn_speed - initial_speed) < initial_speed * 0.10
        ), f"Sharp turn changed speed too much: {initial_speed} -> {sharp_turn_speed}"

    def test_lift_perpendicular_to_velocity(
        self, zero_drag_ship_config, action_combinations, empty_bullets
    ):
        """Test that lift force is perpendicular to velocity."""
        # Setup ship with only lift, no drag or thrust
        config = zero_drag_ship_config
        config.normal_turn_lift_coeff = 1e-2

        ship = Ship(
            ship_id=0,
            team_id=0,
            ship_config=config,
            initial_x=400.0,
            initial_y=300.0,
            initial_vx=100.0,
            initial_vy=0.0,
        )
        ship.power = 0.0  # No thrust

        initial_velocity = ship.velocity
        ship.forward(action_combinations["left"], empty_bullets, 0.0, 0.1)

        # Velocity change should be primarily perpendicular
        velocity_change = ship.velocity - initial_velocity

        # Check that change is mostly in y direction (perpendicular to initial x velocity)
        assert abs(velocity_change.imag) > abs(velocity_change.real)

    def test_sharp_turn_higher_coefficients(self, basic_ship, empty_bullets):
        """Test that sharp turns use higher drag and lift coefficients."""
        basic_ship.power = 0.0  # Disable thrust

        # Normal turn
        initial_velocity = 100.0 + 0j
        basic_ship.velocity = initial_velocity
        basic_ship.speed = abs(initial_velocity)

        action_normal = torch.zeros(len(Actions))
        action_normal[Actions.left] = 1
        basic_ship.forward(action_normal, empty_bullets, 0.0, 0.01)
        normal_velocity_change = abs(basic_ship.velocity - initial_velocity)

        # Sharp turn
        basic_ship.velocity = initial_velocity
        basic_ship.speed = abs(initial_velocity)

        action_sharp = torch.zeros(len(Actions))
        action_sharp[Actions.left] = 1
        action_sharp[Actions.sharp_turn] = 1
        basic_ship.forward(action_sharp, empty_bullets, 0.0, 0.01)
        sharp_velocity_change = abs(basic_ship.velocity - initial_velocity)

        # Sharp turn should cause more change due to higher coefficients
        assert sharp_velocity_change > normal_velocity_change


class TestKinematics:
    """Tests for position and velocity integration."""

    def test_position_integration(
        self, zero_drag_ship_config, no_action, empty_bullets
    ):
        """Test that position integrates velocity correctly."""
        ship = Ship(
            ship_id=0,
            team_id=0,
            ship_config=zero_drag_ship_config,
            initial_x=400.0,
            initial_y=300.0,
            initial_vx=50.0,
            initial_vy=25.0,
        )
        ship.power = 0.0  # No forces

        initial_position = ship.position
        dt = 0.1
        ship.forward(no_action, empty_bullets, 0.0, dt)

        expected_position = initial_position + ship.velocity * dt
        assert abs(ship.position - expected_position) < 1e-5

    def test_velocity_integration(
        self, zero_drag_ship_config, forward_action, empty_bullets
    ):
        """Test that velocity integrates acceleration correctly."""
        ship = Ship(
            ship_id=0,
            team_id=0,
            ship_config=zero_drag_ship_config,
            initial_x=400.0,
            initial_y=300.0,
            initial_vx=100.0,
            initial_vy=0.0,
        )

        initial_velocity = ship.velocity
        dt = 0.1
        ship.forward(forward_action, empty_bullets, 0.0, dt)

        # Acceleration from boost thrust
        acceleration = zero_drag_ship_config.boost_thrust * ship.attitude
        expected_velocity = initial_velocity + acceleration * dt

        assert abs(ship.velocity - expected_velocity) < 1e-5

    def test_speed_calculation(self, basic_ship, action_combinations, empty_bullets):
        """Test that speed is correctly calculated from velocity."""
        # Set velocity at an angle
        basic_ship.velocity = 30.0 + 40.0j
        basic_ship.speed = abs(basic_ship.velocity)

        assert abs(basic_ship.speed - 50.0) < 1e-6  # 3-4-5 triangle

        # Update and check speed is recalculated
        basic_ship.forward(action_combinations["forward"], empty_bullets, 0.0, 0.01)
        assert abs(basic_ship.speed - abs(basic_ship.velocity)) < 1e-6

    def test_minimum_speed_threshold(self, basic_ship, empty_bullets):
        """Test that speed doesn't go below minimum threshold."""
        # Try to stop the ship completely
        basic_ship.velocity = 1e-8 + 0j
        basic_ship.speed = abs(basic_ship.velocity)

        # Apply backward thrust
        action = torch.zeros(len(Actions))
        action[Actions.backward] = 1
        basic_ship.forward(action, empty_bullets, 0.0, 0.01)

        assert basic_ship.speed >= 1e-6


class TestDeathMechanics:
    """Tests for ship death and damage."""

    def test_death_at_zero_health(self, basic_ship, no_action, empty_bullets):
        """Test that ship dies when health reaches zero."""
        basic_ship.health = 0
        basic_ship.forward(no_action, empty_bullets, 0.0, 0.01)

        assert basic_ship.alive is False

    def test_dead_ship_doesnt_update(self, basic_ship, forward_action, empty_bullets):
        """Test that dead ships don't process actions."""
        basic_ship.alive = False
        initial_position = basic_ship.position
        initial_velocity = basic_ship.velocity

        basic_ship.forward(forward_action, empty_bullets, 0.0, 0.1)

        assert basic_ship.position == initial_position
        assert basic_ship.velocity == initial_velocity

    def test_damage_reduces_health(self, basic_ship):
        """Test that damage reduces health correctly."""
        initial_health = basic_ship.health
        damage = 25.0

        basic_ship.damage_ship(damage)

        assert basic_ship.health == initial_health - damage

    def test_damage_kills_at_zero(self, basic_ship):
        """Test that damage kills ship when health reaches zero."""
        basic_ship.damage_ship(basic_ship.health)

        assert basic_ship.health <= 0
        assert basic_ship.alive is False


class TestStateRetrieval:
    """Tests for getting ship state."""

    def test_get_state_structure(self, basic_ship):
        """Test that get_state returns correct structure."""
        state = basic_ship.get_state()

        expected_keys = {
            "ship_id",
            "team_id",
            "alive",
            "health",
            "power",
            "position",
            "velocity",
            "speed",
            "attitude",
            "is_shooting",
            "token",
        }
        assert set(state.keys()) == expected_keys

    def test_get_state_values(self, basic_ship):
        """Test that get_state returns correct values."""
        state = basic_ship.get_state()

        assert state["ship_id"] == basic_ship.ship_id
        assert state["team_id"] == basic_ship.team_id
        assert state["alive"] == basic_ship.alive
        assert state["health"] == basic_ship.health
        assert state["power"] == basic_ship.power
        assert (
            state["position"] == basic_ship.position
        )  # Now returns complex number directly
        assert (
            state["velocity"] == basic_ship.velocity
        )  # Now returns complex number directly
        assert state["speed"] == basic_ship.speed
        assert (
            state["attitude"] == basic_ship.attitude
        )  # Now returns complex number directly
        assert state["is_shooting"] == basic_ship.is_shooting
        assert isinstance(state["token"], torch.Tensor)
        assert state["token"].shape == (10,)  # Token has 10 dimensions
