"""
Tests for ship combat mechanics including shooting and bullet management.
"""

import pytest
import numpy as np
import torch

from src.constants import Actions
from src.bullets import Bullets


class TestShooting:
    """Tests for shooting mechanics."""

    def test_basic_shooting(self, basic_ship, empty_bullets):
        """Test that shooting creates a bullet."""
        action = torch.zeros(len(Actions))
        action[Actions.shoot] = 1

        basic_ship.forward(action, empty_bullets, current_time=1.0, delta_t=0.01)

        assert empty_bullets.num_active == 1
        assert basic_ship.is_shooting is True

    def test_shooting_depletes_energy(self, basic_ship, empty_bullets):
        """Test that shooting consumes energy."""
        action = torch.zeros(len(Actions))
        action[Actions.shoot] = 1

        initial_power = basic_ship.power
        basic_ship.forward(action, empty_bullets, current_time=1.0, delta_t=0.01)

        expected_power = initial_power - basic_ship.config.bullet_energy_cost
        # Account for base power gain during the timestep
        expected_power += basic_ship.config.base_power_gain * 0.01

        assert abs(basic_ship.power - expected_power) < 1e-5

    def test_shooting_respects_cooldown(self, basic_ship, empty_bullets):
        """Test that shooting respects cooldown period."""
        action = torch.zeros(len(Actions))
        action[Actions.shoot] = 1

        # First shot
        basic_ship.forward(action, empty_bullets, current_time=0.0, delta_t=0.01)
        assert empty_bullets.num_active == 1

        # Try to shoot again immediately
        basic_ship.forward(action, empty_bullets, current_time=0.01, delta_t=0.01)
        assert empty_bullets.num_active == 1  # Still just one bullet

        # Wait for cooldown and try again
        cooldown_time = basic_ship.config.firing_cooldown
        basic_ship.forward(
            action, empty_bullets, current_time=cooldown_time + 0.01, delta_t=0.01
        )
        assert empty_bullets.num_active == 2  # Now two bullets

    def test_shooting_without_energy(self, basic_ship, empty_bullets):
        """Test that shooting fails without sufficient energy."""
        action = torch.zeros(len(Actions))
        action[Actions.shoot] = 1

        # Deplete energy
        basic_ship.power = basic_ship.config.bullet_energy_cost - 0.1

        basic_ship.forward(action, empty_bullets, current_time=1.0, delta_t=0.01)

        assert empty_bullets.num_active == 0
        assert basic_ship.is_shooting is False

    def test_bullet_initial_position(self, basic_ship, empty_bullets):
        """Test that bullets spawn at ship position."""
        action = torch.zeros(len(Actions))
        action[Actions.shoot] = 1

        # Capture ship position before shooting (since bullet is created before kinematics update)
        initial_position = basic_ship.position
        basic_ship.forward(action, empty_bullets, current_time=1.0, delta_t=0.01)

        assert abs(empty_bullets.x[0] - initial_position.real) < 1e-5
        assert abs(empty_bullets.y[0] - initial_position.imag) < 1e-5

    def test_bullet_velocity_includes_ship_velocity(self, basic_ship, empty_bullets):
        """Test that bullet velocity includes ship velocity."""
        action = torch.zeros(len(Actions))
        action[Actions.shoot] = 1

        # Set ship velocity
        basic_ship.velocity = 50.0 + 25.0j
        basic_ship.speed = abs(basic_ship.velocity)

        # Capture velocity and attitude before shooting (since bullet is created before kinematics update)
        initial_velocity = basic_ship.velocity
        initial_attitude = basic_ship.attitude

        basic_ship.forward(action, empty_bullets, current_time=1.0, delta_t=0.01)

        # Bullet velocity should be ship velocity plus bullet speed in attitude direction
        expected_vx = (
            initial_velocity.real
            + basic_ship.config.bullet_speed * initial_attitude.real
        )
        expected_vy = (
            initial_velocity.imag
            + basic_ship.config.bullet_speed * initial_attitude.imag
        )

        # Account for spread randomness
        assert (
            abs(empty_bullets.vx[0] - expected_vx) < basic_ship.config.bullet_spread * 3
        )
        assert (
            abs(empty_bullets.vy[0] - expected_vy) < basic_ship.config.bullet_spread * 3
        )

    def test_bullet_spread(self, basic_ship):
        """Test that bullets have random spread."""
        # Create multiple bullets and check spread
        bullets = Bullets(max_bullets=20)
        action = torch.zeros(len(Actions))
        action[Actions.shoot] = 1

        # Fire multiple bullets
        velocities = []
        for i in range(10):
            bullets_temp = Bullets(max_bullets=1)
            basic_ship.last_fired_time = -1.0  # Reset cooldown
            basic_ship.forward(
                action, bullets_temp, current_time=float(i), delta_t=0.01
            )
            if bullets_temp.num_active > 0:
                velocities.append((bullets_temp.vx[0], bullets_temp.vy[0]))

        # Check that velocities are different (spread is working)
        unique_velocities = set(velocities)
        assert len(unique_velocities) > 1

    def test_bullet_lifetime_setting(self, basic_ship, empty_bullets):
        """Test that bullets are created with correct lifetime."""
        action = torch.zeros(len(Actions))
        action[Actions.shoot] = 1

        basic_ship.forward(action, empty_bullets, current_time=1.0, delta_t=0.01)

        assert (
            abs(empty_bullets.time_remaining[0] - basic_ship.config.bullet_lifetime)
            < 1e-5
        )

    def test_bullet_ship_id_tracking(self, basic_ship, empty_bullets):
        """Test that bullets track their originating ship."""
        action = torch.zeros(len(Actions))
        action[Actions.shoot] = 1

        basic_ship.forward(action, empty_bullets, current_time=1.0, delta_t=0.01)

        assert empty_bullets.ship_id[0] == basic_ship.ship_id

    def test_not_shooting_when_not_pressed(self, basic_ship, empty_bullets, no_action):
        """Test that is_shooting is false when shoot not pressed."""
        basic_ship.forward(no_action, empty_bullets, current_time=1.0, delta_t=0.01)

        assert basic_ship.is_shooting is False
        assert empty_bullets.num_active == 0


class TestCombatIntegration:
    """Integration tests for combat scenarios."""

    def test_rapid_fire_sequence(self, basic_ship, empty_bullets):
        """Test rapid firing sequence respecting cooldowns."""
        action = torch.zeros(len(Actions))
        action[Actions.shoot] = 1

        time = 0.0
        dt = 0.01
        expected_bullets = 0

        # Simulate 1 second of continuous firing
        while time < 1.0:
            basic_ship.forward(action, empty_bullets, current_time=time, delta_t=dt)
            time += dt

            # Check if we should have fired
            if time >= expected_bullets * basic_ship.config.firing_cooldown:
                expected_bullets += 1

        # Should have fired approximately 1.0 / cooldown times
        expected = int(1.0 / basic_ship.config.firing_cooldown)
        assert abs(empty_bullets.num_active - expected) <= 1

    def test_shooting_while_moving(self, basic_ship, empty_bullets):
        """Test shooting while performing maneuvers."""
        action = torch.zeros(len(Actions))
        action[Actions.forward] = 1
        action[Actions.left] = 1
        action[Actions.shoot] = 1

        initial_position = basic_ship.position
        basic_ship.forward(action, empty_bullets, current_time=1.0, delta_t=0.1)

        # Ship should have moved
        assert abs(basic_ship.position - initial_position) > 1.0

        # Bullet should have been fired
        assert empty_bullets.num_active == 1

        # Ship should be turning
        assert abs(basic_ship.turn_offset) > 0

    def test_energy_management_combat(self, basic_ship, empty_bullets):
        """Test energy management during combat."""
        # Set power to barely enough for one shot (accounting for base power gain)
        # bullet_energy_cost = 3.0, base_power_gain = 10.0, dt = 0.01
        # After shooting: power - bullet_cost + base_gain * dt = power - 3.0 + 0.1
        # To have insufficient energy for second shot: power - 2.9 < 3.0 -> power < 5.9
        basic_ship.power = 5.8

        shoot_action = torch.zeros(len(Actions))
        shoot_action[Actions.shoot] = 1

        regen_action = torch.zeros(len(Actions))
        regen_action[Actions.backward] = 1  # Regenerative braking

        # Shoot once
        basic_ship.forward(shoot_action, empty_bullets, current_time=0.0, delta_t=0.01)
        assert empty_bullets.num_active == 1

        # Try to shoot again (should fail due to low energy)
        basic_ship.last_fired_time = -1.0  # Reset cooldown
        basic_ship.forward(shoot_action, empty_bullets, current_time=1.0, delta_t=0.01)
        assert empty_bullets.num_active == 1  # Still just one bullet

        # Regenerate energy
        for _ in range(10):
            basic_ship.forward(
                regen_action, empty_bullets, current_time=2.0, delta_t=0.1
            )

        # Now should be able to shoot again
        basic_ship.last_fired_time = -1.0  # Reset cooldown
        basic_ship.forward(shoot_action, empty_bullets, current_time=3.0, delta_t=0.01)
        assert empty_bullets.num_active == 2

    def test_max_bullets_per_ship(self, basic_ship):
        """Test calculation of maximum bullets per ship."""
        max_bullets = basic_ship.max_bullets
        expected = int(
            np.ceil(
                basic_ship.config.bullet_lifetime / basic_ship.config.firing_cooldown
            )
        )
        assert max_bullets == expected

    def test_dead_ship_cant_shoot(self, basic_ship, empty_bullets):
        """Test that dead ships cannot shoot."""
        action = torch.zeros(len(Actions))
        action[Actions.shoot] = 1

        basic_ship.alive = False
        basic_ship.forward(action, empty_bullets, current_time=1.0, delta_t=0.01)

        assert empty_bullets.num_active == 0
