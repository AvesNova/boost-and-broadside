"""
Tests for combat scenarios in the environment.
"""

import pytest
import numpy as np
import torch

from src.constants import Actions


class TestCombatScenarios:
    """Integration tests for combat scenarios."""

    def test_simple_duel(self, basic_env):
        """Test a simple shooting duel."""
        basic_env.reset(game_mode="1v1_old")
        basic_env.state[-1].ships[0].position = 400.0 + 300.0j
        basic_env.state[-1].ships[1].position = 500.0 + 300.0j

        # Both ships shoot at each other
        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        actions[0][Actions.shoot] = 1
        actions[1][Actions.shoot] = 1

        # Run combat for several steps
        for _ in range(10):
            obs, rewards, terminated, truncated, info = basic_env.step(actions)

            # Reset cooldowns to keep shooting
            for ship in basic_env.state[-1].ships.values():
                ship.last_fired_time = -1.0

            if terminated:
                break

        # Should have active bullets
        assert info["active_bullets"] > 0

        # At least one ship should have taken damage
        ship_states = info["ship_states"]
        damaged = any(
            state["health"] < basic_env.state[0].ships[sid].config.max_health
            for sid, state in ship_states.items()
        )
        assert damaged or terminated

    def test_chase_and_shoot(self, basic_env):
        """Test chasing while shooting."""
        basic_env.reset(game_mode="1v1_old")

        # Ship 0 chases and shoots
        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        actions[0][Actions.forward] = 1
        actions[0][Actions.shoot] = 1

        # Ship 1 evades
        actions[1][Actions.forward] = 1
        actions[1][Actions.left] = 1

        initial_distance = abs(
            basic_env.state[-1].ships[0].position
            - basic_env.state[-1].ships[1].position
        )

        # Run for several steps
        for _ in range(20):
            obs, rewards, terminated, truncated, info = basic_env.step(actions)
            if terminated:
                break

        # Distance should have changed
        final_distance = abs(
            basic_env.state[-1].ships[0].position
            - basic_env.state[-1].ships[1].position
        )
        assert abs(final_distance - initial_distance) > 10.0

    def test_combat_to_destruction(self, basic_env):
        """Test combat until one ship is destroyed."""
        basic_env.reset(game_mode="1v1_old")

        # Place ships closer for faster combat
        basic_env.state[-1].ships[0].position = 400.0 + 300.0j
        basic_env.state[-1].ships[1].position = 420.0 + 300.0j
        basic_env.state[-1].ships[0].velocity = 1.0 + 0.0j
        basic_env.state[-1].ships[1].velocity = 1.0 + 0.0j

        # Both shoot continuously
        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        actions[0][Actions.shoot] = 1
        actions[1][Actions.shoot] = 1

        # Add lots of bullets quickly
        for _ in range(20):
            for ship in basic_env.state[-1].ships.values():
                if ship.alive:
                    ship.power = ship.config.max_power  # Unlimited ammo
                    ship.last_fired_time = -1.0  # No cooldown

            obs, rewards, terminated, truncated, info = basic_env.step(actions)

            if terminated:
                # One team should be eliminated
                alive_teams = set(
                    ship.team_id
                    for ship in basic_env.state[-1].ships.values()
                    if ship.alive
                )
                assert len(alive_teams) <= 1
                break

        # Should terminate eventually
        assert terminated

    def test_energy_management_in_combat(self, basic_env):
        """Test energy depletion affects combat."""
        basic_env.reset(game_mode="1v1_old")

        # Ship 0 shoots continuously
        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        actions[0][Actions.shoot] = 1
        actions[0][Actions.forward] = 1  # Also boost (energy drain)

        bullets_fired = 0
        energy_depleted = False

        for step in range(50):
            initial_bullets = basic_env.state[-1].bullets.num_active

            obs, rewards, terminated, truncated, info = basic_env.step(actions)

            # Check if bullet was fired
            if basic_env.state[-1].bullets.num_active > initial_bullets:
                bullets_fired += 1

            # Check energy
            if basic_env.state[-1].ships[0].power <= 0:
                energy_depleted = True
                # Should stop firing when out of energy
                remaining_bullets = basic_env.state[-1].bullets.num_active

                # Continue stepping
                for _ in range(10):
                    obs, rewards, terminated, truncated, info = basic_env.step(actions)
                    # No new bullets should be created
                    assert basic_env.state[-1].bullets.num_active <= remaining_bullets
                break

            # Reset cooldown for continuous firing
            basic_env.state[-1].ships[0].last_fired_time = -1.0

        assert energy_depleted
        assert bullets_fired > 0

    def test_projectile_lifetime(self, basic_env):
        """Test that projectiles expire after lifetime."""
        basic_env.reset(game_mode="1v1_old")

        # Fire a bullet
        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        actions[0][Actions.shoot] = 1

        obs, rewards, terminated, truncated, info = basic_env.step(actions)
        assert info["active_bullets"] > 0

        # Wait for bullet lifetime to expire
        bullet_lifetime = basic_env.state[-1].ships[0].config.bullet_lifetime
        steps_needed = int(np.ceil(bullet_lifetime / basic_env.agent_dt))

        actions[0][Actions.shoot] = 0  # Stop shooting

        for _ in range(steps_needed + 2):
            obs, rewards, terminated, truncated, info = basic_env.step(actions)

        # All bullets should have expired
        assert info["active_bullets"] == 0


class TestMultiShipCombat:
    """Tests for scenarios with multiple ships per team."""

    def test_simultaneous_damage(self, basic_env):
        """Test multiple ships taking damage simultaneously."""
        basic_env.reset(game_mode="1v1_old")

        initial_healths = {
            sid: ship.health for sid, ship in basic_env.state[-1].ships.items()
        }

        # Place bullets at both ships
        for ship_id, ship in basic_env.state[-1].ships.items():
            enemy_id = 1 - ship_id  # Get enemy ship id
            basic_env.state[-1].bullets.add_bullet(
                ship_id=enemy_id,
                x=ship.position.real,
                y=ship.position.imag,
                vx=0.0,
                vy=0.0,
                lifetime=1.0,
            )

        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        obs, rewards, terminated, truncated, info = basic_env.step(actions)

        # Both ships should take damage
        for ship_id in [0, 1]:
            final_health = basic_env.state[-1].ships[ship_id].health
            damage = basic_env.state[-1].ships[ship_id].config.bullet_damage
            expected_health = initial_healths[ship_id] - damage
            assert abs(final_health - expected_health) < 1e-5


class TestCombatEdgeCases:
    """Tests for edge cases in combat."""

    def test_shooting_at_world_boundary(self, basic_env):
        """Test shooting when ship is at world boundary."""
        basic_env.reset(game_mode="1v1")

        # Place ship at boundary
        basic_env.state[-1].ships[0].position = basic_env.world_size[0] - 10.0 + 300.0j

        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        actions[0][Actions.shoot] = 1

        obs, rewards, terminated, truncated, info = basic_env.step(actions)

        # Bullet should be created and wrap if necessary
        assert info["active_bullets"] > 0

        # Check bullet wrapped correctly
        bullets = basic_env.state[-1].bullets
        assert 0 <= bullets.x[0] < basic_env.world_size[0]
        assert 0 <= bullets.y[0] < basic_env.world_size[1]

    def test_zero_damage_bullets(self, basic_env):
        """Test handling of bullets with zero damage."""
        basic_env.reset(game_mode="1v1")

        # Modify bullet damage to zero
        for ship in basic_env.state[-1].ships.values():
            ship.config.bullet_damage = 0.0

        # Place bullet at ship
        ship1 = basic_env.state[-1].ships[1]
        basic_env.state[-1].bullets.add_bullet(
            ship_id=0,
            x=ship1.position.real,
            y=ship1.position.imag,
            vx=0.0,
            vy=0.0,
            lifetime=1.0,
        )

        initial_health = ship1.health

        actions = {0: torch.zeros(len(Actions)), 1: torch.zeros(len(Actions))}
        obs, rewards, terminated, truncated, info = basic_env.step(actions)

        # No damage should occur
        assert ship1.health == initial_health

        # Bullet should still be removed after collision
        assert info["active_bullets"] == 0
