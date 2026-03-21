"""Unit tests for the physics engine.

Tests physical invariants rather than exact floating-point values where possible.
"""

import pytest
import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.constants import PowerActions, TurnActions, ShootActions
from boost_and_broadside.env.physics import update_ships, update_bullets, resolve_collisions
from tests.conftest import make_state


@pytest.fixture
def cfg() -> ShipConfig:
    return ShipConfig()


class TestThrust:
    def test_boost_increases_speed_in_attitude_direction(self, cfg):
        """BOOST action must increase speed along the ship's attitude."""
        state = make_state(num_envs=1, max_ships=1, ship_config=cfg)
        state.ship_attitude[:] = 1.0 + 0j  # pointing East

        actions = torch.zeros((1, 1, 3), dtype=torch.float32)
        actions[0, 0, 0] = PowerActions.BOOST

        state = update_ships(state, actions, cfg)

        # Velocity should have gained a positive real component (East)
        assert state.ship_vel[0, 0].real > 0
        # Imaginary component should remain near zero
        assert abs(state.ship_vel[0, 0].imag) < 1e-5

    def test_reverse_decreases_speed(self, cfg):
        """REVERSE action with no initial velocity produces negative (backward) velocity."""
        state = make_state(num_envs=1, max_ships=1, ship_config=cfg)
        state.ship_attitude[:] = 1.0 + 0j

        actions = torch.zeros((1, 1, 3), dtype=torch.float32)
        actions[0, 0, 0] = PowerActions.REVERSE

        state = update_ships(state, actions, cfg)

        assert state.ship_vel[0, 0].real < 0

    def test_coast_with_no_velocity_produces_small_positive_thrust(self, cfg):
        """COAST provides base thrust — not zero — from standstill."""
        state = make_state(num_envs=1, max_ships=1, ship_config=cfg)
        state.ship_attitude[:] = 1.0 + 0j

        actions = torch.zeros((1, 1, 3), dtype=torch.float32)
        actions[0, 0, 0] = PowerActions.COAST

        state = update_ships(state, actions, cfg)

        assert state.ship_vel[0, 0].real > 0

    def test_no_power_prevents_thrust(self, cfg):
        """Ships with zero power cannot apply thrust."""
        state = make_state(num_envs=1, max_ships=1, ship_config=cfg)
        state.ship_attitude[:] = 1.0 + 0j
        state.ship_power[:] = 0.0

        actions = torch.zeros((1, 1, 3), dtype=torch.float32)
        actions[0, 0, 0] = PowerActions.BOOST

        state = update_ships(state, actions, cfg)

        # With no power, BOOST should not add forward velocity (beyond drag tiny effects)
        # Velocity starts at ~1e-6 (the min clamp value), stays near zero
        assert abs(state.ship_vel[0, 0].real) < 1.0

    def test_boost_velocity_magnitude_matches_expected(self, cfg):
        """After one step of BOOST from rest, velocity ≈ boost_thrust * dt."""
        state = make_state(num_envs=1, max_ships=1, ship_config=cfg)
        state.ship_attitude[:] = 1.0 + 0j

        actions = torch.zeros((1, 1, 3), dtype=torch.float32)
        actions[0, 0, 0] = PowerActions.BOOST

        state = update_ships(state, actions, cfg)

        expected = cfg.boost_thrust * cfg.dt
        assert abs(state.ship_vel[0, 0].real - expected) < 1e-4


class TestTurning:
    def test_turn_left_rotates_attitude_counterclockwise(self, cfg):
        """TURN_LEFT should produce a negative angle change (counter-clockwise)."""
        state = make_state(num_envs=1, max_ships=1, ship_config=cfg)
        state.ship_attitude[:] = 1.0 + 0j
        state.ship_vel[:] = 100.0 + 0j  # needs non-zero velocity

        actions = torch.zeros((1, 1, 3), dtype=torch.float32)
        actions[0, 0, 1] = TurnActions.TURN_LEFT

        state = update_ships(state, actions, cfg)

        angle = torch.angle(state.ship_attitude[0, 0]).item()
        assert angle < 0  # counter-clockwise = negative angle

    def test_turn_right_rotates_attitude_clockwise(self, cfg):
        """TURN_RIGHT should produce a positive angle change (clockwise)."""
        state = make_state(num_envs=1, max_ships=1, ship_config=cfg)
        state.ship_attitude[:] = 1.0 + 0j
        state.ship_vel[:] = 100.0 + 0j

        actions = torch.zeros((1, 1, 3), dtype=torch.float32)
        actions[0, 0, 1] = TurnActions.TURN_RIGHT

        state = update_ships(state, actions, cfg)

        angle = torch.angle(state.ship_attitude[0, 0]).item()
        assert angle > 0

    def test_sharp_turn_larger_angle_than_normal(self, cfg):
        """SHARP_LEFT should produce a larger attitude change than TURN_LEFT."""
        def run_turn(turn_action):
            state = make_state(num_envs=1, max_ships=1, ship_config=cfg)
            state.ship_attitude[:] = 1.0 + 0j
            state.ship_vel[:] = 100.0 + 0j
            actions = torch.zeros((1, 1, 3), dtype=torch.float32)
            actions[0, 0, 1] = turn_action
            state = update_ships(state, actions, cfg)
            return abs(torch.angle(state.ship_attitude[0, 0]).item())

        assert run_turn(TurnActions.SHARP_LEFT) > run_turn(TurnActions.TURN_LEFT)

    def test_go_straight_preserves_attitude(self, cfg):
        """GO_STRAIGHT with forward velocity should not change attitude."""
        state = make_state(num_envs=1, max_ships=1, ship_config=cfg)
        state.ship_attitude[:] = 1.0 + 0j
        state.ship_vel[:] = 100.0 + 0j

        actions = torch.zeros((1, 1, 3), dtype=torch.float32)
        actions[0, 0, 1] = TurnActions.GO_STRAIGHT

        state = update_ships(state, actions, cfg)

        angle = abs(torch.angle(state.ship_attitude[0, 0]).item())
        assert angle < 1e-5


class TestShooting:
    def test_shoot_action_spawns_active_bullet(self, cfg):
        """SHOOT with sufficient power and zero cooldown must activate a bullet."""
        state = make_state(num_envs=1, max_ships=1, ship_config=cfg)
        state.ship_cooldown[:] = 0.0
        state.ship_power[:] = cfg.max_power
        state.ship_attitude[:] = 1.0 + 0j

        actions = torch.zeros((1, 1, 3), dtype=torch.float32)
        actions[0, 0, 2] = ShootActions.SHOOT

        state = update_ships(state, actions, cfg)

        assert state.bullet_active[0, 0, 0].item()

    def test_shoot_deducts_power(self, cfg):
        """Shooting must deduct bullet_energy_cost from ship power."""
        state = make_state(num_envs=1, max_ships=1, ship_config=cfg)
        state.ship_cooldown[:] = 0.0
        state.ship_power[:] = cfg.max_power
        state.ship_attitude[:] = 1.0 + 0j

        actions = torch.zeros((1, 1, 3), dtype=torch.float32)
        actions[0, 0, 2] = ShootActions.SHOOT

        state = update_ships(state, actions, cfg)

        # Power: clamp(max_power + regen*dt, 0, max_power) → max_power (already at cap)
        #        then deduct bullet_energy_cost
        power_after_regen = min(cfg.max_power + cfg.base_power_gain * cfg.dt, cfg.max_power)
        expected          = power_after_regen - cfg.bullet_energy_cost
        assert abs(state.ship_power[0, 0].item() - expected) < 1e-3

    def test_shoot_blocked_by_cooldown(self, cfg):
        """Cannot shoot while cooldown is active."""
        state = make_state(num_envs=1, max_ships=1, ship_config=cfg)
        state.ship_cooldown[:] = cfg.firing_cooldown  # cooldown not expired
        state.ship_power[:] = cfg.max_power

        actions = torch.zeros((1, 1, 3), dtype=torch.float32)
        actions[0, 0, 2] = ShootActions.SHOOT

        state = update_ships(state, actions, cfg)

        assert not state.bullet_active[0, 0, 0].item()

    def test_shoot_blocked_by_insufficient_power(self, cfg):
        """Cannot shoot without enough power."""
        state = make_state(num_envs=1, max_ships=1, ship_config=cfg)
        state.ship_cooldown[:] = 0.0
        state.ship_power[:] = 0.0  # no power

        actions = torch.zeros((1, 1, 3), dtype=torch.float32)
        actions[0, 0, 2] = ShootActions.SHOOT

        state = update_ships(state, actions, cfg)

        assert not state.bullet_active[0, 0, 0].item()


class TestBulletLifetime:
    def test_bullet_expires_after_lifetime(self, cfg):
        """A bullet with bullet_time=0 after dt should be deactivated."""
        state = make_state(num_envs=1, max_ships=1, max_bullets=2, ship_config=cfg)
        # Manually place a bullet with lifetime just under dt — will expire next step
        state.bullet_active[0, 0, 0] = True
        state.bullet_time  [0, 0, 0] = cfg.dt * 0.5  # less than one step

        state = update_bullets(state, cfg)

        assert not state.bullet_active[0, 0, 0].item()

    def test_bullet_with_long_lifetime_remains_active(self, cfg):
        """A bullet with ample lifetime must stay active after one step."""
        state = make_state(num_envs=1, max_ships=1, max_bullets=2, ship_config=cfg)
        state.bullet_active[0, 0, 0] = True
        state.bullet_time  [0, 0, 0] = cfg.bullet_lifetime

        state = update_bullets(state, cfg)

        assert state.bullet_active[0, 0, 0].item()

    def test_bullet_moves_in_velocity_direction(self, cfg):
        """A bullet's position must change by vel * dt each step."""
        state = make_state(num_envs=1, max_ships=1, max_bullets=2, ship_config=cfg)
        init_pos = 100.0 + 200.0j
        bullet_vel = 500.0 + 0j

        state.bullet_active[0, 0, 0] = True
        state.bullet_pos   [0, 0, 0] = init_pos
        state.bullet_vel   [0, 0, 0] = bullet_vel
        state.bullet_time  [0, 0, 0] = cfg.bullet_lifetime

        state = update_bullets(state, cfg)

        expected = init_pos + bullet_vel * cfg.dt
        # Wrap expected position
        w, h = cfg.world_size
        expected_x = expected.real % w
        expected_y = expected.imag % h

        assert abs(state.bullet_pos[0, 0, 0].real - expected_x) < 1e-3
        assert abs(state.bullet_pos[0, 0, 0].imag - expected_y) < 1e-3


class TestCollisions:
    def test_bullet_reduces_target_health(self, cfg):
        """A bullet overlapping a ship must deal damage."""
        # Need 2 ships so own-bullet exclusion doesn't apply
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1

        target_pos = 100.0 + 100.0j
        state.ship_pos[0, 0] = target_pos
        state.ship_pos[0, 1] = 0.0 + 0j

        # Ship 1 fires a bullet at ship 0 (place it on top of ship 0)
        state.bullet_pos   [0, 1, 0] = target_pos
        state.bullet_vel   [0, 1, 0] = 500.0 + 0j
        state.bullet_active[0, 1, 0] = True
        state.bullet_time  [0, 1, 0] = 1.0

        initial_health = state.ship_health[0, 0].item()
        state, _ = resolve_collisions(state, cfg)

        assert state.ship_health[0, 0].item() < initial_health

    def test_ship_dies_when_health_reaches_zero(self, cfg):
        """Ship alive flag must be False when health drops to zero."""
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1
        state.ship_health[0, 0] = 0.01  # near death

        target_pos = 100.0 + 100.0j
        state.ship_pos[0, 0] = target_pos

        # Enough bullets to kill
        for k in range(5):
            state.bullet_pos   [0, 1, k] = target_pos
            state.bullet_vel   [0, 1, k] = 500.0 + 0j
            state.bullet_active[0, 1, k] = True
            state.bullet_time  [0, 1, k] = 1.0

        state, _ = resolve_collisions(state, cfg)

        assert not state.ship_alive[0, 0].item()
        assert state.ship_health[0, 0].item() == 0.0

    def test_own_bullets_do_not_damage_shooter(self, cfg):
        """Ships must not take damage from their own bullets."""
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1

        target_pos = 100.0 + 100.0j
        state.ship_pos[0, 0] = target_pos

        # Ship 0's OWN bullet overlapping ship 0
        state.bullet_pos   [0, 0, 0] = target_pos
        state.bullet_vel   [0, 0, 0] = 500.0 + 0j
        state.bullet_active[0, 0, 0] = True
        state.bullet_time  [0, 0, 0] = 1.0

        initial_health = state.ship_health[0, 0].item()
        state, _ = resolve_collisions(state, cfg)

        # No damage — own bullet excluded
        assert state.ship_health[0, 0].item() == initial_health

    def test_game_over_when_team_eliminated(self, cfg):
        """resolve_collisions must return done=True when one team is fully dead."""
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1
        state.ship_alive  [0, 1] = False  # team 1 already dead

        _, dones = resolve_collisions(state, cfg)

        assert dones[0].item()

    def test_no_game_over_while_both_teams_alive(self, cfg):
        """done must be False while both teams have at least one ship alive."""
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1

        _, dones = resolve_collisions(state, cfg)

        assert not dones[0].item()
