"""Unit tests for the physics engine.

Tests physical invariants rather than exact floating-point values where possible.
"""

from dataclasses import replace

import pytest
import torch

from boost_and_broadside.config import ShipConfig
from boost_and_broadside.constants import (
    DEFAULT_MAX_BULLETS_PER_SHIP,
    PowerActions,
    ShootActions,
    TurnActions,
)
from boost_and_broadside.env.physics import (
    advance_bullets,
    resolve_collisions,
    update_bullets,
    update_ships,
)
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


class TestBelowMinSpeedGating:
    """The stall (turn/lift zeroing) and attitude-hold branches share one threshold."""

    def test_below_min_speed_zeros_turn_and_holds_attitude_together(self, cfg):
        """Below min_speed: turn authority is zeroed AND attitude does not re-align to velocity."""
        state = make_state(num_envs=1, max_ships=1, ship_config=cfg)
        state.ship_attitude[:] = 1.0 + 0j  # facing East
        state.ship_vel[:] = 1j * (cfg.min_speed * 0.5)  # slow, facing North

        actions = torch.zeros((1, 1, 3), dtype=torch.float32)
        actions[0, 0, 1] = TurnActions.SHARP_LEFT

        state = update_ships(state, actions, cfg)

        # Turn authority zeroed by the stall gate
        assert state.ship_ang_vel[0, 0].item() == 0.0
        # Attitude held at its prior value rather than aligning to velocity direction
        assert state.ship_attitude[0, 0].real.item() > 0.9

    def test_above_min_speed_applies_turn_and_aligns_attitude_together(self, cfg):
        """Above min_speed: turn authority applies AND attitude aligns to velocity direction."""
        state = make_state(num_envs=1, max_ships=1, ship_config=cfg)
        state.ship_attitude[:] = 1.0 + 0j  # facing East
        state.ship_vel[:] = 1j * (cfg.min_speed * 5.0)  # fast, facing North

        actions = torch.zeros((1, 1, 3), dtype=torch.float32)
        actions[0, 0, 1] = TurnActions.SHARP_LEFT

        state = update_ships(state, actions, cfg)

        # Turn authority is not zeroed
        assert state.ship_ang_vel[0, 0].item() != 0.0
        # Attitude re-aligned toward velocity direction (North) rather than held at East
        assert state.ship_attitude[0, 0].imag.item() > 0.9


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
        power_after_regen = min(cfg.max_power + cfg.passive_power_gain * cfg.dt, cfg.max_power)
        expected = power_after_regen - cfg.bullet_energy_cost
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

    def test_default_pool_never_overwrites_a_live_bullet(self):
        cfg = ShipConfig(bullet_energy_cost=0.0, passive_power_gain=0.0)
        state = make_state(
            num_envs=1,
            max_ships=1,
            max_bullets=DEFAULT_MAX_BULLETS_PER_SHIP,
            ship_config=cfg,
        )
        actions = torch.tensor([[[0, 0, int(ShootActions.SHOOT)]]])
        shots = 0
        peak_live = 0

        for _ in range(600):
            cursor_before = state.bullet_cursor.item()
            cursor_slot_was_live = state.bullet_active[0, 0, cursor_before].item()
            state = update_ships(state, actions, cfg)
            fired = state.bullet_cursor.item() != cursor_before
            if fired:
                shots += 1
                assert not cursor_slot_was_live
            state = update_bullets(state, cfg)
            peak_live = max(peak_live, int(state.bullet_active.sum().item()))

        assert shots > DEFAULT_MAX_BULLETS_PER_SHIP
        assert peak_live < DEFAULT_MAX_BULLETS_PER_SHIP


class TestBulletLifetime:
    def test_bullet_expires_after_lifetime(self, cfg):
        """A bullet with bullet_time=0 after dt should be deactivated."""
        state = make_state(num_envs=1, max_ships=1, max_bullets=2, ship_config=cfg)
        # Manually place a bullet with lifetime just under dt — will expire next step
        state.bullet_active[0, 0, 0] = True
        state.bullet_time[0, 0, 0] = cfg.dt * 0.5  # less than one step

        state = update_bullets(state, cfg)

        assert not state.bullet_active[0, 0, 0].item()

    def test_bullet_with_long_lifetime_remains_active(self, cfg):
        """A bullet with ample lifetime must stay active after one step."""
        state = make_state(num_envs=1, max_ships=1, max_bullets=2, ship_config=cfg)
        state.bullet_active[0, 0, 0] = True
        state.bullet_time[0, 0, 0] = cfg.bullet_lifetime

        state = update_bullets(state, cfg)

        assert state.bullet_active[0, 0, 0].item()

    def test_bullet_moves_in_velocity_direction(self, cfg):
        """A bullet's position must change by vel * dt each step."""
        state = make_state(num_envs=1, max_ships=1, max_bullets=2, ship_config=cfg)
        init_pos = 100.0 + 200.0j
        bullet_vel = 500.0 + 0j

        state.bullet_active[0, 0, 0] = True
        state.bullet_pos[0, 0, 0] = init_pos
        state.bullet_vel[0, 0, 0] = bullet_vel
        state.bullet_time[0, 0, 0] = cfg.bullet_lifetime

        state = update_bullets(state, cfg)

        half_drag_scale = 1.0 / (1.0 + cfg.bullet_drag_coeff * abs(bullet_vel) * (0.5 * cfg.dt))
        expected = init_pos + bullet_vel * half_drag_scale * cfg.dt
        # Wrap expected position
        w, h = cfg.world_size
        expected_x = expected.real % w
        expected_y = expected.imag % h

        assert abs(state.bullet_pos[0, 0, 0].real - expected_x) < 1e-3
        assert abs(state.bullet_pos[0, 0, 0].imag - expected_y) < 1e-3

    def test_quadratic_drag_uses_exact_speed_solution(self, cfg):
        state = make_state(num_envs=1, max_ships=1, max_bullets=1, ship_config=cfg)
        initial_speed = 500.0
        state.bullet_vel[0, 0, 0] = initial_speed + 0.0j
        state.bullet_time[0, 0, 0] = cfg.bullet_lifetime
        state.bullet_active[0, 0, 0] = True

        state = update_bullets(state, cfg)

        expected = initial_speed / (1.0 + cfg.bullet_drag_coeff * initial_speed * cfg.dt)
        assert state.bullet_vel[0, 0, 0].abs().item() == pytest.approx(expected, rel=1e-6)


class TestCollisions:
    def test_bullet_hit_uses_remaining_damage_potential(self, cfg):
        cfg = replace(cfg, bullet_min_damage_frac=1.0)
        state = make_state(num_envs=1, max_ships=2, max_bullets=1, ship_config=cfg)
        state.ship_pos[0] = torch.tensor([0.0 + 0.0j, 100.0 + 100.0j])
        state.bullet_pos[0, 0, 0] = 100.0 + 100.0j
        state.bullet_active[0, 0, 0] = True
        state.bullet_remaining_damage[0, 0, 0] = 3.0

        health_before = state.ship_health[0, 1].item()
        state, _ = resolve_collisions(state, cfg)

        assert state.ship_health[0, 1].item() == pytest.approx(health_before - 3.0)

    def test_swept_collision_catches_fast_bullet_between_endpoints(self, cfg):
        state = make_state(num_envs=1, max_ships=2, max_bullets=1, ship_config=cfg)
        state.ship_pos[0] = torch.tensor([0.0 + 0.0j, 120.0 + 100.0j])
        state.bullet_pos[0, 0, 0] = 100.0 + 100.0j
        state.bullet_vel[0, 0, 0] = 2400.0 + 0.0j
        state.bullet_time[0, 0, 0] = 1.0
        state.bullet_active[0, 0, 0] = True

        state, start, midpoint = advance_bullets(state, cfg)
        assert abs(state.bullet_pos[0, 0, 0] - state.ship_pos[0, 1]) > cfg.collision_radius

        health_before = state.ship_health[0, 1].item()
        state, _ = resolve_collisions(
            state,
            cfg,
            bullet_start_pos=start,
            bullet_midpoint_pos=midpoint,
        )

        assert state.ship_health[0, 1].item() < health_before
        assert not state.bullet_active[0, 0, 0]

    def test_swept_collision_wraps_across_toroidal_boundary(self, cfg):
        state = make_state(num_envs=1, max_ships=2, max_bullets=1, ship_config=cfg)
        state.ship_pos[0] = torch.tensor([500.0 + 500.0j, 2.0 + 100.0j])
        state.bullet_pos[0, 0, 0] = 1018.0 + 100.0j
        state.bullet_vel[0, 0, 0] = 1200.0 + 0.0j
        state.bullet_time[0, 0, 0] = 1.0
        state.bullet_active[0, 0, 0] = True

        state, start, midpoint = advance_bullets(state, cfg)
        health_before = state.ship_health[0, 1].item()
        state, _ = resolve_collisions(
            state,
            cfg,
            bullet_start_pos=start,
            bullet_midpoint_pos=midpoint,
        )

        assert state.ship_health[0, 1].item() < health_before

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
        state.bullet_pos[0, 1, 0] = target_pos
        state.bullet_vel[0, 1, 0] = 500.0 + 0j
        state.bullet_active[0, 1, 0] = True
        state.bullet_time[0, 1, 0] = 1.0

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
            state.bullet_pos[0, 1, k] = target_pos
            state.bullet_vel[0, 1, k] = 500.0 + 0j
            state.bullet_active[0, 1, k] = True
            state.bullet_time[0, 1, k] = 1.0

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
        state.bullet_pos[0, 0, 0] = target_pos
        state.bullet_vel[0, 0, 0] = 500.0 + 0j
        state.bullet_active[0, 0, 0] = True
        state.bullet_time[0, 0, 0] = 1.0

        initial_health = state.ship_health[0, 0].item()
        state, _ = resolve_collisions(state, cfg)

        # No damage — own bullet excluded
        assert state.ship_health[0, 0].item() == initial_health

    def test_game_over_when_team_eliminated(self, cfg):
        """resolve_collisions must return done=True when one team is fully dead."""
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1
        state.ship_alive[0, 1] = False  # team 1 already dead

        _, dones = resolve_collisions(state, cfg)

        assert dones[0].item()

    def test_no_game_over_while_both_teams_alive(self, cfg):
        """done must be False while both teams have at least one ship alive."""
        state = make_state(num_envs=1, max_ships=2, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1

        _, dones = resolve_collisions(state, cfg)

        assert not dones[0].item()


class TestApplyCombatDamage:
    def test_damage_matrix_and_health_reduction(self, cfg):
        """Combat damage must reduce health, populate damage_matrix, and deactivate bullets."""
        # 3 ships in 1 env
        state = make_state(num_envs=1, max_ships=3, ship_config=cfg)
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1
        state.ship_team_id[0, 2] = 1

        target_pos = 200.0 + 200.0j
        state.ship_pos[0, 0] = target_pos  # target ship
        state.ship_pos[0, 1] = 0.0 + 0j
        state.ship_pos[0, 2] = 0.0 + 0j

        # Ship 1 fires bullet at ship 0
        state.bullet_pos[0, 1, 0] = target_pos
        state.bullet_vel[0, 1, 0] = 500.0 + 0j
        state.bullet_active[0, 1, 0] = True
        state.bullet_time[0, 1, 0] = 1.0

        # Ship 2 fires bullet at ship 0
        state.bullet_pos[0, 2, 0] = target_pos
        state.bullet_vel[0, 2, 0] = 500.0 + 0j
        state.bullet_active[0, 2, 0] = True
        state.bullet_time[0, 2, 0] = 1.0

        initial_health = state.ship_health[0, 0].item()

        # Import target function directly
        from boost_and_broadside.env.physics import _apply_combat_damage

        state = _apply_combat_damage(state, cfg)

        # Health reduced
        assert state.ship_health[0, 0].item() < initial_health
        # Bullets deactivated
        assert not state.bullet_active[0, 1, 0].item()
        assert not state.bullet_active[0, 2, 0].item()

        # Shooter attribution populated
        # dm: (B, N_shooter, N_target)
        assert state.damage_matrix[0, 1, 0].item() > 0
        assert state.damage_matrix[0, 2, 0].item() > 0
        assert state.cumulative_damage_matrix[0, 1, 0].item() > 0
        assert state.cumulative_damage_matrix[0, 2, 0].item() > 0
        # shooter 0 (target itself) has no damage dealt
        assert state.damage_matrix[0, 0, 0].item() == 0

    def test_empty_active_bullets_is_fast_pass(self, cfg):
        """Test that combat damage does not mutate anything when no bullets are active."""
        state = make_state(num_envs=4, max_ships=4, ship_config=cfg)
        state.bullet_active.zero_()

        from boost_and_broadside.env.physics import _apply_combat_damage

        orig_health = state.ship_health.clone()
        orig_matrix = state.damage_matrix.clone()

        state = _apply_combat_damage(state, cfg)

        assert torch.equal(state.ship_health, orig_health)
        assert torch.equal(state.damage_matrix, orig_matrix)
