
import pytest
import torch
from boost_and_broadside.env2.state import TensorState, ShipConfig
from boost_and_broadside.env2.physics import update_ships, resolve_collisions
from boost_and_broadside.core.constants import PowerActions, TurnActions, ShootActions

class TestPhysics:
    @pytest.fixture
    def config(self):
        return ShipConfig()

    def create_state(self, config):
        num_envs = 2
        max_ships = 1
        max_bullets = 5
        device = torch.device("cpu")
        
        return TensorState(
            step_count=torch.zeros((num_envs,), dtype=torch.int32, device=device),
            ship_pos=torch.zeros((num_envs, max_ships), dtype=torch.complex64, device=device),
            ship_vel=torch.zeros((num_envs, max_ships), dtype=torch.complex64, device=device),
            ship_attitude=torch.ones((num_envs, max_ships), dtype=torch.complex64, device=device), # Pointing East (1+0j)
            ship_ang_vel=torch.zeros((num_envs, max_ships), dtype=torch.float32, device=device),
            ship_health=torch.ones((num_envs, max_ships), dtype=torch.float32, device=device) * config.max_health,
            ship_power=torch.ones((num_envs, max_ships), dtype=torch.float32, device=device) * config.max_power,
            ship_cooldown=torch.zeros((num_envs, max_ships), dtype=torch.float32, device=device),
            ship_team_id=torch.zeros((num_envs, max_ships), dtype=torch.int32, device=device),
            ship_alive=torch.ones((num_envs, max_ships), dtype=torch.bool, device=device),
            ship_is_shooting=torch.zeros((num_envs, max_ships), dtype=torch.bool, device=device),
            
            bullet_pos=torch.zeros((num_envs, max_ships, max_bullets), dtype=torch.complex64, device=device),
            bullet_vel=torch.zeros((num_envs, max_ships, max_bullets), dtype=torch.complex64, device=device),
            bullet_time=torch.zeros((num_envs, max_ships, max_bullets), dtype=torch.float32, device=device),
            bullet_active=torch.zeros((num_envs, max_ships, max_bullets), dtype=torch.bool, device=device),
            bullet_cursor=torch.zeros((num_envs, max_ships), dtype=torch.long, device=device)
        )

    @pytest.fixture
    def state(self, config):
        return self.create_state(config)


    def test_thrust(self, state, config):
        # Env 0: BOOST, Env 1: REVERSE
        actions = torch.zeros((2, 1, 3), dtype=torch.float32)
        actions[0, 0, 0] = PowerActions.BOOST
        actions[1, 0, 0] = PowerActions.REVERSE
        
        # Initial velocity 0
        state = update_ships(state, actions, config)
        
        # Check velocity
        # Accel = Thrust / Mass(1) - Drag
        # Drag at 0 speed is 0.
        # Vel = Thrust * dt
        
        dt = config.dt
        expected_vel0 = config.boost_thrust * dt
        expected_vel1 = config.reverse_thrust * dt # Negative
        
        assert torch.allclose(state.ship_vel[0, 0].real, torch.tensor(expected_vel0, dtype=torch.float32), atol=1e-5)
        assert torch.allclose(state.ship_vel[1, 0].real, torch.tensor(expected_vel1, dtype=torch.float32), atol=1e-5)
        
    def test_turn(self, state, config):
        # Env 0: TURN_LEFT
        actions = torch.zeros((2, 1, 3), dtype=torch.float32)
        actions[0, 0, 1] = TurnActions.TURN_LEFT
        
        # Give some forward velocity so turn works normally
        state.ship_vel[:] = 10.0 + 0j
        state.ship_attitude[:] = 1.0 + 0j
        
        state = update_ships(state, actions, config)
        
        # Check attitude rotation
        # Angle should increase by normal_turn_angle
        # Original attitude 0 rad.
        # Expected: exp(1j * angle)
        
        # Note: update_ships logic:
        # 1. attitude = vel/speed * rotation
        # vel is (10, 0) -> dir (1, 0)
        # rotation is exp(i * 0.087...)
        # new attitude should be exp(i * 0.087...)
        
        expected_angle = -config.normal_turn_angle # LEFT is negative in config?
        # Verify config sign
        # src/env/ship.py says LEFT is -angle.
        # But physics.py uses turn_offset_table[TURN_LEFT] = -config.normal_turn_angle
        
        current_att = state.ship_attitude[0, 0]
        current_angle = torch.angle(current_att)
        
        expected_tensor = torch.tensor(expected_angle, dtype=torch.float32)
        assert torch.allclose(current_angle, expected_tensor, atol=1e-5)
        
    def test_shoot(self, state, config):
        # Env 0: Shoot
        actions = torch.zeros((2, 1, 3), dtype=torch.float32)
        actions[0, 0, 2] = ShootActions.SHOOT
        
        # Ensure cooldown 0 and power
        state.ship_cooldown[:] = 0
        state.ship_power[:] = config.max_power
        
        state = update_ships(state, actions, config)
        
        # Check bullet spawned
        assert state.bullet_active[0, 0, 0]
        assert state.bullet_time[0, 0, 0] == config.bullet_lifetime
        
        # Check power consumed
        # Power is clamped at max_power before shooting cost is applied.
        # Since we start at max_power, regen is lost to clamp.
        expected_power = config.max_power - config.bullet_energy_cost
        # Note: Power regeneration happens every step: base_power_gain * dt
        # Then cost deducted.
        assert torch.allclose(state.ship_power[0, 0], torch.tensor(expected_power, dtype=torch.float32), atol=1e-4)

    def test_collisions(self, state, config):
        # Setup bullet hitting ship
        # Env 0, Ship 0 (Target) at (100, 100)
        # Ship 0 (Source - masked) at (0, 0).
        # Wait, strictly N=1 means source and target are same ship?
        # Collision logic ignores own bullets.
        # So we need N=2 ships to test collision properly in same env, or assume cross-env collision which shouldn't happen.
        # Physics collisions are per-env.
        # So need N=2.
        
        # Re-init state with N=2
        num_envs = 1
        max_ships = 2
        max_bullets = 5
        device = torch.device("cpu")
        
        state = TensorState(
            step_count=torch.zeros((num_envs,), dtype=torch.int32, device=device),
            ship_pos=torch.zeros((num_envs, max_ships), dtype=torch.complex64, device=device),
            ship_vel=torch.zeros((num_envs, max_ships), dtype=torch.complex64, device=device),
            ship_attitude=torch.ones((num_envs, max_ships), dtype=torch.complex64, device=device),
            ship_ang_vel=torch.zeros((num_envs, max_ships), dtype=torch.float32, device=device),
            ship_health=torch.ones((num_envs, max_ships), dtype=torch.float32, device=device) * config.max_health,
            ship_power=torch.ones((num_envs, max_ships), dtype=torch.float32, device=device) * config.max_power,
            ship_cooldown=torch.zeros((num_envs, max_ships), dtype=torch.float32, device=device),
            ship_team_id=torch.zeros((num_envs, max_ships), dtype=torch.int32, device=device),
            ship_alive=torch.ones((num_envs, max_ships), dtype=torch.bool, device=device),
            ship_is_shooting=torch.zeros((num_envs, max_ships), dtype=torch.bool, device=device),
            
            bullet_pos=torch.zeros((num_envs, max_ships, max_bullets), dtype=torch.complex64, device=device),
            bullet_vel=torch.zeros((num_envs, max_ships, max_bullets), dtype=torch.complex64, device=device),
            bullet_time=torch.zeros((num_envs, max_ships, max_bullets), dtype=torch.float32, device=device),
            bullet_active=torch.zeros((num_envs, max_ships, max_bullets), dtype=torch.bool, device=device),
            bullet_cursor=torch.zeros((num_envs, max_ships), dtype=torch.long, device=device)
        )
        
        # Ship 1 shoots (idx 1)
        # Bullet placed at Ship 0 location (fake it)
        state.ship_pos[0, 0] = 100 + 100j
        state.ship_team_id[0, 0] = 0
        state.ship_team_id[0, 1] = 1 # Enemy
        
        # Place a bullet owned by Ship 1 at Ship 0 location
        state.bullet_pos[0, 1, 0] = 100 + 100j
        state.bullet_active[0, 1, 0] = True
        state.bullet_time[0, 1, 0] = 1.0
        
        # Run resolution
        state, rewards, dones = resolve_collisions(state, config)
        
        # Check damage
        assert state.ship_health[0, 0] == config.max_health - config.bullet_damage
        
        # Check reward (Ship 1 gets reward)
        # rewards shape (B, N) -> (1, 2)
        # RewardConstants.ENEMY_DAMAGE is likely tiny?
        # rewards[0, 1] should be positive.
        assert rewards[0, 1] > 0
        

if __name__ == "__main__":
    print("Running manual test...")
    t = TestPhysics()
    c = ShipConfig()
    # Mock fixture
    s = t.create_state(c)
    
    t.test_thrust(s, c)
    print("test_thrust passed")
    
    t.test_turn(s, c)
    print("test_turn passed")
    
    t.test_shoot(s, c)
    print("test_shoot passed")
    
    t.test_collisions(s, c)
    print("test_collisions passed")


