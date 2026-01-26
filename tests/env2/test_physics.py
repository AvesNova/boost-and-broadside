
import pytest
import torch
import numpy as np
from src.env2.physics import update_ships, update_bullets, check_collisions

@pytest.fixture
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_update_ships_movement(device):
    """Test basic movement integration (Semi-Implicit Euler)."""
    B, N = 2, 1
    dt = 0.1
    world_size = (100.0, 100.0)
    
    ships_pos = torch.zeros((B, N), dtype=torch.complex64, device=device)
    ships_vel = torch.tensor([[10.0+0j], [0.0+10j]], dtype=torch.complex64, device=device) # Right, Up
    ships_power = torch.ones((B, N), device=device) * 100.0
    ships_cooldown = torch.zeros((B, N), device=device)
    ships_team = torch.zeros((B, N), dtype=torch.int64, device=device)
    ships_alive = torch.ones((B, N), dtype=torch.bool, device=device)
    
    # Action: Coast (0), Straight (0), No Shoot (0)
    actions_power = torch.zeros((B, N), dtype=torch.int64, device=device)
    actions_turn = torch.zeros((B, N), dtype=torch.int64, device=device)
    actions_shoot = torch.zeros((B, N), dtype=torch.int64, device=device)
    
    # Expected:
    # Drag will apply. 
    # v_new = v_old + (acc_thrust + acc_drag) * dt
    # But thrust is non-zero even for COAST (base_thrust=8.0).
    # drag = -coeff * speed * vel
    
    new_pos, new_vel, _, _, _, _, _, _ = update_ships(
        ships_pos, ships_vel, ships_power, ships_cooldown, ships_team, ships_alive,
        actions_power, actions_turn, actions_shoot, dt, world_size
    )
    
    # Check velocity direction roughly maintained
    assert torch.allclose(new_vel[0,0].imag, torch.tensor(0.0, device=device), atol=1e-5)
    assert new_vel[0,0].real > 0
    
    # Check position update: pos + vel*dt
    # Since vel changes, pos should receive updated vel (Semi-Implicit)
    assert torch.allclose(new_pos, ships_pos + new_vel * dt, atol=1e-5)

def test_collision_logic(device):
    """Test collision detection between ships and bullets."""
    B, N = 1, 2
    M, K = 2, 1 # 2 ships (sources), 1 bullet each
    
    ships_pos = torch.tensor([[10.0+0j, 20.0+0j]], dtype=torch.complex64, device=device) # Ship 0 at 10, Ship 1 at 20
    ships_team = torch.tensor([[0, 1]], dtype=torch.int64, device=device)
    ships_alive = torch.ones((1, 2), dtype=torch.bool, device=device)
    
    # Bullets
    # Bullet 0 (from Ship 0) at 20.0 (Hitting Ship 1)
    # Bullet 1 (from Ship 1) at 100.0 (Miss)
    bullets_pos = torch.tensor([
        [[20.0+0j], [100.0+0j]]
    ], dtype=torch.complex64, device=device)
    
    bullets_team = torch.tensor([[[0], [1]]], dtype=torch.int64, device=device) # Teams match sources
    bullets_time = torch.ones((1, 2, 1), device=device)
    
    hits_matrix, bullet_mask = check_collisions(
        ships_pos, ships_team, ships_alive,
        bullets_pos, bullets_team, bullets_time,
        ship_collision_radius=5.0
    )
    
    # Sum over sources to get hits per ship
    hits = hits_matrix.sum(dim=2)
    
    # Expect:
    # Ship 0 hits: 0
    # Ship 1 hits: 1 (from Bullet 0)
    assert hits[0, 0] == 0
    assert hits[0, 1] == 1
    
    # Bullet 0 (src mask 0, index 0): Hit -> True
    # Bullet 1 (src mask 1, index 0): Miss -> False
    assert bullet_mask[0, 0, 0] == True
    assert bullet_mask[0, 1, 0] == False

def test_self_hit_ignored(device):
    """Ensure a ship cannot hit itself."""
    B, N = 1, 1
    M, K = 1, 1
    
    ships_pos = torch.tensor([[10.0+0j]], dtype=torch.complex64, device=device)
    ships_team = torch.zeros((1, 1), dtype=torch.int64, device=device)
    ships_alive = torch.ones((1, 1), dtype=torch.bool, device=device)
    
    # Bullet at same position as ship, from same ship
    bullets_pos = torch.tensor([[[10.0+0j]]], dtype=torch.complex64, device=device)
    bullets_team = torch.zeros((1, 1, 1), dtype=torch.int64, device=device)
    bullets_time = torch.ones((1, 1, 1), device=device)
    
    hits_matrix, bullet_mask = check_collisions(
        ships_pos, ships_team, ships_alive,
        bullets_pos, bullets_team, bullets_time
    )
    
    hits = hits_matrix.sum(dim=2)
    
    assert hits[0, 0] == 0
    assert bullet_mask[0, 0, 0] == False
