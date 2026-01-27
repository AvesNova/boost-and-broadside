
import sys
import os
import torch
import numpy as np

# Add root to path
sys.path.append(os.getcwd())

from src.env.env import Environment
from src.env2.env import TensorEnv

def run_parity():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # 1. Init Env 1 (CPU)
    print("Initializing Env1...")
    env1 = Environment(
        render_mode="none",
        world_size=(1000, 1000),
        memory_size=0,
        max_ships=2,
        agent_dt=0.015,
        physics_dt=0.015,
        random_positioning=False,
        random_speed=False
    )
    obs1, _ = env1.reset(game_mode="1v1")
    
    # 2. Init Env 2 (GPU)
    print("Initializing Env2...")
    env2 = TensorEnv(
        num_envs=1, 
        device=device, 
        max_ships=2,
        dt=0.015,
        world_size=(1000.0, 1000.0)
    )
    env2.reset(options={"team_sizes": (1, 1)})
    
    print(f"Env1 substeps: {env1.physics_substeps}, dt: {env1.physics_dt}")
    print(f"Env2 dt: {env2.dt}")
    
    # 3. Sync State
    print("Syncing State...")
    ship0 = env1.state.ships[0]
    ship1 = env1.state.ships[1]
    
    # Ensure ships are active in env2 (should be by reset)
    with torch.no_grad():
        s = env2.state
        
        # Ship 0
        s.ships_pos[0, 0] = ship0.position
        s.ships_vel[0, 0] = ship0.velocity
        s.ships_power[0, 0] = ship0.power
        s.ships_health[0, 0] = ship0.health
        s.ships_team[0, 0] = ship0.team_id
        
        # Ship 1
        s.ships_pos[0, 1] = ship1.position
        s.ships_vel[0, 1] = ship1.velocity
        s.ships_power[0, 1] = ship1.power
        s.ships_health[0, 1] = ship1.health
        s.ships_team[0, 1] = ship1.team_id
        
        # Update derived
        speed0 = torch.abs(s.ships_vel[0, 0])
        safe_speed0 = max(speed0, 1e-6)
        s.ships_attitude[0, 0] = s.ships_vel[0, 0] / safe_speed0
        
        speed1 = torch.abs(s.ships_vel[0, 1])
        safe_speed1 = max(speed1, 1e-6)
        s.ships_attitude[0, 1] = s.ships_vel[0, 1] / safe_speed1

    # 4. Action
    print("Stepping...")
    # BOOST(1), STRAIGHT(0), NO SHOOT(0)
    action_raw = torch.tensor([1, 0, 0], dtype=torch.long)
    
    actions1 = {0: action_raw, 1: action_raw}
    
    actions2 = torch.zeros((1, 2, 3), dtype=torch.long, device=device)
    actions2[0, 0] = action_raw
    actions2[0, 1] = action_raw
    
    # 5. Step
    env1.step(actions1)
    env2.step({"action": actions2})
    
    # 6. Compare
    pos1_0 = env1.state.ships[0].position
    pos2_0 = s.ships_pos[0, 0].cpu().numpy()
    
    print(f"Env1 Pos: {pos1_0}")
    print(f"Env2 Pos: {pos2_0}")
    
    diff_pos = np.abs(pos1_0 - pos2_0)
    print(f"Diff Pos: {diff_pos}")
    
    if diff_pos < 1e-4:
        print("POS CHECK PASSED")
    else:
        print("POS CHECK FAILED")
        
    vel1_0 = env1.state.ships[0].velocity
    vel2_0 = s.ships_vel[0, 0].cpu().numpy()
    
    print(f"Env1 Vel: {vel1_0}")
    print(f"Env2 Vel: {vel2_0}")
    
    diff_vel = np.abs(vel1_0 - vel2_0)
    print(f"Diff Vel: {diff_vel}")
    
    if diff_vel < 1e-4:
        print("VEL CHECK PASSED")
    else:
        print("VEL CHECK FAILED")

if __name__ == "__main__":
    run_parity()
