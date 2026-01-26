
import sys
import os
import time
import torch
import numpy as np

# Ensure root is in path
sys.path.append(os.getcwd())

from src.env2.env import TensorEnv
from src.env2.adapter import tensor_state_to_cpu_state
from env.renderer import GameRenderer
from env.constants import TurnActions, PowerActions, ShootActions

def play():
    """
    Run a single game with visualization and human control using TensorEnv.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}")
    
    # 1. Setup Env (B=1)
    env = TensorEnv(
        num_envs=1,
        device=device,
        max_ships=4, # 2v2 or 1v1
        dt=0.015,
        world_size=(1500, 1500)
    )
    
    # Reset (1v1 for optimal control)
    env.reset(options={"team_sizes": (1, 1), "random_pos": False})
    
    # 2. Setup Renderer
    renderer = GameRenderer(world_size=(1500, 1500), target_fps=60)
    renderer.initialize()
    
    # Add human player (Ship 0)
    renderer.add_human_player(ship_id=0) # Controls ship index 0 (Team 0)
    # Optional: Control Ship 1 (Team 1) via secondary keys
    # renderer.add_human_player(ship_id=1) 
    
    running = True
    
    while running:
        # 1. Handle Events
        if not renderer.handle_events():
            running = False
            break
            
        renderer.update_human_actions()
        human_actions_dict = renderer.get_human_actions()
        
        # 2. Construct Action Tensor
        # Shape (1, N, 3)
        actions = torch.zeros((1, env.max_ships, 3), dtype=torch.long, device=device)
        
        # Apply Human Actions
        # Map ship_id -> Index in tensor
        # In 1v1 (size 1, 1), indices are 0 and 1.
        for ship_id, act_tensor in human_actions_dict.items():
            # act_tensor is [Power(float), Turn(float), Shoot(float)] from renderer
            # Convert to Long
            if ship_id < env.max_ships:
                actions[0, ship_id, 0] = int(act_tensor[0])
                actions[0, ship_id, 1] = int(act_tensor[1])
                actions[0, ship_id, 2] = int(act_tensor[2])
                
        # AI/Bot Actions for others
        # Simple policy: Spin and Shoot if alive
        for i in range(env.max_ships):
            if i not in human_actions_dict:
                actions[0, i, 0] = PowerActions.BOOST # Boost
                actions[0, i, 1] = TurnActions.TURN_LEFT # Left
                actions[0, i, 2] = ShootActions.SHOOT # Fire
        
        # 3. Step Env
        obs, rewards, done, _, _ = env.step({"action": actions})
        
        # 4. Render
        # Convert State
        cpu_state = tensor_state_to_cpu_state(env.state, batch_idx=0)
        renderer.render(cpu_state)
        
        # Check Done
        if done[0]:
            print("Game Over! Resetting...")
            env.reset(options={"team_sizes": (1, 1)})
            
    renderer.close()

if __name__ == "__main__":
    play()
