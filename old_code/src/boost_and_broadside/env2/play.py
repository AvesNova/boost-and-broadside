
import torch

# Ensure root is in path

from boost_and_broadside.env2.env import TensorEnv
from boost_and_broadside.core.config import ShipConfig
from boost_and_broadside.env2.adapter import tensor_state_to_render_state
from boost_and_broadside.env2.renderer import GameRenderer
from boost_and_broadside.core.constants import TurnActions, PowerActions, ShootActions

def play():
    """Run a single game with visualization and human control using TensorEnv."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running on {device}")
    
    # Configuration
    config = ShipConfig(
        world_size=(1500.0, 1500.0),
        dt=0.015
    )
    
    # 1. Setup Environment (Batch Size = 1)
    env = TensorEnv(
        num_envs=1,
        config=config,
        device=device,
        max_ships=4
    )
    
    # Reset (1v1 for optimal control)
    env.reset(options={"team_sizes": (1, 1)})
    
    # 2. Setup Renderer
    renderer = GameRenderer(config, target_fps=60)
    # renderer.initialize() # Not needed for new logic unless added method
    
    # Add human player (Ship 0) - Controlling Team 0
    renderer.add_human_player(ship_id=0) 
    
    running = True
    
    while running:
        # 1. Handle Events
        if not renderer.handle_events():
            running = False
            break
            
        renderer.update_human_actions()
        human_actions_dict = renderer.get_human_actions()
        
        # 2. Construct Action Tensor
        # Shape (1, NumShips, 3)
        actions = torch.zeros((1, env.max_ships, 3), dtype=torch.long, device=device)
        
        # Apply Human Actions
        for ship_id, act_tensor in human_actions_dict.items():
            if ship_id < env.max_ships:
                actions[0, ship_id, 0] = int(act_tensor[0])
                actions[0, ship_id, 1] = int(act_tensor[1])
                actions[0, ship_id, 2] = int(act_tensor[2])
                
        # AI/Bot Actions for others (Simple spin and shoot)
        for i in range(env.max_ships):
            if i not in human_actions_dict:
                actions[0, i, 0] = PowerActions.BOOST 
                actions[0, i, 1] = TurnActions.TURN_LEFT
                actions[0, i, 2] = ShootActions.SHOOT
        
        # 3. Step Environment
        # Pass actions directly as tensor (Env supports both dict and tensor, but tensor is preferred internally)
        obs, rewards, done, _, _ = env.step(actions)
        
        # 4. Render
        render_state = tensor_state_to_render_state(env.state, config, batch_idx=0)
        renderer.render(render_state)
        
        # Check Done
        if done[0]:
            print("Game Over! Resetting...")
            env.reset(options={"team_sizes": (1, 1)})
            
    renderer.close()

if __name__ == "__main__":
    play()
