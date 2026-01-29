
import torch
import numpy as np
from src.env2.env import TensorEnv
from src.env2.env import TensorEnv
from src.env2.adapter import tensor_state_to_render_state
from src.env2.renderer import GameRenderer
from core.constants import RewardConstants
from core.config import ShipConfig

class TensorEnvWrapper:
    """
    Wraps TensorEnv to provide the API expected by GameCoordinator.
    
    Handles legacy observation format (dict of tensors) and human rendering integration
    via the new backend-agnostic GameRenderer.
    
    Attributes:
        device: The device (cpu/cuda) execution occurs on.
        render_mode: Render mode (human/none).
        world_size: World dimensions.
        max_ships: Max ships per game.
        env: The underlying TensorEnv instance.
        renderer: The renderer instance (if human mode).
    """
    
    def __init__(self, **kwargs):
        """
        Initialize TensorEnvWrapper.
        
        Args:
            **kwargs: Configuration arguments matching Environment signature.
                      Common args: render_mode, world_size, max_ships, physics_dt.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Config params
        self.render_mode = kwargs.get("render_mode", "none")
        self.world_size = kwargs.get("world_size", (2000, 2000))
        self.max_ships = kwargs.get("max_ships", 16)
        self.agent_dt = kwargs.get("agent_dt", 0.015) 
        self.physics_dt = kwargs.get("physics_dt", 0.015)
        self.num_teams = kwargs.get("num_teams", 2)
        
        # Create Config
        self.config = ShipConfig(
            world_size=tuple(float(x) for x in self.world_size),
            dt=self.physics_dt
        )

        # TensorEnv supports batch_size=1 for single game play
        self.env = TensorEnv(
            num_envs=1,
            config=self.config,
            device=self.device,
            max_ships=self.max_ships
        )
        
        # Rendering
        self._renderer = None
        self.target_fps = int(1.0 / self.physics_dt)
        self.human_ship_ids = set()
        
    @property
    def renderer(self):
        """Lazy-load renderer."""
        if self._renderer is None and self.render_mode == "human":
            # Use backend-agnostic GameRenderer
            self._renderer = GameRenderer(self.config, target_fps=self.target_fps)
        return self._renderer

    def add_human_player(self, ship_id: int) -> None:
        """
        Add human player to the renderer.
        
        Args:
            ship_id: The ID of the ship controlled by human.
        """
        self.human_ship_ids.add(ship_id)
        if self.render_mode == "human":
            self.renderer.add_human_player(ship_id)

    def remove_human_player(self, ship_id: int) -> None:
        """
        Remove human player from the renderer.
        
        Args:
            ship_id: The ID of the ship to remove.
        """
        if self.render_mode == "human":
            self.renderer.remove_human_player(ship_id)
            
    def close(self):
        """Clean up resources."""
        if self._renderer:
            self._renderer.close()

    def reset(self, game_mode: str = "nvn", **kwargs) -> tuple[dict, dict]:
        """
        Reset environment.
        
        Args:
            game_mode: The game mode string (1v1, 2v2, nvn).
            **kwargs: Additional reset options.
            
        Returns:
            A tuple (observation, info).
        """
        options = {}
        
        # Determine team sizes based on game mode
        if game_mode == "1v1":
             options["team_sizes"] = (1, 1)
        elif game_mode == "2v2":
             options["team_sizes"] = (2, 2)
        elif game_mode == "nvn":
             # Default half-half
             half = self.max_ships // self.num_teams
             options["team_sizes"] = (half, half)
        
        # Reset TensorEnv
        self.env.reset(options=options)
        
        # Check if we need to render initial frame
        if self.render_mode == "human":
             render_state = tensor_state_to_render_state(self.env.state, self.config, 0)
             self.renderer.render(render_state)
             
        return self.get_observation(), {}

    def step(self, actions: dict[int, torch.Tensor]) -> tuple[dict, dict[int, float], bool, bool, dict]:
        """
        Step environment.
        
        Args:
            actions: Dict mapping ship_id to action tensor (size 3).
            
        Returns:
            A tuple (observation, rewards, terminated, truncated, info).
        """
        # 1. Handle Human Input
        if self.render_mode == "human":
             self.renderer.handle_events()
             self.renderer.update_human_actions()
             human_actions = self.renderer.get_human_actions()
             # Update incoming actions with human ones
             actions.update(human_actions)
             
        # 2. Convert actions dict to Tensor (1, N, 3)
        batch_size = 1
        num_ships = self.max_ships
        action_tensor = torch.zeros((batch_size, num_ships, 3), dtype=torch.long, device=self.device)
        
        for ship_id, act in actions.items():
            if ship_id < num_ships:
                 act = act.to(device=self.device, dtype=torch.long)
                 action_tensor[0, ship_id] = act
                 
        # 3. Step TensorEnv
        obs_tokens, team_rewards, done_mask, _, info = self.env.step(action_tensor)
        
        terminated = bool(done_mask[0].item())
        
        # 4. Convert Rewards to Dict expected by Coordinator
        rewards = {}
        for t in range(self.num_teams):
             rewards[t] = float(team_rewards[0, t].item())
             
        # 5. Render
        if self.render_mode == "human":
             render_state = tensor_state_to_render_state(self.env.state, self.config, 0)
             self.renderer.render(render_state)
             
        # 6. Return standard tuple
        obs_dict = self.get_observation()
        
        return obs_dict, rewards, terminated, False, info

    def get_observation(self) -> dict:
        """
        Construct legacy observation dict from TensorState.
        
        Used by GameCoordinator to build tokens via `observation_to_tokens`.
        
        Returns:
             A dictionary of tensors representing the state.
        """
        state = self.env.state
        batch_idx = 0 # Single batch model in Wrapper
        
        # Helper to get flat tensor for batch 0 and move to CPU
        def to_cpu(tensor):
            return tensor[batch_idx].detach().cpu()
             
        num_ships = self.max_ships
        ship_ids = torch.arange(num_ships)
        
        obs = {
            "ship_id": ship_ids,
            "team_id": to_cpu(state.ship_team_id),
            "alive": to_cpu(state.ship_alive).long(),
            "health": to_cpu(state.ship_health).long(),
            "power": to_cpu(state.ship_power),
            "position": to_cpu(state.ship_pos), # complex
            "velocity": to_cpu(state.ship_vel), # complex
            "acceleration": torch.zeros(num_ships, dtype=torch.complex64), # Not stored in state
            "speed": to_cpu(state.ship_vel).abs(),
            "attitude": to_cpu(state.ship_attitude),
            "angular_velocity": to_cpu(state.ship_ang_vel),
            "is_shooting": to_cpu(state.ship_is_shooting).long()
        }
        
        return obs
