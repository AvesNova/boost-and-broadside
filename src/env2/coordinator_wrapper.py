
import torch
import numpy as np
from src.env2.env import TensorEnv
from src.env2.adapter import tensor_state_to_cpu_state
from env.renderer import create_renderer
from env.constants import RewardConstants
from env.ship import default_ship_config

class TensorEnvWrapper:
    """
    Wraps TensorEnv to provide the API expected by GameCoordinator.
    Handles legacy observation format (dict of tensors) and human rendering integration.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize TensorEnvWrapper.
        Accepts same kwargs as Environment but filters/adapts them.
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Config params
        self.render_mode = kwargs.get("render_mode", "none")
        self.world_size = kwargs.get("world_size", (2000, 2000))
        self.max_ships = kwargs.get("max_ships", 16)
        self.agent_dt = kwargs.get("agent_dt", 0.015) # Assume 1:1 if not specified?
        self.physics_dt = kwargs.get("physics_dt", 0.015)
        self.num_teams = kwargs.get("num_teams", 2)
        
        # TensorEnv supports B=1 for single game play
        self.env = TensorEnv(
            num_envs=1,
            device=self.device,
            max_ships=self.max_ships,
            dt=self.physics_dt,
            world_size=tuple(float(x) for x in self.world_size)
        )
        
        # Rendering
        self._renderer = None
        self.target_fps = int(1.0 / self.physics_dt)
        self.human_ship_ids = set()
        
    @property
    def renderer(self):
        """Lazy-load renderer."""
        if self._renderer is None and self.render_mode == "human":
            # Reuse create_renderer from legacy code
            self._renderer = create_renderer(self.world_size, self.target_fps)
        return self._renderer

    def add_human_player(self, ship_id: int) -> None:
        """Add human player."""
        self.human_ship_ids.add(ship_id)
        if self.render_mode == "human":
            self.renderer.add_human_player(ship_id)

    def remove_human_player(self, ship_id: int) -> None:
        if self.render_mode == "human":
            self.renderer.remove_human_player(ship_id)
            
    def close(self):
        if self._renderer:
            self._renderer.close()

    def reset(self, game_mode: str = "nvn", **kwargs) -> tuple[dict, dict]:
        """Reset environment."""
        # Convert game_mode to TensorEnv options
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
        # default fallback handled by env
        
        # Reset TensorEnv
        self.env.reset(options=options)
        
        # Check if we need to render initial frame
        if self.render_mode == "human":
             cpu_state = tensor_state_to_cpu_state(self.env.state, 0)
             self.renderer.render(cpu_state)
             
        return self.get_observation(), {}

    def step(self, actions: dict[int, torch.Tensor]) -> tuple[dict, dict[int, float], bool, bool, dict]:
        """
        Step environment.
        Args:
            actions: Dict {ship_id: Tensor(3,)} (likely CPU tensors from agents)
        """
        # 1. Handle Human Input
        if self.render_mode == "human":
             self.renderer.handle_events()
             self.renderer.update_human_actions()
             human_actions = self.renderer.get_human_actions()
             # Update incoming actions with human ones
             actions.update(human_actions)
             
        # 2. Convert actions dict to Tensor (1, N, 3)
        # Ensure we cover all alive ships
        B = 1
        N = self.max_ships
        action_tensor = torch.zeros((B, N, 3), dtype=torch.long, device=self.device)
        
        # We need to map discrete actions from agents to tensor
        # Legacy agents return torch tensors (often on CPU).
        # Action space: [Power, Turn, Shoot]
        
        for ship_id, act in actions.items():
            if ship_id < N:
                 # Act might be float or int tensor
                 # GameRenderer returns float tensor. Agents map return discrete ints?
                 # GameCoordinator passes what agents return.
                 # Let's cast to long.
                 act = act.to(device=self.device, dtype=torch.long)
                 action_tensor[0, ship_id] = act
                 
        # 3. Step TensorEnv
        # Note: TensorEnv automatically handles time update
        obs_tokens, team_rewards, done_mask, _, info = self.env.step({"action": action_tensor})
        
        terminated = bool(done_mask[0].item())
        
        # 4. Convert Rewards to Dict expected by Coordinator
        # Coordinator expects {team_id: float_total_reward}
        # TensorEnv returns (B, NumTeams)
        rewards = {}
        for t in range(self.num_teams):
             rewards[t] = float(team_rewards[0, t].item())
             
        # 5. Render
        if self.render_mode == "human":
             cpu_state = tensor_state_to_cpu_state(self.env.state, 0)
             self.renderer.render(cpu_state)
             
        # 6. Return standard tuple
        # Observation must be the Legacy Dict format
        obs_dict = self.get_observation()
        
        return obs_dict, rewards, terminated, False, info

    def get_observation(self) -> dict:
        """
        Construct legacy observation dict from TensorState.
        Used by GameCoordinator to build tokens via `observation_to_tokens`.
        """
        s = self.env.state
        B = 0 # Single batch
        
        # We need to construct a dict where keys are "ship_id", "team_id", etc.
        # and values are Tensors of shape (N,).
        
        # Helper to get flat tensor for batch 0
        def get(attr):
             return attr[B].detach().cpu() # Move to CPU as Legacy code usually expects CPU processing
             
        N = self.max_ships
        
        # Keys from Environment._get_empty_observation map
        
        # ship_id: 0..N-1
        ship_ids = torch.arange(N)
        
        # Complex positions/vels need to certainly be CPU if downstream uses numpy-like ops
        # GameCoordinator uses `observation_to_tokens` which uses `torch` but checks constraints.
        
        obs = {
            "ship_id": ship_ids, # (N,)
            "team_id": get(s.ships_team), # (N,)
            "alive": get(s.ships_alive).long(), # (N,)
            "health": get(s.ships_health).long(), # (N,)
            "power": get(s.ships_power), # (N,)
            "position": get(s.ships_pos), # (N,) complex
            "velocity": get(s.ships_vel), # (N,) complex
            "acceleration": get(s.ships_acc),
            "speed": get(s.ships_vel).abs(),
            "attitude": get(s.ships_attitude),
            "angular_velocity": get(s.ships_ang_vel),
            # "is_shooting" is not explicitly stored in state, derived in Env.
            # But get_observations in TensorEnv computed it in tokens?
            # We can approximate from cooldown?
            # Or use logic similarly:
            "is_shooting": (get(s.ships_cooldown) > (default_ship_config.firing_cooldown * 0.9)).long() 
        }
        
        # Reshape to (N,) or (N, 1) or whatever legacy expects?
        # Env.get_empty_observation returns items of shape (max_ships).
        # GameCoordinator checks `obs["alive"][ship_id]`.
        # So (N,) is correct.
        
        return obs
