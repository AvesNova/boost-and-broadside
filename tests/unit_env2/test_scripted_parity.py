
import torch
import numpy as np
import pytest

from boost_and_broadside.env2.state import TensorState, ShipConfig
from boost_and_broadside.env2.agents.scripted import VectorScriptedAgent
try:
    from boost_and_broadside.agents.scripted import ScriptedAgent 
except ImportError:
    # Fallback if src not in path but running from root
    from boost_and_broadside.agents.scripted import ScriptedAgent

class TestScriptedParity:
    def create_state(self, config, B=2, N=2):
        device = torch.device("cpu")
        w, h = config.world_size
        
        # Random Init
        ship_pos = torch.rand((B, N), dtype=torch.float32, device=device) + 1j * torch.rand((B, N), dtype=torch.float32, device=device)
        ship_pos.real *= w
        ship_pos.imag *= h
        
        ship_vel = torch.randn((B, N), dtype=torch.complex64, device=device) * 10.0
        ship_attitude = torch.randn((B, N), dtype=torch.complex64, device=device)
        ship_attitude = ship_attitude / (torch.abs(ship_attitude) + 1e-8)
        
        ship_team_id = torch.zeros((B, N), dtype=torch.int32, device=device)
        # 0 vs 1
        ship_team_id[:, 1::2] = 1 
        
        return TensorState(
            step_count=torch.zeros((B,), dtype=torch.int32, device=device),
            ship_pos=ship_pos,
            ship_vel=ship_vel,
            ship_attitude=ship_attitude,
            ship_ang_vel=torch.zeros((B, N), dtype=torch.float32, device=device),
            ship_health=torch.ones((B, N), dtype=torch.float32, device=device) * config.max_health,
            ship_power=torch.ones((B, N), dtype=torch.float32, device=device) * config.max_power,
            ship_cooldown=torch.zeros((B, N), dtype=torch.float32, device=device),
            ship_team_id=ship_team_id,
            ship_alive=torch.ones((B, N), dtype=torch.bool, device=device),
            ship_is_shooting=torch.zeros((B, N), dtype=torch.bool, device=device),
            
            bullet_pos=torch.zeros((B, N, 5), dtype=torch.complex64, device=device),
            bullet_vel=torch.zeros((B, N, 5), dtype=torch.complex64, device=device),
            bullet_time=torch.zeros((B, N, 5), dtype=torch.float32, device=device),
            bullet_active=torch.zeros((B, N, 5), dtype=torch.bool, device=device),
            bullet_cursor=torch.zeros((B, N), dtype=torch.long, device=device)
        )

    def test_parity(self):
        config = ShipConfig()
        B, N = 4, 2
        state = self.create_state(config, B, N)
        
        # 1. Run Original ScriptedAgent (Loop over B)
        # Initialize Agent
        original_agent = ScriptedAgent(
            max_shooting_range=600.0, # Default in vector agent
            angle_threshold=10.0, # degrees
            bullet_speed=config.bullet_speed,
            target_radius=20.0,
            radius_multiplier=1.0,
            world_size=config.world_size,
            rng=np.random.default_rng(42)
        )
        
        # We need to construct obs_dict for each env
        original_actions = []
        
        for b in range(B):
            # Extract env slice
            # obs_dict keys: ship_id, team_id, alive, health, power, position, velocity, attitude
            # shapes (N,)
            obs_dict = {
                "ship_id": torch.arange(N), # 0..N-1
                "team_id": state.ship_team_id[b],
                "alive": state.ship_alive[b],
                "health": state.ship_health[b],
                "power": state.ship_power[b],
                "position": state.ship_pos[b],
                "velocity": state.ship_vel[b],
                "attitude": state.ship_attitude[b]
            }
            
            # Run Agent
            # get_actions takes (obs_dict, ship_ids)
            # ship_ids list
            ids = list(range(N))
            actions_dict = original_agent.get_actions(obs_dict, ids)
            
            # Convert to tensor (N, 3)
            # Key 0..N-1
            act_rows = []
            for i in range(N):
                act_rows.append(actions_dict[i])
            original_actions.append(torch.stack(act_rows))
            
        original_actions_tensor = torch.stack(original_actions) # (B, N, 3)
            
        # 2. Run VectorScriptedAgent
        vector_agent = VectorScriptedAgent(config)
        vector_actions = vector_agent.get_actions(state) # (B, N, 3)
        
        # 3. Compare
        # Turn actions might differ if angles are exactly threshold or due to float precision?
        # Shoot actions same.
        # Check agreement rate.
        
        # Power
        diff_power = (original_actions_tensor[..., 0] != vector_actions[..., 0]).float().mean()
        # Turn
        diff_turn = (original_actions_tensor[..., 1] != vector_actions[..., 1]).float().mean()
        # Shoot
        diff_shoot = (original_actions_tensor[..., 2] != vector_actions[..., 2]).float().mean()
        
        print(f"Power Diff Rate: {diff_power:.4f}")
        print(f"Turn Diff Rate: {diff_turn:.4f}")
        print(f"Shoot Diff Rate: {diff_shoot:.4f}")
        
        # We expect very high agreement (>90% or >99%).
        # If float precision causes minor angle diffs around threshold, some divergence expected.
        # But robust logic should match.
        
        # Allow some small divergence
        assert diff_power < 0.05
        assert diff_turn < 0.05
        assert diff_shoot < 0.05

if __name__ == "__main__":
    t = TestScriptedParity()
    t.test_parity()
