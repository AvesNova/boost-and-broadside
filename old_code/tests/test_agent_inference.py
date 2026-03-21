import torch
import pytest
from omegaconf import OmegaConf
from boost_and_broadside.models.yemong.scaffolds import YemongDynamicsInterleaved
from boost_and_broadside.agents.world_model_agent import WorldModelAgent

def test_world_model_agent_inference():
    # Setup dummy model
    from boost_and_broadside.core.constants import STATE_DIM
    config = OmegaConf.create({
        "d_model": 32,
        "n_layers": 1,
        "n_heads": 2,
        "input_dim": STATE_DIM,
        "action_dim": 12,
        "target_dim": 14,
        "use_soft_bin_targets": False,
        "max_ships": 2,
        "loss": {
            "_target_": "boost_and_broadside.models.components.losses.CompositeLoss",
            "losses": []
        }
    })
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = YemongDynamicsInterleaved(config).to(device)
    
    # Initialize Agent
    agent = WorldModelAgent(
        agent_id="test_agent",
        team_id=0,
        squad=[0, 1],
        model=model,
        max_ships=2,
        device=str(device)
    )
    
    # Dummy Observation
    N = 2
    observation = {
        "health": torch.ones(N),
        "power": torch.zeros(N),
        "velocity": torch.zeros(N, dtype=torch.complex64),
        "angular_velocity": torch.zeros(N),
        "position": torch.zeros(N, dtype=torch.complex64),
    }
    
    try:
        actions = agent(observation, ship_ids=[0, 1])
        assert isinstance(actions, dict)
        assert len(actions) == 2
        assert actions[0].shape == (3,) # [p, t, s]
        print("WorldModelAgent Inference Test Passed!")
    except Exception as e:
        pytest.fail(f"Agent Inference failed: {e}")

if __name__ == "__main__":
    test_world_model_agent_inference()
