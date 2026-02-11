
import pytest
import torch
import os
from pathlib import Path
from omegaconf import OmegaConf

from boost_and_broadside.agents.world_model_agent import WorldModelAgent
from boost_and_broadside.agents.mamba_bb import MambaBB, MambaConfig

# Mock config
MOCK_CONFIG = {
    "world_model": {
        "state_dim": 15,
        "embed_dim": 32,
        "n_layers": 2,
        "n_heads": 2,
        "n_ships": 4, # Note: config uses n_ships, agent uses max_ships
        "max_ships": 4,
        "context_len": 16,
        "dropout": 0.0,
    }
}

@pytest.fixture
def mock_checkpoint_dir(tmp_path):
    return tmp_path / "models" / "world_model" / "run_test"

def test_load_full_checkpoint(mock_checkpoint_dir):
    """Test loading a model from a full dictionary checkpoint (.pt)."""
    mock_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a dummy model and save it as a full checkpoint
    cfg = MambaConfig(
        input_dim=15, d_model=32, n_layers=2, n_heads=2, 
        action_dim=12, target_dim=15, loss_type="fixed"
    )
    model = MambaBB(cfg)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": 1,
        "optimizer_state_dict": {},
    }
    
    ckpt_path = mock_checkpoint_dir / "final_world_model.pt"
    torch.save(checkpoint, ckpt_path)
    
    # Create agent
    agent = WorldModelAgent(
        agent_id="test_agent",
        team_id=0,
        squad=[0, 1],
        model_path=str(ckpt_path),
        state_dim=15,
        embed_dim=32,
        n_layers=2,
        n_heads=2,
        max_ships=4,
        context_len=16
    )
    
    # Verify weights loaded (check state_encoder first weight)
    saved_weight = model.state_encoder[0].weight
    loaded_weight = agent.model.state_encoder[0].weight
    
    assert torch.equal(saved_weight, loaded_weight)
    
    # Check another parameter
    saved_weight = model.actor_head[0].weight
    loaded_weight = agent.model.actor_head[0].weight
    
    assert torch.equal(saved_weight, loaded_weight)

def test_load_state_dict(mock_checkpoint_dir):
    """Test loading a model from a raw state dict (.pth)."""
    mock_checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Create a dummy model and save it as raw state dict
    cfg = MambaConfig(
        input_dim=15, d_model=32, n_layers=2, n_heads=2, 
        action_dim=12, target_dim=15, loss_type="fixed"
    )
    model = MambaBB(cfg)
    
    ckpt_path = mock_checkpoint_dir / "best_world_model.pth"
    torch.save(model.state_dict(), ckpt_path)
    
    # Create agent
    agent = WorldModelAgent(
        agent_id="test_agent",
        team_id=0,
        squad=[0, 1],
        model_path=str(ckpt_path),
        state_dim=15,
        embed_dim=32,
        n_layers=2,
        n_heads=2,
        max_ships=4,
        context_len=16
    )
    
    # Verify weights loaded
    saved_weight = model.actor_head[0].weight
    loaded_weight = agent.model.actor_head[0].weight
    
    assert torch.equal(saved_weight, loaded_weight)

def test_agent_forward_pass():
    """Test that the updated agent can process observations."""
    agent = WorldModelAgent(
        agent_id="test_agent",
        team_id=0,
        squad=[0, 1],
        state_dim=15,
        embed_dim=32,
        n_layers=2,
        n_heads=2,
        max_ships=4,
        context_len=16,
        world_size=(100.0, 100.0)
    )
    
    # Create dummy observation (Vectorized format)
    # Keys: position, velocity, attitude, health, power, team_id, is_shooting
    num_ships = 4
    obs = {
        "position": torch.zeros(num_ships, dtype=torch.complex64),
        "velocity": torch.zeros(num_ships, dtype=torch.complex64),
        "attitude": torch.zeros(num_ships, dtype=torch.complex64),
        "health": torch.ones(num_ships) * 100.0,
        "power": torch.ones(num_ships) * 50.0,
        "team_id": torch.zeros(num_ships, dtype=torch.long),
        "is_shooting": torch.zeros(num_ships, dtype=torch.bool),
        "acceleration": torch.zeros(num_ships, dtype=torch.complex64),
        "angular_velocity": torch.zeros(num_ships, dtype=torch.float32),
    }
    
    # Set different values
    for i in range(num_ships):
        obs["position"][i] = complex(10.0 * i, 10.0 * i)
        obs["team_id"][i] = 0 if i < 2 else 1
        
    ship_ids = [0, 1]
    
    actions = agent(obs, ship_ids)
    
    assert len(actions) == 2
    assert 0 in actions
    assert 1 in actions
    
    # Check shape: (3,)
    assert actions[0].shape == (3,)
