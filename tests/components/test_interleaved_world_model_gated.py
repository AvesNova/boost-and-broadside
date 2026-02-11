import torch
import pytest
from boost_and_broadside.agents.interleaved_world_model import InterleavedWorldModel

def test_interleaved_world_model_instantiation():
    model = InterleavedWorldModel(
        state_dim=64,
        action_dim=12,
        embed_dim=128,
        n_layers=2,
        n_heads=4
    )
    assert model is not None
    assert hasattr(model, 'state_encoder')
    assert model.state_encoder.__class__.__name__ == 'GatedStateEncoder'

def test_interleaved_world_model_forward():
    batch_size = 2
    time_steps = 8
    n_ships = 2
    state_dim = 64
    action_dim = 3 # Only using components for shape generation, but input is usually (B, T, N, 3) 
    # Wait, action input to forward is (B, T, N, 3)?
    # Let's check forward signature: actions: (B, T, N, 3)
    
    model = InterleavedWorldModel(
        state_dim=state_dim,
        embed_dim=128,
        n_layers=2
    )
    
    states = torch.randn(batch_size, time_steps, n_ships, state_dim)
    actions = torch.randint(0, 2, (batch_size, time_steps, n_ships, 3))
    # Fix action ranges: Power(3), Turn(7), Shoot(2)
    actions[..., 0] = torch.randint(0, 3, (batch_size, time_steps, n_ships))
    actions[..., 1] = torch.randint(0, 7, (batch_size, time_steps, n_ships))
    actions[..., 2] = torch.randint(0, 2, (batch_size, time_steps, n_ships))
    
    team_ids = torch.randint(0, 2, (batch_size, n_ships))
    
    pred_states, pred_actions, _ = model(states, actions, team_ids)
    
    assert pred_states.shape == (batch_size, time_steps, n_ships, state_dim)
    # pred_actions typically returns logits for all components concatenated?
    # forward returns pred_actions (B, T, N, 3+7+2=12)
    assert pred_actions.shape == (batch_size, time_steps, n_ships, 12)
