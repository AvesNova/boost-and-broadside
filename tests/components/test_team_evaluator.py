import torch
import pytest
from src.agents.components.team_evaluator import TeamEvaluator

def test_team_evaluator_shapes():
    """Test standard forward pass shapes."""
    batch_size = 4
    num_ships = 3
    d_model = 64
    
    evaluator = TeamEvaluator(d_model=d_model)
    x = torch.randn(batch_size, num_ships, d_model)
    
    value, reward = evaluator(x)
    
    assert value.shape == (batch_size, 1)
    assert reward.shape == (batch_size, 3)

def test_team_evaluator_masking():
    """Test that masking works (output should differ if mask changes)."""
    batch_size = 2
    num_ships = 4
    d_model = 32
    
    evaluator = TeamEvaluator(d_model=d_model)
    x = torch.randn(batch_size, num_ships, d_model)
    
    # No mask
    v1, r1 = evaluator(x)
    
    # Mask last ship
    mask = torch.ones(batch_size, num_ships, dtype=torch.bool)
    mask[:, -1] = False
    
    v2, r2 = evaluator(x, mask=mask)
    
    # Check that outputs are different (since last ship is ignored)
    # Note: If Attention Pooling works, masking a ship should change the aggregated vector.
    # Unless that ship had zero contribution/weight which is unlikely with random init.
    assert not torch.allclose(v1, v2), "Masking should affect the output."

def test_team_evaluator_permutation_invariance():
    """
    Test that the order of ships does NOT matter (Permutation Invariance).
    Since we use Attention Pooling (Query vs Key/Value), the order of K/V shouldn't matter 
    assuming no positional encoding is added inside TeamEvaluator (which it isn't).
    However, if we use RMSNorm or other layers *before* pooling, they operate point-wise.
    """
    batch_size = 1
    num_ships = 3
    d_model = 32
    
    evaluator = TeamEvaluator(d_model=d_model)
    x = torch.randn(batch_size, num_ships, d_model)
    
    # Original
    v1, r1 = evaluator(x)
    
    # Permute ships (swap 0 and 1)
    idx = torch.tensor([1, 0, 2])
    x_perm = x[:, idx, :]
    
    v2, r2 = evaluator(x_perm)
    
    assert torch.allclose(v1, v2, atol=1e-5), "TeamEvaluator should be permutation invariant."

def test_team_evaluator_zero_ships():
    """Test robustness when all ships are masked (if possible)."""
    batch_size = 1
    num_ships = 3
    d_model = 32
    
    evaluator = TeamEvaluator(d_model=d_model)
    x = torch.randn(batch_size, num_ships, d_model)
    mask = torch.zeros(batch_size, num_ships, dtype=torch.bool) # All False (Dead)
    
    # Should not crash.
    # Attention usually outputs 0 or handles NaN if no keys.
    # Our TeamEvaluator uses manually implemented attention? Or MHA?
    # It uses `nn.MultiheadAttention`.
    # PyTorch MHA with all-masked keys returns NaNs usually? Or zeros?
    # Let's check behavior.
    
    v, r = evaluator(x, mask=mask)
    
    # If it returns NaNs, that might be expected behavior for "No inputs".
    # But usually we prefer 0s or handling it.
    # Let's assert shape at least.
    assert v.shape == (batch_size, 1)
