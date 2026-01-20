
import pytest
import torch
import torch.nn as nn
from agents.relational_features import RelationalFeatureExtractor

@pytest.fixture
def extractor():
    return RelationalFeatureExtractor(embed_dim=128, num_heads=4)

def test_shape_compatibility(extractor):
    B, Nq, Nk, D = 2, 8, 16, 10
    
    q_states = torch.randn(B, Nq, D)
    k_states = torch.randn(B, Nk, D)
    
    bias = extractor(q_states, k_states)
    
    # Expected shape: (B, num_heads, Nq, Nk)
    assert bias.shape == (B, 4, Nq, Nk)

def test_distance_feature(extractor):
    # Manually construction
    # q at (0, 0), k at (3, 4) -> dist 5
    B, Nq, Nk, D = 1, 1, 1, 10
    q_states = torch.zeros(B, Nq, D) # Pos at indices 3, 4 is 0,0
    k_states = torch.zeros(B, Nk, D)
    k_states[0, 0, 3] = 3.0
    k_states[0, 0, 4] = 4.0
    
    # We can't easily probe intermediate features without hooking or exposing them.
    # But we can check that if we move K further, the bias changes.
    
    bias1 = extractor(q_states, k_states)
    
    k_states_far = k_states.clone()
    k_states_far[0, 0, 3] = 6.0
    k_states_far[0, 0, 4] = 8.0 # Dist 10
    
    bias2 = extractor(q_states, k_states_far)
    
    assert not torch.allclose(bias1, bias2)

def test_gradient_flow(extractor):
    B, Nq, Nk, D = 2, 4, 4, 10
    q_states = torch.randn(B, Nq, D, requires_grad=True)
    k_states = torch.randn(B, Nk, D, requires_grad=True)
    
    bias = extractor(q_states, k_states)
    loss = bias.sum()
    loss.backward()
    
    assert q_states.grad is not None
    assert k_states.grad is not None
    # Ensure gradients are non-zero (implies features effectively use inputs)
    assert torch.any(q_states.grad != 0)
    assert torch.any(k_states.grad != 0)

