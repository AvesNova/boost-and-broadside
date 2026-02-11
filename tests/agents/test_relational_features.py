
import pytest
import torch
from boost_and_broadside.agents.relational_features import RelationalFeatureExtractor

@pytest.fixture
def extractor():
    return RelationalFeatureExtractor(embed_dim=128, num_heads=4)

def test_shape_compatibility(extractor):
    B, Nq, Nk = 2, 8, 16
    # Input is now (B, Nq, Nk, 4)
    fundamental_features = torch.randn(B, Nq, Nk, 4)
    
    bias = extractor(fundamental_features)
    
    # Expected shape: (B, num_heads, Nq, Nk)
    assert bias.shape == (B, 4, Nq, Nk)

def test_distance_feature(extractor):
    # Manually construction
    # Feature 0, 1 are rel_pos_x, rel_pos_y. 
    # (3, 4) -> Dist 5
    B, Nq, Nk = 1, 1, 1
    features_near = torch.zeros(B, Nq, Nk, 4)
    features_near[0, 0, 0, 0] = 3.0
    features_near[0, 0, 0, 1] = 4.0
    
    bias1 = extractor(features_near)
    
    # (6, 8) -> Dist 10
    features_far = torch.zeros(B, Nq, Nk, 4)
    features_far[0, 0, 0, 0] = 6.0
    features_far[0, 0, 0, 1] = 8.0
    
    bias2 = extractor(features_far)
    
    assert not torch.allclose(bias1, bias2)

def test_gradient_flow(extractor):
    B, Nq, Nk = 2, 4, 4
    fundamental_features = torch.randn(B, Nq, Nk, 4, requires_grad=True)
    
    bias = extractor(fundamental_features)
    loss = bias.sum()
    loss.backward()
    
    assert fundamental_features.grad is not None
    # Ensure gradients are non-zero
    assert torch.any(fundamental_features.grad != 0)
