import torch
import torch.nn as nn
import pytest
from agents.interleaved_world_model import InterleavedWorldModel
from agents.relational_features import RelationalFeatureExtractor

def test_relational_feature_extractor_computation():
    """Verify that compute_features returns expected 12D shape."""
    B, N = 2, 3
    # 4D input: [rx, ry, rvx, rvy]
    fundamental = torch.randn(B, N, N, 4)
    
    extractor = RelationalFeatureExtractor(embed_dim=32, num_heads=2)
    features = extractor.compute_features(fundamental)
    
    assert features.shape == (B, N, N, 12)
    # Check that first 4 channels match input
    assert torch.allclose(features[..., 0:4], fundamental)

def test_world_model_relational_prediction_shape():
    """Verify that world model predicts relational features with correct shape."""
    model = InterleavedWorldModel(
        state_dim=64,
        embed_dim=128,
        n_layers=2,
        n_heads=4
    )
    
    B, T, N = 2, 5, 3
    state_dim = 64
    
    states = torch.randn(B, T, N, state_dim)
    actions = torch.randint(0, 2, (B, T, N, 3))
    team_ids = torch.randint(0, 2, (B, N))
    
    # Fake relational features (B, T, N, N, 4)
    rel_features = torch.randn(B, T, N, N, 4)
    
    # Forward pass requesting embeddings and predictions
    pred_states, pred_actions, _, _, features_12d, pred_relational = model(
        states,
        actions,
        team_ids,
        relational_features=rel_features,
        return_embeddings=True
    )
    
    assert features_12d.shape == (B, T, N, N, 12)
    assert pred_relational.shape == (B, T, N, N, 12)

def test_relational_loss_computation():
    """Verify that relational loss is computed and backpropagates."""
    model = InterleavedWorldModel(
        state_dim=64,
        embed_dim=128,
        n_layers=2,
        n_heads=4
    )
    
    B, T, N = 2, 5, 3
    state_dim = 64
    
    states = torch.randn(B, T, N, state_dim)
    actions = torch.randint(0, 2, (B, T, N, 3))
    team_ids = torch.randint(0, 2, (B, N))
    rel_features = torch.randn(B, T, N, N, 4)
    
    # Forward
    pred_states, pred_actions, _, _, features_12d, pred_relational = model(
        states,
        actions,
        team_ids,
        relational_features=rel_features,
        return_embeddings=True
    )
    
    loss_mask = torch.ones(B, T, dtype=torch.bool)
    
    # Fake targets
    target_states = torch.randn(B, T, N, state_dim)
    target_actions = torch.randint(0, 2, (B, T, N, 3))
    
    # Compute Loss
    loss, s_loss, a_loss, r_loss, metrics = model.get_loss(
        pred_states,
        pred_actions,
        target_states,
        target_actions,
        loss_mask,
        target_features_12d=features_12d, # Use computed as target
        pred_relational=pred_relational,
        lambda_relational=1.0 # High weight to ensure gradient
    )
    
    assert r_loss.item() > 0.0
    assert loss.item() > 0.0
    
    # Check gradients
    loss.backward()
    for param in model.relational_head.parameters():
        assert param.grad is not None
