import torch
import pytest
from agents.interleaved_world_model import GatedStateEncoder, GatedSwiGLU

def test_gated_swiglu_shape():
    batch_size = 4
    in_features = 32
    hidden_features = 64
    out_features = 32
    
    model = GatedSwiGLU(in_features, hidden_features, out_features)
    x = torch.randn(batch_size, in_features)
    y = model(x)
    
    assert y.shape == (batch_size, out_features)

def test_gated_state_encoder_shape():
    batch_size = 4
    input_dim = 10
    embed_dim = 128
    
    model = GatedStateEncoder(input_dim=input_dim, embed_dim=embed_dim)
    x = torch.randn(batch_size, input_dim)
    y = model(x)
    
    # Expected output: (Batch, EmbedDim)
    assert y.shape == (batch_size, embed_dim)

def test_gated_state_encoder_fourier_integration():
    """
    Ensure the Fourier features are actually being generated and used.
    """
    input_dim = 4
    embed_dim = 128
    model = GatedStateEncoder(input_dim=input_dim, embed_dim=embed_dim)
    
    x = torch.zeros(1, input_dim)
    y_zeros = model(x)
    
    x_ones = torch.ones(1, input_dim)
    y_ones = model(x_ones)
    
    # Outputs should be different given different inputs
    assert not torch.allclose(y_zeros, y_ones)
