
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DyadicFourierFeatureExtractor(nn.Module):
    """
    Maps continuous inputs to higher dimensional space using Dyadic Fourier Features.
    Features: [sin(2^k * pi * x), cos(2^k * pi * x)] for k in frequencies.
    """
    def __init__(self, input_dim: int, embed_dim: int, num_freqs: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.num_freqs = num_freqs
        self.out_dim = input_dim * num_freqs * 2
        
        # Project to embed_dim (to input into FFN correctly)
        self.projection = nn.Linear(self.out_dim, embed_dim)
        
        # Frequencies: 2^k * PI.
        self.register_buffer("freqs", 2.0 ** torch.arange(num_freqs))

    def forward(self, x):
        # x: (..., input_dim)
        scaled = x.unsqueeze(-1) * self.freqs * torch.pi
        sin_feat = torch.sin(scaled)
        cos_feat = torch.cos(scaled)
        feats = torch.stack([sin_feat, cos_feat], dim=-1)
        feats = feats.view(*x.shape[:-1], -1)
        return self.projection(feats)

class GatedSwiGLU(nn.Module):
    """
    Gated Linear Unit with Swish activation (SwiGLU).
    Maps input -> Hidden (2x) -> Output.
    """
    def __init__(self, in_features, hidden_features, out_features, dropout=0.0):
        super().__init__()
        self.w_gate = nn.Linear(in_features, hidden_features)
        self.w_val = nn.Linear(in_features, hidden_features)
        self.w_out = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.act = nn.SiLU()

    def forward(self, x):
        gate = self.act(self.w_gate(x))
        val = self.w_val(x)
        out = self.w_out(gate * val)
        return self.dropout(out)
