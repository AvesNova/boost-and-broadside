import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from boost_and_broadside.models.components.layers.utils import RMSNorm

class StateEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
    def forward(self, x):
        return self.net(x)

class ActionEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb_power = nn.Embedding(3, 32)
        self.emb_turn = nn.Embedding(7, 64)
        self.emb_shoot = nn.Embedding(2, 32)
        
    def forward(self, action):
        """
        action: (..., 3) [power_idx, turn_idx, shoot_idx]
        """
        p = self.emb_power(action[..., 0].long().clamp(0, 2))
        t = self.emb_turn(action[..., 1].long().clamp(0, 6))
        s = self.emb_shoot(action[..., 2].long().clamp(0, 1))
        return torch.cat([p, t, s], dim=-1)

class RelationalEncoder(nn.Module):
    """
    Physics Trunk: Projects raw geometry into latent space.
    Shared across layers via adapters.
    """
    def __init__(self, d_model: int, n_layers: int, num_bands: int = 8):
        super().__init__()
        self.num_bands = num_bands
        self.raw_dim = 11 + (2 * self.num_bands * 2) # 11 + 32 = 43.
        # Pad to 64 for nice numbers
        self.padded_dim = 64
        
        # Trunk
        self.trunk = nn.Sequential(
            nn.Linear(self.padded_dim, 128),
            RMSNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        
        # Adapters for each layer (1 Actor + N_Layers World Model)
        # We assume n_layers passed is just the world model layers, so we add 1 for Actor
        self.adapters = nn.ModuleList([
            nn.Linear(128, d_model) for _ in range(n_layers + 1)
        ])

    def compute_analytic_features(self, pos, vel, att=None, world_size=(1024.0, 1024.0)):
        """
        Compute analytic edge features from Pos/Vel/Attitude.
        Pos: (B, T, N, 2)
        Vel: (B, T, N, 2)
        Att: (B, T, N, 2) [cos, sin]
        """
        # Delta Pos (Wrapped)
        pos = pos.to(torch.float32)
        
        d_pos = pos.unsqueeze(-2) - pos.unsqueeze(-3) # (..., N, N, 2)
        
        W, H = world_size
        dx = d_pos[..., 0]
        dy = d_pos[..., 1]
        dx = dx - torch.round(dx / W) * W
        dy = dy - torch.round(dy / H) * H
        d_pos = torch.stack([dx, dy], dim=-1)
        
        # Delta Vel
        d_vel = vel.unsqueeze(-2) - vel.unsqueeze(-3)
        
        # Derived
        dist_sq = d_pos.pow(2).sum(dim=-1, keepdim=True)
        dist = dist_sq.sqrt() + 1e-6
        
        # Standard features
        eps = 1e-6
        dir = d_pos / (dist + eps)
        rel_speed = d_vel.pow(2).sum(dim=-1, keepdim=True).sqrt()
        closing = (d_vel * d_pos).sum(dim=-1, keepdim=True) / (dist + eps)
        inv_dist = 1.0 / (dist + 0.1) # 0.1 is safe enough
        log_dist = torch.log(dist + 1.0)
        tti = dist / (closing.clamp(min=1e-3)) # Time to impact can be large, clamp denom
        
        # Relational Geometry (ATA, AA, HCA)
        if att is not None:
             # att is (..., N, 2) [cos, sin]
             att_i = att.unsqueeze(-2) # (..., N, 1, 2)
             att_j = att.unsqueeze(-3) # (..., 1, N, 2)
             
             # ATA: Angle between my heading and target
             cos_ata = (dir * att_i).sum(dim=-1, keepdim=True)
             sin_ata = (dir[..., 0] * att_i[..., 1] - dir[..., 1] * att_i[..., 0]).unsqueeze(-1)
             
             # AA: Angle between target's heading and me (from target's perspective)
             cos_aa = (-dir * att_j).sum(dim=-1, keepdim=True)
             sin_aa = (-dir[..., 0] * att_j[..., 1] + dir[..., 1] * att_j[..., 0]).unsqueeze(-1)
             
             # HCA: Angle between our headings
             cos_hca = (att_i * att_j).sum(dim=-1, keepdim=True)
             sin_hca = (att_i[..., 0] * att_j[..., 1] - att_i[..., 1] * att_j[..., 0]).unsqueeze(-1)
        else:
             # Fallback if no attitude provided (e.g. initial steps of dreaming)
             cos_ata = sin_ata = cos_aa = sin_aa = cos_hca = sin_hca = torch.zeros_like(dist)

        # Fourier Encoding of d_pos (Normalized roughly by world size/scale)
        scale = 2 * math.pi / 1024.0
        scaled_pos = d_pos * scale
        
        fourier_feats = []
        for i in range(self.num_bands):
            freq = 2 ** i
            fourier_feats.append(torch.sin(scaled_pos * freq))
            fourier_feats.append(torch.cos(scaled_pos * freq))
        fourier = torch.cat(fourier_feats, dim=-1) # (..., 32)
        
        # Concatenate: 2(pos)+2(vel)+1(dist)+1(inv)+1(speed)+1(close)+2(dir)+1(log)+1(tti) + 6(geom) = 18 Base
        base_features = torch.cat([
            d_pos, d_vel, dist, inv_dist, 
            rel_speed, closing, dir, log_dist, tti,
            cos_ata, sin_ata, cos_aa, sin_aa, cos_hca, sin_hca
        ], dim=-1)
        
        # Final feature assembly
        fourier = fourier.to(torch.bfloat16)
        features = torch.cat([base_features.to(fourier.dtype), fourier], dim=-1) # 18 + 32 = 50
        
        # Pad to 64
        pad_size = self.padded_dim - features.shape[-1]
        if pad_size > 0:
            features = F.pad(features, (0, pad_size))
        
        # Robust cast to ensure compatibility with Master Weights (Float) when not in Autocast
        target_dtype = self.trunk[0].weight.dtype
        return features.to(target_dtype)

    def forward(self, pos, vel, att=None, layer_idx=None, world_size=(1024.0, 1024.0)):
        raw = self.compute_analytic_features(pos, vel, att=att, world_size=world_size)
        trunk_out = self.trunk(raw)
        
        if layer_idx is not None:
            # Adapter: Linear -> No Act -> Bias for Attention
            return self.adapters[layer_idx](trunk_out)
        return trunk_out, raw
