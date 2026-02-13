import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from boost_and_broadside.models.components.normalizer import FeatureNormalizer
from boost_and_broadside.core.constants import StateFeature, TargetFeature
from boost_and_broadside.models.components.layers.utils import RMSNorm

class StateEncoder(nn.Module):
    def __init__(self, input_dim: int, d_model: int, normalizer: FeatureNormalizer | None = None):
        super().__init__()
        self.normalizer = normalizer
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model),
            RMSNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        
    def forward(self, x):
        if self.normalizer:
            # apply ego state normalization
            # Vectorized normalization
            x = self.normalizer.normalize_ego(x)
            
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

class SeparatedActionEncoder(nn.Module):
    """
    Modular Action Encoder for discrete sub-actions.
    Embeds each component separately and concatenates them.
    Output dim = 3 * embed_dim
    """
    def __init__(self, embed_dim: int = 16):
        super().__init__()
        self.embed_dim = embed_dim
        self.output_dim = embed_dim * 3
        
        self.emb_power = nn.Embedding(3, embed_dim)
        self.emb_turn = nn.Embedding(7, embed_dim)
        self.emb_shoot = nn.Embedding(2, embed_dim)
        
    def forward(self, action):
        """
        action: (..., 3) [power_idx, turn_idx, shoot_idx] indices (Long)
        """
        # Ensure input is Long
        if action.dtype != torch.long:
            action = action.long()
            
        p = self.emb_power(action[..., 0].clamp(0, 2))
        t = self.emb_turn(action[..., 1].clamp(0, 6))
        s = self.emb_shoot(action[..., 2].clamp(0, 1))
        
        return torch.cat([p, t, s], dim=-1)

class RelationalEncoder(nn.Module):
    """
    Physics Trunk: Projects raw geometry into latent space.
    Shared across layers via adapters.
    """
    def __init__(self, d_model: int, n_layers: int, num_bands: int = 10, normalizer: FeatureNormalizer | None = None):
        super().__init__()
        self.num_bands = num_bands
        self.normalizer = normalizer
        # Base: 
        # dvx, dvy (2), rel_speed (1), closing_speed (1), dist (1), tti (1) -> 6
        # heading (unit vector) (2), ata (sin, cos) (2), aa (sin, cos) (2), hca (sin, cos) (2) -> 8
        # Fourier (dx, dy * 2 * num_bands) -> 2 * 10 * 2 = 40
        # Total: 6 + 8 + 40 = 54
        self.raw_dim = 14 + (2 * self.num_bands * 2) 
        self.padded_dim = 64
        
        # Trunk
        self.trunk = nn.Sequential(
            nn.Linear(self.padded_dim, 128),
            RMSNorm(128),
            nn.SiLU(),
            nn.Linear(128, 128)
        )
        
        # Adapters for each layer (1 Actor + N_Layers World Model)
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
        pos = pos.to(torch.float32)
        d_pos = pos.unsqueeze(-2) - pos.unsqueeze(-3) # (..., N, N, 2)
        
        W, H = world_size
        dx = d_pos[..., 0]
        dy = d_pos[..., 1]
        dx = dx - torch.round(dx / W) * W
        dy = dy - torch.round(dy / H) * H
        d_pos = torch.stack([dx, dy], dim=-1) # (..., N, N, 2)
        
        # Delta Vel
        d_vel = vel.unsqueeze(-2) - vel.unsqueeze(-3) # (..., N, N, 2)
        dvx = d_vel[..., 0:1]
        dvy = d_vel[..., 1:2]
        
        # 1. Distances & Speeds
        dist_sq = d_pos.pow(2).sum(dim=-1, keepdim=True)
        dist = dist_sq.sqrt() + 1e-6
        rel_speed = d_vel.pow(2).sum(dim=-1, keepdim=True).sqrt()
        closing = (d_vel * d_pos).sum(dim=-1, keepdim=True) / (dist + 1e-6)
        
        # Normalize/Transform Relational State
        if self.normalizer:
            dvx = self.normalizer.normalize(dvx, "Relational_dvx")
            dvy = self.normalizer.normalize(dvy, "Relational_dvy")
            rel_speed = self.normalizer.normalize(rel_speed, "Relational_rel_speed")
            closing = self.normalizer.normalize(closing, "Relational_closing")
            
            # dist: Log -> Z-Score
            dist_transformed = self.normalizer.transform(dist, "Log")
            dist_norm = self.normalizer.normalize(dist_transformed, "Relational_log_dist")
            
            # tti: Symlog -> Identity
            tti_val = dist / (closing.clamp(min=1e-3))
            tti_norm = self.normalizer.transform(tti_val, "Symlog")
        else:
            # Fallbacks if no normalizer (unlikely in prod)
            dist_norm = dist
            tti_norm = dist / (closing.clamp(min=1e-3))
            
        # 2. Heading (Unit Vector)
        heading = d_pos / (dist + 1e-6) # (..., 2) [x, y] unit vector
        
        # 3. Relational Geometry (ATA, AA, HCA)
        if att is not None:
             # att is (..., N, 2) [cos, sin]
             att_i = att.unsqueeze(-2) # (..., N, 1, 2)
             att_j = att.unsqueeze(-3) # (..., 1, N, 2)
             
             # ATA: Angle between my heading and target
             cos_ata = (heading * att_i).sum(dim=-1, keepdim=True)
             sin_ata = (heading[..., 0] * att_i[..., 1] - heading[..., 1] * att_i[..., 0]).unsqueeze(-1)
             
             # AA: Angle between target's heading and me (from target's perspective)
             cos_aa = (-heading * att_j).sum(dim=-1, keepdim=True)
             sin_aa = (-heading[..., 0] * att_j[..., 1] + heading[..., 1] * att_j[..., 0]).unsqueeze(-1)
             
             # HCA: Angle between our headings
             cos_hca = (att_i * att_j).sum(dim=-1, keepdim=True)
             sin_hca = (att_i[..., 0] * att_j[..., 1] - att_i[..., 1] * att_j[..., 0]).unsqueeze(-1)
        else:
             cos_ata = sin_ata = cos_aa = sin_aa = cos_hca = sin_hca = torch.zeros_like(dist)

        # 4. Fourier Encoding of d_pos (10 frequencies as per spec)
        # Identity normalization since output is bounded.
        scale = 2 * math.pi / 1024.0
        scaled_pos = d_pos * scale
        
        fourier_feats = []
        for i in range(self.num_bands):
            freq = 2 ** i
            fourier_feats.append(torch.sin(scaled_pos * freq)) # (..., 2)
            fourier_feats.append(torch.cos(scaled_pos * freq)) # (..., 2)
        fourier = torch.cat(fourier_feats, dim=-1) # (..., 40)
        
        # Assemble Final Features
        # Spec says Identity for Fourier, Heading, ATA, AA, HCA, TT-I
        features = torch.cat([
            dvx, dvy, rel_speed, closing, dist_norm, tti_norm, # 6
            heading, cos_ata, sin_ata, cos_aa, sin_aa, cos_hca, sin_hca, # 8
            fourier # 40
        ], dim=-1) # Total 54
        
        # Robust cast
        target_dtype = self.trunk[0].weight.dtype
        features = features.to(target_dtype)
        
        # Pad to 64
        pad_size = self.padded_dim - features.shape[-1]
        if pad_size > 0:
            features = F.pad(features, (0, pad_size))
        
        return features

    def forward(self, pos, vel, att=None, layer_idx=None, world_size=(1024.0, 1024.0)):
        raw = self.compute_analytic_features(pos, vel, att=att, world_size=world_size)
        trunk_out = self.trunk(raw)
        
        if layer_idx is not None:
            return self.adapters[layer_idx](trunk_out)
        return trunk_out, raw
