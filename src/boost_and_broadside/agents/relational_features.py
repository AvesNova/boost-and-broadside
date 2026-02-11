
import torch
import torch.nn as nn


class RelationalFeatureExtractor(nn.Module):
    """
    Computes pairwise relational features between ships and projects them to an attention bias.
    
    Modified to accept precomputed fundamental relative features (4D) in local frame.
    
    Input Features (4 dims):
    - rel_pos_x, rel_pos_y (Local Frame)
    - rel_vel_x, rel_vel_y (Local Frame)
    
    Derived Features (computed on the fly):
    - Distance, Inverse Distance
    - Closing Speed (Dot product)
    - Relative Speed
    - Direction Sine/Cosine
    
    Pipeline: Raw(4) -> Derived(N) -> MLP(64) -> Bias(Heads)
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Hyperparameters
        # We input 4 raw + derived.
        # Derived: 
        # 1. Dist
        # 2. InvDist
        # 3. RelSpeed
        # 4. ClosingSpeed (Dot/Dist)
        # 5. ATA (Angle to Agent) -> Since in local frame, this is just atan2(pos). We use cos/sin of it -> Normalized Pos (2)
        # Total: 4 (raw) + 1 + 1 + 1 + 1 = 8 dims.
        # Let's use 12 to be safe/rich (add squares, log dist, etc)
        
        input_dim = 12 
        hidden_dim = 64
        
        # Lightweight MLP
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_heads) 
        )
        
    def compute_features(self, fundamental_features: torch.Tensor) -> torch.Tensor:
        """
        Compute the 12D feature vector from the 4D fundamental features.
        
        Args:
            fundamental_features: (..., 4) [rx, ry, rvx, rvy] in local frame.
            
        Returns:
            features: (..., 12) Full feature vector.
        """
        # Unpack
        # (..., 4)
        rx = fundamental_features[..., 0]
        ry = fundamental_features[..., 1]
        rvx = fundamental_features[..., 2]
        rvy = fundamental_features[..., 3]
        
        # 1. Distance & Direction
        # Add epsilon to gradients
        dist_sq = rx*rx + ry*ry
        dist = torch.sqrt(dist_sq + 1e-6)
        inv_dist = 1.0 / (dist + 0.1)
        
        # Normalized Direction (Cos/Sin of ATA)
        dir_x = rx / (dist + 1e-6)
        dir_y = ry / (dist + 1e-6)
        
        # 2. Velocity Info
        speed_sq = rvx*rvx + rvy*rvy
        rel_speed = torch.sqrt(speed_sq + 1e-6)
        
        # 3. Interaction
        # Closing Speed: Proj of RelVel onto RelPos
        # V . P / |P|
        dot_vp = rvx*rx + rvy*ry
        closing_speed = dot_vp / (dist + 1e-6)
        
        # Time to Closest Approach (TTCA) proxy
        # - dot / speed_sq
        ttca = -dot_vp / (speed_sq + 1e-6)
        ttca = torch.tanh(ttca) # Bound it
        
        # 4. Construct Feature Vector
        # We include raw values and derived ones
        features = torch.stack([
            rx, ry, rvx, rvy,      # 4 Raw
            dist, inv_dist,        # 2 Dist
            rel_speed, closing_speed, # 2 Speed
            dir_x, dir_y,          # 2 Direction
            ttca, dot_vp           # 2 Interaction
        ], dim=-1)
        
        return features

    def project_features(self, features: torch.Tensor) -> torch.Tensor:
        """
        Project 12D features to attention bias.
        
        Args:
            features: (..., 12)
            
        Returns:
            bias: (..., Heads)
        """
        bias = self.mlp(features) # (..., Heads)
        
        # Determine permute dims based on input rank
        # If input (B, N, N, 12) -> bias (B, N, N, H) -> Output (B, H, N, N)
        # If input (B*T, N, N, 12) -> bias (B*T, N, N, H) -> Output (B*T, H, N, N)
        
        if bias.ndim == 4:
            bias = bias.permute(0, 3, 1, 2)
        elif bias.ndim == 5: # (B, T, N, N, H)
             bias = bias.permute(0, 4, 1, 2, 3) 
             
        return bias

    def forward(self, fundamental_features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            fundamental_features: (..., 4) [rx, ry, rvx, rvy]
            
        Returns:
            bias: (..., Heads, N, N)
        """
        features = self.compute_features(fundamental_features)
        return self.project_features(features)
