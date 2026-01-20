
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationalFeatureExtractor(nn.Module):
    """
    Computes pairwise relational features between ships and projects them to an attention bias.
    
    Simplified: Raw features only, lightweight MLP.
    
    Raw Features (12 dims):
    - Distance (2): Dist, InvDist
    - Angle (2): ATA, Aspect
    - Delta (4): dx, dy, dvx, dvy
    - Intercept (2): TTCA, FlightTime
    - Team (2): SameTeam, DeltaSpeed
    
    Pipeline: Raw(12) -> MLP(64) -> Bias(Heads)
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Hyperparameters
        raw_dim = 12
        hidden_dim = 64
        
        # Lightweight MLP
        self.mlp = nn.Sequential(
            nn.Linear(raw_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, num_heads) 
        )
        
    def forward(self, query_states: torch.Tensor, key_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_states: (B, N_q, D_state)
            key_states: (B, N_k, D_state)
            
        Returns:
            bias: (B, num_heads, N_q, N_k)
        """
        B, Nq, D = query_states.shape
        _, Nk, _ = key_states.shape
        
        # Expand for pairwise broadcast: (B, Nq, Nk, D)
        q = query_states.unsqueeze(2).expand(B, Nq, Nk, D)
        k = key_states.unsqueeze(1).expand(B, Nq, Nk, D)
        
        # Unpack indices (Team=0, Pos=3-5, Vel=5-7, Att=7-9)
        q_team = q[..., 0]
        k_team = k[..., 0]
        
        q_pos = q[..., 3:5]
        k_pos = k[..., 3:5]
        
        q_vel = q[..., 5:7]
        k_vel = k[..., 5:7]
        
        q_att = q[..., 7:9]
        k_att_vec = k[..., 7:9]
        
        # --- Raw Feature Computation ---
        
        # 1. Delta
        delta_pos = k_pos - q_pos
        delta_vel = k_vel - q_vel
        
        # 2. Distance
        dist_sq = delta_pos.pow(2).sum(dim=-1, keepdim=True)
        dist = torch.sqrt(dist_sq + 1e-6)
        inv_dist = 1.0 / (dist + 0.1)
        
        # 3. Angle
        bearing = delta_pos / (dist + 1e-6)
        ata = (bearing * q_att).sum(dim=-1, keepdim=True)
        aspect = (bearing * k_att_vec).sum(dim=-1, keepdim=True)
        
        # 4. Intercept
        dot_dp_dv = (delta_pos * delta_vel).sum(dim=-1, keepdim=True)
        speed_sq = delta_vel.pow(2).sum(dim=-1, keepdim=True)
        ttca = -dot_dp_dv / (speed_sq + 1e-6)
        ttca = torch.tanh(torch.relu(ttca))
        flight_time = dist # Normalized distance proxy
        
        # 5. Team / Speed
        same_team = (q_team == k_team).float().unsqueeze(-1)
        q_speed = q_vel.norm(dim=-1, keepdim=True)
        k_speed = k_vel.norm(dim=-1, keepdim=True)
        delta_speed = k_speed - q_speed
        
        # Concatenate All Raw Features
        raw_features = torch.cat([
            dist, inv_dist,
            ata, aspect,
            delta_pos, delta_vel,
            ttca, flight_time,
            same_team, delta_speed
        ], dim=-1)
        
        # Linear Projection
        bias = self.mlp(raw_features) # (..., Heads)
        
        # Permute to (B, Heads, Nq, Nk)
        bias = bias.permute(0, 3, 1, 2)
        
        return bias

