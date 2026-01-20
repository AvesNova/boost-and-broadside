
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from agents.layers import DyadicFourierFeatureExtractor

class RelationalFeatureExtractor(nn.Module):
    """
    Computes pairwise relational features between ships and projects them to an attention bias.
    
    Features computed (Input Dim -> Fourier Dim):
    1. Distance (Dist, 1/Dist): 2 -> 16 (4 freqs)
    2. Angle (Aspect, ATA): 2 -> 16 (4 freqs)
    3. Delta (dx, dy, dvx, dvy): 4 -> 16 (2 freqs)
    4. Intercept (TTCA, FlightTime): 2 -> 8 (2 freqs)
    5. Team/Speed (SameTeam, dSpeed): 2 -> 8 (2 freqs)
    
    Total concatenated dimension: 64
    """
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Feature Extractors
        # 1. Distance: Input 2 (dist, inv_dist) -> 4 freqs * 2 = 8 -> Proj 16? 
        # Wait, DyadicFourierFeatureExtractor projects to a target embed_dim.
        # The prompt specified final sizes.
        
        self.feat_dist = DyadicFourierFeatureExtractor(input_dim=2, embed_dim=16, num_freqs=4)
        self.feat_angle = DyadicFourierFeatureExtractor(input_dim=2, embed_dim=16, num_freqs=4)
        self.feat_delta = DyadicFourierFeatureExtractor(input_dim=4, embed_dim=16, num_freqs=2)
        self.feat_intercept = DyadicFourierFeatureExtractor(input_dim=2, embed_dim=8, num_freqs=2)
        self.feat_team = DyadicFourierFeatureExtractor(input_dim=2, embed_dim=8, num_freqs=2)
        
        total_dim = 16 + 16 + 16 + 8 + 8 # 64
        
        # Projection to Bias
        # Map 64 -> num_heads
        # We use an MLP to allow non-linear combinations of these features
        self.mlp = nn.Sequential(
            nn.Linear(total_dim, 4 * total_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(4 * total_dim, num_heads) 
        )
        
    def forward(self, query_states: torch.Tensor, key_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_states: (B, N_q, D_state) - States of the query ships
            key_states: (B, N_k, D_state) - States of the key ships (context)
            
        Returns:
            bias: (B, num_heads, N_q, N_k)
        """
        B, Nq, D = query_states.shape
        _, Nk, _ = key_states.shape
        
        # Expand for pairwise broadcast: (B, Nq, Nk, D)
        q = query_states.unsqueeze(2).expand(B, Nq, Nk, D)
        k = key_states.unsqueeze(1).expand(B, Nq, Nk, D)
        
        # Extract raw components
        # Assuming state vector indices:
        # 0: Team ID
        # 3, 4: Pos X, Pos Y (Normalized)
        # 5, 6: Vel X, Vel Y (Normalized)
        # 7, 8: Attitude X, Attitude Y
        
        # Unpack
        q_team = q[..., 0]
        k_team = k[..., 0]
        
        q_pos = q[..., 3:5]
        k_pos = k[..., 3:5]
        
        q_vel = q[..., 5:7]
        k_vel = k[..., 5:7]
        
        q_att = q[..., 7:9]
        # k_att = k[..., 7:9] # Unused for now unless we want Aspect Angle relative to K? 
        # Aspect Angle usually means "Angle off bow" of the target.
        # But here we are simply computing pairwise relations.
        
        # --- 1. Delta (dx, dy, dvx, dvy) ---
        delta_pos = k_pos - q_pos
        delta_vel = k_vel - q_vel
        feat_delta_input = torch.cat([delta_pos, delta_vel], dim=-1)
        
        # --- 2. Distance ---
        dist_sq = delta_pos.pow(2).sum(dim=-1, keepdim=True)
        dist = torch.sqrt(dist_sq + 1e-6)
        inv_dist = 1.0 / (dist + 0.1) # Soft inverse
        feat_dist_input = torch.cat([dist, inv_dist], dim=-1)
        
        # --- 3. Angle (Aspect, ATA) ---
        # Antenna Train Angle (ATA): Angle to target relative to my heading
        # We need dot product of Heading (q_att) and Bearing vector (delta_pos normalized)
        
        # Normalize delta_pos for bearing
        bearing = delta_pos / (dist + 1e-6)
        
        # ATA = Dot(Bearing, Q_Attitude)
        ata = (bearing * q_att).sum(dim=-1, keepdim=True)
        
        # Aspect Angle: Angle of ME relative to TARGET's heading?
        # Typically Aspect is defined by Target's tail.
        # Let's use: Dot(Bearing, K_Attitude) ?? 
        # Or Just cross product?
        # The prompt says "Aspect Angle, ATA". 
        # Let's interpret Aspect as "Are they looking at me?" -> Dot(Bearing, K_Attitude)
        # Wait, Bearing is Q->K.
        # If K is looking at Q, K_Attitude should be opposed to Bearing.
        # Let's use Dot(delta_pos, k_vel) or something similar if we don't assume K has attitude? 
        # We DO have attitude at 7,8.
        k_att_vec = k[..., 7:9]
        aspect = (bearing * k_att_vec).sum(dim=-1, keepdim=True)
        
        feat_angle_input = torch.cat([ata, aspect], dim=-1)
        
        # --- 4. Intercept (TTCA, FlightTime) ---
        # TTCA (Time to Closest Approach) = -Dot(RelPos, RelVel) / Dot(RelVel, RelVel)
        # If diverging, clamp to 0 or max.
        
        # RelVel for collision: V_close = V_q - V_k? Or V_k - V_q? 
        # We want time until pos matches.
        # P(t) = P0 + V*t.
        # D(t) = |(Pk + Vk*t) - (Pq + Vq*t)| = |(Pk-Pq) + (Vk-Vq)*t| = |Dpos + Dvel*t|
        # Min at t = -Dot(Dpos, Dvel) / |Dvel|^2
        
        dot_dp_dv = (delta_pos * delta_vel).sum(dim=-1, keepdim=True)
        speed_sq = delta_vel.pow(2).sum(dim=-1, keepdim=True)
        ttca = -dot_dp_dv / (speed_sq + 1e-6)
        ttca = torch.relu(ttca) # Only future intercepts
        ttca = torch.tanh(ttca) # Squash to [0, 1] range roughly
        
        # Bullet Flight Time
        # Time for bullet (speed 500) to travel 'dist'.
        # Since pos is normalized to [0, 1] relative to window size (assume ~2000), 
        # we need to be careful with units.
        # World size is roughly 2000. Token values are Pos/2000.
        # Bullet Speed 500 -> 0.25 units/sec in normalized space?
        # Let's just use Dist. It's proportional.
        flight_time = dist # Just use raw normalized dist as proxy
        
        feat_intercept_input = torch.cat([ttca, flight_time], dim=-1)
        
        # --- 5. Team / Speed ---
        # Same Team: 1 if q_team == k_team, else 0
        same_team = (q_team == k_team).float().unsqueeze(-1)
        
        # Delta Speed
        q_speed = q_vel.norm(dim=-1, keepdim=True)
        k_speed = k_vel.norm(dim=-1, keepdim=True)
        delta_speed = k_speed - q_speed
        
        feat_team_input = torch.cat([same_team, delta_speed], dim=-1)
        
        # --- Projection ---
        out_delta = self.feat_delta(feat_delta_input)
        out_dist = self.feat_dist(feat_dist_input)
        out_angle = self.feat_angle(feat_angle_input)
        out_intercept = self.feat_intercept(feat_intercept_input)
        out_team = self.feat_team(feat_team_input)
        
        combined = torch.cat([out_dist, out_angle, out_delta, out_intercept, out_team], dim=-1)
        
        # MLP -> Bias
        # (B, Nq, Nk, hidden) -> (B, Nq, Nk, Heads)
        bias = self.mlp(combined)
        
        # Permute to (B, Heads, Nq, Nk)
        bias = bias.permute(0, 3, 1, 2)
        
        return bias

