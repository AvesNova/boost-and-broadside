"""Pairwise relational feature extractor for attention bias.

For each pair (i, j) of ships, computes 8 geometric features in ship i's
local frame:
  1-2. Bearing sin/cos:          direction from i to j in i's heading frame
  3-4. Aspect sin/cos:           j's nose facing toward i
  5.   Symlog distance:          compressed distance between ships
  6.   Symlog closing rate:      rate of approach (positive = getting closer)
  7-8. Relative heading sin/cos: θ_j − θ_i heading difference

Output: (..., H, N, N) attention bias added to pre-softmax logits.
"""

import torch
import torch.nn as nn


def _symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log(torch.abs(x) + 1.0)


class RelationalFeatures(nn.Module):
    """Projects pairwise ship geometry to per-head attention biases.

    Args:
        n_heads: Number of attention heads (output channels).
    """

    _FEATURE_DIM = 8

    def __init__(self, n_heads: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(self._FEATURE_DIM, 64),
            nn.SiLU(),
            nn.Linear(64, n_heads),
        )

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute pairwise attention biases from ship observations.

        Args:
            obs: Dict with at least "pos" (..., N, 2), "vel" (..., N, 2),
                 and "att" (..., N, 2) keys. Leading batch dims are arbitrary.

        Returns:
            (..., H, N, N) float — attention bias per head.
        """
        pos = obs["pos"]    # (..., N, 2)  normalized [0, 1]
        vel = obs["vel"]    # (..., N, 2)  world units
        att = obs["att"]    # (..., N, 2)  (cos θ, sin θ) unit vector

        # Pairwise displacement/velocity: (..., N_i, N_j, 2)
        # rel_pos[..., i, j, :] = pos_j - pos_i
        rel_pos = pos.unsqueeze(-3) - pos.unsqueeze(-2)    # (..., N, N, 2)
        rel_vel = vel.unsqueeze(-3) - vel.unsqueeze(-2)    # (..., N, N, 2)

        # Distance and unit direction i → j
        dist2  = (rel_pos * rel_pos).sum(-1, keepdim=True).clamp(min=1e-12)
        dist   = dist2.sqrt()                               # (..., N, N, 1)
        dir_ij = rel_pos / dist                             # (..., N, N, 2)

        # Expand headings for pairwise computation
        # att_i[..., i, j, :] = att[..., i, :]
        att_i = att.unsqueeze(-2).expand_as(rel_pos)       # (..., N, N, 2)
        # att_j[..., i, j, :] = att[..., j, :]
        att_j = att.unsqueeze(-3).expand_as(rel_pos)       # (..., N, N, 2)

        # Bearing from i to j in i's heading frame
        # Complex: conj(att_i) * dir_ij
        bear_cos = att_i[..., 0] * dir_ij[..., 0] + att_i[..., 1] * dir_ij[..., 1]
        bear_sin = att_i[..., 0] * dir_ij[..., 1] - att_i[..., 1] * dir_ij[..., 0]

        # Aspect angle: how much j's nose points toward i (dir_ji = -dir_ij)
        # Complex: conj(att_j) * dir_ji = conj(att_j) * (-dir_ij)
        asp_cos = -(att_j[..., 0] * dir_ij[..., 0] + att_j[..., 1] * dir_ij[..., 1])
        asp_sin = -(att_j[..., 0] * dir_ij[..., 1] - att_j[..., 1] * dir_ij[..., 0])

        # Symlog distance
        symlog_dist = _symlog(dist.squeeze(-1))             # (..., N, N)

        # Closing rate: -dot(rel_vel, dir_ij)  — positive means approaching
        closing      = -(rel_vel[..., 0] * dir_ij[..., 0] + rel_vel[..., 1] * dir_ij[..., 1])
        symlog_close = _symlog(closing)                     # (..., N, N)

        # Relative heading: θ_j − θ_i  — complex: conj(att_i) * att_j
        rel_head_cos = att_i[..., 0] * att_j[..., 0] + att_i[..., 1] * att_j[..., 1]
        rel_head_sin = att_i[..., 0] * att_j[..., 1] - att_i[..., 1] * att_j[..., 0]

        feats = torch.stack([
            bear_sin, bear_cos,
            asp_sin,  asp_cos,
            symlog_dist,
            symlog_close,
            rel_head_sin, rel_head_cos,
        ], dim=-1)                                          # (..., N, N, 8)

        bias = self.mlp(feats)                              # (..., N, N, H)
        return bias.movedim(-1, -3)                         # (..., H, N, N)
