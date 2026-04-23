"""Sketch Isotropic Gaussian Regularizer (SIGReg).

Measures how far the empirical characteristic function of randomly-projected
embeddings deviates from that of a standard isotropic Gaussian (Epps–Pulley
statistic).  Use to regularize a latent space toward N(0,I) during training.
"""

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer.

    Adapted from the LeWorldModel paper.  Single-GPU implementation;
    random projections are re-drawn in-place each forward call using a
    pre-allocated buffer to avoid allocation overhead in the hot path.

    Args:
        d_model:  Embedding dimension — must match the encoder output.
        knots:    Number of trapezoidal integration points.
        num_proj: Number of random projection directions.
    """

    def __init__(self, d_model: int, knots: int = 17, num_proj: int = 1024) -> None:
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)
        # Pre-allocated projection matrix — refilled with randn_ each forward call.
        self.register_buffer("_A", torch.empty(d_model, num_proj), persistent=False)

    def forward(self, proj: torch.Tensor) -> torch.Tensor:
        """Compute the mean Epps–Pulley statistic.

        Args:
            proj: (T, B, D) float — encoder outputs; T timesteps, B samples each.

        Returns:
            Scalar loss — mean statistic over T and projections.
        """
        self._A.normal_()
        self._A.div_(self._A.norm(p=2, dim=0))
        A = self._A.to(proj.dtype)

        t = self.t.to(proj.dtype)
        phi = self.phi.to(proj.dtype)
        weights = self.weights.to(proj.dtype)

        x_t = (proj @ A).unsqueeze(-1) * t          # (T, B, num_proj, knots)
        err = (
            (x_t.cos().mean(1) - phi).square()       # mean over B samples
            + x_t.sin().mean(1).square()
        )                                             # (T, num_proj, knots)
        statistic = (err @ weights) * proj.size(1)   # (T, num_proj), scaled by B
        return statistic.mean()
