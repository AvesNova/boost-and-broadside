"""
soft_bins.py — Soft Binning Utilities for Yemong Targets.

Provides:
  - symlog / symlog_inv
  - soft_bin_uniform   (2-bin triangular interpolation on a uniform grid)
  - soft_bin_angular   (2-bin triangular interpolation with seam wrap-around)
  - SoftBinSpec        (dataclass describing one target field)
  - INTERLEAVED_SOFT_BIN_SPECS  (canonical spec list for YemongDynamicsInterleaved)
  - compute_soft_bin_targets    (convert raw obs to per-spec soft distributions)
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import torch
import torch.nn.functional as F

from boost_and_broadside.core.constants import StateFeature


# ---------------------------------------------------------------------------
# Symlog
# ---------------------------------------------------------------------------

def symlog(x: torch.Tensor, linthresh: float = 1.0) -> torch.Tensor:
    """
    Signed log compression.  Identity near zero, log growth beyond ±linthresh.
    symlog(x) = sign(x) * log(1 + |x| / linthresh)
    """
    return x.sign() * torch.log1p(x.abs() / linthresh)


def symlog_inv(y: torch.Tensor, linthresh: float = 1.0) -> torch.Tensor:
    """Inverse of symlog."""
    return y.sign() * (torch.expm1(y.abs()) * linthresh)


# ---------------------------------------------------------------------------
# 2-bin Soft Binning (triangular / linear-interpolation kernel)
# ---------------------------------------------------------------------------

def soft_bin_uniform(
    x: torch.Tensor,
    n_bins: int,
    lo: float,
    hi: float,
) -> torch.Tensor:
    """
    2-bin soft binning on a uniform grid [lo, hi].

    The grid has `n_bins` bins of equal width w = (hi - lo) / n_bins.
    Bin centres: c_k = lo + (k + 0.5) * w  for k = 0 … n_bins-1.

    Value x is distributed between exactly two adjacent bins via linear
    interpolation (triangle kernel):
        left_bin  = floor((x - lo) / w)          (clamped to [0, n_bins-1])
        right_bin = left_bin + 1                  (clamped to n_bins-1)
        right_w   = (x - lo) / w - left_bin       (in [0, 1])
        left_w    = 1 - right_w

    x is clamped to [lo, hi] before indexing so out-of-range values
    saturate at the boundary bins.

    Args:
        x: (...,) tensor of values.
        n_bins: number of bins.
        lo, hi: range.

    Returns:
        (..., n_bins) float tensor, sums to 1 along last dim.
    """
    dtype = x.dtype
    device = x.device

    w = (hi - lo) / n_bins

    # Clamp to [lo, hi]
    xc = x.clamp(lo, hi)

    # Continuous bin index (fractional), in [0, n_bins]
    frac = (xc - lo) / w  # [...], ∈ [0, n_bins]

    left = frac.floor().long().clamp(0, n_bins - 1)   # [...], ∈ [0, n_bins-1]
    right = (left + 1).clamp(0, n_bins - 1)           # [...], ∈ [0, n_bins-1]

    right_w = (frac - left.float()).clamp(0.0, 1.0)   # weight for right bin
    left_w = 1.0 - right_w                             # weight for left bin

    # Scatter into output
    out = torch.zeros(*x.shape, n_bins, dtype=dtype, device=device)
    out.scatter_add_(-1, left.unsqueeze(-1), left_w.unsqueeze(-1))
    out.scatter_add_(-1, right.unsqueeze(-1), right_w.unsqueeze(-1))

    return out


def soft_bin_angular(x: torch.Tensor, n_bins: int) -> torch.Tensor:
    """
    2-bin soft binning for circular/angular values in [0, 2π).

    Bins are uniformly spaced with width w = 2π / n_bins.
    Bin centres: c_k = (k + 0.5) * w.

    The interpolation is identical to soft_bin_uniform, but wrapped
    modulo n_bins so that the seam between bin n_bins-1 and bin 0
    is handled correctly.

    x is assumed to already be in [0, 2π); no range clamping is applied.
    Values outside this range are wrapped via mod before indexing.

    Args:
        x: (...,) tensor of angles in [0, 2π).
        n_bins: number of bins.

    Returns:
        (..., n_bins) float tensor, sums to 1 along last dim.
    """
    TWO_PI = 2.0 * math.pi
    dtype = x.dtype
    device = x.device

    w = TWO_PI / n_bins

    # Wrap x to [0, 2π)
    xw = x % TWO_PI

    # Continuous bin index, in [0, n_bins)
    frac = xw / w  # [...], ∈ [0, n_bins)

    left = frac.floor().long() % n_bins     # [...], ∈ [0, n_bins-1]
    right = (left + 1) % n_bins             # wraps at seam

    right_w = (frac - frac.floor()).clamp(0.0, 1.0)
    left_w = 1.0 - right_w

    out = torch.zeros(*x.shape, n_bins, dtype=dtype, device=device)
    out.scatter_add_(-1, left.unsqueeze(-1), left_w.unsqueeze(-1))
    out.scatter_add_(-1, right.unsqueeze(-1), right_w.unsqueeze(-1))

    return out


# ---------------------------------------------------------------------------
# Spec Definition
# ---------------------------------------------------------------------------

@dataclass
class SoftBinSpec:
    """Describes one soft-binned target field."""
    name: str
    n_bins: int
    lo: float       # lower bound of the bin range (after symlog if apply_symlog=True)
    hi: float       # upper bound of the bin range
    is_angular: bool = False
    apply_symlog: bool = False
    linthresh: float = 1.0
    is_team_level: bool = False   # True for value/reward (no ship dim)


# Canonical specs for YemongDynamicsInterleaved.
# Order defines the output head order.  Total bins = 768.
INTERLEAVED_SOFT_BIN_SPECS: List[SoftBinSpec] = [
    SoftBinSpec("health",    64,  0.0,             100.0,          is_angular=False, apply_symlog=False, is_team_level=False),
    SoftBinSpec("power",     64,  0.0,             100.0,          is_angular=False, apply_symlog=False, is_team_level=False),
    SoftBinSpec("pos_angle", 128, 0.0,   2*math.pi,               is_angular=True,  apply_symlog=False, is_team_level=False),
    SoftBinSpec("pos_mag",   128, 0.0,             1.0,            is_angular=False, apply_symlog=True,  linthresh=1.0, is_team_level=False),
    SoftBinSpec("vel_angle", 128, 0.0,   2*math.pi,               is_angular=True,  apply_symlog=False, is_team_level=False),
    SoftBinSpec("vel_mag",   128, 0.0,             2.0,            is_angular=False, apply_symlog=True,  linthresh=1.0, is_team_level=False),
    SoftBinSpec("value",      64, 0.0,             2.0,            is_angular=False, apply_symlog=True,  linthresh=1.0, is_team_level=True),
    SoftBinSpec("reward",     64, 0.0,             2.0,            is_angular=False, apply_symlog=True,  linthresh=1.0, is_team_level=True),
]

TOTAL_SOFT_BIN_LOGITS = sum(s.n_bins for s in INTERLEAVED_SOFT_BIN_SPECS)  # 768


# ---------------------------------------------------------------------------
# Target Computation
# ---------------------------------------------------------------------------

def compute_soft_bin_targets(
    state_t: torch.Tensor,        # (B, T, N, state_dim) — current states
    state_tp1: torch.Tensor,      # (B, T, N, state_dim) — next states
    pos_t: torch.Tensor,          # (B, T, N, 2)
    pos_tp1: torch.Tensor,        # (B, T, N, 2)
    vel_t: torch.Tensor,          # (B, T, N, 2)   (vx, vy)
    vel_tp1: torch.Tensor,        # (B, T, N, 2)
    W: float,                     # world width  (for toroidal wrap)
    H: float,                     # world height (for toroidal wrap)
    value: Optional[torch.Tensor] = None,   # (B, T, 1) or (B, T, N)
    reward: Optional[torch.Tensor] = None,  # (B, T, 1) or (B, T, N)
    specs: Optional[List[SoftBinSpec]] = None,
    label_smoothing: float = 0.01,  # mix in this fraction of uniform to prevent zero-probability bins
) -> List[torch.Tensor]:
    """
    Compute soft-binned target distributions for each spec.

    Returns:
        A List[Tensor] in spec order.  Per-ship specs → (B, T, N, n_bins).
        Team-level specs (value, reward) → (B, T, 1, n_bins).
    """
    if specs is None:
        specs = INTERLEAVED_SOFT_BIN_SPECS

    device = state_t.device
    dtype = torch.float32  # always compute in fp32 for numerical stability

    state_tp1 = state_tp1.float()
    pos_t = pos_t.float()
    pos_tp1 = pos_tp1.float()
    vel_t = vel_t.float()
    vel_tp1 = vel_tp1.float()

    # --- Health / Power (absolute next step) ---
    health = state_tp1[..., StateFeature.HEALTH]  # (B, T, N)
    power  = state_tp1[..., StateFeature.POWER]   # (B, T, N)

    # --- Position delta (toroidal) ---
    d_pos = pos_tp1 - pos_t                                     # (B, T, N, 2)
    d_pos[..., 0] -= torch.round(d_pos[..., 0] / W) * W
    d_pos[..., 1] -= torch.round(d_pos[..., 1] / H) * H

    dx, dy = d_pos[..., 0], d_pos[..., 1]

    # Global direction angle: atan2(dy, dx) → [0, 2π)
    pos_angle = torch.atan2(dy, dx) % (2 * math.pi)            # (B, T, N)

    # Magnitude → symlog → clamp to [0, 1]
    pos_mag_raw = torch.sqrt(dx**2 + dy**2)
    pos_mag = symlog(pos_mag_raw, linthresh=1.0).clamp(0.0, 1.0)

    # --- Velocity delta ---
    d_vel = vel_tp1 - vel_t                                     # (B, T, N, 2)
    dvx, dvy = d_vel[..., 0], d_vel[..., 1]

    vel_angle = torch.atan2(dvy, dvx) % (2 * math.pi)          # (B, T, N)

    vel_mag_raw = torch.sqrt(dvx**2 + dvy**2)
    vel_mag = symlog(vel_mag_raw, linthresh=1.0).clamp(0.0, 2.0)

    # --- Value / Reward ---
    def _prepare_scalar(t: Optional[torch.Tensor]) -> torch.Tensor:
        """Return (B, T, 1) float tensor, or zeros if None."""
        if t is None:
            B, T = state_t.shape[:2]
            return torch.zeros(B, T, 1, device=device, dtype=dtype)
        t = t.float()
        if t.ndim == 2:
            t = t.unsqueeze(-1)          # (B, T) → (B, T, 1)
        elif t.ndim == 3 and t.shape[-1] != 1:
            # (B, T, N) → aggregate over N (mean) → (B, T, 1)
            t = t.mean(dim=-1, keepdim=True)
        # t is now (B, T, 1)
        return t.mean(dim=-2, keepdim=False) if t.ndim == 4 else t  # defensive

    val_t  = _prepare_scalar(value)    # (B, T, 1)
    rew_t  = _prepare_scalar(reward)   # (B, T, 1)

    val_sym  = symlog(val_t,  linthresh=1.0).clamp(0.0, 2.0)
    rew_sym  = symlog(rew_t,  linthresh=1.0).clamp(0.0, 2.0)

    # --- Assemble per-spec ---
    _raw = {
        "health":    health,
        "power":     power,
        "pos_angle": pos_angle,
        "pos_mag":   pos_mag,
        "vel_angle": vel_angle,
        "vel_mag":   vel_mag,
        "value":     val_sym,
        "reward":    rew_sym,
    }

    results: List[torch.Tensor] = []
    for spec in specs:
        raw = _raw[spec.name]  # (...,)

        if spec.is_angular:
            soft = soft_bin_angular(raw, spec.n_bins)
        else:
            soft = soft_bin_uniform(raw, spec.n_bins, spec.lo, spec.hi)

        # Label smoothing: mix with uniform distribution to prevent zero-probability bins.
        # smoothed = (1 - ε) * soft + ε / n_bins
        if label_smoothing > 0.0:
            soft = (1.0 - label_smoothing) * soft + label_smoothing / spec.n_bins

        # Ensure per-ship tensors are (B,T,N,bins) and team-level (B,T,1,bins)
        if spec.is_team_level and soft.ndim == 3:
            soft = soft.unsqueeze(2)  # (B, T, 1, bins)

        results.append(soft)

    return results
