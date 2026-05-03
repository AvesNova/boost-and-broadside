"""Composable observation feature pipeline.

Each observation feature is a (source, transforms...) chain. Sources name
a key in the raw obs dict; transforms are applied in order. The encoder
iterates all features, concatenates outputs, and projects to d_model.

Usage example (runs/shared.py):

    OBS_CONFIG = ObsConfig(features=(
        FeatureSpec("pos_x", ("pos", 0), (Fourier(10, period=1024.0),)),
        FeatureSpec("vel",   "vel",       (SymlogVec(),)),
        FeatureSpec("team",  "team_id",   (OneHot(3),)),
    ))
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Union

import torch


_EPS = 1e-6


# ---------------------------------------------------------------------------
# Shared math helpers (also used by encoder)
# ---------------------------------------------------------------------------

def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def fourier_encode(coords: torch.Tensor, n_freqs: int, period: float) -> torch.Tensor:
    """Fourier-encode scalar coordinates. Base-2 power frequencies.

    Args:
        coords: (...) float tensor.
        n_freqs: Number of frequency bands.
        period: Fundamental period (e.g. world_w for x position).

    Returns:
        (..., 2*n_freqs) — interleaved [sin_0, cos_0, sin_1, cos_1, ...].
    """
    k = torch.arange(n_freqs, device=coords.device, dtype=torch.float32)
    freqs = (2.0 * math.pi / period) * (2.0 ** k)
    args = coords.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


def fourier_encode_angle(vec: torch.Tensor, n_freqs: int) -> torch.Tensor:
    """Fourier-encode the angle of a 2D vector with base-2 integer frequencies.

    Args:
        vec: (..., 2) float tensor — any 2D vector (need not be unit).
        n_freqs: Number of frequency bands.

    Returns:
        (..., 2*n_freqs) tensor.
    """
    angle = torch.atan2(vec[..., 1], vec[..., 0])
    k = torch.arange(n_freqs, device=vec.device, dtype=torch.float32)
    freqs = 2.0 ** k
    args = angle.unsqueeze(-1) * freqs
    return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


# ---------------------------------------------------------------------------
# Transform blocks
# ---------------------------------------------------------------------------
# Each block is a frozen dataclass with:
#   out_dim: int          — output feature dimension (static, known at config time)
#   __call__(x) -> Tensor — applies the transform


@dataclass(frozen=True)
class Normalize:
    """Divide by a constant scale. S→S."""
    scale: float

    @property
    def out_dim(self) -> int:
        return 1

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.scale


@dataclass(frozen=True)
class Symlog:
    """Symmetric log: sign(x)*log(1+|x|). S→S."""

    @property
    def out_dim(self) -> int:
        return 1

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return symlog(x)


@dataclass(frozen=True)
class Clamp:
    """Clamp to [lo, hi]. S→S."""
    lo: float
    hi: float

    @property
    def out_dim(self) -> int:
        return 1

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.clamp(self.lo, self.hi)


@dataclass(frozen=True)
class AsFloat:
    """Cast bool or int to float. S→S."""

    @property
    def out_dim(self) -> int:
        return 1

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return x.float()


@dataclass(frozen=True)
class Bucketize:
    """Map a float in [0, 1] to an integer bucket in [0, n-1].

    Rounds to nearest bucket. Input must already be in [0, 1].
    Typically followed by OneHot(n).
    """
    n: int

    @property
    def out_dim(self) -> int:
        return 1  # scalar int (long dtype)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return (x * (self.n - 1)).round().long().clamp(0, self.n - 1)


@dataclass(frozen=True)
class Fourier:
    """Fourier-encode a scalar with base-2 power frequencies. S→V (2*n dims).

    The period must match the signal's natural period so that endpoints
    encode identically (e.g. period=world_w for position x in pixels).
    """
    n: int
    period: float

    @property
    def out_dim(self) -> int:
        return 2 * self.n

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return fourier_encode(x, self.n, self.period)


@dataclass(frozen=True)
class OneHot:
    """Integer → one-hot vector. S→V (n dims)."""
    n: int

    @property
    def out_dim(self) -> int:
        return self.n

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.long()
        if x.shape[-1] == 1:
            x = x.squeeze(-1)
        return torch.nn.functional.one_hot(x, self.n).float()


@dataclass(frozen=True)
class VecMag:
    """Euclidean norm of a 2D vector. V→S (1 dim)."""

    @property
    def out_dim(self) -> int:
        return 1

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return torch.norm(x, dim=-1, keepdim=True)


@dataclass(frozen=True)
class SymlogVec:
    """Direction-preserving symlog on a 2D vector. V→V (2 dims).

    Scales the vector so its magnitude becomes symlog(|v|) while
    preserving the direction ratio vx/vy. When v=0, output is 0.
    """

    @property
    def out_dim(self) -> int:
        return 2

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        mag = torch.norm(x, dim=-1, keepdim=True)
        return x * symlog(mag) / mag.clamp(min=_EPS)


@dataclass(frozen=True)
class FourierAngle:
    """Fourier-encode the angle of a 2D vector with base-2 integer freqs. V→V' (2*n dims).

    Accepts any 2D vector (not required to be unit); angle is extracted via atan2.
    Works for raw velocity, unit attitude vectors, etc.
    """
    n: int

    @property
    def out_dim(self) -> int:
        return 2 * self.n

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return fourier_encode_angle(x, self.n)


# Union type for type annotations
Transform = Union[
    Normalize, Symlog, Clamp, AsFloat, Bucketize,
    Fourier, OneHot, VecMag, SymlogVec, FourierAngle,
]


# ---------------------------------------------------------------------------
# FeatureSpec and ObsConfig
# ---------------------------------------------------------------------------

def _chain_out_dim(transforms: tuple) -> int:
    """Compute output dimension of a transform chain."""
    if not transforms:
        raise ValueError("Transform chain must have at least one block")
    return transforms[-1].out_dim


@dataclass(frozen=True)
class FeatureSpec:
    """A named observation feature: a source plus an ordered transform chain.

    Args:
        name: Human-readable label (used in logging/debugging).
        source: Either a string obs-dict key (returns the whole tensor), or a
                (key, int) tuple that slices one scalar channel from the last dim.
        transforms: Ordered tuple of transform blocks; output of each feeds the next.
    """
    name: str
    source: Union[str, tuple]
    transforms: tuple

    @property
    def out_dim(self) -> int:
        return _chain_out_dim(self.transforms)


@dataclass(frozen=True)
class ObsConfig:
    """Full observation feature pipeline specification.

    A tuple of FeatureSpec entries. The encoder iterates them in order,
    applies each chain, and concatenates results into the raw feature vector
    fed to the linear projection.
    """
    features: tuple

    def raw_dim(self) -> int:
        """Total concatenated feature dimension fed into the encoder's linear layer."""
        return sum(f.out_dim for f in self.features)

    def to_dict(self) -> dict:
        """Serialize to a plain dict suitable for torch.save / JSON logging.

        Each transform block is stored with a ``_type`` key so it can be
        reconstructed exactly via :func:`obs_config_from_dict`.
        """
        def _ser_transform(t) -> dict:
            import dataclasses
            d = dataclasses.asdict(t)
            d["_type"] = type(t).__name__
            return d

        return {
            "features": [
                {
                    "name": f.name,
                    "source": list(f.source) if isinstance(f.source, tuple) else f.source,
                    "transforms": [_ser_transform(t) for t in f.transforms],
                }
                for f in self.features
            ]
        }


_TRANSFORM_REGISTRY: dict[str, type] = {
    "Normalize": Normalize,
    "Symlog": Symlog,
    "Clamp": Clamp,
    "AsFloat": AsFloat,
    "Bucketize": Bucketize,
    "Fourier": Fourier,
    "OneHot": OneHot,
    "VecMag": VecMag,
    "SymlogVec": SymlogVec,
    "FourierAngle": FourierAngle,
}


def obs_config_from_dict(d: dict) -> "ObsConfig":
    """Reconstruct an ObsConfig from the dict produced by :meth:`ObsConfig.to_dict`."""
    features = []
    for f in d["features"]:
        transforms = tuple(
            _TRANSFORM_REGISTRY[t_dict.pop("_type")](**t_dict)
            for t_dict in [dict(t) for t in f["transforms"]]
        )
        src = f["source"]
        source = tuple(src) if isinstance(src, list) else src
        features.append(FeatureSpec(f["name"], source, transforms))
    return ObsConfig(features=tuple(features))
