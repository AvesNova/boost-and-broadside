"""Rotary encoding of world position and attitude for spatial attention.

Spatial attention already had one route to relative geometry: ships and bullets
expand position on a shared Fourier basis, so ``q·k`` contains
``cos(w(x_i - x_j))`` terms. That route is indirect — the encoder MLP has to
preserve those basis functions through two projections and an RMSNorm before the
``W_qkv`` bilinear form can use them, and every dimension it spends doing so is a
dimension not spent on anything else.

Rotary encoding puts the same relative geometry into the comparison directly.
Rotating the Q/K pair ``(t_{2i}, t_{2i+1})`` by ``w_i * x`` makes that pair's
contribution to the score exactly

    |q||k| cos(phi_q - phi_k + w_i (x_q - x_k))

so displacement enters the score as a rotation rather than as something the
trunk must reconstruct, while the token keeps whatever content it had.

**The basis is reused, not reinvented.** Every frequency here comes from
``base2_frequencies``, the same function the encoder's ``Fourier`` transform
calls:

* world x, period = world width, ``position_fourier_frequencies(width)`` harmonics;
* world y, period = world height, likewise;
* attitude, period = ``2*pi``, ``ATTITUDE_FOURIER_FREQUENCIES`` harmonics.

Because every frequency is an integer multiple of ``2*pi / period``, each is
exactly periodic over its own physical period. Wrapping the toroid, or turning
a ship through a full circle, therefore returns the rotation to where it
started — continuity across the seam is exact rather than approximate.

**The attitude axis degrades gracefully.** Tokens without a heading (fields,
zones, the boundary/global token, bullets) carry ``ATT = (0, 0)``, and
``atan2(0, 0)`` is 0 — which is precisely the angle their input Fourier feature
already encodes. Their rotation in the attitude block is the identity, so those
dimensions contribute ``cos(k * theta_query)``: an absolute heading preference
toward map objects rather than a relative-heading comparison. That is the
correct reading of "this thing has no heading", and it needs no second query
tensor and no special case in the attention kernel.

**Dimension budget.** Each frequency consumes one dimension *pair*. The
Frontline world (16384 px, 8 position harmonics per axis) wants
``2*(8 + 8 + 4) = 40`` dimensions, which does not fit a 32-wide head at all and
leaves 24 unrotated dimensions in a 64-wide one. A configuration that would need
more than the head provides raises: silently dropping the coarsest frequency
would break toroidal continuity, and dropping the finest would blur exactly the
short-range geometry this exists to sharpen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

from boost_and_broadside.config import ModelConfig, ShipConfig
from boost_and_broadside.train.rl.checkpoint_schema import (
    ATTITUDE_FOURIER_FREQUENCIES,
    base2_frequencies,
    position_fourier_frequencies,
)


@dataclass(frozen=True)
class RotaryAxis:
    """One physical coordinate the rotation is taken over."""

    name: str
    period: float  # in the coordinate's own units (px for x/y, radians for attitude)
    n_freqs: int

    @property
    def frequencies(self) -> tuple[float, ...]:
        return base2_frequencies(self.period, self.n_freqs)


def spatial_rotary_axes(ship_config: ShipConfig) -> tuple[RotaryAxis, ...]:
    """The three axes spatial Q/K is rotated over, in Q/K dimension order.

    Order is load-bearing: it fixes which head dimensions carry which
    coordinate, and therefore what a trained ``W_qkv`` means. It is recorded in
    the checkpoint's ``ship_config`` by way of ``world_size``.
    """

    world_w, world_h = ship_config.world_size
    return (
        RotaryAxis("position_x", float(world_w), position_fourier_frequencies(float(world_w))),
        RotaryAxis("position_y", float(world_h), position_fourier_frequencies(float(world_h))),
        RotaryAxis("attitude", 2.0 * math.pi, ATTITUDE_FOURIER_FREQUENCIES),
    )


def rotary_pair_count(ship_config: ShipConfig) -> int:
    """Head-dimension *pairs* the rotation consumes; twice this is the width."""

    return sum(axis.n_freqs for axis in spatial_rotary_axes(ship_config))


class RotaryBudgetError(ValueError):
    """A map configuration needs more rotary dimensions than a head provides."""


def check_rotary_budget(model_config: ModelConfig, ship_config: ShipConfig) -> None:
    """Fail loudly when the frequency basis does not fit the spatial head.

    Raised at construction rather than worked around. Every alternative that
    keeps running is worse than a stopped run: truncating the low frequencies
    costs exact toroidal periodicity, truncating the high ones costs the
    short-range resolution the rotation exists for, and substituting a different
    spacing means the encoder's Fourier features and the Q/K rotations no longer
    describe the same geometry.
    """

    pairs = rotary_pair_count(ship_config)
    head_dim = model_config.spatial_head_dim
    if 2 * pairs > head_dim:
        axes = ", ".join(f"{a.name}={a.n_freqs}" for a in spatial_rotary_axes(ship_config))
        raise RotaryBudgetError(
            f"spatial RoPE needs {2 * pairs} of {head_dim} head dimensions for world "
            f"{tuple(ship_config.world_size)} ({axes}). Widen the spatial head "
            f"(d_model={model_config.d_model} / n_spatial_heads="
            f"{model_config.spatial_heads}), or use a world whose position basis is "
            "coarser. Truncating frequencies would silently change what the "
            "rotation measures."
        )


def apply_rotary(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotate the leading ``2R`` dimensions of ``t`` pairwise.

    Args:
        t:   (..., N, H, head_dim) queries or keys, heads last-but-one.
        cos: (..., N, 1, R) cosines of the per-token rotation angles.
        sin: (..., N, 1, R) sines of the same.

    Returns:
        (..., N, H, head_dim) with dimensions ``[0, 2R)`` rotated as interleaved
        ``(even, odd)`` pairs and the remainder passed through unchanged. Every
        head shares one rotation — the angles describe the token, not the head.
    """

    width = 2 * cos.shape[-1]
    rotated, passthrough = t[..., :width], t[..., width:]
    even = rotated[..., 0::2]
    odd = rotated[..., 1::2]
    pairs = torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1)
    return torch.cat((pairs.flatten(-2), passthrough), dim=-1)


class SpatialRotary(nn.Module):
    """Builds the per-token rotation tables the spatial layers share.

    Held by the policy and evaluated once per forward: every spatial sublayer
    rotates against the same tables, so the trigonometry is paid for once rather
    than once per layer. Frequencies live in a non-persistent buffer — they are
    derived from ``ship_config``, which the checkpoint already records, and
    writing them into the state dict would make a world-size change look like a
    weight mismatch instead of the contract error it is.
    """

    def __init__(self, ship_config: ShipConfig, head_dim: int) -> None:
        super().__init__()
        self.axes = spatial_rotary_axes(ship_config)
        self.head_dim = head_dim
        frequencies: list[float] = []
        for axis in self.axes:
            frequencies.extend(axis.frequencies)
        self.pairs = len(frequencies)
        self.rotary_dim = 2 * self.pairs
        if self.rotary_dim > head_dim:
            raise RotaryBudgetError(
                f"spatial RoPE needs {self.rotary_dim} of {head_dim} head dimensions"
            )
        # Split points into the concatenated frequency vector, so a table build
        # can multiply each coordinate by its own block without a gather.
        self._n_x = self.axes[0].n_freqs
        self._n_y = self.axes[1].n_freqs
        self._n_att = self.axes[2].n_freqs
        self.register_buffer(
            "frequencies", torch.tensor(frequencies, dtype=torch.float32), persistent=False
        )

    def extra_repr(self) -> str:
        axes = ", ".join(f"{a.name}:{a.n_freqs}" for a in self.axes)
        return f"pairs={self.pairs}, rotary_dim={self.rotary_dim}/{self.head_dim}, axes=({axes})"

    def tables(
        self, position: torch.Tensor, attitude: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Cosine/sine tables for one set of tokens.

        Args:
            position: (..., N, 2) world x/y.
            attitude: (..., N, 2) Cartesian heading, or None for tokens that have
                none. A missing or zero heading gives angle 0, which makes the
                attitude block's rotation the identity — the same thing the
                encoder's attitude Fourier feature already does with ``(0, 0)``.

        Returns:
            ``(cos, sin)``, each (..., N, 1, R), broadcast-ready against a
            head-major ``(..., N, H, head_dim)`` query or key.
        """

        freqs = self.frequencies.to(position.dtype)
        angle_x = position[..., 0:1] * freqs[: self._n_x]
        angle_y = position[..., 1:2] * freqs[self._n_x : self._n_x + self._n_y]
        att_freqs = freqs[self._n_x + self._n_y :]
        if attitude is None:
            heading = position.new_zeros((*position.shape[:-1], 1))
        else:
            heading = torch.atan2(attitude[..., 1:2], attitude[..., 0:1])
        angle_att = heading * att_freqs
        angles = torch.cat((angle_x, angle_y, angle_att), dim=-1).unsqueeze(-2)
        return angles.cos(), angles.sin()
