"""The pairwise geometry head (`YemongConfig.geometry_head`): a training-only auxiliary head
that reads pairs of the trunk's ship rows and predicts the pair's geometry, so the trunk
learns to carry relative position, attitude and motion in its rows.

A pair is an anchor (an own ship) and a partner (another ship the view sees: an own ship,
or a visible enemy), both alive; hidden ships are never in a pair, so the head never asks
for information the view lacks. Its five targets, from the anchor A to the partner B
(positions p, velocities v, nose headings h; the line of sight lambda = the angle of
p_B - p_A):
- range |p_B - p_A| (m), log-spaced (`RANGE`: a symlog code above zero, knee 10 m, to
  the largest map);
- antenna train angle, wrap(lambda - h_A): B's bearing off A's nose;
- aspect, wrap(lambda - h_B): the line of sight off B's tail (0 with A dead astern of B,
  pi head on);
- heading crossing angle, wrap(h_B - h_A);
- closure rate -(v_B - v_A) . (p_B - p_A) / |p_B - p_A| (m/s, positive closing), a signed
  symlog code finer near zero (`CLOSURE`).
The angles are circular codes of ANGLE_BINS bins with wrapped-Gaussian smoothing (no
padding); range and closure are HL-Gauss codes with 2 padding bins per side (decisions,
"Code padding rule"). Every target is categorical, its loss in nats.

Each anchor has PARTNERS partners per tick (`pairs`): the rows after it in a cycle that
advances PARTNERS rows a tick, so a fleet larger than PARTNERS + 1 rows is covered over
consecutive ticks at a fixed cost per tick. With PARTNERS + 1 rows or fewer, every other
row is a partner every tick.
"""

import dataclasses
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from bnb.contracts.obs import SHIP

from . import codes
from .heads import FIRST_GAIN, LAST_GAIN, _orthogonal
from .layers import norm

PARTNERS = 8  # per anchor and tick
ANGLE_BINS = 48  # 7.5 degrees each
TARGETS = ("range", "antenna_train_angle", "aspect", "heading_crossing_angle", "closure")
ANGLES = TARGETS[1:4]


@dataclasses.dataclass(frozen=True)
class SymlogTarget:
    """A symlog HL-Gauss target (`codes.symlog_encode`): knee and bound (the last real
    bin's centre) in the quantity's units, real bins, and the real bins under zero (None:
    as many as above)."""

    knee: float
    bound: float
    bins: int
    below: int | None = None

    @property
    def width(self) -> int:
        """The target's code width: its bins and the padding bins on both sides."""
        return self.bins + 2 * codes.PADDING_BINS


# Range: 10 m to 64 km (the largest maps, about 62 km across), 48 bins (about 20% apart).
RANGE = SymlogTarget(knee=10.0, bound=64_000.0, bins=48, below=0)
# Closure: 0 to 2000 m/s either way (two of the fastest x2 ships head on, ~900 m/s, and up
# to twice that in thin air; 640 m/s before the x2 game), 20 bins a side.
CLOSURE = SymlogTarget(knee=10.0, bound=2000.0, bins=41)

RUNS: codes.Runs = ((1, RANGE.width), (len(ANGLES), ANGLE_BINS), (1, CLOSURE.width))
WIDTH = sum(groups * size for groups, size in RUNS)


def quantities(
    position: torch.Tensor,
    velocity: torch.Tensor,
    heading: torch.Tensor,
    partners: torch.Tensor,
) -> torch.Tensor:
    """(..., S, K, 5) TARGETS of each anchor (the first S rows) and its partners: position
    and velocity (..., N, 2) in m and m/s, heading (..., N) rad, partners (..., S, K) row
    indices."""
    anchors = partners.shape[-2]

    def gathered(values: torch.Tensor) -> torch.Tensor:
        flat = partners.flatten(-2)  # (..., S K)
        if values.dim() == partners.dim() - 1:  # (..., N)
            return values.gather(-1, flat).unflatten(-1, partners.shape[-2:])
        index = flat[..., None].expand(*flat.shape, values.shape[-1])
        return values.gather(-2, index).unflatten(-2, partners.shape[-2:])

    offset = gathered(position) - position[..., :anchors, None, :]
    distance = torch.linalg.vector_norm(offset, dim=-1)
    sight = torch.atan2(offset[..., 1], offset[..., 0])
    own_heading = heading[..., :anchors, None]
    partner_heading = gathered(heading)
    relative = gathered(velocity) - velocity[..., :anchors, None, :]
    safe = torch.where(distance > 0, distance, 1.0)
    closure = -(relative * offset).sum(-1) / safe

    def wrapped(angle: torch.Tensor) -> torch.Tensor:
        return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi

    return torch.stack(
        [
            distance,
            wrapped(sight - own_heading),
            wrapped(sight - partner_heading),
            wrapped(partner_heading - own_heading),
            closure,
        ],
        -1,
    )


def encode(values: torch.Tensor) -> torch.Tensor:
    """(..., 5) TARGETS to their codes (..., WIDTH) in RUNS' layout."""
    parts = [codes.symlog_encode(values[..., 0], RANGE.knee, RANGE.bound, RANGE.bins, below=0)[0]]
    parts += [codes.wrapped_hl_gauss(values[..., 1 + index], ANGLE_BINS) for index in range(3)]
    parts.append(codes.symlog_encode(values[..., 4], CLOSURE.knee, CLOSURE.bound, CLOSURE.bins)[0])
    return torch.cat(parts, -1)


def partner_rows(rows: int, anchors: int, ticks: int, device=None) -> torch.Tensor:
    """(ticks, anchors, K) int64: anchor a's partners at tick t, the rows
    (a + 1 + (t K + k) mod (rows - 1)) mod rows for k < K = min(PARTNERS, rows - 1): never
    the anchor itself, and distinct within a tick."""
    count = min(PARTNERS, rows - 1)
    if count < 1:  # one row: no pairs
        return torch.zeros((ticks, anchors, 0), dtype=torch.int64, device=device)
    tick = torch.arange(ticks, device=device)[:, None, None]
    anchor = torch.arange(anchors, device=device)[None, :, None]
    step = torch.arange(count, device=device)[None, None, :]
    return (anchor + 1 + torch.remainder(tick * count + step, rows - 1)) % rows


def pairs(ships: torch.Tensor, alive: torch.Tensor, anchors: int) -> tuple[torch.Tensor, ...]:
    """A segment's pairs: ships (L, R, N, columns) the observation, alive (L, R, N) its
    truth, anchors S (the own rows). Returns (partners (L, R, S, K) int64, mask (L, R, S,
    K) bool: anchor and partner alive and the partner visible (an own row always is),
    targets (L, R, S, K, 5) TARGETS from the observation's columns, which are the truth
    where visible)."""
    length, views, rows = ships.shape[0], ships.shape[1], ships.shape[-2]
    partners = partner_rows(rows, anchors, length, ships.device)[:, None].expand(
        length, views, -1, -1
    )
    column = {name: ships[..., SHIP[name]] for name in SHIP}
    seen = alive & (column["visible"] > 0.5)
    mask = alive[..., :anchors, None] & seen.gather(-1, partners.flatten(-2)).unflatten(
        -1, partners.shape[-2:]
    )
    position = torch.stack([column["x"], column["y"]], -1)
    velocity = torch.stack([column["velocity_x"], column["velocity_y"]], -1)
    heading = torch.atan2(column["heading_sin"], column["heading_cos"])
    return partners, mask, quantities(position, velocity, heading, partners)


class GeometryHead(nn.Module):
    """Pair logits (..., WIDTH) from two ship rows: latent(x) is computed once per row (its
    first half the anchor's part, its second the partner's), and a pair's hidden is
    gelu(norm(anchor part + partner part)), then middle, norm, gelu, last; initialised as
    the shot trajectory head (`heads.ShotTrajectoryHead`)."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.latent = nn.Linear(width, 2 * width)
        self.first_norm = norm(width)
        self.middle = nn.Linear(width, width)
        self.middle_norm = norm(width)
        self.last = nn.Linear(width, WIDTH)
        for layer, gain in ((self.latent, FIRST_GAIN), (self.middle, FIRST_GAIN)):
            for block in layer.weight.split(width):
                _orthogonal(block, gain)
            nn.init.zeros_(layer.bias)
        _orthogonal(self.last.weight, LAST_GAIN)
        nn.init.zeros_(self.last.bias)

    def query(self, latents: torch.Tensor, partners: torch.Tensor) -> torch.Tensor:
        """latents (..., N, 2 D) from `self.latent`; partners (..., S, K) row indices, the
        anchors the first S rows. Returns (..., S, K, WIDTH)."""
        anchors = partners.shape[-2]
        own = latents[..., :anchors, : self.width]
        flat = partners.flatten(-2)
        other = latents[..., self.width :].gather(
            -2, flat[..., None].expand(*flat.shape, self.width)
        )
        hidden = own[..., None, :] + other.unflatten(-2, partners.shape[-2:])
        hidden = self.middle(F.gelu(self.first_norm(hidden)))
        return self.last(F.gelu(self.middle_norm(hidden)))
