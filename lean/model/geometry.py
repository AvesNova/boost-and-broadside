"""The relative-geometry heads (representation shaping, queue item 8c, author 2026-11-28):
categorical targets of the current geometry as the team sees it, from the game's own
functions (lean.game.pair_geometry, field_edge_distance, geometry_record's edge column).

- Ship pairs (`PairHead`): per ship two gated projections of its trunk token, u (the ship's
  side, shared) and v (the other side, its entity kind's), combined after the projections, u_i +
  v_j (or u_i x v_j), GELU, then one Linear per target. Ship-field pairs the same with a
  field's own other-side projection of its map memory. Gating only in the per-ship stage.
  The learner picks the pairs (lean.learn.geometry: K valid pairs per view and tick, sampled
  before the pair layers); symmetric and antisymmetric targets train on the upper triangle
  (first < second) only, the order-dependent ones in both orders.
- The map edge: a per-ship column on the auxiliary head group (lean.model.policy).

Every target is a categorical in nats over its own bins, HL-Gauss (`GeometryColumn`): uniform
in bin space (lean.model.value.symlog with the column's knee) across the measured range plus
the buffer columns (`buffered_centres`), or for an angle a circle of bins over (-pi, pi]
whose Gaussian wraps (`circular_centres`, `circular_hl_gauss_targets`), uniform in the
warped angle s(theta) = symlog(theta, knee) (knee 0: uniform in the angle).
"""

import math
from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from lean.model.value import buffered_centres, hl_gauss_targets, symlog


@dataclass(frozen=True)
class GeometryColumn:
    """One geometry target: `bins` bins; a measured range `low` to `high` (target units) in the
    bin space of `knee` (0 linear) with the buffer columns beyond, or `circular` (an angle in
    radians: the circle of bins, `low` and `high` unused). `both_orders`: a pair target that
    depends on the order (trained on both orders of a sampled pair; the others on the upper
    triangle only)."""

    name: str
    bins: int
    knee: float
    low: float = 0.0
    high: float = 0.0
    circular: bool = False
    both_orders: bool = False


# Ranges and knees from the default scripted agent against itself under the training game,
# every 8th tick, both teams' views, the rows the team sees (2026-11-28, queue item 15, the
# core rules with the shrinking zone; docs/rewrite/sections/calibrate-15/): 4v4 (256 of 1,024
# environments x 8,000 ticks: 5.3M ship pairs, 24.7M ship-field pairs, 2.5M ships), 1v1
# and 16v16. Each range is the 99.9% interval rounded out over the three team sizes (the
# buffer columns take the rest), each knee about the finest scale that matters (a
# collision's tens of metres, a hit's 8 mrad aim tolerance at 300 m, the field band's 160 m,
# the boundary ramp's 300 m) traded against the bins' balance. The ship-pair columns (first
# measured 2026-11-28 ~04:00 on the fixed arena) still fit: 99% of the mass in 20-32 of 32
# bins at 4v4 and 16v16, no bin above 13%. Changed by item 15: the field edge (-2,000 to
# 7,000 m, knee 50: the largest bin 21%, perplexity 9.7 -> -1,500 to 6,200 m, knee 200: 15%,
# 13.2; innermost bin 50 m) and the map edge, now the shrinking zone's edge, which every
# ship nears (16 bins, -4,200 to 300 m, knee 100: 99% in 6 of 16, largest 38-46% -> 32
# bins, -3,900 to 300 m, knee 1,000: 99% in 19 of 32, largest 8-15%; innermost bin 77 m).
PAIR_COLUMNS = (
    GeometryColumn("range", 32, 100.0, 0.0, 6700.0),  # m
    GeometryColumn("range_rate", 32, 50.0, -1000.0, 1000.0),  # m/s
    GeometryColumn("line_of_sight_rate", 32, 0.05, -6.0, 6.0),  # rad/s
    GeometryColumn("energy_difference", 32, 0.0, -33000.0, 33000.0),  # store (the full store)
    GeometryColumn("train_angle", 32, 0.0, circular=True, both_orders=True),  # rad
    GeometryColumn("gun_error", 32, 0.02, circular=True, both_orders=True),  # rad
)
FIELD_COLUMN = GeometryColumn("field_edge", 32, 200.0, -1500.0, 6200.0)  # m, negative inside
EDGE_COLUMN = GeometryColumn("map_edge", 32, 1000.0, -3900.0, 300.0)  # m, negative inside


def circular_centres(knee: float, bins: int) -> Tensor:
    """(bins,) float32 bin-space centres of a circle of bins over the angles (-pi, pi]: uniform
    in s = symlog(angle, knee) over [-s(pi), s(pi)), the first bin's lower edge at -s(pi)."""
    half = float(symlog(torch.tensor(math.pi), knee))
    width = 2.0 * half / bins
    return -half + width * (torch.arange(bins, dtype=torch.float32) + 0.5)


def circular_hl_gauss_targets(
    values: Tensor, centres: Tensor, knee: float, sigma_bins: float = 0.75
) -> Tensor:
    """HL-Gauss target probabilities (..., bins) for angles `values` (...) radians on a circle
    of bins (`circular_centres`): the Gaussian centred on s(angle wrapped to (-pi, pi]) in bin
    space, each bin taking the mass over its nearest image (the offset wrapped to one period),
    renormalised. No seam: angles either side of +-pi get neighbouring distributions."""
    centres = centres.float()
    width = centres[1] - centres[0]
    period = width * centres.shape[-1]
    wrapped = torch.atan2(torch.sin(values.float()), torch.cos(values.float()))
    position = symlog(wrapped, knee)
    offset = torch.remainder(centres - position[..., None] + 0.5 * period, period) - 0.5 * period
    scale = sigma_bins * width * math.sqrt(2.0)
    mass = 0.5 * (
        torch.erf((offset + 0.5 * width) / scale) - torch.erf((offset - 0.5 * width) / scale)
    )
    return mass / mass.sum(-1, keepdim=True)


def column_centres(column: GeometryColumn) -> Tensor:
    """The column's bin-space centres, (bins,) float32."""
    if column.circular:
        return circular_centres(column.knee, column.bins)
    return buffered_centres(column.low, column.high, column.knee, column.bins)


def column_targets(column: GeometryColumn, values: Tensor, centres: Tensor, sigma: float) -> Tensor:
    """HL-Gauss probabilities (..., bins) of the column's targets `values` (...)."""
    if column.circular:
        return circular_hl_gauss_targets(values, centres, column.knee, sigma)
    return hl_gauss_targets(values, centres, column.knee, sigma)


def column_cross_entropy(
    column: GeometryColumn, logits: Tensor, values: Tensor, centres: Tensor, sigma: float
) -> Tensor:
    """The cross-entropy in nats (...) of logits (..., bins) against the column's targets."""
    target = column_targets(column, values, centres, sigma)
    return -(target * torch.log_softmax(logits.float(), -1)).sum(-1)


class PairQuery(NamedTuple):
    """The pairs the pairwise head evaluates, per view and tick: ship pairs `first` < `second`
    (T, E, K) long ship slots, and ship-field pairs `field_ship` (T, E, K_f) slots with
    `field_index` (T, E, K_f) field slots (None with no fields)."""

    first: Tensor
    second: Tensor
    field_ship: Tensor | None
    field_index: Tensor | None


def take(tokens: Tensor, index: Tensor) -> Tensor:
    """Rows of tokens (T, E, N, ...) at slots index (T, E, K): (T, E, K, ...)."""
    shape = index.shape + tokens.shape[3:]
    flat = index.reshape(index.shape + (1,) * (tokens.dim() - 3)).expand(shape)
    return tokens.gather(2, flat)


class PairHead(nn.Module):
    """The pairwise relative-geometry head (module docstring). `ship_side`, `other_ship` and
    `other_field` are gated projections (value x sigmoid(gate), one Linear each) of the normed
    trunk token or map memory to `pair_hidden`; a pair is GELU(u_i + v_j) (or of u_i x v_j,
    `pair_combine`), then one Linear per column (`PAIR_COLUMNS` order, then the field
    column). The last layers start orthogonal with gain 0.01 (near-uniform categoricals, as
    every head's), biases zero."""

    def __init__(self, config):
        super().__init__()
        hidden = config.pair_hidden
        self.combine = config.pair_combine
        if self.combine not in ("sum", "product"):
            raise ValueError(f"pair_combine {self.combine!r}, expected 'sum' or 'product'")
        self.ship_norm = nn.RMSNorm(config.width, eps=config.norm_epsilon)
        self.ship_side = nn.Linear(config.width, 2 * hidden)
        self.other_ship = nn.Linear(config.width, 2 * hidden)
        self.field_norm = nn.RMSNorm(config.map_width, eps=config.norm_epsilon)
        self.other_field = nn.Linear(config.map_width, 2 * hidden)
        self.columns = config.pair_columns
        self.ship_lasts = nn.ModuleList(nn.Linear(hidden, column.bins) for column in self.columns)
        self.field_last = nn.Linear(hidden, config.field_column.bins)
        with torch.no_grad():
            for last in (*self.ship_lasts, self.field_last):
                nn.init.orthogonal_(last.weight, 0.01)
                last.bias.zero_()

    @staticmethod
    def _gated(layer: nn.Linear, tokens: Tensor) -> Tensor:
        value, gate = layer(tokens).chunk(2, -1)
        return value * torch.sigmoid(gate)

    def _pair(self, this: Tensor, other: Tensor) -> Tensor:
        return F.gelu(this + other if self.combine == "sum" else this * other)

    def forward(self, ships: Tensor, memory: Tensor, query: PairQuery):
        """ships (T, E, N, width) trunk tokens, memory (T, E, F, map_width). Returns the ship
        pairs' logits, one per column of `pair_columns` ((T, E, K, bins) for the first <
        second order; (T, E, 2K, bins) for a both-orders column: first on second, then second
        on first), and the field pairs' (T, E, K_f, bins) (None with no fields)."""
        normed = self.ship_norm(ships)
        this, other = self._gated(self.ship_side, normed), self._gated(self.other_ship, normed)
        forward = self._pair(take(this, query.first), take(other, query.second))
        needs_reverse = any(column.both_orders for column in self.columns)
        reverse = (
            self._pair(take(this, query.second), take(other, query.first))
            if needs_reverse
            else None
        )
        ship_logits = [
            last(torch.cat([forward, reverse], 2) if column.both_orders else forward)
            for column, last in zip(self.columns, self.ship_lasts, strict=True)
        ]
        field_logits = None
        if query.field_ship is not None:
            fields = self._gated(self.other_field, self.field_norm(memory))
            field_logits = self.field_last(
                self._pair(take(this, query.field_ship), take(fields, query.field_index))
            )
        return ship_logits, field_logits
