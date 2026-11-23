"""Position codes for the token inputs, and the rotary angles of attention.

A position code is one small module: positions (..., 2) metres in, features (..., features)
out, with `features` (the output width) and `span` (metres: the range of the random position
offset, one coarsest period, so the phase of every scale is uniform). The model builds the
one its config names (`make_position_code`), so a compiled graph holds only that one.
Phases are computed in float32 even under bf16 autocast: thousands of metres over a 64 m
period is hundreds of radians, and bf16 keeps 8 bits.
"""

import math

import torch
from torch import Tensor, nn

# Three unit directions 60 degrees apart (a hexagonal layout).
HEXAGONAL_ANGLES = (0.0, math.pi / 3.0, 2.0 * math.pi / 3.0)


def _directions(angles) -> Tensor:
    angles = torch.tensor(angles, dtype=torch.float64)
    return torch.stack([angles.cos(), angles.sin()], -1)


class FourierCode(nn.Module):
    """cos and sin of the position's phase along three directions 60 degrees apart at each
    period: 2 * 3 * len(periods) features (Space2Vec's "theory" layout)."""

    def __init__(self, periods: tuple[float, ...]):
        super().__init__()
        rates = 2.0 * math.pi / torch.tensor(periods, dtype=torch.float64)
        waves = (_directions(HEXAGONAL_ANGLES)[:, None, :] * rates[None, :, None]).reshape(-1, 2)
        self.register_buffer("waves", waves.float(), persistent=False)
        self.features = 2 * waves.shape[0]
        self.span = max(periods)

    def forward(self, positions: Tensor) -> Tensor:
        with torch.autocast(positions.device.type, enabled=False):
            phase = positions.float() @ self.waves.T
            return torch.cat([phase.cos(), phase.sin()], -1)


class GridCellCode(nn.Module):
    """Grid-cell firing maps (Blair, Welday and Zhang 2007, "cosine grating model"): per
    grid spacing L and per spatial phase s, g(cos(k1.(x - s)) + cos(k2.(x - s)) +
    cos(k3.(x - s))) with k1, k2, k3 at 0, 60 and 120 degrees, |k| = 4 pi / (sqrt(3) L), and
    Blair's gain g(u) = exp(0.3 (u + 1.5)) - 1 (zero at the sum's minimum -1.5).

    The phases: k3 = k2 - k1, so a shift s gives phases p1 = k1.s, p2 = k2.s, p3 = p2 - p1.
    Per spacing, `phases` x `phases` shifts with (p1, p2) on an even grid of [0, 2 pi)^2,
    which tiles the grid's unit cell. Features: len(spacings) * phases^2.
    """

    GAIN, OFFSET = 0.3, -1.5  # Blair et al. 2007, a and b

    def __init__(self, spacings: tuple[float, ...], phases: int):
        super().__init__()
        rates = 4.0 * math.pi / (math.sqrt(3.0) * torch.tensor(spacings, dtype=torch.float64))
        waves = (rates[:, None, None] * _directions(HEXAGONAL_ANGLES)[None]).reshape(-1, 2)
        steps = 2.0 * math.pi * torch.arange(phases, dtype=torch.float64) / phases
        first, second = torch.meshgrid(steps, steps, indexing="ij")
        shift = torch.stack([first, second, second - first], -1).reshape(-1, 3)  # (phases^2, 3)
        self.register_buffer("waves", waves.float(), persistent=False)  # (scales * 3, 2)
        self.register_buffer("shift", shift.float(), persistent=False)
        self.scales = len(spacings)
        self.features = self.scales * shift.shape[0]
        self.span = max(spacings)

    def forward(self, positions: Tensor) -> Tensor:
        with torch.autocast(positions.device.type, enabled=False):
            phase = (positions.float() @ self.waves.T).unflatten(-1, (self.scales, 1, 3))
            total = torch.cos(phase - self.shift).sum(-1)  # (..., scales, phases^2)
            return (torch.exp(self.GAIN * (total - self.OFFSET)) - 1.0).flatten(-2)


class ColourCode(nn.Module):
    """v2's categorical position code with no spread (src/bnb/models/yemong/codes.py
    `position_encode` at sigma 0): per axis and level l a 3-colour circular code of the
    coordinate (m mod P / 3^l) * 3^(l + 1) / P (1 - fraction on the colour below it, the
    fraction on the next, cyclically), and per level the outer product of the x and y codes:
    levels * 9 features. P = period, the coarsest; the finest colour is P / 3^levels wide."""

    COLOURS = 3

    def __init__(self, period: float, levels: int):
        super().__init__()
        level_periods = period / self.COLOURS ** torch.arange(levels, dtype=torch.float64)
        self.register_buffer("level_periods", level_periods.float(), persistent=False)
        self.features = levels * self.COLOURS**2
        self.span = period

    def forward(self, positions: Tensor) -> Tensor:
        with torch.autocast(positions.device.type, enabled=False):
            # (..., 2, levels): each level reduced modulo its own period first (float32 keeps
            # the coordinate below 3, as v2's circular_coordinates does).
            coordinate = torch.remainder(positions.float()[..., None], self.level_periods)
            coordinate = coordinate * (self.COLOURS / self.level_periods)
            lower = torch.floor(coordinate)
            fraction = coordinate - lower
            colour = torch.remainder(lower, self.COLOURS)
            colours = torch.arange(self.COLOURS, device=positions.device, dtype=torch.float32)
            above = torch.remainder(colour + 1.0, self.COLOURS)
            code = (1.0 - fraction)[..., None] * (colour[..., None] == colours) + fraction[..., None] * (
                above[..., None] == colours
            )  # (..., 2, levels, 3)
            x, y = code.unbind(-3)
            return (x[..., :, None] * y[..., None, :]).flatten(-3)


POSITION_CODES = ("fourier", "grid_cells", "v2_codes")


def make_position_code(config) -> nn.Module:
    """The position code named by config.position_code (`POSITION_CODES`)."""
    if config.position_code == "fourier":
        return FourierCode(config.position_periods)
    if config.position_code == "grid_cells":
        return GridCellCode(config.position_periods, config.grid_phases)
    if config.position_code == "v2_codes":
        return ColourCode(config.colour_period, config.colour_levels)
    raise ValueError(f"position_code {config.position_code!r} is not one of {POSITION_CODES}")


class Rotary(nn.Module):
    """Rotary angles: per band two perpendicular axes (v2's layout), band k's turned k *
    rotary_turn degrees, at rate 2 pi / period; then the heading harmonics.
    angles(positions (..., 2), heading (...)) -> (..., pairs)."""

    def __init__(self, config):
        super().__init__()
        axes, rates = [], []
        for index, period in enumerate(config.rotary_periods):
            turn = math.radians(index * config.rotary_turn)
            axes += [(math.cos(turn), math.sin(turn)), (-math.sin(turn), math.cos(turn))]
            rates += [2.0 * math.pi / period] * 2
        waves = torch.tensor(axes, dtype=torch.float64) * torch.tensor(rates, dtype=torch.float64)[:, None]
        self.register_buffer("waves", waves.float(), persistent=False)
        self.register_buffer("harmonics", torch.tensor(config.heading_harmonics, dtype=torch.float32), persistent=False)

    def angles(self, positions: Tensor, heading: Tensor) -> Tensor:
        with torch.autocast(positions.device.type, enabled=False):
            return torch.cat([positions.float() @ self.waves.T, heading.float()[..., None] * self.harmonics], -1)
