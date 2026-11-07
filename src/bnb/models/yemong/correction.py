"""The error head's code (`YemongConfig.delta_split`; arms 3-4 brief): for a hidden ship, how
wrong the physics forecast is, e = truth(t + 1) - forecast(t + 1), and the belief advance
that moves the forecast toward the truth by it, with the belief's spread taken from it.

Every quantity of the state code is a symlog HL-Gauss code (`codes.symlog_encode`, exact
zero centre, PADDING_BINS each side) of its uncapped error over the full range a hidden
ship's error can reach. Position and velocity are coded on three world projection axes (0,
60 and 120 degrees, `AXES`), one code each: a hidden ship's error is correlated across any
fixed pair of axes (baselines-bench.md 2.2), and three marginals give the full 2 x 2
covariance. The mean error is their least-squares mean, the covariance the solution of
u_i^T S u_i = var_i, clamped to positive semi-definite (`covariance_of`).

The advance (`correct`): mean(t + 1) = forecast + step per quantity, a rate-limited pseudo
gradient step toward the truth. The step is eta g m: e's median m (per projection axis,
then the least-squares vector), shrunk by the head's confidence in it, g = |m|^2 / (|m|^2 +
sigma^2) with sigma^2 e's variance per dimension (`confident`), times a fixed rate per tick
eta (`YemongConfig.correction_rate`, 0.1: a time constant of about 10 ticks), then capped:
for position and velocity the 2D vector r tanh(|eta g m| / r) m / |m|, for the heading, the
nose rate, the angle of attack, energy and ln n the scalar r tanh(eta g m / r), r the
quantity's one-tick physical bound (`STEP_LIMITS`): the belief moves toward the truth no
faster than the ship could move in a tick. The head re-predicts e every tick from the
corrected belief, so a fraction per tick still converges where it is right, and the loop's
gain stays below 1 where it is wrong or one update stale. The rare-jump quantities
(recharge wait, shield, gun cooldown) step by m itself (the most likely outcome, decisions
D8). The variance is e's second moment about the step, Var e + (E e - step)^2 (for position
and velocity the 2 x 2 covariance plus the outer product of the unapplied part), and
replaces the physics' F P F^T + Q for hidden rows.

Why not the brief's step toward E e (`docs/explanation/yemong.md`, the error head): on a
short CPU run it ran away. The mean of a broad code over +-26 km is set by its tails, and
an early head predicts one offset for every row; stepped toward at up to r per tick and
integrated over a hidden ship's age, that push dominates the next update's errors, which
the head then chases. The median alone did the same a little later. The shrinkage keeps an
unconditioned prediction from moving the belief, and the rate keeps the loop's gain small.
"""

import dataclasses
import math

import torch
import torch.nn as nn

from bnb.game.ship_types import per_ship

from . import codes
from .delta import (
    UNIFORM_HEADING_VARIANCE,
    UNIFORM_NOSE_RATE_VARIANCE,
    DeltaComponent,
    cap_variance,
    wrap,
)
from .state_code import (
    LOG_INDEX_SPAN,
    MEANS,
    NOSE_RATE_LIMIT,
    POSITION_PERIOD,
    SPREADS,
    ShipStateCode,
    StateCodeRanges,
    pack_covariance,
    unpack_covariance,
    velocity_reach,
)

AXES = (0.0, 60.0, 120.0)  # degrees: the world projection axes of position and velocity
PLANAR = ("position", "velocity")
# The quantities, in code order: position and velocity one code per axis, then the scalars.
SCALAR_QUANTITIES = (
    "heading",
    "nose_rate",
    "angle_of_attack",
    "shield_recharge_wait",
    "shield",
    "energy",
    "log_refractive_index",
    "gun_cooldown",
)
QUANTITIES = PLANAR + SCALAR_QUANTITIES
COMPONENT_NAMES = (
    *(f"position_{int(angle)}" for angle in AXES),
    *(f"velocity_{int(angle)}" for angle in AXES),
    *SCALAR_QUANTITIES,
)
# The quantities that step by e's median (a rare jump: the most likely outcome, D8), and
# the bins either side of the median's bin it averages: these codes are coarse (two bins
# per e-fold), so the delta code's window of 4 would reach a jump's mode; 2 bins hold the
# HL-Gauss of one value to 2.7 sigma.
MEDIAN = ("shield_recharge_wait", "shield", "gun_cooldown")
MEDIAN_WINDOW = 2
# Each quantity's one-tick physical bound r (`correct`): the baselines bench's persistence
# maxima (baselines-bench-tables.md, "One tick, pooled", rows P, max; position and velocity
# the norm's), in the code's units (shield and energy fractions of the maximum).
STEP_LIMITS = dict(
    position=7.99,  # m
    velocity=40.6,  # m/s
    heading=0.311,  # rad
    nose_rate=10.2,  # rad/s
    angle_of_attack=0.163,  # rad
    energy=0.0457,  # of the type's maximum
    log_refractive_index=0.232,
)
# The knees: the median |e| per quantity (position and velocity over their three axes; the
# wait and the shield, exactly 0 at 92% and 91%, the median above 1e-6) on a short CPU run
# of this arm at initialisation, 139,922 hidden rows (bench/correction_marginal.py: arm 2's
# BC run's config, 64 environments x 64 ticks x 6 updates, seed 3), where the belief of a
# hidden ship is the physics head's open-loop forecast from its starting marginal.
KNEES = dict(
    position=11.9,  # m, per projection axis
    velocity=9.09,  # m/s, per projection axis
    heading=0.160,  # rad
    nose_rate=0.587,  # rad/s
    angle_of_attack=0.0267,  # rad
    shield_recharge_wait=0.217,  # s
    shield=0.0421,  # of the type's maximum
    energy=0.00918,  # of the type's maximum
    log_refractive_index=3.47e-4,
    gun_cooldown=0.0334,  # s
)
# The starting marginal (`marginal_logits`), from the same run: per component the share of
# errors exactly 0 and 16 quantiles of the rest (at (k + 1/2) / 16).
MARGINAL: dict[str, tuple[float, tuple[float, ...]]] = {
    "position_0": (
        0.0001,
        (
            -414.0,
            -98.93,
            -40.12,
            -19.41,
            -9.822,
            -4.777,
            -2.184,
            -0.7344,
            0.1992,
            1.426,
            3.411,
            7.633,
            16.03,
            34.25,
            91.87,
            444.0,
        ),
    ),
    "position_60": (
        0.0,
        (
            -351.3,
            -94.49,
            -37.34,
            -18.12,
            -8.739,
            -4.084,
            -1.787,
            -0.4602,
            0.4507,
            1.715,
            3.761,
            8.056,
            15.8,
            33.34,
            95.98,
            423.5,
        ),
    ),
    "position_120": (
        0.0,
        (
            -298.6,
            -65.04,
            -27.88,
            -13.69,
            -6.473,
            -3.009,
            -1.253,
            -0.1753,
            0.7648,
            2.159,
            4.649,
            9.599,
            19.04,
            42.74,
            102.8,
            371.0,
        ),
    ),
    "velocity_0": (
        0.0,
        (
            -67.55,
            -41.84,
            -23.33,
            -13.7,
            -8.14,
            -4.593,
            -2.239,
            -0.7753,
            0.2687,
            1.504,
            3.37,
            6.415,
            10.98,
            18.35,
            32.68,
            60.76,
        ),
    ),
    "velocity_60": (
        0.0,
        (
            -67.57,
            -35.02,
            -18.62,
            -10.96,
            -6.502,
            -3.512,
            -1.703,
            -0.4955,
            0.5013,
            1.771,
            3.873,
            7.308,
            13.0,
            21.16,
            37.36,
            65.79,
        ),
    ),
    "velocity_120": (
        0.0,
        (
            -56.79,
            -29.68,
            -16.6,
            -9.156,
            -5.3,
            -2.799,
            -1.208,
            -0.1206,
            0.8933,
            2.507,
            4.865,
            8.322,
            14.1,
            22.89,
            39.18,
            64.24,
        ),
    ),
    "heading": (
        0.0,
        (
            -1.438,
            -0.5715,
            -0.3258,
            -0.2041,
            -0.1259,
            -0.07551,
            -0.03792,
            -0.01229,
            0.007689,
            0.03242,
            0.0701,
            0.1248,
            0.2021,
            0.3257,
            0.5682,
            1.37,
        ),
    ),
    "nose_rate": (
        0.0,
        (
            -2.355,
            -1.501,
            -1.035,
            -0.702,
            -0.4686,
            -0.2755,
            -0.1337,
            -0.02027,
            0.02141,
            0.1375,
            0.2871,
            0.4813,
            0.7164,
            1.045,
            1.496,
            2.367,
        ),
    ),
    "angle_of_attack": (
        0.0,
        (
            -0.1237,
            -0.07556,
            -0.05138,
            -0.03413,
            -0.02102,
            -0.01225,
            -0.006592,
            -0.001816,
            0.001068,
            0.005922,
            0.01165,
            0.02054,
            0.03312,
            0.05039,
            0.07611,
            0.1326,
        ),
    ),
    "shield_recharge_wait": (
        0.9165,
        (
            -1.732,
            -0.9439,
            -0.609,
            -0.348,
            -0.116,
            0.01733,
            0.03467,
            0.052,
            0.07367,
            0.09967,
            0.13,
            0.1777,
            0.2643,
            4.167,
            6.0,
            6.0,
        ),
    ),
    "shield": (
        0.9116,
        (
            -0.4251,
            -0.2192,
            -0.1496,
            -0.114,
            -0.08829,
            -0.07213,
            -0.05361,
            -0.03911,
            -0.02588,
            -0.01427,
            -0.006573,
            -0.003567,
            -0.00218,
            -0.00109,
            -0.0001982,
            0.05153,
        ),
    ),
    "energy": (
        0.0,
        (
            -0.08161,
            -0.0352,
            -0.02316,
            -0.01392,
            -0.008517,
            -0.005099,
            -0.002647,
            -0.000796,
            0.0007899,
            0.002564,
            0.004518,
            0.006789,
            0.009684,
            0.01413,
            0.03011,
            0.1163,
        ),
    ),
    "log_refractive_index": (
        0.0,
        (
            -0.2011,
            -0.003617,
            -0.001689,
            -0.001138,
            -0.0006487,
            -0.0004796,
            -0.000373,
            -0.0002844,
            -0.0002222,
            -0.0001689,
            -0.0001244,
            -8.887e-05,
            -5.332e-05,
            -2.666e-05,
            -1.777e-05,
            0.3458,
        ),
    ),
    "gun_cooldown": (
        0.0,
        (
            -0.06684,
            -0.06674,
            -0.0667,
            -0.06668,
            -0.03357,
            -0.03346,
            -0.03341,
            -0.03338,
            -0.03336,
            -0.03335,
            -0.001604,
            -0.0009701,
            -0.0006386,
            -0.0003437,
            -0.0001072,
            -9.75e-06,
        ),
    ),
}


def odd_bins(knee: float, bound: float) -> int:
    """The real bins of a code from knee to bound: about 1 + 4 ln(1 + bound / knee), odd
    (two bins per e-fold on each side of the zero)."""
    return 2 * round(2.0 * math.log1p(bound / knee)) + 1


def bounds(ranges: StateCodeRanges) -> dict[str, float]:
    """Each quantity's largest |e| (the code's bound, uncapped): position the code's period
    (the largest meaningful offset), velocity twice the velocity code's reach, the heading
    pi, the rest the span of their state-code ranges."""
    return dict(
        position=POSITION_PERIOD,
        velocity=2.0 * velocity_reach(),
        heading=math.pi,
        nose_rate=2.0 * NOSE_RATE_LIMIT,
        angle_of_attack=2.0 * ranges.angle_of_attack_limit,
        shield_recharge_wait=ranges.shield_recharge_delay,
        shield=1.0,
        energy=1.0,
        log_refractive_index=2.0 * LOG_INDEX_SPAN,
        gun_cooldown=(ranges.cooldown_bins - 1) * ranges.tick_length,
    )


def components(ranges: StateCodeRanges) -> tuple[DeltaComponent, ...]:
    """The code's components in COMPONENT_NAMES order."""
    bound, out = bounds(ranges), []
    for name in COMPONENT_NAMES:
        quantity = name.rsplit("_", 1)[0] if name.startswith(PLANAR) else name
        knee = KNEES[quantity]
        out.append(DeltaComponent(name, knee, bound[quantity], odd_bins(knee, bound[quantity])))
    return tuple(out)


def axes(dtype=torch.float64) -> torch.Tensor:
    """(3, 2) the unit projection axes."""
    radians = torch.tensor(AXES, dtype=torch.float64) * (math.pi / 180.0)
    return torch.stack([torch.cos(radians), torch.sin(radians)], -1).to(dtype)


def least_squares(dtype=torch.float64) -> torch.Tensor:
    """(2, 3) the least-squares inverse of the projection: a vector of three projections."""
    return torch.linalg.pinv(axes()).to(dtype)


def variance_solve(dtype=torch.float64) -> torch.Tensor:
    """(3, 3) the inverse of var_i = c_i^2 S_xx + 2 c_i s_i S_xy + s_i^2 S_yy."""
    unit = axes()
    cos, sin = unit[:, 0], unit[:, 1]
    return torch.linalg.inv(torch.stack([cos * cos, 2.0 * cos * sin, sin * sin], -1)).to(dtype)


def project(vector: torch.Tensor, unit: torch.Tensor) -> torch.Tensor:
    """(..., 2) world vectors to their (..., 3) projections on the axes `unit` (3, 2)."""
    return vector @ unit.to(vector.dtype).T


def mean_of(projections: torch.Tensor, solve: torch.Tensor) -> torch.Tensor:
    """The least-squares (..., 2) vector of (..., 3) projected means (solve: `least_squares`)."""
    return projections @ solve.to(projections.dtype).T


def positive_semidefinite(packed: torch.Tensor) -> torch.Tensor:
    """A packed 2 x 2 symmetric (xx, xy, yy) (..., 3) with its eigenvalues clamped at 0."""
    xx, xy, yy = packed.unbind(-1)
    centre, half = 0.5 * (xx + yy), 0.5 * (xx - yy)
    radius = torch.sqrt(half * half + xy * xy)
    major = torch.clamp_min(centre + radius, 0.0)
    minor = torch.clamp_min(centre - radius, 0.0)
    safe = torch.where(radius > 0, radius, 1.0)
    cos = torch.where(radius > 0, half / safe, 1.0)
    sin = torch.where(radius > 0, xy / safe, 0.0)
    middle, spread = 0.5 * (major + minor), 0.5 * (major - minor)
    return torch.stack([middle + spread * cos, spread * sin, middle - spread * cos], -1)


def covariance_of(variances: torch.Tensor, solve: torch.Tensor) -> torch.Tensor:
    """The packed 2 x 2 covariance (xx, xy, yy) (..., 3) whose variance on each axis is
    variances (..., 3) (solve: `variance_solve`), clamped to positive semi-definite."""
    return positive_semidefinite(variances @ solve.to(variances.dtype).T)


def outer(vector: torch.Tensor) -> torch.Tensor:
    """The packed outer product (xx, xy, yy) of (..., 2)."""
    x, y = vector[..., 0], vector[..., 1]
    return torch.stack([x * x, x * y, y * y], -1)


def confident(median: torch.Tensor, square: torch.Tensor, variance: torch.Tensor):
    """The median m shrunk by the head's confidence in it, m |m|^2 / (|m|^2 + sigma^2):
    square |m|^2 (...), variance sigma^2 (...) per dimension; median (...) or (..., 2). A
    broad prediction (sigma >> |m|) moves nothing, a sharp one moves by m."""
    gain = square / torch.clamp_min(square + variance, torch.finfo(square.dtype).tiny)
    return median * (gain[..., None] if median.dim() > gain.dim() else gain)


def capped_step(mean: torch.Tensor, limit: float) -> torch.Tensor:
    """(..., 2) r tanh(|m| / r) m / |m| of mean vectors (..., 2): the full m well below r,
    length r far above it."""
    length = torch.linalg.vector_norm(mean, dim=-1, keepdim=True)
    safe = torch.where(length > 0, length, 1.0)
    return torch.where(length > 0, limit * torch.tanh(length / limit) * mean / safe, 0.0)


@dataclasses.dataclass(frozen=True)
class Moments:
    """The error head's decoded distribution, per quantity (QUANTITIES): mean (..., 2) for
    position and velocity, (...) for the rest; variance, the packed 2 x 2 covariance (...,
    3) or (...); target, what the step moves by before the cap: eta g m, e's median m (per
    projection, then the least-squares vector) shrunk by `confident` and the rate eta (the
    rare jumps': m itself); step, what the advance applies (same shape as mean)."""

    mean: dict[str, torch.Tensor]
    variance: dict[str, torch.Tensor]
    target: dict[str, torch.Tensor]
    step: dict[str, torch.Tensor]

    def applied_variance(self, name: str) -> torch.Tensor:
        """Var e + (E e - step)^2: the variance the belief carries after the step."""
        rest = self.mean[name] - self.step[name]
        if name in PLANAR:
            return self.variance[name] + outer(rest)
        return self.variance[name] + rest * rest


class CorrectionCode(nn.Module):
    """The error head's code of a Yemong build: `errors` (e of a forecast against the truth),
    `targets` (its training code), `moments` (head logits to the decoded distribution and
    the step) and `correct` (the forecast at t + 1 to the corrected belief)."""

    def __init__(self, ranges: StateCodeRanges, rate: float = 0.1) -> None:
        super().__init__()
        if not 0.0 < rate <= 1.0:
            raise ValueError(f"the correction rate {rate} must be in (0, 1]")
        self.rate = rate  # eta: the share of the confident median applied per tick
        self.tick_length = ranges.tick_length
        tensor = lambda values: torch.tensor(values, dtype=torch.float32)  # noqa: E731
        self.register_buffer("max_shield", tensor(ranges.max_shield))
        self.register_buffer("max_energy", tensor(ranges.max_energy))
        self.wait_limit = ranges.shield_recharge_delay
        self.cooldown_bins = ranges.cooldown_bins
        self.cooldown_limit = (ranges.cooldown_bins - 1) * ranges.tick_length
        self.components = components(ranges)
        self.register_buffer("axes", axes(torch.float32), persistent=False)
        self.register_buffer("least_squares", least_squares(torch.float32), persistent=False)
        self.register_buffer("variance_solve", variance_solve(torch.float32), persistent=False)
        self.order = COMPONENT_NAMES
        self.sizes = {component.name: component.width for component in self.components}
        self.runs: codes.Runs = tuple((1, component.width) for component in self.components)
        self._inverse_ends = {}
        for component in self.components:
            table, lowest, highest = codes.symlog_mean_inverse(component.bins)
            self.register_buffer(f"inverse_{component.name}", table.float(), persistent=False)
            self._inverse_ends[component.name] = (lowest, highest)

    @property
    def width(self) -> int:
        """The code's width."""
        return sum(groups * size for groups, size in self.runs)

    def inverse(self, name: str) -> tuple[torch.Tensor, float, float]:
        """A component's `codes.symlog_mean_inverse` table and ends."""
        return (getattr(self, f"inverse_{name}"), *self._inverse_ends[name])

    def _maxima(self, ship_type: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return per_ship(self.max_shield, ship_type), per_ship(self.max_energy, ship_type)

    # ------------------------------------------------------------------ targets

    def errors(
        self, forecast: torch.Tensor, truth: torch.Tensor, ship_type: torch.Tensor
    ) -> torch.Tensor:
        """(..., len(COMPONENT_NAMES)) e = truth - forecast of states (..., len(MEANS)) in the
        code's units: position and velocity projected on AXES, the heading wrapped, shield
        and energy fractions of the type's maximum."""
        index = {name: i for i, name in enumerate(MEANS)}
        max_shield, max_energy = self._maxima(ship_type)
        difference = truth - forecast
        column = lambda name: difference[..., index[name]]  # noqa: E731
        return torch.cat(
            [
                project(difference[..., 0:2], self.axes),
                project(difference[..., 2:4], self.axes),
                torch.stack(
                    [
                        wrap(column("heading")),
                        column("nose_rate"),
                        column("angle_of_attack"),
                        column("shield_recharge_wait"),
                        column("shield") / max_shield,
                        column("energy") / max_energy,
                        column("log_refractive_index"),
                        column("gun_cooldown"),
                    ],
                    -1,
                ),
            ],
            -1,
        )

    def encode(self, errors: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(the code (..., width), (..., components) bool: clamped to the padded range) of
        `errors`."""
        parts, clipped = [], []
        for i, component in enumerate(self.components):
            code, clip = component.encode(errors[..., i])
            parts.append(code)
            clipped.append(clip)
        return torch.cat(parts, -1), torch.stack(clipped, -1)

    def targets(
        self, forecast: torch.Tensor, truth: torch.Tensor, ship_type: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The training code of the forecast's error against the truth, and the clipped flags."""
        return self.encode(self.errors(forecast, truth, ship_type))

    # ------------------------------------------------------------------ decoding

    def split(self, probabilities: torch.Tensor) -> dict[str, torch.Tensor]:
        """The code (..., width) as one (..., size) part per component."""
        return dict(zip(self.order, probabilities.split([self.sizes[n] for n in self.order], -1)))

    def probabilities(self, logits: torch.Tensor) -> dict[str, torch.Tensor]:
        """Head logits (..., width) to each component's probabilities."""
        return self.split(torch.exp(codes.grouped_log_softmax(logits, self.runs)))

    def decode(self, logits: torch.Tensor):
        """Head logits (..., width) to ((..., components) means, variances, medians), in the
        code's units."""
        parts = self.probabilities(logits)
        means, variances, medians = [], [], []
        for c in self.components:
            part, inverse = parts[c.name], self.inverse(c.name)
            mean, variance = codes.symlog_decode(part, c.knee, c.bound, c.bins, inverse=inverse)
            means.append(mean)
            variances.append(variance)
            medians.append(
                codes.symlog_median(
                    part, c.knee, c.bound, c.bins, inverse=inverse, window=MEDIAN_WINDOW
                )
            )
        return torch.stack(means, -1), torch.stack(variances, -1), torch.stack(medians, -1)

    def moments(self, logits: torch.Tensor) -> Moments:
        """The decoded distribution per quantity and the advance's step (module docstring)."""
        means, variances, medians = self.decode(logits)
        mean, variance, target, step = {}, {}, {}, {}
        for k, name in enumerate(PLANAR):
            columns = slice(3 * k, 3 * k + 3)
            mean[name] = mean_of(means[..., columns], self.least_squares)
            variance[name] = covariance_of(variances[..., columns], self.variance_solve)
            median = mean_of(medians[..., columns], self.least_squares)
            spread = 0.5 * (variance[name][..., 0] + variance[name][..., 2])  # per dimension
            target[name] = self.rate * confident(median, median.square().sum(-1), spread)
            step[name] = capped_step(target[name], STEP_LIMITS[name])
        for i, name in enumerate(SCALAR_QUANTITIES, start=3 * len(PLANAR)):
            mean[name], variance[name] = means[..., i], variances[..., i]
            if name in MEDIAN:
                target[name] = step[name] = medians[..., i]
            else:
                median = medians[..., i]
                target[name] = self.rate * confident(median, median.square(), variance[name])
                limit = STEP_LIMITS[name]
                step[name] = limit * torch.tanh(target[name] / limit)
        return Moments(mean=mean, variance=variance, target=target, step=step)

    def marginal_logits(self) -> torch.Tensor:
        """(width,) the starting logits: per component its measured MARGINAL (as
        `delta.DeltaCode.marginal_logits`), or the code of a zero error without one."""
        parts = []
        for component in self.components:
            low, high = component.low, component.width - 1 - component.low
            zero = codes.hl_gauss(
                torch.zeros((), dtype=torch.float64), low, high, codes.HL_GAUSS_SIGMA
            )
            if component.name not in MARGINAL:
                parts.append(zero)
                continue
            share, quantiles = MARGINAL[component.name]
            values = torch.tensor(quantiles, dtype=torch.float64)
            coordinate = codes.symlog_coordinate(values, component.knee, component.spacing)
            coordinate = coordinate.clamp(-float(low), float(high))
            padded = torch.cat([coordinate[:1], coordinate, coordinate[-1:]])
            widths = (0.5 * (padded[2:] - padded[:-2])).clamp_min(codes.HL_GAUSS_SIGMA)
            atoms = torch.stack(
                [
                    codes.hl_gauss(c, low, high, float(w))
                    for c, w in zip(coordinate, widths, strict=True)
                ]
            )
            parts.append(share * zero + (1.0 - share) * atoms.mean(0))
        return torch.log(torch.cat(parts).clamp_min(1e-12)).float()

    # ------------------------------------------------------------------ the advance

    def _cooldown_code(self, cooldown: torch.Tensor, error: torch.Tensor, state_code):
        """The cooldown's code at t + 1: the forecast's sharp code (..., cooldown_bins) moved
        by every bin of e's distribution (..., size), clamped to the range, two-hot coded."""
        component = self.components[COMPONENT_NAMES.index("gun_cooldown")]
        index = torch.arange(component.width, dtype=cooldown.dtype, device=cooldown.device)
        steps = codes.symlog_value(index - component.low, component.knee, component.spacing)
        moved = torch.clamp(cooldown[..., None] + steps, 0.0, self.cooldown_limit)  # (..., size)
        sharp = state_code._cooldown_sharp(moved)  # (..., size, bins)
        return (error[..., :, None] * sharp).sum(-2)

    def correct(
        self,
        forecast: torch.Tensor,
        logits: torch.Tensor,
        ship_type: torch.Tensor,
        state_code: ShipStateCode,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The corrected belief at t + 1 (means (..., len(MEANS)), spreads (...,
        spread_count) in the covariance layout) of the physics forecast at t + 1 (means) and
        the error head's logits: the module docstring's step and variance. The covariance is
        block diagonal: position, velocity, heading and nose rate (the error head codes them
        independently); the heading's variance capped at the uniform circle's and the nose
        rate's at a uniform one over the state code's range, as the physics advance's. The
        shield, energy and the countdowns clamped to their ranges; the cooldown's residual
        that of the forecast's sharp code moved by e's distribution."""
        dtype = forecast.dtype
        moments = self.moments(logits)
        mean = dict(zip(MEANS, forecast.unbind(-1), strict=True))
        step = {name: value.to(dtype) for name, value in moments.step.items()}
        variance = {name: moments.applied_variance(name).to(dtype) for name in QUANTITIES}
        max_shield, max_energy = self._maxima(ship_type)
        position = forecast[..., 0:2] + step["position"]
        velocity = forecast[..., 2:4] + step["velocity"]
        scale = dict(shield=max_shield, energy=max_energy)

        def scalar(name: str):
            factor = scale.get(name, 1.0)
            return mean[name] + step[name] * factor, variance[name] * factor * factor

        attack, attack_variance = scalar("angle_of_attack")
        index_value, index_variance = scalar("log_refractive_index")
        energy, energy_variance = scalar("energy")
        shield, shield_variance = scalar("shield")
        wait, wait_variance = scalar("shield_recharge_wait")
        nose_rate, _ = scalar("nose_rate")
        energy = torch.minimum(torch.clamp_min(energy, 0.0), max_energy)
        shield = torch.minimum(torch.clamp_min(shield, 0.0), max_shield)
        wait = torch.clamp(wait, 0.0, self.wait_limit)
        cooldown = torch.clamp(
            mean["gun_cooldown"] + step["gun_cooldown"], 0.0, self.cooldown_limit
        )
        error = self.probabilities(logits)["gun_cooldown"].to(dtype)
        moved = self._cooldown_code(mean["gun_cooldown"], error, state_code)
        residual = moved - state_code._cooldown_sharp(cooldown)

        zero = torch.zeros_like(attack)
        p, v = variance["position"], variance["velocity"]
        covariance = unpack_covariance(
            torch.stack(
                [
                    *(p[..., 0], p[..., 1], zero, zero, zero, zero),
                    *(p[..., 2], zero, zero, zero, zero),
                    *(v[..., 0], v[..., 1], zero, zero),
                    *(v[..., 2], zero, zero),
                    *(variance["heading"], zero),
                    variance["nose_rate"],
                ],
                -1,
            )
        )
        covariance = cap_variance(covariance, 4, UNIFORM_HEADING_VARIANCE)
        covariance = cap_variance(covariance, 5, UNIFORM_NOSE_RATE_VARIANCE)
        diagonal = torch.diagonal(covariance, dim1=-2, dim2=-1).clamp_min(0.0)
        values = dict(
            x=position[..., 0],
            y=position[..., 1],
            velocity_x=velocity[..., 0],
            velocity_y=velocity[..., 1],
            heading=wrap(mean["heading"] + step["heading"]),
            nose_rate=nose_rate,
            angle_of_attack=attack,
            shield_recharge_wait=wait,
            shield=shield,
            energy=energy,
            gun_cooldown=cooldown,
            log_refractive_index=index_value,
        )
        root = lambda value: torch.sqrt(torch.clamp_min(value, 0.0))  # noqa: E731
        spread = dict(
            position=torch.sqrt(0.5 * (diagonal[..., 0] + diagonal[..., 1])),
            velocity_xx=covariance[..., 2, 2],
            velocity_xy=covariance[..., 2, 3],
            velocity_yy=covariance[..., 3, 3],
            heading=torch.sqrt(diagonal[..., 4]),
            nose_rate=torch.sqrt(diagonal[..., 5]),
            angle_of_attack=root(attack_variance),
            shield_recharge_wait=root(wait_variance),
            shield=root(shield_variance),
            energy=root(energy_variance),
            log_refractive_index=root(index_variance),
        )
        return (
            torch.stack([values[name] for name in MEANS], -1),
            torch.cat(
                [
                    torch.stack([spread[name] for name in SPREADS], -1),
                    residual,
                    pack_covariance(covariance),
                ],
                -1,
            ),
        )
