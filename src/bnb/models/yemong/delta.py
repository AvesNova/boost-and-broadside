"""The next-state delta code (`YemongConfig.next_state = "delta"`, `delta_baseline =
"persistence"`; arm-2 brief): each quantity's one-tick change from persistence (nothing
changes), truth(t + 1) - truth(t), as a symlog HL-Gauss code (`codes.symlog_encode`), and
the belief advance that adds a decoded change back, with a full covariance of the
kinematic state.

Position and velocity are coded along and across the velocity at t (across to the left);
below SLOW_SPEED the frame is the heading (counted: `slow`). The heading's change is
wrapped. The countdowns (shield recharge wait, gun cooldown, in s) and the shield are their
plain changes: a countdown's steady -dt and a reset's jump, the shield's regeneration and a
hit's drop. Shield and energy are fractions of the type's maximum. The target is the
truth's change on every row; the advance applies the decoded change to the belief at t,
rotated from the belief's own velocity frame.

The arms to come (round-2 plan section 4) build on this: `delta_baseline = "extrapolation"`
(arm 4) changes `changes` and the advance's baseline only; a separate error head for the
hidden rows (arm 3) adds a head and a mask and keeps this code as the physics head's.
"""

import dataclasses
import math

import torch
import torch.nn as nn

from bnb.game.rules import Rules
from bnb.game.ship_types import ShipTypes, per_ship, ticks_between_shots

from . import codes
from .state_code import (
    COVARIANCE_STATE,
    MEANS,
    SPREADS,
    ShipStateCode,
    StateCodeRanges,
    pack_covariance,
    unpack_covariance,
)

FINE_BINS = 51  # position across and velocity, per component
COARSE_BINS = 31  # the other quantities
SLOW_SPEED = 1.0  # m/s: below it the frame is the heading, not the velocity
DELTA_BASELINES = ("persistence",)  # `YemongConfig.delta_baseline`


@dataclasses.dataclass(frozen=True)
class DeltaComponent:
    """One component of a symlog code: knee, bound (the last real bin's centre above the
    zero), real bins, and the real bins under the zero centre (None: as many as above)."""

    name: str
    knee: float
    bound: float
    bins: int
    below: int | None = None

    @property
    def width(self) -> int:
        """The component's code width: its real bins and the padding on both sides."""
        return self.bins + 2 * codes.PADDING_BINS

    @property
    def spacing(self) -> float:
        """The bin spacing in symlog units."""
        return codes.symlog_spacing(self.knee, self.bound, self.bins, self.below)

    @property
    def low(self) -> int:
        """The bins under the zero centre, padding included."""
        return codes.symlog_sides(self.bins, self.below)[0] + codes.PADDING_BINS

    def encode(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """`codes.symlog_encode` of this component."""
        return codes.symlog_encode(values, self.knee, self.bound, self.bins, below=self.below)


# Knees: the bench's persistence medians (baselines-bench-tables.md, "One tick, pooled",
# rows P; position and velocity in the track frame; where the median is 0 the median above
# 1e-6, the countdowns' one tick). Bounds: the bench's max, rounded up (position along: the
# state-deltas hard bound, 8.7 m; the cooldown: 3 ticks). Position along the track is the
# whole move, about |v| dt (median 2.6 m, never near 0): 155 real bins above the zero and
# 19 below it (to -0.51 m, a slow ship reversing), so a bin is 4.9 cm wide at the median
# (2.5 cm at 0, 10 cm at the bound).
POSITION_ALONG = DeltaComponent("position_along", 2.6, 8.7, 175, below=19)  # m
COMPONENTS = (
    POSITION_ALONG,
    DeltaComponent("position_across", 0.00628, 0.7, FINE_BINS),  # m
    DeltaComponent("velocity_along", 0.134, 35.0, FINE_BINS),  # m/s
    DeltaComponent("velocity_across", 0.471, 41.0, FINE_BINS),
    DeltaComponent("heading", 0.0101, 0.32, COARSE_BINS),  # rad
    DeltaComponent("nose_rate", 0.052, 10.5, COARSE_BINS),  # rad/s
    DeltaComponent("angle_of_attack", 0.00285, 0.17, COARSE_BINS),  # rad
    DeltaComponent("shield_recharge_wait", 1.0 / 30.0, 6.0, COARSE_BINS),  # s
    DeltaComponent("shield", 0.00399, 0.36, COARSE_BINS),  # of the type's maximum
    DeltaComponent("energy", 0.000544, 0.046, COARSE_BINS),  # of the type's maximum
    DeltaComponent("log_refractive_index", 0.00794, 0.24, COARSE_BINS),
    DeltaComponent("gun_cooldown", 1.0 / 30.0, 0.1, COARSE_BINS),  # s
)
CONTINUOUS = tuple(component.name for component in COMPONENTS)
# The quantities that advance by the decoded median (their change is a rare jump: the most
# likely outcome, decisions D8); the others by the mean.
MEDIAN = ("shield_recharge_wait", "shield", "gun_cooldown")
# The groups of each of the state code's parts, in `head_statistics.STATE_PARTS` order.
PART_GROUPS = (2, 2, 1, 1, 1, 1, 1, 1, 1, 1)

# The starting marginal (`marginal_logits`), measured by `bench/persistence_marginal.py` on
# scripted self-play of this tree (frontline, deathmatch and wide deathmatch, 3 x 1024
# ticks x 128 games each, 10.7M living pairs): per component the share of changes exactly
# 0 and 16 quantiles of the rest (at (k + 1/2) / 16).
MARGINAL = {
    "position_along": (
        0.0,
        (
            1.084,
            1.447,
            1.598,
            1.715,
            1.881,
            2.022,
            2.209,
            2.423,
            2.644,
            2.859,
            3.041,
            3.188,
            3.307,
            3.415,
            3.539,
            3.814,
        ),
    ),
    "position_across": (
        0.0,
        (
            -0.05217,
            -0.03324,
            -0.01941,
            -0.008983,
            -0.00511,
            -0.002632,
            -0.0009532,
            -3.989e-05,
            4.359e-05,
            0.0009713,
            0.002652,
            0.00515,
            0.009196,
            0.0205,
            0.03371,
            0.05308,
        ),
    ),
    "velocity_along": (
        0.0,
        (
            -0.8614,
            -0.4944,
            -0.2288,
            -0.1058,
            -0.04588,
            -0.01076,
            0.008838,
            0.02649,
            0.0469,
            0.07211,
            0.1054,
            0.1495,
            0.205,
            0.2714,
            0.3417,
            0.9909,
        ),
    ),
    "velocity_across": (
        0.0,
        (
            -3.213,
            -2.02,
            -1.265,
            -0.6615,
            -0.3947,
            -0.2054,
            -0.07036,
            -0.0006831,
            0.0008947,
            0.07159,
            0.2075,
            0.3984,
            0.6787,
            1.321,
            2.048,
            3.27,
        ),
    ),
    "heading": (
        0.0295,
        (
            -0.05126,
            -0.04031,
            -0.02583,
            -0.01479,
            -0.009393,
            -0.004662,
            -0.001522,
            -3.82e-05,
            4.401e-05,
            0.001577,
            0.004744,
            0.009521,
            0.01495,
            0.02668,
            0.04068,
            0.05178,
        ),
    ),
    "nose_rate": (
        0.0032,
        (
            -0.9035,
            -0.466,
            -0.2725,
            -0.1273,
            -0.04918,
            -0.01535,
            -0.003787,
            -5.412e-05,
            3.703e-05,
            0.003656,
            0.01525,
            0.04941,
            0.1276,
            0.2727,
            0.4659,
            0.9027,
        ),
    ),
    "angle_of_attack": (
        0.0032,
        (
            -0.02197,
            -0.01197,
            -0.007586,
            -0.005077,
            -0.002548,
            -0.001086,
            -0.0003291,
            -7.738e-06,
            8.008e-06,
            0.0003319,
            0.001098,
            0.002566,
            0.005077,
            0.007589,
            0.01195,
            0.022,
        ),
    ),
    "shield_recharge_wait": (
        0.7988,
        (
            -0.03333,
            -0.03333,
            -0.03333,
            -0.03333,
            -0.03333,
            -0.03333,
            -0.03333,
            -0.03333,
            -0.03333,
            -0.03333,
            -0.03333,
            -0.03333,
            -0.03333,
            -0.03333,
            -0.03333,
            0.03333,
        ),
    ),
    "shield": (
        0.9389,
        (
            -0.09827,
            -0.04811,
            -0.03602,
            -0.02261,
            0.0025,
            0.0025,
            0.0025,
            0.0025,
            0.0025,
            0.0025,
            0.0025,
            0.004,
            0.004,
            0.004,
            0.004,
            0.004,
        ),
    ),
    "energy": (
        0.0,
        (
            -0.006932,
            -0.0003094,
            -0.0001872,
            -0.0001329,
            -7.428e-05,
            0.0003225,
            0.0004406,
            0.0005213,
            0.0005804,
            0.000632,
            0.0006972,
            0.0007574,
            0.0008268,
            0.0009216,
            0.001124,
            0.004249,
        ),
    ),
    "log_refractive_index": (
        0.9652,
        (
            -0.05022,
            -0.02651,
            -0.01508,
            -0.00866,
            -0.00483,
            -0.002338,
            -0.000739,
            -8.702e-06,
            0.0001038,
            0.0008725,
            0.002337,
            0.005017,
            0.009944,
            0.01845,
            0.03171,
            0.0577,
        ),
    ),
    "gun_cooldown": (
        0.9588,
        (
            -0.03333,
            -0.03333,
            -0.03333,
            -0.03333,
            -0.03333,
            -0.03333,
            -0.03333,
            -0.03333,
            -0.03333,
            0.03333,
            0.03333,
            0.03333,
            0.03333,
            0.03333,
            0.06667,
            0.06667,
        ),
    ),
}


@dataclasses.dataclass(frozen=True)
class StateDynamics:
    """What a rule baseline reads from the rules and the ship card, per type
    (`state_dynamics`): shield_recharge_rate (per s), shield_recharge_delay (s),
    ticks_between_shots. Persistence reads none of it, and its builds record None; the
    build record keeps the field for the extrapolation baseline to come (D2)."""

    shield_recharge_rate: tuple[float, ...]
    shield_recharge_delay: tuple[float, ...]
    ticks_between_shots: tuple[int, ...]


def state_dynamics(rules: Rules, types: ShipTypes) -> StateDynamics:
    """The dynamics of the rules and a (types,) card."""
    return StateDynamics(
        shield_recharge_rate=tuple(
            (rules.shield_recharge_rate * types.shield_recharge_rate_scale).tolist()
        ),
        shield_recharge_delay=tuple(
            (rules.shield_recharge_delay * types.shield_recharge_delay_scale).tolist()
        ),
        ticks_between_shots=tuple(
            int(ticks) for ticks in ticks_between_shots(types.fire_rate, rules.tick_length)
        ),
    )


def wrap(angle: torch.Tensor) -> torch.Tensor:
    """angle in [-pi, pi)."""
    return torch.remainder(angle + math.pi, 2.0 * math.pi) - math.pi


def velocity_frame(
    velocity: torch.Tensor, heading: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(unit_x, unit_y, slow), each (...): the direction of velocity (..., 2), or of the
    heading (...) where the speed is below SLOW_SPEED (slow)."""
    speed = torch.linalg.vector_norm(velocity, dim=-1)
    slow = speed < SLOW_SPEED
    safe = torch.where(slow, 1.0, speed)
    unit_x = torch.where(slow, torch.cos(heading), velocity[..., 0] / safe)
    unit_y = torch.where(slow, torch.sin(heading), velocity[..., 1] / safe)
    return unit_x, unit_y, slow


def to_frame(vector: torch.Tensor, unit_x: torch.Tensor, unit_y: torch.Tensor) -> torch.Tensor:
    """(..., 2) world to (along, across) the unit direction (across: to the left)."""
    x, y = vector[..., 0], vector[..., 1]
    return torch.stack([x * unit_x + y * unit_y, y * unit_x - x * unit_y], -1)


def from_frame(vector: torch.Tensor, unit_x: torch.Tensor, unit_y: torch.Tensor) -> torch.Tensor:
    """The inverse of `to_frame`."""
    along, across = vector[..., 0], vector[..., 1]
    return torch.stack([along * unit_x - across * unit_y, along * unit_y + across * unit_x], -1)


def rotate_variances(
    along: torch.Tensor, across: torch.Tensor, unit_x: torch.Tensor, unit_y: torch.Tensor
) -> torch.Tensor:
    """(..., 3) the packed (xx, xy, yy) world covariance of independent along and across
    variances (...) in the frame of the unit direction."""
    return torch.stack(
        [
            along * unit_x * unit_x + across * unit_y * unit_y,
            (along - across) * unit_x * unit_y,
            along * unit_y * unit_y + across * unit_x * unit_x,
        ],
        -1,
    )


def kinematics_jacobian(tick: float, dtype, device) -> torch.Tensor:
    """(6, 6) F of COVARIANCE_STATE over one tick: position += velocity dt, heading +=
    nose rate dt."""
    jacobian = torch.eye(len(COVARIANCE_STATE), dtype=dtype, device=device)
    jacobian[0, 2] = jacobian[1, 3] = jacobian[4, 5] = tick
    return jacobian


class DeltaCode(nn.Module):
    """The delta code of a Yemong build: `targets` (the training code of a pair of states),
    `decode` (head logits to the change's moments), `advance` (the belief at t to the
    forecast at t + 1). Width 624: 179 + 3 x 55 + 8 x 35."""

    def __init__(self, ranges: StateCodeRanges) -> None:
        super().__init__()
        self.tick_length = ranges.tick_length
        tensor = lambda values: torch.tensor(values, dtype=torch.float32)  # noqa: E731
        self.register_buffer("max_shield", tensor(ranges.max_shield))
        self.register_buffer("max_energy", tensor(ranges.max_energy))
        self.wait_limit = ranges.shield_recharge_delay
        self.cooldown_bins = ranges.cooldown_bins
        self.cooldown_limit = (ranges.cooldown_bins - 1) * ranges.tick_length
        self.components = {component.name: component for component in COMPONENTS}
        self.sizes = {component.name: component.width for component in COMPONENTS}
        self.runs: codes.Runs = tuple((1, component.width) for component in COMPONENTS)
        self.order = CONTINUOUS
        self._inverse_ends = {}
        for component in COMPONENTS:
            table, lowest, highest = codes.symlog_mean_inverse(
                component.bins, below=component.below
            )
            self.register_buffer(f"inverse_{component.name}", table.float(), persistent=False)
            self._inverse_ends[component.name] = (lowest, highest)
        # The starting logits, kept for the diagnostics' marginal baseline.
        self.register_buffer("marginal", self.marginal_logits(), persistent=False)

    @property
    def width(self) -> int:
        """The code's width: groups times size, summed over runs."""
        return sum(groups * size for groups, size in self.runs)

    @property
    def groups(self) -> int:
        """The code's softmax groups, summed over runs."""
        return sum(groups for groups, _ in self.runs)

    def inverse(self, name: str) -> tuple[torch.Tensor, float, float]:
        """A component's `codes.symlog_mean_inverse` table and ends."""
        return (getattr(self, f"inverse_{name}"), *self._inverse_ends[name])

    def _maxima(self, ship_type: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return per_ship(self.max_shield, ship_type), per_ship(self.max_energy, ship_type)

    # ------------------------------------------------------------------ targets

    def changes(
        self, means: torch.Tensor, following: torch.Tensor, ship_type: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """((..., len(CONTINUOUS)) the changes from means to following (..., len(MEANS)) in
        the code's frames and units; (...) slow: the frame was the heading)."""
        index = {name: i for i, name in enumerate(MEANS)}
        max_shield, max_energy = self._maxima(ship_type)
        difference = following - means
        unit_x, unit_y, slow = velocity_frame(means[..., 2:4], means[..., index["heading"]])
        position = to_frame(difference[..., 0:2], unit_x, unit_y)
        velocity = to_frame(difference[..., 2:4], unit_x, unit_y)
        column = lambda name: difference[..., index[name]]  # noqa: E731
        continuous = torch.stack(
            [
                position[..., 0],
                position[..., 1],
                velocity[..., 0],
                velocity[..., 1],
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
        )
        return continuous, slow

    def encode(self, continuous: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """(the code (..., width), (..., len(CONTINUOUS)) bool: the changes clamped to the
        padded range) of changes as `changes` gives them."""
        parts, clipped = [], []
        for i, component in enumerate(COMPONENTS):
            code, clip = component.encode(continuous[..., i])
            parts.append(code)
            clipped.append(clip)
        return torch.cat(parts, -1), torch.stack(clipped, -1)

    def targets(
        self, means: torch.Tensor, following: torch.Tensor, ship_type: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The training code of the step from means to following, the clipped flags and
        the slow-frame flags."""
        continuous, slow = self.changes(means, following, ship_type)
        code, clipped = self.encode(continuous)
        return code, clipped, slow

    # ------------------------------------------------------------------ decoding

    def split(self, probabilities: torch.Tensor) -> dict[str, torch.Tensor]:
        """The code (..., width) as one (..., size) part per component."""
        return dict(zip(self.order, probabilities.split([self.sizes[n] for n in self.order], -1)))

    def probabilities(self, logits: torch.Tensor) -> dict[str, torch.Tensor]:
        """Head logits (..., width) to each component's probabilities."""
        return self.split(torch.exp(codes.grouped_log_softmax(logits, self.runs)))

    def decode(self, logits: torch.Tensor):
        """Head logits (..., width) to ((..., len(CONTINUOUS)) the changes' means,
        (..., len(CONTINUOUS)) their variances, (..., len(MEDIAN)) the medians of MEDIAN's
        quantities), in the code's frames and units."""
        parts = self.probabilities(logits)
        means, variances, medians = [], [], []
        for c in COMPONENTS:
            part, inverse = parts[c.name], self.inverse(c.name)
            mean, variance = codes.symlog_decode(
                part, c.knee, c.bound, c.bins, below=c.below, inverse=inverse
            )
            means.append(mean)
            variances.append(variance)
            if c.name in MEDIAN:
                medians.append(
                    codes.symlog_median(
                        part, c.knee, c.bound, c.bins, below=c.below, inverse=inverse
                    )
                )
        return torch.stack(means, -1), torch.stack(variances, -1), torch.stack(medians, -1)

    @staticmethod
    def point(means: torch.Tensor, medians: torch.Tensor) -> torch.Tensor:
        """(..., len(CONTINUOUS)) the change the advance applies, of `decode`'s means and
        medians: MEDIAN's quantities' medians, the others' means."""
        columns = list(means.unbind(-1))
        for i, name in enumerate(MEDIAN):
            columns[CONTINUOUS.index(name)] = medians[..., i]
        return torch.stack(columns, -1)

    def padding_mass(self, probabilities: torch.Tensor) -> torch.Tensor:
        """(..., len(CONTINUOUS)) each component's mass in its padding bins, of probabilities
        (..., width) in the code's layout."""
        parts, pad = self.split(probabilities), codes.PADDING_BINS
        return torch.stack(
            [parts[n][..., :pad].sum(-1) + parts[n][..., -pad:].sum(-1) for n in self.order], -1
        )

    def marginal_logits(self) -> torch.Tensor:
        """(width,) the starting logits: per component (`MARGINAL`) the code of the zero share
        and of each measured quantile, an equal atom whose Gaussian is as wide as its gaps
        to the neighbouring quantiles (in bins; at least the code's own HL_GAUSS_SIGMA), so
        the atoms join into a smooth density where the quantiles spread and stay sharp
        where they coincide (a countdown's -dt)."""
        parts = []
        for component in COMPONENTS:
            share, quantiles = MARGINAL[component.name]
            values = torch.tensor(quantiles, dtype=torch.float64)
            coordinate = codes.symlog_coordinate(values, component.knee, component.spacing)
            low, high = component.low, component.width - 1 - component.low
            coordinate = coordinate.clamp(-float(low), float(high))
            padded = torch.cat([coordinate[:1], coordinate, coordinate[-1:]])
            widths = (0.5 * (padded[2:] - padded[:-2])).clamp_min(codes.HL_GAUSS_SIGMA)
            atoms = torch.stack(
                [
                    codes.hl_gauss(c, low, high, float(w))
                    for c, w in zip(coordinate, widths, strict=True)
                ]
            )
            zero = codes.hl_gauss(
                torch.zeros((), dtype=torch.float64), low, high, codes.HL_GAUSS_SIGMA
            )
            parts.append(share * zero + (1.0 - share) * atoms.mean(0))
        return torch.log(torch.cat(parts).clamp_min(1e-12)).float()

    # ------------------------------------------------------------------ the advance

    def _cooldown_code(self, code: torch.Tensor, change: torch.Tensor, state_code) -> torch.Tensor:
        """The cooldown's code at t + 1: the belief's code (..., cooldown_bins), each of its
        bins moved by every bin of the change's distribution (..., size), clamped to the
        cooldown's range and two-hot coded."""
        component = self.components["gun_cooldown"]
        index = torch.arange(component.width, dtype=code.dtype, device=code.device)
        steps = codes.symlog_value(index - component.low, component.knee, component.spacing)
        centres = self.tick_length * torch.arange(
            self.cooldown_bins, dtype=code.dtype, device=code.device
        )
        moved = torch.clamp(centres[:, None] + steps, 0.0, self.cooldown_limit)  # (bins, size)
        sharp = state_code._cooldown_sharp(moved)  # (bins, size, bins)
        weight = code[..., :, None] * change[..., None, :]  # (..., bins, size)
        return (weight[..., None] * sharp).sum((-3, -2))

    def advance(
        self,
        means: torch.Tensor,
        spreads: torch.Tensor,
        logits: torch.Tensor,
        ship_type: torch.Tensor,
        state_code: ShipStateCode,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The forecast at t + 1 (means (..., len(MEANS)), spreads (..., spread_count) in the
        covariance layout) of the belief at t and the head's logits.

        Means: the belief plus the decoded change (`point`), position and velocity rotated
        from the belief's velocity frame (its heading below SLOW_SPEED) to world; shield,
        energy and the countdowns clamped to their ranges. Spread: the covariance P of
        COVARIANCE_STATE as F P F^T + Q (`kinematics_jacobian`; Q the changes' variances,
        position and velocity rotated to world); the other sigmas the root of the belief's
        variance plus the change's; the cooldown's code the belief's moved by the change's
        distribution (its residual against the advanced mean). SPREADS' kinematic slots
        are read off the new P (position: the per-axis RMS sigma)."""
        tick, dtype = self.tick_length, means.dtype
        decoded, variance, medians = self.decode(logits)
        change, variance = self.point(decoded, medians).to(dtype), variance.to(dtype)
        part = {name: i for i, name in enumerate(CONTINUOUS)}
        mean = dict(zip(MEANS, means.unbind(-1), strict=True))
        sigma = dict(zip(SPREADS, spreads[..., : len(SPREADS)].unbind(-1), strict=True))
        max_shield, max_energy = self._maxima(ship_type)
        unit_x, unit_y, _ = velocity_frame(means[..., 2:4], mean["heading"])

        position = means[..., 0:2] + from_frame(change[..., 0:2], unit_x, unit_y)
        velocity = means[..., 2:4] + from_frame(change[..., 2:4], unit_x, unit_y)

        covariance = unpack_covariance(spreads[..., state_code.covariance_slice])
        jacobian = kinematics_jacobian(tick, dtype, means.device)
        position_noise = rotate_variances(variance[..., 0], variance[..., 1], unit_x, unit_y)
        velocity_noise = rotate_variances(variance[..., 2], variance[..., 3], unit_x, unit_y)
        zero = torch.zeros_like(variance[..., 0])
        noise = unpack_covariance(
            torch.stack(
                [
                    *(position_noise[..., 0], position_noise[..., 1], zero, zero, zero, zero),
                    *(position_noise[..., 2], zero, zero, zero, zero),
                    *(velocity_noise[..., 0], velocity_noise[..., 1], zero, zero),
                    *(velocity_noise[..., 2], zero, zero),
                    *(variance[..., part["heading"]], zero),
                    variance[..., part["nose_rate"]],
                ],
                -1,
            )
        )
        covariance = jacobian @ covariance @ jacobian.transpose(-1, -2) + noise

        def scalar(name: str, scale=1.0):
            value = mean[name] + change[..., part[name]] * scale
            spread = torch.sqrt(sigma[name] ** 2 + variance[..., part[name]] * scale**2)
            return value, spread

        attack, attack_spread = scalar("angle_of_attack")
        index_value, index_spread = scalar("log_refractive_index")
        energy, energy_spread = scalar("energy", max_energy)
        shield, shield_spread = scalar("shield", max_shield)
        wait, wait_spread = scalar("shield_recharge_wait")
        energy = torch.minimum(torch.clamp_min(energy, 0.0), max_energy)
        shield = torch.minimum(torch.clamp_min(shield, 0.0), max_shield)
        wait = torch.clamp(wait, 0.0, self.wait_limit)

        held = torch.clamp_min(
            state_code._cooldown_sharp(mean["gun_cooldown"])
            + spreads[..., state_code.residual_slice],
            0.0,
        )
        cooldown_change = self.probabilities(logits)["gun_cooldown"].to(dtype)
        cooldown_code = self._cooldown_code(held, cooldown_change, state_code)
        cooldown = torch.clamp(
            mean["gun_cooldown"] + change[..., part["gun_cooldown"]], 0.0, self.cooldown_limit
        )
        residual = cooldown_code - state_code._cooldown_sharp(cooldown)

        values = dict(
            x=position[..., 0],
            y=position[..., 1],
            velocity_x=velocity[..., 0],
            velocity_y=velocity[..., 1],
            heading=wrap(mean["heading"] + change[..., part["heading"]]),
            nose_rate=mean["nose_rate"] + change[..., part["nose_rate"]],
            angle_of_attack=attack,
            shield_recharge_wait=wait,
            shield=shield,
            energy=energy,
            gun_cooldown=cooldown,
            log_refractive_index=index_value,
        )
        diagonal = torch.diagonal(covariance, dim1=-2, dim2=-1).clamp_min(0.0)
        spread = dict(
            position=torch.sqrt(0.5 * (diagonal[..., 0] + diagonal[..., 1])),
            velocity_xx=covariance[..., 2, 2],
            velocity_xy=covariance[..., 2, 3],
            velocity_yy=covariance[..., 3, 3],
            heading=torch.sqrt(diagonal[..., 4]),
            nose_rate=torch.sqrt(diagonal[..., 5]),
            angle_of_attack=attack_spread,
            shield_recharge_wait=wait_spread,
            shield=shield_spread,
            energy=energy_spread,
            log_refractive_index=index_spread,
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
