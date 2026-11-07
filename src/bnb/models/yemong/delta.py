"""The next-state delta code (`YemongConfig.next_state = "delta"`; state-deltas report,
section 4): each quantity's one-tick change from its physics baseline, as a symlog HL-Gauss
code (`codes.symlog_encode`), and the belief advance that adds a decoded change back.

The baselines read only the state at t (state_code.MEANS order, physical units):
- position: the constant-nose-rate arc, x + [[A1, -A2], [A2, A1]] v with the game's
  series A1 = (1 - a^2 (1/6 - a^2 / 120)) dt and A2 = a (1/2 - a^2 (1/24 - a^2 / 720)) dt,
  a = omega dt (the game's drift, `bnb.game.physics.flight.flight_tick`, at v_t);
- velocity: v rotated by omega dt;
- heading: heading + omega dt, wrapped;
- nose rate, angle of attack, energy and the log refractive index: persistence;
- shield: after the deterministic recharge, min(max, s + rate clamp(dt - w, 0, dt));
- shield recharge wait and gun cooldown: the countdown, max(w - dt, 0) and max(c - dt, 0).
  Their change is an event, two bins: the wait reset to the type's delay (any damage), the
  gun fired (the cooldown set to the type's interval less the tick).

Position and velocity are coded along and across the heading at t (across: to the left);
shield and energy as fractions of the type's maximum. The target is the truth's change,
truth(t + 1) - baseline(truth(t)), on every row; the belief advance applies the decoded
change to the baseline of the belief at t.
"""

import dataclasses
import math

import torch
import torch.nn as nn

from bnb.game.rules import Rules
from bnb.game.ship_types import ShipTypes, per_ship, ticks_between_shots

from . import codes
from .state_code import MEANS, SPREADS, ShipStateCode, StateCodeRanges

FINE_BINS = 51  # class A: position and velocity, per component
COARSE_BINS = 31  # class B: the other continuous quantities
EVENT_BINS = 2


@dataclasses.dataclass(frozen=True)
class DeltaComponent:
    """One continuous component of the change: knee (the median one-tick change), bound
    (the hard one-tick bound, the last real bin's centre), real bins (odd)."""

    name: str
    knee: float
    bound: float
    bins: int

    @property
    def width(self) -> int:
        """The component's code width: its real bins and the padding on both sides."""
        return self.bins + 2 * codes.PADDING_BINS


# The report's knees (section 4) and hard bounds (section 1.2, the box column).
COMPONENTS = (
    DeltaComponent("position_along", 0.002, 8.7, FINE_BINS),  # m
    DeltaComponent("position_across", 0.003, 8.7, FINE_BINS),
    DeltaComponent("velocity_along", 0.1, 375.0, FINE_BINS),  # m/s
    DeltaComponent("velocity_across", 0.1, 375.0, FINE_BINS),
    DeltaComponent("heading", 0.001, 0.9, COARSE_BINS),  # rad
    DeltaComponent("nose_rate", 0.05, 24.7, COARSE_BINS),  # rad/s
    DeltaComponent("angle_of_attack", 0.003, 0.49, COARSE_BINS),  # rad
    DeltaComponent("shield", 0.001, 1.0, COARSE_BINS),  # of the type's maximum
    DeltaComponent("energy", 0.0005, 0.068, COARSE_BINS),  # of the type's maximum
    DeltaComponent("log_refractive_index", 0.0006, 0.48, COARSE_BINS),
)
CONTINUOUS = tuple(component.name for component in COMPONENTS)
EVENTS = ("shield_recharge_wait", "gun_cooldown")  # reset to the delay; fired
# The code's parts, in the state code's order (`head_statistics.STATE_PARTS`'s names): the
# components each part holds (events by name).
PARTS = (
    ("position", ("position_along", "position_across")),
    ("velocity", ("velocity_along", "velocity_across")),
    ("heading", ("heading",)),
    ("nose_rate", ("nose_rate",)),
    ("angle_of_attack", ("angle_of_attack",)),
    ("shield_recharge_wait", ("shield_recharge_wait",)),
    ("shield", ("shield",)),
    ("energy", ("energy",)),
    ("log_refractive_index", ("log_refractive_index",)),
    ("gun_cooldown", ("gun_cooldown",)),
)
# The share of one-tick changes exactly zero (report 2.1 and the "at 0" column of 4), and
# of the events (report 4): the starting marginal.
ZERO_SHARE = dict(
    position_along=0.0,
    position_across=0.0,
    velocity_along=0.0,
    velocity_across=0.0,
    heading=0.003,
    nose_rate=0.004,
    angle_of_attack=0.004,
    shield=0.935,
    energy=0.006,
    log_refractive_index=0.965,
)
EVENT_RATE = dict(shield_recharge_wait=0.020, gun_cooldown=0.017)
# The marginal's nonzero part: a Laplace in symlog units of scale 1 (its median |change| is
# the knee), discretised by its CDF over the bins.
MARGINAL_SCALE = 1.0


@dataclasses.dataclass(frozen=True)
class StateDynamics:
    """What the baselines read from the rules and the ship card, per type (`state_dynamics`),
    so a built model never reads the current game: shield_recharge_rate (per s),
    shield_recharge_delay (s), ticks_between_shots."""

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


def _arc(angle: torch.Tensor, tick: float) -> tuple[torch.Tensor, torch.Tensor]:
    """The drift's (A1, A2): the game's series in the turn angle."""
    square = angle * angle
    along = (1.0 - square * (1.0 / 6.0 - square / 120.0)) * tick
    across = angle * (0.5 - square * (1.0 / 24.0 - square / 720.0)) * tick
    return along, across


def to_nose(vector: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    """(..., 2) world to (along, across) the heading (...)."""
    cos, sin = torch.cos(heading), torch.sin(heading)
    x, y = vector[..., 0], vector[..., 1]
    return torch.stack([x * cos + y * sin, y * cos - x * sin], -1)


def from_nose(vector: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    """The inverse of `to_nose`."""
    cos, sin = torch.cos(heading), torch.sin(heading)
    along, across = vector[..., 0], vector[..., 1]
    return torch.stack([along * cos - across * sin, along * sin + across * cos], -1)


class DeltaCode(nn.Module):
    """The delta code of a Yemong build: `targets` (the training code of a pair of states),
    `decode` (head logits to the change's moments) and `advance` (the belief at t to the
    forecast at t + 1). Width 434 at the default bins: 2 x 2 x 55 + 6 x 35 + 2 x 2."""

    def __init__(self, ranges: StateCodeRanges, dynamics: StateDynamics) -> None:
        super().__init__()
        self.tick_length = ranges.tick_length
        tensor = lambda values: torch.tensor(values, dtype=torch.float32)  # noqa: E731
        self.register_buffer("max_shield", tensor(ranges.max_shield))
        self.register_buffer("max_energy", tensor(ranges.max_energy))
        self.register_buffer("recharge_rate", tensor(dynamics.shield_recharge_rate))
        self.register_buffer("recharge_delay", tensor(dynamics.shield_recharge_delay))
        # The cooldown one tick after a shot (s): the interval less the tick's clock.
        self.register_buffer(
            "fired_cooldown",
            tensor([(ticks - 1) * ranges.tick_length for ticks in dynamics.ticks_between_shots]),
        )
        self.components = {component.name: component for component in COMPONENTS}
        self.sizes = {
            **{component.name: component.width for component in COMPONENTS},
            **dict.fromkeys(EVENTS, EVENT_BINS),
        }
        self.runs: codes.Runs = tuple((len(names), self.sizes[names[0]]) for _, names in PARTS)
        self.order = tuple(name for _, names in PARTS for name in names)
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

    # ------------------------------------------------------------------ baselines

    def _per_type(self, ship_type: torch.Tensor) -> dict[str, torch.Tensor]:
        return dict(
            max_shield=per_ship(self.max_shield, ship_type),
            max_energy=per_ship(self.max_energy, ship_type),
            rate=per_ship(self.recharge_rate, ship_type),
            delay=per_ship(self.recharge_delay, ship_type),
            fired_cooldown=per_ship(self.fired_cooldown, ship_type),
        )

    def baseline(self, means: torch.Tensor, ship_type: torch.Tensor) -> torch.Tensor:
        """(..., len(MEANS)) the baseline of the state at t (the module's docstring)."""
        tick, kind = self.tick_length, self._per_type(ship_type)
        mean = dict(zip(MEANS, means.unbind(-1), strict=True))
        angle = mean["nose_rate"] * tick
        along, across = _arc(angle, tick)
        vx, vy = mean["velocity_x"], mean["velocity_y"]
        cos, sin = torch.cos(angle), torch.sin(angle)
        shield = torch.minimum(
            kind["max_shield"],
            mean["shield"]
            + kind["rate"] * torch.clamp(tick - mean["shield_recharge_wait"], 0.0, tick),
        )
        values = dict(
            mean,
            x=mean["x"] + vx * along - vy * across,
            y=mean["y"] + vx * across + vy * along,
            velocity_x=vx * cos - vy * sin,
            velocity_y=vx * sin + vy * cos,
            heading=wrap(mean["heading"] + angle),
            shield=shield,
            shield_recharge_wait=torch.clamp_min(mean["shield_recharge_wait"] - tick, 0.0),
            gun_cooldown=torch.clamp_min(mean["gun_cooldown"] - tick, 0.0),
        )
        return torch.stack([values[name] for name in MEANS], -1)

    def changes(
        self, means: torch.Tensor, following: torch.Tensor, ship_type: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """((..., len(CONTINUOUS)) the continuous changes of following (..., len(MEANS)) from
        the baseline of means, in the code's frames and units; (..., len(EVENTS)) bool, the
        events)."""
        base, kind = self.baseline(means, ship_type), self._per_type(ship_type)
        index = {name: i for i, name in enumerate(MEANS)}
        difference = following - base
        heading = means[..., index["heading"]]
        position = to_nose(difference[..., 0:2], heading)
        velocity = to_nose(difference[..., 2:4], heading)
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
                column("shield") / kind["max_shield"],
                column("energy") / kind["max_energy"],
                column("log_refractive_index"),
            ],
            -1,
        )
        half = 0.5 * self.tick_length
        events = torch.stack(
            [column("shield_recharge_wait") > half, column("gun_cooldown") > half], -1
        )
        return continuous, events

    # ------------------------------------------------------------------ the code

    def encode(
        self, continuous: torch.Tensor, events: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """(the code (..., width), (..., len(CONTINUOUS)) bool: the changes clamped to the
        padded range) of changes and events as `changes` gives them."""
        parts, clipped = {}, []
        for i, component in enumerate(COMPONENTS):
            code, clip = codes.symlog_encode(
                continuous[..., i], component.knee, component.bound, component.bins
            )
            parts[component.name] = code
            clipped.append(clip)
        for i, name in enumerate(EVENTS):
            happened = events[..., i].to(continuous.dtype)
            parts[name] = torch.stack([1.0 - happened, happened], -1)
        return torch.cat([parts[name] for name in self.order], -1), torch.stack(clipped, -1)

    def targets(
        self, means: torch.Tensor, following: torch.Tensor, ship_type: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The training code of the step from means to following, and the clipped flags."""
        return self.encode(*self.changes(means, following, ship_type))

    def _split(self, probabilities: torch.Tensor) -> dict[str, torch.Tensor]:
        return dict(zip(self.order, probabilities.split([self.sizes[n] for n in self.order], -1)))

    def decode(self, logits: torch.Tensor):
        """Head logits (..., width) to ((..., len(CONTINUOUS)) the changes' means,
        (..., len(CONTINUOUS)) their variances, (..., len(EVENTS)) the events'
        probabilities), in the code's frames and units."""
        parts = self._split(torch.exp(codes.grouped_log_softmax(logits, self.runs)))
        means, variances = [], []
        for component in COMPONENTS:
            mean, variance = codes.symlog_decode(
                parts[component.name], component.knee, component.bound, component.bins
            )
            means.append(mean)
            variances.append(variance)
        events = torch.stack([parts[name][..., 1] for name in EVENTS], -1)
        return torch.stack(means, -1), torch.stack(variances, -1), events

    def padding_mass(self, probabilities: torch.Tensor) -> torch.Tensor:
        """(..., len(CONTINUOUS)) each component's mass in its padding bins, of probabilities
        (..., width) in the code's layout."""
        parts, pad = self._split(probabilities), codes.PADDING_BINS
        return torch.stack(
            [
                parts[c.name][..., :pad].sum(-1) + parts[c.name][..., -pad:].sum(-1)
                for c in COMPONENTS
            ],
            -1,
        )

    def marginal_logits(self) -> torch.Tensor:
        """(width,) the starting logits: per component the zero share at the exact-zero
        centre (its HL-Gauss code) and the rest a symlog Laplace of scale MARGINAL_SCALE;
        per event its rate (`ZERO_SHARE`, `EVENT_RATE`)."""
        parts = {}
        root = math.sqrt(2.0)
        for component in COMPONENTS:
            half = (component.bins - 1) // 2 + codes.PADDING_BINS
            edges = torch.arange(-half, half + 2, dtype=torch.float64) - 0.5
            spacing = codes.symlog_spacing(component.knee, component.bound, component.bins)
            scale = MARGINAL_SCALE / spacing  # in bins
            laplace = torch.where(
                edges < 0,
                0.5 * torch.exp(edges / scale),
                1.0 - 0.5 * torch.exp(-edges / scale),
            )
            spread = laplace[1:] - laplace[:-1]
            spread = spread / spread.sum()
            cumulative = torch.erf(edges / (codes.HL_GAUSS_SIGMA * root))
            zero = 0.5 * (cumulative[1:] - cumulative[:-1])
            zero = zero / zero.sum()
            share = ZERO_SHARE[component.name]
            parts[component.name] = share * zero + (1.0 - share) * spread
        for name in EVENTS:
            rate = EVENT_RATE[name]
            parts[name] = torch.tensor([1.0 - rate, rate], dtype=torch.float64)
        return torch.log(torch.cat([parts[name] for name in self.order])).float()

    # ------------------------------------------------------------------ the advance

    def advance(
        self,
        means: torch.Tensor,
        spreads: torch.Tensor,
        logits: torch.Tensor,
        ship_type: torch.Tensor,
        state_code: ShipStateCode,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The forecast at t + 1 (means (..., len(MEANS)), spreads (..., spread_count)) of
        the belief at t and the head's logits: the baseline plus the decoded change (rotated
        back from the belief's heading), each variance the belief's carried through the
        baseline (to first order) plus the change's; the events' expected values."""
        tick, kind = self.tick_length, self._per_type(ship_type)
        change, variance, events = self.decode(logits)
        change, variance = change.to(means.dtype), variance.to(means.dtype)
        base = self.baseline(means, ship_type)
        index = {name: i for i, name in enumerate(MEANS)}
        mean = dict(zip(MEANS, means.unbind(-1), strict=True))
        sigma = dict(zip(SPREADS, spreads[..., : len(SPREADS)].unbind(-1), strict=True))
        part = {name: i for i, name in enumerate(CONTINUOUS)}
        heading = mean["heading"]
        position = base[..., 0:2] + from_nose(change[..., 0:2], heading)
        velocity = base[..., 2:4] + from_nose(change[..., 2:4], heading)

        # Carried variances. The nose rate's turns the arc and the velocity by dt omega.
        nose_variance = sigma["nose_rate"] ** 2
        xx, xy, yy = sigma["velocity_xx"], sigma["velocity_xy"], sigma["velocity_yy"]
        angle = mean["nose_rate"] * tick
        cos, sin = torch.cos(angle), torch.sin(angle)
        rotated = torch.stack(
            [
                cos * cos * xx - 2.0 * cos * sin * xy + sin * sin * yy,
                cos * sin * (xx - yy) + (cos * cos - sin * sin) * xy,
                sin * sin * xx + 2.0 * cos * sin * xy + cos * cos * yy,
            ],
            -1,
        )
        turned = torch.stack([-base[..., 3], base[..., 2]], -1)  # d(R v)/d angle
        turn = (tick * tick * nose_variance)[..., None] * torch.stack(
            [turned[..., 0] ** 2, turned[..., 0] * turned[..., 1], turned[..., 1] ** 2], -1
        )
        along_variance, across_variance = variance[..., 2], variance[..., 3]
        cos_h, sin_h = torch.cos(heading), torch.sin(heading)
        changed = torch.stack(
            [
                along_variance * cos_h * cos_h + across_variance * sin_h * sin_h,
                (along_variance - across_variance) * cos_h * sin_h,
                along_variance * sin_h * sin_h + across_variance * cos_h * cos_h,
            ],
            -1,
        )
        velocity_covariance = rotated + turn + changed
        speed_square = mean["velocity_x"] ** 2 + mean["velocity_y"] ** 2
        position_variance = (
            sigma["position"] ** 2
            + tick * tick * 0.5 * (xx + yy)
            + 0.5 * speed_square * (0.5 * tick * tick) ** 2 * nose_variance
            + 0.5 * (variance[..., 0] + variance[..., 1])
        )
        heading_mean = wrap(base[..., index["heading"]] + change[..., part["heading"]])
        heading_variance = (
            sigma["heading"] ** 2 + tick * tick * nose_variance + variance[..., part["heading"]]
        )

        def scalar(name: str, scale=1.0):
            value = base[..., index[name]] + change[..., part[name]] * scale
            spread = torch.sqrt(sigma[name] ** 2 + variance[..., part[name]] * scale**2)
            return value, spread

        nose_rate, nose_spread = scalar("nose_rate")
        attack, attack_spread = scalar("angle_of_attack")
        index_value, index_spread = scalar("log_refractive_index")
        shield, shield_spread = scalar("shield", kind["max_shield"])
        energy, energy_spread = scalar("energy", kind["max_energy"])
        shield = torch.minimum(torch.clamp_min(shield, 0.0), kind["max_shield"])
        energy = torch.minimum(torch.clamp_min(energy, 0.0), kind["max_energy"])

        reset = events[..., 0].to(means.dtype)
        countdown = base[..., index["shield_recharge_wait"]]
        wait = reset * kind["delay"] + (1.0 - reset) * countdown
        wait_spread = torch.sqrt(
            (1.0 - reset) * sigma["shield_recharge_wait"] ** 2
            + reset * (1.0 - reset) * (kind["delay"] - countdown) ** 2
        )

        # The cooldown as its code: the belief's, counted down one bin, mixed with the shot.
        fired = events[..., 1].to(means.dtype)[..., None]
        held = torch.clamp_min(
            state_code._cooldown_sharp(mean["gun_cooldown"]) + spreads[..., len(SPREADS) :], 0.0
        )
        counted = torch.cat(
            [held[..., :1] + held[..., 1:2], held[..., 2:], torch.zeros_like(held[..., :1])], -1
        )
        cooldown_code = (
            fired * state_code._cooldown_sharp(kind["fired_cooldown"]) + (1.0 - fired) * counted
        )
        centres = tick * torch.arange(
            cooldown_code.shape[-1], dtype=means.dtype, device=means.device
        )
        cooldown = (cooldown_code * centres).sum(-1)
        residual = cooldown_code - state_code._cooldown_sharp(cooldown)

        values = dict(
            x=position[..., 0],
            y=position[..., 1],
            velocity_x=velocity[..., 0],
            velocity_y=velocity[..., 1],
            heading=heading_mean,
            nose_rate=nose_rate,
            angle_of_attack=attack,
            shield_recharge_wait=wait,
            shield=shield,
            energy=energy,
            gun_cooldown=cooldown,
            log_refractive_index=index_value,
        )
        spread = dict(
            position=torch.sqrt(position_variance),
            velocity_xx=velocity_covariance[..., 0],
            velocity_xy=velocity_covariance[..., 1],
            velocity_yy=velocity_covariance[..., 2],
            heading=torch.sqrt(heading_variance),
            nose_rate=nose_spread,
            angle_of_attack=attack_spread,
            shield_recharge_wait=wait_spread,
            shield=shield_spread,
            energy=energy_spread,
            log_refractive_index=index_spread,
        )
        return (
            torch.stack([values[name] for name in MEANS], -1),
            torch.cat([torch.stack([spread[name] for name in SPREADS], -1), residual], -1),
        )
