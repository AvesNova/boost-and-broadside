"""An instant estimate of the card meeting a set of handles, for live tuning: no flights.

`solve` is exact but flies the game's code for every iteration (a minute or more on the
CPU). Live tuning needs a card on the next frame, so `Estimate` answers from closed forms
only, anchored on an exact card:

    estimate = Estimate(rules, card, handles)   # handles: the card's own (measure, solve)
    types = estimate.card(new_handles)           # field by field: card * g(new) / g(handles)

`g` is `closed_card`: the solver's first guess (closed forms in the handles), with the bleed
exponent chosen so that the bleed rate's closed form (`bleed_rate_now`) meets its handle.
At the anchor the estimate is the anchor card exactly; away from it, the closed forms'
systematic errors mostly cancel in the ratio, so the estimate's error is the change in that
error between the anchor and the new handles, which grows with the size of the change.
"""

import dataclasses
import math

import torch

from ..physics.flight import airframe_of, lift_coefficient, rates, setting
from ..rules import Rules
from ..ship_types import ShipTypes
from .handles import NAMES, Handles
from .manoeuvres import BLEED_STICK, TURNING_STORE, batched, steady_turn, type_axis
from .solver import EXPONENT_BOUNDS, _batched_handles, _card, first_guess
from .weapons import combat_card

EXPONENT_GRID = 24  # bleed exponents tried at once, log-spaced over the solver's bounds


def bleed_rate_now(rules: Rules, types: ShipTypes) -> torch.Tensor:
    """(B, num_types) the bleed rate handle in closed form: the specific energy rate
    u du/dt + (dE/dt) / m at the bleed manoeuvre's start (corner speed, the turning store,
    the steady angle of attack at BLEED_STICK), which `manoeuvres.fly` measures as the
    flight's starting rate."""
    types = batched(types)
    constants = rules.flight
    airframe = airframe_of(types, type_axis(types))
    corner = airframe.corner_speed
    turning = TURNING_STORE * airframe.max_energy
    stick = torch.full_like(corner, BLEED_STICK)
    angle = steady_turn(rules, airframe, corner, turning, stick).angle
    current = setting(constants, airframe, corner, turning, stick, torch.zeros_like(corner))
    lift = lift_coefficient(constants, airframe, angle)
    speed_rate, energy_rate = rates(
        constants, airframe, current, 1.0, corner, lift, torch.cos(angle), 1.0 / (corner * corner)
    )
    return corner * speed_rate + energy_rate / constants.mass


def _repeat(handles: Handles, combat: dict, copies: int) -> tuple[Handles, dict]:
    def repeat(value):
        return value.repeat(copies, *([1] * (value.dim() - 1)))

    return (
        Handles(**{n: repeat(getattr(handles, n)) for n in NAMES}),
        {k: repeat(v) for k, v in combat.items()},
    )


def closed_card(rules: Rules, handles: Handles) -> ShipTypes:
    """The card the closed forms give for these handles ((B, num_types) fields, float64):
    the first guess at the bleed exponent where the bleed rate's closed form meets its
    handle (interpolated in log exponent over EXPONENT_GRID values, then the first guess at
    it). An exponent that no grid value brackets takes the nearest end."""
    handles = _batched_handles(handles)
    combat = combat_card(rules, handles, handles.cruise_speed)
    B, type_count = handles.ceiling.shape
    low, high = (math.log(bound) for bound in EXPONENT_BOUNDS)
    grid = torch.linspace(low, high, EXPONENT_GRID, dtype=handles.ceiling.dtype)
    repeated, repeated_combat = _repeat(handles, combat, EXPONENT_GRID)
    exponents = grid.exp().repeat_interleave(B).unsqueeze(-1).expand(EXPONENT_GRID * B, type_count)
    guesses = first_guess(rules, repeated, repeated_combat, exponents)
    cards = _card(rules, repeated, guesses, repeated_combat)
    miss = bleed_rate_now(rules, cards) - repeated.bleed_rate
    miss = miss.reshape(EXPONENT_GRID, B, type_count)
    # The first bracketing pair along the grid, interpolated linearly in log exponent.
    sign = miss.sign()
    change = (sign[1:] * sign[:-1] <= 0.0) & torch.isfinite(miss[1:]) & torch.isfinite(miss[:-1])
    has = change.any(0)
    first = change.int().argmax(0)  # (B, types)
    left = miss.gather(0, first[None])[0]
    right = miss.gather(0, (first + 1)[None])[0]
    fraction = (left / (left - right)).nan_to_num(0.0).clamp(0.0, 1.0)
    step = grid[1] - grid[0]
    bracketed = grid[first] + fraction * step
    nearest = grid[miss.abs().nan_to_num(math.inf).argmin(0)]
    exponent = torch.where(has, bracketed, nearest).exp()
    return _card(rules, handles, first_guess(rules, handles, combat, exponent), combat)


@dataclasses.dataclass
class Estimate:
    """The instant card for any handles, anchored on an exact card and its handles (both
    (num_types,) or (B, num_types)). `card` returns float32 fields shaped like the anchor's."""

    rules: Rules
    anchor: ShipTypes
    handles: Handles

    def __post_init__(self) -> None:
        self._one_card = self.anchor.max_energy.dim() == 1
        anchor = batched(self.anchor)
        self._anchor = {name: value.double() for name, value in vars(anchor).items()}
        handles = _batched_handles(self.handles)
        handles = Handles(**{n: getattr(handles, n).double() for n in NAMES})
        self._base = vars(closed_card(self.rules, handles))

    def card(self, handles: Handles) -> ShipTypes:
        """The estimated card for these handles: each field the anchor's times the closed
        forms' ratio between these handles and the anchor's."""
        handles = _batched_handles(handles)
        handles = Handles(**{n: getattr(handles, n).double() for n in NAMES})
        closed = vars(closed_card(self.rules, handles))
        fields = {}
        for name, anchor in self._anchor.items():
            base, value = self._base[name], closed[name]
            ratio = torch.where(base != 0.0, value / base, torch.ones_like(base))
            fields[name] = torch.where(base != 0.0, anchor * ratio, value).float()
        if self._one_card:
            fields = {name: value[0] for name, value in fields.items()}
        return ShipTypes(**fields)
