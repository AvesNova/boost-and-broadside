"""An instant estimate of the card meeting a set of handles, for live tuning: no flights.

`solve` is exact but flies the game's code for every iteration (a minute or more on the
CPU). Live tuning needs a card on the next frame, so `Estimate` answers from closed forms
only, anchored on an exact card:

    estimate = Estimate(rules, card, handles)   # handles: the card's own (measure, solve)
    types = estimate.card(new_handles)           # field by field: card * g(new) / g(handles)

`g` is `closed_card`: the solver's first guess (closed forms in the handles). At the anchor
the estimate is the anchor card exactly; away from it, the closed forms' systematic errors
mostly cancel in the ratio, so the estimate's error is the change in that error between the
anchor and the new handles, which grows with the size of the change.

`bleed_rate_now` gives a card's bleed rate (no longer a handle: the half-lift drag sets the
drag's shape) in closed form, for display.
"""

import dataclasses

import torch

from ..physics.flight import airframe_of, lift_coefficient, rates, setting
from ..rules import Rules
from ..ship_types import ShipTypes
from .handles import NAMES, Handles
from .manoeuvres import BLEED_STICK, TURNING_STORE, batched, steady_turn, type_axis
from .solver import _batched_handles, _card, first_guess
from .weapons import combat_card


def bleed_rate_now(rules: Rules, types: ShipTypes) -> torch.Tensor:
    """(B, num_types) the bleed rate (shown, not set) in closed form: the specific energy rate
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


def closed_card(rules: Rules, handles: Handles) -> ShipTypes:
    """The card the closed forms give for these handles ((B, num_types) fields, float64): the
    solver's first guess, at the bleed exponent the half-lift drag sets."""
    handles = _batched_handles(handles)
    combat = combat_card(rules, handles, handles.cruise_speed)
    return _card(rules, handles, first_guess(rules, handles, combat), combat)


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
