"""Shared cases for the ship-card acceptance tests: varied cards spanning the design
envelope, the perturbations of the independence test, and one batched solve of them all."""

import functools

import torch

from bnb.contracts import GameConfig
from bnb.game import api
from bnb.game.handling import FLIGHT, NAMES, Handles, measure, solve
from bnb.game.handling.handles import stack
from bnb.game.handling.manoeuvres import batched, cruise_speed, ships, type_axis
from bnb.game.physics.bullets import fire
from bnb.game.state import Bullets, replace

VARIED = 8  # cards, each with every type
PERTURBATION = 0.05
TICK_RATES = (25, 30, 60)
# The design envelope, as log-uniform factors on the default card's parameters.
ENVELOPE = dict(max_lift_coefficient=(0.75, 1.33), max_acceleration=(0.75, 1.33),
                zero_lift_drag=(0.75, 1.33), engine_power=(0.75, 1.33),
                static_thrust=(0.75, 1.33), max_energy=(0.7, 1.4),
                corner_nose_frequency=(0.7, 1.4), induced_drag_scale=(0.75, 1.33),
                coast_exchange_scale=(0.7, 1.4), dive_exchange_scale=(0.75, 1.33),
                climb_exchange_scale=(0.75, 1.33), vision_range_scale=(0.7, 1.4),
                stall_angle_scale=(0.67, 1.67), muzzle_speed=(0.8, 1.25),
                bullet_drag_scale=(0.6, 1.6), bullet_energy_cost=(0.7, 1.4), hit_radius=(0.8, 1.25),
                max_shield=(0.7, 1.4), shield_recharge_rate_scale=(0.6, 1.6),
                head_on_fraction_scale=(0.5, 2.0), hit_damage=(0.7, 1.4))


def rules(rate: int = 30):
    return api.rules_of(GameConfig(1, 1, 1000.0), tick_length=1.0 / rate)


def varied_cards(seed: int = 20261101) -> api.ShipTypes:
    """VARIED random cards (B, 2) in float64: the envelope's factors, a full-store power of
    65-80% of the engine's, a bleed exponent from 1.5 to 3, recharge delays of 3, 6 or 9 s
    and a spawn store of 40%. Each satisfies the cruise fix."""
    generator = torch.Generator().manual_seed(seed)
    default = api.ship_types(dtype=torch.float64)

    def uniform(low, high):
        return low + (high - low) * torch.rand(VARIED, 2, generator=generator, dtype=torch.float64)

    values = {}
    for name, (low, high) in ENVELOPE.items():
        factor = torch.exp(uniform(torch.tensor(low).log(), torch.tensor(high).log()))
        values[name] = getattr(default, name) * factor
    values["engine_power_full_store"] = values["engine_power"] * uniform(0.65, 0.8)
    # The bleed exponent reshapes the polar below full lift: the induced drag at full lift,
    # k C_max^e, keeps its envelope factor.
    exponent = uniform(1.5, 3.0)
    values["bleed_exponent"] = exponent
    values["induced_drag_scale"] = (values["induced_drag_scale"]
                                    * values["max_lift_coefficient"] ** (2.0 - exponent))
    delay = torch.randint(1, 4, (VARIED, 2), generator=generator).double() * 0.5
    values["shield_recharge_delay_scale"] = delay
    values["spawn_energy"] = 0.4 * values["max_energy"]
    values["fire_rate"] = torch.full((VARIED, 2), 5.0, dtype=torch.float64)  # whole at 25 Hz
    # The cruise fix: the engine's power below the knee is the full store's plus the charging
    # power at cruise (which the full-store power alone sets), so each card is itself a
    # solution of its handles.
    cards = api.ship_types(dtype=torch.float64, **values)
    cruise = cruise_speed(rules(), cards, cards.max_energy[..., None])[..., 0]
    coast = api.rules_of(GameConfig(1, 1, 1.0)).flight.coast_exchange * cards.coast_exchange_scale
    values["engine_power"] = values["engine_power_full_store"] + coast * cruise
    return api.ship_types(dtype=torch.float64, **values)


def perturbed(base: Handles) -> tuple[Handles, list[tuple[str, float]]]:
    """Each flight handle of base (num_types,) by -5% and +5%, every other held."""
    cards, labels = [], []
    for name in FLIGHT:
        for sign in (-1.0, 1.0):
            cards.append(base.replace(**{name: getattr(base, name) * (1.0 + sign * PERTURBATION)}))
            labels.append((name, sign))
    return stack(cards), labels


class Cases:
    """Every card of the acceptance tests, solved in one batch and measured back.

    Rows: the default card's handles, then VARIED cards' handles, then the perturbations of
    the default's handles (labels)."""

    def __init__(self):
        self.rules = rules()
        self.default = measure(api.ship_types(dtype=torch.float64), self.rules)
        self.varied_handles = measure(varied_cards(), self.rules)
        perturbations, self.labels = perturbed(self.default)
        self.targets = stack([self.default] + [Handles(**{n: getattr(self.varied_handles, n)[i]
                                                          for n in NAMES})
                                               for i in range(VARIED)])
        self.targets = Handles(**{n: torch.cat([getattr(self.targets, n),
                                                getattr(perturbations, n)]) for n in NAMES})
        self.solution = solve(self.targets, self.rules, strict=False)
        self.measured = measure(double(self.solution.types), self.rules)

    @property
    def round_trip(self) -> slice:
        return slice(0, 1 + VARIED)

    @property
    def independence(self) -> slice:
        return slice(1 + VARIED, None)


def double(types: api.ShipTypes) -> api.ShipTypes:
    """A float32 card as float64 (its values exactly), for measuring."""
    return api.ShipTypes(**{name: value.double() for name, value in vars(types).items()})


def seconds_between_shots(rules, types: api.ShipTypes) -> torch.Tensor:
    """(B, num_types) s between a type's first two shots, firing at every chance
    through the game's `fire` (and the step's cooldown count), with a store that never runs
    dry."""
    types = batched(types)
    ship_type = type_axis(types)
    zeros = torch.zeros(ship_type.shape, dtype=types.max_energy.dtype)
    state = ships(zeros + 100.0, zeros + 1e12, zeros, zeros, ship_type)
    empty = torch.zeros(ship_type.shape + (rules.bullet_slots,), dtype=zeros.dtype)
    bullets = Bullets(x=empty, y=empty, direction_x=empty, direction_y=empty, proper_speed=empty,
                      log_refractive_index=empty, log_index_gradient_x=empty,
                      log_index_gradient_y=empty, moves_left=empty.int())
    shoot = torch.ones_like(zeros, dtype=torch.bool)
    shots = []
    for _ in range(round(2.0 / rules.tick_length)):  # 2 s: two shots at 1 Hz or faster
        state, bullets = fire(rules, types, state, bullets, shoot, torch.zeros(zeros.shape + (2,)))
        shots.append(state.fired)
        state = replace(state, gun_cooldown=(state.gun_cooldown - 1).clamp_min(0))
    fired = torch.stack(shots, -1)  # (B, T, ticks)
    assert fired[..., 0].all()  # the first shot is at tick 0
    second = (fired.cumsum(-1) < 2).sum(-1)  # the second shot's tick
    return second.double() * rules.tick_length


@functools.cache
def cases() -> Cases:
    return Cases()
