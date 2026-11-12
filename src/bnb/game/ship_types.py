"""The ship card: parameters that differ by ship type (game spec 1.1, ship-card spec 1), as
tensors.

Each field is a tensor with one entry per type, shape (num_types,), or (B, num_types) to
vary them per environment (a sweep). Code reads a ship's value with `per_ship`, which
fuses into the kernels that use it, so changing a value never recompiles.

The parameters that were shared before the ship card (`SCALED`) are stored as a per-type
scale of their shared value in `Rules` or `FlightConstants`: a type's stall angle is
`flight.stall_angle * stall_angle_scale`. The shared values keep their meaning for every
caller that reads them, and the default card (every scale 1) is exactly the shared game.
"""

from fractions import Fraction

import torch

from bnb.contracts.tree import contract

from . import ships

FIGHTER, HEAVY = 0, 1
RATE_DENOMINATOR = 1000  # a rate is read as a fraction with at most this denominator

# Game spec 1.1: the card's fields, with their units (spec key in brackets). The values, one
# per ship type, live in the ship files (`bnb.game.ships`) and are read into DEFAULTS.
FIELDS = dict(
    max_lift_coefficient="1/m [cl_max]",
    max_acceleration="m/s^2, the g limit [a_max]",
    zero_lift_drag="1/m [cd0]",
    engine_power="W, below the knee [power]",
    engine_power_full_store="W [p_min]",
    static_thrust="N [f_static]",
    max_energy="J [e_max]",
    corner_nose_frequency="rad/s at corner speed [omega_c]",
    max_shield="[shield_max]",
    hit_radius="m",
    hit_damage="[damage]",
    muzzle_speed="m/s, proper",
    bullet_energy_cost="J [bullet_cost]",
    fire_rate="shots/s; 2 and 3 ticks between shots at 30 Hz [fire_interval]",
    bullet_life="s, whole ticks; the shooter's bullets', from its gun range",
    spawn_energy="J",
    bleed_exponent="e: induced drag k (C^2 + (z/u^2)^2)^(e/2)",
    # Per-type scales of shared values (SCALED), 1 by default.
    induced_drag_scale="of flight.induced_drag_factor [k]",
    stall_angle_scale="of flight.stall_angle [alpha_s]",
    coast_exchange_scale="of flight.coast_exchange [a_0]",
    dive_exchange_scale="of flight.dive_exchange [a_d]",
    climb_exchange_scale="of flight.climb_exchange [a_c]",
    vision_range_scale="of rules.vision_range; the observer's [V]",
    bullet_spread_scale="of rules.bullet_spread; the shooter's [sigma_b]",
    bullet_drag_scale="of rules.bullet_drag; the shooter's [k_b]",
    shield_recharge_rate_scale="of rules.shield_recharge_rate; on quiet ticks only",
    shield_recharge_delay_scale="of rules.shield_recharge_delay; counted on quiet ticks only",
    head_on_fraction_scale="of rules.head_on_fraction; the target's [f_0]",
    engine_knee_scale="of flight.engine_knee [e_k]",
)


def _read(ship_set: str) -> dict[str, tuple[float, ...]]:
    """A set's card values by field, one per ship type in the ship files' order."""
    found = ships.load(ship_set)
    for ship in found:
        if set(ship.card) != set(FIELDS):
            missing, unknown = (
                sorted(set(FIELDS) - set(ship.card)),
                sorted(set(ship.card) - set(FIELDS)),
            )
            raise ValueError(f"ship {ship.key}: card lacks {missing} and has unknown {unknown}")
    return {name: tuple(ship.card[name] for ship in found) for name in FIELDS}


STANDARD = ships.STANDARD
DEFAULTS = _read(STANDARD)  # the standard set's card, kept in place by `reload`
_DEFAULTS = {STANDARD: DEFAULTS}  # by ship set, read when first asked for
_NAMES = {STANDARD: ships.names(STANDARD)}


def defaults(ship_set: str = STANDARD) -> dict[str, tuple[float, ...]]:
    """A ship set's default card values by field, one per type (read once, then kept)."""
    if ship_set not in _DEFAULTS:
        _DEFAULTS[ship_set] = _read(ship_set)
    return _DEFAULTS[ship_set]


def type_names(ship_set: str = STANDARD) -> tuple[str, ...]:
    """A ship set's type names, in the card's order (the ship files' `name`)."""
    if ship_set not in _NAMES:
        _NAMES[ship_set] = ships.names(ship_set)
    return _NAMES[ship_set]


def reload() -> None:
    """Read the ship files again (after the tuning panel saved them): every set's default
    card, type names and rules follow them (the standard card in place)."""
    _DEFAULTS.clear()
    _NAMES.clear()
    ships.rules.cache_clear()
    DEFAULTS.clear()
    DEFAULTS.update(_read(STANDARD))
    _DEFAULTS[STANDARD] = DEFAULTS


# Card field -> (owner in Rules, shared field) for the per-type scales.
SCALED = dict(
    induced_drag_scale=("flight", "induced_drag_factor"),
    stall_angle_scale=("flight", "stall_angle"),
    coast_exchange_scale=("flight", "coast_exchange"),
    dive_exchange_scale=("flight", "dive_exchange"),
    climb_exchange_scale=("flight", "climb_exchange"),
    vision_range_scale=("rules", "vision_range"),
    bullet_spread_scale=("rules", "bullet_spread"),
    bullet_drag_scale=("rules", "bullet_drag"),
    shield_recharge_rate_scale=("rules", "shield_recharge_rate"),
    shield_recharge_delay_scale=("rules", "shield_recharge_delay"),
    head_on_fraction_scale=("rules", "head_on_fraction"),
    engine_knee_scale=("flight", "engine_knee"),
)


@contract
class ShipTypes:
    """Ship-card parameters by type, float; see FIELDS for units and SCALED for the
    scales. Times are time-native: the fire rate becomes ticks with `ticks_between_shots`, the
    bullet life with `bullet_moves`."""

    max_lift_coefficient: torch.Tensor
    max_acceleration: torch.Tensor
    zero_lift_drag: torch.Tensor
    engine_power: torch.Tensor
    engine_power_full_store: torch.Tensor
    static_thrust: torch.Tensor
    max_energy: torch.Tensor
    corner_nose_frequency: torch.Tensor
    max_shield: torch.Tensor
    hit_radius: torch.Tensor
    hit_damage: torch.Tensor
    muzzle_speed: torch.Tensor
    bullet_energy_cost: torch.Tensor
    fire_rate: torch.Tensor
    bullet_life: torch.Tensor
    spawn_energy: torch.Tensor
    bleed_exponent: torch.Tensor
    induced_drag_scale: torch.Tensor
    stall_angle_scale: torch.Tensor
    coast_exchange_scale: torch.Tensor
    dive_exchange_scale: torch.Tensor
    climb_exchange_scale: torch.Tensor
    vision_range_scale: torch.Tensor
    bullet_spread_scale: torch.Tensor
    bullet_drag_scale: torch.Tensor
    shield_recharge_rate_scale: torch.Tensor
    shield_recharge_delay_scale: torch.Tensor
    head_on_fraction_scale: torch.Tensor
    engine_knee_scale: torch.Tensor


def default_card(ship_set: str = STANDARD) -> dict[str, tuple[float, ...]]:
    """A ship set's default card values by field, one per type: the JSON form a
    configuration holds, which `ship_types(**card)` turns into tensors."""
    return {
        name: tuple(float(value) for value in values) for name, values in defaults(ship_set).items()
    }


def ship_types(
    device=None, dtype=torch.float32, ship_set: str = STANDARD, **overrides
) -> ShipTypes:
    """A ship set's default card (the standard set's by default). An override is a per-type
    sequence or tensor ((num_types,) or (B, num_types)), or one number for every type. dtype
    is the fields' (offline tools use float64)."""
    values = {}
    for name, default in defaults(ship_set).items():
        value = overrides.pop(name, default)
        value = torch.as_tensor(value, dtype=dtype, device=device)
        values[name] = value.expand(len(default)).clone() if value.dim() == 0 else value
    if overrides:
        raise TypeError(f"unknown ship-card parameters {sorted(overrides)}")
    batched = [v.shape[0] for v in values.values() if v.dim() == 2]
    if batched:  # one card per environment: give every parameter the batch axis
        values = {
            k: v if v.dim() == 2 else v.expand(batched[0], -1).clone() for k, v in values.items()
        }
    return ShipTypes(**values)


def per_ship(table: torch.Tensor, ship_type: torch.Tensor) -> torch.Tensor:
    """A card parameter for each ship, shaped like ship_type (B, ...).

    A select over the (few, statically known) types rather than a gather: it fuses into
    any kernel without an indirect load or bounds check, and needs no host copy, so it is
    also safe in eager CUDA-graph capture.
    """
    if table.dim() == 2:  # (B, num_types): each environment its own card
        table = table.reshape(table.shape[:1] + (1,) * (ship_type.dim() - 1) + table.shape[1:])
    out = table[..., 0].expand(ship_type.shape)
    for i in range(1, table.shape[-1]):
        out = torch.where(ship_type == i, table[..., i], out)
    return out


def bullet_moves(bullet_life: torch.Tensor, tick_length: float) -> torch.Tensor:
    """The moves of each bullet life (s) at this tick, int32: the card holds whole ticks, so
    rounding removes only the float error."""
    return torch.round(bullet_life / tick_length).to(torch.int32)


def ticks_between_shots(fire_rate: torch.Tensor, tick_length: float) -> torch.Tensor:
    """The ticks between shots at this tick of each fire rate (Hz), int32: the tick
    rate over the fire rate, in exact fractions. Raises ValueError, naming both rates, for a
    fire rate that is not a whole number of ticks. The card stores floats, so a rate is read
    as the nearest fraction with a denominator up to RATE_DENOMINATOR: 4.285714 Hz is 30/7,
    seven ticks at 30 Hz."""
    tick_rate = Fraction(1.0 / tick_length).limit_denominator(RATE_DENOMINATOR)
    ticks = []
    for value in fire_rate.flatten().tolist():
        if value <= 0.0:
            raise ValueError(f"a fire rate must be positive, not {value:g} Hz")
        count = tick_rate / Fraction(value).limit_denominator(RATE_DENOMINATOR)
        if count.denominator != 1:
            raise ValueError(
                f"a fire rate of {value:g} Hz is not a whole number of ticks at "
                f"{float(tick_rate):g} Hz ({float(count):g} ticks between shots)"
            )
        ticks.append(count.numerator)
    return torch.tensor(ticks, dtype=torch.int32, device=fire_rate.device).reshape(fire_rate.shape)
