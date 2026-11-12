"""The scripted agent's doctrines: per ship type, how it fights (docs/explanation/game/ship-design/
archetypes.md, section 5). One agent; each own ship reads its type's `Doctrine`, so a mixed
team flies each type by its own.

A doctrine changes the combat layer's aim point and power only (never the shot: a shot
still needs the nose on the lead point). Its weight against the committed target is
`strength` times `in_advantage`, a logistic in the doctrine's advantage, the product of the
own-over-target ratios of the sustained turn rate, the climb power, the cruise speed and
the gun range, each raised to its exponent: so a doctrine can act against one opponent type
and not another. Within it:

- **Extend** (the energy fighter): rather than turn after a target its nose is far from
  (`extend_in_stall_angle`), or engage without an energy advantage (`engage_in_energy_ratio`,
  own over target specific energy, speed^2 / 2 plus the store, of a seen target; and
  `engage_in_store_fraction`, a store floor), it flies away from the target, boosting,
  while within the fight's range (`fight_in_gun_range`), and comes back from beyond it.
- **Slash**: on the attack, boost within `slash_in_gun_range`, and never reverse.
- **Bleed** (the turn fighter): within the fight's range, with the nose far off the target
  (`bleed_in_stall_angle`) and faster than the corner speed (`bleed_in_corner_speed`),
  reverse to slow toward the corner speed, where the turn is fastest.
- **Lag**: aim `lag_in_gun_range` gun ranges behind the target along its track, more the
  farther the ship is from the target's tail (none dead astern), to come round onto it.
- **Deny**: when the target's nose is on the ship (`deny_in_stall_angle`, its angle off the
  target's nose in the target's stall angles) within `deny_in_gun_range`, turn
  `deny_angle` off the line to it, to the side it is already off, so the pass is not a
  mutual head-on.

Each element's weight (`extend`, `slash`, `bleed`, `lag`, `deny`) is 0 by default, as is
`strength`: the default doctrine is off. A `table` packs one doctrine per ship type (T, D),
or per view and type (R, T, D); `per_ship` reads each own ship's row.

The tuned doctrines live in `scripted_doctrines.json` by ship set and type key (`stored`);
the scripted agent flies them by default (`for_card`: the set is the one whose type count
is the card's; a type without a stored doctrine flies none).
"""

import dataclasses
import json
import pathlib

import torch

from .scripted_parameters import Logistic

DOCTRINES = pathlib.Path(__file__).with_name("scripted_doctrines.json")


@dataclasses.dataclass(frozen=True)
class Doctrine:
    """One ship type's doctrine (the module docstring); each value a float, or (R, S) per own
    ship (`per_ship`)."""

    strength: float = 0.0
    turn_exponent: float = -1.0
    climb_exponent: float = 0.0
    speed_exponent: float = 0.0
    range_exponent: float = 0.0
    in_advantage: Logistic = Logistic(1.0, 0.1, True)
    fight_in_gun_range: Logistic = Logistic(1.0, 0.2, False)
    extend: float = 0.0
    extend_in_stall_angle: Logistic = Logistic(4.0, 0.2, True)
    engage_in_energy_ratio: Logistic = Logistic(0.9, 0.03, True)
    engage_in_store_fraction: Logistic = Logistic(0.05, 0.3, True)
    slash: float = 0.0
    slash_in_gun_range: Logistic = Logistic(3.0, 0.2, False)
    bleed: float = 0.0
    bleed_in_corner_speed: Logistic = Logistic(1.2, 0.1, True)
    bleed_in_stall_angle: Logistic = Logistic(2.0, 0.2, True)
    lag: float = 0.0
    lag_in_gun_range: float = 0.5
    deny: float = 0.0
    deny_in_stall_angle: Logistic = Logistic(1.0, 0.2, False)
    deny_in_gun_range: Logistic = Logistic(3.0, 0.2, False)
    deny_angle: float = 0.5  # rad


NAMES = tuple(field.name for field in dataclasses.fields(Doctrine))
_DEFAULT = Doctrine()
# The element weights and the strength, each in [0, 1]; the exponents in [-2, 2].
WEIGHTS = ("strength", "extend", "slash", "bleed", "lag", "deny")
EXPONENTS = ("turn_exponent", "climb_exponent", "speed_exponent", "range_exponent")
# Each logistic's smallest scale (a tenth of a factor e in its quantity, near the combat
# layer's floors in stall angles).
SCALE_FLOORS = {
    "in_advantage": 0.05,
    "fight_in_gun_range": 0.1,
    "extend_in_stall_angle": 0.158,
    "engage_in_energy_ratio": 0.03,
    "engage_in_store_fraction": 0.1,
    "slash_in_gun_range": 0.1,
    "bleed_in_corner_speed": 0.031,
    "bleed_in_stall_angle": 0.158,
    "deny_in_stall_angle": 0.158,
    "deny_in_gun_range": 0.1,
}


def pack(doctrine: Doctrine) -> list[float]:
    """The doctrine as a flat list (a logistic is its centre and scale)."""
    out = []
    for name in NAMES:
        value = getattr(doctrine, name)
        out.extend((value.centre, value.scale) if isinstance(value, Logistic) else (value,))
    return [float(number) for number in out]


WIDTH = len(pack(_DEFAULT))


def unpack(rows: torch.Tensor) -> Doctrine:
    """A doctrine whose values are the last axis's columns of `rows` (..., WIDTH)."""
    values, column = {}, 0
    for name in NAMES:
        kind = getattr(_DEFAULT, name)
        if isinstance(kind, Logistic):
            values[name] = Logistic(rows[..., column], rows[..., column + 1], kind.rising)
            column += 2
        else:
            values[name] = rows[..., column]
            column += 1
    doctrine = object.__new__(Doctrine)  # no generated __init__: callable inside a graph
    for name, value in values.items():
        object.__setattr__(doctrine, name, value)
    return doctrine


def from_row(row: list[float]) -> Doctrine:
    """One doctrine of floats from a packed row: `pack`'s inverse."""
    if len(row) != WIDTH:
        raise ValueError(f"a packed doctrine has {WIDTH} numbers, not {len(row)}")
    values = unpack(torch.tensor(row, dtype=torch.float64))
    return Doctrine(
        **{
            name: (
                Logistic(float(value.centre), float(value.scale), value.rising)
                if isinstance(value, Logistic)
                else float(value)
            )
            for name, value in ((name, getattr(values, name)) for name in NAMES)
        }
    )


def table(doctrines: list[Doctrine], device=None) -> torch.Tensor:
    """(T, WIDTH) float32: one row per ship type, in the card's order."""
    return torch.tensor([pack(d) for d in doctrines], dtype=torch.float32, device=device)


def per_ship(doctrines: torch.Tensor, own_type: torch.Tensor) -> Doctrine:
    """Each own ship's doctrine, (R, S) values: from a table per type (T, WIDTH) or per view
    and type (R, T, WIDTH), at the own ships' types (R, S)."""
    index = own_type.long()
    if doctrines.dim() == 2:
        return unpack(doctrines[index])
    views = torch.arange(index.shape[0], device=index.device)[:, None]
    return unpack(doctrines[views, index])


def named(doctrine: Doctrine, figures: int = 3) -> dict:
    """The doctrine's values by field name (a logistic [centre, scale]), rounded."""

    def rounded(number: float) -> float:
        return float(f"{number:.{figures}g}")

    return {
        name: (
            [rounded(value.centre), rounded(value.scale)]
            if isinstance(value, Logistic)
            else rounded(value)
        )
        for name, value in ((name, getattr(doctrine, name)) for name in NAMES)
    }


def from_named(fields: dict) -> Doctrine:
    """A doctrine from `named`'s fields; a field left out keeps the default."""
    values = {}
    for name, value in fields.items():
        kind = getattr(_DEFAULT, name)
        values[name] = Logistic(*value, kind.rising) if isinstance(kind, Logistic) else value
    return Doctrine(**values)


def stored(path: pathlib.Path = DOCTRINES) -> dict[str, dict[str, Doctrine]]:
    """The tuned doctrines by ship set and type key (`scripted_doctrines.json`)."""
    data = json.loads(path.read_text())
    return {
        ship_set: {key: from_named(entry["doctrine"]) for key, entry in types.items()}
        for ship_set, types in data.items()
    }


def set_table(keys: list[str], ship_set: str, device=None, path=DOCTRINES) -> torch.Tensor:
    """(T, WIDTH) the stored doctrines of a set's types by key, in order (a type without
    one: off)."""
    doctrines = stored(path).get(ship_set, {})
    return table([doctrines.get(key, _DEFAULT) for key in keys], device)


def for_card(characteristics, path: pathlib.Path = DOCTRINES) -> torch.Tensor | None:
    """(T, WIDTH) the stored doctrines for a card's types, on the card's device: those of the
    ship set whose type count is the card's T (the sets' counts differ), by type key; None
    when no set with stored doctrines has T types (every type then flies none)."""
    from bnb.game import ships

    count = characteristics.gun_range.shape[-1]
    device = characteristics.gun_range.device
    for ship_set in stored(path):
        keys = [ship.key for ship in ships.load(ship_set)]
        if len(keys) == count:
            return set_table(keys, ship_set, device, path)
    return None
