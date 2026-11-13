"""The scripted agent's doctrines: per ship type, how it fights (docs/explanation/game/ship-design/
archetypes.md, section 5). One agent; each own ship reads its type's `Doctrine`, so a mixed
team flies each type by its own.

A doctrine changes the combat layer's aim point and power (and, through `hold_fire`, may
withhold a shot; a shot still needs the nose on the lead point). Its weight against the
committed target is `strength` times `in_advantage`, a logistic in the doctrine's advantage,
the product of the own-over-target ratios of the sustained turn rate, the climb power, the
cruise speed and the gun range, each raised to its exponent: so a doctrine can act against
one opponent type and not another. Within it:

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

The maneuvers (the combat layer's `maneuvers`) are weighted on their own, not by `strength`
and `in_advantage`, so they fly in the mirrors too; each reads the committed target:

- **Lead turn**: on a head-on approach (the crossing angle of the two flight paths within
  `lead_turn_in_crossing` radians of opposed) with the merge `lead_turn_in_seconds` away
  (range over closure), start the turn early at `lead_turn_stick` of full stick: toward
  the side the bandit will pass on (the two-circle sense), or, with `lead_turn_one_circle`
  above 1/2, away from it (the one-circle sense).
- **Yo-yo**: after `yoyo_after_seconds` of a close fight without an angle gained (the
  carry's stalemate clock: the nose never on the target while the target's nose is off the
  ship), when the target comes within `yoyo_phase` radians of the nose, ease the stick to
  `yoyo_ease` and climb (reverse: speed into the store) for `yoyo_climb_seconds`, then
  full stick and dive (boost: the store back into speed) for `yoyo_dive_seconds`. The
  low yo-yo (`yoyo_low` above 1/2) dives in the first phase instead (the unloaded
  extension: in the slow lift-limited turn the turn rate grows with the speed) and leaves
  the second's power to the combat layer. The first phase ends early above
  `yoyo_to_corner_speed` corner speeds; a yo-yo never starts with the target's nose within
  2 of its stall angles of the ship.
- **Brake** (overshoot forcing): with the target in the rear hemisphere, its nose on the
  ship (`brake_in_stall_angle`, its stall angles) within `brake_in_gun_range` and closing,
  reverse (the store as a speed brake) while the combat layer turns toward it.
- **Sense** (the post-merge choice): with the target behind the ship's 3-9 line within
  `sense_in_gun_range` and turning, turn at full stick in its rotation sense (the
  two-circle), or against it (the one-circle) with probability `sense_one_circle`, drawn
  once each time the gate opens (the carry's `sense_draw`); weighted
  by `sense_in_lift_ratio`, rising in the own maximum lift coefficient over the target's
  (the inverse of the slow turn's radius: the one-circle goes to the smaller radius), and,
  with `sense_mirror_only` above 1/2, only against the own type. In a crowd (the soft
  count of the other known enemies within `sense_crowd_in_gun_range`, through the falling
  `sense_in_crowd`; open by default) the sense hands over to `crowd_sense`: the same side at
  `crowd_sense_stick` of full stick, boosting with weight `crowd_sense_boost` (whatever
  the types and lift ratio).
- **The post-merge choice by the situation**: the crowd counts an other enemy only with its
  nose (its track) within `sense_crowd_in_nose_angle` radians of the line to the ship (a
  threat), less `sense_crowd_ally_weight` times the soft count of the own ships inside the
  same radius. The crowd extension holds only while the own specific energy over the
  target's is below `crowd_sense_in_energy_ratio` and the own shield over the target's
  below `crowd_sense_in_shield_ratio` (both falling, open by default); where they close, the
  ship presses: the sense's turn (with its own gates) in the crowd. Alone, the **duel
  extension** (`duel_extend`) flies the crowd extension's stick and boost instead of the
  sense's turn when any of the energy, shield or lift ratio is below its gate
  (`duel_extend_in_energy_ratio`, `_in_shield_ratio`, `_in_lift_ratio`; falling, closed by
  default); with `duel_extend_mirror_only` above 1/2, only against the own type; against
  another type, weighted by `duel_extend_cross_in_others`, rising in the count of the other
  known enemies at any range (open by default; at a centre of 1/2, never in a lone 1v1).
- **Hold fire** (`hold_fire`): on a closing approach with the target's nose on the ship
  (`hold_fire_in_stall_angle`, its stall angles) inside `hold_fire_in_gun_range`, the
  shot's probability falls by the weight: the nose-to-nose meeting of a one-circle is not
  shot (the doctrine's only element on the shot).

Each element's weight (`extend`, `slash`, `bleed`, `lag`, `deny`, and the maneuvers'
`lead_turn`, `yoyo`, `brake`, `sense`, `duel_extend`, `hold_fire`) is 0 by default, as is
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
    lead_turn: float = 0.0
    lead_turn_in_seconds: Logistic = Logistic(1.5, 0.2, False)
    lead_turn_in_crossing: Logistic = Logistic(0.5, 0.3, False)
    lead_turn_stick: float = 1.0
    lead_turn_one_circle: float = 0.0
    yoyo: float = 0.0
    yoyo_after_seconds: float = 30.0
    yoyo_phase: float = 1.5  # rad
    yoyo_climb_seconds: float = 4.0
    yoyo_ease: float = 0.4
    yoyo_dive_seconds: float = 4.0
    yoyo_low: float = 0.0
    yoyo_to_corner_speed: float = 10.0
    brake: float = 0.0
    brake_in_stall_angle: Logistic = Logistic(1.5, 0.2, False)
    brake_in_gun_range: Logistic = Logistic(1.0, 0.2, False)
    sense: float = 0.0
    sense_one_circle: float = 0.0
    sense_in_gun_range: Logistic = Logistic(2.0, 0.2, False)
    sense_in_lift_ratio: Logistic = Logistic(1e-3, 0.05, True)
    sense_mirror_only: float = 0.0
    sense_crowd_in_gun_range: Logistic = Logistic(3.0, 0.1, False)
    sense_in_crowd: Logistic = Logistic(1e4, 0.1, False)
    crowd_sense: float = 0.0
    crowd_sense_stick: float = 0.0
    crowd_sense_boost: float = 0.0
    crowd_sense_in_energy_ratio: Logistic = Logistic(1e6, 0.1, False)
    crowd_sense_in_shield_ratio: Logistic = Logistic(1e6, 0.1, False)
    sense_crowd_in_nose_angle: Logistic = Logistic(1e6, 0.1, False)
    sense_crowd_ally_weight: float = 0.0
    duel_extend: float = 0.0
    duel_extend_in_energy_ratio: Logistic = Logistic(1e-6, 0.1, False)
    duel_extend_in_shield_ratio: Logistic = Logistic(1e-6, 0.1, False)
    duel_extend_in_lift_ratio: Logistic = Logistic(1e-6, 0.1, False)
    duel_extend_mirror_only: float = 0.0
    duel_extend_cross_in_others: Logistic = Logistic(1e-9, 0.1, True)
    hold_fire: float = 0.0
    hold_fire_in_stall_angle: Logistic = Logistic(1.0, 0.2, False)
    hold_fire_in_gun_range: Logistic = Logistic(0.5, 0.2, False)


NAMES = tuple(field.name for field in dataclasses.fields(Doctrine))
_DEFAULT = Doctrine()
# The element weights and the strength, each in [0, 1]; the exponents in [-2, 2].
WEIGHTS = (
    "strength",
    "extend",
    "slash",
    "bleed",
    "lag",
    "deny",
    "lead_turn",
    "yoyo",
    "brake",
    "sense",
    "crowd_sense",
    "duel_extend",
    "hold_fire",
)
# The maneuvers' fractions, each in [0, 1] (searched like a weight).
FRACTIONS = (
    "lead_turn_stick",
    "lead_turn_one_circle",
    "yoyo_ease",
    "yoyo_low",
    "sense_one_circle",
    "sense_mirror_only",
    "crowd_sense_stick",
    "crowd_sense_boost",
    "duel_extend_mirror_only",
)
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
    "lead_turn_in_seconds": 0.05,
    "lead_turn_in_crossing": 0.1,
    "brake_in_stall_angle": 0.158,
    "brake_in_gun_range": 0.1,
    "sense_in_gun_range": 0.1,
    "sense_in_lift_ratio": 0.03,
    "sense_crowd_in_gun_range": 0.1,
    "sense_in_crowd": 0.1,
    "crowd_sense_in_energy_ratio": 0.03,
    "crowd_sense_in_shield_ratio": 0.05,
    "sense_crowd_in_nose_angle": 0.1,
    "duel_extend_in_energy_ratio": 0.03,
    "duel_extend_in_shield_ratio": 0.05,
    "duel_extend_in_lift_ratio": 0.03,
    "duel_extend_cross_in_others": 0.1,
    "hold_fire_in_stall_angle": 0.158,
    "hold_fire_in_gun_range": 0.1,
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


# Read once per file and kept: the doctrines are read inside traced code (the play shell's
# exported scripted step builds `Scripted`, whose default reads them), and a traced function
# cannot open files. The module reads the default file when it is imported (below).
_STORED: dict[pathlib.Path, dict[str, dict[str, Doctrine]]] = {}
_SET_KEYS: dict[str, list[str]] = {}


def stored(path: pathlib.Path = DOCTRINES) -> dict[str, dict[str, Doctrine]]:
    """The tuned doctrines by ship set and type key (`scripted_doctrines.json`), read once."""
    if path not in _STORED:
        data = json.loads(path.read_text())
        _STORED[path] = {
            ship_set: {key: from_named(entry["doctrine"]) for key, entry in types.items()}
            for ship_set, types in data.items()
        }
    return _STORED[path]


def _set_keys(ship_set: str) -> list[str]:
    """A ship set's type keys in order, read once (the set's files are read like `stored`)."""
    if ship_set not in _SET_KEYS:
        from bnb.game import ships

        _SET_KEYS[ship_set] = [ship.key for ship in ships.load(ship_set)]
    return _SET_KEYS[ship_set]


def set_table(keys: list[str], ship_set: str, device=None, path=DOCTRINES) -> torch.Tensor:
    """(T, WIDTH) the stored doctrines of a set's types by key, in order (a type without
    one: off)."""
    doctrines = stored(path).get(ship_set, {})
    return table([doctrines.get(key, _DEFAULT) for key in keys], device)


def for_card(characteristics, path: pathlib.Path = DOCTRINES) -> torch.Tensor | None:
    """(T, WIDTH) the stored doctrines for a card's types, on the card's device: those of the
    ship set whose type count is the card's T (the sets' counts differ), by type key; None
    when no set with stored doctrines has T types (every type then flies none)."""
    count = characteristics.gun_range.shape[-1]
    device = characteristics.gun_range.device
    for ship_set in stored(path):
        keys = _set_keys(ship_set)
        if len(keys) == count:
            return set_table(keys, ship_set, device, path)
    return None


def rows_by_count(path: pathlib.Path = DOCTRINES) -> dict[int, list[list[float]]]:
    """A file's stored doctrines packed per card type count (`for_card`'s rows as plain
    floats, by the type count of each set with stored doctrines), for a caller that keeps
    them as constants outside any traced code."""
    out = {}
    for ship_set, doctrines in stored(path).items():
        keys = _set_keys(ship_set)
        out[len(keys)] = [pack(doctrines.get(key, _DEFAULT)) for key in keys]
    return out


def _warm(path: pathlib.Path = DOCTRINES) -> None:
    """Read the default doctrines and their sets' keys now, outside any traced code."""
    for ship_set in stored(path):
        _set_keys(ship_set)


_warm()
