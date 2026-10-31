"""Per-ship values of the constants that differ by ship type (SPEC 1.1).

A ship's type is the state key `ship_type` (B, N). Every per-type rule reads its constant
through these lookups, so the game code has no branch on a type: a constant that every
type shares stays a plain number, and one that differs becomes a gather from a small table.
"""

import dataclasses

import torch

from .constants import Flight, Preset

FLIGHT_FIELDS = tuple(f.name for f in dataclasses.fields(Flight))  # at import: Dynamo cannot trace fields()


def _gather(values: tuple, ship_type: torch.Tensor) -> torch.Tensor:
    dtype = torch.int32 if all(isinstance(v, int) for v in values) else torch.float32
    return torch.tensor(values, dtype=dtype, device=ship_type.device)[ship_type.long()]


def type_value(p: Preset, key: str, ship_type: torch.Tensor) -> torch.Tensor:
    """Per-type constant `key` (constants.PER_TYPE_KEYS) for each ship, shaped like ship_type."""
    return _gather(p.per_type(key), ship_type)


def flight_of(flights: tuple, ship_type: torch.Tensor) -> Flight:
    """The flight constants of each ship: a field that differs by type becomes a tensor
    shaped like `ship_type`, a shared one stays a number; flight.py broadcasts over both."""
    per_ship = {}
    for name in FLIGHT_FIELDS:
        values = tuple(getattr(fl, name) for fl in flights)
        if any(v != values[0] for v in values):
            per_ship[name] = _gather(values, ship_type)
    return dataclasses.replace(flights[0], **per_ship)


def slot_types(p: Preset, batch: int, device) -> torch.Tensor:
    """(B, N) int32 ship_type of every slot (SPEC 1.1)."""
    return torch.tensor(p.slot_types, dtype=torch.int32, device=device).expand(batch, p.num_ships).contiguous()
