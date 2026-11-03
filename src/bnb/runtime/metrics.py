"""The diagnostics collector (diagnostics spec 1): producers give their series as device
tensors in a flat mapping, the collector holds them over an update and reads them back in
one transfer.

A series is named quantity first, `{group}/{quantity}/{slice}`, in full words. Its value is
- a tensor (any shape that holds one number) or a host number: the value itself;
- a `Mean`: a total and a count, divided on the host after the readback, so a mean over no
  samples is absent from the record rather than a 0/0;
- a `Table`: rows of host values, for the views that are not lines (the fired events, the
  league's entries).

Values are raw: nothing here smooths (diagnostics audit, "Rules and method"). Producers own
their series: a new series needs no change here.
"""

from collections.abc import Mapping

import torch

from bnb.contracts.series import Mean, Series, Table

__all__ = ["CADENCE", "Collector", "Mean", "Series", "Table", "due", "read", "read_back"]

# Updates between the cadence groups' measurements (diagnostics spec 1, cadence C).
CADENCE = 10


def read_back(values: torch.Tensor) -> list[float]:
    """The host's copy of a 1-D tensor: the update's one synchronisation (and the early
    stop's per-epoch KL read). Every read the diagnostics make goes through here, so the
    no-sync test can allow exactly these."""
    return values.tolist()


def due(update: int, cadence: int = CADENCE) -> bool:
    """Whether update `update` (counting from 1) measures the cadence groups: every
    `cadence`-th update, and the first, so a short run has one measurement."""
    return update == 1 or update % cadence == 0


class Collector:
    """The series of one update, by name; `read` returns them on the host and empties it."""

    def __init__(self) -> None:
        self.values: dict[str, Series] = {}

    def set(self, series: Mapping[str, Series]) -> None:
        """Hold each series, replacing one of the same name."""
        self.values.update(series)

    def add(self, series: Mapping[str, Series]) -> None:
        """Sum each series into the one of the same name (tensors, host numbers and Means
        add; a new name is held as given)."""
        for name, value in series.items():
            held = self.values.get(name)
            self.values[name] = value if held is None else held + value

    def read(self) -> dict:
        """Every series on the host, by name: tensors in one stacked transfer per device
        (float64, so counts stay exact), a Mean as total / count (absent when the count is
        0), integer and boolean tensors as ints, a table as JSON (`Table.as_json`) and host
        numbers as given. The collector is emptied."""
        values, self.values = self.values, {}
        pending: dict[torch.device, list[torch.Tensor]] = {}
        slots: dict[str, tuple] = {}

        def place(value):
            if isinstance(value, Table):
                return ("host", value.as_json())
            if not isinstance(value, torch.Tensor):
                return ("host", value)
            if value.numel() != 1:
                raise ValueError(f"a series holds one number; got shape {tuple(value.shape)}")
            column = pending.setdefault(value.device, [])
            column.append(value.detach().reshape(()).to(torch.float64))
            integral = not (value.is_floating_point() or value.is_complex())
            return ("device", value.device, len(column) - 1, integral)

        for name, value in values.items():
            if isinstance(value, Mean):
                slots[name] = ("mean", place(value.total), place(value.count))
            else:
                slots[name] = ("value", place(value))
        host = {device: read_back(torch.stack(column)) for device, column in pending.items()}

        def fetch(slot):
            if slot[0] == "host":
                return slot[1]
            number = host[slot[1]][slot[2]]
            return int(number) if slot[3] else number

        out = {}
        for name, slot in slots.items():
            if slot[0] == "mean":
                total, count = fetch(slot[1]), fetch(slot[2])
                if count:
                    out[name] = total / count
            else:
                value = fetch(slot[1])
                if value is not None:
                    out[name] = value
        return out


def read(series: Mapping[str, Series]) -> dict:
    """`series` on the host, in one transfer (a collector of its own)."""
    collector = Collector()
    collector.set(series)
    return collector.read()
