"""What a producer's diagnostic series hold (diagnostics spec 1): a mapping from a name,
`{group}/{quantity}/{slice}` in full words, to a value. A value is a tensor that holds one
number (on the device; the collector, `bnb.runtime.metrics.Collector`, reads every one back
in one transfer), a host number, a `Mean` or a `Table`."""

import dataclasses
import time

import torch


@dataclasses.dataclass(frozen=True)
class Mean:
    """A mean as its total and its count (tensors or host numbers), divided on the host after
    the readback; absent from the record when the count is 0, rather than a 0/0."""

    total: torch.Tensor | float
    count: torch.Tensor | float

    def __add__(self, other: Mean) -> Mean:
        return Mean(self.total + other.total, self.count + other.count)


@dataclasses.dataclass(frozen=True)
class Table:
    """Rows of host values under named columns: a W&B table, a list of rows in the JSONL."""

    columns: tuple[str, ...]
    rows: tuple[tuple, ...]

    def as_json(self) -> dict:
        """{"columns": [...], "rows": [[...], ...]}."""
        return dict(columns=list(self.columns), rows=[list(row) for row in self.rows])


Series = torch.Tensor | Mean | Table | float | int | bool | None


class Stopwatch:
    """A device-side interval: `start` and `stop` record CUDA events (no synchronisation) or,
    on the CPU, the host clock; `seconds` is read after the update's readback, when the
    events have completed (None before a stop)."""

    def __init__(self, device: torch.device) -> None:
        self.cuda = device.type == "cuda"
        self.started = self.stopped = None

    def start(self) -> None:
        self.started, self.stopped = self._now(), None

    def stop(self) -> None:
        self.stopped = self._now()

    def _now(self):
        if not self.cuda:
            return time.perf_counter()
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def seconds(self) -> float | None:
        """The interval in seconds; the CUDA events' wait is a no-op after the readback."""
        if self.stopped is None:
            return None
        if not self.cuda:
            return self.stopped - self.started
        self.stopped.synchronize()
        return self.started.elapsed_time(self.stopped) / 1000.0
