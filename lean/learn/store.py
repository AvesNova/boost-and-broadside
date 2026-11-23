"""Where an update's rollout lives: the segments' per-tick records in pinned host memory,
streamed out while collecting and back in micro-batches while learning.

Layout. U segments of T ticks over S seats give V = U x S views (a view is one seat's
segment); view v = u x S + s. Every per-tick record is (T, V + 1, ...): the last view is a
spare that stays empty (nothing owned, nothing labelled) and pads a short micro-batch, so
every micro-batch has one shape. The large records (the observation, actions, old
log-probabilities, the next-state targets, hit labels, rewards and stored values) are in
pinned host memory on a GPU; the small ones the update indexes by view on the device
(`own`, `done`, `reset`, the deal row), the per-view bootstrap values and loss
denominators, and the update's advantages and value targets stay on the device. The field
observations and the position offset are constant within an episode: they are stored once
per deal in a table (rows per segment: the environments as the segment starts, then the
segment's deals), and each tick records only the seat's row.

Writing. `collect` hands each tick's records to `write_tick`; the copy stream waits for
the compute stream (an event, no host wait), copies them into the pinned arrays
asynchronously and `record_stream` keeps each source alive until its copy has run, so the
next tick is computed while this one is copied. Writing the same view twice is ordered by
the copy stream.

Reading. `ready` makes the host wait once per update for the copies it is about to read.
`batches` gathers a micro-batch's views on the host (index_select into a pinned staging
buffer) and copies it to the device on the copy stream while the compute stream works on
the previous micro-batch: two staging slots, the copy waits for the compute stream to have
released its slot (event), the compute stream waits for the copy (event). The host waits
only for a staging buffer's previous copy before it overwrites it (long finished by then).

On the CPU, or with `host=False` (everything on the device, for measurements and small
runs), the same arrays live on the device and `batches` indexes them directly.
"""

import dataclasses

import torch
from torch import Tensor

# Per-tick records kept on the device (small; indexed by view on the device).
RESIDENT = ("own", "done", "reset", "deal_row")
# Observation fields constant within an episode: stored once per deal (with the position offset).
EPISODE_CONSTANT = ("field", "field_pos", "field_mask")
# Narrow storage types for records whose values allow it (joint actions < 42, hit labels -1..2).
NARROW = {"action": torch.uint8, "teacher": torch.uint8, "hit_label": torch.int8}


class RolloutStore:
    """U segments of a rollout (module docstring). `host`: keep the large records in pinned
    host memory (used only on a GPU). Sized at the first segment; kept across updates."""

    def __init__(self, segments: int, host: bool = True):
        self.segments = segments
        self.host = host
        self.data: dict[str, Tensor] = {}  # per-tick records (T, V + 1, ...), host or device
        self.resident: dict[str, Tensor] = {}  # per-tick records on the device (RESIDENT)
        self.constants: dict[str, Tensor] = {}  # per-deal rows (U x rows, ...) on the device
        self.carry: Tensor | None = None  # initial carries (blocks, V + 1, ...), host or device
        self.observation_type = None
        self.observation_names: tuple[str, ...] = ()
        self._staging: dict = {}
        self._centres: Tensor | None = None

    # ------------------------------------------------------------------ sizes

    def _size(self, initial_carry: Tensor, constants: dict, ticks: int) -> None:
        self.device = initial_carry.device
        self.pinned = self.host and self.device.type == "cuda"
        self.place = torch.device("cpu") if self.pinned else self.device
        self.copy_stream = torch.cuda.Stream(self.device) if self.pinned else None
        self.ticks = ticks
        self.seats = initial_carry.shape[1]
        self.views = self.segments * self.seats
        self.rows = next(iter(constants.values())).shape[0]
        self.carry = self._empty((initial_carry.shape[0], self.views + 1) + initial_carry.shape[2:], initial_carry.dtype, True)
        self.constants = {
            name: torch.empty((self.segments * self.rows,) + table.shape[1:], dtype=table.dtype, device=self.device)
            for name, table in constants.items()
        }
        self.counts = torch.zeros(self.views + 1, 3, device=self.device)  # own, next_valid, labelled per view
        self.paid = torch.zeros(0, dtype=torch.bool, device=self.device)
        self.bootstrap: Tensor | None = None
        self.advantage: Tensor | None = None
        self.target: Tensor | None = None

    def _empty(self, shape, dtype, large: bool) -> Tensor:
        """Zeros (the spare view stays zero): pinned host memory for a large record when pinned."""
        if large and self.pinned:
            return torch.zeros(shape, dtype=dtype, pin_memory=True)
        return torch.zeros(shape, dtype=dtype, device=self.device)

    def _columns(self, segment: int) -> slice:
        return slice(segment * self.seats, (segment + 1) * self.seats)

    # ---------------------------------------------------------------- writing

    def _copy(self, pairs: list[tuple[Tensor, Tensor]]) -> None:
        """Copy (destination, source) pairs: asynchronously on the copy stream when pinned
        (after the work queued so far on the compute stream), else in place."""
        if self.copy_stream is None:
            for destination, source in pairs:
                destination.copy_(source)
            return
        self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(self.copy_stream):
            for destination, source in pairs:
                destination.copy_(source, non_blocking=True)
                source.record_stream(self.copy_stream)

    def begin_segment(self, segment: int, ticks: int, initial_carry: Tensor, constants: dict[str, Tensor]) -> None:
        """Start segment `segment` (from 0): the live policy's carry before its first tick,
        (blocks, S, ...), and its deal table, {name: (rows, ...)} (EPISODE_CONSTANT fields
        and "position_offset"; row r < E the environment r as the segment starts, then the
        segment's deals)."""
        if self.carry is None or initial_carry.shape[1] != self.seats or ticks != self.ticks:
            self._size(initial_carry, constants, ticks)
        columns = self._columns(segment)
        if segment == 0:
            self.paid = torch.zeros(0, dtype=torch.bool, device=self.device)
        self.counts[columns] = 0
        rows = slice(segment * self.rows, (segment + 1) * self.rows)
        for name, table in constants.items():
            self.constants[name][rows] = table
        self._copy([(self.carry[block, columns], initial_carry[block].contiguous()) for block in range(initial_carry.shape[0])])

    def write_tick(self, segment: int, tick: int, records: dict) -> None:
        """One tick's records, each (S, ...): "observation" (an Obs), "deal_row" (S,) the
        seat's row in this segment's deal table, and the Rollout's per-tick fields."""
        columns = self._columns(segment)
        observation = records["observation"]
        fields = {
            field.name: getattr(observation, field.name)
            for field in dataclasses.fields(observation)
            if isinstance(getattr(observation, field.name), Tensor) and field.name not in EPISODE_CONSTANT
        }
        if self.observation_type is None:
            self.observation_type = type(observation)
            self.observation_names = tuple(name for name in fields if name not in RESIDENT)
        items = {**fields, **{name: value for name, value in records.items() if name not in ("observation", "position_offset")}}
        items["deal_row"] = records["deal_row"] + segment * self.rows
        own = items["own"]
        self.counts[columns, 0] += own.sum(-1)
        self.counts[columns, 1] += items["next_valid"].sum(-1)
        paid = (items["reward"] != 0).flatten(0, -2).any(0)
        self.paid = paid if self.paid.numel() == 0 else self.paid | paid
        pairs = []
        for name, value in items.items():
            value = value.to(NARROW.get(name, value.dtype)).contiguous()
            if name in RESIDENT:
                if name not in self.resident:
                    self.resident[name] = torch.zeros((self.ticks, self.views + 1) + value.shape[1:], dtype=value.dtype, device=self.device)
                self.resident[name][tick, columns] = value
                continue
            if name not in self.data:
                self.data[name] = self._empty((self.ticks, self.views + 1) + value.shape[1:], value.dtype, True)
            pairs.append((self.data[name][tick, columns], value))
        self._copy(pairs)

    def end_segment(self, segment: int, hit_label: Tensor, bootstrap: Tensor, centres: Tensor) -> None:
        """Close segment `segment`: its hit labels (T, S, N) (known only once the bullets have
        ended), the bootstrap values (S, N, LEVELS) and the critic's support (LEVELS, BINS)."""
        columns = self._columns(segment)
        if self.bootstrap is None:
            self.bootstrap = torch.zeros((self.views + 1,) + bootstrap.shape[1:], device=self.device)
        self.bootstrap[columns] = bootstrap
        self._centres = centres
        self.counts[columns, 2] = (self.resident["own"][:, columns] & (hit_label >= 0)).sum((0, 2)).float()
        hit_label = hit_label.to(NARROW["hit_label"])
        if "hit_label" not in self.data:
            self.data["hit_label"] = self._empty((self.ticks, self.views + 1) + hit_label.shape[2:], hit_label.dtype, True)
        self._copy([(self.data["hit_label"][tick, columns], hit_label[tick]) for tick in range(self.ticks)])

    @classmethod
    def of(cls, rollout, host: bool = False) -> "RolloutStore":
        """A one-segment store holding a collected Rollout (its per-tick fields on the device)."""
        store = cls(1, host)
        ticks = rollout.done.shape[0]
        store.begin_segment(0, ticks, rollout.initial_carry, rollout.deal_table)
        for tick in range(ticks):
            records = {name: getattr(rollout, name)[tick] for name in rollout.TICK_FIELDS}
            records["observation"] = _tick(rollout.observation, tick)
            store.write_tick(0, tick, records)
        store.end_segment(0, rollout.hit_label, rollout.bootstrap_value, rollout.value_centres)
        return store

    # ---------------------------------------------------------------- reading

    @property
    def centres(self) -> Tensor:
        """The critic's support the stored values were decoded with."""
        return self._centres

    def ready(self) -> None:
        """The host waits for every copy written so far (once per update, before reading)."""
        if self.copy_stream is not None:
            self.copy_stream.synchronize()

    def batch_index(self, order: Tensor, size: int, minibatches: int = 1) -> tuple[Tensor, list[int]]:
        """Micro-batches of views: `order` (V,) split into `minibatches` near-equal parts, each
        into micro-batches of `size` views with the last padded by the spare view. Returns
        the index (M, size) long on the host and each micro-batch's minibatch."""
        rows, owners = [], []
        for part, views in enumerate(torch.tensor_split(order, min(minibatches, order.numel()))):
            for start in range(0, views.numel(), size):
                chunk = views[start:start + size]
                rows.append(torch.cat([chunk, chunk.new_full((size - chunk.numel(),), self.views)]))
                owners.append(part)
        return torch.stack(rows), owners

    def batches(self, names: tuple[str, ...], index: Tensor):
        """Yield one dict per row of `index` (M, B) (host, long): each name of `names` as
        (T, B, ...) on the device (narrow types widened back), "carry" (blocks, B, ...) and
        "views" (B,). A yielded dict is valid until the next one is asked for. From pinned
        memory the next micro-batch is gathered and copied while the caller computes."""
        index_device = index.pin_memory().to(self.device, non_blocking=True) if self.device.type == "cuda" else index
        host_names = [name for name in names if name in self.data]
        resident_names = [name for name in names if name in self.resident]

        def finish(batch: dict, views: Tensor) -> dict:
            for name in resident_names:
                batch[name] = self.resident[name].index_select(1, views)
            for name, dtype in NARROW.items():
                if name in batch:
                    batch[name] = batch[name].long()
            batch["views"] = views
            return batch

        if not self.pinned:
            for row in range(index.shape[0]):
                views = index_device[row]
                batch = {name: self.data[name].index_select(1, views) for name in host_names}
                batch["carry"] = self.carry.index_select(1, views)
                yield finish(batch, views)
            return

        compute = torch.cuda.current_stream(self.device)
        slots = self._slots(tuple(host_names), index.shape[1])

        def issue(row: int) -> None:
            slot = slots[row % 2]
            slot["copied"].synchronize()  # the staging buffer's previous copy has finished
            views = index[row]
            for name in host_names:
                torch.index_select(self.data[name], 1, views, out=slot["staging"][name])
            torch.index_select(self.carry, 1, views, out=slot["staging"]["carry"])
            with torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_event(slot["released"])
                for name, staged in slot["staging"].items():
                    slot["device"][name].copy_(staged, non_blocking=True)
                slot["copied"].record(self.copy_stream)

        issue(0)
        for row in range(index.shape[0]):
            slot = slots[row % 2]
            compute.wait_event(slot["copied"])
            yield finish(dict(slot["device"]), index_device[row])
            slot["released"].record(compute)
            if row + 1 < index.shape[0]:
                issue(row + 1)

    def _slots(self, names: tuple[str, ...], size: int) -> list[dict]:
        """Two staging slots for micro-batches of `size` views of `names`: pinned host and
        device buffers, and the events of their last copy and last release. Kept."""
        key = (names, size)
        if key not in self._staging:
            def shape(stored: Tensor, axis: int) -> tuple:
                return stored.shape[:axis] + (size,) + stored.shape[axis + 1:]

            slots = []
            for _ in range(2):
                sources = {name: (self.data[name], 1) for name in names} | {"carry": (self.carry, 1)}
                slot = {
                    "staging": {name: torch.empty(shape(stored, axis), dtype=stored.dtype, pin_memory=True) for name, (stored, axis) in sources.items()},
                    "device": {name: torch.empty(shape(stored, axis), dtype=stored.dtype, device=self.device) for name, (stored, axis) in sources.items()},
                    "copied": torch.cuda.Event(),
                    "released": torch.cuda.Event(),
                }
                slot["copied"].record(self.copy_stream)
                slot["released"].record(torch.cuda.current_stream(self.device))
                slots.append(slot)
            self._staging[key] = slots
        return self._staging[key]

    def observation(self, batch: dict) -> tuple[object, Tensor]:
        """The batch's observation (the game's Obs, every field (T, B, ...)) and position
        offset (T, B, 2), the episode-constant fields looked up by deal row."""
        rows = batch["deal_row"]
        fields = {name: batch[name] for name in self.observation_names}
        fields["own"] = batch["own"]

        def look_up(table: Tensor) -> Tensor:
            return table.index_select(0, rows.flatten()).view(rows.shape + table.shape[1:])

        for name in EPISODE_CONSTANT:
            if name in self.constants:
                fields[name] = look_up(self.constants[name])
        return self.observation_type(**fields), look_up(self.constants["position_offset"])

    def targets(self, levels: int) -> tuple[Tensor, Tensor]:
        """The update's advantages (T, V + 1, N) and value targets (T, V + 1, N, LEVELS) on the
        device, allocated once."""
        own = self.resident["own"]
        if self.advantage is None or self.advantage.shape != own.shape:
            self.advantage = torch.zeros(own.shape, device=self.device)
            self.target = torch.zeros(own.shape + (levels,), device=self.device)
        return self.advantage, self.target



def _tick(observation, tick: int):
    """One tick of a stacked observation."""
    changes = {
        field.name: getattr(observation, field.name)[tick]
        for field in dataclasses.fields(observation)
        if isinstance(getattr(observation, field.name), Tensor)
    }
    return dataclasses.replace(observation, **changes)
