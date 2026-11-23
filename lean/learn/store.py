"""Where an update's rollout lives: the segments' per-tick records, compact, in pinned host
memory, streamed out while collecting and back in micro-batches while learning.

Layout. U segments of T ticks over S seats give V = U x S views (a view is one seat's
segment); view v = u x S + s. Every per-tick record is (T, V + 1, ...): the last view is a
spare that stays empty (nothing owned, nothing labelled, nothing reset) and pads a short
micro-batch, so every micro-batch has one shape. The initial carries are (V + 1, ...), the
view axis first.

Where. The large records (the observation's per-tick fields, actions, old
log-probabilities, stored values, rewards, next-state targets) and the initial carries are
in pinned host memory on a GPU. The masks and labels the update counts and indexes on the
device (`RESIDENT`: own, done, reset, next_valid, hit_label, deal_row), the per-view
bootstrap values, and the update's advantages and value targets stay on the device. The
observation fields constant within an episode (`EPISODE_CONSTANT`: the fields, their
positions and mask) and the position offset are stored once per deal in a table on the
device (rows per segment: the environments as the segment starts, then the segment's
deals); each tick records only the seat's row (`deal_row`).

Compact types (author 2026-11-23). `stored` narrows a record as it is written: positions,
headings and the PPO numbers (`FLOAT32`) stay float32; every other float is float16; joint
actions and hit labels are int8 (signed: the -1 sentinel stays); the deal row int32; masks
stay bool. `widened` casts back on the device after the copy (floats to float32, integers
to long), before anything reaches the model, so the position code and the rotary angles are
computed from float32 positions. The deal table keeps the game's float32 (it is small).

Writing. `collect` hands each tick's records to `write_tick`. The copy stream waits for the
compute stream (a stream wait, no host wait), copies the large records into the pinned
arrays asynchronously, and `record_stream` keeps each source alive until its copy has run;
the next tick is computed while this one is copied. The resident records are written on
the compute stream.

Reading. `ready` makes the host wait once per update for the copies written so far (the
host reads the pinned arrays itself). `batches` gathers each micro-batch's views on the
host (index_select into a pinned staging buffer) and copies it to the device on the copy
stream while the compute stream works on the previous micro-batch: two staging slots; a
copy waits (event) for the compute stream to have released its slot; the compute stream
waits (event) for the copy. The host waits only for a staging buffer's previous copy before
it overwrites it, which finished a micro-batch earlier.

On the CPU, or with `stream=False`, the same arrays live on the device and `batches`
indexes them directly (the same types, no copies).
"""

import dataclasses

import torch
from torch import Tensor

# Per-tick records kept on the device: one byte per token, counted and indexed there.
RESIDENT = ("own", "done", "reset", "next_valid", "hit_label", "deal_row")
# Observation fields constant within an episode: stored once per deal (with the position offset).
EPISODE_CONSTANT = ("field", "field_pos", "field_mask")
# Floats kept in float32: positions and headings (the position code and the rotary angles
# read them) and the PPO numbers.
FLOAT32 = frozenset(("pos", "heading", "field_pos", "position_offset", "log_probability", "value", "reward"))
# Small integers stored as int8: joint actions (< 42) and hit labels (-1 to 2).
INT8 = frozenset(("action", "teacher", "hit_label"))


def stored(name: str, value: Tensor) -> Tensor:
    """`value` in its storage type (module docstring, "Compact types")."""
    if value.is_floating_point():
        return value.to(torch.float32 if name in FLOAT32 else torch.float16)
    if name in INT8:
        return value.to(torch.int8)
    if name == "deal_row":
        return value.to(torch.int32)
    return value


def widened(value: Tensor) -> Tensor:
    """A stored record back in the type the learner computes with: float32 or long (bool kept)."""
    if value.is_floating_point():
        return value.float()
    if value.dtype in (torch.int8, torch.int32):
        return value.long()
    return value


def view_axis(carry: Tensor) -> int:
    """The seat axis of a recurrent carry (lean.learn.collect.carry_axis)."""
    from lean.learn.collect import carry_axis

    return carry_axis(carry)


class RolloutStore:
    """U segments of a rollout (module docstring). `stream`: keep the large records in pinned
    host memory (used only on a GPU). Sized at the first segment; kept across updates."""

    def __init__(self, segments: int, stream: bool = True):
        self.segments = segments
        self.stream = stream
        self.data: dict[str, Tensor] = {}  # large per-tick records (T, V + 1, ...), host or device
        self.resident: dict[str, Tensor] = {}  # RESIDENT per-tick records (T, V + 1, ...), device
        self.constants: dict[str, Tensor] = {}  # per-deal rows (U x rows, ...), device
        self.carry: Tensor | None = None  # initial carries (V + 1, ...), host or device
        self.ticks = 0
        self.observation_type = None
        self.observation_names: tuple[str, ...] = ()
        self._staging: dict = {}
        self.centres: Tensor | None = None  # the critic's support the stored values were decoded with

    # ------------------------------------------------------------------ sizes

    def _size(self, initial_carry: Tensor, table: dict, ticks: int) -> None:
        self.device = initial_carry.device
        self.pinned = self.stream and self.device.type == "cuda"
        self.copy_stream = torch.cuda.Stream(self.device) if self.pinned else None
        self.ticks = ticks
        self.axis = view_axis(initial_carry)
        self.seats = initial_carry.shape[self.axis]
        self.views = self.segments * self.seats
        self.rows = next(iter(table.values())).shape[0]
        carry = initial_carry.movedim(self.axis, 0)
        self.carry = self._zeros((self.views + 1,) + carry.shape[1:], carry.dtype, self.pinned)
        self.constants = {
            name: torch.zeros((self.segments * self.rows,) + rows.shape[1:], dtype=rows.dtype, device=self.device)
            for name, rows in table.items()
        }
        self.data, self.resident, self._staging = {}, {}, {}
        self.bootstrap: Tensor | None = None
        self.advantage: Tensor | None = None
        self.target: Tensor | None = None

    def _zeros(self, shape, dtype, host: bool) -> Tensor:
        """Zeros (the spare view stays zero): pinned host memory when `host`, else the device."""
        if host:
            return torch.zeros(shape, dtype=dtype, pin_memory=True)
        return torch.zeros(shape, dtype=dtype, device=self.device)

    def _columns(self, segment: int) -> slice:
        return slice(segment * self.seats, (segment + 1) * self.seats)

    def nbytes(self) -> dict[str, int]:
        """Bytes held per record (per-tick records, the deal table, the carries)."""
        sizes = {name: tensor.nbytes for name, tensor in {**self.data, **self.resident}.items()}
        sizes["deal_table"] = sum(tensor.nbytes for tensor in self.constants.values())
        sizes["carry"] = self.carry.nbytes
        return sizes

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

    def begin_segment(self, segment: int, ticks: int, initial_carry: Tensor, table: dict[str, Tensor]) -> None:
        """Start segment `segment` (from 0): the live policy's carry before its first tick
        (seats on carry_axis) and the segment's deal table {name: (rows, ...)}
        (EPISODE_CONSTANT fields and "position_offset"; row r < E is environment r as the
        segment starts, then the segment's deals)."""
        if self.carry is None or initial_carry.shape[view_axis(initial_carry)] != self.seats or ticks != self.ticks:
            self._size(initial_carry, table, ticks)
        rows = slice(segment * self.rows, (segment + 1) * self.rows)
        for name, values in table.items():
            self.constants[name][rows] = values
        self._copy([(self.carry[self._columns(segment)], initial_carry.movedim(self.axis, 0).contiguous())])

    def write_tick(self, segment: int, tick: int, records: dict) -> None:
        """One tick's records, each (S, ...): "observation" (an Obs), "deal_row" (S,) the
        seat's row in this segment's deal table, and the Rollout's other per-tick fields
        (its "position_offset" is not kept: the deal table holds it)."""
        columns = self._columns(segment)
        observation = records["observation"]
        items = {
            field.name: getattr(observation, field.name)
            for field in dataclasses.fields(observation)
            if isinstance(getattr(observation, field.name), Tensor) and field.name not in EPISODE_CONSTANT
        }
        if self.observation_type is None:
            self.observation_type = type(observation)
            self.observation_names = tuple(items)
        items.update((name, value) for name, value in records.items() if name not in ("observation", "position_offset"))
        items["deal_row"] = records["deal_row"] + segment * self.rows
        pairs = []
        for name, value in items.items():
            value = stored(name, value).contiguous()
            if name in RESIDENT:
                if name not in self.resident:
                    self.resident[name] = self._resident(name, value.shape[1:], value.dtype)
                self.resident[name][tick, columns] = value
                continue
            if name not in self.data:
                self.data[name] = self._zeros((self.ticks, self.views + 1) + value.shape[1:], value.dtype, self.pinned)
            pairs.append((self.data[name][tick, columns], value))
        self._copy(pairs)

    def _resident(self, name: str, shape: tuple, dtype) -> Tensor:
        """A RESIDENT record, filled so that the spare view owns, ends, resets and labels nothing."""
        empty = torch.zeros((self.ticks, self.views + 1) + tuple(shape), dtype=dtype, device=self.device)
        return empty.fill_(-1) if name == "hit_label" else empty

    def end_segment(self, segment: int, hit_label: Tensor, bootstrap: Tensor, centres: Tensor) -> None:
        """Close segment `segment`: its hit labels (T, S, N) (known only once the bullets have
        ended), the bootstrap values (S, N, LEVELS) and the critic's support (LEVELS, BINS)."""
        columns = self._columns(segment)
        if self.bootstrap is None or self.bootstrap.shape[1:] != bootstrap.shape[1:]:
            self.bootstrap = torch.zeros((self.views + 1,) + bootstrap.shape[1:], device=self.device)
        self.bootstrap[columns] = bootstrap
        self.centres = centres
        if "hit_label" not in self.resident:
            self.resident["hit_label"] = self._resident("hit_label", hit_label.shape[2:], torch.int8)
        self.resident["hit_label"][:, columns] = stored("hit_label", hit_label)

    @classmethod
    def of(cls, rollout, stream: bool = False) -> RolloutStore:
        """A one-segment store holding a collected (unstreamed) Rollout."""
        store = cls(1, stream)
        ticks = rollout.done.shape[0]
        store.begin_segment(0, ticks, rollout.initial_carry, rollout.deal_table)
        for tick in range(ticks):
            records = {name: getattr(rollout, name)[tick] for name in rollout.TICK_FIELDS}
            records["observation"] = _tick(rollout.observation, tick)
            store.write_tick(0, tick, records)
        store.end_segment(0, rollout.hit_label, rollout.bootstrap_value, rollout.value_centres)
        return store

    # ---------------------------------------------------------------- reading

    def ready(self) -> None:
        """The host waits for every copy written so far (once per update, before reading)."""
        if self.copy_stream is not None:
            self.copy_stream.synchronize()

    def counts(self) -> Tensor:
        """(V + 1, 3) float per view: own tokens, next_valid tokens, labelled own tokens."""
        own = self.resident["own"]
        labelled = own & (self.resident["hit_label"] >= 0)
        return torch.stack([own.sum((0, 2)), self.resident["next_valid"].sum((0, 2)), labelled.sum((0, 2))], 1).float()

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
        (T, B, ...) on the device, widened, plus "carry" (the seat axis where the policy
        has it) and "views" (B,). A yielded dict is valid until the next one is asked for.
        From pinned memory the next micro-batch is gathered and copied while the caller
        computes on this one."""
        index_device = index.pin_memory().to(self.device, non_blocking=True) if self.device.type == "cuda" else index
        large = tuple(name for name in names if name in self.data)
        small = tuple(name for name in names if name in self.resident)

        def finish(batch: dict, views: Tensor) -> dict:
            for name in small:
                batch[name] = self.resident[name].index_select(1, views)
            batch = {name: widened(value) for name, value in batch.items()}
            batch["carry"] = batch["carry"].movedim(0, self.axis).contiguous()
            batch["views"] = views
            return batch

        if not self.pinned:
            for row in range(index.shape[0]):
                views = index_device[row]
                batch = {name: self.data[name].index_select(1, views) for name in large}
                batch["carry"] = self.carry.index_select(0, views)
                yield finish(batch, views)
            return

        compute = torch.cuda.current_stream(self.device)
        slots = self._slots(large, index.shape[1])

        def issue(row: int) -> None:
            slot = slots[row % 2]
            slot["copied"].synchronize()  # the staging buffer's previous copy has finished
            views = index[row]
            for name in large:
                torch.index_select(self.data[name], 1, views, out=slot["staging"][name])
            torch.index_select(self.carry, 0, views, out=slot["staging"]["carry"])
            with torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_event(slot["released"])
                for name, staged in slot["staging"].items():
                    slot["device"][name].copy_(staged, non_blocking=True)
                slot["copied"].record(self.copy_stream)

        issue(0)
        for row in range(index.shape[0]):
            slot = slots[row % 2]
            compute.wait_event(slot["copied"])
            if row + 1 < index.shape[0]:
                issue(row + 1)  # gathered on the host while the device works on this one
            yield finish(dict(slot["device"]), index_device[row])
            slot["released"].record(compute)

    def _slots(self, names: tuple[str, ...], size: int) -> list[dict]:
        """Two staging slots for micro-batches of `size` views of `names`: pinned host and
        device buffers, and the events of their last copy and last release. Kept."""
        key = (names, size)
        if key not in self._staging:
            sources = {name: (self.data[name], 1) for name in names} | {"carry": (self.carry, 0)}

            def shape(stored: Tensor, axis: int) -> tuple:
                return stored.shape[:axis] + (size,) + stored.shape[axis + 1:]

            slots = []
            for _ in range(2):
                slot = {
                    "staging": {name: torch.empty(shape(t, axis), dtype=t.dtype, pin_memory=True) for name, (t, axis) in sources.items()},
                    "device": {name: torch.empty(shape(t, axis), dtype=t.dtype, device=self.device) for name, (t, axis) in sources.items()},
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
        if self.advantage is None or self.advantage.shape != own.shape or self.target.shape[-1] != levels:
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
