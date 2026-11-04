"""Where an update's segments live (learner spec 2): pinned host memory, or the device.

Three kinds of leaves, each a pytree per name:
  per tick     (L, V, ...), V = shards x R views, written block by block from the
               rollout's GPU staging ring;
  per view     (V, ...), written once per segment (the initial carry, the last truth), or
               by the learner for a range of views (its labels, `write_views`);
  lookahead    (n, R, ...), the first n ticks of a name after the update's last shard (the
               lookahead's), each name with its own n: what the learner's targets read
               there and nothing more.

Writes go out on a copy stream, asynchronously: the rollout records an event after each
block and the copy waits on it; before the ring slots (or a segment buffer) are written
again, the compute stream waits on that copy. The update stages micro-batches of views
back the same way, gathering the next one into a pinned buffer and copying it while the
current one computes. Readers wait on the last write's event: the compute stream on the
GPU where the data stays on the device, the host only where it reads pinned memory itself.
On a CPU-only run every copy is a plain copy.
"""

import concurrent.futures
import contextlib

import torch
from torch.utils import _pytree


def _leaves(tree) -> list[torch.Tensor]:
    return [x for x in _pytree.tree_leaves(tree) if isinstance(x, torch.Tensor)]


class SegmentStore:
    """ticks: {name: pytree of tensors shaped (R, ...)}, an example of one tick's record;
    segments: {name: pytree (R, ...)}, an example of a per-segment record. Allocates
    `shards` segments of `length` ticks in pinned host memory (`host=True`, on a CUDA
    device) or on `device`. lookahead: {name of `ticks`: n}, the first n ticks of each name
    the lookahead keeps."""

    def __init__(self, ticks: dict, segments: dict, shards: int, length: int,
                 device: torch.device, host: bool, lookahead: dict[str, int] | None = None) -> None:
        self.device, self.shards, self.length = device, shards, length
        self.shard_views = _leaves(ticks)[0].shape[0]  # R
        self.views = shards * self.shard_views  # V, every shard's
        self.pinned = host and device.type == "cuda"
        where = torch.device("cpu") if self.pinned else device
        total = self.views

        def allocate(example: torch.Tensor, leading: tuple) -> torch.Tensor:
            return torch.empty(leading + tuple(example.shape[1:]), dtype=example.dtype,
                               device=where, pin_memory=self.pinned)

        self.specs, self.data = {}, {}
        for name, example in list(ticks.items()) + list(segments.items()):
            leaves, self.specs[name] = _pytree.tree_flatten(example)
            leading = (length, total) if name in ticks else (total,)
            self.data[name] = [allocate(leaf, leading) for leaf in leaves]
        self.per_tick = tuple(ticks)
        self.names = tuple(self.data)
        self.lookahead_ticks = dict(lookahead or {})
        self.lookahead_data = {
            name: [allocate(leaf, (count, self.shard_views)) for leaf in _leaves(ticks[name])]
            for name, count in self.lookahead_ticks.items()
        }
        self.copy_stream = torch.cuda.Stream(device) if device.type == "cuda" else None
        self._written: torch.cuda.Event | None = None  # the last write, on the copy stream
        self._staging: dict = {}
        self._worker = concurrent.futures.ThreadPoolExecutor(1) if self.pinned else None

    # ---------------------------------------------------------------- writing

    def _copy_context(self):
        return torch.cuda.stream(self.copy_stream) if self.copy_stream is not None \
            else contextlib.nullcontext()

    def write(self, source: dict, shard: int, start: int | None = None) -> torch.cuda.Event | None:
        """Copy {name: pytree} into shard `shard`: per-tick leaves (n, R, ...) at ticks
        start .. start + n, per-view leaves (R, ...). Asynchronous on a GPU: returns the
        event the compute stream must wait on before it writes `source` again."""
        views = slice(shard * self.shard_views, (shard + 1) * self.shard_views)
        return self._copy(source, views, start)

    def write_views(self, source: dict, views: slice) -> None:
        """Copy {name: pytree (n, ...)} into the per-view leaves of `views` (n of them),
        allocating a name's leaves (V, ...) on its first write. The sources may be
        temporaries: the copy stream keeps them until it has read them."""
        for name, tree in source.items():
            if name not in self.data:
                leaves, self.specs[name] = _pytree.tree_flatten(tree)
                where = torch.device("cpu") if self.pinned else self.device
                self.data[name] = [torch.empty((self.views,) + tuple(leaf.shape[1:]),
                                               dtype=leaf.dtype, device=where,
                                               pin_memory=self.pinned) for leaf in leaves]
            if self.copy_stream is not None:
                for leaf in _leaves(tree):
                    leaf.record_stream(self.copy_stream)
        self._copy(source, views, None)

    def write_lookahead(self, source: dict, start: int) -> torch.cuda.Event | None:
        """Copy the lookahead's names of {name: pytree (n, R, ...)}, ticks start .. start + n,
        into the lookahead's leaves, the ticks each keeps; asynchronous as `write`."""
        kept = {}
        for name, count in self.lookahead_ticks.items():
            ticks = min(count - start, _leaves(source[name])[0].shape[0])
            if ticks > 0:
                kept[name] = [leaf[:ticks] for leaf in _leaves(source[name])]
        ready = None
        if self.copy_stream is not None:
            ready = torch.cuda.Event()
            ready.record()
        with self._copy_context():
            if ready is not None:
                self.copy_stream.wait_event(ready)
            for name, leaves in kept.items():
                for stored, given in zip(self.lookahead_data[name], leaves, strict=True):
                    stored[start:start + given.shape[0]].copy_(given, non_blocking=True)
            if self.copy_stream is None:
                return None
            written = torch.cuda.Event()
            written.record(self.copy_stream)
        self._written = written
        return written

    def _copy(self, source: dict, views: slice, start: int | None) -> torch.cuda.Event | None:
        ready = None
        if self.copy_stream is not None:
            ready = torch.cuda.Event()
            ready.record()
        with self._copy_context():
            if ready is not None:
                self.copy_stream.wait_event(ready)
            for name, tree in source.items():
                for stored, given in zip(self.data[name], _leaves(tree), strict=True):
                    if name in self.per_tick:
                        target = stored[start:start + given.shape[0], views]
                        if target.is_contiguous():
                            target.copy_(given, non_blocking=True)
                        else:  # several shards: one copy per tick (each contiguous)
                            for tick in range(given.shape[0]):
                                target[tick].copy_(given[tick], non_blocking=True)
                    else:
                        stored[views].copy_(given, non_blocking=True)
            if self.copy_stream is None:
                return None
            written = torch.cuda.Event()
            written.record(self.copy_stream)
        self._written = written
        return written

    def synchronize(self) -> None:
        """The host waits until every write has landed (before it reads the store)."""
        if self._written is not None:
            self._written.synchronize()

    def _written_before_compute(self) -> None:
        """The compute stream waits on the GPU for every write so far (no host wait)."""
        if self._written is not None:
            torch.cuda.current_stream(self.device).wait_event(self._written)

    # ---------------------------------------------------------------- reading

    def shard(self, names: tuple[str, ...], shard: int) -> dict:
        """Whole-shard copies on the device of the per-tick leaves `names`, (L, R, ...)."""
        if self.pinned:  # a shard of several is a strided slice the host reads to copy it
            self.synchronize()
        else:
            self._written_before_compute()
        views = slice(shard * self.shard_views, (shard + 1) * self.shard_views)
        return {name: _pytree.tree_unflatten(
            [leaf[:, views].to(self.device, non_blocking=True) for leaf in self.data[name]],
            self.specs[name]) for name in names}

    def lookahead(self, names: tuple[str, ...]) -> dict:
        """Copies on the device of the lookahead's leaves `names`, (n, R, ...) each."""
        if self.pinned:
            self.synchronize()
        else:
            self._written_before_compute()
        return {name: _pytree.tree_unflatten(
            [leaf.to(self.device, non_blocking=True) for leaf in self.lookahead_data[name]],
            self.specs[name]) for name in names}

    def following(self, names: tuple[str, ...], views: torch.Tensor, ticks: int):
        """The first `ticks` ticks after each view's segment, on the device: ({name: pytree
        (ticks, n, ...)}, (n,) bool, whether they were played and kept). The segment after a
        view of shard s < shards - 1 is the same row's in shard s + 1; after the last shard it
        is the lookahead (which must keep `ticks` ticks of each name, or the views have none).
        views: int64 on the host. Ticks that were not kept are zeros."""
        if self.pinned:
            self.synchronize()
        else:
            self._written_before_compute()
        rows, last = self.shard_views, (self.shards - 1) * self.shard_views
        where = self.data[names[0]][0].device
        views = views.to(where)
        inside = views < last  # a later shard follows in the store
        kept = all(self.lookahead_ticks.get(name, 0) >= ticks for name in names)
        valid = inside | kept
        gathered = {}
        for name in names:
            leaves = []
            for position, leaf in enumerate(self.data[name]):
                stored = leaf[:ticks].index_select(1, torch.where(inside, views + rows, 0))
                if kept and not bool(inside.all()):
                    tail = self.lookahead_data[name][position][:ticks].index_select(
                        1, torch.where(inside, 0, views - last))
                    shape = (1, -1) + (1,) * (stored.dim() - 2)
                    stored = torch.where(inside.reshape(shape), stored, tail)
                elif not bool(valid.all()):
                    shape = (1, -1) + (1,) * (stored.dim() - 2)
                    stored = torch.where(inside.reshape(shape), stored, torch.zeros_like(stored))
                leaves.append(stored.to(self.device, non_blocking=True))
            gathered[name] = _pytree.tree_unflatten(leaves, self.specs[name])
        return gathered, valid.to(self.device, non_blocking=True)

    def _gather(self, names, views: torch.Tensor, out: dict | None) -> dict:
        gathered = {}
        for name in names:
            axis = 1 if name in self.per_tick else 0
            leaves = []
            for position, leaf in enumerate(self.data[name]):
                target = None if out is None else out[name][position]
                leaves.append(torch.index_select(leaf, axis, views, out=target))
            gathered[name] = leaves
        return gathered

    def _buffers(self, names, count: int, device, pinned: bool) -> dict:
        def allocate(leaf: torch.Tensor, name: str) -> torch.Tensor:
            shape = list(leaf.shape)
            shape[1 if name in self.per_tick else 0] = count
            return torch.empty(shape, dtype=leaf.dtype, device=device, pin_memory=pinned)

        return {name: [allocate(leaf, name) for leaf in self.data[name]] for name in names}

    def _unflatten(self, leaves: dict) -> dict:
        return {name: _pytree.tree_unflatten(value, self.specs[name])
                for name, value in leaves.items()}

    def batches(self, names: tuple[str, ...], view_batches: list[torch.Tensor]):
        """Yield {name: pytree} on the device for each list of views (int64, on the host).
        From pinned memory the next batch is gathered and copied while the caller computes
        on the current one; a yielded batch is valid until the next-but-one. Its staging
        buffers are released for reuse when the caller asks for the next batch (after the
        compute it queued on this one), so a caller runs the generator out: an abandoned one
        never releases its last batch."""
        if not self.pinned:
            self._written_before_compute()
            device = self.data[names[0]][0].device
            for views in view_batches:
                if views.device != device and device.type == "cuda":
                    views = views.pin_memory().to(device, non_blocking=True)
                yield self._unflatten(self._gather(names, views.to(device), None))
            return
        self.synchronize()  # the host gathers from pinned memory the copies wrote
        compute = torch.cuda.current_stream(self.device)
        # Every staging slot this call uses exists before the worker issues a copy: allocated
        # here, on the caller's thread and stream, and released only after. The caching
        # allocator may hand a new device buffer memory the compute stream freed with kernels
        # still queued on it; the first copy into the buffer runs on the copy stream and must
        # wait for them, so `released` is recorded after the allocation, never before (a slot
        # set up on the worker while the caller computes would miss the caller's frees).
        for views in view_batches:
            key = (names, views.numel())
            if key not in self._staging:
                pairs = [dict(host=self._buffers(names, key[1], "cpu", True),
                              device=self._buffers(names, key[1], self.device, False),
                              copied=torch.cuda.Event(), released=torch.cuda.Event())
                         for _ in range(2)]
                for pair in pairs:
                    pair["released"].record(compute)
                self._staging[key] = [pairs, 0]

        def slot(index: int) -> dict:
            """The staging slot of batch `index`: per micro-batch size, two pinned and device
            buffers that alternate, each with the copy that fills it and the compute that
            last read it."""
            key = (names, view_batches[index].numel())
            pairs, turn = self._staging[key]
            self._staging[key][1] = 1 - turn
            return pairs[turn]

        def issue(index: int) -> dict:
            buffers = slot(index)
            buffers["copied"].synchronize()  # the pinned buffer's last copy has finished
            self._gather(names, view_batches[index], buffers["host"])
            with torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_event(buffers["released"])
                for name in names:
                    for target, source in zip(buffers["device"][name], buffers["host"][name],
                                              strict=True):
                        target.copy_(source, non_blocking=True)
                buffers["copied"].record(self.copy_stream)
            return buffers

        # The next batch's gather and copy run on a worker thread (index_select and the copy
        # release the GIL), so they overlap the caller's kernel launches as well as the GPU.
        upcoming = self._worker.submit(issue, 0)
        for index in range(len(view_batches)):
            current = upcoming.result()
            compute.wait_event(current["copied"])
            if index + 1 < len(view_batches):  # its slot's last reader finished recording
                upcoming = self._worker.submit(issue, index + 1)
            yield self._unflatten(current["device"])
            current["released"].record(compute)
