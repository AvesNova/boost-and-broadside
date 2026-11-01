"""Where an update's segments live (learner spec 2): pinned host memory, or the device.

Two kinds of leaves, each a pytree per name:
  per tick     (L, V, ...), V = shards x R views, written block by block from the
               rollout's GPU staging ring;
  per view     (V, ...), written once per segment (the initial carry, the last truth), or
               by the learner for a range of views (its labels, `write_views`).

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
    device) or on `device`."""

    def __init__(self, ticks: dict, segments: dict, shards: int, length: int,
                 device: torch.device, host: bool) -> None:
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
        on the current one; a yielded batch is valid until the next-but-one."""
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

        def slot(index: int) -> dict:
            """The staging slot of batch `index`: per micro-batch size, two pinned and device
            buffers that alternate, each with the copy that fills it and the compute that
            last read it."""
            key = (names, view_batches[index].numel())
            if key not in self._staging:
                pairs = []
                for _ in range(2):
                    released = torch.cuda.Event()
                    released.record(compute)
                    pairs.append(dict(host=self._buffers(names, key[1], "cpu", True),
                                      device=self._buffers(names, key[1], self.device, False),
                                      copied=torch.cuda.Event(), released=released))
                self._staging[key] = [pairs, 0]
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
