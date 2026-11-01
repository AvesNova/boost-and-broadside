"""Where an update's segments live (learner spec 2): pinned host memory, or the device.

Two kinds of leaves, each a pytree per name:
  per tick     (L, V, ...), V = shards x R views, written block by block from the
               rollout's GPU staging ring;
  per segment  (V, ...), written once per segment (the initial carry, the last truth).

Writes go out on a copy stream, asynchronously: the rollout records an event after each
block and the copy waits on it; before the ring slots (or a segment buffer) are written
again, the compute stream waits on that copy. The update stages micro-batches of views
back the same way, gathering the next one into a pinned buffer and copying it while the
current one computes. On a CPU-only run every copy is a plain copy.
"""

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
        self._staging: dict = {}

    # ---------------------------------------------------------------- writing

    def _copy_context(self):
        return torch.cuda.stream(self.copy_stream) if self.copy_stream is not None \
            else contextlib.nullcontext()

    def write(self, source: dict, shard: int, start: int | None = None) -> torch.cuda.Event | None:
        """Copy {name: pytree} into shard `shard`: per-tick leaves (n, R, ...) at ticks
        start .. start + n, per-segment leaves (R, ...). Asynchronous on a GPU: returns the
        event the compute stream must wait on before it writes `source` again."""
        views = slice(shard * self.shard_views, (shard + 1) * self.shard_views)
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
        return written

    def synchronize(self) -> None:
        """Wait until every write has landed (before the host reads the store)."""
        if self.copy_stream is not None:
            self.copy_stream.synchronize()

    # ---------------------------------------------------------------- reading

    def shard(self, names: tuple[str, ...], shard: int) -> dict:
        """Whole-shard copies on the device of the per-tick leaves `names`, (L, R, ...)."""
        self.synchronize()
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
        """Yield {name: pytree} on the device for each list of views (int64, all of one
        size). From pinned memory the next batch is gathered and copied while the caller
        computes on the current one; a yielded batch is valid until the next-but-one."""
        self.synchronize()
        if not self.pinned:
            for views in view_batches:
                yield self._unflatten(self._gather(names, views.to(self.data[names[0]][0].device),
                                                   None))
            return
        count = view_batches[0].numel()
        key = (names, count)
        if key not in self._staging:
            self._staging[key] = [(self._buffers(names, count, "cpu", True),
                                   self._buffers(names, count, self.device, False))
                                  for _ in range(2)]
        staging = self._staging[key]
        copied = [torch.cuda.Event(), torch.cuda.Event()]
        released = [torch.cuda.Event(), torch.cuda.Event()]
        compute = torch.cuda.current_stream(self.device)
        for event in released:
            event.record(compute)

        def issue(index: int) -> None:
            host, device = staging[index % 2]
            copied[index % 2].synchronize()  # the pinned buffer's last copy has finished
            self._gather(names, view_batches[index].cpu(), host)
            with torch.cuda.stream(self.copy_stream):
                self.copy_stream.wait_event(released[index % 2])
                for name in names:
                    for target, source in zip(device[name], host[name], strict=True):
                        target.copy_(source, non_blocking=True)
                copied[index % 2].record(self.copy_stream)

        issue(0)
        for index in range(len(view_batches)):
            compute.wait_event(copied[index % 2])
            yield self._unflatten(staging[index % 2][1])
            released[index % 2].record(compute)
            if index + 1 < len(view_batches):
                issue(index + 1)
