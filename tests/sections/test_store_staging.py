"""The segment store's staging (learner spec 2): a micro-batch read back from pinned memory is
exactly what the store holds, however the caller's compute stream is using and freeing memory
while the store sets up its staging buffers."""

import threading

import pytest
import torch

from bnb.runtime.store import SegmentStore

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]
CUDA = torch.device("cuda") if torch.cuda.is_available() else None
LENGTH, ROWS, WIDTH = 16, 12, 4096
IN_FLIGHT = 200_000_000  # GPU cycles the caller's queued kernel waits: ~0.1 s, far past a copy


def _store() -> tuple[SegmentStore, torch.Tensor]:
    """A pinned store of one per-tick leaf (L, V, WIDTH) int32, filled with joint-action-like
    values in [0, 42)."""
    example = {"action": torch.zeros(ROWS, WIDTH, dtype=torch.int32, device=CUDA)}
    store = SegmentStore(example, {}, 1, LENGTH, CUDA, host=True)
    held = torch.arange(LENGTH * ROWS * WIDTH, dtype=torch.int32).reshape(LENGTH, ROWS, WIDTH)
    held %= 42
    store.data["action"][0].copy_(held)
    return store, held


def test_a_new_staging_buffer_is_filled_after_the_compute_that_last_used_its_memory(monkeypatch):
    """While the caller computes on one micro-batch, the store sets up the next one's staging
    buffers (a new micro-batch size) on its worker thread. Meanwhile the caller frees a
    temporary that a kernel still queued on the compute stream writes; the caching allocator
    may hand that memory to the new staging buffer at once (it is stream-ordered on the
    compute stream). The copy into the buffer runs on the copy stream, so it must wait for
    every compute kernel queued before the buffer was allocated; if it waits only for those
    queued before an earlier point, the queued kernel can land after the copy and overwrite
    the batch (the learner then reads actions out of range).

    The interleaving is forced: the caller's free happens on the caller's thread just before
    the store allocates the device buffer (on whichever thread allocates it)."""
    store, held = _store()
    first, second = torch.arange(1), torch.arange(ROWS)
    shape = (LENGTH, second.numel(), WIDTH)
    for _ in store.batches(("action",), [first]):  # the copy stream and the first size in use
        pass
    # As in a running update: the allocators already hold the memory and the kernels are
    # loaded, so nothing below waits for the device (cudaMalloc, cudaHostAlloc and a lazy
    # module load can, and would hide the race).
    cached = [torch.empty(shape, dtype=torch.int32, device=CUDA) for _ in range(4)]
    cached += [torch.empty(shape, dtype=torch.int32, pin_memory=True) for _ in range(2)]
    torch.cuda._sleep(1)
    cached[0].fill_(-1)
    torch.cuda.synchronize()
    del cached

    def caller_compute():
        """A temporary the size of the staging buffer, written by a kernel that is still
        queued when the temporary is freed."""
        temporary = torch.empty(shape, dtype=torch.int32, device=CUDA)
        torch.cuda._sleep(IN_FLIGHT)
        temporary.fill_(-1)
        del temporary

    in_window, computed = threading.Event(), threading.Event()
    allocate = SegmentStore._buffers

    def allocate_after_the_caller_computes(self, names, count, device, pinned):
        if not pinned and count == second.numel() and not computed.is_set():
            if threading.current_thread() is threading.main_thread():
                caller_compute()
                computed.set()
            else:  # the worker thread: let the caller compute now
                in_window.set()
                assert computed.wait(10.0)
        return allocate(self, names, count, device, pinned)

    monkeypatch.setattr(SegmentStore, "_buffers", allocate_after_the_caller_computes)
    batches = store.batches(("action",), [first, second])
    next(batches)  # the caller's micro-batch; the store prepares the next one meanwhile
    if in_window.wait(2.0):
        caller_compute()
        computed.set()
    assert computed.is_set()
    batch = next(batches)
    assert torch.equal(batch["action"].cpu(), held[:, second])
    batches.close()


def test_micro_batches_of_several_sizes_read_back_what_the_store_holds():
    """Two micro-batch sizes, each with its own alternating staging buffers, over two passes."""
    store, held = _store()
    order = list(torch.randperm(ROWS, generator=torch.Generator().manual_seed(0)).split(5))
    for _ in range(2):
        for views, batch in zip(order, store.batches(("action",), order), strict=True):
            assert torch.equal(batch["action"].cpu(), held[:, views])
