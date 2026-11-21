"""Count the aten ops eager PyTorch dispatches, attributed to named regions.

Every op that reaches the dispatcher is counted once. Ops that only make a new view of
existing memory (OpOverload.is_view: view, reshape-as-view, select, slice, permute, ...)
launch no GPU kernel, so they are kept apart from the compute ops, which do (one or more
kernels each on a GPU). Counts do not depend on the device or the batch size, except where
code loops in Python over a size.

A region is pushed while a wrapped function runs; an op counts toward the innermost region
(exclusive) and toward every region on the stack (inclusive). `fused` wraps a function that
runs as one custom kernel on the GPU: its plain twin runs here for the values, its ops are
not counted, and one compute op is counted in its place.
"""

import collections
import functools

import torch
from torch.utils._python_dispatch import TorchDispatchMode

STACK: list[str] = ["(outside any region)"]
EXCLUSIVE = collections.defaultdict(lambda: [0, 0])  # label -> [compute, view]
INCLUSIVE = collections.defaultdict(lambda: [0, 0])
OPS = collections.defaultdict(collections.Counter)  # label -> op name -> count (compute)
CALLS = collections.Counter()  # label -> times entered
_SUPPRESS = [0]

METADATA = {"aten.sym_size.int", "aten.sym_stride.int", "aten.sym_numel.default",
            "aten.sym_storage_offset.default", "aten.is_contiguous.default",
            "aten.lift_fresh.default", "aten._local_scalar_dense.default"}


class Counter(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        name = str(func)
        if not _SUPPRESS[0] and name not in METADATA:
            kind = 1 if getattr(func, "is_view", False) else 0
            EXCLUSIVE[STACK[-1]][kind] += 1
            for label in set(STACK):
                INCLUSIVE[label][kind] += 1
            if kind == 0:
                OPS[STACK[-1]][name] += 1
        return func(*args, **(kwargs or {}))


def reset() -> None:
    EXCLUSIVE.clear(); INCLUSIVE.clear(); OPS.clear(); CALLS.clear()


def region(owner, attribute: str, label: str) -> None:
    """Wrap owner.attribute (a module-level function or a class's method) as a region."""
    original = getattr(owner, attribute)
    if isinstance(original, staticmethod):
        original = original.__func__

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        STACK.append(label)
        CALLS[label] += 1
        try:
            return original(*args, **kwargs)
        finally:
            STACK.pop()

    if isinstance(owner.__dict__.get(attribute) if hasattr(owner, "__dict__") else None, staticmethod):
        wrapped = staticmethod(wrapped)
    setattr(owner, attribute, wrapped)


def fused(label: str, plain):
    """A stand-in for a GPU kernel entry point: runs `plain` uncounted, counts one op."""

    def wrapped(*args, **kwargs):
        _SUPPRESS[0] += 1
        try:
            out = plain(*args, **kwargs)
        finally:
            _SUPPRESS[0] -= 1
        EXCLUSIVE[STACK[-1]][0] += 1
        for item in set(STACK):
            INCLUSIVE[item][0] += 1
        OPS[STACK[-1]][f"[fused kernel] {label}"] += 1
        return out

    return wrapped


def report(title: str, order: list[str] | None = None, top: int = 0) -> None:
    total_c = sum(c for c, _ in EXCLUSIVE.values())
    total_v = sum(v for _, v in EXCLUSIVE.values())
    print(f"\n== {title}: {total_c} compute ops (+{total_v} view ops) ==")
    labels = order or sorted(EXCLUSIVE, key=lambda k: -EXCLUSIVE[k][0])
    for label in labels:
        if label not in EXCLUSIVE and label not in INCLUSIVE:
            continue
        c, v = EXCLUSIVE.get(label, (0, 0))
        ic, _ = INCLUSIVE.get(label, (0, 0))
        print(f"  {label:<55} exclusive {c:>6}  inclusive {ic:>6}  ({100 * c / max(total_c, 1):4.1f}%)")
        if top:
            for name, n in OPS[label].most_common(top):
                print(f"      {n:>5}  {name}")
