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
BYTES = collections.Counter()  # label -> bytes read and written by compute ops (eager, unfused)
BYTES_EX = collections.Counter()  # the same, by innermost region


def tensor_bytes(tree) -> int:
    total = 0
    for leaf in torch.utils._pytree.tree_leaves(tree):
        if isinstance(leaf, torch.Tensor):
            total += leaf.numel() * leaf.element_size()
    return total
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
        out = func(*args, **(kwargs or {}))
        if not _SUPPRESS[0] and name not in METADATA and not getattr(func, "is_view", False):
            moved = tensor_bytes((args, kwargs)) + tensor_bytes(out)
            BYTES_EX[STACK[-1]] += moved
            for label in set(STACK):
                BYTES[label] += moved
        return out


def reset() -> None:
    EXCLUSIVE.clear(); INCLUSIVE.clear(); OPS.clear(); CALLS.clear(); BYTES.clear(); BYTES_EX.clear()


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
        moved = tensor_bytes((args, kwargs)) + tensor_bytes(out)
        BYTES_EX[STACK[-1]] += moved
        for item in set(STACK):
            INCLUSIVE[item][0] += 1
            BYTES[item] += moved
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


def _count_one(label: str) -> None:
    EXCLUSIVE[STACK[-1]][0] += 1
    for item in set(STACK):
        INCLUSIVE[item][0] += 1
    OPS[STACK[-1]][f"[fused kernel] {label}"] += 1


def fused_autograd(label: str, plain):
    """A GPU kernel with its own backward: forward and backward each count as one op; the plain
    twin computes the values (and, recomputed, the gradients) uncounted. Gradients reach the
    tensor arguments and the parameters of module arguments."""

    simple = fused(label, plain)

    def wrapped(*args, **kwargs):
        if not torch.is_grad_enabled():
            return simple(*args, **kwargs)
        device = next(a for a in list(args) if isinstance(a, torch.Tensor)).device.type
        autocast = (torch.is_autocast_enabled(device), torch.get_autocast_dtype(device))
        modules = [a for a in list(args) + list(kwargs.values()) if isinstance(a, torch.nn.Module)]
        params = [p for m in modules for p in m.parameters()]
        tuples = [t for a in args if isinstance(a, tuple) for t in a if isinstance(t, torch.Tensor)]
        tensor_positions = [i for i, a in enumerate(args) if isinstance(a, torch.Tensor)]
        inputs = [args[i] for i in tensor_positions] + tuples + params

        class Kernel(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *flat):
                _SUPPRESS[0] += 1
                try:
                    out = plain(*args, **kwargs)
                finally:
                    _SUPPRESS[0] -= 1
                _count_one(label)
                BYTES_EX[STACK[-1]] += tensor_bytes(flat) + tensor_bytes(out)
                for item in set(STACK):
                    BYTES[item] += tensor_bytes(flat) + tensor_bytes(out)
                ctx.single = isinstance(out, torch.Tensor)
                outs = (out,) if ctx.single else tuple(out)
                return tuple(o.detach() for o in outs)

            @staticmethod
            def backward(ctx, *grads):
                _SUPPRESS[0] += 1
                try:
                    with torch.enable_grad(), torch.autocast(device, autocast[1], enabled=autocast[0]):
                        copies = list(args)
                        need = []
                        for i in tensor_positions:
                            t = args[i].detach()
                            if args[i].requires_grad:
                                t.requires_grad_(True)
                            copies[i] = t
                            need.append(t)
                        out = plain(*copies, **kwargs)
                        outs = (out,) if isinstance(out, torch.Tensor) else tuple(out)
                        wrt = [t for t in need + tuples + params if t.requires_grad]
                        pairs = [(o, g) for o, g in zip(outs, grads) if g is not None and o.requires_grad]
                        got = torch.autograd.grad([o for o, _ in pairs], wrt, [g for _, g in pairs], allow_unused=True) if pairs and wrt else []
                finally:
                    _SUPPRESS[0] -= 1
                _count_one(label + " (backward)")
                BYTES_EX[STACK[-1]] += 2 * tensor_bytes(grads) + tensor_bytes(got)
                for item in set(STACK):
                    BYTES[item] += 2 * tensor_bytes(grads) + tensor_bytes(got)
                by_id = {id(t): g for t, g in zip(wrt, got)}
                result = []
                for k, i in enumerate(tensor_positions):
                    result.append(by_id.get(id(need[k])) if args[i].requires_grad else None)
                for t in tuples + params:
                    g = by_id.get(id(t))
                    if g is not None and t.is_leaf and t.requires_grad:
                        t.grad = g if t.grad is None else t.grad + g
                        g = None
                    result.append(g)
                return tuple(result)

        outs = Kernel.apply(*inputs)
        return outs[0] if len(outs) == 1 and not isinstance(plain, tuple) and _single(outs) else outs

    return wrapped


def _single(outs) -> bool:
    return len(outs) == 1
