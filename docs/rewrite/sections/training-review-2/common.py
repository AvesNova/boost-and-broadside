"""Shared helpers of the review's measurement scripts (queue item 4): the per-variant compile
cache, the machine's state, a patched `lean.learn.compiled.graphs` (a compile mode per graph),
the update's store, and a Chrome-trace reader (kernel families, GPU busy share, host calls).

Import this module first: it sets TORCHINDUCTOR_CACHE_DIR before torch is imported.
"""

import json
import os
import subprocess
import sys

DATA = "/home/vizia/avesnova/boost-and-broadside/data/review4"


def cache_dir(name: str) -> str:
    """This variant's own inductor and triton cache under the data directory (kept between a
    variant's cold and warm run; deleted by the runner when the grid is done)."""
    path = os.path.join(DATA, "cache", name)
    os.makedirs(path, exist_ok=True)
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = path
    os.environ["TRITON_CACHE_DIR"] = os.path.join(path, "triton")
    return path


def machine() -> dict:
    """The power profile, the load average and the other bnb units at the job's start."""
    profile = subprocess.run(["powerprofilesctl", "get"], capture_output=True, text=True).stdout
    units = subprocess.run(
        ["systemctl", "--user", "list-units", "bnb-*", "--no-legend", "--plain"],
        capture_output=True,
        text=True,
    ).stdout.split("\n")
    return {
        "power": profile.strip(),
        "load": os.getloadavg()[0],
        "bnb_units": [line.split()[0] for line in units if line.strip()],
    }


def emit(record: dict, path: str | None = None) -> None:
    line = json.dumps(record)
    print(line, flush=True)
    if path is not None:
        with open(path, "a") as handle:
            handle.write(line + "\n")


def patch_graphs(modes: dict[str, str]) -> None:
    """lean.learn.compiled.graphs with `modes` {graph name: torch.compile mode} (static
    shapes, like StaticGraph, without its recompile tripwire); the other graphs as the
    trainer compiles them."""
    import functools
    from types import SimpleNamespace

    import torch

    from lean.contract import StaticGraph
    from lean.learn import compiled, rollout, training

    functions = {
        "act": rollout.act,
        "advance": rollout.advance,
        "slot_actions": rollout.slot_actions,
        "merge_league": rollout.merge_league,
        "minibatch_values": training.minibatch_values,
        "minibatch_loss": training.minibatch_loss,
    }

    @functools.cache
    def graphs(shapes: str = "static"):
        built = {}
        for name, function in functions.items():
            mode = modes.get(name, "default")
            if mode == "default":
                built[name] = StaticGraph(function)
            else:
                built[name] = torch.compile(function, dynamic=False, mode=mode)
        return SimpleNamespace(**built)

    compiled.graphs = graphs


FAMILIES = (
    ("adam/optimizer", ("adam", "multi_tensor", "foreach")),
    ("memcpy/memset", ("memcpy", "memset")),
    ("attention (SDPA)", ("fmha", "attention", "flash", "efficient")),
    ("GEMM", ("gemm", "xmma", "cutlass", "nvjet", "cublas", "splitk", "triton_tem", "sm80_", "sm89_", "s16816", "s1688", "_mm_", "bmm")),
    ("linear scan", ("linear_scan", "scan_forward", "scan_backward", "_scan", "_forward_kernel", "_backward_kernel")),
    ("RMSNorm", ("rsqrt", "rms_norm")),
    ("losses (log-softmax, HL-Gauss, erf)", ("log_softmax", "logsumexp", "erf", "_softmax")),
    ("topk/sort/gather", ("topk", "sort", "gather", "scatter", "index")),
    ("GELU / gated MLP", ("gelu",)),
    ("sigmoid gates", ("sigmoid",)),
    ("rotary / position (cos, sin, atan2)", ("cos", "sin", "atan2")),
)


def family(name: str) -> str:
    low = name.lower()
    for label, keys in FAMILIES:
        if any(key in low for key in keys):
            return label
    return "other pointwise / reductions"


def read_trace(path: str) -> dict:
    """Kernel families, GPU busy share, host runtime calls of a Chrome trace."""
    with open(path) as handle:
        trace = json.load(handle)
    events = trace["traceEvents"] if isinstance(trace, dict) else trace
    kernels = [e for e in events if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
    runtime = [e for e in events if e.get("cat") in ("cuda_runtime", "cuda_driver") and "dur" in e]
    regions = [
        e
        for e in events
        if e.get("cat") == "cpu_op" and e.get("name", "").startswith("Torch-Compiled Region")
    ]
    if not kernels:
        return {}
    start = min(e["ts"] for e in kernels)
    end = max(e["ts"] + e["dur"] for e in kernels)
    intervals = sorted((e["ts"], e["ts"] + e["dur"]) for e in kernels)
    busy, current_start, current_end = 0.0, *intervals[0]
    for low, high in intervals[1:]:
        if low > current_end:
            busy += current_end - current_start
            current_start, current_end = low, high
        else:
            current_end = max(current_end, high)
    busy += current_end - current_start
    by_family, by_name = {}, {}
    for e in kernels:
        by_family[family(e["name"])] = by_family.get(family(e["name"]), 0.0) + e["dur"]
        entry = by_name.setdefault(e["name"], [0, 0.0])
        entry[0] += 1
        entry[1] += e["dur"]
    calls = {}
    for e in runtime:
        entry = calls.setdefault(e["name"], [0, 0.0])
        entry[0] += 1
        entry[1] += e["dur"]
    memcpy = {}
    for e in kernels:
        if e.get("cat") == "gpu_memcpy":
            kind = e.get("args", {}).get("kind", e["name"])
            entry = memcpy.setdefault(str(kind) + " " + e["name"][:40], [0, 0.0, 0])
            entry[0] += 1
            entry[1] += e["dur"]
            entry[2] += int(e.get("args", {}).get("bytes", 0) or 0)
    kernel_total = sum(e["dur"] for e in kernels)
    top = sorted(by_name.items(), key=lambda item: -item[1][1])[:40]
    return {
        "window_ms": (end - start) / 1000,
        "busy_ms": busy / 1000,
        "busy_share": busy / (end - start),
        "kernel_ms": kernel_total / 1000,
        "kernels": len(kernels),
        "families_ms": {k: round(v / 1000, 3) for k, v in sorted(by_family.items(), key=lambda kv: -kv[1])},
        "top_kernels": [(name[:110], count, round(dur / 1000, 3)) for name, (count, dur) in top],
        "runtime_calls": {k: (v[0], round(v[1] / 1000, 3)) for k, v in sorted(calls.items(), key=lambda kv: -kv[1][1])},
        "memcpy": {k: (v[0], round(v[1] / 1000, 3), v[2]) for k, v in memcpy.items()},
        "compiled_regions": [(e["name"], round(e["dur"] / 1000, 3), e["ts"]) for e in regions],
    }


def say(*parts) -> None:
    print(*parts, file=sys.stderr, flush=True)
