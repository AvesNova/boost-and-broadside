"""Where a steady PPO update's learn time goes: the kernels of a few warm minibatch steps.

Builds the run's Trainer (`run`: a run directory with config.json), collects and updates
once (compiles and captures), collects again, then profiles `steps` minibatch steps of the
second update (each the minibatch's micro-batches, forward and backward, and the optimizer
step) after `skip` unprofiled ones, with the CUDA activity on. Prints the minibatch step's
wall time, the kernels' total and their share by kind (the name's family: Triton, GEMM,
attention, scan, ...) and the `top` kernels by GPU time, then one JSON line.

Usage: `python bench/learn_profile.py run=DIR [steps=2] [skip=2] [top=40] [trace=PATH]`
"""

import collections
import json
import os
import sys
import time

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")  # as the training entry point
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

# kernel name fragments by kind, the first match wins
KINDS = (
    ("attention", ("attention", "flash", "fmha", "sdpa")),
    ("scan", ("scan",)),
    ("gemm", ("gemm", "sm80_xmma", "sm90_xmma", "cutlass", "ampere_", "cublas", "_mm_")),
    ("optimizer", ("adam", "multi_tensor", "foreach")),
    ("triton reduction", ("triton_red", "triton_per")),
    ("triton pointwise", ("triton_poi",)),
    ("triton other", ("triton",)),
    ("copy", ("memcpy", "memset", "copy_", "CatArrayBatched")),
)


def kind(name: str) -> str:
    """A kernel's family from its name (`KINDS`)."""
    lowered = name.lower()
    for label, fragments in KINDS:
        if any(fragment in lowered for fragment in fragments):
            return label
    return "other"


def main() -> None:
    """Build, warm, profile; print the breakdown and one JSON line."""
    from bnb.cli import compile_limits
    from bnb.runs.serialise import from_json
    from bnb.runtime import metrics
    from bnb.runtime.train import TrainConfig, Trainer

    options = dict(a.split("=", 1) for a in sys.argv[1:])
    steps, skip, top = (int(options.get(k, d)) for k, d in (("steps", 2), ("skip", 2), ("top", 40)))
    compile_limits()
    with open(os.path.join(options["run"], "config.json")) as file:
        config, _ = from_json(TrainConfig, json.load(file))
    torch.manual_seed(0)
    trainer = Trainer(config, torch.device("cuda"))
    learner = trainer.ppo
    store, bootstrap = trainer.collect()
    metrics.read(learner.update(store, bootstrap, trainer.steps_per_update))
    store, bootstrap = trainer.collect()

    original_gradient, original_step = learner.gradient, learner._step
    calls = {"count": 0}
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    )
    walls: list[float] = []

    def gradient(*args, **kwargs):
        if calls["count"] == skip:
            torch.cuda.synchronize()
            profiler.start()
        if skip <= calls["count"] < skip + steps:
            torch.cuda.synchronize()
            calls["started"] = time.perf_counter()
        return original_gradient(*args, **kwargs)

    def step(finite):
        original_step(finite)
        if skip <= calls["count"] < skip + steps:
            torch.cuda.synchronize()
            walls.append(time.perf_counter() - calls["started"])
            if calls["count"] == skip + steps - 1:
                profiler.stop()
        calls["count"] += 1

    learner.gradient, learner._step = gradient, step
    started = time.perf_counter()
    metrics.read(learner.update(store, bootstrap, trainer.steps_per_update))
    update_seconds = time.perf_counter() - started
    learner.gradient, learner._step = original_gradient, original_step

    totals: dict[str, float] = collections.defaultdict(float)
    counts: dict[str, int] = collections.defaultdict(int)
    for event in profiler.events():
        if event.device_type == torch.autograd.DeviceType.CUDA:
            totals[event.name] += event.device_time_total / 1e3  # ms
            counts[event.name] += 1
    if options.get("trace"):
        profiler.export_chrome_trace(options["trace"])
    kernel_ms = sum(totals.values()) / steps
    by_kind: dict[str, float] = collections.defaultdict(float)
    for name, ms in totals.items():
        by_kind[kind(name)] += ms / steps
    wall_ms = 1e3 * sum(walls) / len(walls)
    print(f"update {update_seconds:.1f} s, {calls['count']} minibatch steps")
    print(f"minibatch step: wall {wall_ms:.0f} ms, kernels {kernel_ms:.0f} ms")
    for label, ms in sorted(by_kind.items(), key=lambda item: -item[1]):
        print(f"  {label:18s} {ms:8.1f} ms  {100 * ms / kernel_ms:5.1f}%")
    ranked = sorted(totals.items(), key=lambda item: -item[1])[:top]
    for name, ms in ranked:
        print(f"{ms / steps:8.2f} ms {counts[name] // steps:6d}x  {kind(name):16s} {name[:110]}")
    print(
        json.dumps(
            dict(
                run=options["run"],
                update_s=update_seconds,
                minibatch_steps=calls["count"],
                step_wall_ms=wall_ms,
                step_kernel_ms=kernel_ms,
                by_kind=dict(by_kind),
                top=[dict(name=n, ms=ms / steps, calls=counts[n] // steps) for n, ms in ranked],
            )
        )
    )


if __name__ == "__main__":
    main()
