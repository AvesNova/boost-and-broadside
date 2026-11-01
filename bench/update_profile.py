"""Profile the learner's update at a training cell; prints `RESULT {json}`.

    uv run --no-sync python -m bench.update_profile '{"preset": "deathmatch", "envs": 256}'

Builds the trainer as `bench.cell` does, runs one update (compilation), collects a fresh
store, then:
  stages: per-stage GPU time of one update from CUDA events on the compute stream (the
          shard returns and shot labels, the drift, per micro-batch the staging wait, the
          label codes, the loss forward, the backward, and per minibatch the gradient clip
          and the optimizer step), against its wall time;
  trace:  one update under torch.profiler with CUDA activity: kernel time by category, the
          top kernels, the GPU's busy share of the wall time, and the compile counters.
Options: "precision" (a TrainConfig precision), "learner" (LearnerConfig overrides),
"compile" (default true), "trace_path" (a Chrome trace to write). Takes the benchmark lock.
"""

import collections
import contextlib
import json
import os
import sys
import time

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

import torch  # noqa: E402

from bench.lock import bench_lock  # noqa: E402


def build(cell: dict):
    from bnb.game.modes.rules import PRESETS
    from bnb.learn.ppo import LearnerConfig
    from bnb.runtime.train import TrainConfig, Trainer, mixed, set_precision

    torch.manual_seed(0)
    precision = cell.get("precision", "float32")
    set_precision(precision)
    learner = LearnerConfig(**cell.get("learner", {}))
    options = dict(num_envs=cell["envs"], learner=learner, precision=precision,
                   compile=cell.get("compile", True))
    if cell["preset"] == "mixed":
        game, pools = mixed()
        cfg = TrainConfig(game=game, pools=pools, **options)
    else:
        cfg = TrainConfig(game=PRESETS[cell["preset"]], **options)
    trainer = Trainer(cfg, torch.device("cuda"))
    t0 = time.perf_counter()
    trainer.update()
    torch.cuda.synchronize()
    return trainer, time.perf_counter() - t0


class Events:
    """Named CUDA-event intervals on the compute stream, summed per name."""

    def __init__(self) -> None:
        self.marks: list[tuple[str, torch.cuda.Event, torch.cuda.Event]] = []

    @contextlib.contextmanager
    def span(self, name: str):
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        yield
        end.record()
        self.marks.append((name, start, end))

    def totals(self) -> dict:
        torch.cuda.synchronize()
        out = collections.defaultdict(float)
        for name, start, end in self.marks:
            out[name] += start.elapsed_time(end) / 1e3
        return dict(out)


def instrument(ppo, events: Events) -> None:
    """Wrap the learner's stages in event spans (instance attributes shadow the methods)."""
    learner_class = type(ppo)

    def wrap(name: str, function):
        def wrapped(*args, **kwargs):
            with events.span(name):
                return function(*args, **kwargs)

        return wrapped

    ppo.prepare = wrap("prepare (returns, shot labels)", learner_class.prepare.__get__(ppo))
    ppo.drift = wrap("drift", learner_class.drift.__get__(ppo))
    ppo.labels = wrap("label codes", learner_class.labels.__get__(ppo))
    loss = ppo.loss

    def forward(*args):
        with events.span("loss forward"):
            out = loss(*args)
        # The backward follows at once in `gradient`: its span ends at the next stage.
        start = torch.cuda.Event(enable_timing=True)
        start.record()
        events.pending = start
        return out

    ppo.loss = forward
    staged = learner_class.staged.__get__(ppo)

    def staging(segments, view_batches):
        iterator = staged(segments, view_batches)
        while True:
            pending = getattr(events, "pending", None)
            if pending is not None:  # the previous micro-batch's backward ends here
                end = torch.cuda.Event(enable_timing=True)
                end.record()
                events.marks.append(("backward", pending, end))
                events.pending = None
            with events.span("staging wait"):
                try:
                    item = next(iterator)
                except StopIteration:
                    return
            yield item

    ppo.staged = staging
    clip = torch.nn.utils.clip_grad_norm_

    def clipped(*args, **kwargs):
        with events.span("gradient clip"):
            return clip(*args, **kwargs)

    torch.nn.utils.clip_grad_norm_ = clipped
    step = ppo.optimizer.step

    def stepped(*args, **kwargs):
        with events.span("optimizer step"):
            return step(*args, **kwargs)

    ppo.optimizer.step = stepped


def stages(trainer) -> dict:
    store, bootstraps = trainer.collect()
    torch.cuda.synchronize()
    events = Events()
    instrument(trainer.ppo, events)
    with events.span("update"):
        t0 = time.perf_counter()
        stats = trainer.ppo.update(store, bootstraps)
    torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    totals = events.totals()
    accounted = sum(value for name, value in totals.items() if name != "update")
    return dict(
        wall_s=wall,
        gpu_update_s=totals.pop("update"),
        stages_s=totals,
        unaccounted_s=wall - accounted,
        minibatch_steps=stats["minibatch_steps"],
        losses={k: v for k, v in stats.items() if k.startswith("loss/")},
    )


CATEGORIES = (
    ("triton fused", ("triton",)),
    ("attention", ("flash", "fmha", "attention", "efficient")),
    (
        "matmul",
        (
            "gemm",
            "sgemm",
            "cutlass",
            "xmma",
            "cublas",
            "matmul",
            "splitk",
            "s16816",
            "s1688",
            "wgrad",
            "dgrad",
        ),
    ),
    ("optimizer", ("adam", "multi_tensor", "foreach", "norm_kernel", "_amp_")),
    ("copy", ("memcpy", "memset", "copy", "cat", "index_select", "gather", "index")),
)


def category(name: str) -> str:
    lowered = name.lower()
    for label, keys in CATEGORIES:
        if any(key in lowered for key in keys):
            return label
    return "other eager"


def trace(trainer, trace_path: str | None) -> dict:
    from torch._dynamo.utils import counters
    from torch.profiler import ProfilerActivity, profile

    store, bootstraps = trainer.collect()
    torch.cuda.synchronize()
    counters.clear()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        t0 = time.perf_counter()
        trainer.ppo.update(store, bootstraps)
        torch.cuda.synchronize()
        wall = time.perf_counter() - t0
    if trace_path:
        profiler.export_chrome_trace(trace_path)
    kernels = [
        event for event in profiler.events() if event.device_type == torch.autograd.DeviceType.CUDA
    ]
    by_category, by_name = collections.defaultdict(float), collections.defaultdict(float)
    launches = collections.Counter()
    intervals = []
    for event in kernels:
        seconds = (
            event.device_time / 1e6 if hasattr(event, "device_time") else event.cuda_time / 1e6
        )
        by_category[category(event.name)] += seconds
        by_name[event.name[:120]] += seconds
        launches[category(event.name)] += 1
        intervals.append((event.time_range.start, event.time_range.end))
    intervals.sort()
    busy, end = 0.0, -1.0
    for start, stop in intervals:  # union of kernel intervals (us)
        if start > end:
            busy += stop - start
            end = stop
        elif stop > end:
            busy += stop - end
            end = stop
    top = sorted(by_name.items(), key=lambda item: -item[1])[:30]
    host = profiler.key_averages()
    host_top = sorted(
        ((e.key, e.cpu_time_total / 1e6, e.count) for e in host), key=lambda item: -item[1]
    )[:25]
    return dict(
        wall_s=wall,
        gpu_busy_s=busy / 1e6,
        kernels=len(kernels),
        kernel_s_by_category=dict(by_category),
        launches_by_category=dict(launches),
        top_kernels=top,
        host_top=host_top,
        dynamo_counters={k: dict(v) for k, v in counters.items()},
    )


def main() -> None:
    cell = json.loads(sys.argv[1])
    with bench_lock():
        torch.cuda.reset_peak_memory_stats()
        trainer, compile_s = build(cell)
        result = dict(
            cell=cell,
            compile_s=compile_s,
            views=trainer.rollout.store.views,
            micro_views=trainer.ppo.micro_views,
        )
        if cell.get("mode", "stages") == "stages":
            result.update(stages(trainer))
        else:
            result.update(trace(trainer, cell.get("trace_path")))
        result["peak_vram_gb"] = torch.cuda.max_memory_allocated() / 2**30
    print("RESULT " + json.dumps(result, default=str), flush=True)


if __name__ == "__main__":
    main()
