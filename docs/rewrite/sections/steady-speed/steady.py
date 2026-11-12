"""Steady-state training speed of one tree x stage: `python steady.py config=FILE updates=N
[timed=1] [set=path=value;...] [memory_fraction=F] [process_start=EPOCH]`, one JSON object
to stdout.

Runs the trainer's own update (`Trainer.update`: collect, analytics, learn, the cadence's
diagnostics and heads' sums on its updates), so the steady numbers are a run's, not the
compile bench's (`bench/compile_baseline.py` calls the learner's update with the heads'
sums on every update and no cadence). timed=1 wraps every compiled function with the
compile bench's timer (device syncs around each call: per-unit compile times, for a cold
run); timed=0 (the default) leaves the compiled path as a run has it. The heads' sums are
timed on their calls (device synchronised; they run on cadence updates only). The tree under
test is the one on PYTHONPATH; this file is the same in every tree.
"""

import dataclasses
import json
import os
import sys
import time

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

OPTIONS = dict(a.split("=", 1) for a in sys.argv[1:])
PROCESS_START = float(OPTIONS.get("process_start", time.time()))

import torch  # noqa: E402

EVENTS: list[dict] = []


def since_start() -> float:
    return time.time() - PROCESS_START


def log(message: str) -> None:
    print(f"[{since_start():7.1f} s] {message}", file=sys.stderr, flush=True)


def synchronize() -> None:
    if not torch.cuda.is_current_stream_capturing():
        torch.cuda.synchronize()


if OPTIONS.get("timed") == "1":  # per-unit compile times (the compile bench's wrapper)
    import functools

    import torch._dynamo.utils as dynamo_utils

    _compile = torch.compile

    def _timed_compile(function=None, **options):
        if function is None:
            return lambda f: _timed_compile(f, **options)
        compiled = _compile(function, **options)
        label = getattr(function, "__qualname__", repr(function))

        @functools.wraps(function)
        def call(*arguments, **keywords):
            if torch.cuda.is_current_stream_capturing():
                return compiled(*arguments, **keywords)
            stats = dynamo_utils.counters["stats"]
            frames, ops = stats.get("unique_graphs", 0), stats.get("calls_captured", 0)
            synchronize()
            started = time.perf_counter()
            result = compiled(*arguments, **keywords)
            synchronize()
            if stats.get("unique_graphs", 0) > frames:
                seconds = time.perf_counter() - started
                added = stats.get("calls_captured", 0) - ops
                EVENTS.append(dict(kind="compile", label=label, seconds=seconds, ops=added))
                log(f"compiled {label}: {seconds:.1f} s, {added} ops")
            return result

        return call

    torch.compile = _timed_compile


def main() -> None:
    from bnb.runs.serialise import from_json
    from bnb.runtime.train import TrainConfig, Trainer

    with open(OPTIONS["config"]) as file:
        data = json.load(file)
    for setting in filter(None, OPTIONS.get("set", "").split(";")):
        name, value = setting.split("=", 1)
        *parents, leaf = name.split(".")
        node = data
        for parent in parents:
            node = node[parent]
        if leaf not in node:
            raise SystemExit(f"set {name}: the config has no such field")
        node[leaf] = json.loads(value)
    updates = int(OPTIONS.get("updates", 12))
    config, filled = from_json(TrainConfig, data)
    config = dataclasses.replace(config, updates=max(config.updates, updates))
    torch.manual_seed(0)
    device = torch.device("cuda")
    if "memory_fraction" in OPTIONS:  # emulate a smaller GPU (e.g. 8 GB: 8/16.3 here)
        torch.cuda.set_per_process_memory_fraction(float(OPTIONS["memory_fraction"]), 0)
    log("build ...")
    started = time.perf_counter()
    trainer = Trainer(config, device)
    synchronize()
    build_seconds = time.perf_counter() - started
    log(f"build: {build_seconds:.1f} s")
    note = None
    try:
        from bnb.runs.session import micro_batch_note

        note = micro_batch_note(trainer)
        log(note)
    except ImportError:
        pass
    sums = trainer.ppo.head_sums
    head_sums: list[dict] = []

    def timed_sums(*arguments):
        synchronize()
        begun = time.perf_counter()
        result = sums(*arguments)
        synchronize()
        head_sums.append(dict(update=trainer.updates_done + 1, seconds=time.perf_counter() - begun))
        return result

    trainer.ppo.head_sums = timed_sums
    rows = []
    for index in range(1, updates + 1):
        begun = time.perf_counter()
        row = trainer.update()
        synchronize()
        seconds = time.perf_counter() - begun
        cadence = row.get("train/time/cadence_seconds")
        sums_seconds = sum(c["seconds"] for c in head_sums if c["update"] == index)
        rows.append(
            dict(
                update=index,
                seconds=seconds,
                rollout_seconds=row["train/time/rollout_seconds"],
                learner_seconds=row["train/time/learner_seconds"],
                cadence_seconds=cadence,
                head_sums_seconds=sums_seconds,
                head_sums_calls=sum(1 for c in head_sums if c["update"] == index),
                end_since_start=since_start(),
                peak_reserved_gb=torch.cuda.max_memory_reserved(device) / 2**30,
                lanes={k: v for k, v in row.items() if k.startswith("lanes/")},
                ratings={
                    k: v
                    for k, v in row.items()
                    if k.startswith("rating") and isinstance(v, int | float)
                },
            )
        )
        log(
            f"update {index}: {seconds:.2f} s (rollout {rows[-1]['rollout_seconds']:.2f}, "
            f"learn {rows[-1]['learner_seconds']:.2f}, cadence {cadence}, "
            f"heads' sums {sums_seconds:.2f})"
        )
    print(
        json.dumps(
            dict(
                config=OPTIONS["config"],
                options=OPTIONS,
                filled=filled,
                build_seconds=build_seconds,
                steps_per_update=trainer.steps_per_update,
                environments=config.environments,
                lane_environments=config.lanes.environments if config.lanes else 0,
                micro_batch_rows=config.learner.micro_batch_rows,
                micro_batch_note=note,
                updates=rows,
                compiles=EVENTS,
                compile_seconds=sum(e["seconds"] for e in EVENTS),
                peak_allocated_gb=torch.cuda.max_memory_allocated(device) / 2**30,
                peak_reserved_gb=torch.cuda.max_memory_reserved(device) / 2**30,
                torch=torch.__version__,
                gpu=torch.cuda.get_device_name(device),
            ),
            default=str,
        )
    )


if __name__ == "__main__":
    main()
