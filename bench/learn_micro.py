"""One real micro-batch of the learn, timed alone: the label codes, the loss and its backward.

Two cases, one process each:
  capture  builds the run's Trainer (`run`: a run directory with config.json), collects
           once and saves the update's first micro-batch (its Trajectory, stored shot
           labels, derived part, denominators and return scale) and the model's weights
           to `batch` (a torch.save file)
  time     builds the run's Trainer (no collection), loads `batch`, then runs the micro-batch
           step as `Learner.gradient` does (`labels`, `loss`, backward) `warmup` times and
           `repeats` timed ones (CUDA events per repeat), and prints the median and the
           quartiles; profile=1 adds the kernels by kind of one more step and `top` kernels.
           `values=PATH` saves the loss, terms, and the gradient's norm per parameter of the
           first step (for comparing two trees' maths).

Usage: `python bench/learn_micro.py capture run=DIR batch=PATH` then
`python bench/learn_micro.py time run=DIR batch=PATH [repeats=30] [warmup=5] [profile=0]
[top=30] [values=PATH] [ages=N]`; ages=N queries the shot trajectory head at the first N
bullet ages only (a diagnostic of the head's share). Prints one JSON line last.
"""

import json
import os
import sys
import time

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")  # as the training entry point
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

START = time.perf_counter()


def log(message: str) -> None:
    """Print `message` to stderr with the seconds since the start."""
    print(f"[{time.perf_counter() - START:7.1f} s] {message}", file=sys.stderr, flush=True)


def trainer_of(options: dict):
    """The run's Trainer on the GPU, seeded as the profile's."""
    from bnb.cli import compile_limits
    from bnb.runs.serialise import from_json
    from bnb.runtime.train import TrainConfig, Trainer

    compile_limits()
    with open(os.path.join(options["run"], "config.json")) as file:
        config, _ = from_json(TrainConfig, json.load(file))
    torch.manual_seed(0)
    return Trainer(config, torch.device("cuda"))


class Captured(Exception):
    """Ends the update once the first micro-batch is saved."""


def capture(options: dict) -> None:
    """Collect once and save the update's first micro-batch (see the module docstring)."""
    from bnb.runtime import metrics

    trainer = trainer_of(options)
    learner = trainer.ppo
    log("built; collecting")
    store, bootstrap = trainer.collect()
    log("collected; capturing the first micro-batch")
    saved: dict = {}
    labels = learner.labels

    def labels_saving(trajectory, shot_labels):
        saved.update(trajectory=trajectory, shot_labels=shot_labels)
        return labels(trajectory, shot_labels)

    def loss_saving(trajectory, label, part, denominator, scale):
        saved.update(part=part, denominator=denominator, scale=scale)
        raise Captured

    learner.labels, learner.loss = labels_saving, loss_saving
    try:
        metrics.read(learner.update(store, bootstrap, trainer.steps_per_update))
    except Captured:
        pass
    saved["model"] = trainer.model.state_dict()
    torch.save(saved, options["batch"])
    log(f"saved {options['batch']}")


def time_step(options: dict) -> None:
    """Time the saved micro-batch's step (see the module docstring)."""
    from bench.learn_profile import kind

    repeats, warmup, top = (
        int(options.get(k, d)) for k, d in (("repeats", 30), ("warmup", 5), ("top", 30))
    )
    trainer = trainer_of(options)
    learner, model = trainer.ppo, trainer.model
    saved = torch.load(options["batch"], weights_only=False)
    model.load_state_dict(saved["model"])
    trajectory, shot_labels = saved["trajectory"], saved["shot_labels"]
    part, denominator, scale = saved["part"], saved["denominator"], saved["scale"]
    log("built and loaded")

    ages = int(options.get("ages", 0))  # a diagnostic: the shot head at the first ages only
    if ages:
        learner.ages = learner.ages[:ages]

    def step():
        label = learner.labels(trajectory, shot_labels)
        if ages:
            label = dict(
                label,
                shot_snapshots=label["shot_snapshots"][..., :ages, :],
                shot_ages=label["shot_ages"][..., :ages],
            )
        out = learner.loss(trajectory, label, part, denominator, scale)
        out[0].backward()
        return out

    learner.optimizer.zero_grad(set_to_none=True)
    first = step()
    torch.cuda.synchronize()
    log("compiled (first step)")
    if options.get("values"):
        values = dict(
            loss=first[0].item(),
            terms=first[1].tolist(),
            per_level=first[2].tolist(),
            statistics=first[3].tolist(),
            diagnostics=first[4].tolist(),
            gradients={
                name: p.grad.float().norm().item()
                for name, p in model.named_parameters()
                if p.grad is not None
            },
        )
        with open(options["values"], "w") as file:
            json.dump(values, file)
    for _ in range(warmup):
        learner.optimizer.zero_grad(set_to_none=True)
        step()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        learner.optimizer.zero_grad(set_to_none=True)
        started, ended = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        started.record()
        step()
        ended.record()
        ended.synchronize()
        times.append(started.elapsed_time(ended))
    times.sort()
    result = dict(
        batch=options["batch"],
        median_ms=times[len(times) // 2],
        quartiles_ms=[times[len(times) // 4], times[(3 * len(times)) // 4]],
        peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9,
    )
    log(f"step: median {result['median_ms']:.2f} ms, quartiles {result['quartiles_ms']}")
    if int(options.get("profile", 0)):
        learner.optimizer.zero_grad(set_to_none=True)
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CUDA]) as profiler:
            step()
            torch.cuda.synchronize()
        totals: dict[str, float] = {}
        for event in profiler.events():
            if event.device_type == torch.autograd.DeviceType.CUDA:
                totals[event.name] = totals.get(event.name, 0.0) + event.device_time_total / 1e3
        by_kind: dict[str, float] = {}
        for name, ms in totals.items():
            by_kind[kind(name)] = by_kind.get(kind(name), 0.0) + ms
        kernel_ms = sum(totals.values())
        print(f"kernels {kernel_ms:.1f} ms")
        for label, ms in sorted(by_kind.items(), key=lambda item: -item[1]):
            print(f"  {label:18s} {ms:8.2f} ms  {100 * ms / kernel_ms:5.1f}%")
        ranked = sorted(totals.items(), key=lambda item: -item[1])[:top]
        for name, ms in ranked:
            print(f"{ms:8.3f} ms  {kind(name):16s} {name[:120]}")
        result.update(
            kernel_ms=kernel_ms,
            by_kind=by_kind,
            top=[dict(name=n, ms=ms) for n, ms in ranked],
        )
    print(json.dumps(result))


def main() -> None:
    """Dispatch on the case."""
    case, options = sys.argv[1], dict(a.split("=", 1) for a in sys.argv[2:])
    {"capture": capture, "time": time_step}[case](options)


if __name__ == "__main__":
    main()
