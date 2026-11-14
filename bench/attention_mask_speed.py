"""What masking the absent zone slots out of attention, or gating the attention output,
would cost a training run: the learner's micro-batch step (labels, loss, backward, as
`Learner.gradient`) and the rollout's learner policy step, at a run's own shapes, with each
spatial attention on one path.

Variants, one process each (the attention switch and the patch are read at trace time):
  fused             the default: no key mask, the fused Triton attention
                    (`kernels.attention.attend`)
  fused_masked      every view's absent zone slots masked out (`model.absent_zone_key_mask`,
                    as the `Ablations.absent_zones` evaluation ablation, here on every view
                    and in the learn too), through the fused kernel's key mask
  masked            the same mask with the attention switch off: SDPA with a boolean mask
  plain             no mask, the attention switch off: SDPA without a mask
  gate_elementwise  the fused default with `YemongConfig.gate` "elementwise"
  gate_headwise     the fused default with `YemongConfig.gate` "headwise"

Each process builds the run's Trainer (`run`: a directory with a config.json), collects one
update (the rollout compiles; a realistic carry and store), captures the update's first
micro-batch, then times `warmup` + `repeats` micro-batch steps and `warmup` + `repeats`
replays of the learner's compiled policy step on the rollout's views, captured in a CUDA
graph as the rollout runs its stages (CUDA events per repeat). Prints the medians and the
quartiles as one JSON line last.

Usage: `python bench/attention_mask_speed.py run=DIR variant=VARIANT [repeats=30] [warmup=5]`
"""

import dataclasses
import json
import os
import sys
import time

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")  # as the training entry point
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch  # noqa: E402

START = time.perf_counter()
VARIANTS = ("fused", "fused_masked", "masked", "plain", "gate_elementwise", "gate_headwise")


def log(message: str) -> None:
    """Print `message` to stderr with the seconds since the start."""
    print(f"[{time.perf_counter() - START:7.1f} s] {message}", file=sys.stderr, flush=True)


def apply_variant(variant: str) -> None:
    """Set the attention path every Yemong traced from here on takes."""
    from bnb.models.yemong import kernels
    from bnb.models.yemong.model import Yemong, absent_zone_key_mask

    if variant in ("plain", "masked"):
        kernels.switches.attention = False
    if variant in ("fused_masked", "masked"):
        original = Yemong.tokens

        def tokens(self, *arguments, **keywords):
            built = original(self, *arguments, **keywords)
            every_view = torch.ones(
                built.code.shape[:-2], dtype=torch.bool, device=built.code.device
            )
            return dataclasses.replace(built, key_mask=absent_zone_key_mask(built, every_view))

        Yemong.tokens = tokens


def trainer_of(options: dict, variant: str):
    """The run's Trainer on the GPU, seeded as the profile's (`bench.learn_micro.trainer_of`),
    with the variant's gate."""
    from bnb.cli import compile_limits
    from bnb.runs.serialise import from_json
    from bnb.runtime.train import TrainConfig, Trainer

    compile_limits()
    with open(os.path.join(options["run"], "config.json")) as file:
        config, _ = from_json(TrainConfig, json.load(file))
    if variant.startswith("gate_"):
        gate = variant.removeprefix("gate_")
        config = dataclasses.replace(config, yemong=dataclasses.replace(config.yemong, gate=gate))
    torch.manual_seed(0)
    return Trainer(config, torch.device("cuda"))


def quantiles(times: list[float]) -> dict:
    """The median and the quartiles of `times` (ms)."""
    times = sorted(times)
    count = len(times)
    return dict(
        median_ms=times[count // 2],
        quartiles_ms=[times[count // 4], times[(3 * count) // 4]],
    )


def timed(function, repeats: int, warmup: int) -> dict:
    """`function` run `warmup` times, then `repeats` times each between CUDA events."""
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    times = []
    for _ in range(repeats):
        started = torch.cuda.Event(enable_timing=True)
        ended = torch.cuda.Event(enable_timing=True)
        started.record()
        function()
        ended.record()
        ended.synchronize()
        times.append(started.elapsed_time(ended))
    return quantiles(times)


class Captured(Exception):
    """Ends the update once the first micro-batch is saved."""


def micro_batch(trainer) -> dict:
    """Collect one update and keep its first micro-batch's step inputs."""
    from bnb.runtime import metrics

    learner = trainer.ppo
    store, bootstrap = trainer.collect()
    saved: dict = {}
    labels, loss = learner.labels, learner.loss

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
    learner.labels, learner.loss = labels, loss
    return saved


def learn_step(trainer, saved: dict, repeats: int, warmup: int) -> dict:
    """The micro-batch step as `Learner.gradient` runs it, timed."""
    learner = trainer.ppo

    def step():
        learner.optimizer.zero_grad(set_to_none=True)
        label = learner.labels(saved["trajectory"], saved["shot_labels"])
        out = learner.loss(
            saved["trajectory"], label, saved["part"], saved["denominator"], saved["scale"]
        )
        out[0].backward()

    step()
    torch.cuda.synchronize()
    log("learn step compiled")
    result = timed(step, repeats, warmup)
    rows = saved["trajectory"].done.shape  # (L, R)
    result.update(steps_views=list(rows), peak_allocated_gb=torch.cuda.max_memory_allocated() / 1e9)
    return result


def policy_step(trainer, repeats: int, warmup: int) -> dict:
    """The learner's policy step on the rollout's current views, compiled as the rollout
    compiles its stages and replayed from a CUDA graph."""
    rollout = trainer.rollout
    name = rollout.learner
    controller = rollout.controllers[name]
    carry = rollout.carry
    view = rollout._view(carry, name, rollout.reads + ("globals",))
    from bnb.contracts.obs import GLOBAL

    reset = view.globals[..., GLOBAL["reset"]] > 0.5
    controller_carry = carry["controllers"][name]
    compiled = torch.compile(controller.step, fullgraph=True, dynamic=False)
    with torch.no_grad():
        for _ in range(3):
            compiled(controller_carry, view, reset)
        torch.cuda.synchronize()
        log("policy step compiled")
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                compiled(controller_carry, view, reset)
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            compiled(controller_carry, view, reset)
    result = timed(graph.replay, repeats, warmup)
    result.update(views=int(reset.shape[0]))
    return result


def main() -> None:
    """Build, collect, capture, time (see the module docstring)."""
    options = dict(argument.split("=", 1) for argument in sys.argv[1:])
    variant = options.get("variant", "fused")
    if variant not in VARIANTS:
        raise SystemExit(f"variant one of {VARIANTS}")
    repeats, warmup = int(options.get("repeats", 30)), int(options.get("warmup", 5))
    apply_variant(variant)
    trainer = trainer_of(options, variant)
    log(f"built ({variant}); collecting")
    saved = micro_batch(trainer)
    log("collected; micro-batch captured")
    learn = learn_step(trainer, saved, repeats, warmup)
    log(f"learn step: median {learn['median_ms']:.2f} ms")
    policy = policy_step(trainer, repeats, warmup)
    log(f"policy step: median {policy['median_ms']:.3f} ms")
    print(json.dumps(dict(variant=variant, learn=learn, policy=policy)))


if __name__ == "__main__":
    main()
