"""The marginal GPU cost of each part of the learner's micro-batch; prints `RESULT {json}`.

    uv run --no-sync python -m bench.update_parts '{"preset": "deathmatch", "envs": 256}'

Builds the trainer as `bench.update_profile` does, prepares one update's store, stages its
first micro-batch, and times the compiled forward and backward of the micro-batch loss with
heads removed one group at a time (a removed head's field is None, so its term is skipped
and the compiler drops its work), then the trunk alone, then the encoder alone. Each
variant's cost is its median of `repeats` timed runs, in CUDA events, the variants
interleaved round by round. Takes the benchmark
lock. Options as in `bench.update_profile`, and "repeats".
"""

import dataclasses
import json
import os
import sys

os.environ.setdefault("TRITON_DEFAULT_FP_FUSION", "1")

import torch  # noqa: E402

from bench.lock import bench_lock  # noqa: E402
from bench.update_profile import build  # noqa: E402

SHOT = ("shot_latents", "shot_outcome_logits")
VARIANTS = (
    ("full", ()),
    ("no shot trajectory", ("shot_latents",)),
    ("no shot heads", SHOT),
    ("no next state", ("next_state_logits",)),
    ("no enemy action", ("enemy_action_logits",)),
    ("no density", ("density_log_rates",)),
    (
        "policy and critic only",
        SHOT + ("next_state_logits", "enemy_action_logits", "density_log_rates"),
    ),
)


def timed(function) -> tuple[float, float]:
    """Forward and forward+backward seconds of one `function()` (returns the loss)."""
    start, middle, end = (torch.cuda.Event(enable_timing=True) for _ in range(3))
    start.record()
    loss = function()
    middle.record()
    loss.backward()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(middle) / 1e3, start.elapsed_time(end) / 1e3


def main() -> None:
    from bnb.contracts import Trajectory
    from bnb.contracts.obs import GLOBAL
    from bnb.learn.ppo import COUNTS, Learner

    cell = json.loads(sys.argv[1])
    repeats = cell.get("repeats", 10)
    with bench_lock():
        trainer, _ = build(cell)
        ppo = trainer.ppo
        store, bootstraps = trainer.collect()
        prepared = ppo.prepare(store, bootstraps)
        views = torch.arange(ppo.micro_views)
        trajectory, shot_labels = next(iter(ppo.staged(store, [views])))
        trajectory = Trajectory(
            **{
                field.name: getattr(trajectory, field.name)
                for field in dataclasses.fields(Trajectory)
            }
        )
        device_views = views.to(ppo.device)
        part = {
            name: value.index_select(1, device_views) for name, value in prepared["derived"].items()
        }
        total = prepared["counts"][device_views].sum(0).clamp_min(1.0)
        denominator = dict(zip(COUNTS, total.unbind()))
        scale = prepared["scale"]
        unroll = Learner.unroll.__get__(ppo)
        torch._dynamo.config.cache_size_limit = 64  # one entry per variant (guards on unroll)
        model = ppo.model

        def trunk():
            reset = trajectory.obs.globals[..., GLOBAL["reset"]] > 0.5
            tokens = model.tokens(trajectory.obs, trajectory.replay)
            with model._matmuls(trajectory.obs.ships.device):
                queries, memory, token_tables, map_tables = model._encode(tokens)
                carry = trajectory.initial_carry
                hidden, _, _ = model.trunk.sequence(
                    queries,
                    memory,
                    token_tables,
                    map_tables,
                    carry.state,
                    carry.convolution_buffer,
                    reset,
                )
            return hidden.float().square().mean()

        def spatial():
            """The trunk's spatial sublayers alone (each block's temporal one skipped)."""
            tokens = model.tokens(trajectory.obs, trajectory.replay)
            with model._matmuls(trajectory.obs.ships.device):
                queries, memory, token_tables, map_tables = model._encode(tokens)
                for block in model.trunk.blocks:
                    queries = block.attend(queries, memory, token_tables, map_tables)
            return queries.float().square().mean()

        def encoder():
            tokens = model.tokens(trajectory.obs, trajectory.replay)
            with model._matmuls(trajectory.obs.ships.device):
                queries, memory, _, _ = model._encode(tokens)
            return queries.float().square().mean() + memory.float().square().mean()

        compiled_loss = torch.compile(Learner._loss.__get__(ppo), fullgraph=True, dynamic=False)
        functions = {}
        for name, dropped in VARIANTS:

            def dropping(batch, dropped=dropped):
                return dataclasses.replace(unroll(batch), **dict.fromkeys(dropped))

            def run(dropping=dropping):
                ppo.unroll = dropping
                return compiled_loss(trajectory, shot_labels, part, denominator, scale)[0]

            functions[name] = run
        functions["trunk only"] = torch.compile(trunk, fullgraph=True, dynamic=False)
        functions["spatial sublayers only"] = torch.compile(spatial, fullgraph=True,
                                                            dynamic=False)
        functions["encoder only"] = torch.compile(encoder, fullgraph=True, dynamic=False)
        for function in functions.values():  # compile and warm
            function().backward()
        # Interleaved rounds, so a neighbour on the shared GPU slows every variant alike.
        samples = {name: ([], []) for name in functions}
        for _ in range(repeats):
            for name, function in functions.items():
                forward, total = timed(function)
                samples[name][0].append(forward)
                samples[name][1].append(total)
        results = {
            name: tuple(sorted(values)[len(values) // 2] for values in pair)
            for name, pair in samples.items()
        }
        codes = torch.compile(Learner.labels.__get__(ppo), fullgraph=True, dynamic=False)

        def label_codes():
            return codes(trajectory, shot_labels)

        times = []
        for _ in range(repeats + 2):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            label_codes()
            end.record()
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end) / 1e3)
        results["label codes"] = (sorted(times[2:])[repeats // 2], 0.0)
    out = dict(
        cell=cell,
        micro_views=ppo.micro_views,
        seconds={name: dict(forward=f, forward_backward=t) for name, (f, t) in results.items()},
    )
    print("RESULT " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
