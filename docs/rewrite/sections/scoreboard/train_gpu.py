"""Scoreboard, training on the GPU: collect alone, update alone and the whole loop, eager
against compiled, with peak memory, the first update's time (the compile) and the cost of
streaming the rollout.

    PYTHONPATH=. python docs/rewrite/sections/scoreboard/train_gpu.py \\
        [--compile | --no-compile] [--envs 2048] [--segments 4] [--updates 3] \\
        [--minibatches 32] [--micro 64] [--stream | --no-stream] [--snapshots 8] \\
        [--targets-every-epoch] [--no-bf16] [--out rows.jsonl] [--label text]

Defaults are LearnConfig's (the update at parity with v2: 2,048 environments x 4 segments
x 128 ticks, 32 minibatches, micro-batches of 64 views, the rollout streamed to pinned host
memory, targets once per update). The loop is lean/train.py's (segments into one
RolloutStore, update) with the league filled before the first rollout with `--snapshots`
snapshots of the initial weights (League.after_update, so the EMA and the uniform average
join too), so every network slot plays as in a run's steady state; nothing is added during
the measurement.

Per update: collect s, update s, loop s and environment steps per second of each; the
host's seconds waiting for copies (RolloutStore.ready and the staging buffers) and
gathering micro-batches; the compute stream's seconds waiting for micro-batch copies (CUDA
events). Summary: the first update (the compile, cold or warm by the inductor cache in
TORCHINDUCTOR_CACHE_DIR), the steady mean over the later updates, peak memory allocated
and reserved, and the store's bytes per environment tick. With `--out` each update's row
and the summary are appended to a JSON-lines file as they complete.
"""

import argparse
import dataclasses
import json
import statistics
import time

PROCESS_START = time.perf_counter()

import torch  # noqa: E402

from lean import game, model, scripted  # noqa: E402
from lean.learn import (  # noqa: E402
    League,
    LearnConfig,
    Opponent,
    RolloutStore,
    WeightAverages,
    collect,
    make_optimiser,
    start,
    update,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    defaults = LearnConfig()
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--envs", type=int, default=defaults.environments)
    parser.add_argument("--segments", type=int, default=defaults.segments_per_update)
    parser.add_argument("--updates", type=int, default=3)
    parser.add_argument("--minibatches", type=int, default=defaults.minibatches)
    parser.add_argument(
        "--micro",
        type=int,
        default=defaults.micro_batch_views,
        help="LearnConfig.micro_batch_views",
    )
    parser.add_argument(
        "--stream", action=argparse.BooleanOptionalAction, default=defaults.stream_rollout
    )
    parser.add_argument("--snapshots", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--targets-every-epoch", action="store_true", help="LearnConfig.targets_every_epoch = True"
    )
    parser.add_argument(
        "--bf16", action=argparse.BooleanOptionalAction, default=True, help="LearnConfig.bf16"
    )
    parser.add_argument("--out", default=None, help="append rows to this JSON-lines file")
    parser.add_argument("--label", default="")
    arguments = parser.parse_args()

    device = torch.device("cuda")
    config = dataclasses.replace(
        defaults,
        minibatches=arguments.minibatches,
        micro_batch_views=arguments.micro,
        stream_rollout=arguments.stream,
        segments_per_update=arguments.segments,
        bf16=arguments.bf16,
        targets_every_epoch=arguments.targets_every_epoch,
    )
    envs, ticks, segments = arguments.envs, config.segment_ticks, config.segments_per_update
    torch.manual_seed(arguments.seed)
    generator = torch.Generator(device=device).manual_seed(arguments.seed)
    game_config = game.GameConfig()
    policy = model.Policy(model.ModelConfig()).to(device)
    optimiser = make_optimiser(config, policy)
    league = League(config, device)
    averages = WeightAverages(policy, config)
    averages.update(policy, 10**9)  # past the warm-up: the uniform average holds an update
    nobody = torch.zeros(0, dtype=torch.long, device=device)
    for index in range(arguments.snapshots):
        step = league.schedule.points(10**12)[index]
        league.after_update(policy, averages, step - 1, step, nobody, nobody)
    opponent = Opponent(scripted.ScriptedConfig(), league)
    state, carry = start(config, game_config, policy, opponent, envs, generator)
    store = RolloutStore(segments, config.stream_rollout)
    store.timing = True
    compiled = arguments.compile
    setting = {
        "label": arguments.label,
        "compile": compiled,
        "bf16": config.bf16,
        "envs": envs,
        "segments": segments,
        "ticks": ticks,
        "minibatches": config.minibatches,
        "micro_batch_views": config.micro_batch_views,
        "stream": config.stream_rollout,
        "targets_every_epoch": config.targets_every_epoch,
        "epochs": config.epochs,
        "snapshots": league.snapshots,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(),
    }
    print(setting, flush=True)

    def persist(row: dict) -> None:
        if arguments.out:
            with open(arguments.out, "a") as file:
                file.write(json.dumps({**setting, **row}) + "\n")

    rows = []
    first_update_end = None
    steps = envs * ticks * segments
    steps_per_update = config.epochs * config.minibatches
    for index in range(arguments.updates):
        torch.cuda.synchronize()
        began = time.perf_counter()
        for segment in range(segments):
            rollout = collect(
                config,
                game_config,
                policy,
                opponent,
                state,
                carry,
                ticks,
                generator,
                compiled=compiled,
                store=store,
                segment=segment,
            )
            state, carry = rollout.state, rollout.carry
        del rollout
        torch.cuda.synchronize()
        collected = time.perf_counter()
        store.host_wait = store.host_gather = 0.0
        store.stalls()
        update(
            config,
            policy,
            optimiser,
            store,
            teacher_weight=config.teacher_coefficient,
            step_index=index * steps_per_update,
            generator=generator,
            compiled=compiled,
        )
        torch.cuda.synchronize()
        ended = time.perf_counter()
        stall = store.stalls() if store.pinned else 0.0
        if first_update_end is None:
            first_update_end = ended
        row = {
            "update": index,
            "collect_s": collected - began,
            "update_s": ended - collected,
            "loop_s": ended - began,
            "host_wait_s": store.host_wait,
            "host_gather_s": store.host_gather,
            "compute_stall_s": stall,
        }
        rows.append(row)
        persist(row)
        print(
            f"update {index}: collect {row['collect_s']:.2f} s ({steps / row['collect_s']:.0f}/s) "
            f"update {row['update_s']:.2f} s ({steps / row['update_s']:.0f}/s) loop {row['loop_s']:.2f} s "
            f"({steps / row['loop_s']:.0f}/s) | host wait {store.host_wait:.3f} s gather {store.host_gather:.2f} s "
            f"compute stall {stall:.3f} s",
            flush=True,
        )
    steady = rows[1:] or rows
    mean = {
        name: statistics.mean(row[name] for row in steady) for name in rows[0] if name != "update"
    }
    sizes = store.nbytes()
    summary = {
        "summary": True,
        "first_collect_s": rows[0]["collect_s"],
        "first_update_s": rows[0]["update_s"],
        "collect_per_s": steps / mean["collect_s"],
        "update_per_s": steps / mean["update_s"],
        "loop_per_s": steps / mean["loop_s"],
        "steady_host_wait_s": mean["host_wait_s"],
        "steady_host_gather_s": mean["host_gather_s"],
        "steady_compute_stall_s": mean["compute_stall_s"],
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30,
        "start_to_first_update_end_s": first_update_end - PROCESS_START,
        "store_bytes_per_environment_tick": sum(sizes.values()) / steps,
        "store_bytes": sizes,
    }
    persist(summary)
    print(
        f"SUMMARY {setting} | first update: collect {rows[0]['collect_s']:.1f} s update {rows[0]['update_s']:.1f} s "
        f"| steady steps/s: collect {summary['collect_per_s']:.0f} update {summary['update_per_s']:.0f} "
        f"loop {summary['loop_per_s']:.0f} | host wait {mean['host_wait_s']:.3f} s gather {mean['host_gather_s']:.2f} s "
        f"compute stall {mean['compute_stall_s']:.3f} s per update | peak allocated "
        f"{summary['peak_allocated_gib']:.2f} GiB reserved {summary['peak_reserved_gib']:.2f} GiB | start to first "
        f"update end {summary['start_to_first_update_end_s']:.1f} s | store {summary['store_bytes_per_environment_tick']:.0f} "
        f"bytes per environment tick",
        flush=True,
    )


if __name__ == "__main__":
    main()
