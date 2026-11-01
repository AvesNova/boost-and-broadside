"""torch.profiler tables for one benchmark cell (lock held). Writes the kernel table to
--out and prints its head.

    uv run --no-sync python -m bench.profile training --views 64 --precision bfloat16 \
        --mode compiled --out /tmp/bnb-scratch-yemong-blind/profile.txt
"""

import argparse

import torch
from torch.profiler import ProfilerActivity, profile

from bench import speed
from bench.lock import bench_lock


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("training", "rollout"))
    parser.add_argument("--views", type=int, default=None)
    parser.add_argument("--precision", default="bfloat16")
    parser.add_argument("--mode", default="compiled")
    parser.add_argument("--rows", type=int, default=40)
    parser.add_argument("--out", required=True)
    parser.add_argument("--trace", default=None)
    arguments = parser.parse_args()
    views = arguments.views or (64 if arguments.kind == "training" else 2048)
    with bench_lock():
        model = speed.make_model()
        cell = speed.training_cell if arguments.kind == "training" else speed.rollout_cell
        captured = {}
        original = speed.timed

        def grab(function, repetitions, warmup):
            captured["function"] = function
            return original(function, 3, warmup)

        speed.timed = grab
        cell(model, views, arguments.precision, arguments.mode, 3, 3)
        speed.timed = original
        function = captured["function"]
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
            for _ in range(5):
                function()
            torch.cuda.synchronize()
    table = profiler.key_averages().table(sort_by="cuda_time_total", row_limit=arguments.rows,
                                          max_name_column_width=90)
    with open(arguments.out, "w") as handle:
        handle.write(table)
    if arguments.trace:
        profiler.export_chrome_trace(arguments.trace)
    print("\n".join(table.splitlines()[: arguments.rows + 4]))


if __name__ == "__main__":
    main()
