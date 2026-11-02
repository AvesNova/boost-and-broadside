"""torch.profiler tables for one benchmark cell (lock held). Writes the kernel table to
--out and prints its head.

    uv run --no-sync python -m bench.profile training --views 64 --precision bfloat16 \
        --mode compiled --out /tmp/bnb-scratch-yemong-blind/profile.txt
"""

import argparse
import re

import torch
from torch.profiler import ProfilerActivity, profile

from bench import speed
from bench.lock import bench_lock


GROUPS = (  # (group, pattern on the kernel name), first match wins
    ("custom attention", r"_attention_"), ("custom gated MLP", r"_gated_"),
    ("custom scan", r"_scan_"), ("GEMM", r"gemm|cutlass|triton_mm|_mm_|splitK|cublas"),
    ("fused rms_norm backward", r"rms_norm_backward"), ("fused rms_norm", r"rms_norm"),
    ("fused gelu", r"gelu"), ("fused reduction", r"triton_(per|red)_"),
    ("fused pointwise", r"triton_poi_"), ("copy / fill", r"[Mm]emcpy|[Mm]emset|fill|copy"),
)


def kernel_groups(profiler, iterations: int) -> str:
    """GPU time per iteration by kernel group (ms), from the profiler's device events."""
    totals = {}
    for event in profiler.events():
        if event.device_type != torch.autograd.DeviceType.CUDA:
            continue
        name = event.name
        group = next((g for g, pattern in GROUPS if re.search(pattern, name)), "other")
        totals[group] = totals.get(group, 0.0) + event.device_time / 1e3 / iterations
    whole = sum(totals.values())
    lines = [f"{'group':28s} {'ms / iteration':>14s} {'share':>7s}"]
    for group, value in sorted(totals.items(), key=lambda item: -item[1]):
        lines.append(f"{group:28s} {value:14.3f} {value / whole:7.1%}")
    lines.append(f"{'total':28s} {whole:14.3f}")
    return "\n".join(lines)


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
    groups = kernel_groups(profiler, 5)
    table = profiler.key_averages().table(sort_by="cuda_time_total", row_limit=arguments.rows,
                                          max_name_column_width=90)
    with open(arguments.out, "w") as handle:
        handle.write(table)
    if arguments.trace:
        profiler.export_chrome_trace(arguments.trace)
    print("\n".join(table.splitlines()[: arguments.rows + 4]))
    print(groups)
    with open(arguments.out, "a") as handle:
        handle.write("\n" + groups + "\n")


if __name__ == "__main__":
    main()
