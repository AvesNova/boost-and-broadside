#!/usr/bin/env bash
# The laptop run (RTX 4070 Laptop, native Ubuntu): every suite, one job at a time, PyTorch and JAX
# alternating so slow drift (temperature, background load) falls on both. Records go to
# results/laptop/, one JSON file per invocation; aggregate.py builds the tables.
#
#   bash run_laptop.sh            # from this directory
#
# Float32 is run with the SAME matmul precision in both frameworks: JAX's default on this GPU is
# reduced precision (parity_laptop_cuda_default_precision.txt), so the main runs set
# JAX_DEFAULT_MATMUL_PRECISION=highest to match PyTorch's default ('highest'); that pairing is the
# one parity.py verified (parity_laptop_cuda.txt). A last pass times steady + loop with both on
# TF32 (PyTorch --matmul-precision high, JAX at its default).
set -u
cd "$(dirname "$0")"
TORCH_PY=${TORCH_PY:-$(cd ../../../.. && pwd)/.venv/bin/python}
JAX_PY=${JAX_PY:-$HOME/spike-jax/bin/python}
REPEATS=${REPEATS:-5}
OUT=results/laptop
mkdir -p "$OUT/logs"
LOG="$OUT/progress.log"
SHIPS_LATENCY=10,40,100

stamp() { echo "$(date '+%F %T') load $(cut -d' ' -f1-3 /proc/loadavg) | $*" | tee -a "$LOG"; }
run() {   # run NAME CMD...: skip if NAME's JSON already exists (so the script can be resumed)
    local name=$1; shift
    if [ -s "$OUT/$name.json" ]; then stamp "skip $name (exists)"; return; fi
    stamp "start $name"
    "$@" --out "$OUT/$name.json" > "$OUT/logs/$name.log" 2>&1
    local code=$?
    local failed; failed=$(grep -c "CHILD FAILED" "$OUT/logs/$name.log")
    stamp "end   $name exit $code, failed children $failed"
}

stamp "begin: power profile $(cat /sys/firmware/acpi/platform_profile 2>/dev/null) / $(powerprofilesctl get 2>/dev/null)"
export JAX_DEFAULT_MATMUL_PRECISION=highest

for r in $(seq 1 "$REPEATS"); do
    run "torch_gpu_compile_r$r" "$TORCH_PY" torch_bench.py --device cuda --suite compile
    run "jax_gpu_compile_r$r"   "$JAX_PY"   jax_bench.py   --device cuda --suite compile
    run "torch_gpu_sizes_r$r"   "$TORCH_PY" torch_bench.py --device cuda --suite sizes
    run "jax_gpu_sizes_r$r"     "$JAX_PY"   jax_bench.py   --device cuda --suite sizes
    if [ "$r" = 1 ]; then   # steady and loop take 7 timed repeats inside one invocation
        run torch_gpu_steady "$TORCH_PY" torch_bench.py --device cuda --suite steady
        run jax_gpu_steady   "$JAX_PY"   jax_bench.py   --device cuda --suite steady
        run torch_gpu_loop   "$TORCH_PY" torch_bench.py --device cuda --suite loop
        run jax_gpu_loop     "$JAX_PY"   jax_bench.py   --device cuda --suite loop
        run torch_ops        "$TORCH_PY" torch_bench.py --device cpu  --suite ops
        run jax_ops          "$JAX_PY"   jax_bench.py   --device cpu  --suite ops
    fi
    # Batch size 1: the laptop's CPU (GPU idle), then the GPU.
    run "torch_cpu_latency_r$r" "$TORCH_PY" torch_bench.py --device cpu  --suite latency --latency-ships $SHIPS_LATENCY
    run "jax_cpu_latency_r$r"   "$JAX_PY"   jax_bench.py   --device cpu  --suite latency --latency-ships $SHIPS_LATENCY
    run "torch_gpu_latency_r$r" "$TORCH_PY" torch_bench.py --device cuda --suite latency --latency-ships $SHIPS_LATENCY --latency-threads default
    run "jax_gpu_latency_r$r"   "$JAX_PY"   jax_bench.py   --device cuda --suite latency --latency-ships $SHIPS_LATENCY --latency-threads default
done

# Both frameworks on TF32 for float32 matmuls.
unset JAX_DEFAULT_MATMUL_PRECISION
run torch_gpu_steady_tf32 "$TORCH_PY" torch_bench.py --device cuda --suite steady --matmul-precision high
run jax_gpu_steady_tf32   "$JAX_PY"   jax_bench.py   --device cuda --suite steady
run torch_gpu_loop_tf32   "$TORCH_PY" torch_bench.py --device cuda --suite loop --matmul-precision high
run jax_gpu_loop_tf32     "$JAX_PY"   jax_bench.py   --device cuda --suite loop
stamp "done"
