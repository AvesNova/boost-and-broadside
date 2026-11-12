#!/bin/bash
# measure.sh VARIANT STAGE MODE CONFIG STEPS SETS: one measurement on the box, one JSON line
# appended to $RECORDS/records.jsonl (the compile bench's result, the variant, the GPU's
# state before and after). MODE cold: empty inductor and Triton caches (removed first);
# warm: the same caches a cold run of the variant and stage filled. SETS: the variant's
# config switches (`compile_baseline.py`'s set=, ';'-separated, may be empty).
set -uo pipefail
VARIANT=$1; STAGE=$2; MODE=$3; CONFIG=$4; STEPS=$5; SETS=${6:-}
ROOT=${ROOT:-/workspace/cs}; RECORDS=${RECORDS:-$ROOT/records}
cd /workspace/boost-and-broadside
CACHE=$ROOT/cache/$VARIANT-$STAGE
[ "$MODE" = cold ] && rm -rf "$CACHE"
mkdir -p "$CACHE/inductor" "$CACHE/triton" "$RECORDS" "$ROOT/logs"
export TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor TRITON_CACHE_DIR=$CACHE/triton
export TORCHINDUCTOR_COMPILE_THREADS=${TORCHINDUCTOR_COMPILE_THREADS:-10} OMP_NUM_THREADS=10
export TRITON_DEFAULT_FP_FUSION=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONFAULTHANDLER=1
gpu() { nvidia-smi --query-gpu=pstate,clocks.sm,clocks.mem,clocks.max.sm,power.draw,power.limit,temperature.gpu,clocks_throttle_reasons.active --format=csv,noheader; }
NAME=$VARIANT-$STAGE-$MODE-$(date +%H%M)
BEFORE=$(gpu)
START=$(date +%s.%N)
.venv/bin/python bench/compile_baseline.py train config="$CONFIG" steps="$STEPS" set="$SETS" \
  process_start="$START" > "$ROOT/logs/$NAME.json" 2> "$ROOT/logs/$NAME.log"
END=$(date +%s.%N)
AFTER=$(gpu)
.venv/bin/python - "$ROOT/logs/$NAME.json" "$VARIANT" "$STAGE" "$MODE" "$START" "$END" "$BEFORE" "$AFTER" "$SETS" \
  "$(git rev-parse --short HEAD)" >> "$RECORDS/records.jsonl" <<'PY'
import json, sys
path, variant, stage, mode, start, end, before, after, sets, commit = sys.argv[1:]
try:
    r = json.load(open(path))
except ValueError:  # the process died: what its log says
    log = open(path[:-5] + ".log").read().splitlines()
    print(json.dumps(dict(variant=variant, stage=stage, mode=mode, sets=sets, commit=commit,
        failed=True, process_seconds=float(end) - float(start), gpu_before=before,
        gpu_after=after, last_lines=[l for l in log if "] " in l or "Error" in l][-8:])))
    sys.exit(0)
p, d = r["phases"], r["details"]
steady = [k for k in p if k.startswith("collect_") and not k.endswith("since_start") and k != "collect_1"]
seconds = [p[k] + p[k.replace("collect", "learn")] for k in steady]
compiles = {}
for e in r["events"]:
    if e["kind"] == "compile":
        c = compiles.setdefault(e["label"], [0.0, 0, 0])
        c[0] += e["seconds"]; c[1] += e.get("ops", 0); c[2] += 1
print(json.dumps(dict(
    variant=variant, stage=stage, mode=mode, sets=sets, commit=commit,
    process_seconds=float(end) - float(start),
    first_rollout_since_start=p.get("collect_1_end_since_start"),
    first_update_since_start=p.get("learn_1_end_since_start"),
    phases={k: round(v, 2) for k, v in p.items()},
    steps_per_update=d["steps_per_update"],
    steady_updates=len(seconds),
    steady_steps_per_second=(len(seconds) * d["steps_per_update"] / sum(seconds)) if seconds else None,
    steady_update_seconds=[round(s, 2) for s in seconds],
    peak_allocated_gb=round(d["peak_vram_gb"], 3), peak_reserved_gb=round(d["peak_reserved_gb"], 3),
    head_sums=d.get("head_sums"),
    compile_seconds=round(r["compile_seconds"], 1), capture_seconds=round(r["capture_seconds"], 1),
    compiles={k: [round(v[0], 1), v[1], v[2]] for k, v in compiles.items()},
    gpu_before=before, gpu_after=after, torch=r["torch"], log=path,
)))
PY
echo "$NAME done: $(tail -1 $RECORDS/records.jsonl | head -c 300)"
