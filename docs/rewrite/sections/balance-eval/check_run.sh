#!/bin/bash
# The GPU check on the laptop (2026-11-16): fog speed (dm 5v5, 2048 games of 60 s, fog off /
# on / on with fields), then the worker on a job of Yongma v Jujak at dm 5v5 (256 games, the
# whole 600 s clock) and a Jujak mirror pair. Each step a systemd user unit as laptop_job.sh's
# (MemoryMax 20G, 15 cores, caches on the NVMe), compiling outside the GPU lock and holding
# it only to time or play (--lock).
set -u
here=/home/vizia/avesnova/bnb-balance-eval
data=$here/docs/rewrite/sections/balance-eval/data
lock=/tmp/claude-1000/bnb-gpu.lock
unit() {
  name=$1; shift
  systemd-run --user --unit="$name" --wait --collect \
    -p MemoryMax=20G -p MemorySwapMax=0 -p CPUQuota=1500% -p WorkingDirectory="$here" \
    -E PATH="$PATH" -E PYTHONPATH="$here/src:$here" \
    -E TORCHINDUCTOR_CACHE_DIR=/home/vizia/.cache/bnb-inductor -E TORCHINDUCTOR_COMPILE_THREADS=14 \
    -E TRITON_CACHE_DIR=/home/vizia/.cache/bnb-triton -E TORCH_LOGS=recompiles "$@"
  # the program's output goes to the unit's journal: keep it beside the data
  journalctl --user -u "$name" --no-pager -o short-precise >"$data/$name.journal.txt"
}
py=$here/.venv/bin/python
echo "$(date +%H:%M:%S) start; power $(cat /sys/firmware/acpi/platform_profile)"
for setting in "off off" "on off" "on on"; do
  set -- $setting
  echo "$(date +%H:%M:%S) speed fog $1 fields $2"
  unit balance-eval-speed-$1-$2 $py -m bench.balance_sweep speed --condition deathmatch-5v5 \
    --fog $1 --fields $2 --environments 2048 --seconds 60 --batches 2 --lock $lock \
    >"$data/speed-fog-$1-fields-$2.log" 2>&1
  grep -E "games/s|compiled|Error" "$data/speed-fog-$1-fields-$2.log"
done
queue=$here/docs/rewrite/sections/balance-eval/queue/deathmatch-5v5
mkdir -p "$queue"
cat >"$queue/01-check.json" <<JOB
{"candidates": [{"name": "base: Yongma team v Jujak team"},
                {"name": "pair: Jujak v Jujak", "a": {"ship": "jujak"}, "b": {"ship": "jujak"}}],
 "out": "$data/check-deathmatch-5v5.jsonl", "games": 256, "mixed_games": 0, "seed": 0}
JOB
echo "$(date +%H:%M:%S) worker"
unit balance-eval-worker $py -m bench.balance_worker --condition deathmatch-5v5 --queue "$queue" \
  --environments 256 --variants 32 --once --lock $lock >"$data/worker-check.log" 2>&1
grep -vE "^\s*$" "$data/worker-check.log" | tail -8
echo "$(date +%H:%M:%S) done"
