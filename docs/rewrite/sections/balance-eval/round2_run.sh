#!/bin/bash
# Round 2 on the laptop (2026-11-16, the author's decisions): error bars (a Jujak and a Yongma
# mirror at deathmatch 5v5, 4 seeds x 512 games each, the worker at 1024 environments), then
# one steady batch of every condition at 4096 environments and 32 variants (`timing`). Each a
# systemd user unit as laptop_job.sh's, compiling outside the GPU lock (--lock for the play).
set -u
here=/home/vizia/avesnova/boost-and-broadside/worktrees/bnb-balance-eval
data=$here/docs/rewrite/sections/balance-eval/data
lock=/tmp/claude-1000/bnb-gpu.lock
py=$here/.venv/bin/python
unit() {
  name=$1; shift
  systemd-run --user --unit="$name" --wait --collect \
    -p MemoryMax=20G -p MemorySwapMax=0 -p CPUQuota=1500% -p WorkingDirectory="$here" \
    -E PATH="$PATH" -E PYTHONPATH="$here/src:$here" \
    -E TORCHINDUCTOR_CACHE_DIR=/home/vizia/.cache/bnb-inductor -E TORCHINDUCTOR_COMPILE_THREADS=14 \
    -E TRITON_CACHE_DIR=/home/vizia/.cache/bnb-triton -E TORCH_LOGS=recompiles "$@"
  journalctl --user -u "$name" --no-pager -o short-precise >"$data/$name.journal.txt"
}
echo "$(date +%H:%M:%S) start; power $(cat /sys/firmware/acpi/platform_profile)"
queue=$here/docs/rewrite/sections/balance-eval/queue/mirrors-deathmatch-5v5
mkdir -p "$queue"
for seed in 0 1 2 3; do
cat >"$queue/0$seed-mirrors.json" <<JOB
{"pairs": [{"name": "Jujak mirror", "a": {"ship": "jujak"}, "b": {"ship": "jujak"}},
           {"name": "Yongma mirror", "a": {"ship": "yongma"}, "b": {"ship": "yongma"}}],
 "out": "$data/mirrors-deathmatch-5v5-seed$seed.jsonl", "games": 512, "seed": $seed}
JOB
done
echo "$(date +%H:%M:%S) mirrors"
unit balance-eval-mirrors $py -m bench.balance_worker --condition deathmatch-5v5 --queue "$queue" \
  --environments 1024 --variants 32 --once --lock $lock
grep -E "kill share" "$data/balance-eval-mirrors.journal.txt" | cut -c60-
for condition in deathmatch-5v5 deathmatch-10v10 deathmatch-2v2 deathmatch-1v1 \
    elimination-5v5 elimination-10v10 frontline-5v5 frontline-10v10; do
  echo "$(date +%H:%M:%S) timing $condition"
  unit balance-eval-timing-$condition $py -m bench.balance_sweep timing --condition $condition \
    --environments 4096 --variants 32 --lock $lock --out "$data/timing.jsonl"
  tail -1 "$data/timing.jsonl" | cut -c1-200
done
echo "$(date +%H:%M:%S) done"
