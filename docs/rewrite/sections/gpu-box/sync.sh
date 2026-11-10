#!/bin/bash
# Save the box's runs off the box (it is not a volume: a recycle or a crash loses them).
#  - git (GitHub, branch perf/gpu-box): every run's small records under results/gpu-box/
#    (run.json, config.json, metrics.jsonl, console.txt: the console log without warnings;
#    .txt because the repo ignores *.log), the
#    tournaments, the runner log and scripts; committed by path only, then pushed.
#  - Hugging Face Hub (public model repo julianzia/boost-and-broadside-runs): each run
#    directory whole (policies, checkpoints), skipping half-written `.tmp` checkpoints; the
#    hub skips files it already has.
# Usage: sync.sh            one pass
#        sync.sh --loop N   a pass every N minutes, forever
set -u
REPO=/workspace/boost-and-broadside
RUNS=$REPO/runs
OUT=$REPO/results/gpu-box
cd "$REPO"
set -a; . /workspace/.env; set +a

one_pass() {
  mkdir -p "$OUT/runs" "$OUT/ratings"
  for run in "$RUNS"/2026-*/; do
    run=${run%/}; name=$(basename "$run")
    [ -f "$run/run.json" ] || continue
    mkdir -p "$OUT/runs/$name"
    cp "$run/run.json" "$run/config.json" "$OUT/runs/$name/" 2>/dev/null
    [ -f "$run/metrics.jsonl" ] && cp "$run/metrics.jsonl" "$OUT/runs/$name/"
    [ -f "$RUNS/$name.out" ] && grep -v -E "FutureWarning|^  check\(" "$RUNS/$name.out" \
      > "$OUT/runs/$name/console.txt"
  done
  [ -d "$RUNS/ratings" ] && cp -r "$RUNS/ratings/." "$OUT/ratings/"
  for f in "$RUNS"/ratings-*.out; do
    [ -f "$f" ] && grep -v -E "FutureWarning|^  check\(" "$f" > "$OUT/ratings/$(basename "$f" .out).txt"
  done
  cp "$RUNS/arms.log" "$OUT/runner.txt" 2>/dev/null
  mkdir -p "$OUT/scripts" && cp "$RUNS"/*.sh "$OUT/scripts/" 2>/dev/null
  git add -- results/gpu-box
  if ! git diff --cached --quiet -- results/gpu-box; then
    git commit -q -m "results: gpu-box sync $(date -u +%Y-%m-%dT%H:%MZ)

Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01HJzH7FHcpqQFApfR74WvVb" -- results/gpu-box
  fi
  timeout 300 git push -q origin perf/gpu-box 2>&1 | grep -v "^remote:"
  /venv/main/bin/python - "$RUNS" <<'EOF'
import os, sys, pathlib
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
runs = pathlib.Path(sys.argv[1])
for run in sorted(runs.glob("2026-*")):
    if not run.is_dir() or not (run / "run.json").exists():
        continue
    api.upload_folder(repo_id="julianzia/boost-and-broadside-runs", folder_path=run,
                      path_in_repo=f"runs/{run.name}", ignore_patterns=["*.tmp/*", "*.tmp/**"],
                      commit_message=f"sync {run.name}")
if (runs / "ratings").exists():
    api.upload_folder(repo_id="julianzia/boost-and-broadside-runs", folder_path=runs / "ratings",
                      path_in_repo="ratings", commit_message="sync ratings")
print("hub sync ok")
EOF
  # an upload is not trusted until its hash is checked: re-upload any file whose hub copy
  # differs from its checkpoint record (seen 4 times on 2026-11-09/10: same size, wrong bytes)
  /venv/main/bin/python docs/rewrite/sections/gpu-box/hub_repair.py "$RUNS" 2>&1 | grep "hub repair"
}

disk_guard() {  # the box has 64 GB: below 15 GB free, prune compile-cache files unused 12 h
  local free_gb
  free_gb=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
  echo "disk: ${free_gb} GB free"
  if [ "$free_gb" -lt 15 ]; then
    find /tmp/torchinductor_root -type f -amin +720 -mmin +720 -delete 2>/dev/null
    find /tmp/torchinductor_root -type d -empty -delete 2>/dev/null
    echo "disk: pruned the compile cache, now $(df -BG --output=avail / | tail -1 | tr -dc 0-9) GB free"
  fi
}

if [ "${1:-}" = "--loop" ]; then
  while true; do
    echo "== $(date -u +%H:%M) sync"; one_pass; disk_guard; sleep $(( ${2:-20} * 60 ))
  done
else
  one_pass
fi
