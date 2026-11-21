#!/bin/bash
# fetch_eval.sh CHAIN [DEST]: copy box/eval_queue.sh's outputs from the 5090 box to the laptop
# (DEST, default /home/vizia/avesnova/boost-and-broadside/review/dry-run): $RUNS/CHAIN-eval/ whole (pools, records,
# logs, media, charts; small), the champion's policy file (to play or re-render it here),
# the chain's run records the health chart reads (run.json, config.json, metrics.jsonl), and
# the queue's and the chain's top-level logs. Safe while the queue runs (a partial copy; run
# it again). BOX overrides the ssh host (box5090), RUNS the box's runs directory.
set -euo pipefail
chain=${1:?usage: fetch_eval.sh CHAIN [DEST]}
dest=${2:-/home/vizia/avesnova/boost-and-broadside/review/dry-run}
box=${BOX:-box5090}
runs=${RUNS_BOX:-/workspace/runs}
mkdir -p "$dest/runs"
rsync -a "$box:$runs/$chain-eval/" "$dest/$chain-eval/"
for run in "$chain-bc" "$chain-rl"; do
  mkdir -p "$dest/runs/$run"
  rsync -a "$box:$runs/$run/run.json" "$box:$runs/$run/config.json" \
    "$box:$runs/$run/metrics.jsonl" "$dest/runs/$run/" || echo "no records of $run yet"
done
rsync -a "$box:$runs/$chain.log" "$box:$runs/$chain-eval.launch.log" "$dest/" 2>/dev/null || true
if [ -s "$dest/$chain-eval/champion.json" ]; then
  path=$(python3 -c 'import json, sys; print(json.load(open(sys.argv[1]))["path"])' \
    "$dest/$chain-eval/champion.json")
  mkdir -p "$dest/$chain-eval/champion_policy"
  rsync -a "$box:$path/" "$dest/$chain-eval/champion_policy/"
  echo "champion: $path -> $dest/$chain-eval/champion_policy"
fi
tail -n 20 "$dest/$chain-eval/progress.log" 2>/dev/null || true
cat "$dest/$chain-eval/queue.exit" 2>/dev/null | sed 's/^/queue.exit: /' || true
