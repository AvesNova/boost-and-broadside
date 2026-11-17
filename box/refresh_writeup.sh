#!/bin/bash
# refresh_writeup.sh [--no-fetch]: re-pull the dry runs' small result files from the 5090
# box and re-render the write-up's charts, values, README and review copy from them, on the
# laptop. Safe while the box's evaluator runs (it reads a snapshot of report.json; run it
# again later). It does not commit.
#
# Pulls (into DEST, default review/dry-run-3 of the v2 checkout): dry run 3's run records
# (dry-v26-s1-bc and -rl: run.json, config.json, metrics.jsonl) and the unified evaluator's
# report (dry-v26-v24-unified: report.json, spec.toml, progress.log). Never the box's
# secrets, never the games or policies.
#
# Draws, into docs/results with their values merged into docs/results/values.json:
#   health, behaviour, throughput, lanes   dry run 3's own training records
#   elo                    the unified field, both dry runs on one scale (ELO_MODES, default
#                          elimination; add frontline and deathmatch once their fields finish)
#   champion-break-even    each champion at 5 of its ships against 1 to 20 of Imugi's, once
#                          the evaluator has played such lineups (else it says so, draws none)
#   matchup                the values of both final BC checkpoints against the old Imugi
# The crossover fan, the break-even ratio, the team sizes, the suite and the averaging sweep
# stay dry run 1's (review/dry-run): render them with the commands in their PNGs' text chunks.
#
# Then README.md from its template, a strict MkDocs build into the review copy's site/,
# and README.md, LICENSE and docs/ (without hooks/ and rewrite/) copied beside it.
#
# BOX overrides the ssh host (box5090), RUNS_BOX the box's runs directory, VENV the Python
# environment with bnb's dependencies (the v2 checkout's .venv), REVIEW the review copy.
set -euo pipefail
repo=$(cd "$(dirname "$0")/.." && pwd)
v2=/home/vizia/avesnova/boost-and-broadside-v2
dest=${DEST:-$v2/review/dry-run-3}
review=${REVIEW:-$v2/review/writeup}
venv=${VENV:-$v2/.venv}
matchup=${MATCHUP:-/home/vizia/avesnova/watch-runs/bc-vs-old-imugi/table.txt}
box=${BOX:-box5090}
runs=${RUNS_BOX:-/workspace/runs}
chain=dry-v26-s1
unified=dry-v26-v24-unified
elo_modes=${ELO_MODES:-elimination}
labels=("dry-v26-s1=dry run 3 (v26)" "dry-v24-s1=dry run 1 (v24)")

quiet_ssh() {  # the box's login banner off stderr
  grep -v -e '^Welcome to vast.ai' -e '^Have fun!' -e '^AI agents: READ' >&2 || true
}
bnb() { env PYTHONPATH="$repo/src" "$venv/bin/bnb" "$@"; }

if [ "${1:-}" != "--no-fetch" ]; then
  mkdir -p "$dest/runs" "$dest/$unified"
  for run in "$chain-bc" "$chain-rl"; do
    mkdir -p "$dest/runs/$run"
    rsync -a "$box:$runs/$run/run.json" "$box:$runs/$run/config.json" \
      "$box:$runs/$run/metrics.jsonl" "$dest/runs/$run/" 2> >(quiet_ssh)
  done
  rsync -a "$box:$runs/$unified/report.json" "$box:$runs/$unified/spec.toml" \
    "$box:$runs/$unified/progress.log" "$dest/$unified/" 2> >(quiet_ssh)
  tail -n 2 "$dest/$unified/progress.log"
fi

cd "$repo"
records=("$dest/runs/$chain-bc" "$dest/runs/$chain-rl")
for chart in health behaviour throughput lanes; do
  bnb figures results "$chart" "${records[@]}"
done
read -r -a modes <<<"$elo_modes"
bnb figures results elo "$dest/$unified/report.json" --modes "${modes[@]}" --chain "$chain" \
  --labels "${labels[@]}"
bnb figures results champion-break-even "$dest/$unified/report.json" --modes elimination \
  --chain "$chain" --labels "${labels[@]}"
if [ -s "$matchup" ]; then
  bnb figures results matchup "$matchup" --labels A=dry-v26-s1-bc B=dry-v24-s1-bc \
    --opponent imugi_v24
fi

PYTHONPATH="$repo/src:$repo" "$venv/bin/python" docs/hooks/placeholders.py readme
mkdir -p "$review"
PYTHONPATH="$repo/src:$repo" NO_MKDOCS_2_WARNING=1 "$venv/bin/python" -m mkdocs build \
  --strict --quiet -d "$review/site"
cp README.md LICENSE "$review/"
rsync -a --delete --exclude hooks/ --exclude rewrite/ docs/ "$review/docs/"
echo "refreshed: docs/results, README.md and $review (commit them on docs/v2-writeup)"
