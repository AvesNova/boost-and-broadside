#!/bin/bash
# unified_check.sh CHAIN [HOURS]: the 5090 check of the unified evaluator (`bnb unified`,
# docs/guides/unified-evaluation.md) on the dry run's policies, the box queue's targets
# (box/eval_queue.sh: suite, maps, curves, break-even, team size) in one run:
#   suite      the champion against Imugi, every head-to-head condition (score to +-0.02)
#   maps       Elo of the champion at 0.5x and 2x maps, each 5v5 mode (+-45)
#   curves     5v5 Elo of the scheduled snapshots per mode (+-60)
#   breakeven  the champion's break-even log ratio per mode (+-0.1, at most 2048 games)
#   teamsize   the champion at n v n, n v 1.5n, n v 2n, n = 1..10, per mode (+-0.05)
# Launch detached; it takes the GPU lock:
#   ssh -f box5090 'nohup setsid bash /workspace/wt/ueval/box/unified_check.sh dry-v24-s1 2 \
#     > /workspace/runs/dry-v24-s1-unified.launch.log 2>&1 < /dev/null &'
# A compile-cache archive (docs/rewrite/sections/launch/cache.sh export) in $CACHE_ARCHIVE
# is imported first, so a box of the same GPU model starts warm. Writes $RUNS/CHAIN-unified/
# (progress.log with games per hour, GPU utilisation and the ETA as a clock time;
# games.jsonl, report.json, sizing.json); a rerun resumes from games.jsonl.
#
# What to expect (the laptop's measurement scaled, docs/rewrite/sections/unified-eval/
# results.md): the 5090 has 170 SMs at about 2.4 GHz against the laptop 4070's 36 at about
# 2.2 GHz, 5.2x by SMs times clock; the measured training-rollout ratio is 3.8x. Each
# bucket's sizing.json should hold about 4x the laptop's environments (32 GB against 8),
# and games per hour should come out 3.8-5x the laptop's per bucket. The GPU should stay
# above 90% busy once warm (progress.log's GPU column); lower means the host loop is the
# limit (check the python process's CPU in top).
set -uo pipefail
chain=${1:?usage: unified_check.sh CHAIN [HOURS]}
hours=${2:-2}
EVAL_ENV=${EVAL_ENV:-/workspace/env.sh}
[ "$EVAL_ENV" != none ] && source "$EVAL_ENV"
UEVAL=${UEVAL:-/workspace/wt/ueval}
PY=${PY:-/workspace/venv/bin/python}
RUNS=${RUNS:-/workspace/runs}
export PYTHONPATH=$UEVAL/src:$UEVAL
out=$RUNS/$chain-unified
mkdir -p "$out"
if [ -n "${CACHE_ARCHIVE:-}" ] && [ -s "$CACHE_ARCHIVE" ]; then
  bash "$UEVAL/docs/rewrite/sections/launch/cache.sh" import "$CACHE_ARCHIVE"
fi
champion=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["path"])' \
  "$RUNS/$chain-eval/champion.json" 2>/dev/null) || champion=""
if [ -z "$champion" ]; then
  last=$(ls "$RUNS/$chain-rl/policies" | sed -n 's/^step_\([0-9]*\)$/\1/p' | sort -n | tail -1)
  champion=$RUNS/$chain-rl/policies/step_$last/exponential_average
fi
"$PY" - "$chain" "$champion" "$RUNS" > "$out/spec.toml" <<'PY'
import pathlib, sys
chain, champion, runs = sys.argv[1], sys.argv[2], pathlib.Path(sys.argv[3])
def steps(run):
    return sorted(int(p.name[5:]) for p in (runs / run / "policies").glob("step_*") if p.name[5:].isdigit() and int(p.name[5:]) > 0)
rl, bc = steps(f"{chain}-rl"), steps(f"{chain}-bc")
def at(found, millions):
    return next((s for s in found if s >= millions * 1_000_000), found[-1])
snapshots = {f"rl_{m}m": f"{chain}-rl/policies/step_{at(rl, m)}/exponential_average" for m in (5, 10, 20, 35, 50, 75)}
snapshots["rl_last"] = f"{chain}-rl/policies/step_{rl[-1]}/exponential_average"
for share in (25, 50, 75, 100):
    snapshots[f"bc_{share}"] = f"{chain}-bc/policies/step_{bc[max(len(bc) * share // 100 - 1, 0)]}/exponential_average"
modes = {"frontline": "real_scale_frontline", "deathmatch": "real_scale_deathmatch", "elimination": "real_scale_elimination"}
print('lineup_types = ["Yongma", "Jujak"]\nbuckets = [5, 10, 20]\nseed = 0\n\n[players]')
print(f'champion = "{champion}"')
for name, path in snapshots.items():
    print(f'{name} = "{path}"')
field = '["scripted", "champion", ' + ", ".join(f'"{n}"' for n in snapshots) + "]"
for mode, preset in modes.items():
    sizes = [[5, 5], [10, 10]] + ([[1, 1], [2, 2]] if mode != "frontline" else [])
    print(f'\n[[win_rate]]\nname = "suite"\npreset = "{preset}"\nfirst = "champion"\nsecond = "scripted"\nlineups = {sizes}\nstderr = 0.02\nbudget = 512')
    for scale in (0.5, 2.0):
        print(f'\n[[elo]]\nname = "maps/{mode}/{scale}"\npreset = "{preset}"\narena_scale = {scale}\nplayers = ["scripted", "champion"]\nstderr = 45.0\nbudget = 4096')
    print(f'\n[[elo]]\nname = "curves/{mode}"\npreset = "{preset}"\nplayers = {field}\nstderr = 60.0\nbudget = 12288')
    print(f'\n[[break_even]]\npreset = "{preset}"\npolicy = "champion"\nopponent = "scripted"\nstderr = 0.1\nbudget = 2048')
    lineups = []
    for n in range(1, 11):
        lineups.append([n, n])
        if n % 2 == 0:
            lineups.append([n, 3 * n // 2])
        lineups.append([n, 2 * n])
    if mode == "frontline":
        lineups = [l for l in lineups if min(l) >= 3]
    print(f'\n[[win_rate]]\nname = "teamsize"\npreset = "{preset}"\nfirst = "champion"\nsecond = "scripted"\nlineups = {lineups}\nstderr = 0.05\nbudget = 128')
PY
echo "spec: $out/spec.toml (champion $champion)"
exec 9> "${GPU_LOCK:-/tmp/gpu.lock}"
flock 9
"$PY" -c 'import sys; from bnb.cli import main; sys.argv[0] = "bnb"; sys.exit(main())' \
  unified "$out/spec.toml" --out "$out" --runs-directory "$RUNS" --hours "$hours"
code=$?
echo "$code" > "$out/exit"
if [ -n "${CACHE_EXPORT:-}" ]; then
  bash "$UEVAL/docs/rewrite/sections/launch/cache.sh" export "$CACHE_EXPORT"
fi
exit $code
