#!/bin/bash
# check_config.sh bc|rl: the arm's resolved configuration (`bnb train ... --print-config`)
# against the champion's recorded config.json (champion/<stage>-config.json); exits 1 on any
# difference besides the expected ones: yemong.heads (2 -> 1), and for BC stop.budget_steps
# (the champion's was 30M at creation, raised to 60M then 120M by its resumes), and for BC
# yemong.physics_hidden_fade absent -> False (the switch was merged after the champion's BC
# started on fadd6065; False is its default, the champion's later segments ran with it off).
set -u
cd /workspace/boost-and-broadside
# A hold (runs/HOLD-RL): the pipeline stops before RL starts (this check fails, the driver
# exits 2, which supervisor does not restart).
if [ "$1" = rl ] && [ -f runs/HOLD-RL ]; then
  echo "$(date -u +%FT%TZ) RL held (runs/HOLD-RL): the pipeline stops after BC" | tee -a runs/pipeline.log
  exit 1
fi
stage=$1
here=docs/rewrite/sections/gpubox2
out=runs/check-config-$stage.json
mkdir -p runs
if [ "$stage" = bc ]; then
  eval "$(sed -n '/^BC_ARGS=(/,/)$/p' $here/pipeline.sh)"
  .venv/bin/bnb train --name check "${BC_ARGS[@]}" --print-config > $out || exit 1
  allowed="yemong.heads stop.budget_steps yemong.physics_hidden_fade"
else
  INIT=runs/2026-11-12-bc-onehead/policies/step_120586240/uniform_average
  eval "$(sed -n '/^RL_ARGS=(/,/)$/p' $here/pipeline.sh)"
  .venv/bin/bnb train --name check "${RL_ARGS[@]}" --print-config > $out || exit 1
  allowed="yemong.heads"
fi
.venv/bin/python - $here/champion/$stage-config.json $out "$allowed" <<'PY' | tee -a runs/pipeline.log
import json, sys
def flat(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        out.update(flat(v, key + ".") if isinstance(v, dict) else {key: v})
    return out
champion, arm = (flat(json.load(open(p))) for p in sys.argv[1:3])
allowed = set(sys.argv[3].split())
diffs = sorted(k for k in champion.keys() | arm.keys() if champion.get(k, "<absent>") != arm.get(k, "<absent>"))
bad = [k for k in diffs if k not in allowed]
for k in diffs:
    print(f"config diff {k}: champion {champion.get(k, '<absent>')!r} arm {arm.get(k, '<absent>')!r}"
          + ("" if k in allowed else "  <-- UNEXPECTED"))
print(f"config check {sys.argv[2]}: {len(diffs)} differences, {len(bad)} unexpected")
sys.exit(1 if bad else 0)
PY
status=${PIPESTATUS[0]}
[ $status = 0 ] && touch runs/$stage-config-checked
exit $status
