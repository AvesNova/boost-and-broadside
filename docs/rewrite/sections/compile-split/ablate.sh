#!/bin/bash
# ablate.sh BACKEND NAME...: the rollout's compiles (collect only) of the ablation configs
# (ablations.py) one after another, fresh caches each; BACKEND eager: Dynamo's trace alone
# (op counts), inductor: the real compile. One JSON line each in $ROOT/records/ablations.jsonl.
set -uo pipefail
BACKEND=$1; shift
ROOT=/workspace/cs; cd /workspace/boost-and-broadside
for name in "$@"; do
  CACHE=$ROOT/cache/ablation-$name; rm -rf "$CACHE"; mkdir -p "$CACHE"
  LOG=$ROOT/logs/ablation-$BACKEND-$name-$(date +%H%M)
  START=$(date +%s.%N)
  env TORCHINDUCTOR_CACHE_DIR=$CACHE/inductor TRITON_CACHE_DIR=$CACHE/triton \
    TORCHINDUCTOR_COMPILE_THREADS=${THREADS:-3} OMP_NUM_THREADS=3 TRITON_DEFAULT_FP_FUSION=1 \
    $([ "$BACKEND" = eager ] && echo BENCH_BACKEND=eager) \
    .venv/bin/python bench/compile_baseline.py train config=$ROOT/configs/ablations/$name.json \
    steps=1 learn=0 process_start=$START > $LOG.json 2> $LOG.log
  .venv/bin/python -c "
import json,sys
r=json.load(open('$LOG.json'))
c={}
for e in r['events']:
    if e['kind']=='compile':
        x=c.setdefault(e['label'],[0.0,0]); x[0]+=e['seconds']; x[1]+=e.get('ops',0)
print(json.dumps(dict(name='$name', backend='$BACKEND', compiles={k:[round(v[0],1),v[1]] for k,v in c.items()}, phases={k:round(v,1) for k,v in r['phases'].items()})))
" >> $ROOT/records/ablations.jsonl || echo "{\"name\": \"$name\", \"backend\": \"$BACKEND\", \"failed\": \"$LOG.log\"}" >> $ROOT/records/ablations.jsonl
  echo "ablation $name done: $(tail -1 $ROOT/records/ablations.jsonl | head -c 400)"
done
