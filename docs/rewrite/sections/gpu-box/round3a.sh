#!/bin/bash
# Round 3, part a (after tournament 1): the Brier check, then deadreckon-split-fastdecay.
# The Fourier and hidden-physics arms follow in part b once their builds are merged.
set -u
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
LOG=runs/arms.log
say() { echo "$(date -u +%H:%M) $*" | tee -a "$LOG"; }

until grep -q "tournament1 done\|tournament1 FAILED (runs/ratings-round2.out)$" <(tail -n 1 "$LOG"); do
  sleep 30
done

# 1. Brier check: absolute (C-fixed) to 10.5M on UNMODIFIED v2 (fp32 shot head, old _step),
# with only the two training-neutral settings; compare with the 3090 C-fixed at 10.5M.
CHECK=2026-11-09-bc-absolute-v2clean-check
say "start $CHECK (Brier check on unmodified v2)"
if PYTHONPATH=/workspace/bnb-v2-clean/src .venv/bin/python -c \
    "import sys; from bnb.cli import main; sys.argv = ['bnb'] + sys.argv[1:]; main()" \
    train --name $CHECK --runs-directory /workspace/boost-and-broadside/runs \
    --preset mixed --objective bc --bc-budget-steps 10485760 --set yemong.order=temporal_first \
    --micro-batch-rows 2560 --set host_store=false > runs/$CHECK.out 2>&1; then
  say "done $CHECK"
else
  say "FAILED $CHECK (runs/$CHECK.out)"
fi

# 2. deadreckon-split with the held rates decaying 2.5x faster (tau 0.5 -> 0.2 s).
FAST=2026-11-09-bc-deadreckon-split-fastdecay-3090
say "start $FAST"
if .venv/bin/bnb train --name $FAST --preset mixed --objective bc --bc-budget-steps 30000000 \
    --micro-batch-rows 2560 --set host_store=false \
    --set yemong.order=temporal_first --set yemong.next_state=delta \
    --set yemong.delta_baseline=extrapolation --set yemong.spread_inputs=scalar \
    --set learner.next_state=0.0033 --set yemong.delta_split=true \
    --set learner.correction=0.0033 --set yemong.extrapolation_decay=0.2 \
    > runs/$FAST.out 2>&1; then
  say "done $FAST"
else
  say "FAILED $FAST (runs/$FAST.out)"
fi
say "round3a done"
