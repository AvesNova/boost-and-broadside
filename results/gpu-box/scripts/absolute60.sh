#!/bin/bash
# absolute (C-fixed) to 60M, the 60M round's baseline (the author). Its hub copy of the 30.4M
# checkpoint fails its own record (training.safetensors: the only mismatch of 38 checked), so it
# resumes from 23.1M (verified by SHA-256 against its record); the 30.4M policy is the original.
set -u
cd /workspace/boost-and-broadside
set -a; . /workspace/.env; set +a
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
A=2026-11-09-bc-absolute
echo "$(date -u +%H:%M) start resume $A to 60M from 23.1M" >> runs/arms.log
if .venv/bin/bnb train --resume $A --at 23068672 --bc-budget-steps 60000000 >> runs/$A.out 2>&1; then
  echo "$(date -u +%H:%M) done resume $A" >> runs/arms.log
else
  echo "$(date -u +%H:%M) FAILED resume $A (runs/$A.out)" >> runs/arms.log
fi
docs/rewrite/sections/gpu-box/sync.sh > /dev/null 2>&1
