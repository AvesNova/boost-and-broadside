#!/bin/bash
# remote.sh NAME ARGS...: maneuvers-2's sizes.py on gpubox7 in tmux session ships2-NAME,
# logging to ships-r9m2/NAME.log (run from /workspace/ships2/repo)
name=$1; shift
cd /workspace/ships2/repo
tmux new-session -d -s "ships2-$name" "cd /workspace/ships2/repo && \
  PYTHONPATH=src:. PYTHONUNBUFFERED=1 TORCHINDUCTOR_COMPILE_THREADS=3 TRITON_DEFAULT_FP_FUSION=1 \
  TORCHINDUCTOR_CACHE_DIR=/workspace/ships2/cache/inductor TRITON_CACHE_DIR=/workspace/ships2/cache/triton \
  /workspace/ships2/venv/bin/python $* > docs/rewrite/sections/ships-r9m2/$name.log 2>&1"
