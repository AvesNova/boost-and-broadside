#!/bin/bash
# remote.sh NAME ARGS...: a python run on gpubox7 in tmux, logging to NAME/log.txt (the
# venv, caches and clone under /workspace/armour).
name=$1; shift
cd /workspace/armour/repo
mkdir -p docs/rewrite/sections/tank-armour/$name
env="TORCHINDUCTOR_CACHE_DIR=/workspace/armour/cache/inductor TRITON_CACHE_DIR=/workspace/armour/cache/triton TORCHINDUCTOR_COMPILE_THREADS=3 TRITON_DEFAULT_FP_FUSION=1 PYTHONPATH=src:. PYTHONUNBUFFERED=1"
tmux new-session -d -s "armour-$name" \
  "env $env .venv/bin/python $* > docs/rewrite/sections/tank-armour/$name/log.txt 2>&1"
