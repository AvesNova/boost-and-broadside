#!/bin/bash
# remote.sh NAME ARGS...: a python run on gpubox7 in tmux (session arch-NAME), logging to
# NAME/log.txt; the clone, venv and caches under /workspace/arch. ARGS keep their quoting.
name=$1; shift
cd /workspace/arch/repo
mkdir -p docs/rewrite/sections/archetypes/$name
args=$(printf '%q ' "$@")
env="TORCHINDUCTOR_CACHE_DIR=/workspace/arch/cache/inductor TRITON_CACHE_DIR=/workspace/arch/cache/triton TORCHINDUCTOR_COMPILE_THREADS=3 TRITON_DEFAULT_FP_FUSION=1 PYTHONPATH=src:. PYTHONUNBUFFERED=1"
tmux new-session -d -s "arch-$name" \
  "env $env .venv/bin/python $args > docs/rewrite/sections/archetypes/$name/log.txt 2>&1"
