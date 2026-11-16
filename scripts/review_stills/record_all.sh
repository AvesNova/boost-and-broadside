#!/bin/bash
# One game at a time, CPU only, under a 6G scope.
cd /home/vizia/avesnova/bnb-istills
export PYTHONPATH=$PWD/src:$PWD CUDA_VISIBLE_DEVICES=""
PY=/home/vizia/avesnova/boost-and-broadside-v2/.venv/bin/python
R=scratch_records
run() { [ -f "$R/$1.pt" ] && { echo "skip $1"; return; }
  echo "$(date +%H:%M:%S) start $1"
  systemd-run --user --scope -q -p MemoryMax=6G $PY scripts/review_stills/record.py "${@:2}" "$R/$1.pt" 14 || echo "FAILED $1"; }
run dm1v1_s0 real_scale_deathmatch 1 density 0 180
run dm1v1_s1 real_scale_deathmatch 1 density 1 180
run dm2v2_s0 real_scale_deathmatch 2 density 0 150
run dm5v5_s0 real_scale_deathmatch 5 preset 0 240
run elim5v5_s0 real_scale_elimination 5 preset 0 300
echo "$(date +%H:%M:%S) all done"
