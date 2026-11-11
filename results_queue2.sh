#!/bin/bash
# The wrapped-set runs, started whenever fewer than 4 probe runs are playing.
cd "$(dirname "$0")"
PY=~/avesnova/boost-and-broadside-v2/.venv/bin/python
C=runs/2026-11-09-rl-deadreckon-split-hiddenphysics/policies
N=runs/2026-11-09-rl-absolute/policies
JOBS=(
"dm_champion_121M_wrapped|$C/step_121634816/uniform_average|--preset deathmatch --variants wrapped --environments 32 --seconds 30"
"dm_null_121M_wrapped|$N/step_121634816/uniform_average|--preset deathmatch --variants wrapped --environments 32 --seconds 30"
)
for job in "${JOBS[@]}"; do
  sleep 20
  while [ "$(pgrep -f 'bench.position_aliasing play' | wc -l)" -ge 4 ]; do sleep 10; done
  IFS='|' read -r name policy extra <<< "$job"
  mkdir -p results/$name
  ( OMP_NUM_THREADS=4 PYTHONPATH=src systemd-run --user --scope -q -p MemoryMax=8G \
    $PY -m bench.position_aliasing play "$policy" --out results/$name --device cpu $extra \
    > results/$name/stdout.log 2>&1 \
  && PYTHONPATH=src $PY -m bench.position_aliasing report --out results/$name > /dev/null 2>> results/$name/stdout.log ) &
done
wait
echo ALL DONE
