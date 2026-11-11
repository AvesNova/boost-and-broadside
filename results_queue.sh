#!/bin/bash
# The probe's runs, three at a time: name|policy|extra args
cd "$(dirname "$0")"
PY=~/avesnova/boost-and-broadside-v2/.venv/bin/python
C=runs/2026-11-09-rl-deadreckon-split-hiddenphysics/policies
N=runs/2026-11-09-rl-absolute/policies
JOBS=(
"dm_champion_121M|$C/step_121634816/uniform_average|--preset deathmatch"
"dm_null_121M|$N/step_121634816/uniform_average|--preset deathmatch"
"fl_champion_121M|$C/step_121634816/uniform_average|--preset frontline"
"fl_null_121M|$N/step_121634816/uniform_average|--preset frontline"
"dm_champion_60M|$C/step_60817408/uniform_average|--preset deathmatch"
"dm_null_60M|$N/step_60817408/uniform_average|--preset deathmatch"
"dm_champion_121M_arena12500|$C/step_121634816/uniform_average|--preset deathmatch --arena 12500"
"dm_champion_121M_fog|$C/step_121634816/uniform_average|--preset deathmatch --fog 1.0"
"dm_champion_121M_bf16|$C/step_121634816/uniform_average|--preset deathmatch --bf16 --environments 16 --seconds 20"
)
run() {
  IFS='|' read -r name policy extra <<< "$1"
  [ -f results/$name/report.md ] && return
  mkdir -p results/$name
  OMP_NUM_THREADS=4 PYTHONPATH=src systemd-run --user --scope -q -p MemoryMax=8G \
    $PY -m bench.position_aliasing play "$policy" --out results/$name \
    --environments 64 --seconds 40 --device cpu $extra > results/$name/stdout.log 2>&1 \
  && PYTHONPATH=src $PY -m bench.position_aliasing report --out results/$name > /dev/null 2>> results/$name/stdout.log
}
for job in "${JOBS[@]}"; do
  while [ "$(jobs -rp | wc -l)" -ge 4 ]; do sleep 5; done
  run "$job" &
  sleep 2
done
wait
echo ALL DONE
