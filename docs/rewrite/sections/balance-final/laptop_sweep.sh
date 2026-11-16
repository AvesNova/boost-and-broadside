#!/bin/bash
# The laptop sweep wrapper: GPU flock + a systemd unit (MemoryMax 8G). usage: laptop_sweep.sh unit mode candidates out [sweep args]
# usage: sweep.sh unit mode candidates out [extra args]; candidates/out relative to the main worktree.
# The code (src, sweep.py, before/) comes from the sweep worktree at the base card, so the
# main worktree's src may carry the after card.
unit=$1; mode=$2; cand=$3; out=$4; shift 4
MAIN=/home/vizia/avesnova/bnb-balance-final
CODE=/home/vizia/avesnova/bnb-balance-final
cd $CODE
flock /tmp/claude-1000/bnb-gpu.lock systemd-run --user --unit=$unit -p MemoryMax=8G --wait --collect -E PATH=$PATH -E PYTHONPATH=$CODE/src:$CODE --working-directory=$CODE uv run --no-sync python docs/rewrite/sections/balance-final/sweep.py --mode $mode --candidates $MAIN/$cand --out $MAIN/$out "$@"
