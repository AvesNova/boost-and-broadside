---
covers: [bench/ship_design_search.py]
---

# Runbook: a ship-design search

For an agent running one step of the [ship design](../ship-design.md) with no other
context. Every step is a command and a rule; if something does not match, stop and report
rather than improvising a fix to the method.

## Ground rules

- Work in the repository's worktree on branch `feat/real-scale`; commit and push after each
  step with your model's `Co-Authored-By` trailer; `git add` only the paths named below,
  never `git add -A`; never force-push, never merge into another branch, never a bare
  `git stash`; `git pull --rebase` before `git push`.
- Python as `uv run --no-sync python ...`. Tests under
  `systemd-run --user --scope -p MemoryMax=8G -q uv run --no-sync pytest ...`.
- One GPU job at a time, as a systemd user unit (below). Read the power profile first
  (`powerprofilesctl get`, expect `performance`) and note it.
- Progress notes (tried / worked / didn't / next, ETAs as clock times such as
  "ETA 15:40 (worst 16:10)") in the notes file your brief names.

## 0. Check the start (2 minutes, CPU)

    git status --short     # nothing staged, nothing modified but untracked run data
    nvidia-smi             # no other GPU process
    systemd-run --user --scope -p MemoryMax=8G -q uv run --no-sync pytest -q \
        tests/sections/test_ship_design.py -m "not slow and not gpu"

## 1. Search one stage's spec

A spec is a JSON file in `docs/explanation/game/ship-design/stage-N/` (the stage-1 specs:
`flight.json`, `guns.json`, `reach.json`). For each spec, one unit (about 30 s a generation
for a three-type cycle of 12 candidates; 30 generations, about 20 minutes with the first
compile):

    SPEC=flight   # then guns, then reach
    OUT=runs/ship-design/stage-1/$SPEC
    mkdir -p $OUT
    systemd-run --user --unit=design-$SPEC --working-directory=$PWD \
      -p StandardOutput=file:$PWD/$OUT/unit.log -p StandardError=file:$PWD/$OUT/unit.log \
      $PWD/.venv/bin/python -m bench.ship_design_search search \
        --spec docs/explanation/game/ship-design/stage-1/$SPEC.json --out $OUT --hours 2

Watch it no faster than every few minutes: `tail -2 $OUT/unit.log` prints each generation's
mean loss, edge scores, realism, best loss, sigma and the ETA. It resumes from
`$OUT/state.json` and `$OUT/search.jsonl` if rerun after a stop. The bound is `--hours`.

**Stopping rule** (read off `search.jsonl`; the search stops itself at the spec's
`generations`): stop early, by `systemctl --user stop design-$SPEC`, when the mean's loss
has been 0 in three generations in a row (every edge in the band and no penalty). Report a
search whose last five means all have a loss above 1 as **failed**: the mechanism does not
form a cycle within the envelope at these starts.

## 2. Check the result (about 10 minutes each)

    uv run --no-sync python -m bench.ship_design_search check \
      --spec docs/explanation/game/ship-design/stage-1/$SPEC.json --out $OUT --games 1024
    uv run --no-sync python -m bench.ship_design_search summary \
      --spec docs/explanation/game/ship-design/stage-1/$SPEC.json --out $OUT --card estimate

(Run the check as a unit like the search when the GPU is shared.) The summary prints the
verdict against the targets: every edge within 0.58-0.72, ratings within 35 Elo of the mean,
exploitability at most 0.02, a full-support Nash equilibrium, and (with seven types) the
mixed-team test. **Decision rule:** `"pass": true` -> step 3; else report the failing
measures and stop (do not rerun with other settings).

## 3. Exact cards, then the same check

The search plays instant estimates of the flight; the result is solved exactly before it is
accepted (CPU, one process a design, minutes each):

    uv run --no-sync python -m bench.ship_design_search export \
      --spec docs/explanation/game/ship-design/stage-1/$SPEC.json --out $OUT \
      --ship-set design-$SPEC
    uv run --no-sync python -m bench.ship_design_search check \
      --spec docs/explanation/game/ship-design/stage-1/$SPEC.json --out $OUT --games 1024 \
      --card design-$SPEC
    uv run --no-sync python -m bench.ship_design_search summary \
      --spec docs/explanation/game/ship-design/stage-1/$SPEC.json --out $OUT --card design-$SPEC

Every design's export line must say `converged True`; one that does not is reported with its
worst flown error. **Decision rule:** the exact cards' summary passes -> step 4; an edge that
left the band by the exact solve (it passed on the estimate) -> rerun the search from where it
stopped for five more generations (raise the spec's `generations` by 5) and repeat steps 2-3
once; then report.

## 4. Frontline

The same check with `--preset real_scale_frontline`, and its summary with
`--preset real_scale_frontline`. Report whether each edge keeps its direction (score above
0.5) and how far it moved.

## 5. Commit

    git add docs/explanation/game/ship-design/stage-N/ src/bnb/game/ships/design-$SPEC/
    git commit -m "data: ship design stage N <spec>: <pass or fail, the three edges>"
    git pull --rebase && git push origin feat/real-scale

The run directories (`runs/`) are not committed; copy `summary-*.json` and the check's
`check-*.jsonl` into `docs/explanation/game/ship-design/stage-N/<spec>/` first.
