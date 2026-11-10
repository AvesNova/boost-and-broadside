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
  (`powerprofilesctl get`, expect performance) and note it.
- Progress notes (tried / worked / didn't / next, ETAs as clock times such as
  "ETA 15:40 (worst 16:10)") in the notes file your brief names.

## 0. Check the start (2 minutes, CPU)

    git status --short     # nothing staged, nothing modified but untracked run data
    nvidia-smi             # no other GPU process
    systemd-run --user --scope -p MemoryMax=8G -q uv run --no-sync pytest -q \
        tests/sections/test_ship_design.py -m "not slow and not gpu"

## A case study (one trade-off; about 10 to 20 minutes of GPU)

A study is a spec in `docs/explanation/game/case-studies/` ([Case studies](../case-studies.md)
explains the fields; copy the nearest existing spec and change only the variants or the
airframes). At most 36 slots (airframes x variants).

    NAME=guns-la7   # the spec's file name without .json
    mkdir -p runs/case-studies
    systemd-run --user --unit=case-$NAME --working-directory=$PWD \
      -p StandardOutput=file:$PWD/runs/case-studies/$NAME.log \
      -p StandardError=file:$PWD/runs/case-studies/$NAME.log \
      $PWD/.venv/bin/python -m bench.case_study play \
        --spec docs/explanation/game/case-studies/$NAME.json --out runs/case-studies/$NAME --hours 1
    # when the unit has ended:
    uv run --no-sync python -m bench.case_study report \
      --spec docs/explanation/game/case-studies/$NAME.json --out runs/case-studies/$NAME

Each batch prints the largest standard error and an ETA; the play stops at the spec's
`target_stderr` (20 Elo) or `budget`, or at `--hours`, and resumes from its record if rerun.
Copy `report.json` and `batches.jsonl` into `docs/explanation/game/case-studies/$NAME/` and add
a section to the case-studies page in the shape of the others: **Question**, **Design** (the
airframe, the variants, the games), the table of Elo against the baseline, the line's Elo per
doubling, and **What it says** (plain statements of what is steep, what is free, what
interacts, whether any majority cycle has a weakest edge above 0.55). Do not draw a
conclusion the numbers do not carry: a difference under two standard errors is "within
noise".

## 1. Search one stage's spec

A spec is a JSON file in a stage's directory, `docs/explanation/game/ship-design/stage-1/` for
the first stage (the stage-1 specs:
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

## 3b. Continuing a search

To continue from a search's best mean with other settings (more games, a fixed direction),
write a new spec whose designs' `start` values are that mean's knobs (`search.jsonl`, the
row with `"mean": true` of lowest loss in the last five generations), `edges` in the
direction the exact cards went round, `"edges_are_cycle": true`, and a smaller `sigma`
(`stage-1/seed-2.json` continues `seed.json` this way). Run steps 1 to 3 with it.

## The seven types (after three cycles pass)

    uv run --no-sync python -m bench.ship_design_search compose \
      --specs A.json,B.json,C.json --outs runs/.../A,runs/.../B,runs/.../C \
      --names N1,N2,N3,N4,N5,N6,N7 --write docs/explanation/game/ship-design/stage-2/seven.json

then steps 1 to 4 with that spec (21 edges, 5 candidates a generation, about 3 minutes a
generation). Its check includes the mixed-team test: the summary's `mixed_pass` must be true.

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
