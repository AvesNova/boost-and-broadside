# Brief: the extrapolation baseline without the acceleration inputs

**Decided by the author (2026-11-08 09:35):** each round builds on the winner. Arm 1 (the
acceleration inputs) did not beat C-fixed, so arms 2-4 must read C-fixed's inputs: no
acceleration codes in the ship token. Arm 4's extrapolation baseline still needs last tick's
rates, but as a baseline computed in code, not as a model input. Today
`YemongConfig.delta_baseline == "extrapolation"` requires `acceleration_inputs` (model.py
validation): decouple them.

## What to build

- With `delta_baseline=extrapolation` and `acceleration_inputs=false`, the baseline gets its
  rates exactly as today (visible rows from the observation's acceleration columns; hidden
  rows the held last-sighting rates, decaying with `extrapolation_decay`), but the ship token
  carries no acceleration codes or flag. Carry what the baseline needs (`Carry.motion`,
  `Carry.held`) whenever either switch needs it.
- With `acceleration_inputs=true` everything is unchanged bit for bit (arms 1, 2, 2b, 3b/4b
  digests; record any you need before changing code, as earlier agents did).
- Remove the validation; update the docs (`yemong.md`: the switch table, and the note that a
  residual from a baseline whose rates the model cannot see is the author's choice here).

## Tests (CPU)

- The new combination builds, steps, unrolls (= a loop of `step`), trains a BC and a PPO step;
  its baseline equals the inputs-on baseline on the same recorded segment; its token width
  equals C-fixed's plus the delta-head arms' own additions (no acceleration columns).
- Every existing golden unchanged; the quick tier (`pytest -m "not slow and not gpu"`), the
  slow dashboards test, ruff.

## Where you work

Worktree `/home/vizia/avesnova/bnb-extrap-noinputs`, branch `fix/extrapolation-without-inputs`
(from v2); scratch `/tmp/bnb-scratch-extrap-noinputs`. CPU only (`CUDA_VISIBLE_DEVICES=""`,
`nice -n 15`, at most 8 threads); never the bench lock, queue file, units or run directories.
Long commands under `systemd-run --user --scope -p MemoryMax=8G -p MemorySwapMax=0`. Never
`git stash`. `uv run --no-sync ...`. Progress notes in the scratch `progress.md` (`## HH:MM`,
1-3 bullets, clock-time ETAs). Commit and push (author Julian Kizanis <julianzia@gmail.com>,
`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`). Final message: what changed, the
exact `--set` line for arm 4c (arm 3c's line, `--set yemong.acceleration_inputs=false
--set yemong.delta_split=true --set learner.correction=0.0033`, with
`delta_baseline=extrapolation`), tests, the time.
