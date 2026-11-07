# Brief: acceleration inputs (the round-2 arm 1: C-fixed + accelerations)

**Decided by the author (2026-11-08):** the model gets each ship's **linear and angular
acceleration** as inputs (physical accelerations, not the enemies' previous actions). This
arm is C-fixed (`--set yemong.order=temporal_first`, `spread_fit` "coarsest", the absolute
next-state head) plus these inputs and nothing else. Later arms (persistence and dead-
reckoning delta heads) build on it; they are not part of this brief.

Why: the physics is deterministic, a = f(state, control), so a visible ship's change over the
last tick reveals the control it used, and controls are sticky. Today the model can only
difference two state codes across ticks, and the velocity code's bins (2.5 (1 + |v|/50) m/s:
17.5 m/s at 300 m/s) are wider than any one-tick change (at most ~7 m/s from thrust), so it
cannot resolve accelerations. The later dead-reckoning arm also needs them as inputs (a
residual from a baseline the model cannot see is unexplained variance).

Read first: `bnb.models.yemong` (`inputs.py`, `state_code.py`, `codes.py`, `belief.py`,
`model.py`, `network.py`), the observation contract (`bnb.contracts.obs`,
`bnb.game.perception.observe`), `docs/rewrite/sections/state-deltas.md` (the measured
one-tick changes and hard bounds, table "Summary" and 1.2), and `bnb.models.yemong.delta`
(the symlog HL-Gauss code to reuse).

## The quantities

Per ship row, from the rows the model reads (truth for allies and visible enemies, the
belief's forecast for hidden enemies), at tick t:

| Input | Definition | Frame |
|---|---|---|
| along | (v_t - v_{t-1}) / dt, component along v_t | velocity frame at t |
| across | the same, component across v_t (left positive) | velocity frame at t |
| angular | (omega_t - omega_{t-1}) / dt, omega the nose rate | - |

- The velocity frame: the unit vector of v_t. Below 1 m/s use the heading at t instead, and
  count it (`inputs/acceleration_slow_frame`, the fraction of rows).
- **Zero** (and a validity flag 0) when the difference is not a real tick of motion: dead at
  t or t-1, respawned or spawned between them, the first tick of a game, and **just spotted**
  (hidden at t-1, visible at t: the difference would be truth minus belief, not motion).
  Just hidden (visible at t-1, hidden at t) is fine: the belief's forecast minus the last
  truth.
- No respawn teleports, no wrap (v2 has none).
- Hidden ships carry no spread for these (mean only).

## Where it is computed

In the model, from the rows it already reads: the previous tick's row (velocity, nose rate,
alive/visible/spawn state) is carried in the rollout's recurrent state for `step`, and
shifted in `unroll`; the learner's sequence start must get the true previous row (from the
stored recurrent state or the replay). `unroll` must equal a loop of `step`. If computing it
in the observation is clearly simpler and leaks nothing (the rule above: only when the team
saw the ship at both ticks), say why and do that instead.

## The code

The symlog HL-Gauss code of `delta.py` (exact-zero centre, sigma 0.75 bins, 2 padding bins
per side, clamp, no fold), **31 real bins** per component. Knee at the measured median
|change| / dt, range at the hard one-tick bound / dt (state-deltas: velocity persistence,
nose-rate persistence), excluding respawns. Say the knees and ranges you chose and how many
samples fall in the padding on the recorded segments (`/tmp/bnb-scratch-deltas/segments`,
read only). The validity flag is its own input. Embedded with the other ship inputs, the
same way.

## Switch

`YemongConfig.acceleration_inputs: bool`, default False, so every existing build record
(C-fixed's included: it must still load and play for the tournament) is unchanged bit for
bit. Recorded in the build and config.json. The arm runs with
`--set yemong.acceleration_inputs=true`.

## Diagnostics (logged)

Per component: the input's mean |value| for visible and hidden rows, the fraction invalid,
the slow-frame fraction, the padding share.

## Tests (CPU)

- The quantities on a scripted trajectory: a known straight acceleration gives along = a,
  across 0; a steady turn gives across = |v| x path rate; the frame fallback below 1 m/s.
- Invalid cases zero with flag 0: death, respawn, spawn, first tick, just spotted.
- `unroll` equals a loop of `step` with the switch on; sequence start in the learner uses the
  true previous row; `false` unchanged bit for bit (a recorded step/unroll); a policy-file
  round trip; BC and PPO learner steps run.
- The code's round trip (reuse delta's tests where they fit).
- The quick tier (`pytest -m "not slow and not gpu"`), ruff, docs (`yemong.md`, the inputs
  section).

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-acceleration`, branch `feat/acceleration-inputs`
  (from v2). Scratch `/tmp/bnb-scratch-acceleration`.
- **CPU only:** `CUDA_VISIBLE_DEVICES=""`, `nice -n 15`, at most 8 threads; a BC run owns the
  GPU. Never take the bench lock. No training runs.
- **Memory:** anything long runs under `systemd-run --user --scope -p MemoryMax=8G
  -p MemorySwapMax=0`, streaming to disk.
- **Git:** never use `git stash` (the repository has old shared stashes).
- **Python:** `uv sync --all-extras`, then `uv run --no-sync ...`.
- **Progress notes:** `/tmp/bnb-scratch-acceleration/progress.md`, `## HH:MM` from
  `date +%H:%M`, 1-3 bullets, ETAs as clock times. Start with a plan note.
- If something here is wrong, missing or ambiguous, say so in a note and in the final
  message; make a sensible choice and keep going.

## Done

- Commit and push to `feat/acceleration-inputs` in logical commits. Author Julian Kizanis
  <julianzia@gmail.com>; end each message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Final message: where it is computed and why, the knees, ranges and padding shares, the
  tests, the exact `--set` line, anything open, the time.
