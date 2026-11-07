# Brief: round-2 arm 2, persistence delta heads (one head on every row)

**Decided by the author (2026-11-08, `round2-plan.md` section 4 and O1-O10):** arm 2 is arm 1
(the acceleration inputs, now read from the observation) plus a next-state head that predicts
each quantity's **change from persistence** (nothing changes), trained on **every row**
(allies, visible and hidden enemies, as round 1), with the belief's spread carried as a full
covariance through the physics and fed back as **ellipse** inputs. Arms 3 and 4 (a separate
error head; extrapolation baselines) come later and build on this; leave room for them, do
not build them.

Read first: `round2-plan.md` (4, 5.2-5.6, 7), `round2-log.md`, `baselines-bench.md` and its
tables (the persistence rows: your knees and ranges), `delta-heads-round2-notes.md` (what
round 1 got wrong), `delta-heads-brief.md` (round 1's head, most of which you reuse), the
code: `bnb.models.yemong` (`delta.py`, `state_code.py`, `codes.py`, `belief.py`, `heads.py`,
`inputs.py`, `model.py`), `bnb.learn.labels.next_state_targets`, the next-state term in
`bnb.learn.ppo`.

## Switches (no legacy)

- `yemong.next_state`: "absolute" (C-fixed, arm 1; unchanged bit for bit) or "delta".
- `yemong.delta_baseline`: "persistence" (this arm). Arm 4 will add "extrapolation".
  **Round 1's rule baselines are removed** (no arm uses them; the aborted round-1 arm's
  policies will no longer load: say so in the docs).
- `yemong.spread_inputs`: "scalar" (today: one position sigma, velocity xx/xy/yy) or
  "ellipse" (below). Default "scalar".
- Leave a clear seam for arm 3 (`delta_split`: the physics head on visible rows only plus a
  separate error head for hidden rows); do not implement it.

Arm 2 runs with `--set yemong.order=temporal_first --set yemong.acceleration_inputs=true
--set yemong.next_state=delta --set yemong.delta_baseline=persistence
--set yemong.spread_inputs=ellipse` (say the exact line).

## Targets

- Every quantity in the state code: truth(t+1) - truth(t) (hidden rows: open loop, the same,
  as round 1).
- Position and velocity: two components in the **velocity frame at t** (along, across the
  track; below 1 m/s the nose frame, counted). Heading wrapped.
- **Countdowns and shield** under persistence: the change itself (a countdown's steady -dt,
  a reset's jump; the shield's regeneration, a hit's drop). The model learns the linear
  regeneration and the countdown (the author wants that in this arm).
- Codes: symlog HL-Gauss (sigma 0.75 bins), exact-zero centre, 2 padding bins per side, clamp
  (as round 1). Knee and range per component from the bench's persistence rows. **Position
  along the track** is the whole move (about |v| dt: median 2.6 m, max 8 m, never near 0):
  choose its bins so the decoded value resolves the move well (bin width at the median
  <= 5 cm if the cost allows, up to about 201 bins; measure the learner's memory and update
  time for your choice and say them). Others: 51 (position, velocity) and 31 (the rest), or
  say why not.
- Output bias initialised at the marginal, as round 1.

## The belief advance (hidden ships)

- Mean: belief + the decoded change, rotated from the belief's velocity frame to world.
- **Most likely, not the expected value** (D8): for the quantities whose change is a rare jump
  (recharge wait, gun cooldown, shield), advance by the decoded distribution's median (equal
  to the most likely outcome when a jump has p < 0.5); the others by the mean. Diagnostics
  the same.
- **Spread: a full covariance** of (x, y, vx, vy, heading, nose rate) in the belief's
  recurrent state. Each tick P' = F P F^T + Q with F the kinematics' Jacobian (position +=
  velocity dt, heading += nose rate dt) and Q the decoded changes' variances, velocity frame
  rotated to world. Other quantities: their variance plus the change's, as now.
  Risk to watch: under persistence the head's position-change spread on hidden rows already
  includes the belief's velocity error, so F may count it twice (sigma too large). Log
  coverage (truth within 1 and 2 sigma) by hidden age; say what you see on a short CPU run.
- Visible rows: the truth, zero spread.

## Ellipse spread inputs (`spread_inputs=ellipse`)

For position (from P's x-y block) and velocity (vx-vy block): log sigma_major, log
sigma_minor, and the doubled angle (cos 2 theta, sin 2 theta) times the anisotropy
(sigma_major - sigma_minor) / (sigma_major + sigma_minor), world frame. Coded in the input
code's style (say how), replacing the scalar position sigma and the velocity xx/xy/yy.

## Loss weight

Measure the next-state term's trunk-gradient share at initialisation and on a short CPU run;
set `learner.next_state` so the share matches the absolute head's in C-fixed (say the number
and how you got it), as round 1 did.

## Diagnostics (logged)

Per quantity: decoded one-tick error for visible ships against the bench's persistence,
rule and extrapolation medians; cross-entropy; padding and clip shares; the belief's error
and its 1 and 2 sigma coverage by hidden age; the ellipse inputs' anisotropy; the slow-frame
share.

## Tests (CPU)

- The codes: round trip below 1e-3 bins everywhere including the padding, the exact zero,
  folding only beyond the padded range; position along the track resolves to your stated
  width.
- The frame rotation and its inverse; the velocity-frame fallback.
- The advance: a known change moves the mean exactly; the median rule on a bimodal change;
  P' = F P F^T + Q on a constant-velocity case (variance grows as the analytic n^3 term).
- The ellipse: a round covariance gives zero angle features; a rotated one the right angle.
- "absolute" unchanged bit for bit (C-fixed's digests, `tests/sections/golden/`); arm 1's
  switches unchanged; `unroll` equals a loop of `step` with the arm's switches; BC and PPO
  learner steps; a policy-file round trip.
- The quick tier (`pytest -m "not slow and not gpu"`), the slow dashboards test, ruff, docs
  (`yemong.md`, `learner.md`, the state-code section).

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-arm2`, branch `feat/arm2-persistence` (from v2).
  Scratch `/tmp/bnb-scratch-arm2`.
- **CPU only:** `CUDA_VISIBLE_DEVICES=""`, `nice -n 15`, at most 8 threads; a BC run owns the
  GPU. Never take the bench lock; never touch the queue file, units or run directories (read
  only). No training runs.
- **Memory:** anything long runs under `systemd-run --user --scope -p MemoryMax=8G
  -p MemorySwapMax=0`, streaming to disk.
- **Git:** never use `git stash` (the repository has old shared stashes).
- **Python:** `uv sync --all-extras`, then `uv run --no-sync ...`.
- **Progress notes:** `/tmp/bnb-scratch-arm2/progress.md`, `## HH:MM` from `date +%H:%M`, 1-3
  bullets, ETAs as clock times. Start with a plan note.
- If something here is wrong, missing or ambiguous, say so in a note and in the final
  message; make a sensible choice and keep going.

## Done

- Commit and push to `feat/arm2-persistence` in logical commits. Author Julian Kizanis
  <julianzia@gmail.com>; end each message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Final message: the codes (bins, knees, ranges per component), the position-along choice and
  its memory and time cost, the loss coefficient and why, the coverage seen on the short run,
  the tests, the exact `--set` line, anything open, the time.
