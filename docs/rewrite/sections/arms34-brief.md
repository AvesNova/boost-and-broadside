# Brief: round-2 arms 3 and 4 (the error head; extrapolation baselines)

**Decided by the author (2026-11-08, `round2-plan.md` sections 4-5, O3-O8 settled in
`round2-log.md`):** arm 3 is arm 2 with the next-state work **split**: the physics head
trains on allies and visible enemies only, and a separate **error head** serves hidden
enemies, predicting how wrong the forecast is; the belief's sigma for hidden ships comes from
it. Arm 4 is arm 3 with **extrapolation** baselines for the physics head. Build both behind
switches; each must run alone with its `--set` line.

Read first: `round2-plan.md`, `round2-log.md` (the O3 and O8 calls, the heading fix),
`arm2-persistence-brief.md` and the arm-2 code it produced (`bnb.models.yemong.delta`,
`belief.py`, `heads.py`, `inputs.py`, `model.py`; `bnb.learn.labels`, the next-state term in
`bnb.learn.ppo`, `head_statistics.py`), `baselines-bench.md` (the extrapolation definitions
and `bench/baselines_core.py`).

## Arm 3: `yemong.delta_split=true` (with `delta_baseline=persistence`)

- **Physics head:** arm 2's head and targets, its loss masked to allies and visible enemies.
  It still runs on hidden rows (the forecast uses it); no gradient from them. Log its error on
  hidden rows by hidden age (against the truth) as a diagnostic.
- **Error head:** a second head on the same tokens, hidden enemy rows only. Target
  e = truth(t+1) - forecast(t+1), where forecast(t+1) = belief(t) + the physics head's decoded
  change, with **the gradient stopped through the physics term** (O7). Every quantity in the
  state code (O4; the countdowns and shield as continuous changes, as in arm 2).
- **2D quantities on three projection axes** (O8): position and velocity errors projected on
  world directions at 0, 60 and 120 degrees, one code each. The mean error by least squares
  from the three projected means; the 2 x 2 covariance from the three projected variances
  (u_i^T S u_i = var_i solves for S's three entries), clamped to positive semi-definite.
  Heading wrapped; the rest scalar.
- **Codes:** symlog HL-Gauss as arm 2, exact-zero centre, 2 padding bins, **uncapped target**
  (O6) over the full range a hidden ship's error can reach: position +-26 244 m (the code's
  period, the largest meaningful offset), velocity +-2 x the top speed, heading +-pi, the rest
  their code ranges. Knees: measure the error's median per quantity on a short CPU run of
  this arm (the belief starts as persistence forecasts; say the numbers); bins about 1 + 4
  ln(1 + range / knee), odd.
- **The advance for hidden rows:** mean(t+1) = forecast(t+1) + step, step = r tanh(|E e| /
  r) E e / |E e| per quantity (per 2D vector for position and velocity), with **r = the
  quantity's one-tick physical bound** (the bench's persistence maxima: the belief may move
  toward the truth no faster than the ship could move in a tick; say the values). Rare-jump
  quantities (wait, cooldown, shield): the median rule of arm 2 on e's distribution.
- **Sigma for hidden rows from the error head** (O5): Var e + (E e - step)^2 per quantity
  (the 2 x 2 for position and velocity from the projections, plus the outer product of the
  unapplied part). This replaces arm 2's F P F^T + Q propagation for hidden rows. The ellipse
  inputs and the per-axis blur read this covariance.
- **Loss weights:** the physics head keeps 0.0033; the error head gets its own coefficient
  (`learner.correction` or similar): start equal per row, measure both heads' trunk-gradient
  shares on a short CPU run, and say them; change the error head's only if its share is more
  than 1.5x off the physics head's per row, and say why.

## Arm 4: `yemong.delta_baseline=extrapolation` (with `delta_split=true`)

The physics head's baselines per quantity, as settled (O3, `round2-log.md` 02:00): position
the arc form with the tangential acceleration; velocity v + a dt with the tangential and
normal parts applied in the velocity frame; heading + omega dt + omega_dot dt^2/2; **nose rate
persistence**; angle of attack and energy last tick's change; shield and countdowns the rule
(last tick's rate only without an event, D2); ln n the gradient baseline ln n + grad . v dt.
The rates come from the acceleration inputs (visible rows: the observation; hidden rows: the
belief's own change, as the inputs do). The targets' knees and ranges from the bench's E rows
(re-measure on this tree as arm 2 did). The forecast for hidden rows uses the same baselines
on the belief; the error head is unchanged.

## Diagnostics (logged)

Arm 2's, plus: the error head's predicted vs realised error by hidden age (mean and sigma),
the step's size against r (the share capped), the physics head's error on hidden rows, both
heads' trunk shares.

## Tests (CPU)

- The projections: a known 2D error and covariance round-trip through the three axes; the
  positive semi-definite clamp.
- The advance: the capped step (below r: the full E e; far above: r); sigma's unapplied term.
- The physics head's mask; the stop-gradient (the error head's loss gives the physics head
  no gradient).
- Arm 4's baselines equal `bench/baselines_core.py`'s on recorded segments (float32).
- Arm 2's switches unchanged bit for bit (record digests before you change anything, as the
  acceleration agent did for C-fixed); C-fixed's and arm 1's goldens; `unroll` equals a loop
  of `step` for arms 3 and 4; BC and PPO steps; a policy-file round trip.
- The quick tier, the slow dashboards test, ruff, docs.

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-arms34`, branch `feat/arms34-error-head` (from v2).
  Scratch `/tmp/bnb-scratch-arms34`.
- **CPU only:** `CUDA_VISIBLE_DEVICES=""`, `nice -n 15`, at most 8 threads; a BC run owns the
  GPU. Never take the bench lock; never touch the queue file, units or run directories (read
  only). No training runs beyond short capped CPU measurement runs.
- **Memory:** anything long runs under `systemd-run --user --scope -p MemoryMax=8G
  -p MemorySwapMax=0`, streaming to disk.
- **Git:** never use `git stash` (the repository has old shared stashes).
- **Python:** `uv sync --all-extras`, then `uv run --no-sync ...`.
- **Progress notes:** `/tmp/bnb-scratch-arms34/progress.md`, `## HH:MM` from `date +%H:%M`,
  1-3 bullets, ETAs as clock times. Start with a plan note.
- If something here is wrong, missing or ambiguous, say so in a note and in the final
  message; make a sensible choice and keep going. Build arm 3 first and commit it working
  before arm 4.

## Done

- Commit and push to `feat/arms34-error-head` in logical commits. Author Julian Kizanis
  <julianzia@gmail.com>; end each message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Final message: the error head's codes and r per quantity, the knees you measured, the
  coefficients and shares, coverage on a short run (arm 3, and arm 4 if time), the tests, the
  exact `--set` lines for arms 3 and 4, anything open, the time.
