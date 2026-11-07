# Brief: a robust spread for the error head (fix before arms 3 and 4 rerun)

**Found 2026-11-08 07:40 (lead), on arm 3's GPU run at update 3:** the error head's
predicted position variance is 140x too wide at short hidden ages (47 683 m^2, sigma 218 m,
at 0-0.1 s, against a realised mean square error of 2.6 m^2; 100% within 1 sigma at every
age up to 2 s). Cause: `correction.decode` takes the variance as the second moment of the
categorical over symlog bins; the outer bins are kilometres wide, so a sliver of mass there
(1e-3 in a 10 km bin is 1e5 m^2) dominates it. The same pathology the mean had (fixed by the
median). Consequences: the hidden ships' inputs are blurred to nothing (the per-axis blur and
the ellipse read this sigma), and the confidence gain m^2 / (m^2 + sigma^2) is about 0, so
the correction never acts: arm 3 as built cannot test the error head. Arm 3's GPU run was
stopped at update ~4 for this.

## The fix

In the error head only (`bnb.models.yemong.correction`; arms 3 and 4), replace the second
moment with a **quantile spread**, per code (each projection axis and each scalar):
- In the code's coordinate (bin index), the 16th, 50th and 84th percentiles c16, c50, c84 by
  linear interpolation of the cumulative mass across each bin.
- sigma_c = (c84 - c16) / 2, less the code's own smoothing: sqrt(max(sigma_c^2 - 0.75^2, 0)).
- Physical: sigma = (v(c50 + sigma_c) - v(c50 - sigma_c)) / 2, v the code's value map
  (symlog, with its padding and any asymmetric sides).
- Variance = sigma^2. Position and velocity: the three projected variances to the 2 x 2 by
  the existing solve and the PSD clamp.
- Everywhere the error head's variance is used: the confidence gain; the spread carried after
  the step, now sigma^2 about the median plus (median - step)^2 (the median is the location
  the step moves toward); the inputs (blur, ellipse); the diagnostics
  (`correction/predicted_variance`, within 1 and 2 sigma).
- If a rare-jump quantity (wait, shield, cooldown) needs a different rule to stay sane, say
  so and choose.

Leave the physics head's variance as it is (its codes are bounded tightly; arm 2's coverage
was too narrow, not too wide), unless you find the same problem there; then say so.

## Tests (CPU)

- An encoded value decodes to sigma about 0; an encoded Gaussian of known sigma (several,
  across the knee) to that sigma within 5%; a sharp code plus 1e-3 mass in a far bin keeps
  its sigma (the bug's case); the 2 x 2 from three projections of a known covariance.
- C-fixed's, arm 1's and arm 2's goldens unchanged (digests in `tests/sections/golden/`);
  arms 3 and 4 `unroll` equals a loop of `step`; BC and PPO steps; the quick tier
  (`pytest -m "not slow and not gpu"`), the slow dashboards test, ruff, docs (`yemong.md`:
  the error head's spread and why).
- A short CPU run of arm 3 (as the arms-3/4 agent ran: 64 envs x 32 ticks, 20 updates, seed 1)
  and one of arm 4: report by hidden age the predicted sigma against the realised RMS error,
  within 1 and 2 sigma, the belief's error against arm 2's, the step's share of r and the gain.

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-error-spread`, branch `fix/error-head-spread` (from
  v2). Scratch `/tmp/bnb-scratch-error-spread`. Earlier short-run records for comparison:
  `/tmp/bnb-scratch-arms34/arm3_short*.jsonl`, `arm4_short*.jsonl`.
- **CPU only:** `CUDA_VISIBLE_DEVICES=""`, `nice -n 15`, at most 8 threads; a BC run owns the
  GPU. Never take the bench lock; never touch the queue file, units or run directories (read
  only). Only short capped CPU runs.
- **Memory:** anything long under `systemd-run --user --scope -p MemoryMax=8G
  -p MemorySwapMax=0`.
- **Git:** never use `git stash`. **Python:** `uv run --no-sync ...`.
- **Progress notes:** `/tmp/bnb-scratch-error-spread/progress.md`, `## HH:MM`, 1-3 bullets,
  ETAs as clock times. Start with a plan note.

## Done

Commit and push to `fix/error-head-spread` (author Julian Kizanis <julianzia@gmail.com>,
`Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`). Final message: the rule as built,
the short-run numbers above, the tests, anything open, the time.
