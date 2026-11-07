# Brief: next-state delta heads (the round-1 challenger: C-fixed + delta heads)

**Decided by the author (2026-11-07):** the next-state head predicts each quantity's
**change from its physics baseline**, coded as **symlog HL-Gauss** (sigma 0.75 bins) with
an **exact-zero centre**, **2 padding bins per side** (decisions.md, "Code padding rule":
no folding at the edges), a **per-quantity knee at the median change** and **range at the
hard physical bound**, **two bin-count classes** (position and velocity finer, the rest
coarser), and position and velocity on **two axes in the nose (body) frame**. Everything
else stays as C-fixed (`--set yemong.order=temporal_first`, `spread_fit` "coarsest").

Read first: `docs/rewrite/sections/state-deltas.md` (the measured changes, the hard bounds
in 1.2, the recommendation in 4) and `docs/rewrite/sections/belief-baselines.md`; the
current head and targets: `bnb.learn.labels.next_state_targets`, `bnb.models.yemong`
(`state_code.py`, `codes.py`, `heads.py`, `belief.py`, `decode.py`), the next-state term
in `bnb.learn.ppo`.

## The targets, in the state code's order

| Quantity | Change from (baseline computable from the row's state at t) | Frame | Class |
|---|---|---|---|
| position | constant-nose-rate arc dead reckoning (report 1.1/2.4 "arc") | along / across heading_t | A |
| velocity | v_t rotated by omega_t dt | along / across heading_t | A |
| heading | heading_t + omega_t dt, wrapped | - | B |
| nose rate, angle of attack, energy (fraction of max) | persistence | - | B |
| log refractive index | persistence (no gradient term for now) | - | B |
| shield (fraction of max) | after the deterministic recharge | - | B |
| shield recharge wait | event: reset to the delay, or not (2 bins) | - | event |
| gun cooldown | event: fired, or not (2 bins) | - | event |

- Knees: the report's medians (table in 4). Ranges: the report's hard bounds, as the edge
  of the real bins, then 2 padding bins per side beyond them.
- Bins: class A **51** real bins per component, class B **31**, odd, exact-zero centre
  (the centre bin's value is exactly 0). Plus the padding.
- The code: map the change to its symlog bin coordinate, then HL-Gauss (the Gaussian CDF
  integrated over each bin) with sigma 0.75 bins; the padding bins hold the spill; only a
  value beyond the padded range folds into the last bin (count it:
  `next_state/support_clip/<quantity>`). Reuse `codes.py` where it fits; no folding at the
  real edge.
- The head outputs logits over this code (about 2 x 2 x 55 + 6 x 35 + 4 units); no
  persistence log-baseline term (the baseline is in the target's frame); initialise the
  output bias to the measured marginal (the zero-centre share, report 2.1), so the first
  loss is near the marginal entropy.
- The head still trains on every row it trains on today (allies, visible and hidden
  enemies), same mask.

## The belief advance (C's forecast feedback)

The forecast becomes: mean_{t+1} = baseline(belief_t) + decoded change (frame rotated
back); spread_{t+1}^2 = the belief's spread_t^2 carried through the baseline plus the
decoded change's variance (per quantity, in the same frame). Events: the expected value.
Then encoded into today's absolute input codes as today (inputs unchanged).

## Switch

`YemongConfig.next_state`: `"absolute"` (today, the default, so every existing build
record is unchanged bit for bit) or `"delta"`. The challenger runs with
`--set yemong.next_state=delta`. Recorded in the build and config.json.

## Diagnostics (logged)

Per quantity: decoded one-tick error for visible ships against persistence and the arc /
rotation baselines (the report's 2.4 numbers), cross-entropy by situation (in a field band,
hit, boost vs coast), support use (mass in padding, clipped), and the belief's error and
within-one-sigma by hidden age (existing series must keep working).

## Loss weight

Nats per token change with the code. Measure the next-state term's trunk-gradient share at
initialisation and on a short CPU run for both settings; set the coefficient so the delta
head's share matches the absolute head's in C-fixed (say the number and how you got it).

## Tests (CPU)

- The code: round trip (encode then decode) mean bias below 1e-3 bins everywhere,
  including at the real edges (the padding); the exact zero; the symlog spacing; the
  folding only beyond the padded range.
- The baselines equal the sim's next state on recorded steps where the dynamics are
  linear (no controls, no field), within float32; the frame rotation and its inverse.
- The belief advance: a known change moves the mean exactly; the spread grows by the
  change's variance.
- `"absolute"` unchanged bit for bit (a recorded step/unroll); `unroll` equals a loop of
  `step` with `"delta"`; the term in the learner (BC and PPO); a policy-file round trip.
- The quick tier (`pytest -m "not slow and not gpu"`), ruff, docs (`yemong.md`,
  `learner.md`, the state-code section).

## Where you work

- **Worktree:** `/home/vizia/avesnova/bnb-delta-heads`, branch `feat/delta-heads` (from
  v2). Scratch `/tmp/bnb-scratch-delta-heads`.
- **CPU only:** `CUDA_VISIBLE_DEVICES=""`, `nice -n 15`, at most 8 threads; a BC run owns
  the GPU. Never take the bench lock. No training runs.
- **Memory:** anything long runs under `systemd-run --user --scope -p MemoryMax=8G
  -p MemorySwapMax=0`, streaming to disk.
- **Git:** never use `git stash` (the repository has old shared stashes).
- **Python:** `uv sync --all-extras`, then `uv run --no-sync ...`.
- **Progress notes:** `/tmp/bnb-scratch-delta-heads/progress.md`, `## HH:MM` from
  `date +%H:%M`, 1-3 bullets, ETAs as clock times. Start with a plan note.
- If something here is wrong, missing or ambiguous, say so in a note and in the final
  message; make a sensible choice and keep going.

## Done

- Commit and push to `feat/delta-heads` in logical commits. Author Julian Kizanis
  <julianzia@gmail.com>; end each message with
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Final message: the codes (bins, knees, ranges per quantity), the loss coefficient and
  why, the tests, the exact `--set` line for the challenger, anything open, the time.
