# Rebuilding the live Elo ladder

Working notes and implementation plan. Delete once the first run on the new
estimator has landed and the results are folded into `evaluation.md`.

## The problem, as measured

Run 727 was resumed at 128.3M. Across the seam its `live_elo` stepped +85 and
stayed there. Two independent instruments disagree with it:

| instrument | 719 | 727 | implied gap |
|---|---:|---:|---:|
| `live_elo`, 135–178M | 1427.6 ± 5.6 | 1520.5 ± 6.3 | **+93** |
| win rate vs scripted, same band | 0.915 ± 0.004 | 0.929 ± 0.004 | **+34** |
| calibrated tournament @127.4M | — | — | **+37** |

So `live_elo` over-reports by roughly 2.7x. Nothing published is affected —
`elo-calibrate` replays frozen checkpoints and never reads `live_elo` — but
`live_elo` gates ladder promotion and `best_training` selection, so the error
degrades the ladder as a measuring instrument over a long run.

### Root cause

`config/live_elo.py` defines the gauge only up to scripted: random 0, scripted
1000, semi-random at `p` → 1000·p. **Above scripted nothing is defined**, and
the K-factor filter extrapolates against the run's own checkpoints.

Fisher information per game goes as `p(1−p)`:

* live 1500 vs scripted 1000 → p ≈ 0.94, information ≈ 0.056
* live 1500 vs a rung 200 below → p ≈ 0.76, information ≈ 0.18

Slot 2 guarantees scripted a fixed 512 envs, so scripted is never starved of
*games* — it is starved of *information*, about 3:1. A K-factor filter settles
where the opposing pulls cancel, weighted by information, so the live rating
equilibrates ~70% toward what the self-generated rungs say. Solving
`0.3·34 + 0.7·X = 93` gives X ≈ 118: the rung chain believes 727 is ~118 ahead.

`MAX_CHECKPOINT_ANCHORS = 2` exists to damp this ("damping the random walk a
single-link ladder accumulates") — damping, not elimination.

### What is *not* the cause

* The seed of a new rung. `min_games_to_freeze = 1000` at `k_factor = 4` washes
  a 100-Elo seeding error out well before freezing.
* The floating-Elo jump and ladder win-rate drop at promotion. Those are
  ordinary promotion behaviour, verified against 727's own promotions at 41.8M
  and 76.6M where `live_elo` held flat (1019→997, 1205→1204).

## Why a floor is enough

Elo's unit is fixed by the logistic model — 400 points is 10:1 odds by
construction — so within a connected pool the MLE pins every rating
*difference* and only one additive constant is free. Two runs under identical
physics therefore need exactly **one** shared point, not a shared ladder.
Scripted-at-the-floor already supplies it.

(The "682 and 719 cannot share a scale" note is not a counterexample: there the
*physics* differed, so random-to-scripted genuinely spanned 1335 Elo in one game
and 862 in the other. Runs 719–727 share physics.)

So the problem is not that the anchor is too weak to be informative at the top.
It is that **the offset is carried from floor to top through a chain, and the
chain is estimated badly.** Fleet-size handicaps against scripted were
considered and rejected: adding weak opponents changes the task rather than its
difficulty, rewarding crowd control over skill.

## The two facts the design rests on

Both verified numerically against the stored artifact
`artifacts/elo-calibration/20260830T191704Z-9ee350c4/`:

1. **The Fisher information of a BT model is a weighted graph Laplacian**, with
   edge weight `w_ij = games_ij · c²·p_ij(1−p_ij)`, `c = ln10/400`.
2. **`Var(r_i − r_j)` is the effective resistance** between i and j in that
   graph. Checked against the fitter's own standard errors: 8.85 vs 8.85,
   12.96 vs 12.96, 18.25 vs 18.25.

### The allocation rule that follows

For a target functional such as `Var(r_live − r_scripted)`, Sherman-Morrison
gives the marginal value of one more game on edge (i,j):

```
gain_ij = c²·p_ij(1−p_ij) · b_ij²     where   b = φ_i − φ_j,   L φ = e_live − e_scripted
```

`b_ij` is the potential drop across edge (i,j) under unit current injected at
the live agent and drawn off at the anchor. The rule reads: **play the matches
that carry the most current between the live policy and the anchor.**

Structurally that is *local information × global position*. The current
allocator has only the first factor. On the real pool, targeting
`Var(727_final − scripted)` at a current SE of 11.04 Elo:

| top pair, c-optimal | score | local info |
|---|---:|---:|
| scripted vs r727_final | **4.8e−02** | 3.2e−06 (lowest in pool) |
| r725_ladder_133M vs r727_final | 2.8e−02 | 8.2e−06 |

| top pair, local `p(1−p)` only | score | local info |
|---|---:|---:|
| semi_scripted:0.5 vs r719_ladder_10M | **4.8e−10** | 8.3e−06 (max) |

Eight orders of magnitude apart. The local rule spends its budget on a perfectly
balanced matchup between two players whose relative rating nobody is asking
about, and ranks the single most valuable game — live against the anchor — last,
because it is saturated.

### The sub-1000 caveat handles itself

Summing every edge touching the weak end, against 4.8e−02 for the best single
edge: `random` 6.2e−04, `semi_scripted:0.5` 7.9e−03, `semi_scripted:0.8`
2.5e−02. `random` falls out by two orders of magnitude — no current flows
through a dead-end branch — while `semi_scripted:0.8` stays competitive because
it *is* on the path carrying the offset upward. Better than a hard threshold,
which would have cut a load-bearing link.

One consequence worth keeping: the direct live-vs-scripted edge wins now because
`b²` is large, but as the policy strengthens `p→1` and `info→0` faster than `b²`
grows, so the rule shifts budget onto the chain by itself. "When does the anchor
saturate" gets answered quantitatively, at the moment it happens.

## Design

### Estimator: two stages, every update

Cost measured with the repo's own `fit_bradley_terry`: 28 players 11.7 ms,
60 players 27 ms, 120 players 122 ms, single-rating solve 1.2 ms — against a
~350,000 ms update. There is no reason to batch it, and running every update
removes the periodic-bump problem by construction.

* **Stage 1 — the ladder.** Frozen rungs and stationary references only.
  Accumulate counts *forever*; these players do not move. Refit every update,
  warm-started from the previous solution. Converges like 1/√N and stabilises.
* **Stage 2 — the live policy.** `fit_single_rating` over a sliding window of
  recent games against the now-known ladder. Its noise is honest sampling noise
  from the window, with no filter lag and no K-factor fixed point.

Both functions already exist in `train/rl/bradley_terry.py`.

### Allocator

* Live slots: c-optimal, target `(live, scripted)`.
* Background rung-vs-rung slots: D-optimal (`gain_ij = info_ij · R_ij`), which
  prioritises poorly-measured pairs. Rung ratings are a compounding investment
  and the whole ladder must be sound for cross-run comparison.
* Allocate greedily with the Sherman-Morrison update
  `R_st ← R_st − δ·b_ij²/(1 + δ·R_ij)` so the batch spreads instead of piling
  onto one edge.

Asymmetry to respect: **rung-rung games accumulate forever; live games are
consumed every update.** Different objectives, different budgets.

### Guards — these are not optional

Earlier attempts at this degenerated with meaningless swings of hundreds of Elo.
Ranked by likelihood, with the fix. `elo-calibrate` fits the same model on the
same kind of data across a 1476-Elo spread without degenerating, so the online
version should port its guards rather than be written fresh.

1. **Complete separation → unbounded MLE.** Any player winning 100% of its games
   sends the likelihood to r → +∞; the iteration stops wherever the cap lands,
   so refits on near-identical counts land hundreds of points apart. The pool
   guarantees this (strong rungs beat `random` every game). Fix: `prior_games`,
   already a parameter, already 1.0 in `ELO_CALIBRATE`.
2. **Pool-dependent anchoring.** Centring on the pool mean, or on a per-fit
   reference, makes every rating jump when the pool changes. Always shift so
   scripted reads its fixed gauge value.
3. **Connectivity loss.** Information-weighted allocation can starve the link
   between the rung block and the stationary references; a weakly-connected
   component swings on small count changes. Fix: a floor of games on every
   rung-to-stationary edge, plus a Fiedler-value monitor.
4. **Non-stationary players in the accumulator.** The still-floating rung, the
   averaged policy, or the live policy must never enter the accumulate-forever
   matrix. `_NON_STATIONARY` in the calibrator is the precedent.
5. **Dropping draws.** Against random, a whole-run record of 2794W/10L/1120T
   yields Fisher information 10 under decisive-only and 487 under half-win.
   Keep half-win.

Regression alarms to log every refit:

* movement per unit evidence — a rung with 50,000 games moving 50 Elo on 100 new
  ones is a bug;
* Fiedler value of the weighted Laplacian;
* max |rating|, which trips immediately on separation;
* `live_elo` minus the scripted-implied Elo — the drift detector that would have
  caught 727 on the day.

## Phases

Each phase is independently landable and independently valuable.

**Phase 0 — drift detector and diagnostics.** Log the four alarms above against
the *existing* K-factor estimator. No behaviour change. Cheap, and it gives a
baseline to compare the new estimator against.

**Phase 1 — persistent match matrix.** Accumulate wins/ties among stationary
players across the whole run, checkpointed and restored. Nothing reads it yet.

**Phase 2 — the two-stage estimator.** Replace the K-factor filter. Stage 1
refits the ladder every update from the accumulated matrix with `prior_games`,
a fixed scripted anchor, half-win draws, warm-started. Stage 2 rates the live
policy with `fit_single_rating` over a sliding window. Keep the old `live_elo`
logged alongside under a different key for one run.

**Phase 3 — resistance-based allocation.** Replace the `p(1−p)` multinomial with
the c-optimal rule for live slots. One linear solve plus an O(n²) scoring pass
per batch. Keep the connectivity floor.

**Phase 4 — background rung tournament.** Reallocate eval slots so some fraction
plays rung-vs-rung under the D-optimal rule. This is what makes the ladder
itself well-estimated rather than only the live policy.

**Phase 5 (optional, later).** Exploiters as extra pool members for stylistic
diversity, gated on the Hodge cyclic share staying low. Currently 2.5%, so BT is
the right model and this is not needed yet.

## Validation

**Offline, no GPU, quantitative.** 719's `elo_history.jsonl` records per-update
match counts by opponent label, and its post-hoc calibration
(`checkpoints/good-leaf-719/artifacts/elo-calibration/`) contains
`curve[].live_calibrated` — the best available ground truth for what the live
policy was worth at each update. Replay the new estimator over the recorded
counts and check it tracks `live_calibrated` better than the recorded
`live_elo` does. Report RMS error against the calibrated curve for both.

Caveat: those counts came from the *old* allocator, so this validates the
estimator only, not the allocator.

**Allocator, in simulation.** Synthetic pool with known ratings; simulate
batches under (a) `p(1−p)` multinomial, (b) c-optimal, (c) uniform; compare the
SE of the target functional against games spent. Deterministic and cheap, so it
belongs in the test suite.

**Live.** One run with both estimators logged side by side, then a post-hoc
calibration; the new curve should sit closer to the calibrated one.

## Non-goals

* Replacing Bradley-Terry. The pool is 2.5% cyclic; BT is the right model. mElo
  is held in reserve behind a pre-committed threshold (adopt only if it beats BT
  on held-out log-likelihood once the cyclic share passes ~10%).
* Nash averaging or α-Rank as the rating. Both are pool-relative, so they cannot
  be compared across runs. Useful as within-run league diversity audits only.
* Any use of another run's weights, or of scripted above the floor.
* Changing what `elo-calibrate` publishes. `live_elo` remains a training
  instrument and must never be published as a rating.
