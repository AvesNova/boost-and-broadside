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

### Root cause — the leading hypothesis, not a measurement

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

**Read that as a hypothesis.** The 0.3/0.7 mixing weights are derived from the
same information ratio they are being used to explain, so the argument is
self-consistent rather than independently confirmed. It is consistent with every
number in the table, and no competing explanation survives the checks below —
but nothing yet rules out a second mechanism contributing. Phase 0 exists partly
to test it: if the drift detector shows `live_elo` minus scripted-implied Elo
growing smoothly with policy strength, the story holds; if the drift is
concentrated at discrete events, something else is also at work and Phase 2
alone will not fix it.

### What is *not* the cause

* The seed of a new rung. `min_games_to_freeze = 1000` at `k_factor = 4` washes
  a 100-Elo seeding error out well before freezing.
* The floating-Elo jump and ladder win-rate drop at promotion. Those are
  ordinary promotion behaviour, verified against 727's own promotions at 41.8M
  and 76.6M where `live_elo` held flat (1019→997, 1205→1204).

## Why a floor is enough

> **Superseded by the Phase 0 control.** Read this section with the measurement
> in *Phase 2 as designed is refuted* below. A shared floor is enough to
> **define** the scale, and that part stands. It is not enough to **estimate**
> against: a floor-only fit is off by ~50 Elo in RMS and in bias late in a run,
> because the direct edge saturates and a single saturated edge is not a
> connected graph. The argument below is right about identification and wrong
> about what follows from it for an estimator.

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

### The ceiling this design has, and why calibration stays the publication path

A floor is enough to *define* the scale, but it gets progressively more
expensive to *measure* against. `Var(r_live − r_scripted)` is the effective
resistance from the live policy to the anchor, and as the policy strengthens the
direct edge saturates — `p → 1`, `info → 0` — so the current is forced through
the rung chain, adding resistances in series. The variance of the floor-anchored
offset therefore grows over a run no matter how well the games are allocated.
Better allocation changes the constant and the rate; it cannot make the quantity
stationary.

That is the honest limit of a live rating, and it is precisely why frozen
checkpoint calibration remains the thing we publish. `elo-calibrate` gets to
replay any pair it likes, including live-era checkpoints against each other, so
it can short the chain that training had to traverse one link at a time. The
non-goal below — never publish `live_elo` — is a consequence of this paragraph,
not an independent policy choice.

Phase 0 should log the running `Var(r_live − r_scripted)` so we find out how fast
this actually degrades in practice. If the SE at 400M is small enough, the
distinction is academic; if it is 40 Elo, cross-run claims late in a run need the
calibrator and we should know that before making one.

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

The local rule spends its budget on a perfectly balanced matchup between two
players whose relative rating nobody is asking about, and ranks the single most
valuable game — live against the anchor — last, because it is saturated.

Do not read the raw ratio between those two numbers as the expected improvement.
It compares each rule's *best* pair, and `p(1−p)`'s best pair is bad by
construction, because that rule has no notion of which rating we are asking
about. The honest baselines are uniform allocation and `p(1−p)` restricted to
edges touching the live policy; both are far better than the unrestricted local
rule, and the realistic gain over them is a modest constant factor, not orders of
magnitude. The simulation in *Validation* measures against all three, and that
measurement — not this table — is what decides whether Phase 3 earns its
complexity.

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

> **Revised after Phase 0.** Stage 2 must fit against the **whole pool**, not
> the gauge references — floor-anchored fits lose to the current filter by a
> factor of four. And the sliding window is a cost at every length tested, so
> the window is one update. What remains worth building is the *ladder* half:
> refitting rung ratings from the accumulated matrix rather than filtering them.

Cost measured with the repo's own `fit_bradley_terry`: 28 players 11.7 ms,
60 players 27 ms, 120 players 122 ms, single-rating solve 1.2 ms — against a
~350,000 ms update. There is no reason to batch it, and running every update
removes the periodic-bump problem by construction.

* **Stage 1 — the ladder.** Frozen rungs and stationary references only.
  Accumulate counts *forever*; these players do not move. Refit every update,
  warm-started from the previous solution. Converges like 1/√N and stabilises.
* **Stage 2 — the live policy.** `fit_single_rating` over a sliding window of
  recent games against the now-known ladder.

Both functions already exist in `train/rl/bradley_terry.py`. Draws are folded in
at half a win before the fit, matching every existing caller — `fit_bradley_terry`
takes a decisive-win matrix and excludes draws by contract.

**What stage 2 does and does not buy.** It is not lag-free, and the window length
is a tuned constant with the same bias/variance tradeoff the K-factor had: the
live policy is improving *within* the window, so a boxcar of length W estimates
what the policy was worth around W/2 updates ago. The two real gains are that the
estimate has no self-referential fixed point — it is a direct solve against
ratings that were not themselves derived from the live policy — and that its
error bar is honest sampling noise the fitter reports, rather than a filter
state whose spread nobody can quote. Pick W by the offline replay against 719's
`live_calibrated` curve rather than by feel, and log the fitted standard error
next to the rating so the lag/noise tradeoff stays visible.

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
4. **Non-stationary players in the accumulator.** The averaged policy and the
   live policy must never enter the accumulate-forever matrix.
   `_NON_STATIONARY` in the calibrator is the precedent.

   *Corrected while implementing Phase 1.* This guard originally named the
   still-floating rung too, which is wrong: admission is decided by **weights,
   not by ratings**. A floating checkpoint's weights are fixed at snapshot time
   and never move again — only its *rating* is unsettled, and the accumulator
   stores counts, which the fit re-estimates from scratch. Its rating being
   unsettled is the reason to accumulate its games, not a reason to withhold
   them. The live and averaged policies are excluded on the correct grounds:
   they change strength under the record, and a count matrix cannot say when a
   game was played, so pooling them fits the average of something that was never
   the same twice.
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

Each phase is independently landable and independently valuable, and the
sequence is designed to be abandonable after Phase 2 if the measurements say the
rest is not worth it.

**Phase 0 — drift detector and diagnostics.** Log the four alarms above, plus
the running `Var(r_live − r_scripted)`, against the *existing* K-factor
estimator. No behaviour change. Cheap, and it gives a baseline to compare the new
estimator against.

*Landed.* `train/rl/elo_diagnostics.py`, logged under `elo_diag/*` and
documented in `training.md`.

It was then replayed over **719 as a control** — the run whose post-hoc
calibration shows its filter tracking the truth to −4.1 Elo on average — and the
control overturned three things this document previously asserted. What follows
replaces them.

### The floor-anchored estimate is biased, not just noisy

719's filter is accurate, and `drift_vs_gauge` still reads **+53 on average**,
rising past +75, with a correlation to the filter's true error of only +0.18.
The mechanism is saturation: once the policy beats every defined reference
almost always, its record against them stops being able to say *how far* above
them it is, and the fitted rating settles below the truth.

So the drift is a **paired** instrument. Two runs under the same physics share
the bias almost exactly, and their difference at matched steps is meaningful
even though neither number is:

| band | 727 − 719 drift | n |
|---|---:|---:|
| 40–128M (before 727's resume) | **+4.6 ± 33** | 22 |
| 130–400M (after) | **+60.5 ± 30** | 54 |

That is the cleanest measurement of the original problem yet taken, and it
agrees with the two independent instruments (+34 win-rate-implied, +37
calibrated tournament at 127.4M, both measured *at* the seam before the shift
completed). The earlier "+130 at 400M" reading in this document was the shared
bias, not the effect.

### The seam is a regime change, not a detectable event

`movement_z` at the seam is 7.4, which sits inside 719's ordinary range — 719's
distribution is median 1.33, p95 4.34, max 14.8, with values of 6.8 and 9.7 at
unremarkable updates. And the seam's single-update change in drift is the **53rd
percentile** of 719's. No per-update alarm fires, and the earlier claim that this
"would have caught 727 on the day" is withdrawn.

What is real is the persistent shift in the table above: +4.6 → +60.5, sustained
over 270M steps. It takes tens of updates of pooled evidence to see, not one.
Any drift alarm has to be built on an accumulated comparison, not a per-update
threshold.

### The measurement ceiling is real and run-independent

`SE(live − scripted)` climbs 6.0 → 22.1 over 400M steps in both runs, on a pool
whose Fiedler value sits near 1e−8. This one survives unchanged: it is a
property of the information graph rather than of any estimator, and at 400M the
floor-anchored offset is worth ±22 Elo before estimator error.

### Phase 2 as designed is refuted

Scored against 719's `live_calibrated` curve over all 1004 updates, RMS error in
Elo:

| estimator | RMS | bias |
|---|---:|---:|
| `live_elo`, the current K-factor filter | 17.6 | −4.1 |
| **all opponents, window 1 update** | **15.4** | −4.9 |
| all opponents, window 8 | 36.7 | −10.3 |
| gauge references only, window 1 | 65.9 | −48.1 |
| gauge references only, window 8 | 69.5 | −57.2 |
| scripted only, window 4 | 63.0 | −52.0 |

Three conclusions, in order of how much they change the plan.

**The chain is not the problem — it is the solution.** This document argued the
offset should be carried from the floor and that the self-generated rungs were
the weak link. The opposite holds: floor-anchored estimation is off by ~50 Elo
in RMS *and* in bias, and the only estimator that tracks the calibrated truth is
the one that uses the whole pool including the run's own rungs. Bradley-Terry
pinning differences from one shared anchor is true of the MLE over a connected
graph; it is not true of a single saturated edge, which is what a floor-only fit
degenerates into late in a run. Stage 2 must not be built against the floor.

**The sliding window is a cost, not a benefit.** Every mode is monotonically
worse as W grows, and W=1 wins outright. The policy improves fast enough that
pooling even two updates costs more in lag than it buys in variance. This kills
the window-length sweep as a tuning exercise: the answer is one update.

**The headroom is 12%.** The best implementable alternative beats the current
filter 15.4 to 17.6. That is a real improvement and it comes with an honest
error bar the filter cannot produce, but it is not the several-fold gain this
plan was scoped around. The stop/go checkpoint below should be read as already
half-fired.

One thing the replay could not test: 719's history carries no accumulated
rung-vs-rung record, so "all opponents" had to use rung ratings that the
K-factor filter itself produced. Whether refitting those from the Phase 1 matrix
improves on 15.4 is the open question, and it is the question Phase 4 exists to
answer. That is now the phase carrying the plan's value.

**Phase 1 — persistent match matrix.** *Landed.* `train/rl/match_matrix.py`,
saved as `match_matrix.json` in the run directory.

The games were already being played and thrown away. The evaluator's slot 4
runs the floating checkpoint against a stationary anchor every update to settle
the floating rating, then discards the outcome; both players are weight-frozen,
so over a run those results build exactly the rung-to-reference and
rung-to-rung graph the ladder estimator needs. Nothing new is computed and no
eval budget moves — Phase 4 is where budget gets reallocated.

Kept out of `elo_history.jsonl` on purpose. That file holds the run's
*irreplaceable* measurements, the live and averaged policies' records, which
exist in one form for one update. Everything in the match matrix can be
replayed from disk later at any precision; it is persisted because the training
run needs it *now*, which is a different reason and belongs in a different file.

Accumulate wins/ties among weight-frozen players across the whole run. Persist it as a sidecar next to `roster.json` in
the checkpoint directory rather than inside the `.pt` payload — the roster
already works this way, and keeping the tensor payload untouched means the
compatibility question never arises for inference. A run that finds no matrix
file starts an empty one, which is also what a resumed pre-Phase-1 run does.
Nothing reads it yet.

*Compatibility, stated once:* old checkpoints must stay **loadable for post-hoc
inference** — `elo-calibrate`, the league, tournament replay. They need not stay
*resumable*. Any state added for this work therefore goes outside the `.pt`
payload, or is optional with an empty default; neither is allowed to become a
required key that `load_checkpoint_payload` would trip on.

**Phase 2 — the two-stage estimator.** *Rescoped by the Phase 0 control.*
Stage 1 refits the ladder every update from the accumulated matrix with
`prior_games`, a fixed scripted anchor, half-win draws, warm-started — this is
the half that survives, and it is the half Phase 1 now feeds. Stage 2 rates the
live policy with `fit_single_rating` against **the whole pool at those refitted
ratings, over a single update**: not against the gauge references, which lose to
the current filter four to one, and not over a sliding window, which was worse at
every length tested.

The measured headroom for stage 2 alone is 17.6 → 15.4 RMS against 719's
calibrated curve, using filter-produced rung ratings. Whether stage 1 improves
on that is untested and untestable offline, because no run has an accumulated
rung-vs-rung record yet. Build stage 1 first, log stage 2 beside the filter, and
let the first run on the new matrix answer it.

Two rules for the side-by-side run. Log the old `live_elo` under a different key
— and **the old estimator keeps gating promotion and `best_training` selection
for the whole comparison run.** `live_elo` gates ladder advancement, so an
estimator that gates also changes promotion timing, which changes the pool, which
changes the ratings. If both estimators gate in different runs there is no
controlled comparison left. The new estimator observes only, until it has been
accepted.

**Checkpoint here — and it has already half-fired.** The 719 replay puts stage
2's headroom at 12%, not the several-fold gain this plan was scoped around, and
it refuted the floor-anchored design the plan was built on. What the replay could
not test is whether refitting the rungs helps, because no run had the record to
refit from until Phase 1 landed.

So the ordering changes. **Phase 4 now carries the plan's value, not Phase 3.**
The estimator that tracks the calibrated truth is the one that uses the run's own
rungs, which means rung quality is the binding constraint, which is exactly what
a rung-vs-rung tournament buys. Phase 3 optimises how the live policy spends its
games and is a refinement of a term that is no longer the largest one.

Recommended sequence from here: land stage 1 and a whole-pool stage 2 logged
beside the filter, then Phase 4, then reassess Phase 3 on measurement rather than
on the argument in this document.

*Phases 2 and 4 landed together*, since stage 1 needs games that only the
Phase 4 allocator produces in useful proportions. `train/rl/live_rating.py` and
`train/rl/allocation.py`, logged under `two_stage/*`, gating nothing.

The allocator's simulated margin, standard error of a rung's floor offset:

| rule | 10 batches | 40 | 160 |
|---|---:|---:|---:|
| `p(1−p)` — the rule being replaced | 39.5 | 26.1 | 14.5 |
| uniform | 33.9 | 20.4 | 10.9 |
| **current-flow weighted** | **23.6** | **12.2** | **6.1** |

Roughly three times fewer games for equal precision against the honest baseline.
Worth recording that the old rule is *worse than uniform*: targeting information
without asking what it is information about is not merely suboptimal.

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

Sweep the stage-2 window length W here and pick it on this curve. Two caveats,
the second more serious than the first:

* those counts came from the *old* allocator, so this validates the estimator
  only, not the allocator;
* **this tests static estimation quality, and the failure we actually observed
  was dynamic.** 727's fault was an +85 *step at a resume seam* that then
  persisted. A replay over 719's smooth run can show a healthy RMS improvement
  while saying nothing about whether the new estimator steps across a seam. Do
  not treat a good replay number as the plan succeeding.

**The seam test.** The direct instrument for the observed fault: run the new
estimator across a stop/resume and check it does not step. 726 is parked at
103.5M with its ladder intact and is the cheap way to get one — resume it, seam
and all, with both estimators logged. If it no longer resumes cleanly, the
fallback costs nothing extra: deliberately stop and resume the Phase 2
comparison run at a chosen update and read the same seam off it. Either way this
is a required gate, not an optional extra, and it is the one that maps onto the
bug.

Note that 727's own seam is confounded with a coincident promotion and cannot
serve as the ground truth here — which is exactly why a clean seam has to be
manufactured.

**Allocator, in simulation.** Synthetic pool with known ratings; simulate
batches under (a) `p(1−p)` multinomial, (b) c-optimal, (c) uniform, and
(d) `p(1−p)` restricted to edges touching the live policy; compare the SE of the
target functional against games spent. (c) and (d) are the baselines that decide
whether Phase 3 is worth building — (a) is a floor, not a fair comparison.
Deterministic and cheap, so it belongs in the test suite.

**Live.** One run with both estimators logged side by side — old one gating, new
one observing — then a post-hoc calibration; the new curve should sit closer to
the calibrated one.

## Non-goals

* Replacing Bradley-Terry. The pool is 2.5% cyclic; BT is the right model. mElo
  is held in reserve behind a pre-committed threshold (adopt only if it beats BT
  on held-out log-likelihood once the cyclic share passes ~10%).
* Nash averaging or α-Rank as the rating. Both are pool-relative, so they cannot
  be compared across runs. Useful as within-run league diversity audits only.
* Any use of another run's weights, or of scripted above the floor.
* Changing what `elo-calibrate` publishes. `live_elo` remains a training
  instrument and must never be published as a rating — see *The ceiling this
  design has* for why that stays true however good the estimator gets.
* Making old checkpoints resumable. They must stay loadable for post-hoc
  inference; resume compatibility across this work is explicitly not maintained.
