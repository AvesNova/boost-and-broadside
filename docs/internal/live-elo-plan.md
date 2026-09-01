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

Each phase is independently landable and independently valuable, and the
sequence is designed to be abandonable after Phase 2 if the measurements say the
rest is not worth it.

**Phase 0 — drift detector and diagnostics.** Log the four alarms above, plus
the running `Var(r_live − r_scripted)`, against the *existing* K-factor
estimator. No behaviour change. Cheap, and it gives a baseline to compare the new
estimator against.

*Landed.* `train/rl/elo_diagnostics.py`, logged under `elo_diag/*` and
documented in `training.md`. Replaying 727's own `elo_history.jsonl` through it
gives the baseline the rest of this work is measured against:

| update | Mstep | live_elo | gauge-implied | drift | SE(live−scripted) | movement z |
|---:|---:|---:|---:|---:|---:|---:|
| 40 | 39.8 | 961.9 | 973.1 | −11.2 | 6.0 | 0.82 |
| 127 | 126.4 | 1393.4 | 1375.8 | +17.6 | 9.2 | 0.23 |
| 128 | 127.4 | 1361.5 | 1380.8 | −19.2 | 8.5 | 2.86 |
| **129** | **129.3** | **1451.0** | **1387.2** | **+63.7** | **10.4** | **7.40** |
| 130 | 130.3 | 1476.8 | 1391.5 | +85.3 | 11.2 | 2.25 |
| 200 | 200.0 | 1620.3 | 1524.9 | +95.5 | 16.8 | 0.92 |
| 320 | 319.4 | 1696.4 | 1570.2 | +126.2 | 20.1 | 1.18 |
| 400 | 399.0 | 1729.1 | 1599.6 | +129.6 | 22.1 | 0.82 |

Three things this settles.

**The detector works, and the seam is the event.** Update 129 is the resume.
Drift opens from −19 to +64 in one update at `movement_z` 7.4, and never closes.
The alarm fires on the day, which is what Phase 0 was for.

**The problem is worse than the 135–178M window suggested, and it compounds.**
Drift keeps growing after the seam — +85 at 130M, +96 at 200M, +130 at 400M
against a fitted standard error of 15. That is 8σ, and it is not a step that
settles; the rate of growth is roughly constant. The seam started it but is not
all of it, which supports the information-starvation account over a one-off
resume bug.

**The ceiling is now a number rather than an argument.** `SE(live − scripted)`
grows 6.0 → 22.1 over 400M steps, monotonically, on a pool whose Fiedler value
sits around 1e−8. At 400M the floor-anchored offset is worth ±22 Elo before any
estimator error at all, which is the size of the effects we routinely compare
between runs. Phase 4 is the phase that attacks this term.

**Phase 1 — persistent match matrix.** Accumulate wins/ties among stationary
players across the whole run. Persist it as a sidecar next to `roster.json` in
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

**Phase 2 — the two-stage estimator.** Replace the K-factor filter. Stage 1
refits the ladder every update from the accumulated matrix with `prior_games`,
a fixed scripted anchor, half-win draws, warm-started. Stage 2 rates the live
policy with `fit_single_rating` over a sliding window, with W chosen by the
offline replay.

Two rules for the side-by-side run. Log the old `live_elo` under a different key
— and **the old estimator keeps gating promotion and `best_training` selection
for the whole comparison run.** `live_elo` gates ladder advancement, so an
estimator that gates also changes promotion timing, which changes the pool, which
changes the ratings. If both estimators gate in different runs there is no
controlled comparison left. The new estimator observes only, until it has been
accepted.

**Checkpoint here.** Phases 3 and 4 are refinements of allocation; Phase 2 alone
plausibly captures most of the benefit, because the diagnosed fault is *how the
chain is estimated*, not which games were played. Measure against the 719 replay
and the seam test before committing further, and be willing to stop with Phase 2
shipped. If we do continue, note that the phases are numbered in increasing order
of implementation cost, not of expected value: **Phase 4 addresses the root cause
more directly than Phase 3 does**, since a well-estimated ladder is what the
floor-to-top offset actually rides on. Reordering them is reasonable if the
Phase 2 results point that way.

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
