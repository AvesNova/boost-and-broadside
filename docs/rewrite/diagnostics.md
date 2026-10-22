# Diagnostics audit

Every metric the trainer logs today (`train/rl/*.py`), grouped by prefix, with a first
verdict for the port. Status: **draft, to go through group by group with the author.**
Templated names (`{name}`, `{group}`, ...) expand per reward level, code group,
population or ladder entry, so the real series count is several hundred.

Verdicts: **keep**, **drop** (dead by a decision, constant, or duplicate), **rename**,
**replace** (the thing it measured changed), **check** (needs the author's call).

| Group | Series | What it is | First verdict |
|---|---|---|---|
| `overview/*` | 16 | Curated copies of other series for one dashboard | keep the idea, rebuild from surviving series: drop `score_vs_*`, `bc_coef`, `loss_proxy_bc`, `win_rate_vs_scripted`; add live rating and pool shares |
| `loss/*` | 16 | Unweighted loss per term | keep; drop `sigreg`; drop `next_state_cont` (equal to `next_state`) |
| `loss_proxy/*` | 10 | Weighted loss per term (coefficient x loss) | rename to `loss_weighted/*`; drop `sigreg` |
| `policy/*` | 8 | KL, clip fraction, ratio mean/max, entropy per factor (power, turn, shoot) | keep; add joint 42-way entropy; factor entropies become marginals of the joint |
| `critic/*`, `value/{level}`, `returns/*`, `return_normalizer/*` | ~20 | Explained variance per level, value loss, return/value means, advantage std, percentile scale | keep; add HL-Gauss support use per level (share of target mass in padding bins or clipped at the edge) |
| `outcome/*` | 5 + bins | Predicted vs realised win, Brier, calibration bins | keep |
| `next_state*`, `next_state_ce/{pop}/{group}`, `next_state_baseline_ce/...`, `next_state_population/...` | many | Next-state CE per population and code group, vs the "nothing changes" baseline | keep, consolidate under one prefix; per-population averaging is decided (3c) |
| `belief/{prefix}/{name}` | many | Physical error of the decoded belief (visible / hidden splits) | keep, check naming |
| `enemy_action/*` | 10 | Realised probability, entropy, Brier, accuracy, persistence baseline, uniform baseline | keep; drop the four `uniform_*` series (constants: ln 42 etc., document instead) |
| `global_density/*` + `loss/global_density_*` | 6 | Poisson deviance, ally and enemy | replace: shape KL, count deviance, deviance explained per channel, enemy split by hidden count (3e) |
| `shot/*` | 10 | Shot head realised vs predicted rates, trajectory CE per channel | keep (heads ported as is; item 15) |
| `episode/*`, `episode/event/{name}`, `episode/reward_{name}` | 12 + per level | Win/draw rates, lengths, lifespans, rewards, event counts | keep; `team0_win_rate` / `team1_win_rate` become per-view |
| `fog/*` | 10 | Visibility, occlusion, hidden age, reacquisitions | keep (game analytics) |
| `physics/*`, `frontline/*`, `map_scale/*` | 12 | Speed, power, damage and deaths by source, front advances, respawns, captures, map scale | keep (game analytics) |
| `reward/event_share/*`, `reward/zero_sum_residual/*` | per level | Share of reward by event; zero-sum residual | keep; `zero_sum_residual` depends on tabled item 13 |
| `seat/*` | 2 | `ego_pass` pending-action validity per seat | replace with per-view checks, or drop |
| `schedule/*` | 12 | Current schedule values | keep `learning_rate`, `entropy_coef`, `offensive_bias`, tier scales; drop `bc_cutoff_streak`, `bc_decay_factor`, `behavior_cloning_coef`, `scripted_win_rate`, `policy_gradient_coef`, `target_kl` (all gone or fixed) |
| `train/*` | 11 | Throughput, gradient norms, actor share, clip fire rate, epochs completed, non-finite fraction | keep; `nonfinite_grad_fraction` -> skipped-step count (4d); add time breakdown (rollout, learner, host transfer stall) |
| `counters/*` | 5 | Env steps, tokens, updates, hours | keep |
| `grad_norm/`, `grad_share/`, `grad_cos/`, `grad_coherent_*`, `grad_halves_cos/`, `grad_diag/*` | many | Per-term gradient decomposition at the diagnostic cadence | keep; consolidate under one `grad/` prefix |
| `elo_diag/*`, `eval/*`, `ladder/*`, `two_stage/*`, `live_elo/*`, `matches/*` | ~45 | The dedicated evaluator and its ladder | replace with the in-run joint fit: `rating/live`, `rating/avg`, `rating/stderr`, `pool/share/{kind}`, `matches/{entry}/score` from league games |
| `hist/logprob`, `hist/returns/{level}` | 3 + per level | Histograms every 10 updates | check: likely drop `hist/logprob`; returns histograms either kept at low cadence or replaced by the support-use scalars above |

**Candidates to add** (from decisions so far): skipped-step count and dump path (4d);
carry-staleness check (post-implementation TODO); per-view win rate and actor
share; pool composition and per-entry game counts (4b); HL-Gauss support use for
every code; density deviance explained (3e); timing breakdown including stream
stalls (4d); league drain progress.

## Generic checklist (BC and RL)

What any BC or RL project needs, independent of today's metrics. Each entry names the
question it answers. **Status:** have (passes), partial, missing, decided (already
settled in an earlier part). **Cost:** U = per update, on device, read back with the
update's one sync; C = diagnostics cadence (10 updates); T = a test, not a plot;
O = offline or low cadence, off the critical path. Status: **proposed, to go through
with the author.**

### A. Run health and throughput

| ID | Diagnostic | Question it answers | Status | Cost |
|---|---|---|---|---|
| A1 | Skipped (non-finite) steps: count, offending term, dump path | Is the run numerically sound, and where did it break? | partial (`train/nonfinite_grad_fraction`; 4d decided count + dump) | U |
| A2 | Env steps/s, tokens/s | Is the run as fast as it should be? | have | U |
| A3 | Time breakdown: rollout, learner, stream stall, logging | Where does wall clock go; is the stream keeping up? | missing (candidate) | U |
| A4 | Device memory: compiled peak vs budget | How close are we to OOM; did a change grow memory? | missing (`vram_probe` is a separate tool) | free, at compile |
| A5 | Recompile count after warm-up | Did a shape or Python value leak into a trace and silently recompile? | missing (JAX-specific) | U, host counter |
| A6 | Host syncs per update | Did someone add a sync on the hot path? | missing | T |
| A7 | Progress counters | Where are we in the run? | have | U |
| A8 | Same seed gives bitwise-same first updates | Is the run reproducible, so A/B results mean something? | missing | T |
| A9 | Checkpoint round trip: resume equals uninterrupted | Does resuming change the run (lost optimizer state, keys, pool)? | missing | T |

### B. Optimisation

| ID | Diagnostic | Question it answers | Status | Cost |
|---|---|---|---|---|
| B1 | Loss per term, raw and weighted | Is each objective going down, and which one dominates? | have | U |
| B2 | Global grad norm (pre-clip), clip fire rate | Are gradients exploding or always clipped? | have | U |
| B3 | Per-term gradient norms, shares, conflicts | Which term drives the trunk; do terms fight? | have (`grad_*`) | C |
| B4 | Parameter norm per module | Are weights growing without bound (a known cause of lost plasticity)? | missing | U |
| B5 | Update-to-parameter ratio per module, ‖Δθ‖/‖θ‖ | Is the effective step size sane per layer; is a module frozen or too hot? | missing | U |
| B6 | Dormant-unit share and activation RMS per block | Are units dying or saturating over a long run (plasticity loss)? | missing | C |
| B7 | Schedule values | What were LR, entropy coefficient etc. at this point? | have | U |

### C. Data and inputs

| ID | Diagnostic | Question it answers | Status | Cost |
|---|---|---|---|---|
| C1 | Observation feature ranges: mean, std, share at clip limits, per feature | Are inputs in range; did a physics or encoding change break normalisation? | missing | C |
| C2 | Reward per term: scale and share | Which reward term drives learning; is one term swamping the rest? | have (`reward/event_share`, `episode/reward_*`) | U |
| C3 | Episode end reasons: win, loss, draw, time limit | Are games decided or timing out; is bootstrapping at truncation exercised? | partial (no time-limit share) | U |
| C4 | Behaviour vs learner mismatch: KL and ratio between stored and recomputed log-probs, first minibatch of epoch 0, before any step | Does the learner see the same policy that acted (stale carry, dtype, feature drift)? Should be ~0. | missing (ratio is only averaged over all epochs); subsumes the carry-staleness TODO | U |
| C5 | Sample composition: learner tokens per view, mode, opponent kind | Is the batch made of what we think it is? | partial (pool shares decided, 4b) | U |
| C6 | Label validity: share of valid targets per auxiliary head | Are delayed labels (shots, 29 ticks) and rollout-edge masks leaving enough data? | partial (`seat/*` only) | U |

### D. Policy

| ID | Diagnostic | Question it answers | Status | Cost |
|---|---|---|---|---|
| D1 | Entropy per factor and joint | Is exploration collapsing or stuck high? | partial (joint missing) | U |
| D2 | Entropy quantiles, near-certain share | Is the policy decisive in some states and random in others? | missing (group 1 proposal) | U |
| D3 | Action usage per factor | Are all actions used (new stick level); is the style drifting? | missing (group 1 proposal) | U |
| D4 | Action switch rate (own action changes step to step) | Is the policy jittery? Bears on goal A (fun, human-like opponents). | missing | U |
| D5 | KL, clip fraction, ratio, epochs completed | Is each update the size we intend? | have | U |
| D6 | Advantage mean, std, share clipped by the normaliser | Is the actor's learning signal well scaled? | partial (std only) | U |

### E. Value

| ID | Diagnostic | Question it answers | Status | Cost |
|---|---|---|---|---|
| E1 | Explained variance per level | Does the critic predict returns better than a constant? | have | U |
| E2 | Value bias per level, mean(prediction − target) | Is the critic systematically over- or under-estimating? | partial (both means logged, not the difference) | U |
| E3 | Support use per code (mass in padding bins or clipped) | Do targets fit the categorical range? | missing (candidate) | U |
| E4 | Outcome calibration | Are win probabilities honest? | have | U |
| E5 | Return normaliser scale | Is the normaliser stable? | have | U |

### F. Auxiliary heads and representation

| ID | Diagnostic | Question it answers | Status | Cost |
|---|---|---|---|---|
| F1 | Each auxiliary head vs its trivial baseline | Does the head learn anything beyond persistence or a prior? | have for next state and enemy action; decided for density (3e); partial for shots | U |
| F2 | Effective rank of trunk features | Is the representation collapsing? With SIGReg dropped, nothing else watches this. | missing | C |
| F3 | Belief error, visible vs hidden | Does the model track what it cannot see? | have | U |

### G. Performance

| ID | Diagnostic | Question it answers | Status | Cost |
|---|---|---|---|---|
| G1 | Live rating with standard error | Is the agent getting stronger, and how sure are we? | decided (4b, joint fit) | U |
| G2 | Score per pool entry kind | Against whom is it improving or losing? | decided (4b) | U |
| G3 | Score vs past checkpoints over time | Is it forgetting how to beat old strategies (cycling)? | partial (match records exist; no plot) | C |
| G4 | Performance per game mode and map | Does one mode lag or regress? | missing | U |
| G5 | Reward trend vs rating trend | Is reward rising without strength rising (reward hacking)? | derived from G1 and C2 | — |
| G6 | Seed variance | Is a difference between runs real? | process (two seeds for decisions) | O |

### H. Behaviour

| ID | Diagnostic | Question it answers | Status | Cost |
|---|---|---|---|---|
| H1 | Replays: recorded games at low cadence, viewable | What does the agent actually do? Most exploits and bugs are found by watching. | missing | O |
| H2 | Game analytics (fog, physics, frontline) | How is the game being played in aggregate? | have | U |
| H3 | Degenerate-strategy detectors (edge camping, never shooting, ...) | Has it found a cheap exploit? | partial (via H2) | U |

### I. BC phase (BC is its own phase in the rewrite, 4c)

| ID | Diagnostic | Question it answers | Status | Cost |
|---|---|---|---|---|
| I1 | Train and held-out loss, and their gap | Is the student learning the teacher, and overfitting? | missing | U |
| I2 | Top-1 agreement with the teacher per factor | How often does the student pick the teacher's action? | missing | U |
| I3 | Student entropy vs target entropy | Does the student match the entropy we shaped the targets to (4c)? | missing | U |
| I4 | Loss on the student's own states vs teacher-driven states | Is compounding error (covariate shift) present; is the DAgger mix right? | missing | U |
| I5 | Closed-loop score vs the teacher | Does imitation translate into play? | partial (old BC win-rate logic) | C |

### J. League

| ID | Diagnostic | Question it answers | Status | Cost |
|---|---|---|---|---|
| J1 | Pool composition and games per entry | Who is in the pool and how often each is played? | decided (4b) | U |
| J2 | Drain progress | Are retiring entries leaving cleanly? | decided (candidate) | U |
| J3 | Entropy of the opponent-sampling distribution | Is sampling collapsing onto a few entries (the p(1-p) noise concern)? | missing | U |

### Checklist outcomes (author)

1. Adopt every missing and partial entry; T entries become tests, not plots.
2. H1 replays: state trajectories of one env from the existing pool every N updates,
   readable by the current viewer.
3. G4 per mode: short list only (win rate, rating, episode end reasons).
4. Open: warning triggers and a live dashboard (proposal below).

### Proposal: warning triggers and a watch dashboard

**Feasible, and cheap.** Every scalar is already on the host after the one readback
per update, so rules run in Python on numbers we already have: no extra sync, no
device work. Rule state (running means, peaks, cooldowns) is a few floats, saved in
the checkpoint so resume does not re-arm everything.

**Rule kinds**
- *Invariant:* the value has a known correct range, no calibration needed (skipped
  step, recompile after warm-up, C4 mismatch above ~0, memory above 90% of budget,
  support clipping).
- *Anomaly:* compared with the run's own history: z-score against an EMA mean and
  variance, or a drop from the running peak. No absolute threshold to choose before
  we have calibrated runs.
- *Trend:* a slope over a window (parameter norm growth, dormant share rising,
  reward rising while rating is flat).

**Severity**
- *Alert:* flag plus `wandb.alert` (email/Slack through W&B), rate-limited per rule.
- *Warning:* flag only, visible on the dashboard.

**Against alert fatigue:** warm-up suppression (no rule fires in the first N
updates), hysteresis (must hold k consecutive updates), per-rule cooldown, few
alert-level rules. Thresholds get tuned after the first runs (post-implementation
TODO).

**Outputs**
- `alerts/{rule}`: 0/1 per update, so a flag lines up in time with its plot.
- `alerts/active`: count of active flags, the one number to glance at.
- A W&B table of fired events (update, rule, value, threshold, which plot to open).

**Dashboard:** defined in code (W&B's `wandb-workspaces` package) and versioned in
the repo, so every run opens with the same layout. Sections, top to bottom:
Alerts; Strength (rating and SE, pool scores); Health (throughput, time breakdown,
skipped steps, recompiles, memory); Policy (entropy, KL, clip, C4); Value (EV, bias,
support use); then every other group in collapsed sections. Fallback if that
package's API proves unstable: a saved workspace made once by hand.

### Trigger and dashboard outcomes (author)

1. Design and starting rule list adopted (rules as listed in chat: alerts for skipped
   step, recompile, memory over 90%, C4 mismatch, throughput below 50% of its EMA,
   rating more than 2 SE below peak; warnings for entropy collapse, EV below 0,
   support clipping over 1%, feature rank 30% below peak, update ratio out of band,
   dormant share rising, observation clip share over 1%, low label validity, low
   sampling entropy, reward up while rating flat).
2. Delivery: dashboard only for now; no `wandb.alert`.
3. Only the skipped-step rule acts (checkpoint plus batch dump, 4d); the rest report.

## Group 1: histograms

**What exists** (`ppo.py` `_update_epochs`, logged by `logging.py`): every
`histogram_interval` (10) updates, from the **last primary minibatch of the last
epoch** (1/32 of the batch), alive ships only:
- `hist/logprob`: log-probability of the taken action under the current policy.
- `hist/returns/{level}`: the lambda-return targets per reward level (one W&B
  histogram per level).
The same interval also gates the actor/critic gradient-split measurement
(`measure_split = record_histograms and not diagnose_update`), an unrelated
coupling.

**Assessment**
- `hist/logprob`: its mean is (minus) the entropy already logged per factor. What a
  histogram adds is shape, e.g. many near-certain decisions plus a random tail. That is
  better read as scalars: per-decision entropy quantiles (p10, p50, p90) and the share
  of near-deterministic decisions (max probability > 0.9). Scalars plot as lines over
  training; histograms in W&B are hard to compare across time. **Proposal: drop,
  replace with entropy quantiles.**
- `hist/returns/{level}`: what matters for the categorical critic is whether targets
  fit the code's range and how spread they are per level. **Proposal: replace with
  scalars:** return p5/p50/p95 per level (the summed return's p5/p95 is already logged
  by the normaliser) and HL-Gauss support use per level (share of target mass in the
  padding bins or clipped at the edge).
- Raw arrays for offline plots: optional dump of the sampled minibatch's returns,
  advantages and log-probabilities to the run's artifact folder at a low cadence (for
  example every 100 updates), not to W&B.
- **Decouple** the actor/critic split from the histogram flag; everything diagnostic
  runs on the one diagnostics cadence (10 updates).

**Candidate addition: action usage.** Share of each action factor (power 3, turn 7,
shoot 2: 12 scalars) among committed learner decisions, every update (cheap: a
count). It is the direct check that the new 2/3 stick level is used at all, and it
shows drift in style (boost share, shooting rate) over training.

## Rules and method (author)

- **Avoid host syncs.** Metrics stay on device through the rollout and the update and
  are read back once per update (or once per diagnostic cadence). No per-step or
  per-minibatch `.item()`; the per-epoch KL read for early stopping is the one
  deliberate exception.
- **Cost context** (`docs/engineering/rl-throughput.md`): histograms, scalar metrics,
  logging, schedule, checkpoints, GAE and labels total 1.7 s per update (1.1%). The
  actor/critic split costs one forward and two backwards on one micro-batch per
  cadence. The full gradient decomposition costs up to 17 traversals and forces an
  eager update (in JAX: a separately jitted function at the cadence only). The
  dedicated evaluator was 53.9 s per update, ~35% of wall clock (removed, 4b).
- **Lens for every plot:** what is its purpose, and what failure or question does it
  diagnose? A series with no answer is dropped.
- **Log raw values, never pre-smoothed** (author). Smoothing is for display (W&B's
  built-in smoothing) or post-processing. A logged value may come from an EMA only when
  that EMA is itself the quantity being reported, with the justification written down.
  Current justified cases: the return normaliser's p5/p95/scale (the EMA state is the
  value the update actually divides by) and the live rating (a fit, not a smoothing).
  Trigger rules keep EMAs as internal state; those are not logged.
- **No expected-behaviour assumptions** (author). A diagnostic is never pruned
  because the quantity "should" be flat or zero; that is what it must be able to refute.
- **Coupled series: name quantity first** (`{group}/{quantity}/{slice}`) and consolidate
  in the dashboard (one panel per quantity, slices as lines), not by dropping series.
  Report-only views (heatmaps over training time) are built offline.
- **"Derivable" is not a reason to drop** (author). A derived series can serve its own
  purpose (a headline, a direct answer). A series is dropped only when it is an
  identical duplicate, answers no question that a kept series answers better, or is
  misleading (for example a mixture whose proportions drift and so reads as learning).
- **Order:** first a generic checklist of diagnostics any BC or RL project needs
  (finds what we already have, which passes automatically, and what is missing);
  then the audit of today's groups against it. Group 1 (histograms) above is
  provisional until then.

**Status after the checklist:** the three questions are answered by adopted checklist
entries: `hist/logprob` dropped for D2 (entropy quantiles, near-certain share);
`hist/returns/{level}` dropped for per-level return p5/p50/p95 plus E3 (support use);
D3 adds action usage. Decoupling the actor/critic split from the histogram flag stands.
Raw dump: **dropped** (author); the 4d skipped-step dump covers the case that needs raw data.

## Group 2: `overview/*`

**What exists** (`logging.py`): 16 copies of other series under one prefix, plus a
mean of explained variance across levels, so one W&B section shows the headline plots.

**Purpose:** a curated first screen. The dashboard defined in code now does that job
by referencing the original series. **Proposal: drop the prefix entirely.** The
cross-level EV mean also goes: levels have different scales and the dashboard shows
them per level.

## Group 3: `loss/*` and `loss_proxy/*`

**What exists** (`ppo.py` `_update_epochs`): minibatch means of every loss term,
raw (`loss/*`) and times its coefficient (`loss_proxy/*`), averaged over the update.

**Assessment, term by term**
- `loss/total`: a sum of incommensurable terms; no failure shows here first (non-finite
  values are A1, spikes are visible in the terms). Drop.
- `loss/policy_gradient`: the PPO surrogate with normalised advantages averages near
  zero whatever happens; its trend does not mean learning. The update-size signals
  (KL, clip fraction, ratio) are the diagnostics. Drop.
- `loss/entropy`: equal to minus the entropy already in `policy/`. Drop.
- `loss/value`: equal to the sum of `critic/value_loss` per level. Drop here, keep per
  level under `critic/`.
- BC terms (`behavioral_cloning`, `behavioral_cloning_kl`, `scripted_entropy`): gone
  with the joint paradigm (4c); the BC phase has I1-I4.
- `sigreg`, `next_state_cont`: dropped earlier.
- Auxiliary terms (next state, enemy action, density, shot trajectory, shot outcome):
  real supervised learning curves, keep. **Proposal: move each into its head's own
  group next to its baseline** (`next_state/loss`, `enemy_action/loss`,
  `density/loss`, `shot/loss_trajectory`, ...), so a head's loss and the number that
  says whether it is good are on one panel.
- `loss_proxy/*` (coefficient times loss): meant to say which term dominates, but the
  size of a loss says little about the size of its gradient (a 3-nat CE can have a
  flat gradient). The gradient shares (B3) answer that question directly, at the
  diagnostic cadence. **Proposal: drop the prefix.** The renaming to
  `loss_weighted/*` from the first verdict lapses.

Net: the `loss/` and `loss_proxy/` prefixes disappear; per-head losses move into their
head groups; the value loss stays per level under `critic/`.

**Outcomes (author):** group 2 dropped. Group 3 as proposed, aggregates dropped too
(`loss/total`, `loss/value`, `loss/policy_gradient`, `loss/entropy`; free to log, but
none answers a question its parts do not answer better).

## Group 4: `policy/*`

**What exists** (`ppo.py` `_loss_and_diag`, `_update_epochs`), each a mean over every
minibatch of every epoch, alive learner ships only:
- `policy/kl`: k3 estimator of KL(behaviour || current), measured before each
  minibatch's step. The per-epoch mean also drives the `target_kl` early stop (the one
  `.item()` sync, kept).
- `policy/clip_fraction`: share with |ratio - 1| > clip.
- `policy/ratio_mean`, `policy/ratio_max`.
- `policy/entropy_{power,turn,shoot}`: entropies of the marginals of the joint 42-way
  softmax (diagnostic only; the objective uses the joint entropy, which is not logged).
- `policy/{level}`: coefficient times the clipped surrogate per reward level.

**Assessment**
- `policy/kl` mixes two different things: the first minibatch of epoch 0 is the
  behaviour-vs-learner mismatch (C4, should be ~0), later ones measure drift during the
  update. **Proposal: split** into `policy/kl_first` (first minibatch, before any step:
  C4) and `policy/kl` (mean over the last epoch: how far this update moved, the number
  `target_kl` is about). Both are already computed; no extra cost.
- `policy/clip_fraction`: keep (update size, standard).
- `policy/ratio_mean`: drop. Sampling under the behaviour policy makes E[ratio] = 1 by
  construction; a deviation is a weak mismatch signal that C4 measures directly.
- `policy/ratio_max`: replace. One maximum over 12.78M tokens is set by the rarest
  low-probability action and is noise. Proposed: `policy/far_ratio_share`, share with
  |ratio - 1| > 2 x clip (tokens pushed well past the clip, the tail that matters).
  A mask and a mean, no sort.
- `policy/entropy_{power,turn,shoot}`: keep. **Add** `policy/entropy` (joint, D1;
  already computed for the loss) and `policy/factor_coupling` = sum of marginal
  entropies minus joint entropy (>= 0; the total correlation). It says whether the
  policy coordinates its factors (for example "boost only while turning") instead of
  choosing each independently. Free.
- `policy/{level}`: ~~drop~~ **correction (group 5 audit):** these are not logged
  scalars. They are the term names of the per-level gradient decomposition
  (`_reward_policy_terms`, grouped as `reward_policy`), so they only appear inside the
  `grad_*` series, which is exactly the B3 view. Nothing to drop; they move with the
  decomposition under `grad/`. The same holds for `value/{level}`.
- Move `train/epochs_completed` here as `policy/epochs_completed`: it is the result of
  the update-size control.
- Adopted checklist entries that land in this group: D2 (`policy/entropy_p10/p50/p90`,
  `policy/near_certain_share`), D3 (`policy/usage/{factor}/{option}`), D4
  (`policy/switch_rate` per factor, computed in the rollout).

**Outcomes (author):** all four accepted (KL split, `far_ratio_share` replaces
`ratio_max`, `ratio_mean` dropped, joint entropy and `factor_coupling` added,
`epochs_completed` moved). The `policy/{level}` item is corrected above.

## Group 5: critic, returns, normaliser

**What exists**
- `critic/explained_variance/{level}`: from the **last epoch** only (`ev_epoch` resets
  each epoch), so the critic has already fitted these targets for 3 epochs.
- `critic/value_loss/{level}`: the HL-Gauss cross-entropy per level (includes the
  target's own entropy, so its floor is not zero).
- `critic/return_mean/{level}`, `critic/value_pred_mean/{level}`.
- `critic/outcome_cross_entropy`: the outcome head.
- `returns/{level}` (`returns/component`): team-mixed return mean per level, from GAE.
- `returns/advantage_std/{level}`: RMS of the per-level advantage (no mean removed).
- `returns/aggregate`, `returns/aggregate_std`: mean and std of the summed return.
- `returns/scale`: snapshot of the normaliser's scale.
- `return_normalizer/{p5,p95,scale,floor_bound}`: the DreamerV3 normaliser; `floor_bound`
  is 1 while the p95-p5 spread is under the floor (one win).
- `value/{level}`: not logged; gradient-decomposition term names (see group 4).

**Assessment**
- **Explained variance, measure it before fitting.** Last-epoch EV is in-sample and
  flatters the critic. **Proposal:** `critic/ev/{level}` from epoch 0 (each minibatch
  before its step: the critic has not trained on these returns, an honest predictive
  score) and `critic/ev_fit/{level}` from the last epoch. The gap is how much the critic
  fits this batch beyond what generalises. Both come from moments already accumulated.
- **Value loss -> KL.** Report `critic/value_kl/{level}` (CE minus target entropy, zero at
  a perfect fit), as decided for HL-Gauss codes. CE stays the trained quantity.
- **Value bias (E2):** replace the two means with `critic/value_bias/{level}` =
  mean(prediction - return target). Keep `critic/return_mean/{level}` (reward drift per
  level); drop `critic/value_pred_mean` (it is return mean plus bias).
- **Support use (E3):** add `critic/support_clip/{level}`, share of target mass in the
  padding bins or clipped at the edge.
- `returns/{level}`: drop. With team spirit 0 (decided) the mixing is the identity, so it
  equals `critic/return_mean/{level}` up to the mask.
- `returns/scale`: drop, duplicate of `return_normalizer/scale`.
- `returns/aggregate`, `returns/aggregate_std`: drop. The normaliser's p5/p95 already
  describe the summed return's spread, and its drift shows per level in `return_mean`.
- **Advantages (D6).** Per-level RMS is not what the actor sees: it sees the summed,
  normalised advantage. **Proposal:** `returns/adv_rms` (RMS of the normalised aggregate
  advantage: the actor's signal strength against the entropy bonus) and
  `returns/adv_mean` (should be near 0; a drift means value bias reaches the actor).
  Drop the per-level RMS. "Share clipped by the normaliser" from the checklist does not
  apply: the DreamerV3 normaliser divides, it does not clip; D6 is amended.
- Normaliser: keep `p5`, `p95`, `scale`, `floor_bound` (`floor_bound` says advantages
  are not being amplified while returns barely spread).
- `critic/outcome_cross_entropy`: move to the `outcome/` group (next).
- Prefixes: `critic/` for the value function, `returns/` for targets, advantages and the
  normaliser (merging `return_normalizer/` into it).
- Return percentiles per level (adopted in group 1, p5/p50/p95): under `returns/`.

**Outcomes (author):** all accepted (epoch-0 EV plus `ev_fit`, `value_kl`, `value_bias`
with `return_mean`, the advantage aggregate, normaliser merged into `returns/`), with
one amendment: **team spirit may not stay at zero.** `returns/{level}` is then the
team-mixed return the actor is credited with, which differs from the critic's own
return. Kept as `returns/mixed/{level}`, logged only for levels whose team spirit is
non-zero (a static config check at trace time: no cost while spirit is zero).

## Group 6: `outcome/*`

**What exists** (`critic.py` `outcome_calibration`, the 4-class outcome head: win,
loss, tie, unresolved; targets are categorical lambda-returns):
- `outcome/calibration_brier`, `outcome/predicted_win`, `outcome/realised_win`.
- `outcome/calibration_bin_{0..9}/{predicted,realised}`: a 10-bin reliability diagram
  as 20 line series.
- `critic/outcome_cross_entropy` (moved here in group 5).

**Finding: the calibration only sees the last seconds of each game.** A transition is
scored only if its episode ends **inside the same rollout**. A rollout is 128 decisions
(about 4.3 s at 30 Hz); a game is up to 9,000 (5 min). So every number above measures
the last ~4 s before the end, where the winner is nearly obvious. The question "are
win probabilities honest?" is unanswered for 98% of the game.

**Proposals**
- **Prediction ledger, on device.** Per env, a fixed ring of slots: every 150 steps
  (5 s) store P(win | resolved) and the elapsed time. At episode end, score every
  filled slot against the result and add into a fixed table on device (time-to-end
  bucket x calibration bin: sums and counts). 60 slots per env, static shapes, no sync;
  read with the update's one readback. Gives:
  - `outcome/brier_skill/{bucket}`: Brier skill score (1 - Brier / Brier of the
    base rate) per time-to-end bucket (for example 0-5 s, 5-15 s, 15-30 s, 30-60 s,
    1-2 min, 2+ min). Raw Brier depends on the win rate; the skill score does not.
    The curve says how early the critic knows who will win.
  - `outcome/ece`: expected calibration error over the whole game, one scalar for the
    question "are the probabilities honest?".
  - `outcome/bias`: mean predicted minus realised win (replaces the two means).
- **Reliability bins:** drop the 20 line series; log the table as one W&B table at the
  diagnostic cadence, for drilling in when `ece` moves.
- **Loss:** `outcome/kl` (cross-entropy minus target entropy), as for the value codes.
- Games that never resolve within the ledger (time limit) score as a tie, as today.

**Revision (author asked "why 150 steps, why not all?"):** 150 was arbitrary (sized to
the smallest bucket). Better: score **every step**, exactly, with running sums.
Brier for a binary result w is (p - w)^2 = p^2 - 2pw + w (w is 0 or 1), so each env
only needs, per 1-second block of elapsed time (30 steps), the sums n, sum p, sum p^2.
At episode end the result w turns each block's sums into its Brier sum, and the block
is assigned to its time-to-end bucket (boundary error under 1 s). For the whole-game
ECE, each env also keeps n and sum p per probability bin (10 bins).
- Memory: 300 blocks x 3 floats x 3,840 envs = 3.5M floats, about 14 MB, plus 20 floats
  per env for ECE. Storing every raw prediction instead would be about 69 MB in bf16
  and gives nothing the sums do not.
- Sampling density is not the limit: predictions 33 ms apart are nearly identical, so
  the information comes from the number of finished games (about 55 per update if
  games run long). Numbers are read per update but are meaningful smoothed over the
  diagnostic cadence. (Superseded wording: nothing is logged smoothed; see the rule
below. Smoothing is display-side only.)

**Outcomes (author):** ledger as exact running sums with the buckets above; `ece`
plus a reliability table at the diagnostic cadence replace the 20 bin series;
`outcome/kl` and `outcome/bias` adopted.

## Group 7: next state

**What exists** (`ppo.py`, `next_state_populations.py`). G = number of code groups
(position levels, 3 velocity, attitude levels, angular velocity, shield delay, health,
power, local log index, cooldown):
- `next_state/{group}`: CE per code group over all supervised tokens (G series).
- `next_state_visible/{group}`, `next_state_hidden/{group}`: the same split by sight
  (2G).
- `next_state_ce/{population}/{group}` and `next_state_baseline_ce/{population}/{group}`
  for ally, enemy_visible, enemy_hidden: the head and the "nothing changes" baseline
  (6G).
- `next_state_population/{population}/{token_share, ce, baseline_ce}` (9).
- Gradient split by population and environment half: belongs to `grad/` (later group).

About 9G + 9 series, all from sums already accumulated on device.

**Purpose:** is the head learning dynamics (allies, visible enemies) and belief (hidden
enemies) better than "nothing changes", per population and per code group?

**Assessment**
- `next_state/`, `next_state_visible/`, `next_state_hidden/` are all derivable from the
  population table: hidden is exactly enemy_hidden (allies are always visible), visible
  is ally plus enemy_visible weighted by token share, the aggregate is all three. They
  are mixtures whose proportions move with the fog, which is why the population split
  was added. **Drop all three (3G series).**
- CE and baseline CE as separate series: the question is the gap. Both are CEs against
  the same HL-Gauss target, so their difference cancels the target's entropy and is in
  pure nats. **Replace** with `next_state/gain/{population}/{group}` = baseline CE - CE
  (nats per token the head beats "nothing changes" by; below 0 means it is worse than
  copying its input), and per population `next_state/gain/{population}` (mean over
  groups) and `next_state/kl/{population}` (CE minus target entropy, as for the other
  codes: how far from perfect).
- `token_share/{population}`: keep (what the batch teaches; under 3c's per-population
  averaging it also says how noisy each population's mean is).
- **Add (C6):** `next_state/supervised_share`, share of alive tokens with a valid
  next-state label (rollout edge and death masks).
- Prefix: everything under `next_state/`.

Net: about 9G + 9 series become 3G + 10, with the question answered directly.

**Reasoning restated (author: "derivable" alone is not a reason):**
- `next_state_hidden/{group}`: identical duplicate of the enemy_hidden population.
- `next_state/{group}`: misleading. Its population mix drifts with the fog, so it moves
  without learning; and after 3c it is not the optimised objective (a weighted sum of
  per-population means). The training loss stays as `next_state/loss` (group 3).
- `next_state_visible/{group}`: its purpose (one dynamics headline) is served by the
  ally and enemy_visible gains; the token-weighted pooling drifts with the visible
  share. If a single headline is wanted, a fixed-weight mean of those two gains.
- Earlier drops re-checked: identical duplicates (`returns/scale`, `overview/*`,
  `returns/{level}` at spirit 0); no question answered (`ratio_mean`, surrogate,
  `loss/total`, `loss_proxy/*`); `value_pred_mean` answers neither the bias question
  (`value_bias`) nor the drift question (`return_mean`).

**Outcomes (author):** the three mixtures dropped (no fixed-weight headline); `gain` per
population and group plus `kl` per population replace head and baseline; 
`supervised_share` added.

**Update after "Memory and belief" (2026-10-21, decisions.md):** the head now has two
outputs, residual (allies, visible enemies) and absolute (all three populations), so
every series gains an output slice, e.g. `next_state/kl/{population}/{output}` (one
panel per quantity, populations and outputs as lines). Hidden ships no longer have an
input code, so their "nothing changes" baseline is undefined. **Decided (author):** KL
only for hidden ships here; the physical-unit comparison against dead reckoning lives in
group 8. Token share stays; it is also the input to the tempered weights (alpha = 0.5).

## Group 8: `belief/*`

**What exists** (`ppo.py` `_precompute_belief_diagnostics`, once per update from the
rollout's stored forecasts against privileged truth, enemies only):
- 14 error names: absolute error per physical channel (position px, velocity px/s,
  attitude rad, angular velocity, shield delay s, health, power, cooldown s, local log
  index), `position_beyond_legal_frac` (share beyond 10 calibrated position scales,
  25 px), and baselines: persistence (position, velocity) and dead reckoning (position).
- 10 slices: `visible`, `hidden`, and hidden by age: 0-0.1, 0.1-0.5, 0.5-1, 1-2, 2-5,
  5-10, 10-30, 30+ s.
- 140 series.

**Purpose:** the next-state gains (group 7) say in nats whether the head beats
"nothing changes". This group says it in physical units against the strongest cheap
baseline (dead reckoning), and how fast tracking degrades the longer an enemy stays
hidden. Neither question is answered elsewhere. Keep the group.

**Assessment**
- **The `hidden` aggregate is misleading:** a mixture over ages whose proportions follow
  play style. If the agent learns to keep enemies hidden longer, it rises with no change
  in the belief. Same case as group 7: **drop it**; the age buckets are the answer.
- **Age buckets for every channel (112 series):** the age curve is about drift.
  Position and velocity drift; the resource channels evolve almost deterministically
  (regeneration, cooldown ticks) and their error is small and flat. **Proposal:** age
  buckets only for position, velocity, the baselines and `beyond_legal` (6 x 8 = 48);
  the other seven channels once over all hidden enemies (7). Purpose of the
  all-hidden form there: "is the resource belief broken", not a curve.
- **Visible slice:** keep all 14. `position_beyond_legal_frac` on visible enemies is a
  bug detector: on a trained head it should be about 0, and a non-zero value means a
  belief lifecycle leak (the comment says so). **Add a warning rule:** visible
  beyond-legal share above ~0 after warm-up.
- **Add token share per age bucket** (`belief/hidden_share/{bucket}`, 8 series): says
  which buckets rest on few tokens and so are noisy, and shows the age distribution the
  buckets sample. (Overlaps `fog/` hidden-age stats; check in that group.)
- ~~Torus wrap in the position error: drop it.~~ **Correction (author):** the world is
  always a torus; map centres are random and the playable area can straddle a seam,
  especially on large maps. The minimum-image error stays (as 3e already recorded:
  "deltas keep the minimum image"). My mistake: I conflated "no torus-only modes" with
  "no torus".
- Source: the forecasts are the behaviour policy's (one update stale). State it in the
  description; no change.

Net: 140 series become 14 + 48 + 7 + 8 = 77, plus one warning rule.

**Revision (author):**
1. **Torus:** minimum-image error kept (correction above).
2. **No expected-behaviour assumptions in diagnostics:** "health should be flat with age"
   is exactly what a diagnostic must be able to refute. All 14 errors keep all age
   buckets. Rule added to the method.
3. **Consolidation by plot type, not by dropping.** The coupling is a layout problem:
   - **Log flat raw scalars** (needed by trigger rules and post-processing), but name
     them **quantity first**: `belief/{quantity}/{slice}` (for example
     `belief/position_px/age_2_5s`, `belief/position_px/visible`, baselines as
     `belief/reckon_position_px/{slice}`). Today it is `belief/{slice}/{quantity}`.
   - **One dashboard panel per quantity**, every slice as a line: visible plus the 8
     age buckets on a sequential colour ramp (light = young, dark = old), log y.
     Position and velocity panels overlay their baselines dashed. 14 panels instead of
     140 charts; each answers "how does this error grow with age, and is it improving".
   - The `hidden` all-ages aggregate: dropped (misleading mixture, as group 7). With
     the slices in one panel nothing is lost.
   - **Age x training-time heatmaps** (one per quantity: x = update, y = age bucket,
     colour = error) for reports, built in post-processing by the repo's offline
     figure set (`charts/`), not logged.
   - Token share per age bucket: one panel, stacked area.
   This naming and panel rule applies to every coupled group (next-state gain
   population x group, outcome skill by bucket, action usage), not only belief.

**Units (author):** "px" is a legacy name for a world unit; everything outside the
renderer is in metres (decisions.md, Units). Belief series become `position_m`,
`velocity_m_s`, `reckon_position_m`, and so on.

**Outcomes (author, group 8):** quantity-first naming and one panel per quantity,
here and as the general rule; offline heatmaps later; no visible-leak warning rule and
no per-bucket token share.

**Update after "Memory and belief" (2026-10-21, decisions.md):** there is no belief
store; the group measures the **absolute output's** forecast (decoded mean), which is
also what sets damped rotary geometry for hidden ships. Slices, baselines and naming are
unchanged. **Add** the decided sigma calibration: per quantity and slice, the mean of
(error / sigma)^2 (1 when calibrated) and the share of errors within 1 and 2 sigma.
Group name `belief/` kept (it is the model's belief, now a readout).

## Group 9: `enemy_action/*`

**What exists** (`ppo.py`, the enemy-action head: the enemy's committed joint command,
42-way, privileged label; its output becomes the pending-action belief in the next
observation). Over enemy tokens, alive and committed:
- `loss/enemy_action` (moving to `enemy_action/loss`, group 3).
- `realized_probability` (mean probability on the true action), `entropy`, `brier`,
  `accuracy` (top-1).
- `persistence_accuracy`: share of enemy decisions equal to the previous one.
- `persistence_cross_entropy`: CE of a "repeat the last action" predictor with a fixed
  0.99 on repeat and the rest spread over the other 41.
- `uniform_*`: constants, dropped earlier.

**Purpose:** does the head predict what enemies are doing better than the cheap
alternatives, and is the belief it writes into the next observation honest?

**Update after "Memory and belief" (2026-10-21, decisions.md):** the prediction is no
longer written into the next observation; the head is an auxiliary loss only. The
second half of the purpose, and the calibration argument below ("fed back as an input"),
lose their main motivation. The four questions stay open for when the audit resumes.

**Assessment**
- **The persistence baseline is set by an arbitrary constant.** With 0.99 hard-coded,
  "beats persistence" depends on that number. **Replace** with the best persistence
  predictor for this batch: repeat with probability r (the measured repeat rate, which
  `persistence_accuracy` already is), else uniform over the other 41. Its CE is
  -r ln r - (1 - r) ln((1 - r)/41), from one count. **Add** a second baseline, the
  batch's action marginal (always predict the empirical action frequencies): its CE is
  the entropy of the action histogram, also one count per action.
- **Gain, as in group 7:** `enemy_action/gain/{baseline}` = baseline CE - CE, nats per
  token the head beats repeat and marginal by.
- **Visible vs hidden enemies: the same mixture problem as groups 7 and 8.** Predicting
  a visible enemy's next command and a hidden one's are different tasks, and the mix
  drifts with the fog. **Proposal:** every series sliced by `visible` / `hidden`
  (quantity first: `enemy_action/{quantity}/{slice}`).
- **Calibration of the belief:** the head's distribution is fed back as an input, so
  over- or under-confidence matters directly. **Replace `brier` and
  `realized_probability`** (proper-score views of what CE already measures) with
  `enemy_action/confidence` (mean top probability) beside `accuracy`: confidence above
  accuracy means overconfident.
- `entropy`: keep (how sharp the belief the model receives is).
- **Per factor:** CE of the marginals for power, turn and shoot, like the policy's
  marginals: which part of the enemy's command is predictable (shooting may be easy,
  turning hard). Free from the joint.
- **Optional slice by opponent kind** (self-play, checkpoint, average, scripted,
  semi-scripted): modelling the scripted agent and modelling itself are different
  tasks. Uses the per-env opponent id the league already keeps.

**Outcomes (author, group 9, 2026-10-21, after the feedback loop was removed):**
1. Baselines from the batch replace the hard-coded 0.99: optimal repeat-rate
   persistence, CE = -r ln r - (1 - r) ln((1 - r)/41), and the batch action marginal
   (entropy of the action histogram); `enemy_action/gain/{repeat, marginal}` in nats.
2. Every series sliced `visible` / `hidden`, quantity first.
3. Calibration dropped: no consumer of the probabilities any more and CE already
   scores it. `brier` and `realized_probability` dropped (restate CE); `accuracy` and
   `entropy` kept; no confidence series.
4. Per-factor CE (power, turn, shoot marginals of the joint) added; opponent-kind slice
   (self-play, checkpoint, average, scripted) on gains and accuracy only.
5. Whether the head earns its cost as an auxiliary loss is a gradient question for the
   `grad_*` group.

**Residual vs absolute logging (author's question, decided):** separate, as a slice.
- Group 7: `next_state/{kl, gain}/{population}/{output}`, output in {residual,
  absolute}; residual exists only for ally and enemy_visible. The "nothing changes"
  baseline is the same for both outputs, so the gains compare directly.
- Group 8: the hidden-age slices measure the absolute output (the one that sets rotary
  geometry); the `visible` slice logs both outputs, so the comparison is also in metres.
- Gradient share per output (how much each output moves the trunk): `grad_*` group.

## Group 10: `global_density/`

**What exists** (`ppo.py` `_global_density_loss`; the hex density head of 3e, Poisson
log-rate per hex cell for allies and enemies, target the true soft counts, privileged):
`loss/global_density`, `loss/global_density_{ally,enemy}`, `global_density/deviance`,
`global_density/{ally,enemy}_deviance`, `loss_proxy/global_density` (dropped in group 3).
Decided in 3e but not built: fraction of deviance explained per team (EMA per-cell null)
and the enemy channel split by number of hidden enemies.

**Purpose:** ally field = sanity check (allies are always visible); enemy field = how
well the model knows the enemy fleet's layout, hidden ships included (with Griffin-only
memory, a direct readout of what memory holds).

**Outcomes (author, 2026-10-21):** kept minimal (a first, larger proposal was judged
overcomplicated).
- `global_density/loss`: the optimised objective.
- `global_density/deviance/{ally, enemy}`: excess nats per cell over a perfect
  predictor (0 = exact). A Poisson loss has no natural zero, so the per-team losses are
  dropped; the combined deviance is dropped (the per-team lines answer it better).
- `global_density/r2/{ally, enemy}`: fraction of deviance explained,
  `1 - D_model / D_null`, with D_null the deviance of the batch's per-cell mean count
  (1 perfect, 0 no better than the average, below 0 worse). Comparable across modes,
  maps and fleet sizes, unlike raw deviance. The null is the batch mean, not 3e's EMA:
  exact, stateless, no smoothing.
- Not built: the enemy split by hidden ships (group 8 already measures hidden tracking
  by age); can return as a targeted check.
- Panels: one for deviance, one for R2, ally and enemy as lines. 5 series.


## Group 11: `shot/`

**What exists** (`ppo.py` `_shot_losses`; the counterfactual shot heads of 3f, ported
as they are until the item 15 benchmarks): `loss/shot_{trajectory, outcome}`, trajectory
CE per part (displacement, direction, speed, local log index), and realized vs
predicted rates for enemy hit, ally hit and clear miss. 12 series.

**Purpose:** does the head learn where shots go and what they hit, and are its hit
rates right?

**Outcomes (author, 2026-10-21):**
- `shot/loss/{trajectory, outcome}` (group 3 move).
- `shot/trajectory_kl/{displacement, direction, speed, index}`: KL replaces CE (CE
  includes the target's entropy and has no zero; KL is 0 when perfect).
- `shot/{enemy_hit, ally_hit, clear_miss}_rate/{realized, predicted}`: three panels, two
  lines each; the gap is calibration. The realized enemy-hit rate doubles as the
  policy's accuracy, logged nowhere else.
- **Add** `shot/supervised_share`: share of own-ship rows with a valid shot label
  (checklist C6; the end-of-rollout label loss, ~23%, is part of the item 15 benchmarks).
- 13 series.

## Group 12: `episode/`

**What exists** (`logging.py`, finished episodes, ~55 games per update):
`reward_{mean,min,max}` per ship-episode, `reward_{level}`, `event/{name}`,
`length_mean`, `lifespan_mean` (steps), `win_rate` (per ship-episode, self-play and
league pooled), `team0_win_rate`, `team1_win_rate`, `draw_rate`, and `alive_fraction`
(`ppo.py`, share of ship-steps alive in the batch). In Frontline a timed-out match is
awarded by front position (`apply_timeout_result`).

**Purpose:** how games go in aggregate: length, how they end, what happens, what each
reward term pays.

**Outcomes (author, 2026-10-21):**
- Kept (seconds per the units rule): `episode/length_s`, `episode/lifespan_s`,
  `episode/reward_mean`, `episode/reward_{level}`, `episode/event/{name}`,
  `episode/alive_share`, and `episode/reward_{min,max}` (only catch for a single
  ship's reward exploding).
- `episode/result/{team0, team1, draw}`: side-balance check (spawn or map bias in
  self-play), not performance (G1, G2).
- Dropped `episode/win_rate`: pools self-play and league, so it moves with the opponent
  mix, not skill.
- **Add** `episode/end/{decided, timeout}` (checklist C3): timed-out Frontline games are
  adjudicated and look decided in the results otherwise.
- **Add** `episode/count`: finished games per update, the sample size behind every
  series here.
- About 13 series plus one per reward level and one per event.

## Group 13: `fog/`

**What exists** (`logging.py`, perception stats per update): `visible_fraction`,
`range_only_visible_fraction`, `field_occluded_fraction_of_in_range`,
`never_seen_fraction`, `individual_visible_fraction`, `team_shared_gain`,
`mean_hidden_age_steps`, `reacquisitions` (raw count), `visible_fraction/scale_{bucket}`,
`occlusion_duration_le_{edge}` (raw counts).

**Purpose:** game analytics (H2): how much is hidden, why (range or fields), for how
long, how much team vision helps; context for the memory diagnostics (group 8).

**Outcomes (author, 2026-10-21):**
- Computed over **fogged envs only**, plus `fog/fogged_env_share` (mixed fog decision:
  fog-free envs would make every share drift with the fogged share).
- `fog/never_seen_share` kept as an **invariant**: must be 0 (every ship is revealed at
  spawn); alert trigger above 0.
- Kept, renamed (share, seconds): `fog/visible_share/{all, scale_{bucket}}`,
  `fog/range_visible_share`, `fog/field_occluded_share`, `fog/individual_visible_share`,
  `fog/mean_hidden_age_s`.
- Dropped `team_shared_gain`: exactly the gap between the team and individual lines in
  one panel.
- Batch-size independent: `fog/reacquisitions_per_slot_min` (per enemy slot per minute)
  and `fog/occlusion_share/{bucket}` (histogram as shares).

## Group 14: `physics/` + `frontline/` + `map_scale/`

**What exists** (`logging.py`, wrapper source stats, mostly per live ship-step):
`physics/{combat,boundary}_damage_per_live_ship_step`,
`physics/{combat,boundary}_deaths_per_million_live_ship_steps`,
`physics/nonambient_live_ship_fraction` (local refractive index != 1, i.e. inside a
field), `physics/mean_power`, `physics/mean_speed`, `physics/out_of_power_fraction`;
`frontline/shield_recharge_per_live_ship_step`,
`frontline/respawns_per_million_live_ship_steps`, `frontline/front_advances` (raw sum of
|front delta|), `frontline/simultaneous_captures` (raw count); `map_scale/mean`,
`map_scale/enemy_slot_share/{1, 1_2, 2_plus}`.

**Purpose:** game analytics (H2) and degenerate-strategy detection (H3: boundary
camping, never shooting, power starvation).

**Outcomes (author, 2026-10-21):**
- Readable units, batch-size independent: `physics/damage_per_ship_s/{combat,
  boundary}`, `physics/deaths_per_ship_min/{combat, boundary}`, `physics/speed_m_s`,
  `physics/power_mean`, `physics/out_of_power_share`, `physics/in_field_share`.
- `frontline/front_speed_m_s` (mean |front speed| over running env-time) replaces the
  raw summed distance; `frontline/simultaneous_captures_per_env_min` replaces the raw
  count; `frontline/shield_recharge_per_ship_s` kept.
- Dropped respawns: in Frontline every death respawns, so it repeats the death rates.
- `map_scale/mean` and `map_scale/slot_share/{1, 1_2, 2_plus}` kept (C5: the batch has
  the map-scale mix the config asks for).
- 15 series.

## Group 15: `reward/`

**What exists** (`logging.py`, over all running ticks): `reward/event_share/{event}`
(each event's share of its level's absolute reward) and
`reward/zero_sum_residual/{level}` (per-tick |sum over ships| of a level's reward; 0 when
zero-sum, positive by design under the offensive premium).

**Purpose:** checklist C2: which reward term drives learning; is one swamping the rest?
`event_share` answers it within a level only, and `episode/reward_{level}` cannot answer
it across levels (in zero-sum self-play each level's mean is ~0 whatever its scale).

**Outcomes (author, 2026-10-22):**
- `reward/event_share/{event}` kept.
- **Add** `reward/level_share/{level}`: each level's share of total absolute reward
  across levels (the cross-level half of C2).
- `reward/zero_sum_residual/{level}` kept, units reward per env-second; revisit with
  tabled item 13.

## Group 16: `seat/`

**What exists** (`ppo.py` `_seat_symmetry_metrics`, host syncs every update):
`seat/own_pending_invalid` (own pending action not an exact one-hot) and
`seat/enemy_pending_invalid` (enemy pending action not a valid distribution; that was
the fed-back enemy-action belief).

**Outcomes (author, 2026-10-22):** group dropped.
- The enemy check has nothing to validate (enemy pending action is masked: zeros plus
  the visible flag); the own check is a deterministic adapter property.
- Replaced by an **adapter test**: own pending action exactly one-hot, enemy pending
  action masked, in both teams' views (with the per-view conformance tests).
- Runtime input sanity across all features is checklist C1 (to assign with the
  checklist items without a group).
