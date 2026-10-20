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
- **Order:** first a generic checklist of diagnostics any BC or RL project needs
  (finds what we already have, which passes automatically, and what is missing);
  then the audit of today's groups against it. Group 1 (histograms) above is
  provisional until then.

**Status after the checklist:** the three questions are answered by adopted checklist
entries: `hist/logprob` dropped for D2 (entropy quantiles, near-certain share);
`hist/returns/{level}` dropped for per-level return p5/p50/p95 plus E3 (support use);
D3 adds action usage. Decoupling the actor/critic split from the histogram flag stands.
Open: the optional raw dump (proposed yes).

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
