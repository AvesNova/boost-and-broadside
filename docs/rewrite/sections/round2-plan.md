# Round 2 plan: accelerations, next-state baselines, the correction head (2026-11-08)

One place for the plan: what is decided, what runs, what is open, and what could fool us.
It supersedes `delta-heads-round2-notes.md` (kept for its findings). Every item marked
**open** is a proposal until the author decides it.

## 1. Goal and how we judge

- **Product:** the averaged agent (uniform average of BC, later RL). MVP: a BC clone, then RL
  to about 1500 Elo (scripted = 1000).
- **Method:** champion against challenger. Each challenger trains to 30M steps and is rated
  (final uniform average) against the champion's 30M uniform average, in frontline and
  deathmatch. The winner becomes the champion; the GPU otherwise continues the champion.
- **Judges, in order:** the tournament; then KL to the teacher and agreement; outcome Brier
  skill; the heads' own errors (next state, belief by hidden age).
- **Noise:** seed to seed about 30 Elo in frontline (round 0: C and C-seed1 played the same).
  A challenger must beat the champion by about 50 in frontline (about 40 on the mean of the
  two modes) to count. Brier skill moved by 0.2 with the seed alone: weak evidence.

## 2. Where we stand

| | |
|---|---|
| Champion | **C-fixed**: temporal-first order, belief spread fix (`spread_fit` coarsest), absolute next-state head. 30M: frontline -221 (C -266), KL 0.500, Brier skill 0.557. Continued: KL 0.415 at 60M, 0.349 at 105M |
| Round-1 delta arm | aborted at update ~5 (findings in `delta-heads-round2-notes.md`): the velocity baseline over-rotated; hidden shields never regenerated (expected-value events); position spread missed Cov(x, v) |
| Running | **Arm 1** (accelerations), queue job 4, ETA 03:25 (worst 03:45); its tournament job 5, ETA 04:30 (worst 05:15). Champion continuation (to 150M) resumes after |
| Running (CPU) | **Baselines bench** (`baselines-bench-brief.md`), measurements ~01:35, doc ~03:30 |

## 3. Decided

| # | Decision | When |
|---|---|---|
| D1 | Physical accelerations as inputs (along / across the velocity, angular), not the enemies' previous actions | 11-08 |
| D2 | In an extrapolation baseline, shield and countdowns use last tick's rate only when last tick had no event, else the rule rate | 11-08 |
| D3 | Three separate arms: (1) accelerations, (2) persistence deltas, (3) dead-reckoning (extrapolation) deltas | 11-08 |
| D4 | A CPU bench of the baselines before arms 2 and 3 | 11-08 |
| D5 | Physics head trained on allies and visible enemies only; still applied to hidden ships | 11-07 |
| D6 | A correction for hidden ships, in world coordinates | 11-07 (scope open: 5.4) |
| D7 | Velocity frame for the physics head's position and velocity (below 1 m/s: the nose, with a warning) | 11-07 |
| D8 | Events (recharge wait, gun cooldown) advance by the most likely outcome, not the expected value | 11-07 |

## 4. The arms (revised by the author, 2026-11-08 ~01:40)

| Arm | Contents | Compared with | Isolates | State |
|---|---|---|---|---|
| 1. Accelerations | C-fixed + `yemong.acceleration_inputs` (velocity-frame codes, computed in the model), absolute head | C-fixed 30M | the inputs | **running** |
| 2. Persistence, one head | arm 1 + persistence delta heads, **one next-state head trained on every row** (allies, visible and hidden enemies, as round 1); ellipse spread inputs, sigma by the physical-Jacobian propagation | arm 1 | the delta targets | to build |
| 3. Persistence, split | arm 2, but the physics head trains on visible rows only and a **separate error (correction) head** serves hidden rows (O4-O8; sigma from it) | arm 2 | the separation | to build |
| 4. Dead reckoning | arm 3 with the extrapolation baselines (O3: where the bench shows a gain) | arm 3 | the baseline | to build |

The author: arms 2 and 3 use the same target types, so 2 against 3 compares like with like;
then persistence against dead reckoning. All four carry the acceleration inputs (C8).

## 5. Components

### 5.1 Acceleration inputs (built, arm 1)

- (v_t - v_{t-1}) / dt along and across v_t; (omega_t - omega_{t-1}) / dt. Symlog HL-Gauss, 31
  bins, knees 3.8 / 14 m/s^2 and 1.55 rad/s^2, ranges 7740 m/s^2 and 741 rad/s^2, nothing in
  the padding. Zero with a flag on death, spawn, the first tick, just spotted.
- Today computed in the model from the rows it reads (truth if visible, the belief's forecast
  if hidden), with the previous row carried in `Carry.motion`.
- **Open O1, where:** move the visible ships' value into the observation as world-frame
  columns (ax, ay, angular) like every other hidden column; the model still fills hidden rows
  from the belief. Same numbers for visible ships, so arm 1's result carries over.
  Recommended: in the arms 2/3 build. Needs 9 (below) for old policies.
- **Open O2, frame of the input codes:** velocity frame (recommended: reveals the control
  directly, matches arms 2/3's targets), world (consistent with position and velocity; easier
  for arm 1's absolute head), or both (about 70 more columns).

### 5.2 Baselines (arms 2 and 3)

| Quantity | Persistence (arm 2) | Extrapolation (arm 3) |
|---|---|---|
| position | nothing changes | x + v dt + a dt^2/2, or the arc with the tangential acceleration (bench has both) |
| velocity | nothing changes | v + a dt, the tangential and normal parts applied in the velocity frame |
| heading | nothing changes | + omega dt + omega_dot dt^2/2 |
| nose rate | nothing changes | + omega_dot dt (**risk**: the nose oscillates; early bench: the tail overshoots, p99.9 9.7 vs 5.6 rad/s) |
| angle of attack, energy, ln n | nothing changes | last tick's change |
| shield, countdowns | nothing changes (+ event bins) | the rule (D2) (+ event bins) |

- **Open O3:** per quantity, take extrapolation only where the bench shows it helps (for
  example the nose rate may stay on persistence in arm 3).

### 5.3 Physics head (arms 2 and 3)

- Predicts each quantity's change from its baseline, velocity frame for position and
  velocity; symlog HL-Gauss, exact-zero centre, 2 padding bins; trained on allies and
  visible enemies (D5).
- Its spread is the one-tick process noise for a ship whose state is known.
- **Risk:** on hidden rows its inputs (wide belief codes) are unlike anything it trained on.
  Log its error on hidden rows by hidden age.

### 5.4 Correction head (arms 2 and 3)

- Trained on hidden rows only: target e = truth(t+1) - forecast(t+1), world coordinates.
- **Open O4, scope:** every continuous quantity (recommended: a wrong velocity belief would
  otherwise fight the position correction every tick), and for the event quantities, its own
  event probabilities for hidden ships (the visible rate is wrong for hidden ships).
- **Open O5, sigma from this head:** its predicted variance is the belief's error variance,
  calibrated against the truth; it replaces any analytic covariance propagation (no
  F P F^T for hidden ships). Recommended.
- **Open O6, cap:** on the step, not the target. Target: the uncapped e (symlog: +-26 km is
  ~67 bins); applied step r tanh(|E e| / r) toward E e; reported variance
  Var e + (E e - step)^2. Capping the target would cut the tails and understate sigma. r is
  open (per quantity).
- **Open O7, the label's physics term:** e uses the physics head's output, so the label moves
  as physics trains. Stop the gradient through it (recommended: physics learned on the many
  visible rows still helps hidden ones), or define e against the baseline only (a fixed
  label; the correction head then learns hidden physics itself).
- **Open O8, 2D shape:** heads code each axis independently (no correlation). Three
  projection axes at 0, 60, 120 degrees give the full 2 x 2 covariance from marginals (one
  more code group; clamp to positive semi-definite). Or two world axes (no tilt). The bench's
  frame correlation (6) informs this.

### 5.5 Spread inputs (arm 2 onward)

- Today position has one sigma; velocity has (xx, xy, yy).
- **Open O9:** both as an ellipse: log sigma_major, log sigma_minor, and the doubled angle
  (cos 2 theta, sin 2 theta) scaled by the anisotropy (sigma_major - sigma_minor) /
  (sigma_major + sigma_minor), so the angle fades out when the ellipse is round (raw tilt is
  undefined there).

### 5.6 Events

- Physics head: 2-bin events (reset / fired) for visible rows; advance by the most likely
  outcome (D8). Hidden rows: from the correction head if O4 is taken.
- Known limit: a hidden ship in a fight is hit with p 5-20% per tick; the most likely outcome
  never resets. Later: the wait and shield as a joint distribution (particles).

## 6. The baselines bench (running)

Recorded play (scripted and calm-fern, three presets). Per quantity and baseline: residual
quantiles, share exactly 0, entropy in a common code; split by control held / changed, field
band, hit. Axis correlation and ellipse shape per frame (world, nose, velocity), one tick and
dead reckoned 0.1-5 s (a stand-in for a hidden ship's error). Early read, one player and
mode, provisional: extrapolation cuts position and velocity medians 3-10x against the rule;
with the control held its residual is near float32 resolution; the nose rate overshoots.

## 7. Confounds and risks

| # | What could fool us | Mitigation |
|---|---|---|
| C1 | **Persistence + acceleration inputs is almost extrapolation.** With a_{t-1} as an input, the persistence head can learn "dv = a_{t-1} dt" as a copy. Arm 2 vs 3 then tests "the network does the copy" vs "the code does it", not physics learned vs given | Read arms 2/3 with that in mind; the bench's held / changed split shows how much is a copy |
| C2 | **Hand-holding.** Under extrapolation, held-control ticks leave ~0 to learn; the gradient lands on control changes (a few % of ticks), which is mostly predicting the others' policy. The trunk may learn less physics, or more intent | This is the question arms 2 vs 3 answer; watch play, not head loss |
| C3 | Loss scale: each arm's residual has different entropy, so the same coefficient gives a different trunk-gradient share | Measure the share at initialisation and set each arm's coefficient to match C-fixed's (as in round 1) |
| C4 | Arm 2 vs arm 1 changes the code (colour codes to symlog deltas), the head, the correction and the spread inputs at once | Accepted bundle; if it loses, split it |
| C5 | Physics head on hidden rows (out of distribution) | Logged; the correction absorbs the mean error |
| C6 | The correction label moves with the physics head (O7) | Stop-gradient, or the fixed label |
| C7 | Seed noise ~30 Elo; Brier skill varies 0.2 by seed | The bar in section 1; a second seed for a close call |
| C8 | Arm 1 decides whether arms 2/3 carry acceleration inputs, but arm 3 needs them regardless | If arm 1 loses, arms 2/3 still carry them (both, so 2 vs 3 stays clean) |
| C9 | The bench's data comes from the scripted agent and calm-fern (an old RL build), not from the BC clones being trained | Both players are reported separately; differences flag it |
| C10 | The champion's 30M snapshot is the reference; a challenger's 30M is compared with it, not with the 100M+ champion | Fine for the method (equal budgets); the product is still the longest-trained winner |

## 8. Infrastructure: old policies and new inputs

- Today a policy file stores its build record (so a model of any width rebuilds from it) and
  the **schema versions** of the observation and action contracts; loading refuses a policy
  whose versions differ from the code's. Yemong reads observation columns **by name**
  (`SHIP["velocity_x"]`), the toy MLP by width.
- So adding observation columns would bump the version and lock out every old policy (C-fixed
  included), although Yemong would read the old columns fine.
- **Open O10 (recommended):** record the observation's column names in the policy file, the
  way the build is recorded; at load, give an old model the columns it was trained with, by
  name, from the current observation. Adding a column then needs no version bump; removing
  or renaming one a policy uses fails loudly. Rule: a column's meaning never changes under
  the same name (a meaning change is a new name). The version stays for real breaks.
  Needed before O1.

## 9. Later (parked)

- Negative information (drop belief mass inside ally-visible regions).
- Particles (one shared head conditioned on each particle; winner-take-all corrections;
  joint wait and shield).
- Whether in-flight bullets reach the model.
- The padding rule (decisions.md) for the existing inputs and the critic.
- Backlog: compile time; CPU ratings `KeyError buf52`; the tournament stopping rule with
  sweeps.

## 10. Decisions O1-O10

**The author accepted every recommendation below (2026-11-08 ~01:40)**, and left the rest
to my judgment while away. O3 and O8 are settled from the bench's numbers.

| # | Question | Options | Recommended (accepted) | Blocks |
|---|---|---|---|---|
| O1 | Accelerations in the observation | now / with arms 2-3 / never | with arms 2-3 | arms 2-3 build |
| O2 | Frame of the acceleration codes | velocity / world / both | velocity | arms 2-3 build |
| O3 | Extrapolation per quantity | all / only where the bench shows a gain | where it helps | arm 3 |
| O4 | Correction scope | position (+heading) / all continuous + hidden events | all | arms 2-3 |
| O5 | Sigma from the correction head | yes / analytic propagation | yes | arms 2-3 |
| O6 | Cap on the step, uncapped target; r | yes / cap the target | step | arms 2-3 |
| O7 | Correction label's physics term | stop-gradient / baseline only | stop-gradient | arms 2-3 |
| O8 | 2D correction shape | 3 projection axes / 2 world axes | after the bench | arms 2-3 |
| O9 | Ellipse spread inputs | ellipse / today's | ellipse | arms 2-3 |
| O10 | Observation layout in the policy file | yes / no | yes | O1 |

## 11. Next

1. 01:35-03:30: bench results; I report the headline per quantity and update O3 and O8.
2. 03:25: arm 1 finishes; 04:30 (worst 05:15): its tournament against C-fixed 30M.
3. After the author decides O1-O10: one brief for the arms 2/3 build (with O10 first if O1
   is taken), then the runs through the queue.
