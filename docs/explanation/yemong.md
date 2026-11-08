---
covers: [src/bnb/models/yemong/, src/bnb/controllers/yemong.py]
reviewed: 2026-11-08 01:02
---

# The model (Yemong)

Yemong is the model, in `bnb.models.yemong`: v1's model with the same ideas and math, in
new code and the v2 units. The fixtures in `tests/reference/yemong` pin each module
against v1's on fixed inputs and weights. It is built from a build record alone
(`bnb.models.yemong.model.YemongBuild`, from `bnb.models.yemong.model.yemong_build`): its
config and the numbers it reads from the rules and the ship card, so a saved Yemong is
rebuilt without the current game ([Runs and checkpoints](runs.md)). The spec is
`docs/rewrite/sections/yemong-spec.md`.

The model and its controller (`bnb.controllers.yemong.YemongController`) run in the rollout,
and the learner trains it (see [The learner](learner.md)).

Nothing in the weights depends on the number of ships, the number of fields or the arena,
so one set of weights plays at every size.

## The model owns its inputs

Yemong reads the raw observation and composes its own tokens. Everything it remembers
lives in its [carry](glossary.md#carry) (`bnb.models.yemong.model.Carry`), per
[view](glossary.md#view):

- the trunk's recurrent state and convolution buffer;
- the **belief**, per ship row (`bnb.models.yemong.belief.Belief`): whether it holds a
  belief, its age, the ship state's means and spreads, and a forecast of the ship's next
  joint action;
- the own ships' previous joint actions;
- the view's **position offset**, drawn uniformly at every reset;
- with the acceleration inputs or the extrapolation baseline (below), each enemy row's
  velocity, nose rate and presence as the model read them at the last decision
  (`bnb.models.yemong.inputs.Motion`).

### The belief

Before each decision the belief assimilates sight; after it, it stores the forecast:

- a visible ship is truth;
- a hidden ship the team has seen since it last spawned keeps the model's forecast, with
  its spread;
- a ship not seen since it spawned holds no belief.

The forecast comes from the model's own next-state head, decoded back to physical units,
and from its enemy-action head.

`YemongConfig.forecast_feedback` (on by default) chooses whether the forecast is stored.
Off, nothing is written back after a decision: a hidden ship's row is what the team last
saw, its state at its last sighting with no spread, while its age keeps counting, and
every enemy row's action stays uniform. Everything that reads the belief follows: the
ship's state code, its position for rotary attention and local presence, its heading, and
its pending action. The trunk's temporal layers then carry the memory of hidden ships. The
next-state and enemy-action heads still train as auxiliary losses, unchanged; their
predictions are only not fed back. Select it with `--set yemong.forecast_feedback=false`.

`YemongConfig.rotary` (on by default) chooses whether attention is rotary. Off, the
rotation is the identity everywhere: every angle 0 (cos 1, sin 0) in the token and the map
tables, in the plain and the fused kernel path alike, so attention reads no relative
position or heading; positions reach the model only through the tokens' position codes.
The model trains and plays that way, and the switch is recorded in the build and the run's
config. Select it with `--set yemong.rotary=false`.

**Inference-time ablations.** `Yemong.step` also takes an `Ablations` (per view, bool):
`rotary`, the rotary tables are the identity on the view's rows (every angle 0, cos 1 and
sin 0, for the token and the map tables, so the fused attention kernel reads the same
tables); `forecast`, the view's belief is not advanced, as with `forecast_feedback` off.
A view with neither computes exactly what it would without them. The tournament uses them
to play an ablated copy of a trained policy (see the league page's published ratings);
they measure reliance on a feature the weights were trained with, not a model trained
without it.

### The position offset

Every position the model reads, of ships, fields, zones and the map centre, is shifted by
the view's offset. The categorical position codes then cannot memorize absolute map
positions, while rotary attention and local presence read only differences, where the
offset cancels.
`tests/sections/test_yemong_model.py::test_the_offset_reaches_the_outputs_only_through_the_position_codes`
checks it.

## Tokens

The query tokens are the ship rows, own ships first, then one global token. Fields and the
mode's zones are key and value memory only.

- **A ship token** is the **state code** of its belief (see below), then its team and type
  one-hots, its flags, the time since it was seen, its respawn wait as Fourier features,
  its pending action, the refractive-index gradient, and the local presence of allies and
  enemies around it; with `YemongConfig.acceleration_inputs`, its acceleration inputs last.
  With `YemongConfig.position_inputs` "fourier" the position and heading enter as Fourier
  features in place of their code runs (see below).
- **The global token** reads the map centre's code, episode progress, the arena radius,
  the fog flag and the mode's block.
- **A field token** reads its centre's code, radius, band width and level; **a zone token**
  its centre's code, radius, role, meter and the captures left. An absent zone is one
  learned vector (see [Mixed batches](game/mixed-batches.md)).

### The state code

The state code (`bnb.models.yemong.state_code`) writes a ship's believed state, each
quantity's mean and its spread, as one vector made of **softmax groups**: short runs of
probabilities, each run summing to one, like a probability distribution over a few bins.
The next-state head predicts the same vector, so a forecast carries its own uncertainty.

- **Binned codes** (`bnb.models.yemong.codes.scalar_encode`): a scalar such as the shield
  or the energy is spread over a fixed set of bins between a low and a high value. The two
  bins around the mean share its weight, and a wider spread moves weight to further bins.
  Velocity uses a two-dimensional binned code on a scale compressed above a knee speed.
- **Multi-level circular codes** (`bnb.models.yemong.codes.circular_encode`): a periodic
  quantity, the heading or a position coordinate, is coded at several levels, each with a
  period a fixed factor shorter than the one before. Each level is a softmax group over a
  few "colours" around its circle, so the coarse levels place the value roughly and the
  fine ones precisely. A spread larger than a level's period flattens that level toward
  uniform. Decoding reads the spread at the coarsest level that has broadened: a
  predicted code need not be Gaussian (a sharp core, two modes, flat fine levels), and its
  fine levels then say little about how far off the mean may be.
  `YemongConfig.spread_fit` chooses this reading, `"coarsest"`, or v1's fit, `"v1"`. The v1
  fit weights the levels so the finest informative one decides. On the codes a trained
  head predicts for a hidden ship it reported metres of spread while the ship was hundreds
  of metres off, and a 1-sigma range that held the truth 0-5% of the time past half a
  second. The config's default is `"v1"`, so every build record written before the option
  decodes as it trained (play, warm starts, tournaments, where the two builds are separate
  groups). A new run builds `"coarsest"` (`TrainConfig.yemong`). `"v1"` is to be removed
  once no rated policy needs it.

### The next-state head: absolute or delta

`YemongConfig.next_state` chooses what the next-state head predicts. Its inputs, the
state code above, are the same either way.

- **`"absolute"`** (the default): the state code of the state at `t + 1`. The head's logits
  add to the persistence baseline `log(code_t + 1e-3)` (`bnb.models.yemong.heads.next_state_logits`),
  so it starts as "nothing changes", and the forecast is the decoded code.
- **`"delta"`**: each quantity's one-tick change from its baseline
  (`bnb.models.yemong.delta`), chosen by `YemongConfig.delta_baseline`
  (`"persistence"` here; `"extrapolation"` [below](#the-extrapolation-baseline)). Under
  `"persistence"` nothing changes, so the target is
  `truth(t + 1) - truth(t)` (the measurements behind it are in
  `docs/rewrite/sections/baselines-bench.md`). Position and velocity are coded along and
  across the velocity at `t` (across to the left); below 1 m/s the frame is the heading
  (counted, `next_state/slow_frame_share`). The heading's change is wrapped. The shield
  recharge wait and the gun cooldown (in seconds) and the shield are their plain changes:
  a countdown's steady `-dt` and a reset's jump, the shield's regeneration and a hit's
  drop, so the model learns the linear regeneration and the countdowns. Shield and energy
  are fractions of the type's maximum.

  Each component is a **symlog HL-Gauss code**
  (`bnb.models.yemong.codes.symlog_encode`): the change's coordinate
  `sign(x) ln(1 + |x| / knee)` in bins, its centres evenly spaced with an exact zero at
  one of them, two padding bins on each side, and a Gaussian of 0.75 bins integrated over
  each bin. A change past the padded range is clamped to it (and counted); nothing folds:
  the Gaussian's mass past the outer edges is dropped and the code divided by its sum.

  | Component | Knee | Bound (the last real centre) | Real bins |
  |---|---|---|---|
  | position along the track (m) | 2.6 | 8.7 | 175 (19 below zero) |
  | position across the track (m) | 0.00628 | 0.7 | 51 |
  | velocity, along / across (m/s) | 0.134 / 0.471 | 35 / 41 | 51 |
  | heading (rad) | 0.0101 | 0.32 | 31 |
  | nose rate (rad/s) | 0.052 | 10.5 | 31 |
  | angle of attack (rad) | 0.00285 | 0.17 | 31 |
  | shield recharge wait (s) | 1/30 | 6 | 31 |
  | shield (of the maximum) | 0.00399 | 0.36 | 31 |
  | energy (of the maximum) | 0.000544 | 0.046 | 31 |
  | log refractive index | 0.00794 | 0.24 | 31 |
  | gun cooldown (s) | 1/30 | 0.1 | 31 |

  The knees are the bench's persistence medians (where the median is 0, the median above
  1e-6), the bounds its maxima rounded up. Position along the track is the whole move,
  about `|v| dt` (median 2.6 m, never near 0), so it has its own asymmetric code: 155 real
  bins above the zero and 19 below (to -0.51 m, a slow ship reversing), a bin 4.9 cm wide at
  the median move (2.5 cm at 0, 10 cm at 8.7 m). The code is 624 outputs (179 + 3 x 55 + 8
  x 35), against the absolute code's 490. The head's last weight starts at zero and its
  bias at the marginal measured on scripted play (`bench/persistence_marginal.py`): per
  component the share of exact zeros and 16 quantiles of the rest, each an atom whose
  Gaussian is as wide as its gaps to its neighbours
  (`bnb.models.yemong.delta.DeltaCode.marginal_logits`). Its first cross-entropy there is
  30.9 nats per row, against the targets' marginal entropy of 30.3.

  Decoding (`bnb.models.yemong.codes.symlog_decode`) reads the mean at the mean
  coordinate through a table that undoes the padded ends' lost spill
  (`codes.symlog_mean_inverse`): an encoded change decodes to itself within 2e-4 bins
  everywhere, padding included. The variance is the bins' physical variance less the
  code's own smoothing. The shield, the wait and the cooldown advance by the **median**
  instead (`codes.symlog_median`: the mean over the bins within 4 of the one where the
  cumulative mass reaches one half), the most likely outcome when a jump has a probability
  below one half: a hidden ship's wait counts down and its shield regenerates.

  The forecast fed back (`bnb.models.yemong.delta.DeltaCode.advance`, through
  `Yemong.forecast`) is the belief plus the decoded change, turned from the belief's own
  velocity frame to world; shield, energy and the countdowns are clamped to their ranges. A
  delta build's belief carries the **full covariance** P of (x, y, vx, vy, heading, nose
  rate), appended to its spreads (`state_code.COVARIANCE_STATE`, 21 entries); each tick
  `P' = F P F^T + Q`, with F the kinematics' Jacobian (position plus velocity `dt`) and Q
  the decoded changes' variances, position and velocity turned to world; the heading's
  variance is capped at a uniform circle's, `pi^2 / 3`, and the nose rate's at a uniform
  one over the state code's range, `8^2 / 3` (the nose rate is a bounded oscillator; with
  its variance growing as a random walk, a hidden ship's nose rate held the truth within
  one sigma 100% of the time past 0.5 s). The heading is not coupled to the
  nose rate in F: the head's heading change is the whole turn, nose rate `dt` included,
  and the nose rate is a damped oscillator, so the coupling counted the turn twice and
  grew the heading's sigma past pi by 1 s on a short CPU run, while the belief's heading
  error was 0.55 rad at 1-2 s and grew as the root of the age, as the heading change's
  own variance does. The other sigmas add the change's variance to the belief's; the cooldown's code
  is the belief's moved by the change's distribution. The spreads' usual slots (the
  position's per-axis RMS sigma, the velocity's covariance, the heading's and nose rate's
  sigmas) are read off P, so play, figures and diagnostics read a delta build as any other.
  A visible row is the truth with zero spread. Under persistence the head's position change
  for a hidden row already holds the belief's velocity error, which F adds again: the
  belief's coverage by hidden age (`belief/within_one_sigma/*`) is where that would show.
  On a short CPU run it showed the opposite: the position's sigma too small (14% within
  one sigma at 0.1-0.5 s, 5-9% past it).

  The target is the truth's change on every row (allies, visible and hidden enemies) whose
  ship is alive at `t` and `t + 1` (a death or a spawn is not a change of state). Round 1's
  rule baselines (the arc, the turned velocity, the recharge and the countdown rules, the
  two-bin events) are removed: a policy trained with them does not load (its build record
  holds the dynamics they read, and the model refuses it). Select the arm with
  `--set yemong.next_state=delta --set yemong.delta_baseline=persistence`.

### The extrapolation baseline

`YemongConfig.delta_baseline = "extrapolation"` (arm 4) carries last tick's rates one tick (`bnb.models.yemong.delta.DeltaCode.baseline`, the
baselines bench's E with the choices of `docs/rewrite/sections/round2-log.md`, O3):

- position: the arc of last tick's path turn at its mean speed, `x + Arc(turn) (v + du/2 u)
  dt`, exact on a steady turn and on constant acceleration along a line;
- velocity: `v + a dt`, last tick's tangential and normal acceleration carried from its
  velocity frame into the frame at `t`;
- heading: `+ omega dt + omega_dot dt^2 / 2`; the nose rate: persistence (E overshoots the
  damped nose);
- angle of attack and energy: last tick's change (energy: persistence after a shot);
- shield, recharge wait and gun cooldown: last tick's change without an event, else the
  rule (the shield's recharge once the wait is over, the countdowns; D2). The events are a
  reset of the wait (a hit always resets it) and a jump of the cooldown or the
  observation's shot flag;
- ln n: `ln n + grad . v dt` where the gradient is known (a visible row's observation),
  else last tick's change.

The rates (`delta.Rates`, `Yemong.rates`): a visible row reads the observation's
acceleration and its validity, its shot flag and ln n's gradient, and the other
quantities' last-tick change against the row as the model read it the tick before (which
rides in the carry, `Carry.previous`, and which `unroll` shifts within a segment). Where a
row has no valid rate (a first tick, just spotted, spawned) every rate is 0 and the events
take the rule. A hidden row **holds its last sighting's rates** (`Carry.held`): the
acceleration along and across its velocity (turned back with the belief's velocity), the
nose's and the scalars' last change, each decayed by `exp(-dt / tau)` per tick, tau
`YemongConfig.extrapolation_decay` (0.5 s), while the belief holds the ship; its shield
and countdowns take the rule. The brief had a hidden row difference its own belief, as
its acceleration inputs do. On a short CPU run that ran away: the belief's change holds the
head's decoded change, which came back as the next tick's acceleration and compounded, to
a velocity error of 1040 m/s at 2-5 s by update 8. With the held rates decaying at 0.5 s the
same short run (arm 2's BC config scaled to 64 environments of 32 ticks, float32, seed 1,
20 updates; the error head as arm 3's, eta 0.1) stayed stable, so no clamp on the speed
was needed. A hidden ship's heading also turns by its nose rate times that decay since the
sighting (`delta.Rates.turn`), so it stops turning. Updates 10-20, by hidden age 0-0.1 /
0.1-0.5 / 0.5-1 / 1-2 / 2-5 / 5-10 s: position error 0.04 / 1.8 / 11 / 39 / 142 / 388 m
(arm 2: 1.0 / 6.5 / 20 / 46 / 143 / 377; arm 3: 1.2 / 7.1 / 21 / 54 / 147 / 379),
velocity 1.6 / 11.8 / 28.1 / 47.6 / 64 / 82 m/s (arm 2: 1.9 / 12.4 / 29.2 / 39.9 / 62 /
80), heading 0.03 / 0.22 / 0.56 / 1.00 / 1.35 / 1.47 rad (arm 2: 0.03 / 0.17 / 0.37 / 0.51
/ 0.82 / 1.08). The turn's decay helped the heading a little (without it 0.25 / 0.63 / 1.07
/ 1.49 / 1.56), and it is still worse than arm 2's.

On recorded play the baseline equals the bench's E to float32
(`tests/sections/test_yemong_extrapolation.py`); below 1 m/s the bench takes last tick's
frame from last tick's heading, the model from today's (no living ship was that slow on the
bench's data). The forecast of a hidden row applies the same baseline to the belief, and
the error head is unchanged. The target is the truth at `t + 1` less the baseline of the
truth at `t` under the rates the model read: exact for the rows seen at `t`, and, for a
hidden row (a diagnostic under the split), its held rates on the true state. The build
records the shield rule's dynamics (`delta.StateDynamics`). Its code:

| Component | Knee | Bound (the last real centre) | Real bins |
|---|---|---|---|
| position, along / across the track (m) | 2.62e-4 / 1.17e-3 | 0.3 / 0.4 | 51 |
| velocity, along / across (m/s) | 0.0188 / 0.0731 | 26 / 26 | 51 |
| heading (rad) | 1.28e-3 | 0.38 | 31 |
| nose rate (rad/s; persistence) | 0.0848 | 10.5 | 31 |
| angle of attack (rad) | 2.11e-3 | 0.23 | 31 |
| shield recharge wait (s) | 0.0667 | 6 | 31 |
| shield (of the maximum) | 0.0363 | 0.36 | 31 |
| energy (of the maximum) | 8.08e-6 | 0.06 | 31 |
| log refractive index | 5.65e-4 | 0.09 | 31 |
| gun cooldown (s) | 1/30 | 0.1 | 31 |

The knees are the medians of the baseline's residual (where the median is 0: the wait 97%,
the shield 98%, ln n 97%, the cooldown 96%, the median above 1e-6) on the arm-2 recordings
of scripted self-play on this tree (10.6M living pairs with a tick before them;
`bench/persistence_marginal.py --baseline extrapolation`), the bounds its maxima rounded up;
nothing falls in the padding. The position along the track is a residual here, not the
whole move, so it has an ordinary symmetric code. 500 outputs. The head's bias starts at
the measured marginal (`delta.EXTRAPOLATION_MARGINAL`); its first cross-entropy there was
29.4 nats per row against the targets' marginal entropy of 29.2, in the code of the bench's
E knees the measurement started from (the re-measured knees differ by up to 1.6x). Select the arm with
`--set yemong.delta_baseline=extrapolation` on arm 3's line.

The baseline does not need the acceleration inputs. Its rates are computed the same way
with `acceleration_inputs` on or off (the enemy rows' `Motion` rides in the carry
whenever either needs it, and so do `Carry.previous` and `Carry.held`); the switch decides
only whether the ship token reads the accelerations. With it off (arm 4c, on C-fixed's
inputs) the token is C-fixed's plus the delta arms' own additions, and the head learns a
residual from a baseline whose rates it cannot see. That is the author's choice for round
2 (each round builds on the winner, and the acceleration inputs did not win): the residual
then carries what last tick's rates do not explain, and the model cannot tell when the
baseline took a rate from the tick before. On the same recorded segment its rates,
baseline and targets equal the inputs-on arm's exactly
(`tests/sections/test_yemong_extrapolation.py`). Select it with
`--set yemong.acceleration_inputs=false` on arm 4's line.

### The error head: the next-state work split

`YemongConfig.delta_split` (with `next_state = "delta"`) splits the next-state work in two
(arms 3 and 4, `docs/rewrite/sections/arms34-brief.md`):

- The next-state head becomes the **physics head**: the same head, code and targets as
  above, its loss masked to the own rows and the enemies seen at `t`. It still runs on the
  hidden rows, whose forecast reads it; they give it no gradient.
- A second head on the same tokens, the **error head** (`bnb.models.yemong.correction`),
  trains on the hidden enemies only. Its target is how wrong the physics forecast is,
  `e = truth(t + 1) - forecast(t + 1)`, where `forecast(t + 1)` is the belief at `t` plus the
  physics head's decoded change. The physics term is detached, so the error head's loss
  gives the physics head no gradient. The head is built after every other layer (the rest
  initialise as without it), its last weight zero and its bias the measured marginal.

Every quantity of the state code has a symlog HL-Gauss code of its **uncapped** error over
the full range a hidden ship's error can reach. Position and velocity are coded on **three
world projection axes**, at 0, 60 and 120 degrees, one code each. A hidden ship's error is
correlated across any fixed pair of axes, with a sign set by its turn
(`docs/rewrite/sections/baselines-bench.md`, 2.2), and three marginals recover the full 2 x 2
covariance. The error's location is the least-squares vector of the three projected
medians. The covariance S solves `u_i^T S u_i = var_i` for its three entries, clamped to
positive semi-definite (`correction.covariance_of`). The heading's error is wrapped. The others are
scalars, with shield and energy as fractions of the type's maximum. Bins are about
`1 + 4 ln(1 + range / knee)`, odd, so two per e-fold on each side of the zero:

| Quantity | Knee | Bound (the last real centre) | Real bins |
|---|---|---|---|
| position, per axis (m) | 11.9 | 26 244 (the position code's period) | 31 |
| velocity, per axis (m/s) | 9.09 | 639 (twice the velocity code's reach) | 19 |
| heading (rad) | 0.160 | pi | 13 |
| nose rate (rad/s) | 0.587 | 16 | 15 |
| angle of attack (rad) | 0.0267 | 1.05 | 15 |
| shield recharge wait (s) | 0.217 | 6 | 15 |
| shield (of the maximum) | 0.0421 | 1 | 13 |
| energy (of the maximum) | 0.00918 | 1 | 19 |
| log refractive index | 3.47e-4 | 1.39 | 35 |
| gun cooldown (s) | 0.0334 | 0.1 | 7 |

The rest's bounds are the spans of their state-code ranges. The knees are the median |e| on
a short CPU run of the arm at initialisation (`bench/correction_marginal.py`, 139 922 hidden
rows). The belief of a hidden ship is then the physics head's open-loop forecast. Where the
error is exactly 0 most of the time (the wait 92%, the shield 91%) the knee is the median
above 1e-6. The code is 338 outputs. The rare-jump quantities read their median within 2
bins (`correction.MEDIAN_WINDOW`; these codes are coarse, and the delta code's window of 4
bins would reach a jump's mode).

The advance of a hidden row (`correction.CorrectionCode.correct`, through
`Yemong.forecast`) is a rate-limited step toward the truth: `mean(t + 1) = forecast(t + 1)
+ step` per quantity, with the step `eta g m`, then capped. Here m is e's median (per
projection axis, then the least-squares vector), g the head's confidence in it, `|m|^2 /
(|m|^2 + sigma^2)` with sigma^2 e's variance per dimension (`correction.confident`), and eta
a fixed rate per tick (`YemongConfig.correction_rate`, default 0.1: a time constant of
about 10 ticks, 0.33 s). The cap is `r tanh(|eta g m| / r)` along `m` for position and
velocity, and `r tanh(eta g m / r)` for the heading, the nose rate, the angle of attack,
energy and ln n. Here r is the quantity's one-tick physical bound (`correction.STEP_LIMITS`,
the bench's persistence maxima: 7.99 m, 40.6 m/s, 0.311 rad, 10.2 rad/s, 0.163 rad, 0.0457
of the maximum energy, 0.232), so the belief moves toward the truth no faster than the ship
could move in a tick. The recharge wait, the shield and the gun cooldown step by e's median
itself, the most likely outcome. The head predicts e again every tick from the corrected
belief, so a fraction per tick still converges where it is right, and the loop's gain stays
below 1 where it is wrong or an update stale.

The brief asked for a step toward E e, capped at r. On a short CPU run that ran away: the
mean of a broad code over +-26 km is set by its tails, and an early head predicts one
offset for every hidden row. Stepped toward at up to r per tick and integrated over a
hidden ship's age, the push dominated the next update's errors (the replay is one update
stale), which the head then chased. By update 4 the step was at r for every hidden row and
the belief's position error at 1-2 s went 51, 27, 94, 128 then 459 m, against arm 2's
20-58 m. The median alone did the same a little later (118 m at update 4).

With the confidence gain and eta 0.1 the same short run (arm 2's BC config scaled to 64
environments of 32 ticks, float32, seed 1, 20 updates) stayed stable, its step at 6% of r
for position on average (1.4% of rows past r), and its belief matched arm 2's by hidden
age (updates 10-20, position error 1.2 / 7.1 / 21 / 54 / 147 m at 0-0.1 / 0.1-0.5 / 0.5-1
/ 1-2 / 2-5 s; arm 2 1.0 / 6.5 / 20 / 46 / 143 m). The sigma it reports now errs wide
where arm 2's erred narrow: the truth within one sigma of position 0.74 / 1.00 / 1.00 /
1.00 / 0.98 (arm 2: 0.42 / 0.11 / 0.07 / 0.08 / 0.04), velocity 0.80 / 0.98 / 0.93 / 0.84
/ 0.72 (arm 2: 0.66 / 0.31 / 0.21 / 0.21 / 0.13), heading 0.86 / 0.99 / 0.96 / 0.86 / 0.75
(arm 2: 0.70 / 0.50 / 0.43 / 0.43 / 0.35). The early error head is broad: its position
sigma is 450-680 m at every age against a realised spread of 3-130 m, and the GPU run's
`correction/predicted_variance/*` against `correction/realised_square_error/*` shows
whether it sharpens.

#### The error head's spread

Each code's sigma is a quantile spread (`codes.symlog_spread`), not its second moment. In
the code's bin coordinate it takes the 16th, 50th and 84th percentiles c16, c50 and c84,
with the cumulative mass interpolated across the bin that holds each level linearly in its
probit. Each side's half-width loses the code's own smoothing in quadrature, `a =
sqrt(max((c84 - c50)^2 - 0.75^2, 0))` and `b` likewise below. The physical sigma is `(v(c50
+ a) - v(c50 - b)) / 2`, v the code's value map. Its square is e's variance per code; for
position and velocity the three projected variances give the 2 x 2 as above.

The second moment failed on arm 3's first GPU run (stopped at update 4): the position
sigma was 218 m at 0-0.1 s of hidden age against a realised RMS error of 1.6 m, and every
error lay within 1 sigma up to 2 s. The outer bins of a code over +-26 km are kilometres
wide, so a sliver of mass there sets the second moment (1e-3 in a 10 km bin is 1e5 m^2).
That is the mean's failure again. The gain `|m|^2 / (|m|^2 + sigma^2)` was then about 0, so
the correction never acted, and the hidden ships' blurred inputs carried nothing. A quantile
moves by about the sliver's mass instead.

Two choices depart from the plain rule, each measured on encoded codes. Linear
interpolation of the cumulative mass widens an encoded value's half-width to 0.82-0.90
bins, depending on where it sits in its bin, so less 0.75 bins it never reaches 0 (3.5 m
at the centre of the position code). A Gaussian's probit is linear, so the probit
interpolation gives an HL-Gauss its exact quantiles and an encoded value decodes to 0. Both
sides are deconvolved apart because a symlog code folds near 0: a Gaussian off the centre
is lopsided in the bin coordinate. The symmetric `(c84 - c16) / 2` read a 200 m Gaussian at
20 m as 280 m; apart, it reads 192 m. A Gaussian of 1 to 50 knees decodes to its sigma
within -5% to +2% (low where the fold makes it two-humped in the coordinate). Below about a
bin's width at its location the code cannot resolve a sigma, and it reads less.

The rare jumps (the wait, the shield and the cooldown) take the same rule. Their second
moment erred wide as well (the wait's sigma 1.4 s against a realised 0.23 s at 0-0.1 s on
the short run above). A jump of probability above about 0.16 widens the spread by about
half its size. A rarer one leaves it near the spread at zero, so the spread reads the most
likely outcome, as the step does.

On the stopped GPU run's policy at update 4 (decoded on a CPU rollout of 64 environments
x 32 ticks, learning rate 0), the position sigma by hidden age is 7.1 / 6.7 / 9.8 / 12 /
29 / 78 m at 0-0.1 / 0.1-0.5 / 0.5-1 / 1-2 / 2-5 / 5-10 s, against a realised RMS error
about the median of 0.85 / 4.9 / 18 / 51 / 166 / 366 m. The rule now errs narrow past
0.5 s (0.22 / 0.09 / 0.03 within 1 sigma at 0.5-1 / 1-2 / 2-5 s), and so does the head:
its own raw 16th-84th percentile interval holds 0.39 / 0.23 / 0.13 of the errors there
(position on the 0-degree axis), and the rule's interval 0.31 / 0.19 / 0.11. Its median is
unbiased (0.47-0.54 of the errors fall below it). The narrowness belongs to the early
head's overconfident core, which the rule reports as it is; the second moment hid it under
the far bins' mass. On 20 CPU updates from initialisation (arm 3 as above) the head stays
near its unconditioned marginal, and the two rules read about the same: a position sigma
of 470-720 m against the second moment's 445-815 m.

The belief's spread after the step is e's about the median plus the unapplied part,
`sigma^2 + (m - step)^2`: for position and velocity the 2 x 2 covariance plus the outer
product. The median is the location the step moves toward. It replaces the physics' `F P
F^T + Q` for hidden rows. The covariance is block diagonal (the
error head codes position, velocity, heading and the nose rate independently), with the
same caps on the heading's and the nose rate's variance. The cooldown's code is the
forecast's sharp code moved by e's distribution. The ellipse inputs and the per-axis blur
read this covariance. A row seen at `t` takes the
physics forecast, as without the split. Select the arm with `--set yemong.delta_split=true`
on arm 2's line.

`YemongConfig.physics_hidden_decay` (tau in seconds, with `delta_split`; off by default)
also trains the physics head on the hidden enemies, each row weighted `exp(-age / tau)` by
the time since its ship was last seen (the replay's age at `t`; a row of the segment's
first tick weighs 0, its truth lacking the tick before). Its target is truth to truth: the
code of the true change from the baseline of the true state at `t`, the extrapolation
baseline's rates read off the truth as a row seen at `t - 1` and `t` reads them, which is
the target a seen row of that ship gets. Its inputs stay the belief it reads; the error
head is unchanged. The physics term keeps its denominator, so a seen row weighs what it
did and the hidden rows add on top. Only the labels read the switch; it is in the build
record, and a policy file without it loads it as off. Select it with `--set
yemong.physics_hidden_decay=0.5` on the deadreckon-split line.

### Spread inputs: scalar or ellipse

`YemongConfig.spread_inputs` chooses how a ship row reads its belief's spread. The spread
enters the state code by blurring it: a sigma larger than a level's period flattens that
level, so the first layer sees which levels to trust.

- **`"scalar"`** (the default): the position code is blurred by one sigma for both axes,
  the velocity code by its covariance (its three projections' variances).
- **`"ellipse"`**: the position code is blurred per world axis by the position
  covariance's marginal sigmas (`sqrt(P_xx)`, `sqrt(P_yy)`), the velocity code as before,
  and the ship token ends with the position and velocity **ellipses**
  (`ShipStateCode.ellipse_code`): log sigma_major, log sigma_minor, and the doubled angle
  `(cos 2 theta, sin 2 theta)` times the anisotropy `(sigma_major - sigma_minor) /
  (sigma_major + sigma_minor)`, world frame, so the angle fades out where the ellipse is
  round. Each feature is a two-hot code of 21 bins, as the state code's scalars: a sigma as
  `ln(1 + sigma / knee)` from 0 to its bound's (position: knee 1 m, bound the position
  code's period; velocity: knee 0.1 m/s, bound the velocity code's reach), the angle
  features over [-1, 1]. 168 inputs, their first-projection columns initialised as the
  state code's. Without a covariance (an absolute build) the position ellipse is round.

### Position inputs: colour or Fourier

`YemongConfig.position_inputs` chooses what a ship row reads for its position and heading
(`ShipStateCode.inputs`). **`"colour"`** (the default) reads the state code's runs.
**`"fourier"`** reads `a_l (cos, sin)` of the phase at each level of the same periods
(`bnb.models.yemong.codes.fourier_encode`): per world axis `P / 3^l` for `l = 0..8` (the
finest 4 m), the heading `2 pi / 4^l` for `l = 0..3`. a_l is the colour code's
attenuation by the row's sigma, so a blurred level fades toward 0 as the colour level fades
toward uniform. The phase reduces the value modulo each level's period first, as the colour
code does, so float32 keeps the 4 m level. 36 position and 8 heading columns replace 81
and 16 (the ship token 500 wide for 553 on arm 4d). Only the encoder's input changes: the
tokens' `code`, the global, field and zone tokens, every head and every target stay the
colour codes. The absolute head predicts the colour code from the colour inputs, so
"fourier" needs `next_state="delta"`. Select it with `--set yemong.position_inputs=fourier`
on arm 4d's line.

### The acceleration inputs

The physics is deterministic, so a ship's change over the last tick reveals the control it
used, and controls are sticky. The state code cannot show that change: a velocity bin is
2.5 (1 + |v| / 50) m/s wide (17.5 m/s at 300 m/s), wider than any one-tick change from
thrust (about 7 m/s). `YemongConfig.acceleration_inputs` (off by default) gives each ship
row its acceleration over the last tick (`bnb.models.yemong.inputs.accelerations`):

| Input | Definition | Knee | Bound |
|---|---|---|---|
| along | (v_t - v_{t-1}) / dt along v_t | 3.8 m/s^2 | 7740 m/s^2 |
| across | the same across v_t, to the left | 14 m/s^2 | 7740 m/s^2 |
| angular | (omega_t - omega_{t-1}) / dt, omega the nose rate | 1.55 rad/s^2 | 741 rad/s^2 |

The rows are the ones the model reads. A visible row (every own row) reads the
observation's acceleration columns (`acceleration_x`, `acceleration_y`,
`nose_acceleration`, world frame, and `acceleration_valid`; see
[contracts](contracts.md#the-observation)) and turns them into the velocity frame at `t`. A
hidden row differences the belief's forecast against the row the model read the tick
before (a mean only; no spread). Below 1 m/s the frame is the heading at `t`. Each component is a symlog HL-Gauss code (`bnb.models.yemong.codes.symlog_encode`:
31 real bins, an exact-zero centre, sigma 0.75 bins, 2 padding bins per side, clamped,
nothing folded), and a validity flag follows the three. They end the ship token, and the
first projection's columns of the codes start as the state code's do. An input is zero with
the flag 0 when the difference is not a tick of motion: the ship dead at either tick, spawned
or respawned at `t`, the first tick of a game, or just spotted (hidden at `t - 1`, visible at
`t`, where the difference would be truth minus belief). For a visible row that is the
observation's flag, which is 1 only when the team saw the ship alive at both ticks. Just
hidden is valid: the forecast minus the last truth.

The knees are the median `|value|` of living ships' one-tick pairs on the recorded
segments of the state-delta measurements (respawns excluded; both players, three presets,
pooled at equal weight; `bench/acceleration_inputs_measure.py`). Across is dominated by
the turn: `|v|` times the path rate. The bounds are the hard one-tick bounds of the change
under persistence (258 m/s and 24.7 rad/s; `docs/rewrite/sections/state-deltas.md`,
section 1.2) over the tick. Those bounds come from a loose box: the largest observed values
are 1214 m/s^2 and 307 rad/s^2, and none of the 16.3M samples reached the padding.

For hidden rows the previous tick's enemy rows ride in the carry (`Carry.motion`; also
under the extrapolation baseline, whose rates read the same accelerations), so
`step` differences against what it read last; `unroll` shifts the segment's own rows by
one tick and takes the first tick's from the carry at the segment's start, which the
runtime stores with the segment. Own rows need no carry: they are always visible. Until
2026-11-08 the model differenced every row itself (run `2026-11-08-bc-arm-C-accel` trained
that way); the visible rows' numbers are the same, so that policy plays unchanged. The learner logs `input/acceleration/mean_abs/{component}/{visible, hidden}`,
`input/acceleration/padding_share/{component}`, `input/acceleration/invalid_share` (of the
living rows) and `input/acceleration/slow_frame` (of the valid rows). Select it with
`--set yemong.acceleration_inputs=true`.

## Network

- **Encoder:** one first projection per token kind, then a shared one. Fields and zones are
  then projected into the map memory, of width `map_width`.
- **Trunk:** blocks of spatial sublayers and one temporal sublayer each, in the order
  `YemongConfig.order` gives (see [Trunk order and the rotary read-out](#trunk-order-and-the-rotary-read-out)):
  by default the spatial sublayers first.
  - A **spatial sublayer** is attention of the queries over themselves and the map memory,
    with rotary encodings of position and heading, then a gated MLP. There is no mask.
  - A **temporal sublayer** runs per query token over time: a short causal convolution and
    an RG-LRU recurrence, gated, then a gated MLP. An episode start cuts both. The RG-LRU
    (real-gated linear recurrent unit) keeps one state vector per token and updates it as
    `h = a * h + sqrt(1 - a**2) * (i * x)`, where the decay `a` and the input gate `i` are
    computed from the input `x`, so the layer learns how long to remember.
- **Heads**, each an MLP:

| Head | Reads | Gives |
|---|---|---|
| action | own ships | logits over the joint actions |
| value | own ships | categorical logits per local reward level |
| enemy action | ship rows | the next joint action's logits |
| next state | ship rows | the next state code, as a change from the current one (absolute), or each quantity's change from persistence (delta) |
| error (with `delta_split`) | ship rows | the physics forecast's error for a hidden ship, per quantity |
| outcome | global token | win, tie, loss, unresolved |
| density | global token | ally and enemy rates on a hex grid that scales with the arena |
| shot outcome | ship rows | how a shot fired now would end |
| shot trajectory | ship rows, at bullet ages | where a shot fired now would be at each age |
| rotary read-out (optional) | ship rows, after each block's temporal sublayer | per block, a (cos, sin) pair per rotary frequency |

### Trunk order and the rotary read-out

`YemongConfig.order` sets each block's sublayer order: `"spatial_first"` (spatial,
spatial, temporal) or `"temporal_first"` (temporal, spatial, spatial). Temporal first,
every spatial sublayer reads tokens that have already passed through the first temporal
sublayer, so attention can use what the recurrence remembers. Either way `unroll` equals a
loop of `step`, and the carry is the same shape. A new run builds `"temporal_first"` (the
`TrainConfig.yemong` default, arm C's order); `"spatial_first"` is the config's own
default, so a build record from before the setting existed loads as it was trained.
`--set yemong.order=spatial_first` selects the other order.

`YemongConfig.rotary_readout` (off by default; it needs `temporal_first` and `rotary`) adds
one linear head per block, on that block's temporal sublayer's ship rows: every block does
the same thing, whichever it is. Each head gives a (cos, sin) pair for every rotary
frequency: x and y for each of the `rotary_octaves` position octaves, then each
heading harmonic (2 x (2 x 8 + 4) = 40 numbers at the defaults). The pairs are not
normalised: the read-out is trained toward the true phasors, whose expectation under the
model's uncertainty is shorter than 1, so a short pair means the model is unsure at that
frequency, and attention at that frequency is damped.

With the read-out, each block's spatial sublayers rotate a ship's row by:

- a **visible** ship: its true tables, from its observed position and heading;
- a **hidden** ship: that block's read-out pairs, in place of the belief's tables;
- the global token and the map memory (fields, zones): their exact tables, as without it.

The read-outs enter the tables detached, so each learns only from its own loss (the
learner's `rotary_readout` term, the mean over blocks of each block's error against the
true phasors of every living ship; see the [learner page](learner.md)). That keeps each
optimum the expected phasor and keeps the fused
attention kernel in use in the learner (the kernel gives no gradient to the tables). The
inference-time `rotary` ablation still makes every table the identity, the read-out's
included. Select it with `--set yemong.rotary_readout=true`.

**The temporal-first arms.** The BC comparison of the trunk order runs two arms beside
the baseline:

- **C**, temporal first with forecast feedback: hidden ships' rows and rotary positions come
  from the belief, as in the baseline (`--set yemong.order=temporal_first`);
- **D**, temporal first, Griffin only (no forecast fed back between ticks), with hidden
  ships' rotary pairs from each block's read-out: memory places a hidden ship where it thinks the
  ship is, not at its last sighting (`--set yemong.order=temporal_first --set
  yemong.rotary_readout=true --set yemong.forecast_feedback=false`).

The critic gives one value per reward level (`bnb.contracts.REWARD_LEVELS`): a per-ship
categorical value over symlog-spaced bins for each local level, and the outcome level as
`P(win) - P(loss)` from the global token (`bnb.models.yemong.critic`).

## step and unroll

- **`step(carry, obs, reset)`** is one decision for a batch of views. It clears the carry
  where the environment reset, assimilates sight into the belief, composes the tokens, runs
  the trunk one step and the heads, and advances the belief with the forecast (unless
  `forecast_feedback` is off). It skips
  rows nothing reads: the enemy-action and next-state heads run on enemy rows only.
- **`unroll(carry, obs, replay, reset)`** recomputes a stored segment: the encoder and the
  spatial sublayers run over every step at once, and the temporal sublayers scan over
  time. It computes every head on every row, for the learner. With the acceleration
  inputs, the enemy rows' motion before each tick is the tick before's, the carry's at the
  start.

`step` returns a **replay record** (`bnb.models.yemong.inputs.Replay`): exactly what the
decision read from its carry. It holds the belief of the enemy rows only: an own ship is
always visible, so its row is the observation's truth with no spread, and
`bnb.models.yemong.inputs.believed` rebuilds it for `step` and `unroll` alike. The runtime
stores the record, so `unroll` composes the same tokens without replaying the belief. **`unroll` equals a loop of `step`** over the same segment;
`tests/sections/test_yemong_model.py` and `tests/sections/test_modes_yemong.py` check it.

The controller samples the joint action from the policy logits by Gumbel-max
(`bnb.models.yemong.policy.sample`), stores the sample as the previous action, and
converts it to the factored `Action` with `bnb.contracts.action.from_joint`.

### Decoding the heads

`bnb.models.yemong.decode` turns the heads' outputs into what they predict, in the game's
units and frame: the shot path at each age with its sigma (in the launch heading's frame,
and `bnb.models.yemong.decode.launch_frame_to_world` to world coordinates), the next state
and the beliefs (positions less the view's offset, as the representative nearest a
reference), the density head's expected counts on the hex cells, the enemy action
distribution with its factor marginals, and the outcome's class probabilities. The figures
(`bnb figures`, [Figures](../guides/figures.md)) read the model through it only.

## Kernels

On CUDA, five pieces of the network run as Triton kernels (`bnb.models.yemong.kernels`):

| Switch | Replaces | Runs in |
|---|---|---|
| `scan` | the RG-LRU's linear recurrence over a segment | either precision (the recurrence is float32 in both) |
| `attention` | a spatial sublayer's rotary rotation and attention | either precision |
| `gated_mlp` | the gated MLP's hidden layer, `gelu(gate x) * (up x)` | bf16 only |
| `temporal_mix` | a temporal sublayer's gates, recurrence and output gate over a segment (it takes over from `scan` there) | either precision, segments only |
| `heads` | linear, norm, gelu, linear: the heads, the shot trajectory head's tail and the encoder's projections | bf16 only |

Each kernel computes the same function as the plain PyTorch code it replaces, which stays
in place as its twin: the plain path runs on the CPU, in float64 (the reference fixtures),
wherever a kernel does not apply, and wherever its switch is off.
`tests/sections/test_yemong_kernels.py` holds each kernel to its twin, forward and
backward, and checks that every kernel gives bitwise-identical results on identical
inputs, as the exact-resume tests need. Each is a `torch.library.triton_op` with autograd,
so it compiles under `fullgraph=True` and captures in the rollout's CUDA graph.

The switches are the fields of `bnb.models.yemong.kernels.Switches`, process-wide in
`bnb.models.yemong.kernels.switches`, and read when a function is traced: set them before
compiling. `bnb.models.yemong.kernels.plain` turns them all off for a block of code:

<!-- runnable -->
```python
from bnb.models.yemong import kernels

kernels.switches.heads = False  # one kernel off, from here on
with kernels.plain():  # every kernel off inside the block
    assert not kernels.switches.scan
kernels.switches.heads = True
```

A change to a piece with a kernel is made twice, in the plain path and in the kernel, and
the test keeps them equal; a new piece can be tried with the switch off first. A speed
comparison between model types runs both plain or both with their kernels.

Each kernel autotunes its tiles and warps for each shape it meets, over a few
configurations (`triton.autotune`), so every GPU picks its own; a configuration a GPU
cannot run is skipped. Tuning a new shape costs about a second, once.
`tests/sections/test_yemong_kernels.py::test_every_autotune_configuration_matches_the_plain_path`
checks every configuration against the plain path. The scan's and the gated MLP's
configurations give the same bits; the others' can differ in the last bits, so the choice
is cached on disk (Triton's and Inductor's autotune caches), and a later process on the
same machine, such as a resumed run, makes the same choice and repeats the bits.

## Configuration

`bnb.models.yemong.model.YemongConfig`. Its defaults are the configuration of v1's
trained model:

<!-- generated: yemong-config -->

`bfloat16_matmuls` runs the matrix products of the encoder, the trunk and the heads under
bfloat16 autocast. With it on, `step` and `unroll` agree only to bfloat16 rounding. Training
sets it from `TrainConfig.precision`, whose default is `"bfloat16"`; the residual stream,
the norms, the recurrent state, the codes, the belief and the critic's expectations stay
float32 under it.

## Export

A model's single-step forward must export to ONNX, so a client can run agents in ONNX
Runtime. `tests/contract/test_yemong_export.py::test_yemong_step_exports_to_onnx` checks
that Yemong's does and that the export matches the eager step.
