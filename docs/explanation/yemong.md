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
- with the acceleration inputs (below), each row's velocity, nose rate and sight at the
  last decision (`bnb.models.yemong.inputs.Motion`).

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
- **`"delta"`**: each quantity's one-tick change from its physics baseline
  (`bnb.models.yemong.delta`; the measurements behind it are in the state-deltas report,
  `docs/rewrite/sections/state-deltas.md`). The baselines read only the state at `t`:

  | Quantity | Baseline | Coded as |
  |---|---|---|
  | position | the constant-nose-rate arc, the game's drift at v_t | along and across the heading at `t` |
  | velocity | v_t turned by omega_t dt | along and across the heading at `t` |
  | heading | `heading_t + omega_t dt`, wrapped | |
  | nose rate, angle of attack, log refractive index | persistence | |
  | shield, energy | after the deterministic recharge; persistence | fractions of the type's maximum |
  | shield recharge wait | the countdown | an event, two bins: reset to the type's delay |
  | gun cooldown | the countdown | an event, two bins: fired (the type's interval less a tick) |

  Each continuous component is a **symlog HL-Gauss code**
  (`bnb.models.yemong.codes.symlog_encode`): the change's coordinate
  `sign(x) ln(1 + |x| / knee)` in bins, its centres evenly spaced from minus to plus the
  hard one-tick bound with an exact zero at the middle one, two padding bins on each side,
  and a Gaussian of 0.75 bins integrated over each bin. A change past the padded range is
  clamped to it (and counted); nothing folds: the Gaussian's mass past the outer edges is
  dropped and the code divided by its sum. Position and velocity have 51 real bins per
  component, the other six 31: 434 outputs in all (2 x 2 x 55 + 6 x 35 + 2 x 2), against
  the absolute code's 490.

  | Component | Knee | Bound (the last real centre) | Real bins |
  |---|---|---|---|
  | position, along / across (m) | 0.002 / 0.003 | 8.7 | 51 |
  | velocity, along / across (m/s) | 0.1 | 375 | 51 |
  | heading (rad) | 0.001 | 0.9 | 31 |
  | nose rate (rad/s) | 0.05 | 24.7 | 31 |
  | angle of attack (rad) | 0.003 | 0.49 | 31 |
  | shield (of the maximum) | 0.001 | 1 | 31 |
  | energy (of the maximum) | 0.0005 | 0.068 | 31 |
  | log refractive index | 0.0006 | 0.48 | 31 |

  The knees are the report's medians, the bounds its hard bounds (section 1.2, the box). The
  head's last weight starts at zero and its bias at the measured marginal: the share of
  exact zeros at the zero centre's code and the rest a Laplace of scale 1 in symlog units
  (median `|change|` at the knee), the events at their rates
  (`bnb.models.yemong.delta.DeltaCode.marginal_logits`).

  Decoding (`bnb.models.yemong.codes.symlog_decode`) reads the mean at the mean
  coordinate, exact on an encoded change (the round trip's bias is below 1e-3 bins up to
  0.06 bins short of the last real centre and 1.3e-3 bins at it), and the variance as the
  bins' physical variance less the code's own smoothing. The forecast fed back
  (`bnb.models.yemong.delta.DeltaCode.advance`, through `Yemong.forecast`) is the
  belief's baseline plus the decoded change, turned back from the belief's heading. Each
  variance is the belief's carried through the baseline to first order (the velocity's
  covariance turned, `dt^2` of it into the position, the nose rate's `dt^2` into the heading
  and the arcs) plus the change's; the events' means are their expected values, the
  cooldown's code the belief's counted down one bin mixed with the shot's. The training
  target is the truth's change, `truth(t + 1) - baseline(truth(t))`, on the same rows as the
  absolute head. The baselines' constants per ship type (recharge rate and delay, the ticks
  between shots) are in the build record (`YemongBuild.dynamics`), only for a delta build.
  Select it with `--set yemong.next_state=delta`.

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

The rows are the ones the model reads: truth where the ship is visible, the belief's
forecast where it is hidden (a mean only; no spread). Below 1 m/s the frame is the heading
at `t`. Each component is a symlog HL-Gauss code (`bnb.models.yemong.codes.symlog_encode`:
31 real bins, an exact-zero centre, sigma 0.75 bins, 2 padding bins per side, clamped,
nothing folded), and a validity flag follows the three. They end the ship token, and the
first projection's columns of the codes start as the state code's do. An input is zero with
the flag 0 when the difference is not a tick of motion: the ship dead at either tick, spawned
or respawned at `t`, the first tick of a game, or just spotted (hidden at `t - 1`, visible at
`t`, where the difference would be truth minus belief). Just hidden is valid: the forecast
minus the last truth.

The knees are the median `|value|` of living ships' one-tick pairs on the recorded
segments of the state-delta measurements (respawns excluded; both players, three presets,
pooled at equal weight; `bench/acceleration_inputs_measure.py`). Across is dominated by
the turn: `|v|` times the path rate. The bounds are the hard one-tick bounds of the change
under persistence (258 m/s and 24.7 rad/s; `docs/rewrite/sections/state-deltas.md`,
section 1.2) over the tick. Those bounds come from a loose box: the largest observed values
are 1214 m/s^2 and 307 rad/s^2, and none of the 16.3M samples reached the padding.

The previous tick's motion rides in the carry (`Carry.motion`), so `step` differences
against what it read last; `unroll` shifts the segment's own rows by one tick and takes
the first tick's from the carry at the segment's start, which the runtime stores with the
segment. The learner logs `input/acceleration/mean_abs/{component}/{visible, hidden}`,
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
| next state | ship rows | the next state code, as a change from the current one (absolute), or each quantity's change from its physics baseline (delta) |
| outcome | global token | win, tie, loss, unresolved |
| density | global token | ally and enemy rates on a hex grid that scales with the arena |
| shot outcome | ship rows | how a shot fired now would end |
| shot trajectory | ship rows, at bullet ages | where a shot fired now would be at each age |
| rotary read-out (optional) | ship rows, after each block's temporal sublayer | per block, a (cos, sin) pair per rotary frequency |

### Trunk order and the rotary read-out

`YemongConfig.order` sets each block's sublayer order: `"spatial_first"` (the default:
spatial, spatial, temporal) or `"temporal_first"` (temporal, spatial, spatial). Temporal
first, every spatial sublayer reads tokens that have already passed through the first
temporal sublayer, so attention can use what the recurrence remembers. Either way `unroll`
equals a loop of `step`, and the carry is the same shape. Select it with
`--set yemong.order=temporal_first`.

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
  inputs, the motion before each tick is the tick before's, the carry's at the start.

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
