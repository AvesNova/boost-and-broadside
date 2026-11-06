---
covers: [src/bnb/models/yemong/, src/bnb/controllers/yemong.py]
reviewed: 2026-11-06 15:30
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
- the view's **position offset**, drawn uniformly at every reset.

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
  enemies around it.
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
  uniform.

## Network

- **Encoder:** one first projection per token kind, then a shared one. Fields and zones are
  then projected into the map memory, of width `map_width`.
- **Trunk:** blocks of spatial sublayers followed by one temporal sublayer.
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
| next state | ship rows | the next state code, as a change from the current one |
| outcome | global token | win, tie, loss, unresolved |
| density | global token | ally and enemy rates on a hex grid that scales with the arena |
| shot outcome | ship rows | how a shot fired now would end |
| shot trajectory | ship rows, at bullet ages | where a shot fired now would be at each age |

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
  time. It computes every head on every row, for the learner.

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
