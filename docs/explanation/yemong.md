---
covers: [src/bnb/models/yemong/, src/bnb/controllers/yemong.py]
reviewed: 2026-11-01
---

# The model (Yemong)

Yemong is the model, in `bnb.models.yemong`: v1's model with the same ideas and math, in
new code and the v2 units. The fixtures in `tests/reference/yemong` pin each module
against v1's on fixed inputs and weights. The spec is
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
  the trunk one step and the heads, and advances the belief with the forecast. It skips
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

## Configuration

`bnb.models.yemong.model.YemongConfig`. Its defaults are the configuration of v1's
trained model:

<!-- generated: yemong-config -->

`bfloat16_matmuls` runs the matrix products of the encoder, the trunk and the heads under
bfloat16 autocast. With it on, `step` and `unroll` agree only to bfloat16 rounding.

## Export

A model's single-step forward must export to ONNX, so a client can run agents in ONNX
Runtime. `tests/contract/test_yemong_export.py::test_yemong_step_exports_to_onnx` checks
that Yemong's does and that the export matches the eager step.
