# Policy architecture

`YemongPolicy` is a centralized recurrent controller. It reads the full scene, exchanges
information across entities with spatial attention, carries per-entity memory through
time, and emits a factored action for every ship in the learned fleet. Its name (*Yemong*,
from the Korean 예몽, a dream that foretells the future) comes from the auxiliary head that
learns to predict the next state of the world. Zero-shot transfer across team sizes comes
from its variable-cardinality design, described [below](#why-team-size-can-change).

![YemongPolicy architecture](policy_architecture.png)

The diagram shows the reference-run configuration; component counts are set by
[`ModelConfig`](../src/boost_and_broadside/config/core.py), not fixed properties of the
architecture.

## Data flow

```text
global tensor state
    ↓
ship + refractive-field entity tokens          bullets (N·K)
    ↓                                              ↓
FeatureCoordinator → encoder MLP            bullet encoder
    ↓                                              ↓
[spatial Transformer ×S  ← cross-attends ──────────┘
 → temporal Griffin/RG-LRU ×T] × blocks
    ↓
ship tokens only
    ├── factored action distributions (per ship)
    ├── decomposed value estimates (per ship/component)
    └── next-state predictions (per ship)
```

For a batch `B`, `N` ships, `M` fields, and embedding width `D`, the shared trunk works
on `(B, N+M, D)` entity tokens. Fields participate in attention but not in recurrence;
bullets are key/value-only inputs to cross-attention and are never entity tokens at all.
The three output heads slice the first `N` ship tokens.

## Block structure

Every Yemong block has the same shape, set by [`ModelConfig`](../src/boost_and_broadside/config/core.py):
`n_spatial_per_block` spatial sublayers followed by `n_temporal_per_block` temporal ones,
repeated `n_yemong_blocks` times. The reference configuration is two blocks of
`2 spatial + 1 temporal`, so the trunk is `S S T | S S T`.

The ratio is deliberate. At these token counts a spatial sublayer costs roughly a quarter
of a temporal one at equal parameter count, so relational depth is the cheap axis to spend
on.

Four spatial layers is more depth than the reasoning appears to need. *"Fight the enemy
that is not already swarmed"* is two hops: one layer for each ship to aggregate its own
local situation, one for the decision. The tighter limit is head count, which sets how
many distinct aggregates a single layer can hold at once.

`n_bullet_cross_per_block` spatial sublayers cross-attend to bullets, counted from the
first. The front is where they have to go. A bullet read must precede at least one more
entity-to-entity layer; otherwise a ship can react to fire aimed at itself but can never
reason about fire aimed at an ally it might support.

## Observation and feature coordination

[`observation_from_state`](../src/boost_and_broadside/env/observation.py) exposes global
position, velocity, attitude, angular velocity, shield/health, power, cooldown, shield
recharge delay, team identity, alive state, radius, previous action, ship-local encoded
log index, and the local
refractive-index gradient. Fields are appended as always-alive entity tokens with team ID
2, zero motion/action channels, and numeric physical features: transition width, absolute
target log index.
Field properties are unchanged by team flipping.

The index gradient is the force term in `a = F/m + 0.5|v|² grad(log m) - (v·grad(log m))v`.
Without it a ship can see which medium it occupies but not which way that medium is
changing. It would feel the resulting acceleration with nothing in its input to explain
it.

The policy does not hand-encode those channels. The canonical
[`FeatureCoordinator`](../src/boost_and_broadside/train/rl/features.py) binds each raw
channel to:

1. an accessor from the observation dictionary;
2. a network-facing input transform;
3. where applicable, a target transform and predictor for the auxiliary dynamics loss.

| Feature | Network encoding | Auxiliary target |
|---|---|---|
| position x/y | four-frequency Fourier features over the toroidal period | phase delta |
| velocity | direction scaled by [symlog](https://arxiv.org/abs/2301.04104) speed | additive velocity delta |
| attitude | four-frequency Fourier features of the angle itself | phase delta |
| angular velocity | symlog scalar | next absolute value |
| health, power, cooldown | circular bounded encoding | phase delta |
| team identity | three-way one-hot | none |
| alive state | scalar | none |
| currently visible | scalar | none |
| belief token valid | scalar; also the attention/existence mask | none |
| time since observation | symlog scalar | none |
| previous power/turn/shoot | categorical one-hot | none |
| radius | shared ship/field scalar divided by half the shorter world dimension | none |
| field width | normalized scalar | none |
| field target log index | normalized physical scalar | none |
| shield recharge delay | symlog seconds | absolute prediction |
| ship-local log index | `log(n)/(2 log(s))` | additive next-step delta |
| ship-local index gradient | normalized `grad(n)` pair | none |
| ally / enemy presence | two `log1p` Gaussian aggregates, optional | none |

Phase targets make wraparound natural: crossing the map boundary is a small rotation, not
a large coordinate jump. Feature dimensions and prediction layout are derived from the
registered features rather than hardcoded in model code.

Ally and enemy presence are the two channels `local_presence` adds. Softmax attention
returns *proportions*, which is the invariant that survives a change in fleet size and is
exactly why it cannot report cardinality: "outnumbered two to one" reads the same at any
scale, while "three enemies within weapons range" does not, and nothing else in the
observation says it. Each scalar is a Gaussian kernel over toroidal distance summed over
every contributing ship and compressed with `log1p` — permutation invariant, self-excluded
on the ally channel, and masked by the same belief validity attention keys on, so a ship
never counts a neighbour it is not allowed to see.

The 500 px radius and the `log1p` compression are measured rather than assumed
([`presence_density_study.py`](../benchmarks/presence_density_study.py)). At 250 px the
5v5 enemy channel is dead; at 1000 px the 5v5 ally spread collapses because every ship
reads crowded. A bounded `s/(s+k)` compresses the crowded end into 0.02 of its range at
50 ships a side, where `log1p` keeps 0.38 — the difference between a count and a sense of
crowding, which is the semantics wanted.

The index gradient is an input only. Given the static field map it is a deterministic
function of position. Making it a target would also mean inventing a `label_scale`, since
those are `1/std` estimates and there is nothing here to estimate one from. A mis-set
scale can quietly dominate or vanish inside the aux loss, so it stays out until
measured.

Bullets have their own feature set on a separate axis, built by `build_bullet_coordinator`:

| Bullet feature | Network encoding |
|---|---|
| position x/y | four-frequency Fourier, **identical basis to ships** |
| velocity | direction scaled by symlog speed, as for ships |
| remaining lifetime | normalized scalar |
| local log index, local index gradient | normalized physical scalars |
| shooter team | two-way one-hot, never a per-ship index |
| active | key-padding mask, not a feature |

Both pipelines are derived from `ShipConfig` by
[`policy_io.build_policy`](../src/boost_and_broadside/train/rl/policy_io.py), the single
path that constructs a policy. Which encoders exist therefore follows from the config,
not from what a call site passes. The bullet encoder exists exactly when
`n_bullet_cross_per_block > 0`.

## Entity encoder

[`ShipEncoder`](../src/boost_and_broadside/models/yemong/encoder.py) concatenates the
encoded features and projects each entity independently into `d_model` with a two-layer
MLP and RMS normalization. Team and alive information remain part of the token; alive
masks are also passed to attention so dead entities cannot act as keys.

Setting `encoder_split` gives ships and fields a separate first projection over shared
plus own channels, followed by a *shared* second projection. Each feature declares a
`FeatureScope`, so a field token no longer spends most of its input width on ship-only
channels that are hard zeros for it. The shared output layer keeps both token types in one
latent space. Spatial layers apply a single `W_qkv` to ships and fields alike, and cannot
reconcile two spaces that have drifted apart.

[`BulletEncoder`](../src/boost_and_broadside/models/yemong/encoder.py) is separate and
deliberately narrow. It runs over `N·K` entities where the entity encoder runs over `N+M`,
so its width is what sets encoder cost. A bullet is also a much simpler entity to
describe.

The reference policy uses `d_model=128` and four attention heads.

## Spatial attention

Within each timestep, [`TransformerBlock`](../src/boost_and_broadside/models/yemong/attention.py)
applies pre-normalized multi-head self-attention and a gated MLP with residual connections.
Every live ship can therefore condition its action on every other live ship and field.

`n_spatial_heads` sets the head count here alone, separately from the pooling attention
in the value head. Head *width* is what bounds how much relative geometry a single
comparison can carry, and the critic's `TeamPMA` has no reason to follow a change made
for that reason. Two 64-wide heads and four 32-wide ones are the same weights read
differently — the parameter count is identical.

### Rotary position and attitude

With `spatial_rope` set, [`SpatialRotary`](../src/boost_and_broadside/models/yemong/rope.py)
rotates spatial Q/K by world x, world y, and entity attitude before the score is taken.
One rotated dimension pair then contributes

```text
|q| |k| cos(phi_q - phi_k + w (x_q - x_k))
```

so displacement enters the comparison directly instead of being something the trunk must
reconstruct from absolute-position features it first has to preserve through two
projections and a norm.

The frequencies are not a second scheme. They come from `base2_frequencies`, the same
function the encoder's `Fourier` transform calls, at the same periods: world width, world
height, and `2*pi`. Every frequency is an integer multiple of `2*pi / period`, so each is
exactly periodic over its own physical period — crossing the toroidal seam or turning
through a full circle returns the rotation to where it started, exactly rather than
approximately. The explicit Fourier features stay in the token; the rotation is additive
to them, using the same basis in a second place on purpose.

Each frequency costs one dimension pair. The Frontline world wants `2*(8 + 8 + 4) = 40`
of them, which does not fit a 32-wide head at all and leaves 24 unrotated dimensions in a
64-wide one — which is why the head-width change and the rotation arrive together. A
configuration needing more than the head provides raises rather than truncating: dropping
the coarsest frequency costs toroidal periodicity, and dropping the finest costs exactly
the short-range resolution the rotation exists to sharpen.

Cross-attention follows coordinate semantics. A bullet is a position on the same toroid,
so its key rotates on the same x/y basis as the ship query it meets. It has no heading, so
its attitude rotation is the identity — which is already what the encoder does with that
token's `ATT = (0, 0)`. Map objects carry the same zero attitude and behave the same way,
making those dimensions an absolute-heading preference toward map features rather than a
relative-heading comparison. Tables are built once per forward pass and shared by every
spatial sublayer.

### Relational attention bias

With `relational_bias` set, each spatial sublayer adds a shared pairwise term to its
attention scores, computed by one linear map from six scalars per ordered pair:
proximity, the ego-frame bearing cosine and sine both proximity-weighted and raw, and a
proximity-weighted range rate. See
[`relation.py`](../src/boost_and_broadside/models/yemong/relation.py).

It exists for the two things the rotation structurally cannot say. A base-2 ladder of
`cos(w dx)` is not a monotone sense of range. And the rotation's attitude block compares
*headings*, never the angle between a ship's nose and the direction to another ship —
which is the quantity the behaviour-cloning turn-head diagnostics identify as limiting.

One linear map per sublayer is `6*H` weights, twelve at two heads, shared by every pair.
Permutation equivariance is structural, and no parameter has a fleet-size dimension. The
bias is added into the same additive mask the key padding already uses, so a masked key
stays masked: the relational term is finite where the padding term is not.

Entity self-attention only. Bullets and K/V map memories are separate softmaxes over
tokens of a different kind, and one relation function would have to mean the same thing
for a ship pair and a ship/bullet pair.

### Attention kernel

The spatial layers pass an additive `(B, 1, 1, K)` alive-mask bias on every call, and
flash attention does not accept an additive mask — so this attention has always run on
PyTorch's memory-efficient backend, not flash. The relational bias does not change that;
it adds a `(B, H, N, N)` term to a tensor that already existed. Its measured cost is
materialization, not a lost kernel. Note that a compiled policy cannot be probed with
`sdpa_kernel(...)`: inductor lowers the attention itself and never consults it. See
[`frontline_inference_scaling.py`](../benchmarks/frontline_inference_scaling.py).

## Bullet cross-attention

Bullets are observed directly rather than inferred. Refractive fields make inference
impractical: a bullet curves under `grad(n)`, refracts, can totally internally reflect,
and travels at `500/n` locally, so dead-reckoning one
from the shooter's pose amounts to integrating an ODE inside the recurrent state.

They enter as **key/value-only** tokens: no query, no output projection, no FFN, no
recurrence. A bullet therefore costs `2·D²` per token against `16·D²` for a full entity,
cheap enough to attend over all of them instead of selecting a top-k. Nothing persists
between steps either, so a recycled ring-buffer slot cannot carry stale state.

Bullet position and velocity use the *same* encodings as ships. Attention computes relative
geometry as a bilinear form over Fourier features, and `q·k` reduces to a function of the
displacement only when both sides expand on one shared frequency basis; mismatched
frequencies leave cross terms that never form relative geometry at all. Shooter identity is
carried as a team one-hot and never as an index over ships, which would fix `N` in the
weights and break zero-shot transfer.

Softmax normalises, so this read conveys *which* bullets are relevant but not *how many*.
Threat intensity is still not available on the bullet axis. The ship axis now has the
transfer-safe form of that answer — the ally/enemy presence scalars above — and a
projectile equivalent would follow the same recipe: a smooth aggregate compressed so it
stays bounded, never a raw count, which would grow without bound as fleets scale.

## Temporal recurrence

After spatial mixing, [`GriffinTemporalBlock`](../src/boost_and_broadside/models/yemong/griffin.py)
updates each ship through a causal depthwise convolution and real-gated linear recurrent
unit, following [Griffin](https://arxiv.org/abs/2402.19427) (De et al., 2024), followed by
a gated MLP. Each ship carries its own temporal state, while attention supplies current
cross-entity context.

Only ships are recurrent. A field is static within an episode, so a recurrence over it
converges to a fixed point and carries nothing the encoder did not already supply, while
costing the expensive half of every block. Field tokens instead take
`forward_nonrecurrent`, which replaces the causal conv and RG-LRU with a per-sublayer
linear and keeps `norm1`, `linear1`, `linear2`, `linear_out`, `norm2`, and `gated_mlp`
shared with the ship path. Running both types through the same weights leaves the next
spatial layer's single `W_qkv` no divergence to undo. The substitute linear supplies the
one thing shared weights cannot: a type-specific linear map. It is initialised to the
identity, because `b1_out` feeds a multiplicative gate and zeroing it would erase the
branch entirely.

The recurrent state therefore covers ships only, `(n_yemong_blocks · n_temporal_per_block,
B·N, CONV_KERNEL·D)`, a third smaller than in the four-field profile before.

The implementation supports both execution patterns required by recurrent PPO:

- step-by-step rollout, where the recurrent state is updated once per environment step;
- full-sequence re-evaluation during PPO updates, where the same causal computation runs
  over the stored rollout.

Tests in [`tests/models/test_encoder.py`](../tests/models/test_encoder.py) pin recurrent
equivalence, attention masking, dtype behavior, and gradient checkpointing.

## Per-ship action head

The action head emits 12 logits for each ship and splits them into categorical power,
turn, and shoot distributions with sizes 3, 7, and 2. Actions and entropy remain factored;
the joint log probability is the sum of the three selected sub-action log probabilities.

The output shape is `(B, N, 3)` action indices.

## Decomposed value head

The critic produces one value per ship and active reward component. Most components use a
local token projection. Win/loss components use TeamPMA, which pools by multi-head
attention in the style of the [Set Transformer](https://arxiv.org/abs/1810.00825)
(Lee et al., 2019): learned seeds attend over the live ships of each team and feed a
dedicated outcome-value projection. That gives global outcome targets an explicitly
pooled team representation while retaining per-ship critic outputs.

Returns are normalized per component by the training system before value loss. Reward
semantics, aggregation, and horizons are documented in [training](training.md#reward-decomposition).

## Auxiliary next-state head

The next-state head predicts the coordinator's registered target channels for every ship:
position and attitude phase deltas, velocity deltas, resource phase deltas, absolute
angular velocity, and ship-local log-index delta. Static field material channels are
inputs, not prediction targets; the local index target makes entering and leaving a
medium visible to the learned dynamics model.

Training applies:

- normalized per-step mean-squared error across prediction channels; and
- a triangle-window cumulative loss for position and velocity, which penalizes systematic
  multi-step drift more strongly than zero-mean step noise.

With finite vision, visible ships refresh a policy-local point-estimate cache and the head's
forecast becomes the next hidden input recursively. Hidden tokens receive privileged
next-state supervision without exposing that truth to the actor or critic; death-to-respawn
teleport labels are masked. Each policy/perspective owns its cache, including frozen league
and evaluation policies.

Both fleets see the whole board for the opening tick of an episode, so no token is ever in
the never-observed state after deployment. That is what makes the supervision above cover
every enemy rather than only sighted ones, and it is why the trunk needs no key mask: token
validity is a constant, not something attention has to be told.

The label is the step from the *believed* current state to the true next one, not truth to
truth. The head's output is applied to the cache, so a truth-to-truth label would ask it to
reproduce a transition it is never in a position to apply -- the belief error would be
carried forward unchanged at every step, with nothing in the objective able to remove it.
Re-basing on the belief makes the target the correction back onto truth, which for a visible
ship is the same quantity as before and for a stale one is the shrinkage the point estimate
needs. The residual is only partly predictable, so the head learns the conditional mean of
that correction and the label distribution is correspondingly wider than a one-step delta.

The measured channel errors are shown in [evaluation](evaluation.md#auxiliary-dynamics-learning),
with deeper autoregressive diagnostics in the reference run's
[autoregressive report](../checkpoints/good-leaf-719/artifacts/figures/ar_report_4v4/) and
[noise calibration](../checkpoints/good-leaf-719/artifacts/figures/noise_calibration/).

## Why team size can change

No learned weight matrix has a ship-count dimension. Attention and recurrence operate
over the current token axis, and the heads apply to however many ship tokens are present.
That makes new team sizes executable without retraining; the
[crossover sweep](evaluation.md#zero-shot-crossover) tests whether the learned behavior
remains effective as the fleet grows.

The bullet path preserves this. Cross-attention is linear in bullet count, and shooter
identity is a team one-hot with no per-ship index anywhere. Softmax attention yields
proportions, which is the invariant that survives a change in fleet size: "outnumbered two
to one" means the same thing at any scale, while "five enemies nearby" does not.
