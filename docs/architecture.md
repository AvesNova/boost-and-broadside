# Policy architecture

`YemongPolicy` — *Yemong*, from the Korean 예몽, a dream that foretells the future — is a
centralized recurrent controller: it reads the full scene, exchanges information across
entities with spatial attention, carries per-entity memory through time, and emits a
factored action for every ship in the learned fleet. The name comes from its auxiliary
head, which learns to predict the next state of the world. The variable-cardinality
design is what makes zero-shot transfer across team sizes possible.

![YemongPolicy architecture](policy_architecture.png)

The diagram shows the reference-run configuration; component counts are set by
[`ModelConfig`](../src/boost_and_broadside/config/core.py), not fixed properties of the
architecture.

## Data flow

```text
global tensor state
    ↓
ship + refractive-field entity tokens
    ↓
FeatureCoordinator → encoder MLP
    ↓
[spatial Transformer → temporal Griffin/RG-LRU] × blocks
    ↓
ship tokens only
    ├── factored action distributions (per ship)
    ├── decomposed value estimates (per ship/component)
    └── next-state predictions (per ship)
```

For a batch `B`, `N` ships, `M` fields, and embedding width `D`, the shared trunk works
on `(B, N+M, D)` entity tokens. Fields participate in attention and retain temporal
state, but the three output heads slice the first `N` ship tokens.

## Observation and feature coordination

[`observation_from_state`](../src/boost_and_broadside/env/observation.py) exposes global
position, velocity, attitude, angular velocity, health, power, cooldown, team identity,
alive state, radius, previous action, and ship-local encoded log index. Fields are appended
as always-alive entity tokens with team ID 2, zero motion/action channels, and numeric
physical features: transition width, absolute inside and parent/outside log index, log
index ratio, and normalized interface damage. Field properties are unchanged by team
flipping.

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
| attitude | four-frequency Fourier features | phase delta |
| angular velocity | symlog scalar | next absolute value |
| health, power, cooldown | circular bounded encoding | phase delta |
| team identity | three-way one-hot | none |
| alive state | scalar | none |
| previous power/turn/shoot | categorical one-hot | none |
| radius | shared ship/field scalar divided by half the shorter world dimension | none |
| field width | normalized scalar | none |
| field inside/outside log index and ratio | normalized physical scalars | none |
| interface damage | normalized scalar | none |
| ship-local log index | `log(n)/(2 log(s))` | additive next-step delta |

Phase targets make wraparound natural: crossing the map boundary is a small rotation, not
a large coordinate jump. Feature dimensions and prediction layout are derived from the
registered features rather than hardcoded in model code.

## Entity encoder

[`ShipEncoder`](../src/boost_and_broadside/models/yemong/encoder.py) concatenates the
encoded features and projects each entity independently into `d_model` with a two-layer
MLP and RMS normalization. Team and alive information remain part of the token; alive
masks are also passed to attention so dead entities cannot act as keys.

The reference policy uses `d_model=128`, four attention heads, and two Yemong blocks, as
recorded in the [reference-run configuration](../checkpoints/resilient-resonance-682/wandb_export/config.json).

## Spatial attention

Within each timestep, [`TransformerBlock`](../src/boost_and_broadside/models/yemong/attention.py)
applies pre-normalized multi-head self-attention and a gated MLP with residual connections.
Every live ship can therefore condition its action on every other live ship and field.

## Temporal recurrence

After spatial mixing, [`GriffinTemporalBlock`](../src/boost_and_broadside/models/yemong/griffin.py)
updates each entity through a causal depthwise convolution and real-gated linear recurrent
unit, following [Griffin](https://arxiv.org/abs/2402.19427) (De et al., 2024), followed by
a gated MLP. Each entity carries its own temporal state, while attention supplies current
cross-entity context.

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
local token projection. Win/loss components instead use TeamPMA — pooling by multi-head
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

The measured channel errors are shown in [evaluation](evaluation.md#auxiliary-dynamics-learning),
with deeper autoregressive diagnostics under [`docs/ar_report/`](ar_report/) and noise
analysis under [`docs/noise_calibration/`](noise_calibration/).

## Why team size can change

No learned weight matrix has a ship-count dimension. Attention and recurrence operate
over the current token axis, and the heads apply to however many ship tokens are present.
That makes new team sizes executable without retraining; the
[crossover sweep](evaluation.md#zero-shot-crossover) tests whether the learned behavior
remains effective as the fleet grows.
