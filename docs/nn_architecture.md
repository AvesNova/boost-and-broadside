# MVPPolicy — Network Architecture Reference

A complete, diagram-oriented description of the policy network used in Boost and Broadside.
Every shape, layer, activation, mask and parameter count below is read directly from the
implementation (`src/boost_and_broadside/models/mvp/`, `train/rl/features.py`,
`train/rl/ppo.py`) and the shipped run profile (`runs/shared.py`, `runs/rl.py`).

---

## 1. One-paragraph summary

`MVPPolicy` is a recurrent, entity-centric actor–critic. Every ship and every obstacle in an
environment is encoded as one token. A shared trunk of **Yemong blocks** alternates *spatial*
processing (multi-head self-attention across the entities of a single timestep) with *temporal*
processing (a Griffin RG-LRU linear recurrence run independently per entity across time). The
trunk is fully shared: one network produces the behaviour of every ship. Obstacle tokens
participate in attention and carry recurrent state but are sliced away before the output heads.
Four heads read the ship tokens: a factored categorical **action head**, an auxiliary
**next-state prediction head** (self-supervised dynamics signal), a **per-component value head**
(one scalar per reward component), and a **team-pooled value path** (pooling-by-multi-head-
attention over each team's alive ships) that supplies the win/loss value components with global
team context.

---

## 2. Symbols and concrete values

| Symbol | Meaning | Value in the shipped RL profile |
|---|---|---|
| `B` | parallel environments | 5856 per rollout (≈6 M tokens per update ÷ `N+M` ÷ `T`, rounded to a multiple of the 32 minibatches → 183 envs per minibatch) |
| `T` | rollout length (timesteps per PPO update segment) | 128 |
| `N` | ships per environment (both teams combined) | 8 |
| `M` | obstacles per environment | 0 in `runs/rl.py`, 8 in `runs/rl_obstacles.py` |
| `N+M` | entity tokens per environment per timestep | 8 (combat) / 16 (obstacle run) |
| `D` = `d_model` | token embedding width | **128** |
| `H` = `n_heads` | attention heads | **4** (head dim 32) |
| `L` = `n_transformer_blocks` | number of Yemong blocks | **2** |
| `F` | encoder input width (concatenated encoded features) | **58** |
| `A` | action logits | **12** = 3 power + 7 turn + 2 shoot |
| `P` | next-state prediction width | **9** |
| `Ptgt` | next-state *target* width (pre-prediction space) | 15 |
| `K` | active reward components = value head outputs | **11** |
| `K_win` | components routed through TeamPMA | **2** (`ally_win`, `enemy_win`) |
| `CONV_KERNEL` | causal depthwise conv kernel | **4** |

Total trainable parameters: **1,340,578 (≈ 1.34 M)**.

---

## 3. Top-level dataflow

```
                    MVPObservation  (per entity: pos, vel, att, ang_vel,
                                     health, power, cooldown, team_id,
                                     alive, radius, previous_action)
                             │
                    ┌────────▼─────────┐
                    │ FeatureCoordinator│   14 features → encode → concat
                    │   (non-learned)   │
                    └────────┬─────────┘
                             │  (B, N+M, 58)
                    ┌────────▼─────────┐
                    │   ShipEncoder    │   2-layer MLP, shared by ships & obstacles
                    └────────┬─────────┘
                             │  (B, N+M, 128)   ← "entity tokens"
              ┌──────────────▼──────────────┐
              │      YemongBlock  × 2       │  ← recurrent state in/out
              │  ┌───────────────────────┐  │
              │  │ TransformerBlock      │  │  spatial: entities ↔ entities,
              │  │   (MHSA + GatedMLP)   │  │  within one timestep
              │  ├───────────────────────┤  │
              │  │ GriffinTemporalBlock  │  │  temporal: per entity, across time
              │  │   (conv + RG-LRU +    │  │
              │  │    GatedMLP)          │  │
              │  └───────────────────────┘  │
              └──────────────┬──────────────┘
                             │  (B, N+M, 128)
                     slice [:, :N, :]         ← obstacles dropped here
                             │  (B, N, 128)  = x_ships
        ┌────────────────────┼────────────────────┬────────────────────┐
        │                    │                    │                    │
 ┌──────▼──────┐    ┌────────▼────────┐   ┌───────▼───────┐   ┌────────▼────────┐
 │ ActionHead  │    │ NextStateHead   │   │ ValueHead     │   │    TeamPMA      │
 │             │    │   (auxiliary)   │   │   (local)     │   │  (per-team pool)│
 └──────┬──────┘    └────────┬────────┘   └───────┬───────┘   └────────┬────────┘
        │ (B,N,12)           │ (B,N,9)            │ (B,N,11)           │ (B,N,128)
        │                    │                    │            ┌───────▼────────┐
        │                    │                    │            │ ValueHead(win) │
        │                    │                    │            └───────┬────────┘
        │                    │                    │                    │ (B,N,2)
        │                    │                    └────── merge ───────┘
        │                    │                             │  indices 0,1 of K
        │                    │                             │  come from the win path
   factored                 aux                        (B, N, 11)
   Categorical            MSE loss                  per-component value
   (power|turn|shoot)                               (normalized space)
```

---

## 4. Stage 1 — Observation → feature vector (`FeatureCoordinator`, not learned)

This stage is a fixed, hand-designed encoding, not a learned layer. It runs identically for
every entity token. Each `Feature` = (accessor → input encoder), and predicted features
additionally carry (target encoder → predictor) used only for the auxiliary loss.

| # | Feature | Raw channels | Input encoding | Enc. dims | Predicted? | Prediction type | Pred. dims | `label_scale` |
|---|---|---|---|---|---|---|---|---|
| 1 | `position_x` | pos[0] | `Fourier(4 freqs, period = world_w = 1024)` | 8 | ✓ | phase delta (toroidal-exact) | 1 | 177.4 |
| 2 | `position_y` | pos[1] | `Fourier(4, period = world_h = 1024)` | 8 | ✓ | phase delta | 1 | 177.4 |
| 3 | `velocity` | vel[0:2] | `SymlogVelocity` = direction · symlog(‖v‖) | 2 | ✓ | additive delta | 2 | (20.0, 20.0) |
| 4 | `attitude` | att[0:2] = (cos θ, sin θ) | `Fourier(4, period = 2π)` on both channels | 16 | ✓ | phase delta (rotation) | 1 | 1.5 |
| 5 | `angular_velocity` | ang_vel | `Symlog` | 1 | ✓ | absolute | 1 | 0.447 |
| 6 | `health` | health | `UnitCircle(max_health=100)` → quarter-wave (sin, cos) | 2 | ✓ | phase delta | 1 | 36.0 |
| 7 | `power` | power | `UnitCircle(max_power=100)` | 2 | ✓ | phase delta | 1 | 93.0 |
| 8 | `cooldown` | cooldown | `UnitCircle(firing_cooldown=0.1)` | 2 | ✓ | phase delta | 1 | 2.1 |
| 9 | `team_id` | team_id | `OneHot(3)` — team 0 / team 1 / obstacle | 3 | — | — | — | — |
| 10 | `alive` | alive | identity | 1 | — | — | — | — |
| 11 | `prev_power` | previous_action[0] | `OneHot(3)` | 3 | — | — | — | — |
| 12 | `prev_turn` | previous_action[1] | `OneHot(7)` | 7 | — | — | — | — |
| 13 | `prev_shoot` | previous_action[2] | `OneHot(2)` | 2 | — | — | — | — |
| 14 | `radius` | radius | `Normalize(40.0)` | 1 | — | — | — | — |
| | | | **concat → encoder input** | **58** | | **→ NextStateHead output** | **9** | |

Notes worth showing in a figure:

- `Fourier(n, period)` emits `[sin(2π·2^k·x/period), cos(2π·2^k·x/period)]` for `k = 0…n−1`,
  i.e. `2n` dims per scalar channel — an octave-spaced positional encoding whose lowest
  frequency wraps exactly with the toroidal world.
- Positions are absolute world coordinates, **not** ego-relative. There is no positional
  encoding on the attention: the model is permutation-equivariant over entity slots, and all
  spatial information enters through these features.
- Obstacles use the same 58-dim layout: `team_id = 2`, `alive = True` always, `health` set to
  max, `power`/`cooldown`/`ang_vel` zero, `attitude` = velocity direction, `radius` = the
  obstacle's actual radius (ships use the constant collision radius 10).
- `symlog(x) = sign(x)·log(1+|x|)`.

---

## 5. Stage 2 — `ShipEncoder` (learned, shared across all entities)

Applied independently to every token; the leading `(…, N+M)` dims are treated as batch.

```
(…, 58)
  → Linear(58 → 256)          [bias]
  → RMSNorm(256)
  → GELU
  → Linear(256 → 128)         [bias]
  → RMSNorm(128)
(…, 128)
```

Parameters: **48,384**. One encoder instance; ships and obstacles share weights (distinguished
only by their feature content, chiefly the `team_id` one-hot and `radius`).

---

## 6. Stage 3 — `YemongBlock` × 2 (the trunk)

Each block is `TransformerBlock` (spatial) followed by `GriffinTemporalBlock` (temporal).
Both are pre-norm with residual connections. Blocks are stacked; each block owns an independent
slice of the recurrent state.

### 6a. `TransformerBlock` — spatial (entities ↔ entities, within one timestep)

```
x  ─┬────────────────────────────────────────────────► (+) ─┬──────────────────────────► (+) ──► out
    │                                                   ▲    │                             ▲
    └─ RMSNorm ─► MHSA(4 heads, head_dim 32) ─► out_proj┘    └─ RMSNorm ─► GatedMLP ───────┘
                       ▲
                  alive mask (keys)
```

- `qkv`: `Linear(128 → 384, bias=False)`, split into Q, K, V; reshaped to `(B, 4, N+M, 32)`.
- Attention: `F.scaled_dot_product_attention`, no dropout, no causal mask — **full attention
  over the `N+M` entities of one timestep**.
- **Alive masking:** an additive bias `(1 − alive) · (−large)` of shape `(B, 1, 1, N+M)` is
  applied to the *key* axis. Dead ships cannot emit information; they still form queries
  (their outputs are simply masked out of the losses). Obstacles are always alive.
- `out_proj`: `Linear(128 → 128, bias=False)`.
- `GatedMLP` (SwiGLU-shaped, GELU gate, 4× expansion, all bias-free):
  `down_proj( GELU(gate_proj(x)) ⊙ up_proj(x) )` with `gate_proj, up_proj: 128 → 512`,
  `down_proj: 512 → 128`.

Parameters per block: **262,400**.

### 6b. `GriffinTemporalBlock` — temporal (per entity, across time)

Runs with the entity axis folded into the batch: batch = `B·(N+M)`, sequence = `T`.
There is **no** mixing between entities here.

```
x_seq ─┬──────────────────────────────────────────────────────► (+) ─┬────────────────────► (+) ──► out
       │                                                          ▲   │                       ▲
       └─ RMSNorm ─┬─ Linear1(128→128) ─► CausalDepthwiseConv1d ──┐   └─ RMSNorm ─► GatedMLP ─┘
                   │        (kernel 4, groups=128, bias)          │
                   │                                              ▼
                   │                                          RG-LRU  ──┐
                   │                                                    ⊙ ──► Linear_out(128→128)
                   └─ Linear2(128→128) ─► GELU ─────────────────────────┘
```

- `Linear1`, `Linear2`, `Linear_out`: `128 → 128`, `bias=False`.
- **Causal depthwise conv:** `Conv1d(128, 128, kernel_size=4, groups=128, bias=True)`, i.e.
  one length-4 filter per channel, no cross-channel mixing. Causality is enforced by
  *prepending a stored buffer* of the last `CONV_KERNEL−1 = 3` `Linear1` outputs instead of
  zero-padding, so a single-step rollout (`T=1`) and a full-sequence PPO re-evaluation see
  bit-identical causal context.
- **RG-LRU** (Real-Gated Linear Recurrent Unit, Griffin), element-wise over the 128 channels:

  ```
  r_t = σ(W_a x_t + b_a)                      recurrence gate     (Linear 128→128, bias)
  i_t = σ(W_x x_t + b_x)                      input gate          (Linear 128→128, bias)
  a_t = σ(Λ)^(c · r_t),           c = 8       per-channel decay
  h_t = a_t ⊙ h_{t−1} + √((1−a_t)(1+a_t)) ⊙ (i_t ⊙ x_t)
  ```

  `Λ` is a learnable 128-vector initialised to `linspace(0, 4)`, so the decay rates `σ(Λ)`
  start spread from ≈0.50 (fast-forgetting channels) to ≈0.982 (long-memory channels) — a
  multi-timescale memory bank. The normaliser is written as `(1−a)(1+a)` (clamped at 1e-6) to
  stay numerically safe in bfloat16 as `a → 1`. The output of the RG-LRU *is* `h_t`.
- The two branches multiply element-wise (`RG-LRU output ⊙ GELU(Linear2(·))`) before
  `Linear_out`. First residual adds this to the block input; a second pre-norm `GatedMLP`
  (identical shape to the spatial one: 128→512→128, bias-free) with its own residual closes
  the block.
- **Episode boundaries:** at update time a `done_mask` zeroes `a_{t+1}` at the step after a
  terminal, so recurrence never leaks across episodes. At rollout time the equivalent is done
  externally by `reset_hidden_for_envs`, which zeroes every token of a finished environment.

Parameters per block: **279,808**.

### 6c. Recurrent state layout

One tensor carries everything the trunk needs between steps:

```
hidden : (L, B·(N+M), CONV_KERNEL · D) = (2, B·(N+M), 512)   float32

   hidden[l, :,   0:128]  → RG-LRU state h for layer l
   hidden[l, :, 128:512]  → causal-conv buffer: last 3 Linear1 outputs, flattened (3 × 128)
```

Every entity token — ships **and** obstacles — carries its own recurrent state.

---

## 7. Stage 4 — token slicing

```
x        : (B, N+M, 128)     trunk output
x_ships  = x[:, :N, :]       (B, N, 128)      ← everything downstream uses only this
```

Token order is fixed: indices `0 … N−1` are ships, indices `N … N+M−1` are obstacles.
Within the ship range the two teams are assigned to slots by an independent random permutation
per environment on every reset, so slot index carries no team information — the only team signal
the network sees is the `team_id` one-hot inside each token. Obstacles receive **no** action,
value, or prediction head.

---

## 8. Stage 5 — Output heads

All four heads share the same 2-layer MLP shape: `Linear(128 → 256) → RMSNorm(256) → GELU →
Linear(256 → out)`, both Linears with bias. Initialisation is standard PPO orthogonal: first
Linear gain `√2`, last Linear gain `0.01`, biases zero.

### 8a. ActionHead → `(B, N, 12)`

Output is **three concatenated categorical logit blocks**, not one 12-way softmax:

| Slice | Sub-action | Size | Values |
|---|---|---|---|
| `[0:3]` | power | 3 | COAST, BOOST, REVERSE |
| `[3:10]` | turn | 7 | GO_STRAIGHT, TURN_LEFT, TURN_RIGHT, SHARP_LEFT, SHARP_RIGHT, AIR_BRAKE, SHARP_AIR_BRAKE |
| `[10:12]` | shoot | 2 | NO_SHOOT, SHOOT |

Each block is an independent `Categorical`; the sampled action is the triple
`(power, turn, shoot)` and the joint log-prob / entropy are the **sums** over the three
sub-distributions. Effective joint action space: 3 × 7 × 2 = 42, factored into 12 logits.

### 8b. NextStateHead → `(B, N, 9)` — auxiliary, training-only

`Linear(128 → 256) → RMSNorm → GELU → Linear(256 → 9)`. Predicts the next-step change of the
ship's own physical state, in a scaled space (targets are multiplied by `label_scale` so all
nine outputs are O(1)). Output layout, in order:

```
[0] position_x phase delta   [1] position_y phase delta
[2] velocity Δ(vx_norm)      [3] velocity Δ(vy_norm)
[4] attitude phase delta     [5] angular_velocity (absolute)
[6] health phase delta       [7] power phase delta      [8] cooldown phase delta
```

Phase predictions are *applied as rotations* of the corresponding `(sin, cos)` target pair, so
the reconstructed state stays on the unit circle by construction and toroidal position wrap is
exact. This head is a dense dynamics-learning signal (a world-model-style auxiliary task); it
does not feed back into the policy at inference and is not used to act.

### 8c. ValueHead — local path → `(B, N, 11)`

`Linear(128 → 256) → RMSNorm → GELU → Linear(256 → K=11)`. One scalar per active reward
component, read straight from the ship's own token. Outputs live in a *normalized* space
maintained by `ReturnScaler` (per-component EMA of the 5th/95th percentiles of returns in
symlog-reward space, mapping returns to roughly `[−1, 1]`), so a single MSE with uniform
weighting is well-conditioned across components whose natural scales differ by orders of
magnitude.

The 11 active components, in value-head output order:

| k | Component | Aggregation | γ | λ |
|---|---|---|---|---|
| 0 | `ally_win` | team (TeamPMA path) | 0.999 | 0.97 |
| 1 | `enemy_win` | team, λ = −1 for enemies (TeamPMA path) | 0.999 | 0.97 |
| 2 | `facing` | local (self-only) | 0.975 | 0.80 |
| 3 | `closing_speed` | local | 0.975 | 0.80 |
| 4 | `shoot_quality` | local | 0.975 | 0.80 |
| 5 | `kill_shot` | local | 0.995 | 0.87 |
| 6 | `kill_assist` | local | 0.995 | 0.97 |
| 7 | `damage_taken` | local | 0.991 | 0.90 |
| 8 | `damage_dealt_enemy` | local | 0.991 | 0.90 |
| 9 | `damage_dealt_ally` | local | 0.991 | 0.90 |
| 10 | `death` | local | 0.995 | 0.95 |

(The registry in `rewards.py` has 21 components; the 10 with weight 0 in the shipped profile —
the `ally_*`/`enemy_*` damage and death pairs, the four obstacle penalties, `shooting_penalty`,
`speed` — are dropped before the model is built, so `K` is 11 here. `K` is profile-dependent.)

### 8d. TeamPMA + win/loss ValueHead → overrides `k = 0, 1`

Pooling by Multi-head Attention, giving the win/loss critic global team context that a single
ship's token cannot carry.

```
for each team t ∈ {0, 1}:
    query  = seeds[t]                     learned (2, 128) parameter, init N(0, 0.02²)
    keys   = values = x_ships              (B, N, 128)
    mask   = ¬((team_id == t) & alive)     key_padding_mask — other team + dead ships ignored
    out_t  = MultiheadAttention(d=128, heads=4, bias=False)(query, keys, values)   → (B, 128)
    out_t  = nan_to_num(out_t)             guard: a fully-dead team yields all-masked → NaN
    out_t  = out_t · 1[team t has ≥1 alive ship]
    pool_t = RMSNorm(out_t)
```

The two pooled team embeddings are then **gathered back per ship** (`team_pool[team_id(ship)]`,
team id clamped to 0/1), restoring a `(B, N, 128)` tensor, and passed through a dedicated
`ValueHead(128 → 256 → K_win=2)`. Its two outputs replace the local head's outputs at indices
0 and 1. All other components keep the local per-ship estimate. Both paths receive gradients
(at update time the merge is a `cat` of slices, not an in-place write).

TeamPMA is only instantiated when `team_pma_k` is non-empty; the attention `in_proj` is
orthogonal-initialised with gain `√2`, `out_proj` with gain 1.0.

---

## 9. Parameter budget

| Module | Params | Share |
|---|---:|---:|
| `ShipEncoder` | 48,384 | 3.6 % |
| `YemongBlock × 2` — spatial (`TransformerBlock`) | 524,800 | 39.1 % |
| `YemongBlock × 2` — temporal (`GriffinTemporalBlock`) | 559,616 | 41.7 % |
| `TeamPMA` | 65,920 | 4.9 % |
| `ActionHead` | 36,364 | 2.7 % |
| `ValueHead` (local, K=11) | 36,107 | 2.7 % |
| `ValueHead` (win, K_win=2) | 33,794 | 2.5 % |
| `NextStateHead` | 35,593 | 2.7 % |
| **Total** | **1,340,578** | 100 % |

Per-Yemong-block breakdown (×2 in the shipped config):

| Sub-module | Params |
|---|---:|
| spatial `qkv` (128→384) | 49,152 |
| spatial `out_proj` (128→128) | 16,384 |
| spatial `GatedMLP` (3 × 128↔512) | 196,608 |
| spatial RMSNorms (×2) | 256 |
| temporal `Linear1`/`Linear2`/`Linear_out` | 49,152 |
| temporal depthwise conv (128×4 + bias) | 640 |
| temporal RG-LRU (`Λ` + two gated Linears) | 33,152 |
| temporal `GatedMLP` | 196,608 |
| temporal RMSNorms (×2) | 256 |
| **block total** | **542,208** |

---

## 10. Two execution modes (same weights, same math)

The figure benefits from showing these as two "reading modes" of the same trunk.

**A. Rollout / inference — `get_action_and_value`, `T = 1`, `torch.no_grad`**

```
obs (B, N+M, …) + hidden (2, B·(N+M), 512)
  → encoder → for each Yemong layer: layer.step(...)
  → spatial attention over (B, N+M, D)
  → temporal recurrence advanced by exactly one step (conv buffer supplies the
    previous 3 inputs, RG-LRU advanced once)
  → slice ships → sample action, read value + pred_next
  → new hidden
```

**B. PPO update / re-evaluation — `evaluate_actions`, full `T = 128`**

```
obs (T, B, N+M, …) + initial_hidden (2, B·(N+M), 512)
  → encoder over all T·B·(N+M) tokens in parallel
  → per Yemong layer:
        spatial: fold T into batch → attention over (T·B, N+M, D), all timesteps at once
        temporal: fold B·(N+M) into batch → RG-LRU over T via a Hillis–Steele
                  parallel scan (log₂T rounds of the associative operator
                  (a₂,b₂)∘(a₁,b₁) = (a₂a₁, a₂b₁+b₂)), plus done-mask resets
  → slice ships → logprob, entropy, values, logits, pred_next
```

The parallel scan is mathematically identical to the sequential recurrence; it exists only to
make the `T = 128` re-evaluation fast on GPU. The stored conv buffer is what makes mode A and
mode B agree exactly.

**Ego-pass detail (training only, worth a footnote rather than a box):** the shipped profile
runs `paradigm="ego_pass"`, i.e. each rollout step performs a single batched forward over `2B`
environments — the raw observation and a team-flipped copy — so that every ship acts from a
perspective where its own team is labelled 0. Only the raw-perspective half contributes
log-probs, values and predictions to the loss; opponents always play team 1. Each perspective
carries its own hidden state.

---

## 11. What trains what (loss → head arrows)

Total loss, per PPO minibatch, all terms masked to alive ships:

```
L = pg_coef · L_PPO-clip
  + vf_coef · L_value
  + ent_coef · L_entropy
  + bc_coef  · L_BC
  + ns_coef  · L_next_state
  + win_coef · L_windowed
  ( + sigreg_coef · L_SIGReg  — 0.0 in every shipped profile )
```

| Loss | Reads | Detail |
|---|---|---|
| `L_PPO-clip` | ActionHead logits | Clipped surrogate, `clip_coef = 0.15`, advantages normalised by an RMS estimate; advantages come from per-component GAE combined through a **lambda aggregation matrix** (teammates/enemies see each other's global components; local components use a diagonal, self-only lambda). |
| `L_entropy` | ActionHead logits | Sum of the three sub-action entropies; coefficient 0.005. |
| `L_BC` | ActionHead logits | Cross-entropy against the stochastic scripted agent's action distribution, per sub-action block. Schedule-gated: decays to zero as the policy's win rate against the scripted agent reaches `bc_winrate_target = 0.45`. Logged alongside the scripted agent's own entropy so `CE − H` reads as a KL. |
| `L_value` | ValueHead (local + win paths) | MSE against `ReturnScaler`-normalized per-component returns, averaged over alive ships and over `K`. |
| `L_next_state` | NextStateHead | Per-step weighted MSE against `label_scale`-scaled labels, masked to alive & non-terminal steps. `next_state_coef = 0.2`. |
| `L_windowed` | NextStateHead | Triangle-kernel (window 32) convolution of the per-step errors of `position_x`, `position_y`, `velocity` along time, squared. Amplifies systematic drift (∝ window²) relative to per-step noise (∝ window) — catches bias that teacher-forced per-step MSE cannot see. `windowed_loss_coef = 0.1` (default; not overridden in `runs/rl.py`). |
| `L_SIGReg` | encoder output `z` | Optional embedding regulariser on the raw encoder output, 64 random projections. Disabled (`sigreg_coef = 0.0`) in every profile — safe to omit from the figure or show greyed. |

Training runs under `torch.autocast(bfloat16)`; gradients are clipped at global norm 1.0 with
Adam (`eps = 1e-5`), 4 epochs per update, 32 minibatches, each split into 5 gradient-
accumulation micro-batches.

---

## 12. Notes for the figure

Things that are architecturally load-bearing and worth making visually explicit:

1. **Two axes of mixing.** Spatial = across entities within a timestep (attention). Temporal =
   across time within an entity (RG-LRU). They never mix in the same operator; the Yemong block
   is exactly the alternation of the two. Consider drawing the trunk as a 2-D grid of tokens
   (entities × time) with horizontal arrows for attention and vertical arrows for recurrence.
2. **One shared network, many ships.** There is no per-agent network. The batch axis is
   `environments × entities`; ship identity exists only in the features.
3. **Obstacles enter and leave.** They are encoded, attended to, and carry recurrent state, but
   are sliced off before all heads. A dashed token colour that stops at the slice line reads well.
4. **Two value paths.** The local path is per-ship; the win/loss path detours through TeamPMA
   and rejoins as components 0 and 1. This is the only place the network aggregates across ships
   *after* the trunk.
5. **Masking legend.** Three distinct masks appear: (a) `alive` as an attention key mask in the
   spatial block, (b) `alive & team` as the TeamPMA key-padding mask, (c) `alive & non-terminal`
   as the loss mask. Worth a small legend rather than three separate annotations.
6. **Auxiliary vs. control.** ActionHead is the only head used at inference. Value and
   NextState heads are training-only; consider a lighter weight or dashed border for those two
   plus TeamPMA/win-value.
7. **Recurrent state is a single packed tensor** — RG-LRU state and conv buffer side by side,
   `(2, B·(N+M), 512)`. Showing the packing explicitly explains why rollout and update agree.
8. **Suggested block colours:** non-learned preprocessing (FeatureCoordinator) in a neutral
   grey; shared trunk (encoder + Yemong) in the primary colour; heads in a secondary colour;
   training-only paths desaturated. Shape annotations on every arrow — the shapes are the most
   information-dense part of this architecture.
