# Mamba-2D-Dogfight: Factorized World Model & Actor Specification

## 1. High-Level System Architecture

This architecture utilizes a **Space-Time Factorization**: `Mamba2` handles temporal mixing (time-axis), while `Relational Attention` handles spatial mixing (ship-axis) within a single time-step.

The system uses a **Y-Structure** flow:

1. **Backbone Update:** The World Model backbone (Mamba + Attention) processes the current state and previous actions to update history.
2. **Actor Pass:** Uses current **State** and **History** to predict the current **Action**.
3. **Evaluation Pass:** Predicts the **Value** (Team Return) and **Reward** for the current snapshot.

| Component | Specification |
| --- | --- |
| **Model Type** | Factorized Mamba-Transformer (Space-Time) |
| **Kernel Constraint** | **Strictly Optimized** (`mamba_ssm`). No Python-side state manipulation. |
| **Model Dimension ($d_{model}$)** | 256 (Configurable) |
| **Depth** | 6 Blocks (Each = 1 Mamba2 + 1 Relational Attention) |
| **Sequence Management** | Continuous Stream (Concatenated Episodes). **No Padding.** |
| **Sequence Packing** | Uses `seq_idx` tensor to trigger kernel-level state resets. |
| **Reset Semantics** | **Additive Learned Vector** at $S_0$ + **History Zeroing** at resets. |
| **Dead Logic** | **Replacement Embedding** for state + **Masked out** of Attention. |
| **Normalization** | **RMSNorm** (Root Mean Square Norm) used universally. |
| **Activation** | **SiLU** (Swish) used universally. |

---

## 2. Input Specifications

Inputs are factorized into **Intrinsic** (Self) and **Relational** (Edge). Absolute position is explicitly excluded to force relational invariance.

### A. Intrinsic State Tensor ($S_t$)

* **Shape:** `[Batch, Time, Ships, 9]`
* **Encoder:** Shared MLP.

**1. State Layout:**

| Index | Feature | Description |
| --- | --- | --- |
| 0 | Team | Team ID (0 or 1). |
| 1 | Health | Current Health (normalized). |
| 2 | Power | Engine Power setting. |
| 3, 4| Velocity | $V_x, V_y$ (normalized). |
| 5, 6| Attitude | $\cos(\theta), \sin(\theta)$ (Heading). |
| 7 | Shoot | Is shooting flag. |
| 8 | AngVel| Angular Velocity (normalized). |

**2. Embeddings & Special Tokens:**

* **Identity Embeddings:**
  * `ShipID_Embed`: (1-8) Identity.
  * `TeamID_Embed`: (0-1) Affiliation.
* **Replacement Logic (Dead Ships):**
  * If `HP <= 0`: The entire state vector is **replaced** by a learnable `[DEAD]` embedding vector.
* **Reset Semantic:**
  * If `seq_idx` marks a new episode: An additive `[RESET]` embedding is added to the state input.

### B. Relational Edge Tensor ($E_t$)

* **Shape:** `[Batch, Time, Ships, Ships, 64]`
* **Encoder:** Shared Physics Trunk + Layer-Specific Adapters.

**Features (Analytic Calculation):**

* **Base Features (18):** `d_pos` (2), `d_vel` (2), `dist` (1), `inv_dist` (1), `rel_speed` (1), `closing` (1), `dir` (2), `log_dist` (1), `tti` (1), `ATA/AA/HCA` (6).
* **Fourier Features (32):** 8 frequency bands on `d_pos` (sin/cos).
* **Total:** 50 active features, padded to 64.

### C. Action Tensor ($A_t$)

* **Shape:** `[Batch, Time, Ships, 3]`
* **Structure:** Multi-Discrete Enum embeddings.
* `Power`: 3 classes (Coast, Boost, Reverse).
* `Turn`: 7 classes (Straight, Turn L/R, Sharp L/R, AirBrake, Sharp AirBrake).
* `Shoot`: 2 classes (NoShoot, Shoot).

---

## 3. Detailed Layer Specifications

### A. Normalization & Encoders

* **Norm:** **RMSNorm** (`eps=1e-5`) is applied *before* every sub-layer (Pre-Norm architecture).
* **State Encoder:** 2-layer MLP (Dim -> 256 -> 256) with SiLU.
* **Physics Trunk:** 2-layer MLP (64 -> 128 -> 128) with SiLU. Layer Adapters project 128-dim trunk output to 256-dim attention bias.

### B. The Actor Pass (Space-Time Factorized)

The Actor pass determines the action $A_t$ for the current timestep.

1. **Stage 1: Spatial Self-Attention:**
   - Ships attend to each other using current state $S_t$ plus Relational Bias.
   - Purpose: Direct awareness of immediate surroundings.
2. **Stage 2: Temporal Global Cross-Attention:**
   - **Query:** Output of Stage 1 (current spatial snapshot).
   - **Key/Value:** History (Backbone outputs from $t-1$).
   - Purpose: Contextualizes current state with past trajectory.
3. **Actor Heads:** 2-layer MLP (256 -> 256 -> 12) outputs logits for all action components.

### C. The World Model Pass (Backbone)

The World Model backbone integrates history and choices.

1. **Action Injection:**
   - Previous action $prev\_action$ is embedded and fused with the encoded state $S_t$.
   - **Fusion:** `Linear(d_model + 128 -> d_model)`.
2. **Temporal Sub-Layer (Mamba2):**
   - Processes the fused state across time.
   - Uses `seq_idx` to trigger kernel-level state resets.
3. **Spatial Sub-Layer (Relational Attention):**
   - Multi-Head Attention (4 heads) modulated by relational bias from Physics Trunk.
   - `Key = Key + Bias`, `Value = Value + Bias`.
   - **Masking:** NaN-safe softmax ensures zero output for dead ships.

### D. The Evaluation Pass (Team Evaluator)

1. **Attention Pooling:** A learned "Team Token" queries all ship latents to produce a single Team Vector.
2. **Heads:**
   - **Value:** Predicts discounted future returns ($G_t$).
   - **Reward:** Predicts immediate frame reward ($R_t$).

---

## 4. Output Targets (Delta Predictions)

The World Model predicts **Residuals** (Deltas) applied to the current state $S_t$.

**Specific Targets:**
* `Health`, `Power`, `Velocity (X, Y)`, `Shoot`, `Angular Velocity`.
* *Note:* Attitude is typically integrated analytically from velocity/actions but the model predicts a delta for stability.

---

## 5. Training Protocol

### A. Loss Masking & Focal Loss
* **Masking:** Loss is dropped for transition frames and dead entities.
* **Focal Loss:** Used for action prediction to handle class imbalance (e.g., shooting being rare).

### B. Uncertainty Weighting
The model uses **Learnable Log Variance** ($\log \sigma^2$) for each loss component to automatically balance tasks:
$$L_{total} = \sum_{i} \left( \frac{1}{2\sigma_i^2} L_i + \frac{1}{2} \log \sigma_i^2 \right)$$
Components: State, Power, Turn, Shoot, Value, Reward.

---

## 6. Inference Modes

### Mode A: Pure Dreaming (Hallucination)
1. **Initialize:** $S_0$ from real world.
2. **Loop:**
   - **Actor:** Predict $A_t$ using $S_t$ and internal SSM state.
   - **World:** Predict $S_{t+1}$ using $S_t, A_t$.
   - **Update:** $S_{t+1} \to S_{next}$, update SSM hidden state.

### Mode B: Physics-in-the-Loop
1. **Observation:** Engine provides $S_t$.
2. **Actor:** Predict $A_t$.
3. **Execution:** $A_t$ sent to Game Engine.
4. **Mamba Update:** World model runs in "blind" mode to maintain hidden state $h_t$ for the next prediction.