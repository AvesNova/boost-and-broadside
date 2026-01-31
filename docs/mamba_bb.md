# Mamba-2D-Dogfight: Factorized World Model & Actor Specification

## 1. High-Level System Architecture

This architecture utilizes a **Space-Time Factorization**: `Mamba2` handles temporal mixing (time-axis), while `Relational Attention` handles spatial mixing (ship-axis) within a single time-step.

The system uses a **Y-Structure** flow:

1. **Actor Pass:** Uses  +  to predict .
2. **World Model Pass:** Uses  +  (Fused) to predict .

| Component | Specification |
| --- | --- |
| **Model Type** | Factorized Mamba-Transformer (Space-Time) |
| **Kernel Constraint** | **Strictly Optimized** (`mamba_ssm`). No Python-side state manipulation. |
| **Total Parameters** | ~3.5 Million |
| **Model Dimension ()** | 256 |
| **Depth** | 6 Blocks (Each = 1 Mamba2 + 1 Relational Attention) |
| **Sequence Management** | Continuous Stream (Concatenated Episodes). **No Padding.** |
| **Sequence Packing** | Uses `seq_idx` tensor to trigger kernel-level state resets. |
| **Reset Semantics** | **Additive Learned Vector** at  (Semantics for Attention). |
| **Dead Logic** | **Replacement Embedding** for state + **Masked out** of Attention. |
| **Normalization** | **RMSNorm** (Root Mean Square Norm) used universally. |
| **Activation** | **SiLU** (Swish) used universally. |

---

## 2. Input Specifications

Inputs are factorized into **Intrinsic** (Self) and **Relational** (Edge). Absolute position is explicitly excluded to force relational invariance.

### A. Intrinsic State Tensor ()

* **Shape:** `[Batch, Time, Ships, Features]`
* **Encoder:** Shared MLP.

**1. Continuous Features (Float):**

* `Velocity` () — *3 dims*
* `Acceleration` () — *2 dims*
* `Attitude` () — *2 dims*
* `Speed`, `Health`, `Power` — *3 dims*
* `Is_Shooting` — *1 dim*

**2. Embeddings & Special Tokens:**

* **Additive Embeddings:**
* `+ ShipID_Embed`: (1-8) Identity.
* `+ TeamID_Embed`: (Red/Blue) Affiliation.


* **Replacement Logic (Dead Ships):**
* If `HP <= 0`: The entire continuous feature vector is **replaced** by a learnable `[DEAD]` embedding vector.



### B. Relational Edge Tensor ()

* **Shape:** `[Batch, Time, Ships, Ships, Edge_Features]`
* **Encoder:** Shared Physics Trunk + Layer-Specific Adapters.

**Features (Analytic Calculation):**

* `Fourier(d_pos)`: Log-linear bands on Toroidal Shortest Path.
* `d_vel`: Relative Velocity vector.
* `Sin/Cos Encodings`: Angle to Target (ATA), Aspect Angle (AA), Heading Crossing Angle (HCA).
* `Scalars`: Log(Distance), Closing Speed, Time to Intercept (TTI).

### C. Action Tensor ()

* **Shape:** `[Batch, Time, Ships, Action_Dim]`
* **Structure:** Multi-Discrete Enum embeddings.
* `Power`: 3 classes.
* `Turn`: 7 classes.
* `Shoot`: 2 classes.


* **Null Action:** The "Null/No-Op" action is used as the initial dummy action for  or padding.

---

## 3. Detailed Layer Specifications

### A. Normalization & Activation Policy

* **Norm:** **RMSNorm** (`eps=1e-5`) is applied *before* every block (Pre-Norm architecture).
* **Activation:** **SiLU** is used in all MLPs and Encoders.

### B. Encoders (The Senses)

#### 1. State Encoder (Shared)

Used by both World Model and Actor.

* **Input:** Intrinsic State Features + Embeddings (~64 dim).
* **Architecture:**
* `Linear(In -> 256)`
* `RMSNorm(256)`
* `SiLU`
* `Linear(256 -> 256)`


* **Output:** `[Batch, Ships, 256]` ()

#### 2. Relational Encoder (Physics Trunk)

A shared "Physics Engine" that projects raw geometry into latent space, reused across all layers.

* **Shared Trunk:**
* **Input:** Analytic Edge Features (~64 dim).
* **Layers:** `Linear(In -> 128)`  `RMSNorm`  `SiLU`  `Linear(128 -> 128)`.
* **Output:** `Base_Edge_Embed` [128].


* **Layer Adapters:**
* **Structure:** 7 Separate Linear Layers (6 for World Model, 1 for Actor).
* **Layer :** `Linear(128 -> 256)` (No activation).
* **Purpose:** Projects the physics trunk output into the specific "Key/Value Injection" space for that specific depth.



---

### C. The Actor Pass (Phase 1)

This phase runs **first** at each timestep . It sees the current state and past history, but **not** the current action.

#### 1. Actor Inputs

* `State_t` (from State Encoder).
* `History_t-1` (from Backbone Output of previous step).
* `Edges_t` (from Physics Trunk + Actor Adapter).

#### 2. Actor Fusion Block

* `Concat(State, History)`  `Linear(512 -> 256)`  `RMSNorm`  `SiLU`.

#### 3. Tactical Layer (Relational Self-Attention)

* **Type:** 1 Layer of Self-Attention (4 Heads).
* **Injection:** Uses `Edges_t` to modulate Keys/Values.
* **Purpose:** Gives the ship awareness of its neighbors' current positions/velocities *before* deciding an action.

#### 4. Action Network (Heads)

* **Trunk:** `Linear(256 -> 256)`  `RMSNorm`  `SiLU`.
* **Heads:**
* `Head_Power`: `Linear(256 -> 3)`
* `Head_Turn`: `Linear(256 -> 7)`
* `Head_Shoot`: `Linear(256 -> 2)`



---

### D. The World Model Pass (Phase 2)

This phase runs **second**. It fuses the State with the Action (determined in Phase 1) to predict the future.

#### 1. Action Injection (The "Y" Junction)

We combine the state with the chosen action.

* **Source Selection:**
* **Training:** Use Ground Truth Actions (Teacher Forcing).
* **Inference:** Use `Action_t` predicted by the Actor Head in Phase 1.


* **Embedding:**
* `Emb_Power(3->32)`, `Emb_Turn(7->64)`, `Emb_Shoot(2->32)`.
* Concat Embeddings  `Action_Vector` (128 dim).


* **Fusion:**
* `Concat(State_t, Action_Vector)`  `Linear((256+128) -> 256)`  `RMSNorm`  `SiLU`.


* **Output:** `Fused_Input_t` (256 dim).

#### 2. The Backbone (6 Blocks)

The `Fused_Input_t` enters the deep stack.

**Block :**

1. **Temporal Sub-Layer (Mamba2):**
* **Norm:** `RMSNorm(256)`.
* **Core:** `Mamba2(d_model=256, d_state=128, expand=2)`.
* *Note:* Uses `seq_idx` during training to reset SSM state at episode boundaries.
* **Residual:** .


2. **Spatial Sub-Layer (Relational Attention):**
* **Norm:** `RMSNorm(256)`.
* **Edge Projection:** .
* **Attention:** Multi-Head Attention (4 Heads, Head Dim=64).
*  (*Broadcast add: Geometry biases the Keys*)
*  (*Broadcast add: Geometry biases the Values*)


* **Mask:** Dead ships set to  in score matrix.
* **Residual:** .



---

### E. The World Head (The Dreamer)

* **Input:** Backbone Output .
* **Layers:** `Linear(256 -> 256)`  `RMSNorm`  `SiLU`  `Linear(256 -> Targets)`.

---

## 4. Output Targets (Delta Predictions)

The World Model predicts **Residuals** (Deltas) to be applied to the current state.

**Targets:**

*  (Displacement Vector) — *Used to update Latent/Ghost Position.*
*  (Linear Acceleration).
*  (Angular Acceleration).
* , , .
* **Discrete:** `Is_Alive_Next` (Logits), `Is_Shooting_Next` (Logits).

---

## 5. Training Protocol (Continuous Stream)

### A. Data Loader & Sequence Packing

To minimize wasted compute, we treat the data as a continuous stream of tokens rather than padded batches.

* **Source:** A massive buffer of concatenated episodes: `[Ep1, Ep2, Ep3...]`.
* **Sampling:** Uniformly sample a start index . Grab chunk `[i : i + 1024]`.
* **Sequence Indexing (`seq_idx`):**
* The loader generates an integer tensor `seq_idx` of shape `[Batch, Time]`.
* Value increments (e.g., ) at the frame index  where Ep1 ends and Ep2 starts.
* This tensor is passed to `Mamba2` to trigger **Instant State Reset** in the kernel.


* **Semantic Reset:**
* At the transition index , we also add the `Reset_Vector` to the input state  to inform the Attention layers (which do not see `seq_idx`) that the context has changed.



### B. Loss Masking

We generate a boolean mask tensor `M` of shape `[Batch, Time, Ships]`.

**Mask  (Drop Loss) when:**

1. **Transition Frame:** The target is part of a new episode relative to the input.
2. **Dead Entity:** `HP <= 0` in the *target* step.

---

## 6. Inference Modes

### Mode A: Pure Dreaming (Hallucination)

*Used for long-term planning, MCTS, or visualization.*

1. **Initialize:**  from real world. .
2. **Loop:**
* **Actor Step:**
* Encode .
* Run Actor Pass (Fusion + Spatial + Head)  Predict .


* **World Step:**
* Embed & Fuse  into .
* Run Mamba Backbone  Predict .


* **Integration:**
* .
* Update Ghost Positions/Edges for next step.





### Mode B: Physics-in-the-Loop (Agent Gameplay)

*Used when playing the game or training the Actor (RL).*

1. **State:** The **Game Engine** holds the truth.
2. **Loop:**
* **Observation:** Engine provides .
* **Actor Step:**
* Encode  + .
* Run Actor Pass  Predict .


* **Execution:**  is sent to the Game Engine.
* **Physics Step:** Game Engine integrates physics (60Hz -> 15Hz).
* **Mamba Update (Open Loop):**
* To keep the Mamba hidden state  current, we must run the World Model "blindly."
* Fuse  +  (Executed).
* Run Mamba Backbone to update state to .