# Interleaved World Model Architecture & Training Pipeline

 ## Overview

 The **Interleaved World Model** is a transformer-based autoregressive model designed to simulate the *Boost and Broadside* environment. It models the game state evolution by processing an interleaved sequence of **State ($S$)** and **Action ($A$)** tokens for multiple agents (ships).

 $$ S_0 \to A_0 \to S_1 \to A_1 \dots $$

 *   **Input**: Sequence of $N$ ships' states and actions.
 *   **Objective**: Predict the next token in the sequence.
     *   $S_t \to A_t$: Policy / Behavior Cloning (predict what action a ship takes given its state).
     *   $A_t \to S_{t+1}$: Physics / Environment Model (predict the next state given current state and action).

 ---

 ## Model Architecture

 ### 1. Tokenization & Embedding

 #### State Encoding ($S_t$)
 *   **Input**: Continuous vector of size `state_dim` (default 15).
     *   Includes: Position (sin/cos), Velocity, Attitude, Health, Power, etc.
 *   **Architecture**:
     1.  **Fourier Features**: Maps input low-dim coords to high-freq features (`DyadicFourierFeatureExtractor`). This helps learn high-frequency functions (sharp boundaries).
     2.  **Gated SwiGLU**: A GLU variant with Swish activation, mapping `embed_dim` $\to$ `2 * embed_dim` $\to$ `embed_dim`.
     3.  **Residual + LayerNorm**: Standard transformer block structure.

 #### Action Encoding ($A_t$)
 *   **Input**: Discrete indices for [Power (3), Turn (7), Shoot (2)].
 *   **Architecture**:
     *   Separate `nn.Embedding` for each action component.
     *   Concatenation $\to$ Linear Projection $\to$ SiLU activation.

 #### Positional & Identity Embeddings
 *   **Ship ID**: Learned embedding per ship index ($0 \dots N-1$).
 *   **Team ID**: Learned embedding for Team 0 vs Team 1.
 *   **Type ID**: Differentiates State tokens from Action tokens.
 *   **Additivity**: These are added to the base token embeddings:
     $$ E_{final} = E_{content} + E_{ship} + E_{team} + E_{type} $$

 ### 2. Transformer Backbone

 The backbone consists of $L$ layers alternating between **Temporal** and **Spatial** attention mechanisms.

 #### Temporal Attention (Self-History)
 *   **Scope**: Each ship attends *only to its own history*. Ship $i$ at time $t$ cannot see Ship $j$.
 *   **Mechanism**: Causal Self-Attention.
 *   **Positional Encoding**: **RoPE (Rotary Position Embeddings)** is applied here to encode relative time distances.
     *   **Optimization**: Uses a static precomputed cache of length `2 * max_context_len` (rounded up to the next power of 2) for device efficiency.
 *   **Benefit**: Allows the model to learn momentum, reload times, and temporal dynamics independent of other agents.

 #### Spatial Attention (Inter-Agent)
 *   **Scope**: Each ship attends to *all other ships* within the **Current Step** only.
 *   **Mechanism**: Non-causal spatial attention over the $N$ ships. 
     *   $S_t$ attends only to $S_t$ of all ships.
     *   $A_t$ attends only to $A_t$ of all ships.
 *   **Relational Bias**: A computed bias is added to the attention scores to inject spatial awareness explicitly.
     $$ \text{Attn}(Q, K) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}} + \text{Bias}_{\text{rel}}(i, j)\right) $$
     *   Bias is derived from pairwise features (distance, relative velocity, heading difference) via a projected 12D feature vector.

 ### 3. Prediction Heads

 #### State Head
 *   **Input**: Action Token ($A_t$).
 *   **Output**: Predicted Next State Vector ($S_{t+1}$).
 *   **Loss**: MSE.

 #### Action Head
 *   **Input**: State Token ($S_t$).
 *   **Output**: Logits for Power, Turn, Shoot.
 *   **Loss**: Cross-Entropy (sum of 3 components).

 #### Relational Head
 *   **Input**: Action Token ($A_t$).
 *   **Output**: Pairwise Relational Features for $t+1$ (12D vector per pair).
 *   **Architecture**: **Bilinear Layer**.
     *   Avoids constructing massive $(B, T, N, N, E)$ tensors.
     *   Uses low-rank factorization to predict edge features directly from node embeddings: $Rel(i, j) = U(e_i)^T V(e_j)$.
 *   **Loss**: MSE against ground-truth geometric features.

 ---

 ## Training Pipeline

 ### Data Loading
 *   **Source**: HDF5 (`aggregated_data.h5`).
 *   **Batching Strategy**:
     *   **Short Batches**: 32 samples (random short windows, length ~32). Good for IID sampling and stable gradients.
     *   **Long Batches**: 128 samples (long chunks, length ~128). Necessary for learning long-term dependencies and verifying stability.
     *   **Context Window**: 96 steps.
     *   **Ratio**: Tunable (default 4:1 Short:Long).
 *   **Curriculum**: "Sobol" sampling or fixed patterns can be used to mix these.

 ### Optimization
 *   **Loss**: Weighted sum of State, Action, and Relational losses.
     $$ \mathcal{L} = \lambda_s \mathcal{L}_{state} + \lambda_a \mathcal{L}_{action} + \lambda_r \mathcal{L}_{rel} $$
 *   **Scheduled Sampling (Training Rollouts)**:
     *   *Code Implementation*: The trainer has a `perform_rollout` function capable of in-place scheduled sampling.
     *   *Current Status*: **Disabled by default** in configuration (`rollout.enabled: false`). Training proceeds with 100% Teacher Forcing unless this is enabled.
     *   *Purpose*: If enabled, it bridges the "exposure bias" gap between training and inference.
 *   **Gradient Accumulation**: Supports accumulating gradients over multiple micro-batches (effective for large effective batch sizes).
 *   **SWA (Stochastic Weight Averaging)**:
     *   Starts after a warmup period or fixed epoch.
     *   Maintains a moving average of weights to find flatter minima and improve generalization.

 ### Validation
 1.  **Teacher Forcing**: Standard validation using ground-truth history. Measures one-step prediction error.
 2.  **Autoregressive Rollout (Dreaming)**:
     *   **Eval Only**: Runs the model in "dream mode" (closed loop) for $K$ steps (e.g., 50).
     *   Feeds its own predictions back as input.
     *   *Metric*: Calculates `val_rollout_mse_state` (Mean MSE) and `val_rollout_mse_step` (Per-step MSE).
     *   *Current Logging*: Currently only the scalar Mean MSE is logged to WandB. The per-step vector is computed but filtered out by the logger.

 ---
