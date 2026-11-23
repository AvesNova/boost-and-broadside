# Literature check: five model choices in lean (2026-11-23)

Scope: the lean model on branch `core` (`lean/model/policy.py`, `position.py`, `temporal.py`,
`NOTES.md`), the open items in `author-review.md` §6 and the author's answers in
`core-decisions.md` ("Answers from the author review, sections 2 to 6"). The rule being
served: every output trained in nats with cross-entropy, and otherwise follow the literature.

This builds on `provenance-check.md` (HL-Gauss sigma and spacing §1, DreamerV3's symlog §2,
gated attention and null keys §3, Griffin's initialisation §4, grid cells §6, partial rotary
§7) and `rotary-and-gate-check.md` (rotary outside language models §1, gate initialisation §2).
What those settled is referred to, not repeated.

Labels on each claim:
- **measured**: a primary source reports an experiment.
- **default**: what released code or a paper does, without an ablation.
- **inference**: my reasoning or arithmetic; nothing measured it.

Everything cited was fetched on 2026-11-23 and the named passage read, unless it is marked
otherwise. Code was downloaded at the commits named. Lean line numbers are from the tree at
6c7a1555.

---

## 1. Next-state head: binned targets, bin spacing, target form

### What lean does now

`next_state_target` (`policy.py:119-131`) returns seven columns: the one-tick change of x, y,
heading cos, heading sin, speed, energy and shield, each divided by a fixed scale
(`policy.py:78-86`). It is trained as a regression (`learn/update.py:104`, squared error). The
author has decided it moves to binned cross-entropy (core-decisions 6.3).

### Sources

**Imani and White, "Improving Regression Performance with Distributional Losses", ICML 2018**,
https://arxiv.org/abs/1806.04613 (abstract read). HL-Gauss comes from here: a supervised
**regression** loss, not a value loss. The abstract says it "significantly improves prediction
accuracy". The authors attribute the gain to easier optimisation ("better-behaved gradients"),
not to less overfitting or better representations. (measured)

**Imani, Luedemann, Scholnick-Hughes, Elelimy, White, "Investigating the Histogram Loss in
Regression"**, arXiv 2402.13425, https://arxiv.org/html/2402.13425 (first 100k characters read).
- Section 4: HL-Gauss against ℓ2, ℓ1, an MDN, a one-bin target and others on four regression
  datasets (CT Position, Song Year, Bike Sharing, Pole). §5.1 covers time series (LSTM and
  GRU helped, a linear model did not, a transformer about tied); §5.2 covers Atari state
  values. (measured)
- §4.7: "the performance gain is not due to better representations"; HL-Gauss still beats ℓ2
  on a fixed random representation. §4.8: "more stable gradients than ℓ2". §4.6: most of
  HL-Gauss's lead over a one-bin target comes from the one-bin target's bias. §4.10: the study
  assumes the target range is known. (measured)

**Farebrother et al. 2024, "Stop Regressing"**, https://arxiv.org/abs/2403.03950 (PDF read).
Spacing and sigma are in provenance §1. What matters here:
- §5.1.2 and Figure 13: HL-Gauss beats Two-Hot across a wide range of σ. Two-Hot "corresponds
  to not using any label smoothing". (measured, Atari values)
- Appendix (Figure 17, regression to targets of growing magnitude): "HL-Gauss outperforms
  Two-Hot, consistent with our previous findings". A synthetic non-stationary regression,
  101 bins on [−40, 40]. (measured)
- §6: two-hot was used in MuZero, DreamerV3 and TD-MPC2 "without any analysis"; "in our
  experiments, Two-Hot performs worse than other cross-entropy losses and MSE". (measured)

**DreamerV3**, Hafner et al., arXiv 2301.04104 (PDF read); code
https://github.com/danijar/dreamerv3 at e01491f.
- "For potentially stochastic targets, such as rewards or returns, we introduce the symexp
  twohot loss" (Robust predictions, Eqs. 9-11). Vector observations are decoded with the
  **symlog squared loss**, not two-hot: "Dreamer transforms vector observations using the
  symlog functions, both for the encoder inputs and the decoder targets and employs the symexp
  twohot loss for the reward predictor and critic". (default)
- Code: `dreamerv3/rssm.py:299` picks `categorical` for discrete observation keys and
  `symlog_mse` for continuous ones; `configs.yaml:98,101` give `symexp_twohot`, 255 bins, for
  the reward head and the critic. (default)
- The decoder predicts the **current** observation from the latent (reconstruction); there is
  no next-state delta head. (default)
- Ablation (Figure 6): removing symexp two-hot (Huber instead) or observation symlog each hurts
  on some tasks. (measured, against Huber and no-symlog, not against HL-Gauss)

**TD-MPC2**, Hansen, Su, Wang 2024; code https://github.com/nicklashansen/tdmpc2 at 332f19e.
- `common/math.py:58-72`: two-hot in **symlog space** over `linspace(vmin, vmax, num_bins)`;
  `config.yaml:50-52`: 101 bins on [−10, 10]. Used for reward and Q only. (default)
- `tdmpc2.py:272-275`: dynamics are trained by **MSE between latents** (`consistency_loss`),
  not by predicting observations. (default)

**M3**, Cohen, Wang, Kang, Gadot, Mannor, arXiv 2502.11537v2,
https://arxiv.org/html/2502.11537v2 (read through a summarising fetch).
- §2.3 and App. A.3.1: the reward head uses **HL-Gauss applied to symlog(y)** over 128 uniform
  bins on (−15, 15), σ = 0.75 bin widths (Table 3: 0.1758), read out as symexp of the
  expectation. HL-Gauss was chosen on the strength of Farebrother et al. (default)
- App. A.2.2: continuous observation features are quantised to 125 values in [−6, 6] and
  predicted with per-token cross-entropy. (default)
- §3.3 ablates "classification-based predictions" as a whole; the text gives no numbers and
  no HL-Gauss against two-hot comparison. (not usable as a measurement)

**Trajectory Transformer**, Janner, Li, Levine 2021, arXiv 2106.02039 (PDF read).
- §3.1: "we discretize each dimension independently"; V = 100 bins per dimension (Table in
  App. A); trained with teacher-forced cross-entropy. Two schemes: uniform bins over
  [min, max], or quantile bins (equal data mass per bin). (default)
- §4.2: "uniform and quantile, perform similarly on all environments except for
  HalfCheetah-Medium-Expert, where the large range of the velocities prevents the uniform
  discretization scheme from recovering the precise actuation … the quantile discretization
  approach achieves a return of more than twice that of the uniform discretization". (measured)
- It predicts **absolute** next states, not changes. (default)

**IRIS**, Micheli, Alonso, Fleuret 2023, arXiv 2209.00588 (read through a summarising fetch).
§2.1-2.2: frames become 16 tokens from a 512-entry codebook; next-frame tokens and termination
by cross-entropy, reward by "a mean-squared error loss or a cross-entropy loss … depending on the
reward function". (default)

**MuZero**, Schrittwieser et al., arXiv 1911.08265 (PDF read), App. F: rewards and values are
scaled by h(x) = sign(x)(√(|x| + 1) − 1) + εx, then put on "a discrete support set of size 601
with one support for every integer between −300 and 300" (two-hot style). No state or
observation prediction. (default)

**EfficientZero** (Ye et al., arXiv 2111.00210, §4.1) and **SPR** (Schwarzer et al.,
arXiv 2007.05929, §2): next-step targets are **latents** (SimSiam-style consistency;
cosine similarity to a target encoder). SPR App.: "Using a quadratic loss causes collapse".
(default; EfficientZero's §4.1 ablation reports the consistency loss is critical, measured)

**UNREAL**, Jaderberg et al., arXiv 1611.05397 (PDF read), "Reward Prediction" section: the reward-prediction
auxiliary uses "a multiclass cross-entropy classification loss across three classes (zero,
positive, or negative reward)". (default)

**Change as the target.** Nagabandi, Kahn, Fearing, Levine 2018, "Neural Network Dynamics for
Model-Based Deep RL with Model-Free Fine-Tuning", arXiv 1708.02596 (PDF read), §III.A: s_{t+1}
from s_t "can be difficult to learn when the states s_t and s_{t+1} are too similar … this
difficulty becomes more pronounced as the time between states Δt becomes smaller … We overcome
this issue by instead learning a dynamics function that predicts the change in state". Stated
reasoning, no ablation in the passage read. (default)

**Change relative to motion already known.** Sanchez-Gonzalez et al. 2020, "Learning to Simulate
Complex Physics with Graph Networks", arXiv 2002.09405 (PDF read): the network predicts
per-particle **accelerations**, integrated by semi-implicit Euler (§2, App. "Update mechanism";
p̈ = p_{t+1} − 2p_t + p_{t−1}, App. "Input and output representations"). With the previous
velocity as input, the acceleration is exactly the position change minus the constant-velocity
extrapolation, i.e. form (c). The ablations (Figure 4) cover message passing steps, shared
weights, radius, noise and relative encoders, **not** the target form. (default)

No paper found that compares (a), (b) and (c) as targets of an RL auxiliary loss.

### What it means here

- **HL-Gauss for an auxiliary target.** HL-Gauss was introduced and tested as a general
  regression loss (Imani and White 2018; Imani et al. 2024, including time series), and it
  beat Two-Hot for values in two separate settings in Farebrother et al. So "HL-Gauss was for
  values" understates its record: the one head-to-head against Two-Hot is on values and on a
  synthetic regression, and the supervised-regression record is against ℓ2, not Two-Hot. No
  world-model or auxiliary-loss paper measures HL-Gauss against Two-Hot (M3 uses HL-Gauss on
  rewards without that ablation). Two-hot in DreamerV3, TD-MPC2 and MuZero is a default with no
  comparison against HL-Gauss. (inference from measured pieces)
- **What DreamerV3 actually does for states.** Its two-hot is for rewards and values; continuous
  state is a symlog squared error. Under the author's "everything in nats" rule DreamerV3 is not
  a precedent for binned state targets. Trajectory Transformer and M3 are: per-dimension bins
  with cross-entropy, uniform or quantile bins. (default)
- **The present energy problem** (author-review 6.3: energy's mean square about 1,470, loss
  209.8 at weight 0.1) is the scale coupling that DreamerV3 and Imani et al. describe. A
  cross-entropy per column is bounded by ln(bins) at a uniform prediction, so no column can
  dominate by its units. (inference)
- **Linear or symlog bins.** HL-Gauss was tested only on uniform bins. The one precedent for
  HL-Gauss with symlog is M3: the Gaussian is applied to symlog(y) and the bins are uniform in
  symlog space, so "σ = 0.75 bin widths" keeps its meaning. The lean columns have known
  physical bounds (turn rate, top speed, the store, the shield), so linear bins over a measured
  range lose nothing for the columns whose changes are mostly of one size. A column whose change
  is usually near 0 with rare large jumps (shield on a hit, energy on firing) wastes most
  linear bins; symlog-space bins (as M3) or quantile bins (as Trajectory Transformer, which
  measured a 2x return gain where one dimension had a large range) suit it better. (inference,
  with the Trajectory Transformer result measured)
- **Target form.**
  - (a) absolute next value: Trajectory Transformer. With bins over the arena (8.4 km) a 64-bin
    code has 130 m bins, against a one-tick motion of at most about 11 m (325 m/s / 30). The
    change is lost below the bin width. Not suitable. (inference, arithmetic)
  - (b) change from the current value: Nagabandi et al.'s reasoned default, and the common
    model-based RL form. For x and y the change is mostly v·Δt, which the ship's own features
    already carry (`contract.py:110-111`, velocity_x, velocity_y), so much of the head's work
    is copying an input. (inference)
  - (c) change relative to dead reckoning: GNS's acceleration target. For x and y it isolates
    what the input does not give: thrust, drag, turn and collisions. For heading, speed,
    energy and shield, dead reckoning is "no change" unless a deterministic rule is subtracted
    (regeneration, recharge), so (b) and (c) coincide unless that rule is modelled.
    (inference)
- **Stochasticity.** Today the head reads the tick-t token, which does not contain the action
  sampled at t, while lean applies that action on tick t. The target is then a mixture over the
  ship's own 42 actions. A categorical head represents that mixture; a regression averages it.
  Once actions apply one tick late (decided in 3.3, not built), the pending action is an input
  and the target becomes close to deterministic apart from other ships. HL-Gauss is still
  appropriate then: its decoded mean is unbiased three bin widths inside the support (`value.py`
  docstring). (inference)

### Recommendation

HL-Gauss cross-entropy per column, σ = 0.75 bin widths, 64 bins per column like the critic. Use
linear bins for columns of bounded, mostly one-sized change, and bins uniform in symlog space
(M3's form) for the heavy-tailed shield and energy changes, each column's range set from
measured changes. Target form (c) for x and y (the change minus velocity × Δt, using the game's
own integration step), and (b) for the other five. Strength: **moderate** for HL-Gauss over
Two-Hot (measured on values and one regression; no auxiliary-target test exists), **weak** for
the spacing per column and for (c) over (b) (reasoned from GNS and Nagabandi; nobody measured it
for an auxiliary loss). Cheap check before a run: per-column histograms of the target under (b)
and (c) from one collection, and the share of mass in the central bins.

---

## 2. Gated attention: where the gate acts, and the value back-rotation

### Sources

**Qiu et al. 2025, "Gated Attention for Large Language Models"**, arXiv 2505.06708,
https://arxiv.org/html/2505.06708 (read through a summarising fetch; §2.2 and Table 1).
- Positions (§2.2, Figure 1, Eq. 5, Y′ = Y ⊙ σ(XW_θ)): **G1** on the SDPA output, before the
  output projection W_O; **G2** on the value projection; **G3** on the keys; **G4** on the
  queries; **G5** after W_O. X is "the hidden states after pre-normalization" (§2.2, footnote 1).
  G5 "does not address the lack of non-linearity between W_V and W_O" (§4.1).
- Table 1 (15A2B MoE, 400B tokens), average PPL / MMLU: baseline 6.026 / 58.79; G1 elementwise
  5.761 / 60.82; G1 headwise 5.792 / 60.05; G2 elementwise 5.820 / 59.17; G2 headwise
  5.808 / 59.32. (measured)
- The paper does not discuss gating together with RoPE.

**Official code**, https://github.com/qiuzh20/gated_attention at f4c2a5f, `modeling_qwen3.py`:
- Lines 276-279: gate columns appended to `q_proj`. Lines 314-317: the gate is split off
  **before** `apply_rotary_pos_emb` (line 331), which rotates only the query and key: the gate
  is never rotated.
- Lines 352-366: `attn_output = attn_weights @ value_states`; transpose; `attn_output *
  torch.sigmoid(gate_score)` (line 362); reshape; `o_proj`. So G1 is per head, on the
  attention-weighted sum, immediately before the output projection. Values are not rotated in
  this model, so the code has no back-rotation step to place the gate against. (default)

**GTA**, Miyato et al., arXiv 2310.10375 (Eq. 6, read in rotary-and-gate-check §1b):
O = P_g ⊚ Attn(P_gᵀQ, P_g⁻¹K, P_g⁻¹V). The back-transform P_g is part of the attention output,
and dropping the value transform costs 2.4 dB PSNR (measured there). GTA has no output gate.

No paper or code found that combines an output gate with rotated values.

### What lean does

`policy.py:225-229`: SDPA, then the back-rotation by the query's angles (line 227), then
`attended * sigmoid(gate)` (line 228), then the output projection. The gate block is the fourth
part of `query_key_value_gate` and is not rotated (line 218: only the first three parts). That is
G1 on the same kind of tensor as the released code: the per-head attention output in the
coordinates the output projection reads.

### Does the order matter

Yes. Let y = Σ_j a_j R(θ_j) v_j be the attended sum with values rotated by the keys' angles, and
g the gate (computed from the query token's content). With the gate after the back-rotation the
output is g ⊙ R(−θ_q) y = g ⊙ Σ_j a_j R(θ_j − θ_q) v_j, which depends only on relative angles.
With the gate before it, R(−θ_q)(g ⊙ y): a common shift δ of every angle turns y into R(δ) y,
and R(−θ_q − δ)(g ⊙ R(δ) y) equals the unshifted output only when the two gate values of each
rotated pair are equal. An element-wise gate therefore breaks the "attention reads only relative
positions" property if it acts before the back-rotation; a head-wise gate (one scalar per head)
would commute with the rotation and the order would not matter. (inference, algebra; the
lean test `test_attention_reads_only_relative_positions` would catch the wrong order with an
element-wise gate)

### Recommendation

Keep G1, element-wise, after the back-rotation and before the output projection, as lean has it.
Strength: **strong** for the order (it follows from the algebra and from the role of the
back-rotation as part of the attention output, as in GTA's Eq. 6), **moderate** for G1
element-wise as the position (measured best in Qiu et al. Table 1, on language models only). The
gate's initialisation is in rotary-and-gate-check §2 and item 3 below.

---

## 3. Initialisation: residual outputs, the reset, the heads

### Sources

**recurrentgemma**, https://github.com/google-deepmind/recurrentgemma at 2efa84d,
`recurrentgemma/torch/`:
- `griffin.py:62-75`: every `ResidualBlock` gets `final_w_init_variance_scale = 2.0 /
  num_layers`; `num_layers` counts residual blocks, each a temporal-mixing block plus an MLP
  block. (default)
- `modules.py:357-376` (`LocalAttentionBlock`): q, k, v N(0, 1/width); final projection
  N(0, s/width), bias zero. `modules.py:565-583` (`RecurrentBlock`): the two input branches
  N(0, 1/width), output N(0, s/lru_width). `modules.py:728-737` (`MLPBlock`): up projection by
  `Einsum` (`layers.py:690-697`, N(0, 1/width), bias zero), down projection N(0, s/expanded),
  bias zero. So the same rule covers attention, recurrent and MLP sublayers. (default)
- `layers.py:335, 351-353`: `reset = segment_pos == 0`; `multiplier = reset + ~reset *
  sqrt(1 − a²)`: the input is **not** scaled at a segment's first position. (default)

**Griffin paper**, arXiv 2402.19427 (PDF read), §2.4 Eq. 4: h_t = a_t ⊙ h_{t−1} + √(1 − a_t²) ⊙
(i_t ⊙ x_t), "similar to Orvieto et al. (2023b)". The paper does not mention resets.

**LRU**, Orvieto et al. 2023, "Resurrecting Recurrent Neural Networks for Long Sequences",
arXiv 2303.06349 (PDF read), §3.4, Proposition 3.3: with eigenvalues near the unit circle the
state's squared norm **as k → ∞** grows by about 1/(1 − r²) under white-noise input; the γ =
√(1 − |λ|²) normalisation removes that blow-up (Table 3, measured on LRA). The rationale is the
stationary scale of the state, which does not hold at the first step after a zero state.

**GPT-2**, Radford et al. 2019 (PDF read, §2.3): "We scale the weights of residual layers at
initialization by a factor of 1/√N where N is the number of residual layers". (default)

**T-Fixup** (Huang, Perez, Ba, Volkovs, ICML 2020) and **DeepNet / DeepNorm** (Wang, Ma, Dong,
Huang, Zhang, Wei, arXiv 2203.00555; abstract read: "combines the best of two worlds, i.e., good
performance of Post-LN and stable training of Pre-LN"): both are schemes for Post-LN or
norm-free transformers of tens to hundreds of layers. T-Fixup's paper was not read. Lean is
pre-norm with 6 residual blocks, so neither applies. (inference)

**PPO heads.** CleanRL, https://github.com/vwxyzjn/cleanrl at fe8d8a0, `cleanrl/ppo.py:94-115`:
orthogonal gain √2 for hidden layers, **0.01** for the policy's last layer, **1.0** for the scalar
value's last layer. (default) Andrychowicz et al., "What Matters in On-Policy RL? A Large-Scale
Empirical Study", arXiv 2006.05990 (PDF read), §3.2: "Initialize the last policy layer with 100×
smaller weights"; "The scale of the last layer initialization matters much less for the value
MLP (C58) than for the policy MLP". (measured, continuous-control MuJoCo with Gaussian policies)

**Categorical heads.** DreamerV3, Methods: "The output weights of twohot distributions are
initialized to zero to ensure that the agent does not hallucinate rewards and values at
initialization"; `configs.yaml:98-101`: `outscale: 0.0` for the reward and value heads, `0.01`
for the policy. TD-MPC2 `world_model.py:32`: `init.zero_` on the last layers of the reward head and
of the Q ensemble. (default)

**Rare classes.** Lin et al., "Focal Loss for Dense Object Detection", arXiv 1708.02002 (PDF read),
§3.3 and §5.1: set the last layer's bias so the initial probability of the rare class is a prior
π (b = −log((1 − π)/π)). Standard cross-entropy without it "fails quickly, with the network
diverging"; with π = 0.01 it trains, and results are insensitive to π. (measured, detection with
heavy class imbalance)

### What it means for each item

- **Depth scale.** Lean applies recurrentgemma's rule (s = 2/6) to the temporal sublayer only
  (`temporal.py:72-79`); the spatial sublayers keep PyTorch's default. Arithmetic: PyTorch's
  `nn.Linear` default (Kaiming uniform, a = √5) has variance 1/(3·fan_in), and s/fan_in at s = 1/3
  is the same number. So at 6 blocks the spatial **output** projections already match the rule in
  variance (uniform instead of normal). The departure is in the spatial **input** projections
  (query, key, value, gate and the MLP's input): 1/(3·width) against recurrentgemma's 1/width,
  a third of the variance. (inference, arithmetic)
  - Counting: recurrentgemma counts residual blocks of (mixing + MLP). Lean's temporal sublayer
    and each spatial sublayer are each one such block, so 2 × (1 + 2) = 6 is the matching count.
    GPT-2 counts each residual branch (12 here) and scales the std by 1/√N relative to its base;
    the two rules are different conventions, and at 6 blocks the choice changes the output std by
    a small factor. (inference)
- **sqrt(1 − a²) at a reset.** With the state zeroed, scaling the first input by √(1 − a²) makes
  the state start far below its stationary scale on the slow channels. Lean's decay init puts
  a^c in [0.9, 0.999] (`temporal.py:80-82`); at the start-of-training gate r ≈ 0.5 the per-tick
  retention reaches 0.9995, so the first input is scaled by about 0.03, and that channel takes on
  the order of 1/(1 − a²) ≈ 1,000 ticks (about 33 s) to reach its stationary scale. The released
  code's 1 at a reset starts the state at the input's scale. The cost: the first tick's input then
  outweighs later single inputs on the slow channels until it decays. (inference, arithmetic; the
  choice itself is the released code's default, not an ablation)
  The same reasoning applies to the tick after a masked row (lean zeroes the carry there too,
  `temporal.py:120-121`).
- **Policy head.** 0.01 orthogonal makes the initial policy close to uniform over the 42 actions.
  Measured as the most important init choice in Andrychowicz et al. (continuous actions); the
  categorical case follows by the same reasoning (maximum entropy at the start). (measured /
  inference)
- **Value levels.** The 1.0 convention is for a scalar value output. For categorical logits a
  gain of 1.0 on a 256-wide hidden layer gives random logits of order 1 and random initial value
  estimates; DreamerV3 and TD-MPC2 zero the last layer so the initial distribution is uniform.
  Lean's 0.01 is nearly the same. A uniform distribution decodes to the support's midpoint, which
  is 0 only for a symmetric support; NOTES reports the first value loss at ln 64, consistent with
  near-uniform logits. (default / inference)
- **Next-state head.** The same as the value levels once it is categorical: zero or 0.01. Zero
  leaves the first layer without a gradient for one step only (NOTES choice 7); both are within
  the published practice. (default / inference)
- **Hit head.** Three classes with very unequal frequencies (most shots miss). Lin et al.'s prior
  bias, b_c = log p_c from a measured class frequency, gives the right marginal at step 0 at no
  cost. With 0.01 weights and zero bias the initial loss is ln 3 ≈ 1.10 nats regardless of how
  rare hits are. (measured in detection / inference here)

### Recommendation

- Spatial sublayers: initialise like recurrentgemma's attention and MLP blocks, the same rule
  as the temporal sublayer: query, key, value and gate N(0, 1/width), output projection
  N(0, s/width), MLP input N(0, 1/width), MLP down projection N(0, s/hidden), s = 2/6. One rule
  for all six residual blocks. Strength: **weak**. It is the released code's default and makes
  the model consistent; the measured effect of init scale at 6 pre-norm blocks is likely small,
  and nothing here measured it.
- sqrt(1 − a²): use 1 at a reset (recurrentgemma), and also on the tick after a masked row.
  Strength: **moderate**. This is the released code, with the LRU rationale (the normalisation
  is about the stationary state); no ablation was found.
- Heads: policy last layer orthogonal 0.01 (**moderate**: measured, continuous control); value
  levels and next-state last layers zero or 0.01, keep 0.01 (**moderate**: the DreamerV3 and
  TD-MPC2 default for categorical heads; 1.0 does not carry over from scalar values); hit head
  last weights 0.01 with biases at log class frequencies measured from one collection
  (**weak to moderate**: measured in detection, cheap, not tested in RL). T-Fixup and DeepNet:
  not applicable (Post-LN, deep).

---

## 4. Outcome and reward-level values: per ship or on the game token

### Sources

**MAPPO**, Yu et al., "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games",
arXiv 2103.01955, https://arxiv.org/html/2103.01955 (first 100k characters read through a
summarising fetch).
- §5.2, Figures 3-4: value inputs compared on SMAC: concatenated local observations (CL)
  "ineffective, particularly in maps with many agents"; the environment's global state (EP)
  "subpar performance in more difficult maps"; **agent-specific global state** (AS: global state
  plus the agent's own observation) and its feature-pruned form (FP) strong. Table 1:
  27m_vs_30m FP 93.8, AS 85.9, IPPO 69.5. (measured)
- One critic network shared by all agents, evaluated per agent on agent-specific inputs (§3.3,
  App. C.2). Separate networks per agent are much worse (MMM2 13.0 against 90.6). (measured)
- App. C.3, Figure 10: **death masking** (a dead agent's critic input zeroed except its ID)
  "significantly outperforms" no masking and dropping dead samples; agents are dead in about
  20% of steps. (measured)
- Hanabi (Table 3): IPPO equals MAPPO at 2 players and falls behind at 5. (measured)

**Lyu, Xiao, Daley, Amato, "Contrasting Centralized and Decentralized Critics in Multi-Agent
RL"**, AAMAS 2021, arXiv 2102.04402 (abstract read): "the centralized critic design is not
strictly beneficial"; each has trade-offs (bias and variance). (measured and theory; body not
read)

**OpenAI Five**, Berner et al., arXiv 1912.06680 (read through a summarising fetch). App. H: one
replica network per hero, "the value function is computed as another linear projection of the
LSTM state", so a value per hero. App. G: per-hero reward r_i = (1 − τ)ρ_i + τρ̄ ("team spirit"),
"Ultimately we care about optimizing for team spirit τ = 1"; lower τ "reduce[s] gradient variance
in early training"; Table 2: τ 0.3 → 0.8 (Five), 0.3 → 1.0 (Rerun); App. O: τ = 0 best very early,
0.5 at medium skill. (default; App. O measured, not read here in full) §5 notes AlphaStar's value
network "observed full information about the game state (including observations hidden from the
policy)" as a direction OpenAI Five did not take.

**AlphaStar** controls all units with one agent, so its per-agent versus team question does not
arise; its critic's privileged information (above, as reported by Berner et al.) is a different
axis. Not read further.

### What it means here

- Every lean ship token attends to the whole team and the game token, so a per-ship value already
  reads the team's information plus its own: MAPPO's AS form, with one shared network. The game
  token also attends to every ship, so a game-token value is closer to EP plus learned pooling;
  MAPPO's EP lacked any agent-specific features, which the game token can still reach by
  attention. The measured EP against AS gap therefore overstates the difference here.
  (inference)
- **Per-ship reward levels** (damage, kill_death, objective_progress): the target differs per
  ship, so the value must be per ship. (inference, by definition)
- **Outcome**: one number per team, the same target for every ship. Per ship gives each ship its
  own baseline from its own token (AS, measured strong; OpenAI Five's per-hero value at τ = 1 is
  the same arrangement). A game-token value gives one baseline shared by the team; it persists
  through deaths, whereas a dead ship's row is masked in lean (MAPPO measured death masking as
  better than dropping dead agents' samples, but in lean a dead ship takes no actions and needs
  no advantage). Nothing measured a pooled-token value against per-agent values for a shared
  team reward. (inference)

### Recommendation

Keep every level per ship, the outcome included (MAPPO's agent-specific value with a shared
network; OpenAI Five's per-hero value). An outcome head on the game token is a reasonable arm,
not a default. Strength: **moderate** for per ship on the reward levels (forced by the targets),
**weak to moderate** for the outcome (MAPPO's AS over EP is measured, but lean's game token is not
MAPPO's EP).

---

## 5. Grid cells in RL, and the spacing of multi-scale position codes

### Sources

Banino et al. 2018, Cueva and Wei 2018, Blair et al. 2007 and Space2Vec 2020 are summarised in
provenance §6 (the grid code in Banino is learned, and fed to an A3C agent; nobody tested a fixed
hexagonal code in RL). Added here:

**Yu, Behrens, Burgess, "Prediction and Generalisation over Directed Actions by Grid Cells"**,
arXiv 2006.03355v2 (read through a summarising fetch): grid cells as Fourier modes of a
translation-invariant transition matrix; a "gc-DQN" with 50 low-frequency Fourier modes on
discretised CartPole "greatly accelerates learning" against DQN and Dyna-Q, 5 seeds, results in a
figure only, described by the authors as "preliminary". (measured, weak)

**Yang, Ajay, Agrawal, "Overcoming the Spectral Bias of Neural Value Approximation"**,
arXiv 2206.04672 (abstract read): Fourier-feature inputs to the Q-network, "a single line change",
state-of-the-art continuous control in their tests. Fourier features help RL value fitting; this
is not a grid-cell comparison. (measured, abstract only)

**Wei, Prentice, Balasubramanian, "A principle of economy predicts the functional architecture of
grid cells"**, eLife 4:e08362 (2015), https://pmc.ncbi.nlm.nih.gov/articles/PMC4616244 (first
100k characters read through a summarising fetch).
- Abstract: a hierarchical code that "economizes the number of neurons required to encode
  location with a given resolution across a range equal to the largest period" predicts a
  triangular lattice, a geometric progression of scales, and an adjacent-scale ratio of √e
  (≈ 1.65) for idealised neurons, 1.4 to 1.7 for realistic ones.
- 1D: ratio e (≈ 2.72) with winner-take-all decoding, ≈ 2.3 with a probabilistic decoder. 2D:
  √e ≈ 1.65 (winner-take-all) and ≈ 1.44 (probabilistic). Within 5% of the minimum neuron count
  for ratios in (1.43, 1.96) (winner-take-all) or (1.28, 1.66) (probabilistic).
- The nested condition ("Intuitions from a simplified model"): "Avoiding ambiguity requires that
  λ_{i+1}, the period at scale i + 1, must exceed l_i, the grid field width at scale i": each
  finer module must resolve the coarser module's uncertainty.
- Data they compare to: Stensola et al. 2012, adjacent-scale ratio 1.42 ± 0.17 (n = 24); Barry
  et al. 2007, 1.64 ± 0.09 (n = 6). (theory, and measured biology)

**Stensola, Stensola, Solstad, Frøland, Moser, Moser, "The entorhinal grid map is discretized"**,
Nature 492, 72-78 (2012), doi 10.1038/nature11649. **Not read** (abstract and secondary sources
only: the arXiv survey 1810.07429 and Wei et al.). Reported: modules each with their own scale,
orientation and asymmetry; spacing ratio about 1.42 (√2, each hexagon's area doubling); at most
five modules recorded per animal, the total estimated in the upper single digits.

**Fiete, Burak, Brookings, "What grid cells convey about rat location"**, J. Neurosci. 28(27),
6858-6871 (2008), https://fietelab.mit.edu/wp-content/uploads/2018/12/rns_fieteetal08_jneurosci.pdf
(PDF read).
- Abstract: the code "is fully combinatorial in capacity. The resulting range for unambiguous
  position representation is vastly greater than the ~1-10 m periods of individual lattices",
  "with excess capacity that could be used for error correction".
- Eq. 2: for integer periods the range is the least common multiple. For real periods with phase
  uncertainty Δφ there is no closed form; capacity is found numerically and grows as
  (1/Δφ)^(N_eff − 1), N_eff ≈ 10.7 for N = 12 (Eq. 5, Figure 2). Figure 3: twelve lattices with
  periods 30 to 74 cm, Δφ = 0.2, cover about 2.2 km at 6.6 cm resolution. (theory, numerical)
- Note the regime: periods in a narrow band (30-74 cm), range from the residue arithmetic, not
  from a coarse module covering it.

**Sreenivasan and Fiete, "Grid cells generate an analog error-correcting code for singularly
precise neural computation"**, Nature Neuroscience 14, 1330-1337 (2011), doi 10.1038/nn.2901.
**Abstract only** (via PubMed and the publisher page): the error relative to the coding range
goes to zero at a non-zero information rate, and "a simple neural network can effectively correct
the grid code". (theory)

**Mathis, Herz, Stemmler, "Resolution of nested neuronal representations can be exponential in
the number of neurons"**, PRL 109, 018103 (2012). **Abstract only**: nested periodic codes resolve
a real-valued signal with resolution exponential in the number of noisy neurons; mirrors grid
cells. (theory)

Machine-learning practice, for scale: NeRF's positional encoding (Mildenhall et al., arXiv
2003.08934, Eq. 4, not re-read here) uses octaves (ratio 2); GTA uses octaves; axial RoPE in ViT
uses 100^(1/16) ≈ 1.33 per step at a 64-wide head (rotary-and-gate-check §1a). (default)

**Not found**: any RL paper with a fixed (not learned) grid-cell firing-map code as the input,
compared against Fourier features or raw coordinates.

### Grid cells against Fourier features

A grid module is a fixed function G of position on a hexagonal lattice; its phases are shifted
copies G(x − s). By Fourier expansion, the phase-shifted copies span the module's Fourier
components: the three first-harmonic plane waves (exactly lean's hexagonal Fourier code at that
period) plus the higher harmonics the nonlinear gain adds. So for a network whose first layer is
linear, a full set of grid phases carries the Fourier code plus harmonics and non-negativity; a
Fourier code carries the first harmonics alone. The literature has no measured advantage of one
over the other as an RL input (Space2Vec's small hexagonal edge is for sinusoids on three
directions, not firing maps). (inference; the absence of a test is a finding)

**A defect in lean's grid-cell code.** With `grid_phases = 2` the shifts are the half-lattice
points (p1, p2 ∈ {0, π}), and every such shift is its own negative. Since G is a sum of cosines
(even), each feature satisfies f_s(−x) = G(−x − s) = G(x + s) = G(x − s) = f_s(x): the whole code is
symmetric under x → −x, at every spacing at once. Checked numerically on the tree:
`GridCellCode(periods, 2)` gives max |f(x) − f(−x)| = 1.9e-5 on random positions (float32
rounding), against 2.35 between unrelated positions; with 3 phases the difference is 2.73; the
Fourier code gives 2.0. In lean's frame the input is pos + offset, so a ship at p and a ship at
−p − 2·offset receive identical position features. With 3 × 3 phases (or any phase set where no
shift is its own negative) the reflected position maps to a permutation of the features, which
the fixed feature slots tell apart. The present test ("a grid cell repeats over its lattice and
not over half of it") does not cover point reflection. (measured here, numerically)

### Design guidance, related to lean's numbers

Numbers: arena radius 4,200 m (diameter 8,400 m); top speed about 325 m/s at 30 ticks per second,
so one tick moves a ship at most about 10.8 m and two ships' separation at most about 21.7 m.
Lean now: Fourier and grid spacings 64 m to 20 km in 6 steps (ratio 3.16, `policy.py:23-24`);
rotary 64 m to 8,603 m in 5 bands (ratio 3.405, `policy.py:71-73`), two axes per band, 35° turn.

1. **Coarsest scale, input code.** The policy compares a ship's code with the game token's code
   (the arena centre, `policy.py:362`). A phase difference along a wave of period P is
   unambiguous for |projection| < P/2, and positions lie within 4,200 m of the centre, so
   P > 8,400 m. For a grid module, the plane-wave period is √3·L/2 for spacing L
   (`position.py:58`, |k| = 4π/(√3 L)), and the unit cell's inscribed radius is L/2, so
   L > 8,400 m. The 20 km coarsest of both codes satisfies this; the offset span is that coarsest
   period, so every phase is uniform. (inference, arithmetic)
2. **Coarsest band, rotary.** Ship-to-ship separations reach 8,400 m, in either sign along an
   axis, so one band alone is unambiguous only if P > 16,800 m. 8,603 m covers 4,300 m alone and
   relies on finer bands beyond that (NOTES choice 2). The author's first criterion ("the coarsest
   band unambiguous on the largest map") asks for a band above 16.8 km. (inference, arithmetic)
3. **Finest band.** The author's criterion "no aliasing from one tick's motion": a pairwise phase
   step below half a period per tick needs P > 2 × 21.7 ≈ 43 m. At 64 m the largest per-tick
   step is 122°, inside the limit. The resolution needed in close fights sets how fine it must be;
   that is a game question, not a literature one. (inference, arithmetic)
4. **Ratio between scales.** The nested condition (Wei et al.; Mathis et al.) is the operative
   one for a network that reads the scales together: the coarser scale's position error must be
   smaller than half the finer period. If a scale's phase is read with error ε (as a fraction of
   its period) and k standard deviations of margin are wanted, ε·λ_coarse·k < λ_fine/2, i.e.
   r < 1/(2kε). At k = 2: lean's 3.405 tolerates ε < 0.073 (26° of phase); r = 2 tolerates
   ε < 0.125 (45°); r = 1.65 tolerates ε < 0.15 (55°). The residue codes of Fiete et al. reach
   ranges beyond the coarsest period with narrow-band periods, but decoding them is non-local
   arithmetic; nothing suggests a policy network learns it, so keep the coarsest scale over the
   whole range and choose r by the nested condition. (inference; the condition is from the
   cited theory)
   - Biology and theory put r at 1.4 to 1.7 in 2D (measured 1.42 and 1.64; predicted √e or 1.44);
     ML codes use 1.33 to 2. Lean's 3.16 and 3.405 are coarser than all of them.
5. **What the ratio costs.**
   - Input code (cheap): from 64 m to about 10 km at r = 1.65 is 11 scales. Fourier: 3 directions
     × 2 × 11 = 66 features. Grid cells with 3 × 3 phases: 99 features. (arithmetic)
   - Rotary (each band two pairs; 30 position pairs available in a 64-wide head after 2 heading
     pairs): 64 m to 16.8 km needs 1 + ⌈ln(262.5)/ln r⌉ bands. At r = 2: 10 bands, 64 m to
     32.8 km, 20 + 2 = 22 pairs, 69% of the head (near Barbero et al.'s 75%, provenance §7). At
     r = 1.65: 13 bands, 28 pairs (88%). At r = 1.42: 17 bands, 34 pairs, which does not fit.
     (arithmetic)
6. **Orientation.** Biology: modules differ in orientation (Stensola, secondary sources). For a
   fixed code the aim is that no two scales share an axis, so their aliasing patterns do not
   line up. Hexagonal codes repeat every 60°, two-axis rotary bands every 90°. Steps of an
   irrational fraction of that period avoid realignment: 60°/φ ≈ 37.1° for hexagonal modules,
   90°/φ ≈ 55.6° (or lean's 35°) for rotary bands. (inference)
7. **Phases.** Biology: phases within a module cover the unit cell (Hafting et al. 2005, not read
   here). For the code: at least 3 × 3 phases per module, never the 2 × 2 half-lattice set (see
   the defect above). 3 × 3 also gives enough features (9 ≥ 7) to span the first harmonics and
   the mean linearly. (inference, and the numerical check above)
8. **Gain.** Blair's exponential gain is one choice; nothing in the sources compares it with a
   rectifier for a network input. (none found)

### Recommendation

- Keep the Fourier code as the default input. Grid cells only as an arm, and only with 3 × 3
  phases (the 2 × 2 code is reflection-symmetric). Strength: **weak**. No RL paper compares a
  fixed grid code with Fourier features; the learned grid code in Banino is a different thing.
- For both codes and the rotary: a geometric progression with ratio about 1.65 to 2, the finest
  near 64 m. For the input code, coarsest above 8.4 km (20 km is fine). For the rotary, coarsest
  above 16.8 km, e.g. 10 octave bands from 64 m to 32.8 km (22 pairs with heading, 69% of the
  head). Orientations stepped by an irrational fraction of the code's symmetry period. Strength:
  **weak to moderate**. The ratio rests on grid-cell theory with measured biology (1.42, 1.64) and
  on ML convention (octaves); the coarsest and finest limits are arithmetic from the author's
  criteria. No ML ablation of the ratio for 2D position codes was found.

---

## Could not verify

- Stensola et al. 2012 (Nature): full text not read; the ratio 1.42, module count and orientation
  statements come from Wei et al. 2015 and secondary sources.
- Sreenivasan and Fiete 2011 and Mathis, Herz, Stemmler 2012: abstracts only.
- Wei et al. 2015: read through a summarising fetch of the first 100k characters; the eLife PDF
  could not be downloaded.
- Lyu et al. 2021: abstract only. OpenAI Five App. O (the team-spirit ablation): only as
  summarised. MAPPO's last 18k characters were not read.
- Qiu et al. 2025, M3, IRIS: read through summarising fetches, not line by line.
- T-Fixup (Huang et al. 2020): not read; described from general knowledge as a Post-LN or
  norm-free scheme. NeRF's encoding and Hafting et al. 2005: cited from general knowledge, not
  re-read.
- No paper found that compares HL-Gauss with two-hot for world-model or auxiliary targets, that
  compares absolute, change and dead-reckoning targets for an RL auxiliary loss, that combines an
  output gate with rotated values, that compares a pooled team-token value with per-agent values
  for a shared reward, or that uses a fixed grid-cell firing-map code as an RL input.
