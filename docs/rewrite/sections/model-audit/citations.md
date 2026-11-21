# Citation check for the proposed baseline

Method. Every paper below was fetched as PDF from arxiv.org (`https://arxiv.org/pdf/<id>`) or reused from
`audit/papers/`, converted with `pdftotext`, and the relevant passage read. Line references ("L123") are to
`audit/papers/<id>.txt` so each quote can be re-found. Labels: **[FULL]** = the relevant passage was read in the
full text; **[ABSTRACT]** = only abstract-level text read. "Target setting" = centralised entity-attention policy,
PPO self-play, two-team air combat, fog of war, variable entity counts.

Things I could not read are stated as such; nothing here is filled in from memory unless explicitly flagged
"(recollection, NOT verified)".

---

## 1. PPO: clipped surrogate, multiple epochs of minibatches, clip 0.2; does the paper normalise advantages?

- Paper: Schulman, Wolski, Dhariwal, Radford, Klimov (2017), "Proximal Policy Optimization Algorithms", arXiv 1707.06347 (v2).
- Verdict: **CONFIRMED** (with one nuance on "default 0.2"). [FULL]
- Passages:
  - Eq. (7), Sec. 3: `L^CLIP(θ) = Ê_t[ min( r_t(θ) Â_t, clip(r_t(θ), 1−ε, 1+ε) Â_t ) ]`, "where epsilon is a hyperparameter, say, ε = 0.2."
  - Abstract: "a novel objective function that enables multiple epochs of minibatch updates"; Algorithm 1: "Optimize surrogate L wrt θ, with K epochs and minibatch size M ≤ NT".
  - Table 1 (Sec. 6.1, 7 MuJoCo tasks): clipping ε=0.1 → 0.76, ε=0.2 → **0.82** (best), ε=0.3 → 0.70.
  - Nuance: 0.2 is the MuJoCo/Roboschool choice. Atari (Table 5) uses clipping parameter **0.1 × α**, α annealed 1→0, 3 epochs, minibatch 32×8, VF coeff 1, entropy coeff 0.01. MuJoCo (Table 3): 10 epochs, minibatch 64, γ 0.99, λ 0.95.
  - Advantage normalisation: **not in the paper.** The only occurrences of "normal" in the whole text are "avg. normalized score" in Table 1 and its caption. The value loss in the paper is the plain squared error `(V_θ(s_t) − V_t^targ)^2` (Eq. 9), so value-loss clipping is not in the paper either (Andrychowicz et al. say the same: "present in the original PPO implementation but not mentioned in the PPO paper").
- So the belief is correct: per-minibatch advantage normalisation is an implementation convention with no support in 1707.06347.
- Caveat for transfer: experiments are single-agent MuJoCo/Roboschool/Atari with small MLP/CNN policies; no self-play, no attention, no recurrent policy on the benchmark tasks.

## 2. What Andrychowicz et al. and Engstrom et al. actually conclude

### 2A. Andrychowicz et al. 2020

- Paper: Andrychowicz, Raichuk, Stańczyk, Orsini, Girgin, Marinier, Hussenot, Geist, Pietquin, Michalski, Gelly, Bachem (2020), "What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study", arXiv 2006.05990.
- Verdict: **CONFIRMED as a source for (a)-(e)**, with the stated recommendations below. [FULL] (Sec. 3.1-3.5 and App. B.1/B.2 read.)
- (a) Per-minibatch advantage normalisation (choice C67), Sec. 3.3: "per-minibatch advantage normalization (C67) seems not to affect the performance too much (Fig. 35)". It is **not** in their recommendation, and their default is `False` (Table, C67). Definition (App. B): "We normalize advantages in each minibatch by subtracting their mean and dividing by their standard deviation for the policy loss."
- (b) Value-loss clipping (C13), Sec. 3.4: "PPO-style value loss clipping (C13) hurts the performance regardless of the clipping threshold (Fig. 43)", footnote "This is consistent with prior work [27]". Recommendation: "Use GAE with λ = 0.9 but neither Huber loss nor PPO-style value loss clipping."
- (c) Observation normalisation (C64), Sec. 3.3: "Input normalization (C64) is crucial for good performance on all environments apart from Hopper (Fig. 33)." Recommendation: "Always use observation normalization and check if value function normalization improves performance. Gradient clipping might slightly help but is of secondary importance." (Value-function normalisation helped strongly on HalfCheetah and Humanoid and "significantly hurts the performance on Walker2d".)
- (d) Number of epochs (C3, swept over {1, 3, 10}), Sec. 3.5: "going over the experience multiple times appears to be crucial for good sample complexity (Fig. 54)". Recommendation: "Go over experience multiple times." No specific count is recommended; their default config uses 10.
- (e) Recomputing advantages, Sec. 3.5: "we propose to recompute the advantages at the beginning of each pass over the data instead of just once per iteration ... stale advantages can in fact hurt performance and that recomputing them at the beginning of each pass at least partially mitigates the problem and performs best among all variants." Recommendation: "Shuffle individual transitions before assigning them to minibatches and recompute advantages once per data pass (See App. B.1 for the details)."
- Also relevant: "Use the PPO policy loss. Start with the clipping threshold set to 0.25 but also try lower and higher values if possible." (Sec. 3.1)
- Caveat for transfer: five MuJoCo continuous-control tasks, MLP policies, no recurrence, no self-play, dense reward. Epoch count in particular does not transfer cleanly (see MAPPO, claim 10, and OpenAI Five, claim 12e, which point the other way).

### 2B. Engstrom et al. 2020

- Paper: Engstrom, Ilyas, Santurkar, Tsipras, Janoos, Rudolph, Madry (ICLR 2020), "Implementation Matters in Deep Policy Gradients: A Case Study on PPO and TRPO", arXiv 2005.12729.
- Verdict: **PARTLY** — it supports far less of (a)-(e) than it is usually cited for. [FULL] (Sec. 3, Fig. 1 caption, Sec. 5 and hyperparameter appendix read.)
- What it does: lists nine "code-level optimizations" (Sec. 3): value function clipping, reward scaling, orthogonal init and layer scaling, Adam LR annealing, reward clipping, observation normalisation, observation clipping, tanh activations, global gradient clipping. It runs a **full ablation only on the first four** (Fig. 1: "value clipping, reward scaling, network initialization, and learning rate annealing", Humanoid-v2 and Walker2d-v2).
- (a) Per-minibatch advantage normalisation: **not studied and not in their list.** NOT SUPPORTED as a source for this.
- (b) Value-function clipping: ablated, and conspicuously absent from the list of things that mattered. Fig. 1 caption: "reward normalization, Adam annealing, and network initialization each significantly impact the rewards landscape with respect to hyperparameters, and were necessary for attaining the highest PPO reward within the tested hyperparameter grid." The paper makes no explicit statement that value clipping hurts; the reading "value clipping did not matter" is an inference from omission (it is how Andrychowicz et al. cite it).
- (c) Observation normalisation: listed as optimisation #6 but **not ablated**. Not a source for its benefit.
- (d) Epochs: not studied (fixed at 10 policy epochs / 10 value epochs in the appendix tables).
- (e) Recomputing advantages: not studied.
- Headline conclusion (Sec. 5, Table 2-3): code-level optimisations account for most of PPO's gain over TRPO; "the clipping mechanism is not necessary to achieve high performance" (PPO-NoClip with the optimisations matches PPO).
- Caveat for transfer: MuJoCo only (Humanoid, Walker2d, Hopper), MLP policies; gives no recommendations list, only an attribution argument.

## 3. GAE: estimator and recommended λ range

- Paper: Schulman, Moritz, Levine, Jordan, Abbeel (ICLR 2016), "High-Dimensional Continuous Control Using Generalized Advantage Estimation", arXiv 1506.02438.
- Verdict: **CONFIRMED.** [FULL]
- Estimator, Eq. (16): `Â_t^GAE(γ,λ) = Σ_{l=0}^∞ (γλ)^l δ^V_{t+l}`, with `δ^V_t = r_t + γV(s_{t+1}) − V(s_t)`. Special cases Eq. (17)-(18): λ=0 gives the one-step TD residual, λ=1 gives `Σ γ^l r_{t+l} − V(s_t)`.
- λ range: Sec. 6.2 (cart-pole) "the best results are obtained at intermediate values of the parameters: γ ∈ [0.96, 0.99] and λ ∈ [0.92, 0.99]"; Fig. 2 caption "fastest policy improvement is obtain by intermediate values of λ in the range [0.92, 0.98]"; Sec. 7 discussion: "choosing an appropriate intermediate value of λ in the range **[0.9, 0.99]** usually results in the best performance." Sec. 3: "Empirically, we find that the best value of λ is much lower than the best value of γ".
- Caveat for transfer: validated with TRPO on simulated locomotion; the λ range is an empirical locomotion result, not a general theorem.

## 4. Griffin / Hawk RG-LRU: equations, c, Λ initialisation; comparison with the project's init

- Paper: De, Smith, Fernando, Botev, Cristian-Muraru, Gu, Haroun, Berrada, Chen, Srinivasan, Desjardins, Doucet, Budden, Teh, Pascanu, De Freitas, Gulcehre (2024), "Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models", arXiv 2402.19427.
- Verdict on the recollection: **CONFIRMED** (c = 8; init such that a^c ~ U[0.9, 0.999]). Verdict on the project implementation matching the paper: **NOT SUPPORTED** — the project's initialisation is far outside the paper's range. [FULL]
- Equations, Sec. 2.4, Eq. (1)-(4):
  - `r_t = σ(W_a x_t + b_a)` (recurrence gate)
  - `i_t = σ(W_x x_t + b_x)` (input gate)
  - `a_t = a^(c·r_t)`
  - `h_t = a_t ⊙ h_{t−1} + sqrt(1 − a_t²) ⊙ (i_t ⊙ x_t)`
  - "We parameterize a in Equation (3) as a = σ(Λ), where Λ is a learnable parameter. ... The variable c is a scalar-valued constant set to 8."
- Initialisation, quoted: "We initialize both W_a and W_x using LeCun init (LeCun et al., 2002). We initialize Λ such that a^c is uniformly distributed between 0.9 and 0.999 at the start of training, similar to Orvieto et al. (2023b)."
- App. A, Eq. (6): `log a_t = log a^(c r_t) = log σ(Λ) · c r_t = −c softplus(Λ) ⊙ r_t`. Note: with a = σ(Λ), log σ(Λ) = −softplus(−Λ), so Eq. (6) as printed has a sign-convention inconsistency with the main text. The project's `exp(8 · r · logsigmoid(logit))` matches the main-text definition a = σ(Λ) exactly.
- Computed comparison (decay per step d = exp(8·r·logsigmoid(logit)); time constant τ = −1/ln d; half-life = ln 0.5 / ln d):

| logit Λ | r | per-step decay d | τ (steps) | half-life (steps) |
|---|---|---|---|---|
| 0 | 1.0 | 0.0039 | 0.18 | 0.12 |
| 0 | 0.5 | 0.0625 | 0.36 | 0.25 |
| 2 | 1.0 | 0.3622 | 0.98 | 0.68 |
| 2 | 0.5 | 0.6019 | 1.97 | 1.37 |
| 4 | 1.0 | 0.8648 | 6.89 | 4.77 |
| 4 | 0.5 | 0.9300 | 13.8 | 9.55 |
| paper low end (a^c = 0.9, Λ ≈ 4.32) | 1.0 | 0.9000 | 9.49 | 6.58 |
| paper low end | 0.5 | 0.9487 | 19.0 | 13.2 |
| paper high end (a^c = 0.999, Λ ≈ 8.99) | 1.0 | 0.9990 | 999.5 | 692.8 |
| paper high end | 0.5 | 0.9995 | 1999 | 1386 |

  - The paper's init corresponds to Λ ∈ [≈4.32, ≈8.99]. The project's `linspace(0, 4)` lies **entirely below** the paper's shortest-memory channel: at r = 1 the project's longest-memory channel (logit 4, d = 0.865, τ ≈ 6.9 steps) forgets faster than the paper's shortest (d = 0.9, τ ≈ 9.5 steps), and half the project's channels (logit ≤ 2) retain under 37% per step, i.e. effectively no memory. The paper's range spans τ ≈ 10 to 1000 steps (20 to 2000 at r = 0.5); the project spans τ ≈ 0.2 to 7 (0.4 to 14 at r = 0.5).
  - Also note the distribution shape differs: the paper is uniform in a^c; `linspace` in logit space is not.
- Caveat for transfer: Griffin's init was chosen for language modelling with sequence lengths in the thousands; the right range for an RL recurrent core depends on the env step rate and episode length, so the paper's numbers are provenance for the *form*, not proof the same range is optimal here.

## 5. Rotary position embedding; rotary on continuous 2D/ND coordinates

- Papers fetched and read:
  - Su, Lu, Pan, Murtadha, Wen, Liu (2021), "RoFormer: Enhanced Transformer with Rotary Position Embedding", arXiv 2104.09864.
  - Heo, Park, Han, Yun (2024), "Rotary Position Embedding for Vision Transformer", arXiv 2403.13298 (v2).
  - Ostmeier, Axelrod, Varma, Moseley, Chaudhari, Langlotz (2024), "LieRE: Lie Rotational Positional Encodings", arXiv 2406.10322.
  - Schenck, Reid, Jacob, Bewley, Ainslie, et al. (2025), "Learning the RoPEs: Better 2D and 3D Position Encodings with STRING", arXiv 2502.02562.
  - Miyato, Jaeger, Welling, Geiger (ICLR 2024), "GTA: A Geometry-Aware Attention Mechanism for Multi-View Transformers", arXiv 2310.10375.
- Verdict: **PARTLY.** RoPE itself confirmed; published rotary encodings on multi-dimensional and real-valued coordinates exist; **none found in RL entity transformers, and none found using metric world coordinates of freely moving entities.** [FULL] for the five papers above; the "none found in RL" statement rests on a limited arXiv API search, not an exhaustive review.
- What RoPE is (RoFormer abstract + Sec. 3): "the proposed RoPE encodes the absolute position with a rotation matrix and meanwhile incorporates the explicit relative position dependency in self-attention formulation" — queries and keys are rotated by angle (position × θ_i) in 2D sub-planes, so the q·k inner product depends only on relative position. Positions are integer token indices; frequencies fixed θ_i = 10000^(−2i/d).
- Heo et al. (Sec. 3.2): 2D position `p_n = (p^x_n, p^y_n)` with `p^x_n ∈ {0,1,...,W}, p^y_n ∈ {0,1,...,H}` — **integer patch-grid indices**, not real-valued coordinates. Two variants: axial (`R(n,2t) = e^{iθ_t p^x_n}`, `R(n,2t+1) = e^{iθ_t p^y_n}`) and "RoPE-Mixed", Eq. (14) `R(n,t) = e^{i(θ^x_t p^x_n + θ^y_t p^y_n)}` with **learnable** per-head, per-layer frequencies, motivated by axial frequencies being unable to represent diagonal directions. Result: better multi-resolution (extrapolation) performance on ImageNet/COCO/ADE20k.
- LieRE (Sec. 3): "When encoding positions p ∈ R^n, LieRE learns a skew-symmetric basis of matrices {A_i} ... R(p) = exp(Σ p_i A_i)". Formulated for real vectors, described as "continuous" encodings, but evaluated on 2D/3D image and video patch positions (grid positions, "scaled proportionally to the image dimensions").
- STRING (abstract, Sec. 3): generalises RoPE to "token coordinates of arbitrary dimensionality" with exact translation invariance `R(r_i)^T R(r_j) = R(r_j − r_i) ∀ r_i, r_j ∈ R^{d_c}`, via commuting skew-symmetric generators. It **does use real-valued coordinates**: per-patch depth ("we incorporated depth as a third coordinate for 3D position representations"; "compute the height for each patch via mean-pooling across depth values"; metric mono-depth from Depth-Anything-V2), and it explicitly notes that positions "differ across training examples and batches, which occurs e.g for point cloud data or color plus depth channel (RGB-D) images". Applications: open-vocabulary 2D/3D detection and robot manipulation policies (imitation-style controllers, not PPO).
- GTA: related but different mechanism — applies group representations of relative camera transforms (SE(3) × SO(2) × SO(2)) to attention for novel-view synthesis; not RoPE on entity coordinates.
- Answer to "does any use real-valued metric coordinates?": STRING, partly (x, y are still patch-grid positions; only the depth axis is a real-valued, metric quantity). LieRE in formulation only. I found no paper that rotates by continuous 2D world positions of a variable set of entities, and none in an RL entity-attention policy. An arXiv listing search also surfaced titles I did **not** read (e.g. 2606.12146 "nD-RoPE", 2604.24169 "PointTransformerX", 2509.22259 "Rotary Position Encodings for Graphs"); they are unverified leads, not support.
- Caveat for transfer: all evidence is for dense, bounded, grid-like positions in supervised vision. Two things are unverified for the target: (i) frequency scale selection for unbounded/large metric ranges (Heo et al. had to learn frequencies; Gorishniy and Tancik both report frequency scale as the critical hyperparameter); (ii) pure-translation invariance is the property obtained — rotation/heading invariance, which matters in air combat, is not provided by RoPE.

## 6. Fourier features for coordinate inputs

- Paper: Tancik, Srinivasan, Mildenhall, Fridovich-Keil, Raghavan, Singhal, Ramamoorthi, Barron, Ng (NeurIPS 2020), "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains", arXiv 2006.10739.
- Verdict: **CONFIRMED.** [FULL]
- Abstract: "passing input points through a simple Fourier feature mapping enables a multilayer perceptron (MLP) to learn high-frequency functions in low-dimensional problem domains ... Using tools from the neural tangent kernel (NTK) literature, we show that a standard MLP fails to learn high frequencies both in theory and in practice. To overcome this spectral bias, we use a Fourier feature mapping to transform the effective NTK into a stationary kernel with a tunable bandwidth."
- Mapping (Sec. 6): Gaussian `γ(v) = [cos(2πBv), sin(2πBv)]`, B_ij ~ N(0, σ²); also "Basic" `[cos 2πv, sin 2πv]` and log-linear "positional encoding". Finding: "the Gaussian RFF mapping performs best"; Sec. 5: performance depends on the scale σ, "exhibiting underfitting (slow convergence) when the standard deviation of sampled frequencies is too low and overfitting when it is too high", while the distribution's shape matters less than its scale.
- Caveat for transfer: the result is about fitting a fixed target function from coordinates (image/shape/NeRF regression), where the *output* must vary at high frequency in the coordinate. In an RL policy, coordinates are inputs to a decision function whose required frequency content is unknown, the target is non-stationary, and σ must be tuned — the paper warns too-high σ overfits. No RL experiment in the paper.

## 7. Classification for value functions (HL-Gauss)

- Papers: Farebrother, Orbay, Vuong, Taïga, Chebotar, Xiao, Irpan, Levine, Castro, Faust, Kumar, Agarwal (2024), "Stop Regressing: Training Value Functions via Classification for Scalable Deep RL", arXiv 2403.03950. Imani & White (ICML 2018), "Improving Regression Performance with Distributional Losses", arXiv 1806.04613.
- Verdict: **PARTLY.** HL-Gauss description and headline results confirmed; **there is no PPO experiment** — "works with on-policy PPO" is not supported by this paper. [FULL] for Farebrother; [ABSTRACT] plus Sec. 3 headings for Imani & White.
- What HL-Gauss is (Sec. 3.1): project a Gaussian centred on the scalar TD target onto a fixed histogram; `p_i = F(z_i + ς/2) − F(z_i − ς/2)` for bins of width `ς = (v_max − v_min)/m`, with `Y ~ N(μ = target, σ²)`, then train with categorical cross-entropy. "akin to label smoothing". Tuning advice: "we recommend tuning σ/ς ... Unless specified otherwise, we set σ/ς = 0.75 for our experiments, which distributes mass to approximately 6 locations." Requires choosing [v_min, v_max].
- Compared against: MSE regression, Two-Hot (Schrittwieser et al. 2020 style), and C51 (categorical distributional RL).
- Results: Sec. 4.1 online DQN on 60 Atari games, 200M frames: "HL-Gauss substantially outperforms the Two-Hot and MSE losses ... also improves upon categorical distributional RL (C51)"; Fig. 4 caption: "Two-Hot loss underperforms MSE but is more stable" (offline). Headline (Sec. 1): "30% better performance when scaling parameters with Mixture-of-Experts in single-task RL on Atari; 1.8−2.1× performance in multi-task setups on Atari; 40% better performance in the language-agent task of Wordle; 70% improvement for playing chess without search; and 67% better performance on large-scale robotic manipulation with transformers".
- On-policy / actor-critic coverage: the text never mentions PPO (zero matches for "PPO", "proximal", "on-policy" as an algorithm). The only actor-critic experiment is **IMPALA** (V-trace, distributed, slightly off-policy) for multi-task Atari variants (Sec. 4.2.2: "We employ a distributed actor-critic method, IMPALA ... and compare the standard MSE critic loss with the cross-entropy based HL-Gauss loss"; HL-Gauss "consistently outperforms MSE" and scales better with ResNet size). Everything else is value-based (DQN, CQL, Q-transformer) or supervised distillation (chess).
- Important for the project: in this paper **two-hot is the weakest classification variant** and underperforms MSE in the online DQN setting; so this paper is evidence for HL-Gauss, and mild evidence *against* plain two-hot.
- Imani & White 2018: origin of the histogram loss with a (truncated) Gaussian target, "HL-Gaussian"; abstract: "We introduce a novel distributional regression loss, and similarly find it significantly improves prediction accuracy ... we instead find evidence for an alternative hypothesis: this loss is easier to optimize, with better behaved gradients". Supervised regression only, not RL.
- Caveat for transfer: no PPO, no self-play, no non-stationary zero-sum returns; nearest evidence is IMPALA on multi-task Atari. Bounded support must be set, which interacts with any return normalisation.

## 8. DreamerV3: symlog two-hot, return normalisation, entropy scale

- Paper: Hafner, Pasukonis, Ba, Lillicrap, "Mastering Diverse Domains through World Models", arXiv 2301.04104. **Two versions differ materially**; both were read: v2 (17 Apr 2024, the current arXiv default) and v1 (Jan 2023).
- Verdict: **PARTLY** — the numbers are right, but (a) the bin count 255 is only stated in v1, and (b) what is divided differs between versions; the common shorthand "DreamerV3 divides returns by max(1, S)" is v1, the current version scales the advantage-like term. [FULL]
- (a) Bins and range.
  - v2, Eq. (10): `ŷ = softmax(f(x))^T B`, `B = symexp([−20 ... +20])`; "the network outputs the logits for a softmax distribution over exponentially spaced bins b_i ∈ B". Loss Eq. (11): `L(θ) = −twohot(y)^T log softmax(f(x, θ))`. **The number of bins is not stated anywhere in the v2 text** (no occurrence of 255).
  - v1, Sec. "Critic learning": "we transform returns using the symlog function and discretize the resulting range into a sequence B of **K = 255** equally spaced buckets", Eq. (8) `B = [−20 ... +20]`.
  - So: range [−20, 20] in symlog space in both; 255 is CONFIRMED for v1 only, COULD NOT CHECK for v2 from the paper (it would need the source code).
  - Also: "The output weights of twohot distributions are initialized to zero".
- (b) Return normalisation.
  - v2, Eq. (6): `L(θ) = −Σ_t sg( (R^λ_t − v_ψ(s_t)) / max(1, S) ) log π_θ(a_t|s_t) + η H[π_θ(a_t|s_t)]`
  - v2, Eq. (7): `S = EMA( Per(R^λ_t, 95) − Per(R^λ_t, 5), 0.99 )`
  - Text: "we normalize returns to be approximately contained in the interval [0, 1]. In practice, substracting an offset from the returns does not change the actor gradient and thus dividing by the range S is sufficient. Moreover, to avoid amplifying noise from function approximation under sparse rewards, we only scale down large return magnitudes but leave small returns below the threshold of L = 1 untouched." and "Previous work typically normalizes advantages rather than returns".
  - So in v2 the quantity divided is the **baseline-subtracted return (R^λ − v), i.e. an advantage**, but the scale S is a statistic of the **returns** (5th-95th percentile range, EMA decay 0.99, floor max(1, S)). This is deliberately *not* advantage standardisation: it never scales small signals up.
  - v1, Eq. (11)-(12): `L(θ) = Σ_t E[ sg(R^λ_t)/max(1, S) ] − η H[π]`, `S = Per(R^λ_t, 95) − Per(R^λ_t, 5)`, described as "an exponentially decaying average of the range from their 5th to their 95th batch percentile"; Table: "Return normalization decay 0.99". Here the **returns** are divided.
  - Hyperparameter table (v2, Table 4): "Actor RetNorm scale S = Per(R, 95) − Per(R, 5); Actor RetNorm limit L = 1; Actor RetNorm decay 0.99".
- (c) Entropy scale: v2 "To use a fixed entropy scale of η = 3 × 10^−4 across domains"; Table 4 "Actor entropy regularizer η 3 × 10^−4". CONFIRMED. The paper is explicit that this fixed η works *because of* the return normalisation; the two are a package.
- Caveat for transfer: model-based agent, actor trained on imagined rollouts (horizon 15) with Reinforce and λ-returns, not PPO; single-agent, no self-play. The fixed η = 3e-4 is tuned jointly with percentile return scaling and a 1% uniform-mix on the actor; lifting η alone into a PPO loss with standardised advantages has no support here.

## 9. RND: two value heads, different discounts, summed advantages

- Paper: Burda, Edwards, Storkey, Klimov (2018), "Exploration by Random Network Distillation", arXiv 1810.12894.
- Verdict: **CONFIRMED.** [FULL]
- Sec. 2.3: "we can fit two value heads V_E and V_I separately using their respective returns, and combine them to give the value function V = V_E + V_I. This same idea can also be used to combine reward streams with different discount factors."
- Algorithm 1 (App. A.1): "Calculate returns R_{I,i} and advantages A_{I,i} for intrinsic reward; Calculate returns R_{E,i} and advantages A_{E,i} for extrinsic reward; Calculate combined advantages A_i = A_{I,i} + A_{E,i}". Hyperparameter table: "Coefficient of extrinsic reward 2, Coefficient of intrinsic reward 1", λ = 0.95, rollout length 128 — so the sum is a weighted sum (2·A_E + 1·A_I).
- Discounts: Sec. 3.3-3.4 "increasing γ_E to 0.999 while holding γ_I at 0.99 greatly improves performance ... further increasing γ_I to 0.999 hurts performance"; final: "the intrinsic rewards are non-episodic with **γ_I = 0.99**, and **γ_E = 0.999**". Intrinsic stream is non-episodic, extrinsic episodic.
- Nuance: Sec. 3.2 "Contrary to our expectations using two value heads did not show any benefit over a single head in the episodic setting. Nevertheless having two value heads is necessary for combining reward streams with different characteristics". So the paper supports two heads as an enabler for mixing discounts/episodicity, not as a gain in itself.
- Caveat for transfer: single-agent Atari hard-exploration; the second stream there is a non-stationary intrinsic bonus, normalised by a running std of intrinsic returns. Using the construction for two *extrinsic* reward components in self-play is an extension the paper only gestures at.

## 10. MAPPO recommendations

- Paper: Yu, Velu, Vinitsky, Gao, Wang, Bayen, Wu (2021/2022), "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games", arXiv 2103.01955.
- Verdict: **PARTLY** — the suggestions are confirmed, but the value-normalisation suggestion as worded does **not** name PopArt. [FULL]
- Exact wording: "**Suggestion 1: Utilize value normalization to stabilize value learning.**" Sec. 5.1 describes it as: "we standardize the targets of the value function by using running estimates of the average and standard deviation of the value targets. Concretely, during value learning, the value network regresses to normalized target values. When computing the GAE, we use the running average to denormalize the output of the value network ... We find that using value normalization never hurts training and often improves the final performance of MAPPO significantly."
  - "PopArt" appears only in Algorithm 1 (appendix pseudocode): "Compute advantage estimate Â via GAE on τ, using PopArt; Compute reward-to-go R̂ on τ and normalize with PopArt". The main-text mechanism described is running mean/std target standardisation; the output-preserving ("Pop") weight correction is not described. Citing MAPPO for "use PopArt" is therefore loose; citing it for "normalise value targets with running statistics" is exact.
- Epochs and minibatching: "**Suggestion 3: Use at most 10 training epochs on difficult environments and 15 training epochs on easy environments. Additionally, avoid splitting data into mini-batches.**" Evidence: "when using 4 mini-batches, MAPPO fails to solve any of the selected maps while using 1 mini-batch produces the best performance on 22/23 maps"; "We hypothesize that this pattern could be a consequence of non-stationarity in MARL: using fewer epochs per update limits the change in the agents' policies".
- Other suggestions: "Suggestion 2: When available, include both local, agent-specific features and global features in the value function input." "Suggestion 4: For the best PPO performance, maintain a clipping ratio ε under 0.2; within this range, tune ε as a trade-off between training stability and fast convergence." "Suggestion 5: Utilize a large batch size to achieve best task performance with MAPPO."
- Parameter sharing (Sec. 3.3): "In benchmark environments with homogeneous agents ... we utilize parameter sharing; past works have shown that this improves the efficiency of learning, which is also consistent with our empirical findings. In these settings, agents share both the policy and value function parameters." App. C.2: MAPPO with sharing "outperforms MAPPO without parameter sharing by a clear margin". It is an implementation choice with an ablation, not one of the five numbered suggestions.
- Caveat for transfer: cooperative games only (MPE, SMAC, Hanabi, GRF) against scripted or built-in opponents; decentralised per-agent policies with a centralised value function; small MLP/RNN networks. No self-play and no competitive non-stationarity; a centralised policy over all own entities is a different factorisation.

## 11. PopArt

- Papers: van Hasselt, Guez, Hessel, Mnih, Silver (2016), "Learning values across many orders of magnitude", arXiv 1602.07714. Hessel, Soyer, Espeholt, Czarnecki, Schmitt, van Hasselt (2018), "Multi-task Deep Reinforcement Learning with PopArt", arXiv 1809.04474.
- Verdict: **CONFIRMED.** [FULL] for 1602.07714 Sec. 1-2; [ABSTRACT] plus section headings for 1809.04474.
- Problem (1602.07714 abstract): "Most learning algorithms are not invariant to the scale of the function that is being approximated. We propose to adaptively normalize the targets used in learning. This is useful in value-based reinforcement learning, where the magnitude of appropriate value approximations can change over time when we update the policy of behavior." Motivation is removing DQN's reward clipping to [−1, 1], which "can result in qualitatively different behavior".
- How (Sec. 2): learn a normalised function g with targets `Ỹ = Σ^{-1}(Y − μ)`, where μ, Σ are adapted online ("Adaptively Rescaling Targets"), and simultaneously "Preserve Outputs Precisely": when (Σ, μ) → (Σ_new, μ_new), set `W_new = Σ_new^{-1} Σ W` and `b_new = Σ_new^{-1}(Σ b + μ − μ_new)` (scalar case `W_new = (σ/σ_new) W`) so the *unnormalised* output is unchanged for all inputs.
- Multi-task (1809.04474 abstract): in multi-task learning agents "can get distracted by certain tasks ... because of the density or magnitude of the in-task rewards"; the paper derives "scale invariant updates" for the actor-critic IMPALA agent with per-task normalisation statistics (section "Scale invariant updates for multi-task learning").
- Caveat for transfer: original is value-based DQN on Atari; the multi-task version is IMPALA across 57 Atari games/DMLab-30. In self-play the return scale drifts with opponent strength rather than across tasks; the mechanism applies but the specific gains are not evidence for that regime.

## 12. OpenAI Five

- Paper: Berner, Brockman, Chan, Cheung, Dębiak, Dennison, Farhi, Fischer, Hashme, Hesse, Józefowicz, Gray, Olsson, Pachocki, Petrov, Pinto, Raiman, Salimans, Schlatter, Schneider, Sidor, Sutskever, Tang, Wolski, Zhang (2019), "Dota 2 with Large Scale Deep Reinforcement Learning", arXiv 1912.06680.
- Verdict: **CONFIRMED on all six, with details that differ from common paraphrase** on (a) and (f). [FULL] (Sec. 2-3, App. C, E, G, H, M, O; Figure 17 read as a rendered page.)
- (a) Fogged enemies, App. E: "Units in fog of war are not observed. When enemy heroes are in fog of war, we reuse the observation from the last time step when the unit was visible." Slots are fixed-count per category ("189 units: heroes (5), creeps (30), buildings (21), wards (30), and courier (1) for each team, plus 15 neutrals"); "If the number of visible units in a category is less than the allotted number, the rest are padded with zeroes. If more, we observe only the units closest to allied heroes." So: non-hero fogged units are simply absent (zero padding); fogged enemy **heroes** carry a stale last-seen observation. The paper does not describe attention masking — it uses pooling over zero-padded slots.
- (b) Team spirit, App. G, Eq. (12): `r_i = (1 − τ) ρ_i + τ ρ̄`, "with scalar ρ̄ being equal to mean of ρ". "Ultimately we care about optimizing for team spirit τ = 1 ... However we find that lower team spirit reduces gradient variance in early training". Schedule (Table, App. C): OpenAI Five 0.3 → 1.0; Rerun 0.3 → 0.8 (planned to reach 1.0 but stopped earlier); changes "applied gradually over the course of 1-2 days". No formula for the schedule is given — it was hand-adjusted during training.
- (c) Zero-sum and normalisation, App. G: "We ensure that all our rewards are zero-sum, by subtracting from each hero's reward the average of the enemies' rewards." Table footnote c: "We normalize rewards using a running estimate of the standard deviation, and the value loss weight is applied post-normalization."
- (d) Pooling, Figure 17a (diagram, read as image): "Process Set — Summarizes an unordered set of N elements into a vector of size S": N×K → **2x FC → max-pool**, with the pre-pool N×S "Embedding output" kept as per-unit "Unit Embeddings" used by target-selection action heads. So yes, max-pool over unit embeddings. Separately, Fig. 17b: a "cross-hero pool" in which "we maxpool the first 25% of the vector across the five replica networks".
- (e) PPO settings (App. C table, OpenAI Five column): GAE λ = 0.95; GAE horizon 60 s → 840 s (Rerun 180 → 360 s); PPO clipping 0.2; value loss weight 0.25 ↔ 1.0 (Rerun 1.0); entropy coefficient 0.01 → 0.001; learning rate 5e-5 ↔ 1e-6; Adam β1 0.9, β2 0.999; LSTM unroll 16; 20% of games against past opponents; sample reuse 0.8 ↔ 2.7 (Rerun 1.0 ↔ 1.1). On reuse (Sec. 4.2): "reusing the same data even 2-3 times can cause a factor of two slowdown, and reusing it 8 times may prevent the learning of a competent policy altogether. Our final system targets sample reuse ∼ 1 in all our experiments." App. M.3 adds that sample reuse below 1 "can be beneficial".
- (f) LSTM per hero: yes, as weight-shared replicas. Sec. 2 / Fig. 1: "Separate replicas of the same policy function (with identical parameters θ) are used to control each of the five heroes"; "Each of the five heroes on the team is controlled by a replica of this network with nearly identical inputs, each with its own hidden state"; single-layer 4096-unit LSTM, "84% of the model's total parameter count". The replicas are coupled only through shared observations and the cross-hero max-pool. Value and policy share the network ("share a network and share gradients").
- Caveat for transfer: fixed unit budget with zero padding and pooling (no attention, no entity mask); one LSTM replica per controlled hero rather than one centralised policy; batch sizes of ~1-3M timesteps make "sample reuse ≈ 1" affordable — at small scale the same setting costs sample efficiency, so the reuse finding and Andrychowicz's "multiple passes" recommendation pull in opposite directions and neither is in the target regime.

## 13. Registers as support for a learned always-present "null" key

- Paper: Darcet, Oquab, Mairal, Bojanowski (ICLR 2024), "Vision Transformers Need Registers", arXiv 2309.16588.
- Verdict: paper's finding **CONFIRMED**; as support for the project's null-key design it is **PARTLY** — an analogy, not evidence. [FULL] (Sec. 1-2 and the register-count ablation read.)
- Finding (abstract): "we identify and characterize artifacts in feature maps of both supervised and self-supervised ViT networks. The artifacts correspond to high-norm tokens appearing during inference primarily in low-informative background areas of images, that are repurposed for internal computations. We propose a simple yet effective solution based on providing additional tokens to the input sequence of the Vision Transformer to fill that role." Details: outliers are "around 2%" of tokens; appear only in sufficiently large, sufficiently trained models ("only appear after one third of training", "only the three largest models exhibit outliers"); "High-norm tokens appear where patch information is redundant", "hold little local information", but hold global information (linear probing on outlier tokens beats normal patches). Fig. 8: "While one register is sufficient to remove artefacts, using more leads to improved downstream performance."
- The gap, stated plainly:
  1. Registers are extra **tokens that are themselves updated through every layer** (they attend and are attended to; they act as writable scratch memory). A single learned "null" **key** that can only be attended *to* is a different object: it gives attention somewhere to put probability mass (a no-op sink), but it cannot store or broadcast global computation. The paper's mechanism (global aggregation in repurposed tokens) is not what a static null key provides.
  2. The paper's problem is repurposing of **real, valid, low-information input tokens**. It says nothing about masked/absent placeholders. If absent entities are properly masked out of attention they cannot be repurposed at all, and the register argument does not arise; if they are unmasked zero vectors, the relevant fix is masking (claim 20), not registers.
  3. The demonstrated harm is to dense feature-map quality (segmentation, object discovery) in large ViTs (ViT-L and up, late in training). Classification accuracy was largely unaffected. No RL, no small models, no set inputs of 10-100 entities.
  4. The stronger, more direct motivation for a null key is the standard softmax constraint that weights must sum to one over the unmasked set, including the degenerate case where a query has *no* valid key (all masked → NaN). That is an engineering argument, not something Darcet et al. show.
- Reasonable phrasing: "by analogy with registers (Darcet et al.), which show transformers benefit from dedicated content-free tokens" is defensible; "Darcet et al. show a null key is better than letting the model use absent-entity placeholders" is not supported.

## 14. Hsu, Jeong, et al.: k-nearest masking

- Paper: Hsu, Jeong, Pappas, Chaudhari (2020), "Scalable Reinforcement Learning Policies for Multi-Agent Control", arXiv 2011.08055.
- Verdict: **CONFIRMED** on all four points, with two qualifications about what is masked and when. [FULL]
- Training sizes: "a stochastic policy trained on the task space of n ∈ [1, 4] and m ∈ [1, 4] denoted as '4a4t' ... n ∈ [1, 8] and m ∈ [1, 8] denoted as '8a8t' ... n ∈ [1, 20] and m ∈ [1, 20] denoted as '20a20t'" (n pursuers, m targets).
- Test sizes: "scaling to 8, 20, 100, and 1000 agents"; headline "problems as large as 1000 pursuers tracking 1000 targets".
- k: masks of k = 1 ("greedy heuristic") and k = 4. "Policies evaluated with masks of k = 4 outperform the mask of k = 1" (Fig. 8 caption); "applying a k = 4 heuristic on the stochastic 4a4t and 8a8t policies outperforms all other policies".
- Full observation degraded: "Evaluating policies with the full observation state results in a degradation in performance as the task space expands past their training space. Policies 4a4t and 8a8t marginally outperform the greedy heuristic baseline for tasks within the training space but degrade outside of it." Also: "the 20a20t policy does not work well even on tasks inside its training space" — training on the larger space was itself worse.
- Density held constant: yes. "When evaluating policies on larger task spaces, we also scale the map to fit the density of the originally trained 4a4t task where the empty map has an area of 2500 m². With each agent able to observe an area of 25π m², we maintain the same density when scaling to 8, 20, 100, and 1000 agents."
- Qualifications:
  1. The mask exposes the k nearest **target beliefs** to each pursuer ("each agent's observation state is masked to expose the k nearest target beliefs"), not k nearest agents/entities generally.
  2. "In this paper, the heuristic mask is used **during evaluation** to improve scalability of trained policies" — it is a test-time heuristic applied to a policy trained without it, not a training-time architecture choice.
- Caveat for transfer: cooperative target tracking with decentralised, parameter-shared pursuers; soft double-Q learning (off-policy, discrete actions), not PPO; no adversary, targets follow fixed dynamics; density deliberately held constant, so it says nothing about generalising across densities; evidence is for a test-time mask.

## 15. Veličković et al.: softmax dispersion numbers

- Paper: Veličković, Perivolaropoulos, Barbero, Pascanu (2024), "Softmax is not Enough (for Sharp Size Generalisation)", arXiv 2410.01104.
- Verdict: **CONFIRMED** on the numbers; "recovered" overstates what adaptive temperature did. [FULL] (Table 1 and Sec. 4.1.)
- Table 1 (max-retrieval task, single attention head, trained on sets of ≤ 16 items, ten seeds):

| items | 16 (ID) | 32 | 64 | 128 | 256 | 512 | 1,024 | 2,048 | 4,096 | 8,192 | 16,384 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Baseline | 98.6% | 97.1% | 94.3% | 89.7% | 81.3% | 70.1% | 53.8% | 35.7% | 22.6% | 15.7% | 12.4% |
| Adaptive θ | 98.6% | 97.1% | 94.5% | 89.9% | 82.1% | 72.5% | 57.7% | 39.4% | 24.9% | 17.5% | 14.0% |

- So at 16 / 64 / 256 / 16384: baseline 98.6 / 94.3 / 81.3 / 12.4; adaptive temperature 98.6 / 94.5 / 82.1 / 14.0. Gains are statistically significant (paired t-test p-values in the table) but small: +0.2 points at 64, +0.8 at 256, at most +3.9 (at 1,024), +1.6 at 16,384. Adaptive temperature **does not recover** in-distribution accuracy; it shaves a few points off the degradation. The authors say as much: the key contribution "is not the adaptive temperature proposal; it is the fact that" dispersion is unavoidable for softmax with bounded logits. Adaptive temperature is applied at inference "without changing the parameters".
- Caveat for transfer: a synthetic argmax-retrieval task over sets, with degradation becoming material at 4× to 1000× the training size. A policy trained across its whole operating range of entity counts is not in the out-of-distribution regime the table measures; the result is a warning about extrapolating entity counts, not about attention within range.

## 16. Auxiliary prediction tasks

- Papers: Jaderberg, Mnih, Czarnecki, Schaul, Leibo, Silver, Kavukcuoglu (2016), "Reinforcement Learning with Unsupervised Auxiliary Tasks" (UNREAL), arXiv 1611.05397. Gelada, Kumar, Buckman, Nachum, Bellemare (2019), "DeepMDP: Learning Continuous Latent Space Models for Representation Learning", arXiv 1906.02736. Schwarzer, Anand, Goel, Hjelm, Courville, Bachman (ICLR 2021), "Data-Efficient Reinforcement Learning with Self-Predictive Representations" (SPR), arXiv 2007.05929.
- Verdict: **PARTLY** — auxiliary losses helping on-policy actor-critic is supported by UNREAL; a *next-state / forward-dynamics* auxiliary loss helping an on-policy actor-critic is **not** established by any of the three, and UNREAL contains a cautionary result. [FULL] for UNREAL; [FULL] abstract and Sec. 8.2-8.3 for DeepMDP; [ABSTRACT] plus Sec. 1 for SPR.
- UNREAL: base agent is A3C (on-policy actor-critic, CNN-LSTM). Auxiliary tasks: pixel control (off-policy Q-learning on pseudo-rewards), reward prediction (from a skewed replay buffer), value function replay. Eq. (2): `L_UNREAL = L_A3C + λ_VR L_VR + λ_PC Σ L_Q + λ_RP L_RP`. Results: "averaging 880% expert human performance" on Atari; Labyrinth "a mean speedup in learning of 10× and averaging 87% expert human performance" vs 54% for A3C. Ablation (Labyrinth): UNREAL 87%, A3C+PC 81%, A3C+RP+VR 79%, A3C+RP 72%.
  - Directly relevant ablation (Sec. 4.1, Fig. 5): they compare against "A3C augmented with a pixel reconstruction loss" and "A3C augmented with an input change prediction loss". "learning to control pixel changes is indeed better than simply predicting immediate pixel changes, which in turn is better than simply learning to reconstruct the input. In fact, learning to reconstruct only led to faster initial learning and actually made the final scores worse when compared to vanilla A3C. Our hypothesis is that input reconstruction hurts final performance because it puts too much focus on reconstructing irrelevant parts of the visual input". So the closest on-policy evidence says: prediction-of-change helped somewhat, reconstruction hurt final score, and control-style tasks were best. The auxiliary losses use a replay buffer, so the auxiliary part is off-policy.
- DeepMDP: latent model trained by reward prediction + next-latent-state distribution prediction, with theory tying these losses to bisimulation; empirical claim: "learning a DeepMDP as an auxiliary task in the Atari 2600 domain leads to large performance improvements over model-free RL". Base agent is **C51** (value-based, off-policy), 60 Atari games; "performance of the agent is highly dependent on the architecture" of the transition model (Sec. 8.3).
- SPR: "trains an agent to predict its own latent state representations multiple steps into the future", targets from an EMA encoder, plus augmentation; "median human-normalized score of 0.415 on Atari in a setting limited to 100k steps". Base agent is data-efficient **Rainbow** (value-based, replay); the regime is extreme low-data.
- I did not find, in this pass, a well-known result for forward-dynamics auxiliary losses specifically with PPO/on-policy actor-critic; I did not search exhaustively and make no claim that none exists.
- Caveat for transfer: all three are pixel observations, single-agent; two are value-based with replay. With compact entity-state inputs (not pixels) the representation-learning motive is weaker, and under fog of war a next-state target is partly unpredictable — neither case is covered.

## 17. Behaviour cloning then RL: protecting the cloned policy

- Papers: Schmitt, Hudson, Zidek, Osindero, Doersch, Czarnecki, Leibo, Kuttler, Zisserman, Simonyan, Eslami (2018), "Kickstarting Deep Reinforcement Learning", arXiv 1803.03835 [FULL]. Mathieu et al. (2023), "AlphaStar Unplugged: Large-Scale Offline Reinforcement Learning", arXiv 2308.03526 [FULL, relevant sections]. Vinyals et al. (2019), Nature 575 — **COULD NOT CHECK**: nature.com returned only the abstract and reference list (paywalled); Methods not readable. Secondary descriptions read instead: Han et al. (2020), "TStarBot-X", arXiv 2011.13729 [FULL, Sec. 4.5-4.6]; Wang et al. (2020), "SCC: an Efficient Deep Reinforcement Learning Agent Mastering the Game of StarCraft II", arXiv 2012.13169 [FULL, RL section].
- Verdict: **PARTLY.** Kickstarting confirmed in full. AlphaStar's KL-to-supervised term is confirmed only **second-hand**; its schedule could not be checked. AlphaStar Unplugged does **not** contain the claimed mechanism.
- Kickstarting: loss `l_kick(ω, x, t) = l_RL(ω, x, t) + λ_k H(π_T(a|x_t) ‖ π_S(a|x_t, ω))` — cross-entropy from teacher to student, evaluated on **student-generated (on-policy) trajectories**, added to A3C/IMPALA loss which keeps its own entropy bonus `−β H(π_S)`. Schedule of λ_k: adapted by Population Based Training; "showing a reduction to almost negligible weight after 2 billion steps". Sec. 5.1.1 / Table 2 compares PBT with hand schedules ("linear from 1 to 0 at 1B / 2B / 4B", "linear from 2 to 0 at 1B", constants 1 and 2): "constant setting of λ_k perform well at first, but their performance ultimately plateaus. Linear schedules can work well, but only if chosen correctly. PBT performance is competitive and alleviates the need for sweeps". Result: student matches a from-scratch agent "in almost 10× fewer steps, and surpassing its final performance by 42%" (DMLab-30).
  - Note the setting: the student is **randomly initialised** and learns from a teacher *network* online. It is not "BC-initialised weights then RL". What it supports is the loss form and the need to decay λ to zero.
- AlphaStar Unplugged: purely **offline** RL. "The Behavior Cloning policy μ̂ can be used as the starting point for π (i.e. used to initialize the weights)"; the offline actor-critic keeps the critic fixed at V^μ because TD learning of V^π diverged offline; MuZero with MCTS training targets "would collapse quickly to a few actions". It has no online RL stage and no KL/distillation-to-supervised term with a schedule. **NOT SUPPORTED** as a source for the claim; it is only (useful) evidence that policies initialised from BC drift and collapse without a constraint.
- AlphaStar (second-hand only): TStarBot-X Sec. 4.6: "Following AlphaStar, our RL loss consists of a V-trace loss, an UPGO loss, an entropy regularizer, and a policy distillation term targeting to the supervised model. L_RL = L_V-trace + L_UPGO + L_entropy + L_distill", "in L_distill the teacher policy is always fixed as the supervised agent". SCC: "In addition to the standard PPO loss, we also added extra loss terms, i.e., a standard entropy regularization loss, and a KL divergence loss with respect to the final supervised policy to prevent RL policy deviating from humans (Vinyals et al., 2019b). L_RL = L_PPO + L_entropy + L_KL." (SCC is a PPO example of the construction.) Neither states AlphaStar's weight or schedule for the term, and I could not read the Nature Methods to get it.
- Bearing on the project (RL after BC, entropy coefficient 0, no distillation term): every source read that does BC→RL (AlphaStar per TStarBot-X/SCC, SCC itself, TStarBot-X) keeps **both** an entropy regulariser and a KL/distillation term to the supervised policy; Kickstarting keeps the entropy bonus and decays the distillation weight rather than omitting it. None of the sources ran the project's configuration, so they do not *prove* it fails — but the configuration has no provenance, and AlphaStar Unplugged's collapse reports are a warning in the same direction.
- Caveat for transfer: the StarCraft distillation term anchors to *human* play to preserve strategic diversity; if the project's BC teacher is a scripted or weak policy, anchoring to it trades stability against the ceiling, which is exactly why Kickstarting decays λ.

## 18. Prioritised fictitious self-play / league

- Source: Vinyals et al. (2019), Nature — **COULD NOT CHECK** (paywalled; see 17). arXiv-accessible descriptions read: SCC, arXiv 2012.13169; TStarBot-X, arXiv 2011.13729.
- Verdict: **PARTLY** — attribution to AlphaStar and the qualitative definition are confirmed second-hand; the exact weighting function could not be verified from a readable primary source. [FULL] for the two secondary papers.
- SCC, Sec. 2 (describing AlphaStar's league): "prioritized fictitious self-play (PFSP) mechanism that adapts the mixture probabilities proportionally to the win rate of each opponent against the agent, to dynamically focus more on the difficult opponents. Second, main exploiters play only against current main agents to find weaknesses in main agents. Third, league exploiters use a similar PFSP mechanism against all agents to find global blind spots in the league."
- TStarBot-X, Sec. 4.5: cites PFSP to AlphaStar ([1]); "The MA will use 25% Self-Play, 60% PFSP and 15% of PFSP matches against forgotten main players, similar to AlphaStar." Also a caution: "using PFSP is not sufficient to let the agent discover robust or novel policies. Diversity in the policy population thus plays an important role".
- Not verified: the specific weighting functions usually quoted for AlphaStar (recollection, NOT verified: f_hard(x) = (1 − x)^p and f_var(x) = x(1 − x) applied to the win probability against each candidate opponent). Treat as unverified until the Nature Methods text is read.
- Related, verified, simpler alternative: OpenAI Five plays 80% against the current policy and 20% against past versions ("Past opponents 20%", App. C), with past-opponent sampling by a learned quality score (App. N, not read in detail here).
- Caveat for transfer: league machinery was built for a game with strong strategy cycles and a population of many agents on very large compute; for a single-learner setup the readable evidence supports "mix in past opponents, weighted toward ones you lose to", not the full league.

## 19. Policy weight averaging (EMA) for evaluation / league

- Papers: Hilton, Cobbe, Schulman (2021), "Batch size-invariance for policy optimization", arXiv 2110.00641 [FULL, Sec. 4 and 5.3]. Nikishin et al. (2018), "Improving Stability in Deep Reinforcement Learning with Weight Averaging" — **COULD NOT CHECK**: not found on arXiv by title search; not read.
- Verdict: **NOT SUPPORTED** as stated (EMA of policy weights used as the *evaluation or league* policy). The one source read uses EMA weights for a different purpose.
- Hilton et al.: "Maintain an exponentially-weighted moving average (EWMA) of the policy network, updating it after every policy gradient step using some decay rate β_prox. Use this as the network for the **proximal policy** in one of the decoupled policy objectives." That is, the EMA network is the anchor for the PPO trust-region/KL term, decoupled from the behaviour policy used for importance ratios. It is not used for acting, evaluation or as an opponent. Result (Sec. 5.3): "We found the benefit of the EWMA to be small but remarkably consistent across environments and algorithms, outperforming the baseline on all of the last 8 environments for both PPO and PPG. We believe that this is the result of the EWMA reducing the variance of the proximal policy." Cost: "increases the computational cost of PPO by 30% and of PPG by 2.3%". They cite Izmailov et al. 2018 (SWA) for "averaging in parameter space may actually improve the proximal policy" — a supervised-learning reference.
- Also seen while reading other claims (not policy-weight averaging for evaluation): DreamerV3 regularises the critic toward an EMA of its own parameters (decay 0.98); SPR uses an EMA target encoder.
- Net: no source read in this pass shows that evaluating, checkpointing or league-seeding with EMA policy weights improves on-policy RL. If the baseline includes it, it should be listed as an unprovenanced choice (or justified by a project experiment).
- Caveat for transfer: Hilton et al. is Procgen, single-agent PPO/PPG.

## 20. Entity masking as standard practice

- Papers: Baker, Kanitscheider, Markov, Wu, Powell, McGrew, Mordatch (ICLR 2020), "Emergent Tool Use From Multi-Agent Autocurricula", arXiv 1909.07528 [FULL]. Iqbal, Schroeder de Witt, Peng, Böhmer, Whiteson, Sha (2021), "Randomized Entity-wise Factorization for Multi-Agent Reinforcement Learning" (REFIL), arXiv 2006.04222 [FULL]. Hu, Zhu, Chang, Liang (ICLR 2021), "UPDeT: Universal Multi-agent Reinforcement Learning via Policy Decoupling with Transformers", arXiv 2101.08001 [FULL].
- Verdict: **PARTLY** — Baker et al. and REFIL confirm masking of unobserved entities in attention; **UPDeT does not** (its masking is of the action space), and none of the three text passages explicitly discusses *dead / padded* entities as distinct from unobserved ones.
- Baker et al.: Sec. 3: "If entities are not in line-of-sight of the agent or not in a 135 degree cone in front of the agent, then they are masked out in the policy." App. B: "All the embedded entities are then passed through a residual self-attention block ... We then average-pool entity embeddings ... Note that in the policy network the entities not observed by each agent are masked away through self-attention and pooling." (Fig. 2: "masked residual self-attention", "masked average pooling".) The value function is omniscient (unmasked); App. A ablation: "training with a masked value function never progressed past stage 3". This is PPO + GAE self-play with entity attention and a visibility mask — the closest architectural precedent to the target setting.
- REFIL: App. A defines attention with "M, a binary obserability mask specifying which entities each query entity can observe (i.e. M_{i,j} = 1 when i ∈ S can incorporate information from x_j ...)" and "The mask(Y, M) operation takes two equal sized matrices and fills the entries of Y with −∞ in the indices where M is" [zero]. Sec. 3: "Masking enables 1) decentralized execution by providing the mask M^μ ... such that agents can only see entities observable by them" and 2) the randomised sub-group factorisation. Varying entity counts are handled through this same masking. Q-learning (QMIX-style), cooperative StarCraft.
- UPDeT: the only masking described is "Dead enemy units will be masked out from the action space to ensure that the executed action is valid" and "Once an agent or enemy has died, we mask corresponding unavailable action in the action select stage". Nothing in the text says dead entities are masked in attention. Do not cite UPDeT for attention masking.
- Contrast: OpenAI Five (claim 12a) does *not* mask — it zero-pads and max-pools, which is safe for max-pool only to the extent that post-FC embeddings of zero inputs do not win the max.
- Caveat for transfer: Baker et al. mask by line-of-sight per agent with an omniscient critic; a centralised policy under team-shared fog has one mask per team, and "absent because dead" vs "absent because unseen" vs "last-seen stale" (OpenAI Five's choice for heroes) are distinct states that none of these sources separates.

## 21. Two-hot / binned encodings for continuous inputs

- Paper: Gorishniy, Rubachev, Babenko (NeurIPS 2022), "On Embeddings for Numerical Features in Tabular Deep Learning", arXiv 2203.05556 (v4) [FULL]. Also DreamerV3 (2301.04104) for what an RL agent does with inputs [FULL].
- Verdict: **PARTLY** — there is a source for binned encodings of scalar inputs outperforming raw scalars, but it is tabular supervised learning, the encoding is piecewise-linear (thermometer-like), not two-hot, and the paper's best performer is a different (periodic) embedding. No RL source for two-hot *inputs* was found.
- Encoding, Eq. (1): `PLE(x) = [e_1, ..., e_T]`, with `e_t = 0` if x < b_{t−1} (t > 1); `e_t = 1` if x ≥ b_t (t < T); otherwise `(x − b_{t−1})/(b_t − b_{t−1})`. This is a **cumulative** code (all bins below the value are 1, one bin is fractional), not two-hot (two adjacent non-zero entries summing to 1). Bins from quantiles or from target-aware decision-tree splits. "For T = 1, the PLE-representation is effectively equivalent to the scalar representation." Motivation: "vanilla MLP is known to be a universal approximator, in practice, due to optimization peculiarities, it has limitations".
- Findings (Sec. 4.5-4.7 takeaways): "The piecewise linear encoding is often beneficial for both types of architectures (MLP and Transformer) and the profit can be significant"; "For most datasets, embeddings for numerical features can provide noticeable improvements for three different backbones"; "The simplest LR embedding is a good baseline solution: although the performance gains are not dramatic, its main advantage is consistency"; "The PLR module [periodic → linear → ReLU] provides the best average performance. Empirically, we observe σ to be an important hyperparameter that should be tuned"; "after the MLP-like architectures are coupled with embeddings for numerical features, they perform on par with the Transformer-based models". Heavy variants "do not justify the extra costs".
- RL practice actually found: DreamerV3 uses two-hot only for **targets** (reward, critic); for inputs it says "we transform vector observations using the symlog function ... both for the encoder inputs and the decoder targets" — i.e. a squashed scalar, not a binned code.
- So the honest statement is: binned/periodic input embeddings have supervised tabular provenance (with periodic+linear the best on average and plain linear the consistent one); two-hot input encoding in RL has **no source found**.
- Caveat for transfer: i.i.d. tabular supervised learning with fixed feature distributions (quantile bins are meaningful); in RL the input distribution drifts with the policy, bins must be fixed in advance, and for coordinates the binned code competes with the Fourier/rotary treatments in claims 5-6.

---

## Summary table

| # | Claim | Verdict | Label |
|---|---|---|---|
| 1 | PPO clipped surrogate, epochs/minibatches, ε = 0.2; no advantage normalisation in paper | CONFIRMED (0.2 is MuJoCo; Atari uses 0.1×α) | FULL |
| 2A | Andrychowicz et al. on (a)-(e) | CONFIRMED (adv. norm: no effect; value clip: hurts; obs norm: always; epochs: multiple; recompute advantages per pass) | FULL |
| 2B | Engstrom et al. on (a)-(e) | PARTLY (only value clipping ablated, with no explicit conclusion; a, c, d, e not studied) | FULL |
| 3 | GAE estimator and λ range | CONFIRMED (λ ∈ [0.9, 0.99]) | FULL |
| 4 | RG-LRU equations, c = 8, init a^c ~ U[0.9, 0.999] | CONFIRMED; project init linspace(0,4) NOT SUPPORTED by paper (decay 0.004-0.865 vs 0.9-0.999) | FULL |
| 5 | RoPE; rotary on continuous ND coordinates | PARTLY (exists for grids and RGB-D depth in vision/robotics; none in RL, none on entity world coordinates) | FULL |
| 6 | Fourier features fix spectral bias | CONFIRMED (supervised coordinate regression; scale must be tuned) | FULL |
| 7 | HL-Gauss, comparisons, results, PPO coverage | PARTLY (no PPO; only IMPALA actor-critic; two-hot underperforms) | FULL (Imani & White: ABSTRACT) |
| 8 | DreamerV3 two-hot bins/range, return norm, η | PARTLY (255 bins only in v1; v2 divides (R − v), v1 divides R; rest confirmed) | FULL |
| 9 | RND two heads, discounts, summed advantages | CONFIRMED (γ_E = 0.999, γ_I = 0.99; coefficients 2 and 1) | FULL |
| 10 | MAPPO recommendations | PARTLY (suggestion says "value normalization", PopArt only in pseudocode; rest confirmed) | FULL |
| 11 | PopArt | CONFIRMED | FULL (1809.04474: ABSTRACT) |
| 12 | OpenAI Five (a)-(f) | CONFIRMED (details differ from paraphrase on fog and LSTM) | FULL |
| 13 | Registers as support for a null key | PARTLY (finding confirmed; support is by analogy only) | FULL |
| 14 | Hsu et al. k-nearest masking | CONFIRMED (mask is over nearest targets, applied at evaluation only) | FULL |
| 15 | Veličković et al. numbers | CONFIRMED (adaptive temperature gains ≤ 3.9 points; does not recover) | FULL |
| 16 | Auxiliary prediction tasks | PARTLY (UNREAL on-policy; forward-dynamics results are value-based; reconstruction hurt in UNREAL) | FULL (SPR: ABSTRACT) |
| 17 | BC then RL: KL/distillation to supervised policy | PARTLY (Kickstarting confirmed; AlphaStar only second-hand; AlphaStar Unplugged does not contain it; Nature COULD NOT CHECK) | FULL (secondary sources) |
| 18 | PFSP / league definition | PARTLY (second-hand via SCC and TStarBot-X; primary COULD NOT CHECK) | FULL (secondary sources) |
| 19 | EMA policy weights for evaluation/league | NOT SUPPORTED (Hilton et al. use EMA as proximal policy only; Nikishin et al. COULD NOT CHECK) | FULL |
| 20 | Entity masking as standard practice | PARTLY (Baker, REFIL confirm; UPDeT masks actions, not attention) | FULL |
| 21 | Two-hot/binned continuous inputs | PARTLY (tabular PLE, not two-hot, not RL; DreamerV3 uses symlog inputs) | FULL |
