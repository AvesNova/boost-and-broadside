# Provenance check: seven design questions (2026-11-22)

Every citation below was fetched on 2026-11-22 and the named passage read, unless it is
marked **could not fetch**. arXiv PDFs are the latest version on that date. Page numbers
are the PDF's own. Code and configs were read from the official repositories named.

---

## 1. HL-Gauss: bin spacing, sigma, bin count

**Source (read).** Farebrother, Orbay, Vuong, Taïga, Chebotar, Xiao, Irpan, Levine, Castro,
Faust, Kumar, Agarwal, "Stop Regressing: Training Value Functions via Classification for
Scalable Deep RL", ICML 2024. https://arxiv.org/abs/2403.03950

**Passages.** §3.1, Eq. 3.3 and the paragraph "How should we tune σ in practice?" (p. 6);
§5.1.2 and Figure 13 (p. 12-13); Appendix A, Listing 1 (Jax and PyTorch code);
Appendix B, Tables B.2 and B.3 (p. 23); Appendix B.2, Chess (p. 24).

**Answer.**
- **Spacing: linear.** Bins have one width ς = (v_max − v_min)/m (§3.1, before Eq. 3.3);
  the reference code builds the edges with `linspace(min_value, max_value, num_bins + 1)`
  (Listing 1). No symlog or other transform appears anywhere in the paper.
- **Sigma: σ/ς = 0.75.** "a more interpretable hyper-parameter that we recommend tuning is
  σ/ς: setting it to K/6 distributes most of the probability mass to ⌈K⌉ + 1 neighbouring
  locations … Unless specified otherwise, we set σ/ς = 0.75 for our experiments, which
  distributes mass to approximately 6 locations" (p. 6). Table B.3: smoothing ratio 0.75.
- **Too small or too large.** The paper never says "too hot". The sweep (§5.1.2, Figure 13:
  13 Atari games, bins {21, 51, 101, 201}, σ/ς {0.25, 0.5, 0.75, 1.0, 2.0}) says "a wide range
  of σ values for HL-Gauss outperform Two-Hot" and that "the optimal value of σ seems to be
  independent of number of bins". Reading the figure (my reading of the plot, not text): the
  x axis is σ in value units; the smallest σ (about 0.03) score well *below* Two-Hot (≈0.1
  against ≈0.6 IQM); the score peaks at σ ≈ 0.3 to 0.4 for every bin count; the largest σ
  (2.0 with 21 bins) falls back to about Two-Hot's level. So too narrow is the worse failure
  (it approaches a lossy one-hot), too wide degrades more gently.
- **Bin count.** Atari: 51 bins on [−10, 10] (Table B.2, inherited by B.3). Chess: m = 128
  bins on [0, 1] (B.2). Wordle, robotics and others reuse Table B.3. The sweep shows little
  dependence on bin count once σ is fixed in value units.

**What it means here.** The paper's recommendation is really a width in value units
(0.75 × 20/51 ≈ 0.29, about 1.5% of the support); "0.75 bins" carries over only if the
project's bins span a comparable fraction of its return range. The linear-against-symlog
question is not covered: HL-Gauss was tested only on uniform bins. Applying the Gaussian in
symlog space (uniform there) is an extrapolation, kind (d). The paper is value-based RL
(DQN, CQL, Q-transformers) on single-scale returns, not a PPO critic over several reward
levels.

---

## 2. Symlog in DreamerV3: why

**Source (read).** Hafner, Pasukonis, Ba, Lillicrap, "Mastering Diverse Domains through
World Models", arXiv 2301.04104 (v2, 2024; Nature 2025). https://arxiv.org/abs/2301.04104
Bin count from the official config (read): https://github.com/danijar/dreamerv3,
`dreamerv3/configs.yaml`, `rewhead` and `value`: `output: symexp_twohot, bins: 255`.

**Passages.** "Critic learning" (p. 5-6); "Robust predictions", Eqs. 8-11 (p. 7);
Methods, "Distributions", Eq. 12 (p. 19 area); Ablations (Figure 6 text).

**Answer.** The paper gives three reasons, all framed by scale varying across domains:
- "the scale of these quantities can vary across domains. Predicting large targets using a
  squared loss can lead to divergence whereas absolute and Huber losses stagnate learning.
  On the other hand, normalizing targets based on running statistics introduces
  non-stationarity into the optimization." (Robust predictions, p. 7)
- The critic is categorical over "exponentially spaced bins, decoupling the scale of
  gradients from the prediction targets" because "the return distribution can have multiple
  modes and vary by orders of magnitude across environments" (Critic learning, p. 5-6).
- The loss "only depends on the probabilities assigned to the bins but not on the continuous
  values associated with the bin locations, decoupling the size of the gradients from the
  size of the targets" (Eq. 11). B = symexp(−20 … +20) (Eq. 10). Symlog "approximates the
  identity around the origin so that it does not affect learning of targets that are
  already small enough". Unlike alternatives it avoids "truncating large targets",
  "non-stationary from normalization" or "adjusting network weights when new extreme values
  are detected".

The whole paper is framed as "fixed hyperparameters across all domains" (abstract,
Figure 1). The ablation (Figure 6, 14 tasks) finds each robustness technique "critical on a
subset of tasks but may not affect performance on other tasks".

**Verdict on the author's reading.** Mostly confirmed: the stated purpose is one
configuration across domains whose reward and return scales differ by orders of magnitude.
One correction: the paper also motivates symlog against *running normalisation*
(non-stationarity) and against targets growing during training, which can happen within one
game. With a scale per level (as decided), symlog is needed only if a level's returns span
orders of magnitude or drift during training; otherwise linear bins over a known range
suffice. The two-hot targets are DreamerV3's; HL-Gauss (Q1) was not tested with symexp bins.

---

## 3. Attention that need not land on a token, and training stability

**Candidates read:**

| Work | Mechanism | Claim | Removes the need for a null key? |
|---|---|---|---|
| Qiu et al., "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free", Qwen team, arXiv 2505.06708 (2025). **Read**: abstract, §1, §2.2 Eq. 5, Table 1 rows 5 and 10, §4.3, Table 4, Figure 2 | Head-specific sigmoid gate on each head's SDPA output, computed from the query token's (normed) hidden state: Y' = Y ⊙ σ(XW) (Eq. 5, position G1), before the output projection | Best of 30 variants on 15B MoE and 1.7B dense models (3.5T tokens): "nearly eliminating loss spikes, enabling larger learning rates"; first-token attention falls from 46.7% to 4.8% (Figure 2); massive activations fall from 1053 to 94 (Table 4). Attributed to non-linearity between W_V and W_O and query-dependent sparsity | Functionally yes: a head can output nothing by closing its gate, so it no longer needs a token to park attention on. The softmax still sums to one; the mass goes somewhere and is zeroed after. Headwise and elementwise both reduce the sink (Table 4: F-Attn 0.073 headwise, 0.048 elementwise) |
| Kimi Linear, Moonshot, arXiv 2510.26692 (2025). **Read**: §4 Eq. 10, §5.2 and Table 1 | Adopts Qiu et al.'s sigmoid output gate (low-rank) on its KDA layers, citing it for "alleviating the Attention Sink"; NoPE on its full-attention (MLA) layers | Removing the gate raises perplexity (Table 1: 5.65 → 5.67 validation); a swish gate is "substantially worse" than sigmoid | Same mechanism as Qiu et al.; Moonshot's use of it is probably what the author remembers |
| Kimi K2 technical report, Moonshot, arXiv 2507.20534 (2025). **Read**: §2.1 "MuonClip" | QK-Clip: after each optimiser step, rescale W_q and W_k per head whenever that head's maximum pre-softmax logit exceeds τ = 100 | 15.5T tokens "with zero loss spike"; counters attention-logit explosion seen with Muon | No. It bounds logits; it says nothing about sinks or null keys |
| gpt-oss model card, OpenAI, arXiv 2508.10925 (2025). **Read**: §2.2 "Attention" | "Each attention head has a learned bias in the denominator of the softmax, similar to off-by-one attention and attention sinks, which enables the attention mechanism to pay no attention to any tokens" | Architectural statement; no ablation in the card | Yes, and it is literally a null key: one learned logit per head with a zero value, without a token or projection |
| Xiao et al., "Efficient Streaming Language Models with Attention Sinks" (StreamingLLM), ICLR 2024, arXiv 2309.17453. **Read**: §3.1, §3.3, Table 3, Eq. 2 | Explains sinks: softmax forces the weights to sum to one, so models dump spare attention on always-visible tokens. Tests SoftMax₁ ("Zero Sink", a key and value of zeros) and a learnable sink token prepended in pre-training | 160M-parameter models: Zero Sink "alleviates the attention sink problem to some extent" but the model still uses other initial tokens; the learnable sink token "is highly effective in stabilizing the attention mechanism" (stabilising here means streaming perplexity, not training stability) | The learnable sink token *is* a null key (with a learned value), the form lean uses now |
| Evan Miller, "Attention Is Off By One", blog, 2023. **Read**: https://www.evanmiller.org/attention-is-off-by-one.html | softmax₁(x)ᵢ = exp(xᵢ)/(1 + Σⱼ exp(xⱼ)) | Opinion piece; no experiment | Yes in principle (a zero-logit, zero-value null key) |

**Answer.** The "elegant replacement that also stabilised training" matches Qiu et al.'s
gated attention (Qwen, 2025): stability (no loss spikes, larger learning rates) and no
attention sink, from one sigmoid gate per head. Moonshot's Kimi Linear adopted it and
cites it for the sink. Kimi K2's QK-Clip is the Moonshot stability fix, but it has nothing to
do with sinks. I found no Moonshot paper proposing its own sink replacement.

**What it means here.** v2 already has Qiu et al.'s G1 gate (`spatial.py`, `GATES`,
"elementwise"/"headwise"), the variant the paper found best. Qiu et al. do not give the
gate's initialisation; v2's (weight zero, bias 2.0, all gates 0.88 open) is the project's
own choice, kind (d). "Drop the null key" is supported only if the gate is on: the gate gives a
head a way to abstain. Without the gate, gpt-oss's per-head learned denominator bias is the
cheapest null key (one scalar per head, no token, no projection). Every result above is a
causal language model with thousands of tokens; the project attends over a few dozen ship,
field and zone tokens per query, where sinks may never form. Whether they do here is
measurable (attention share on the game token).

---

## 4. Griffin's initialisation, and v2's and lean's

**Sources (read).** De, Smith, Fernando, Botev, Cristian-Muraru, Gu, Haroun, Berrada, Chen,
Srinivasan, Desjardins, Doucet, Budden, Teh, Pascanu, De Freitas, Gulcehre, "Griffin:
Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models",
arXiv 2402.19427 (2024). §2.2 MLP block, §2.3 Recurrent block (p. 4), §2.4 RG-LRU,
Eqs. 1-4 (p. 4), §4.1 (p. 7), Appendix A Eq. 6.
Official code (read): https://github.com/google-deepmind/recurrentgemma,
`recurrentgemma/torch/layers.py` (`rnn_param_init`, `RGLRU`, `BlockDiagonalLinear`,
`Conv1D`), `torch/modules.py` (`RecurrentBlock`), `torch/griffin.py`.

**What the paper specifies.**
- a = σ(Λ), Λ learnable; a_t = a^(c·r_t); "c is a scalar-valued constant set to 8" (§2.4).
- "We initialize Λ such that a^c is uniformly distributed between 0.9 and 0.999 at the
  start of training, similar to Orvieto et al. (2023b)" (§2.4, p. 4).
- "We initialize both W_a and W_x using LeCun init" (§2.4). The gates have biases b_a, b_x
  (Eqs. 1-2); their initialisation is not stated.
- Gates are block-diagonal, 16 blocks each (§4.1, p. 7), a sharding choice.
- Recurrent block: two parallel linear layers of width D_RNN; one branch a "small separable
  Conv1D … with a temporal filter dimension of 4", then the RG-LRU; the other branch GeLU;
  multiply, then a final linear layer (§2.3). D_RNN ≈ 4D/3 (§3, "Hawk"). MLP expansion
  M = 3 (§2.2). The paper gives no initialisation for the projections or the convolution.
- log a_t computed as −c·softplus(−Λ)·r_t (Appendix A, Eq. 6).

**What the official code does (where the paper is silent, and one place it disagrees).**
- `rnn_param_init(min_rad=0.9, max_rad=0.999)`: draws u ~ U[0.9², 0.999²] and sets
  softplus(a_param) = −½ ln u, with log a_t = −8·r_t·softplus(a_param). So at r = 1,
  a_t = u⁴: **the code puts a (not a^c) in [0.9, 0.999]**, uniform in a² ("uniformly on a
  ring"), giving a^c in [0.43, 0.992] (half-lives at r = 1: 0.8 to 87 steps). The paper's
  text says a^c in [0.9, 0.999] (half-lives 6.6 to 693 steps). They disagree by a factor of 8 in
  the exponent. I cannot tell from the sources which one trained the published models.
- Gates: block-diagonal, N(0, 1/block_width) weights, zero biases.
- Input projections (x and y branches): N(0, 1/D), zero biases. Output projection:
  N(0, s/D_RNN) with s = 2/num_layers (`griffin.py`), zero bias.
- Conv1D: N(0, 0.01/4) weights (std 0.05), zero bias.
- sqrt(1 − a_t²) is replaced by 1 at a segment's first position (`reset`), and its gradient
  is clipped (`SqrtBoundDerivative`).

**v2 against the paper and the code** (`src/bnb/models/yemong/temporal.py`,
`src/bnb/models/initialise.py`):

| Item | Paper / code | v2 | Effect |
|---|---|---|---|
| Λ | a^c ~ U[0.9, 0.999] (paper); a ~ ring-uniform [0.9, 0.999] (code), random | `linspace(0, 4)`, deterministic: a = σ(Λ) ∈ [0.50, 0.982], **a^c ∈ [0.004, 0.865]** | Far shorter memory: half-lives at r = 1 of 0.12 to 4.8 steps against 6.6 to 693 (paper) or 0.8 to 87 (code). At init r ≈ 0.5, so a_t ∈ [0.06, 0.93]. Most channels forget within a step or two |
| Gates W_a, W_x | LeCun normal, var 1/fan_in; block-diagonal 16 blocks | Dense `nn.Linear` default: Kaiming-uniform (a = √5), var 1/(3·fan_in) | A third of the variance; dense rather than block-diagonal (the paper's reason for blocks is sharding, so dense is defensible) |
| Gate biases | Zero (code) | Uniform ±1/√D | Small random offsets |
| Input projection, gate branch | N(0, 1/D), with zero bias (code) | Kaiming-uniform default, var 1/(3D), no bias | A third of the variance |
| Output projection | N(0, 2/(L·D_RNN)) (code), depth-scaled | Kaiming-uniform default, var 1/(3D), not depth-scaled | Equal only at L = 6 blocks; v2 does not scale with depth |
| Conv1D | Width 4 ✓; N(0, 0.0025), zero bias (code) | Uniform ±½ (std 0.29), bias uniform ±½ | About 6× the weight scale, plus a random bias of order 0.3 per channel added to every input of the recurrence |
| c | 8 ✓ | 8 ✓ | Same |
| √(1 − a²) at reset | 1 at the first step (code) | Always √(1 − a²), floored at 1e-6 | The first input after a reset is scaled down by √(1 − a²) |
| D_RNN, MLP | D_RNN ≈ 4D/3, MLP M = 3 | D_RNN = D, MLP 4D | Width choices, not initialisation |

**Verdict.** The author is right about the core: v2's decay initialisation is far from
Griffin's (deterministic and short, a^c ≤ 0.865 against ≥ 0.9). Calling it "completely" wrong
overstates it: c = 8, the gate formulas, the √(1 − a²) normalisation and the block layout
match. Every linear layer and the convolution use torch defaults instead of the
paper's LeCun or the code's scaled normals. The convolution's ±½ weights and biases are the
largest numerical departure after Λ.

**lean** (`lean/model/policy.py`, class `Temporal`): Λ set so a^c ~ U[0.9, 0.999]
(`retention_low/high`), matching the paper's text, not the official code. Gates are dense
`nn.Linear` with default init (same notes as v2). But lean's `Temporal` is a **bare RG-LRU**:
no input projection, no Conv1D, no GeLU gate branch, no output projection. It adds the state
straight to the residual stream. That departs from both Griffin and the decision "Memory is
the Griffin temporal block (v2's block, not a bare recurrence)". lean also scales the input
by √(1 − a²) at an episode's first tick.

---

## 5. AlphaStar's PFSP weights and opponent shares; OpenAI Five

**Sources.**
- Vinyals et al., "Grandmaster level in StarCraft II using multi-agent reinforcement
  learning", Nature 575, 350-354 (2019). https://doi.org/10.1038/s41586-019-1724-z.
  **Methods section: could not fetch** (Nature paywall; the open page carries figures and
  references only). **Read**: the unformatted main text released by DeepMind
  (https://storage.googleapis.com/deepmind-media/research/alphastar/AlphaStar_unformatted.pdf),
  "League" paragraphs and Figures 1C and 3 captions. Also read: the paper's
  **Supplementary Data pseudocode** (Nature ESM, `41586_2019_1724_MOESM2_ESM.zip` →
  `pseudocode/multiagent.py`), functions `pfsp`, `Payoff`, `MainPlayer.get_match`,
  `MainExploiter.get_match`, `LeagueExploiter.get_match`.
- Berner et al. (OpenAI), "Dota 2 with Large Scale Deep Reinforcement Learning",
  arXiv 1912.06680 (2019). **Read**: §3 (p. 6) and Appendix N "Self-play", Eq. 14.

**Answer (AlphaStar, from the pseudocode), with x the learner's win rate against the
opponent:**
- `variance`: x(1 − x) (f_var). `linear`: 1 − x. `linear_capped`: min(0.5, 1 − x).
  `squared`: (1 − x)² (f_hard in the paper's terms; the Methods text naming it could not
  be read). Probabilities are the weights normalised; uniform if they sum below 1e-10.
- Win rate counts draws as half; an unplayed pair is 0.5; every game decays that pair's
  counts by 0.99 first (an exponential moving window).
- **Main agent** per game: 50% PFSP `squared` over all past players; 15% a "verification"
  branch (exploiter checkpoints with `squared` if any win rate < 0.3; else the chosen main
  agent's own checkpoints with `squared` if any < 0.7, after dropping a monotonic suffix);
  otherwise, and whenever verification finds nothing, self-play against a main agent (35%
  or more). In self-play, if the learner's win rate against it is ≤ 0.3, it plays that
  agent's past checkpoints with `variance` instead.
- **Main exploiter**: the current main agents if its win rate against them > 0.1; else those
  agents' checkpoints with `variance`. Reset to the supervised weights at every checkpoint.
- **League exploiter**: all past players with `linear_capped`; reset with probability 0.25.
- Main text: "With fixed probability, a main agent is selected as an opponent; this recovers
  the rapid learning of self-play". Figure 3 ablations used "a 50%-50% mix of self-play and
  PFSP"; naive self-play "has high Elo, but is more forgetful".

**OpenAI Five.** "80% of the games are played against the latest set of parameters, and 20%
play against past versions" (Appendix N; also §3). Past opponents are sampled by softmax
over quality scores q_i. A new snapshot (every 10 iterations) enters at the current maximum
q, and when the learner beats opponent i, q_i ← q_i − η/(N·p_i), with η = 0.01 (Eq. 14).

**What it means here.** The proposal "(1 − p)² with a small uniform floor" is AlphaStar's
`squared` weighting. AlphaStar has no uniform floor; its coverage comes from the 0.5
prior for unplayed pairs, the 0.99 decay, the verification branch and the exploiters. The
decided half self-play, half league split matches AlphaStar's 50/50 ablation setting and is
more league-heavy than OpenAI Five's 80/20. Both are two-player (team) zero-sum with one
learner per side. The project's single shared policy with a scripted league member is a
smaller league, so the shares are a starting point, not evidence.

---

## 6. Grid cells in artificial agents

**Sources.**
- Banino, Barry, Uria, Blundell, Lillicrap, Mirowski, Pritzel, Chadwick, Degris, Modayil,
  Wayne, Soyer, Viola, Zhang, Goroshin, Rabinowitz, Pascanu, Beattie, Petersen, Sadik,
  Gaffney, King, Kavukcuoglu, Hassabis, Hadsell, Kumaran, "Vector-based navigation using
  grid-like representations in artificial agents", Nature 557, 429-433 (2018).
  **Read** (full text, course-site copy of the Nature PDF:
  https://www.its.caltech.edu/~jkenny/nb250c/papers/Banino-2018.pdf): main text p. 429-432.
  Methods and supplement not read.
- Cueva and Wei, "Emergence of grid-like representations by training recurrent neural
  networks to perform spatial localization", ICLR 2018, arXiv 1803.07770. **Read**: §2
  (model, inputs/outputs, training), §3 (results), Figure 1.
- Solstad, Moser, Einevoll, "From grid cells to place cells: a mathematical model",
  Hippocampus 16, 1026-1031 (2006), doi 10.1002/hipo.20244. **Could not fetch** (Wiley
  paywall; not in PubMed Central).
- In its place for the three-wave model: Blair, Welday, Zhang, "Scale-invariant memory
  representations emerge from moiré interference between grid fields that produce theta
  oscillations", J. Neurosci. 27, 3211-3229 (2007).
  https://www.jneurosci.org/content/27/12/3211. **Read** (via fetch; the equation is an
  image, the text around it was read): Materials and Methods, "Cosine grating model of
  theta grids".
- Mai, Janowicz, Yan, Zhu, Cai, Lao, "Multi-Scale Representation Learning for Spatial
  Feature Distributions using Grid Cells" (Space2Vec), ICLR 2020, arXiv 2003.00824.
  **Read**: §4.1 Eqs. 3-4, §5.1.1 and Table 1.

**Answers.**
- **Standard firing-map model.** Blair et al. 2007: the rate of a grid at location r is "a
  sum of three cosine gratings", "oriented along three vectors ω1, ω2, ω3, which were 60°
  apart" (at θ − 30°, θ + 30°, θ + 90°). Spacing λ·|ω| = 4π/√3. The sum goes through a gain
  g(x) = exp[a(x − b)] − 1 (a = 0.3, b = −3/2). Solstad et al. 2006 is cited there for
  place cells built by summing grid inputs; its own equation could not be checked.
- **Banino 2018 (RL).** The grid code is *learned*, not imposed. An LSTM given velocity
  (later also noisy velocity and vision) is trained by supervision to predict place-cell and
  head-direction-cell activity, through a linear layer with 50% dropout. 25.2% of that
  layer's units became grid-like, and "grid-like representations did not emerge in networks
  without regularization (for example, dropout)". The linear layer's activity (current and
  goal "grid codes") is fed to an A3C policy LSTM. Result: the grid-cell agent beat a
  place-cell agent, A3C baselines and an expert human, generalised to larger environments
  ("only the grid cell agent was able to generalize effectively") and took shortcuts.
  Silencing grid units hurt more than silencing others. It contrasts itself with
  "approaches using pre-configured grid cells".
- **Cueva and Wei 2018.** An RNN given speed and heading, trained to output (x, y) with
  weight and metabolic penalties. Grid-like, band and border units *emerge*; square arenas
  tend to give rectangular grids. It is a representation study with no downstream task.
- **Space2Vec 2020 (explicit, fixed code).** The location goes through sin/cos of its
  projection on three unit vectors 120° apart, at S geometric scales λ_min…λ_max (Eq. 3),
  then an MLP. On a points-of-interest task, "theory" (3 directions) gets test MRR 0.144
  against 0.139 for the same code on the x and y axes ("grid", Eq. 4), 0.123 for RBF and
  0.120 for tiles (Table 1). A small edge for the hexagonal layout and a large one for being
  multi-scale and periodic.

**What it means here.** The literature supports a multi-scale periodic code with three
directions 60° (equivalently 120°) apart. lean's Fourier position code is that layout. The
evidence that it helps comes from learned grid codes in navigation (Banino) and from a
supervised geographic task (Space2Vec). It does not come from a fixed hexagonal code in RL,
which I did not find. Hexagonal over x/y is a small effect (0.144 against 0.139), so it is
kind (d) for this project, not (b).

---

## 7. Partial rotary embeddings

**Sources (read).** Official `config.json` files on Hugging Face (fetched 2026-11-22);
transformers v4.46.0 `modeling_rope_utils.py` (`_compute_default_rope_parameters`: rotated
dim = head_dim × `partial_rotary_factor`, default 1.0 when the field is absent); Black et al.,
"GPT-NeoX-20B", arXiv 2204.06745, §2.1.1 (p. 2-3); Barbero et al., "Round and Round We Go!
What makes Rotary Positional Encodings useful?", ICLR 2025, arXiv 2410.06205, §6.1, Table 2,
Appendix D; DeepSeek-V3 and Kimi K2 configs; gpt-oss card §2.2.

| Model | Config | Rotated share of each head |
|---|---|---|
| GPT-J-6B | `rotary_dim` 64, head 4096/16 = 256 | 25% |
| GPT-NeoX-20B | `rotary_pct` 0.25 | 25% |
| Phi-1.5 | `partial_rotary_factor` 0.5 | 50% |
| Phi-2 | `partial_rotary_factor` 0.4 (head 80 → 32) | 40% |
| StableLM-2-1.6B | `partial_rotary_factor` 0.25 | 25% |
| Qwen3-Next-80B-A3B | `partial_rotary_factor` 0.25, head 256 → 64 | 25% |
| DeepSeek-V3, Kimi K2 (MLA) | `qk_rope_head_dim` 64, `qk_nope_head_dim` 128 | 33% of the query-key dims (decoupled rotary part) |
| Phi-3-mini, Mistral-7B, Llama-2-7B (NousResearch mirror; Meta's repo is gated), gpt-oss-20b | No partial field → 1.0 | 100% |
| Kimi Linear | NoPE on its full-attention layers (arXiv 2510.26692, §4) | 0% there |

**Ablations.**
- GPT-NeoX-20B §2.1.1: "we follow Wang and Komatsuzaki (2021) and instead apply it only to
  the first 25% of embedding vector dimensions. Our initial experiments indicate that this
  strikes the best balance of performance and computational efficiency." No numbers given.
- Barbero et al. §6.1, Table 2 (Gemma 2B trained from scratch, 8k context, validation
  perplexity Wiki / FlanV2): NoPE 4.859 / 6.643; RoPE 4.463 / 6.443; 0.25-RoPE
  4.530 / 6.511; 0.75-RoPE (lowest 25% of frequencies removed) 4.441 / 6.442; 0.75-RoPE
  reversed (highest removed) 4.459 / 6.468; 0.75-RoPE-partial (25% of dims unrotated)
  4.454 / 6.456. Keeping 75% is as good as or slightly better than full; 25% is worse; which
  frequencies are dropped matters (drop the lowest).

**What it means here.** 25% to 50% rotated is common and harmless in language models;
the one controlled ablation finds 75% best and 25% worse. Barbero et al.'s reason (the
lowest frequencies act as position-free "semantic" channels over long contexts) concerns
1D token distance. The project rotates by 2D position at several spatial periods plus
heading harmonics, over a few dozen tokens. lean now rotates 21 pairs (18 position + 3
heading) of a 64-wide head, about 66%. A fraction carries over only as a starting point,
kind (d).

---

## Draft rows for `docs/provenance.md`

Kinds: (a) standard; (b) published beats standard; (c) measured here; (d) reasoned,
untested.

| Component | What it does | Problem it solves | Kind | Source and passage | How this project's setting differs |
|---|---|---|---|---|---|
| HL-Gauss value targets | Cross-entropy against a Gaussian (σ in bin widths) integrated over each bin | MSE value regression scales badly and is hurt by noisy targets | b | Farebrother et al. 2024, §3.1 Eq. 3.3, p. 6; §4 and §5.1-5.2; Fig. 13 (read) | Value-based RL (DQN/CQL), single reward scale, uniform bins; here a PPO critic with several reward levels |
| σ = 0.75 bin widths | Width of the target Gaussian | Too narrow ≈ lossy one-hot (worse than Two-Hot); too wide blurs | b, tuned there | Farebrother et al. 2024, p. 6; Table B.3; Fig. 13 (read) | Their optimum is a width in value units (~0.3 on [−10, 10], 51 bins); "0.75 bins" carries over only with a comparable bins-to-range ratio |
| Linear against symlog bins | Spacing of the value support | Returns spanning orders of magnitude (symlog) against resolution near typical values (linear) | d | HL-Gauss tested only linear bins (Listing 1, read); DreamerV3 used symexp two-hot (Hafner et al. 2023, Eqs. 9-11, p. 7, read) | DreamerV3's reason is one config across domains plus avoiding running normalisation; one game with a scale per level may not need it. Test: per-level return histograms over training |
| Same bin count for every level | One support shape for all reward levels | Uniform code and scaling | d | None | Untested |
| Attention output gate (G1) | Per-head (or per-element) sigmoid gate on SDPA output from the query's hidden state | Attention sinks, loss spikes, low-rank W_V·W_O | b | Qiu et al. 2025, §2.2 Eq. 5, Table 1, §4.3, Table 4, Fig. 2 (read); adopted in Kimi Linear §4 Eq. 10, Table 1 (read) | LLMs with thousands of tokens; here a few dozen tokens per query; gate init (bias 2.0) is the project's own, kind d |
| Null key / learned sink | Lets a head attend to nothing | Softmax forces the weights to sum to one | a/b | StreamingLLM §3.3, Table 3 (read); gpt-oss card §2.2 (read); Miller 2023 (read, no experiment) | Streaming LLM inference; whether sinks form over the project's short token sets is unmeasured. Dropping it is reasoned only with the gate on (d) |
| QK-Clip | Rescales W_q, W_k per head when the max logit exceeds τ | Attention-logit explosion (with Muon) | b, for Muon | Kimi K2 report §2.1 (read) | Only relevant if Muon or logit growth is seen; not proposed |
| RG-LRU recurrence | Gated diagonal linear recurrence a_t = σ(Λ)^(c·r_t), c = 8 | Per-token memory across ticks at linear cost | b | De et al. 2024, §2.4 Eqs. 1-4, App. A Eq. 6 (read) | Language modelling at 2k-8k tokens; here per-ship memory over decision steps |
| RG-LRU decay init | a^c ~ U[0.9, 0.999] (paper) or a ~ ring-uniform [0.9, 0.999] (official code) | Memory timescales from steps to hundreds of steps at start | a (paper) | De et al. 2024, §2.4 p. 4 (read); recurrentgemma `rnn_param_init` (read) | Paper and code disagree by a factor of 8 in the exponent; v2 uses neither (a^c ∈ [0.004, 0.865]); the right range depends on decision rate and episode length here |
| Griffin recurrent block | Linear → Conv1D(4) → RG-LRU, times GeLU(linear), → linear | Local mixing plus gated memory | b | De et al. 2024, §2.3 p. 4 (read) | lean's `Temporal` is a bare RG-LRU, which departs from the decision |
| Griffin block init | LeCun gates; code: N(0, 1/D) projections, depth-scaled output, Conv1D std 0.05, zero biases | Well-scaled start | a | De et al. 2024 §2.4 (read); recurrentgemma `modules.py`, `layers.py` (read) | v2 and lean use torch defaults (a third of the variance, random biases, Conv1D ±½) |
| PFSP (1 − x)² | Opponent weight from the learner's win rate x | Focus on opponents not yet beaten | b | AlphaStar Supplementary pseudocode `multiagent.py` `pfsp`, `MainPlayer.get_match` (read); Methods text could not fetch | StarCraft league with exploiters and 0.5 prior plus 0.99 decay; no uniform floor; here one policy plus a scripted member |
| Self-play share 50% | Share of games against the current policy | Fast learning against forgetting | b | AlphaStar main text Fig. 3 caption, 50/50 (read); pseudocode 50/15/35 (read); OpenAI Five App. N, 80/20 (read) | Different games and league sizes; shares were not tuned for transfer |
| Hexagonal Fourier position code | sin/cos of position on 3 directions 60° apart, multi-scale | Even resolution in every direction; multi-scale location | d | Blair et al. 2007, Methods "Cosine grating model" (read); Space2Vec Table 1, theory 0.144 against x/y 0.139 MRR (read); Banino 2018 (read, learned code); Solstad 2006 could not fetch | No fixed hexagonal code tested in RL; the edge over x/y is small and from a supervised geographic task |
| Partial rotary | Rotate only part of each head's query/key dims | Keeps position-free channels; cheaper | b (LLMs) | GPT-NeoX-20B §2.1.1 (read, no numbers); Barbero et al. 2025 Table 2 (read); HF configs (read) | 1D token distance in LLMs; here 2D position and heading over few tokens; lean rotates ~66% |
