# Rotary outside language models, and the attention-gate initialisation (2026-11-22)

Sources only: everything below was fetched and read on 2026-11-22. Paper passages were read
through arXiv HTML (or a PDF converted to text); code was downloaded and read directly, with
line numbers from the downloaded file. Where a paper page was only partly read (the fetch tool
stops at 100k characters), this is noted. Anything not fetched is listed at the end.

The language-model findings (GPT-NeoX 25%, Barbero et al. 75% vs 25%) are in
`provenance-check.md` §7 and are not repeated here.

---

## 1. Rotary or relative position outside language models

### 1a. Vision transformers

**Heo, Park, Han, Yun (NAVER AI Lab), "Rotary Position Embedding for Vision Transformer",
arXiv 2403.13298v2 (2024)**, https://arxiv.org/html/2403.13298 ; code
https://github.com/naver-ai/rope-vit (commit 48d8df5, `deit/models_v2_rope.py`).
- §3.2, Eq. 12: RoPE-Axial "divide[s] embedding dimensions into two and apply position embedding
  for the x-axis and y-axis separately"; even pairs rotate by x, odd by y. The whole head is
  rotated (no pass-through part).
- §3.2, Eq. 13: θ_t = 100^(−t/(d_head/4)), t = 0 … d_head/4; "ViT-B has 32 frequencies per head"
  (§3.3). Code lines 47-56: `compute_axial_cis`, 1/θ^(arange(0, dim, 4)/dim) → d_head/4
  frequencies per axis (16 per axis for a 64-wide head), base `rope_theta=100`, positions in
  patch units 0 … 13 (`init_t_xy`, lines 58-62). So the wavelengths run from 2π ≈ 6.3 patches
  to about 2π·100^(15/16) ≈ 470 patches on a 14-patch grid (my arithmetic).
- §3.2, Eq. 14: RoPE-Mixed learns the frequencies, "separate sets of learnable frequencies for
  each head and every self-attention layer", about 0.01% of ViT-B's parameters. The paper page
  does not say how they are initialised; the code does (lines 20-33, 140-150): magnitudes
  1/10^(k/(d/4)) (base `rope_theta=10` for every Mixed model, lines 240-261), each used along
  two perpendicular directions (angle a and a + π/2) with a random a per head; the full head is
  rotated.
- Ablations (Appendix A.1, Tables A.1-A.2): learnable frequencies help only at high resolution
  (ViT-S at 512: Axial 76.1, Axial + learned 77.8, Mixed 79.1). Combining with an absolute
  embedding (Tables 1, 3, A.4) is mixed: helps at 96×96 (68.1 → 70.2), hurts at 512 (82.9 →
  82.4); §4.1 says APE helps interpolation and reduces the extrapolation gain. **No ablation of
  the rotated share or of the band range** in the part read.

**Fang, Sun, Wang, Huang, Wang, Cao, "EVA-02: A Visual Representation for Neon Genesis",
arXiv 2303.11331v2 (2023)**, https://arxiv.org/html/2303.11331 (first 100k characters read);
code https://github.com/baaivision/EVA, `EVA-02/asuka/rope.py` and `modeling_finetune.py`.
- §2.1 / Table 2: 2D RoPE vs absolute PE, ImageNet-1K fine-tune 85.6 vs 85.2 (+0.4); "the
  standard relative position embedding suffers from unstable pre-training" (diverged).
- Code: `modeling_finetune.py` lines 515-521 build `VisionRotaryEmbeddingFast(dim=head_dim//2)`;
  `rope.py` lines 95-133: frequencies 1/10000^(2i/(head_dim/2)) per axis, positions in patch
  units, the x and y tables concatenated, so **the full head is rotated, half by x and half by
  y**. The class token is not rotated (`modeling_finetune.py` lines 213-220).

**Schenck, Reid, Jacob et al., "Learning the RoPEs: Better 2D and 3D Position Encodings with
STRING", arXiv 2502.02562v1 (2025)**, https://arxiv.org/html/2502.02562 (first 100k read).
- Generalises RoPE to learned commuting skew-symmetric generators (Def. 3.1); no rotated-share
  figure is given. Orientation is not encoded as a position input (rotation invariance is only a
  proposed extension, §3.3).
- Comparisons (not share ablations): ImageNet top-1 absolute 80.04, RoPE 80.18, RoPE-Mixed 80.86,
  Circulant-STRING 81.22, Cayley-STRING 81.09 (Table 1); ALOHA simulated MultiTask success ViT
  0.37, RoPE 0.42, STRING 0.46 (Table 5); real-robot 3D STRING ~74% vs 2D ~65% (§4.4.2).

**Ostmeier et al., "LieRE: Lie Rotational Positional Encodings", arXiv 2406.10322 (v5, 2025)**,
https://arxiv.org/html/2406.10322.
- RoPE-Mixed is LieRE with 2×2 blocks (§4). Block-size ablation (§5.4, Table 10, CIFAR-100):
  2 → 68.84, 4 → 69.28, 8 → 70.32, 16 → 69.85, 64 → 69.99; per-head, per-layer parameters best
  (Table 4). No rotated-share or frequency-range ablation.

### 1b. Poses with orientation: GTA, and the SE(2) trajectory models

**Miyato, Jaeger, Welling, Geiger, "GTA: A Geometry-Aware Attention Mechanism for Multi-View
Transformers", arXiv 2310.10375 (v3 2024, ICLR 2024)**, https://arxiv.org/html/2310.10375
(first 100k read).
- Eq. 6 (§3.1): the group representation is applied to Q, K **and V**, and the inverse to the
  output: O = P_g ⊚ Attn(P_gᵀQ, P_g⁻¹K, P_g⁻¹V).
- Eq. 8-9, App. A.2, Table 7: each head is split **2:1:1** — half the dims carry the camera
  SE(3) pose, a quarter the camera rotation (SO(3), Wigner-D l = 1, 2), a quarter the 2D image
  position as SO(2) rotations with d/16 octave frequencies, maximum 1. Concretely d = 64 uses
  4 position frequencies (1 … 1/2³); d = 96 uses 6 (1 … 1/2⁵).
- Ablations: removing the transform on V, PSNR 38.99 → 36.54 (CLEVR-TR) and 24.58 → 23.77
  (MSN-Hard) (Table 5 left); position as translation T(2) instead of SO(2) rotations 37.20 /
  23.69; one SO(2) frequency 38.82 / 23.98; several frequencies 38.99 / 24.58 (Table 6b);
  dropping camera or position encoding degrades (Table 6a); the SO(3) part gives "moderate
  improvement on MSN-Hard and no improvement on CLEVR-TR". Pure 2D (DiT-B/2 ImageNet 256,
  Table 5 right): FID 7.02 plain, 6.26 with 2D RoPE, 5.87 with GTA. The paper notes its own
  MSN-Hard number differs between tables (24.58 vs 24.80).

**Zhao, Ban, Liu, Liu, Zhou, Wang, Zhou, Qin, Yang, Li, "DRoPE: Directional Rotary Position
Embedding for Efficient Agent Interaction Modeling", arXiv 2503.15029v1 (2025)**,
https://arxiv.org/html/2503.15029v1. This is the closest match: RoPE on agent position plus a
rotation by agent heading, for multi-agent trajectory generation.
- §III-D, Eq. 20: for the heading rotation "all scalar parameters θ_l [are set] to unity", so
  every heading pair rotates by the same angle (only the first harmonic); Proposition 4: the
  score then depends only on the relative heading mod 2π. Position uses ordinary RoPE
  frequencies θ_l = 10000^(−l/d_k) (Eq. 13).
- Applied to Q and K only; V is not rotated (Eqs. 21, 25, 30).
- §IV-A: two ways to combine: **head-by-head** (even heads position, odd heads heading,
  Eq. 23-25) or **intra-head** (each head split into a position and a heading sub-vector,
  Eqs. 26-30, split sizes not given).
- Table II (§V-E), minADE (lower better) / kinematic / interactive / map-based:
  RPE (50-nearest-neighbour relative encoding) 1.3910 / 0.4820 / 0.7878 / 0.8416;
  intra-head 1.4289 / 0.4804 / 0.7843 / 0.8345; head-by-head 1.3745 / 0.4827 / 0.7894 / 0.8449.
  The authors attribute the intra-head drop to it "mix[ing] direction and positional features".
  No ablation of the share, the frequencies, or removing heading. Tokens are 64-wide (§V-B);
  head width not stated.

**Pronovost, Boloor, Schleede, Hendy, Morales, Roy (Zoox), "Linear Memory SE(2) Invariant
Attention", arXiv 2507.18597v1 (2025)**, https://arxiv.org/html/2507.18597v1.
- §III: relative heading is a standard RoPE block (Eq. 10); the relative x, y in the query's
  frame (which does not factor into query × key) is approximated by a Fourier series in heading
  with F terms (Eqs. 11-17); query side ρ(−θ_n), key side ρ(θ_m) (Eq. 19); several such blocks
  stacked with scaled x, y for "different spatial resolutions". No head share given.
- Table I (mean of 3 seeds), NLL / error on stationary / straight / turning trajectories:
  absolute positions 0.193 / 0.24 / 1.90 / 2.98; 2D RoPE 0.190 / 0.23 / 1.78 / 2.69;
  SE(2) representation (GTA-like) 0.191 / 0.23 / 1.82 / 2.70; SE(2) Fourier 0.190 / 0.23 /
  1.79 / 2.60. Gains are on turning trajectories only. The introduction claims better than GTA,
  but Table I has no row named GTA. No ablations (§V defers them).

### 1c. Trajectory prediction with relative-pose features (not rotary)

**QCNet** (Zhou et al., CVPR 2023, "Query-Centric Trajectory Prediction"). The CVF paper PDF
returned 403; read from code only, https://github.com/ZikangZhou/QCNet (commit 55cacb4).
- `modules/qcnet_agent_encoder.py` lines 170-176: for each agent pair, the edge feature is
  [distance, angle of the neighbour in the centre agent's heading frame, relative heading
  (wrapped)]; time edges add the time gap (lines 132-139).
- `layers/fourier_embedding.py` lines 23-67: each scalar gets **learnable** frequencies
  (`nn.Embedding(input_dim, num_freq_bands)`), cos/sin plus the raw value, a per-input MLP,
  summed. `predictors/qcnet.py` defaults: `num_freq_bands` 64, `hidden_dim` 128, 8 heads of 16.
- `layers/attention_layer.py` lines 93-95: the embedded relative pose is **added to keys and
  values** (k_j + W_kr r_ij, v_j + W_vr r_ij), Transformer-XL style, not a rotation.
  Ablations not read (paper not fetched).

**Wayformer** (Nayakanti, Al-Rfou, Zhou, Goel, Refaat, Sapp, arXiv 2207.05844v1, 2022),
https://arxiv.org/html/2207.05844: ego-centric frame (scene centred and rotated to the ego
agent, §3), other agents' features transformed into that frame (§2), learned positional
embeddings initialised at zero (§3); no pairwise relative pose inside attention; no ablation of
position encoding.

**MTR** (Shi, Jiang, Dai, Schiele, arXiv 2209.13508 v2, 2023), https://arxiv.org/html/2209.13508:
agent-centric normalisation (§3.1), sinusoidal PE of each token's latest position / polyline
centre added in the encoder (Eq. 2), heading as an input feature (App. A); frequency count not
given; no PE ablation (local vs global attention ablated, Table 4).

### 1d. Multi-agent RL with entity attention

**AlphaStar** (Vinyals et al., Nature 575, 2019), Supplementary Data `detailed-architecture.txt`
(downloaded from the Nature ESM zip): entity encoder lines 34-35 "x_position: Binary encoding
of entity x-coordinate, in game units" (same for y); line 66: 3-layer transformer, 2 heads,
keys/queries/values of 128. Positions are absolute per-entity input features; **no positional
encoding or relative pose in the attention**, no facing feature (grep found none).

**OpenAI Five** (Berner et al., arXiv 1912.06680v1, 2019), https://arxiv.org/html/1912.06680v1:
App. E Table 3: per unit "position (x, y, z)", "facing angle (cos, sin)", and "vector to me
(dx, dy, length)" as input features, normalised by running mean/std; App. H: sets go through a
"Process Set" module into a 4096-unit LSTM; no attention with positional terms described.

**SMAC** (Samvelyan et al., arXiv 1902.04043v5, 2019), §4: each agent sees, for units in sight
range 9, "distance, relative x, relative y, health, shield, and unit_type", normalised by
maximum; no facing. **UPDeT** (Hu, Zhu, Chang, Liang, 2021, arXiv 2101.08001, PDF text): each
observed entity is embedded as a token (§3.2, Eq. 1) from these SMAC features ("relative
distance of the observed units", App. A); no positional encoding is mentioned.

### 1e. Physics simulators with attention

**FluidFormer** (arXiv 2508.01537v1, 2025), §4.1 Eqs. 6-10: 3D RoPE, base 10000, x/y/z blocks
on Q and K; share and axis split not stated. Table 4 (§5.4): without 3D-RoPE nn-SE 29.131 vs
24.442, density error 0.017 vs 0.008.

**RigidFormer** (Dou, Guo, Wu, Roble, Stuyck, Matusik, arXiv 2605.09196v2, 2026), §3.3 Eqs. 3-4:
anchor-based RoPE, 32 log-spaced phase channels per coordinate (96-D per anchor), mean-pooled;
heads split into "a rotary part and a pass-through part" (size not stated); orientation does not
enter the rotary ("ARoPE encodes where each object is, while its shape reaches attention through
the object tokens"). Table 8: ARoPE vs sinusoidal / learned / OBB / PCA / SE(3) encodings —
best or tied position error in 8 of 9 cells, margins small (e.g. 0.161 vs 0.172 m at step 1).

### Answer to question 1

- **Share rotated.** Every vision RoPE read (Heo et al. axial and mixed, EVA-02) rotates the
  **whole head**, split half x and half y; STRING and LieRE likewise rotate the full head.
  GTA spends the whole head on geometry but divides it 2:1:1 (pose : rotation : 2D position),
  so 2D position gets only a quarter. RigidFormer has a pass-through part of unstated size.
  **No source outside language models ablates the rotated share.**
- **Bands.** Vision: d_head/4 frequencies per axis (16 per axis at width 64), base 100 for axial
  ViT (wavelengths ~6 to ~470 patches on a 14-patch grid), 10 for learnable Mixed, 10000 in
  EVA-02. GTA: octave frequencies, 4 for d = 64, highest wavelength 2π in its units. QCNet: 64
  learnable frequencies per relative-pose scalar. The only band ablations are GTA's (several
  frequencies > one frequency; rotations > translations) and Heo et al.'s learned-frequency
  result (helps only at extrapolated resolutions).
- **Orientation in the rotary.** DRoPE (heading rotation with the first harmonic only, Q/K
  only), Pronovost et al. (heading as a RoPE block plus a heading-Fourier approximation of the
  relative position in the query's frame), GTA (camera rotation on Q, K, V). The one ablation of
  how to place heading next to position is DRoPE Table II: separate heads (head-by-head) 1.3745
  minADE beat sharing a head (intra-head) 1.4289. Pronovost et al.: an SE(2) relative frame beats
  plain 2D RoPE on turning trajectories (2.60 vs 2.69) and ties it otherwise. Older trajectory
  models (QCNet) and all MARL systems read pass relative pose as features (QCNet adds it to keys
  and values; SMAC/AlphaStar/OpenAI Five put positions or offsets in the token).

**What it means here.** Outside language models, nobody leaves part of a spatial head
unrotated by design, and no one has measured the share, so lean's ~66% is neither supported nor
contradicted; full-head 2D rotation is the common default. The one direct evidence on heading
is DRoPE's: giving position and heading separate heads beat mixing them inside a head, which is
worth an arm against lean's 18 + 3 pairs in one head. GTA's value-transform ablation (−2.4 dB)
bears on the field memory: rotating values as well as keys is the step that mattered most there.

---

## 2. Gated attention initialisation

**Paper.** Qiu et al., "Gated Attention for Large Language Models", arXiv 2505.06708v1,
https://arxiv.org/html/2505.06708: no initialisation for the query-dependent gate W_θ and no
gate bias is described; the only zero-initialised gate is the input-independent ablation (§4.2,
"introduce input-independent gating by zero-initializing learnable parameters"). Trained gates
end sparse: mean score 0.116 for the elementwise SDPA gate, 0.172 headwise (Table 4); "Most
gating scores are less than 0.5" (Fig. 3).

**Official repository**, https://github.com/qiuzh20/gated_attention (commit f4c2a5f),
`modeling_qwen3.py`:
- Lines 276-279: the gate is extra output columns of `q_proj`
  (`num_heads * head_dim * 2` elementwise, `+ num_heads` headwise), `bias=config.qkv_bias`.
- Lines 361-362: `attn_output * torch.sigmoid(gate_score)`.
- Lines 733-738: `_init_weights` draws every `nn.Linear` weight from N(0, `initializer_range`)
  and zeroes biases. The released 1B checkpoints' config
  (https://huggingface.co/QwQZh/gated_attention, `1B_gate_elementwise/config.json`):
  `initializer_range` 0.02, `qkv_bias` false, hidden 2048.

**Qwen3-Next, Hugging Face transformers** (`src/transformers/models/qwen3_next/
modeling_qwen3_next.py`, last change 56fd541, 2026-11-18):
- Lines 240-242: `q_proj` outputs `num_attention_heads * head_dim * 2`, `bias=config.attention_bias`;
  lines 270-273 split it into query and gate; line 301 `attn_output * torch.sigmoid(gate)`.
- Lines 956-972: `_init_weights` has no gate-specific case (special cases only for the
  GatedDeltaNet `dt_bias`/`A_log`, RMSNorm, experts, router), so `q_proj` gets the base class
  rule, `modeling_utils.py` lines 2304-2324: weight N(0, `initializer_range`), bias zero.
- Released config (https://huggingface.co/Qwen/Qwen3-Next-80B-A3B-Instruct, `config.json`):
  `initializer_range` 0.02, hidden 2048, `head_dim` 256, no `attention_bias` field
  (configuration default false, `configuration_qwen3_next.py` line 104).

**Kimi Linear** (released code, https://huggingface.co/moonshotai/Kimi-Linear-48B-A3B-Instruct,
`modeling_kimi.py`):
- The output gate is on the **KDA (linear attention) layers only**: lines 497-503 low-rank
  `g_a_proj` (hidden → head_dim 128) and `g_b_proj` (128 → heads × 128), both `bias=False`;
  lines 596-598 `o_norm(o, g)`, a `FusedRMSNormGated` with `activation='sigmoid'`, i.e.
  sigmoid(g) ⊙ RMSNorm(o). The MLA full-attention layers have no gate (lines 437-440 go straight
  to `o_proj`).
- Lines 887-892: every `nn.Linear` weight N(0, `initializer_range` = 0.02), biases zero.
  The transformers port (`models/kimi_linear/modeling_kimi_linear.py` lines 858-874) adds no
  gate-specific init.
- Paper, Zhang et al. (Kimi team), arXiv 2510.26692 §4 Eq. 10 and §5.2 / Table 1 (first 100k
  characters read): low-rank sigmoid output gate; removing it 9.23 → 9.25 train PPL,
  5.65 → 5.67 validation; a Swish gate is worse (9.43 / 5.81). No initialisation stated.

**Other adopters found in transformers main (fetched 2026-11-22)** — all `bias=False` and no
gate-specific `_init_weights`, so N(0, 0.02) weights by the base rule:
- `qwen3_5` (Qwen 3.5): same fused `q_proj` × 2 gate as Qwen3-Next (lines 754-756, 812).
- `afmoe` (Arcee AI): separate `gate_proj` hidden → heads × head_dim (line 337), line 384.
- `step3p7` (StepFun): `g_proj` hidden → heads (per-head) or heads × head_dim (lines 693-695),
  line 737.
- `lfm3`: per-head `g_proj` hidden → heads, on by default (`use_attention_gate` true), lines
  238-241, 282.
- **Megatron-LM** (NVIDIA, main e54d389): `attention_output_gate` (transformer_config.py line
  341) appends the gate columns to the fused `linear_qkv` (attention.py lines 1936-1944), so it
  takes `config.init_method`, N(0, `init_method_std` = 0.02) (transformer_config.py lines
  553-565); applied as `x * torch.sigmoid(gate)` (lines 1844-1848). Bias init for that layer
  not read.

### Answer to question 2

Every released implementation read gives the gate **no special initialisation**: the same
N(0, 0.02) weights as its query projection and **no bias** (bias false in every released
config). The gate therefore starts centred at **sigmoid(0) = 0.5**, spread by the input.
My arithmetic, assuming unit-RMS normed inputs: Qwen3-Next and the official 1B (hidden 2048)
give logits with std 0.02·√2048 ≈ 0.9, so gates mostly 0.29-0.71; Kimi Linear's low-rank
gate gives std ≈ 0.02·√2304 · 0.02·√128 ≈ 0.22, gates about 0.45-0.55. Caveat: these are the
libraries' from-scratch init functions; none of the labs documents its pretraining init for the
gate, so this is what the code does, not a stated recipe.

**What it means here.** v2's weight 0, bias 2 (0.88, every gate equal) departs from all released
code, which starts near 0.5 with input-dependent spread and lets training drive the mean down to
~0.1-0.2. Nothing read tested an open start, so 0.88 is a defensible near-identity choice
rather than a copy; if it should match the published models, use the query projection's init
with no bias (or a bias of 0).

---

## Not settled

- **No ablation of the rotated share outside language models was found**; the vision, robotics
  and trajectory papers read either rotate the full head or do not state the split.
- QCNet's paper (CVF PDF 403) was not read, so its relative-pose ablations are unknown; only
  its code was read.
- DRoPE does not give the intra-head split sizes, the head width, or an ablation without the
  heading part; Pronovost et al. give no head share and no downstream ablations.
- Not checked: MAT (Wen et al. 2022), SMACv2, robotics control transformers beyond STRING
  (RT-1, Octo and similar), HiVT, GoRela, PRoPE ("cameras as relative positional encoding"),
  and DRoPE's code.
- Later parts of long pages were not read (GTA, STRING, EVA-02, RigidFormer, Kimi Linear past
  100k characters).
- Gate: no lab documents the gate init used in its own pretraining; Megatron's bias init for
  the gated `linear_qkv` (if `add_qkv_bias` is on) was not read.
