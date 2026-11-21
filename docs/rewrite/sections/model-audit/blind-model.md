# Blind review of the Yemong (v2) model architecture

Reviewer's basis: the inventory (sections 1-6, 12, "Options present but off", "v1 equivalents", "Uncertain") and three spot-checks of source (`models/yemong/temporal.py:82-87`, `models/yemong/model.py:424-431, 575, 1140-1152`, `contracts/config.py:118-213`). Nothing under `docs/` was read. Nothing was run: every verdict below is reasoning, not measurement.

Citation tags: [FETCHED] = I queried the arXiv API by id this session and the returned title matches (I checked the title only, not the paper's content; what I say the paper shows is still from memory). [MEMORY] = recalled, not checked.

Provenance categories: a = standard choice; b = published work shows it beats the standard for a comparable purpose; c-needed = only a project measurement could justify it; d = reasoned, untested; none.

## Three findings that drive most verdicts

1. **The model has never seen a count other than its training count, and nothing is masked.** Weights are size-free (12.1, 2.9), which is the necessary condition and it is met. But the sufficient condition is that the statistics each token sees at 50v50 resemble training, and they will not: one softmax over 26 keys becomes one over 116 (12.5); dead, never-seen and absent entities are live keys stacked at the map centre (12.6, 12.29); teams are always full and equal (12.3); the arena has two values (12.33). The cheapest remedy is standard and is not in the design: key masks on every non-existent entity, and training on a mixture of sizes (which the masks make possible in one batch, removing 12.46).
2. **Almost all the cost and almost all the bespoke machinery is the distributional state code and the belief loop.** Once the belief goes, the reason for a 542-wide probabilistic input code goes with it (there is no uncertainty left to blur), and so do the extrapolation baseline, the rates, the delta decode, the correction decode and the feedback. By the inventory's own counts these are 3,111 + 10,626 (which includes 9,068 correction and 1,519 delta decode) of 27,019 ops per rollout tick.
3. **The trunk is good and close to published work.** Per-kind entity encoders, a set transformer with no slot parameters, rotary relative position, and a Griffin recurrent block per token. I would keep nearly all of it.

---

## Row-by-row verdicts

### 1a. Raw observation

**1.1 Ship rows (29 columns, world coordinates, own team first)**
- Purpose: full kinematic and status state per entity.
- Standard?: an entity list in world coordinates is standard for one-forward-per-team entity models (AlphaStar, Vinyals et al. 2019, Nature [MEMORY]; OpenAI Five [FETCHED 1912.06680]). The alternative standard, egocentric per-agent observation (Baker et al. [FETCHED 1909.07528]), costs one forward per ship and is ruled out by the cost goal.
- Size generalisation: neutral, except "own team's S rows first", which is a positional convention (see 2.8).
- Verdict: KEEP the columns; REPLACE the ordering convention with an explicit per-row team column plus an `exists` mask so that rows can be padded and teams unequal. `spawned`, `benched`, `visible` collapse into that mask plus one `visible` flag.
- Provenance: a. Confidence: high.

**1.2 Columns observed but unread (`proper_speed`, `time_since_seen`)**
- Verdict: DROP `proper_speed` from the contract. `time_since_seen` becomes the input once the model's own belief age (4.1) is gone, so USE it rather than drop it.
- Provenance: a (dead inputs are removed). Confidence: high.

**1.3 Field rows** — KEEP. Standard entity rows. The `present` column must become a key mask (see 3.4). Provenance a. Confidence high.

**1.4 Zone rows (fixed Z = 5, 5 roles)**
- Purpose: objective state.
- Size generalisation: hurts. Z = 5 is a contract constant, the role one-hot has exactly 5 classes (12.28), and training shows either all 5 or none (12.29). "Different numbers of capture zones" is a stated goal and is untrained.
- Verdict: SIMPLIFY the contract to variable Z with a mask. Replace the 5-way role with features that survive other zone counts: owner (own / neutral / enemy, 3-way) and is-spawn (1 bit). The `offensive_distance` / `defensive_distance` (captures to win/loss) are counts that grow with Z; keep them but as clipped scalars (min(d, 4)/4), since "far from winning" is all that needs to extrapolate.
- Provenance: d. Confidence: medium (I have not read how roles are used by the mode rules).

**1.5 Global row (15 columns)**
- Size generalisation: `score_own/target` and `score_enemy/target` are ratios and fine. `symlog(score_target)` is 25 in training and 100-250 at scale (12.9): one constant per mode in training, so the model has learned it as a mode flag, and at scale it is out of distribution. `arena_radius`: see 1.28/1.29. `fogged` is constant 1 (1.7).
- Verdict: SIMPLIFY. Drop `fogged` and `symlog(score_target)`; keep ratios, mode flags, progress, front, boundary fraction.
- Provenance: d. Confidence: medium-high.

**1.6 No bullet observation**
- Purpose (inferred): bullets live 11 ticks (about 0.37 s); a bullet token set would be large and too short-lived to act on. The `fired` flag carries the event.
- Standard?: v1 had bullets; many shooters observe projectiles. No source known for either choice in this regime.
- Size generalisation: helps (bullets scale with N and would multiply keys).
- Verdict: KEEP. Provenance: d. Confidence: medium. Would change if bullets are slow relative to ship manoeuvre (see "What the inventory does not tell me").

**1.7 Fog share 1.0** — KEEP if the human always plays fogged; then the `fogged` input is a constant and goes. Provenance a (train on the deployment distribution). Confidence high.

### 1b. Ship-token feature groups

**1.8 + 1.20-1.25 State code as input (542 wide, 23 softmax groups) — judged together**
- Purpose: represent each physical quantity as a soft categorical so that a belief's uncertainty can be shown as blur, and so that inputs and prediction targets share one code.
- Standard?: No. The standard entity input is a short vector of scaled scalars: sin/cos for angles, value/max for bounded quantities, symlog for unbounded ones (DreamerV3 [FETCHED 2301.04104] for symlog; OpenAI Five and AlphaStar for the general recipe). There is published support for binned / piecewise-linear / periodic encodings of scalars beating raw scalars, but in tabular supervised learning (Gorishniy et al. [FETCHED 2203.05556]), not in RL entity encoders; no source known for RL.
- Size generalisation: neutral.
- Verdict: REPLACE with about 20 scalars per ship: heading cos/sin; velocity as symlog-radial (vx, vy)/100 (the existing radial log compression is fine as a scalar transform) plus speed; nose rate / 8; angle of attack / 0.54; shield and energy as fractions; recharge wait / 3; cooldown (1 scalar); log index / ln 2; gradient x, y. The 99-bin three-axis velocity code, the 4x4 heading code, the two-hot/moment-matched forms and `match_moments` all go. With the belief removed there is nothing to blur, which was the only thing a scalar could not express.
- If the project has a measurement that the code beats scalars as an *input* with the belief off, that is category c and it may stay as an experiment arm; without it, it has no provenance.
- Provenance: scalars a; the existing code c-needed. Confidence: high that scalars are the right baseline; medium that nothing is lost.

**1.9 Team one-hot** — KEEP (one bit suffices). a. High.

**1.10 Ship-type one-hot** — KEEP. a. High. Structural only in the number of types (12.25), which is not a goal.

**1.11 Flags (alive, visible, fired, benched)** — SIMPLIFY: keep alive, visible, fired as inputs; `benched` becomes a mask, not an input (it is always 0 in training, so as an input it is untrained, 12.3). a. High.

**1.12 Time since seen (symlog)** — KEEP, fed from the observation column. Only meaningful if hidden-but-seen enemies remain as tokens (see "ghost tokens" in the baseline). a. High.

**1.13 Respawn wait: scalar + 4 harmonics (9 inputs)** and **1.30 Timer code (9 inputs x 4 uses)**
- Purpose: let a linear first layer resolve a countdown finely.
- Standard?: a clipped scalar. Harmonics of a 16 s horizon are Fourier features (Tancik et al. [FETCHED 2006.10739]); justified for high-frequency functions of the input, which a respawn countdown is not.
- Size generalisation: neutral (12.44 only if clocks change).
- Verdict: SIMPLIFY to min(t, 16)/16, one input each. Provenance a. Confidence: medium-high.

**1.14 Pending action (42 inputs)**
- Own rows, previous action: standard for recurrent agents (IMPALA, Espeholt et al. 2018, arXiv:1802.01561 [MEMORY]). KEEP.
- Enemy rows, stored forecast: part of the belief; goes. Enemy rows read zeros.
- Provenance: a (own half). Confidence: high.

**1.15 Refractive-index gradient** — KEEP as two scaled scalars. Game physics the ship needs. a. High.

**1.16 Local presence (2 inputs, 150 m Gaussian sum, N x N)**
- Purpose: tell a ship how crowded its surroundings are. This is a real gap in softmax attention, which averages and cannot count; an explicit count is the usual patch.
- Standard?: No; hand-built feature. No source known.
- Size generalisation: hurts mildly. It is a sum, so its range rises with local density (12.8); ln(1 + .) softens it but values above the training maximum are extrapolation, and Xu et al. [FETCHED 2009.11848] is the standard reference for why MLPs extrapolate poorly outside the training range.
- Verdict: DROP from the baseline; it returns as an experiment. Cheapest test, evaluation only on the existing model: (i) zero both inputs at 5v5 and see whether win rate against Imugi moves; (ii) at 20v20 clamp them to the training maximum and compare with unclamped.
- Provenance: none (c-needed). Confidence: medium.

**1.17 Row with no belief placed at the map centre** — DROP; mask the row instead (3.4). It is the single worst size interaction in the inputs: at the start of a fogged 50v50 there are 50 identical unmasked keys on top of the global token. a (masking). High.

### 1c. Encodings

**1.18 Random position offset per view (uniform over 236 km)**
- Purpose: make absolute position uninformative so the model can only use relative position; this is translation invariance by augmentation, and it also makes the position inputs independent of arena size.
- Standard?: translation augmentation is standard in spirit; this exact form is bespoke. For the rotary bands it is a mathematical no-op (relative phase is already invariant). It matters only for position features fed as token inputs.
- Size generalisation: helps (the one mechanism in the inputs designed for it).
- Verdict: KEEP if absolute position features remain as inputs (I keep them, see 1.19); otherwise drop. Provenance d. Confidence: medium-high.

**1.19 Position colour code (9 levels x 3x3, 81 values, belief blur)**
- Purpose: a multi-resolution periodic code of absolute (offset) position, so that two tokens' codes can be compared for relative displacement inside the MLP, with more precision than attention weights alone give.
- Standard?: No. The standard for the same job is sin/cos Fourier features at geometric frequencies (Vaswani et al. 2017, arXiv:1706.03762 [MEMORY]; Tancik et al. [FETCHED 2006.10739]). The code's own "fourier" option exists and is off, so the project may have measured this.
- Size generalisation: neutral for fine levels (all phases seen through the offset); the coarsest levels (periods of tens to hundreds of km) barely turn across a training view (12.38), so they are an untrained input direction that starts moving on a larger map.
- Do I need position inputs at all, given rotary attention? I think yes, at reduced size. Rotary puts relative position in the attention *weights* only; values carry none (3.3). Precise bearing for aiming then has to be read from which of 2 heads x 4 layers fired, which is coarse. Position features in the token let the gated MLP (which is multiplicative) form phase differences. The existence of a geometry head (5.10) suggests the designers also found pair geometry was not free.
- Verdict: REPLACE with Fourier features, sin/cos per axis at about 6 periods from 64 m to about 16 km (24 inputs), no blur. Cap the longest period near the largest separation actually trained. TEST (one training arm each): colour code vs Fourier vs no position input.
- Provenance: Fourier a; colour code c-needed. Confidence: medium.

**1.26 Rotary position bands (8 bands, 64 m to 340 km, axes turned 35 degrees per band)**
- Purpose: translation-invariant relative position in attention.
- Standard?: RoPE is standard for sequences (Su et al. [FETCHED 2104.09864]); 2D variants with mixed-axis frequencies are published for vision (Heo et al. [FETCHED 2403.13298]). A continuous-metre, geometric-ladder, rotated-axis version is a reasonable adaptation; the specific ladder (ratio 3.405, 35 degrees) is bespoke.
- Size generalisation: helps for count; mixed for arena. Bands with period above about 30 km turn through less than half a cycle in training (12.38), so the model can use them as near-linear range features; on a 4x arena those features leave their trained range. Also RoPE has no built-in decay with distance, and in language it is known to extrapolate weakly beyond trained lengths while a simple additive distance penalty extrapolates well (ALiBi, Press et al. [FETCHED 2108.12409]). That is an analogy from 1D sequences, not evidence about this domain.
- Verdict: KEEP rotary; SIMPLIFY the ladder to bands whose period is at most about 4x the largest trained separation (drop the 340 km and 100 km bands, or keep them and sample arena sizes in training). TEST as experiments: (i) the inference ablation switch "rotary off" already exists, which gives a zero-cost reading of how much the trained model depends on it; (ii) an additive per-head distance bias, -m_h x distance / 1 km, passed as the float mask of `scaled_dot_product_attention`.
- Provenance: b (by analogy; "comparable purpose" is generous). Confidence: medium.

**1.27 Rotary heading harmonics (3 pairs)**
- Purpose: let attention depend on heading difference between two ships.
- Standard?: No; no source known. Heading difference is the crossing angle only; the tactically important angles (bearing off own nose, aspect) need bearing as well, which this does not supply.
- Size generalisation: neutral.
- Verdict: DROP from the baseline (heading stays as cos/sin input); experiment later. Frees 6 of 64 dims per head.
- Provenance: none (c-needed). Confidence: medium-low.

**1.28 Arena code (25-bin HL-Gauss of ln radius) and 1.29 raw arena scalar**
- Purpose: tell the model how big the map is.
- Standard?: one scalar. Two encodings of a quantity that takes two values in training (12.33) is redundant, and neither can generalise: the model has seen two points, so both act as a mode flag.
- Size generalisation: hurts. Any other arena lights untrained bins (12.34) or an untrained scalar range (12.35).
- Verdict: REPLACE both with a local, saturating feature on each ship: distance to the boundary, clipped at about 2 km and divided by 2 km, plus the unit vector toward the map centre (3 inputs). This is what a ship acts on, and it is identical on any arena. Keep the global token at the map centre for rotary purposes. Independently, turn on arena-size sampling in training (`arena_scale` exists and is off).
- Provenance: d. Confidence: medium-high that the current pair is wrong; medium on the replacement.

**1.31 symlog scalars** — KEEP. b (DreamerV3 [FETCHED 2301.04104]). High.

### 2. Tokenisation and embeddings

**2.1 Ship token** — KEEP the concept (one token per ship, a query). Its input shrinks from 607 to roughly 70 (20 state + 24 position + 3 boundary + 1 team + 3 type + 3 flags + 1 time since seen + 1 respawn + 12 or 42 previous action). a. High.

**2.2 Global token (136 inputs, at the map centre, heading 0)**
- Standard?: a summary token attending over a set is standard (the [CLS] idea; pooling by attention in Set Transformer [FETCHED 1810.00825]).
- Size generalisation: neutral to mildly hurts: its softmax pooling is an average, so fractions (alive share per team) survive a size change but counts do not. Its 81-wide position code and 25-wide arena code go.
- Verdict: KEEP, with about 15 scalar inputs. a. High.

**2.3 Field tokens and 2.4 zone tokens as memory-only (keys/values, never queries)**
- Purpose: static or slow entities do not need a recurrent state or outputs; this saves the quadratic part and the recurrence for them.
- Standard?: equivalent to encoder-memory cross-attention (Vaswani et al. [MEMORY]). Sound.
- Size generalisation: helps (cost is linear in map entities).
- Verdict: KEEP. Inputs shrink to position features plus 3 scalars (fields) and about 10 scalars (zones). Field scales in metres (12.31) are fine and consistent with "metres, not normalised".
- Provenance: d (close to a). Confidence: high.

**2.5 Absent-zone learned vector at the map centre**
- Standard?: No. The standard is a key padding mask.
- Size generalisation: hurts. Five identical unmasked keys in two of three modes; partial zone sets never seen (12.29).
- Verdict: REPLACE with a key mask. A mode with no zones is already signalled by the mode flags. Evaluation-only test available now: the `absent_zones` ablation (`model.py:1142`).
- Provenance: a. Confidence: high.

**2.6 Kind projections (per-kind Linear to 256, norm, GELU, shared Linear to 128, norm)**
- Standard?: Yes: per-type entity encoders into a shared width (AlphaStar [MEMORY]; OpenAI Five [FETCHED 1912.06680]). No slot or index embedding is correct.
- Verdict: KEEP the structure; SIMPLIFY the width (inputs of 15-70 do not need a 256 first layer; go straight to 128, or 128 then 128). The encoder FLOPs (18.72 M of 60.6 M) fall roughly with input width.
- Provenance: a. Confidence: high.

**2.7 Map-memory projection to 64 wide, RMSNorm per sublayer**
- Purpose: cheaper keys/values for map tokens.
- Verdict: SIMPLIFY: keep map tokens at trunk width, normalise once in the encoder, one bias-free key/value projection per sublayer. The saving from 64-wide is negligible at 15 tokens and it adds a second width to reason about. The record's `map_norm 79.96 M` FLOPs cannot be right for an RMSNorm over 15 x 64 values (the listed sub-items sum to more than the block total), so I do not trust the cost attribution here.
- Provenance: a. Confidence: medium.

**2.8 Token order and slicing by position; S from `previous_action` width**
- Size generalisation: breaks for unequal teams (12.2); the only route is benched slots, never trained (12.3).
- Verdict: REPLACE. Every head runs on every ship token; which rows a loss or the sampler uses is decided by masks built from the team column and `exists`. This also removes the play/learn difference in which rows are computed.
- Provenance: a. Confidence: high.

**2.9 Nothing indexed by slot** — KEEP. This is the property the headline goal rests on (permutation equivariance: Deep Sets, Zaheer et al. 2017, arXiv:1703.06114 [MEMORY]; transformers used this way for variable agent counts: UPDeT [FETCHED 2101.08001], REFIL [FETCHED 2006.04222]). a. High.

**2.10 Input-column initialisation by code group** — DROP; it exists only because of the code inputs. Default initialisation on scalar inputs. a. High.

### 3. Trunk

**3.1 Block layout (2 blocks; temporal, then 2 spatial; width 128)**
- Standard?: the common published layout is a spatial entity encoder followed by one recurrent core (AlphaStar, OpenAI Five). Interleaving, with memory first, is a reasoned variant: under fog, a token should recall before it communicates.
- Size generalisation: neutral. Four rounds of dense attention is enough for information to cross any fleet.
- Verdict: KEEP. The trunk is the cheap part; there is no cost case for touching it. Provenance d. Confidence: medium.

**3.2 Spatial sublayer (pre-norm attention over queries + map memory in one softmax, 2 heads of 64, gated MLP)**
- Standard?: pre-norm attention is standard. Two heads of 64 is unusual (4 x 32 is the usual choice at width 128) and follows from spending 38 dims per head on rotary pairs. One softmax shared by ships and map tokens means the share of attention the map gets falls as ships are added (15 of 26 keys in training, 15 of 116 at 50v50).
- Size generalisation: hurts (12.5).
- Verdict: KEEP the sublayer; TEST two changes as separate arms: (i) 4 heads of 32 with fewer rotary pairs (needs training); (ii) a separate softmax for map tokens, i.e. true cross-attention, so that map attention does not depend on ship count (needs training). The size remedy I would put in the baseline is masks plus mixed-size training, not a structural change.
- Provenance: a for the sublayer; d for one-softmax. Confidence: medium.

**3.3 Rotary attention; values not rotated; no additive bias** — KEEP (see 1.26). Standard RoPE form [FETCHED 2104.09864]. Consequence already noted: geometry reaches values only through inputs, hence 1.19 stays. b. Medium.

**3.4 No attention masking in training**
- Standard?: No. Padding/key masks are standard (Vaswani et al. [MEMORY]; masked entity attention in Baker et al. [FETCHED 1909.07528]); v1 masked dead entities.
- Size generalisation: hurts badly (12.6, 12.29, 12.30), and blocks mixed-size batches.
- Verdict: REPLACE: a key mask covering padding, benched, absent fields/zones, never-seen enemies, and dead ships that will not respawn. Dead ships awaiting respawn stay as queries (the critic trains on them, 7.12) but are masked as keys. The global token is always a valid key, so no query has an empty key set. The two runtime switches become the only path, trained on.
- Provenance: a. Confidence: high.

**3.5 No neighbour limit (dense attention)**
- Standard?: dense is standard at these sizes.
- Size generalisation: hurts. Softmax attention disperses as the number of keys grows, so a head that picked out one neighbour among 25 is less sharp among 115 (Veličković et al., "Softmax is not enough" [FETCHED 2410.01104]).
- Verdict: KEEP dense in the baseline (simplest), train on mixed sizes, and TEST a nearest-K key mask. The test is evaluation-only on the existing model: at 20v20, restrict each query's ship keys to its 9 nearest ships (the training count) and compare win rate with dense. If nearest-K wins clearly, the baseline should train with it.
- Provenance: a (dense); nearest-K d. Confidence: medium.

**3.6 Fused Triton attention kernel** — DROP (decided). `scaled_dot_product_attention` with the mask. It also removes 12.7. a. High.

**3.7 Gated MLP (GeGLU, hidden 512, shared gate/up matrix, no biases)**
- Standard?: this is the gated MLP of Griffin [FETCHED 2402.19427] and of GLU variants (Shazeer 2020, arXiv:2002.05202 [MEMORY]).
- Verdict: KEEP. b. High. (A plain GELU MLP would be category a and equally fine; not worth a change.)

**3.8 Temporal sublayer (norm, Linear, 4-tap causal conv, RG-LRU, GELU-gated branch, Linear) and 3.9 RG-LRU**
- Standard?: this is Griffin's recurrent block, including the width-4 temporal convolution and the c = 8 exponent [FETCHED 2402.19427]. The RL standard is an LSTM or GRU per agent (OpenAI Five, AlphaStar, MAPPO [FETCHED 2103.01955]). Griffin's evidence is for language modelling, so "comparable purpose" is a stretch; what it buys here is a diagonal recurrence that is cheap and stable in a long scan.
- One departure I would change: the decay initialisation. `log_decay = linspace(0, 4)` with a = exp(8 r logsigmoid(Lambda)) gives, at r = 0.5, per-tick retention between about 0.06 and 0.93 (I computed these from the formula), i.e. memory of at most roughly half a second at 30 Hz at initialisation. Griffin initialises so that a^c lies between 0.9 and 0.999 [MEMORY for the exact numbers]. Once the hand-built belief is removed, this recurrence is the only memory of hidden enemies, and it needs time constants of seconds to tens of seconds (hundreds of ticks).
- Size generalisation: neutral (per token, no mixing across tokens; needs row identity to be stable in time, which slots give).
- Verdict: KEEP the block; REPLACE the initialisation with a^c spread over about [0.9, 0.9995]. The Triton scan goes; the chunked scan stays.
- Provenance: b for the block; d for the initialisation change. Confidence: medium-high for the block, medium for the initialisation.

**3.10 Trunk carry** — KEEP. Required by 3.8. a. High.

**3.11 RMSNorm everywhere** — KEEP (Zhang & Sennrich 2019, arXiv:1910.07467 [MEMORY]; used in Griffin). No batch statistics, which is right for size transfer (12.48). a/b. High.

**3.12 No final trunk norm; each head group does Linear then RMSNorm**
- Standard?: pre-norm stacks end in one final norm. The per-head norm does the same job.
- Verdict: SIMPLIFY (minor): one final RMSNorm, heads as Linear-GELU-Linear. Low stakes. a. Medium.

**3.13 Segment evaluation with stored carry** — KEEP. Standard recurrent PPO: stored initial state, backprop through the segment (R2D2, Kapturowski et al. 2019, ICLR [MEMORY], for the stored-state idea). The belief replay record disappears. a. High.

**3.14 Initialisation keyed by seed and parameter path** — SIMPLIFY to framework defaults under a seed; path-keyed draws serve bit-exact reproducibility, which is no longer required. a. High.

### 4. Belief and forecast machinery (4.1-4.9)

Removal is decided; I agree on the merits (it is 40% of the rollout tick, has no provenance in categories a/b, and its coarse fixed constants such as the 26 km error bound in 12.39 are arena-dependent).

What else becomes unnecessary once it goes:
- The probabilistic input code and everything that blurs it: 1.19's blur, 1.20-1.25, `match_moments`, spreads, `spread_fit`, `spread_inputs`, 2.10.
- 1.17 (centre placement), the enemy half of 1.14, the model's own age counter in 1.12, the valid-belief rule in 1.16.
- 4.3 extrapolation baseline and 4.4 rates and held rates, provided the next-state head predicts change from the current state instead (see 5.4). The observation's acceleration columns then have no reader and can leave the contract too.
- 4.5 delta decode at play (a training-only head needs no decode), 4.6, 4.7, most of 4.8 (the replay record keeps only previous own actions and the offset), the non-finite forecast guard, the int16 fixed-point action forecast.
- 5.5 and its loss; 5.6 at play (the enemy-action head no longer feeds anything).
- At play the model computes policy, value and outcome only.

One thing must replace it, and the choice is consequential. "Hidden enemies leave the observation; recurrent memory carries them" has two readings:
- (A) the hidden enemy's token is masked out entirely. Memory of it lives in the own ships' and the global token's recurrent state: a fixed 128 numbers per token regardless of how many enemies are hidden. Capacity per remembered enemy falls as fleets grow.
- (B) a hidden enemy that has been seen since spawning keeps its token ("ghost"): inputs zeroed except `visible = 0` and time since seen, positioned for rotary at its last seen position, and its own per-token recurrent state holds what was known. Memory then scales with the number of entities, which is the size-generalising form. A never-seen enemy is masked.
I recommend (B) as the baseline (category d, no source known) and (A) as the comparison arm. (B) costs nothing extra, because the per-token recurrence already exists.

### 5. Output heads

General form (Linear, RMSNorm, GELU, Linear; orthogonal, last layer gain 0.01): standard PPO practice (Andrychowicz et al. [FETCHED 2006.05990]; Huang et al. 2022, "The 37 implementation details of PPO", ICLR blog track [MEMORY]). KEEP. Sharing a first matrix across heads (`HeadGroup`) is harmless; keep or drop by code simplicity.

**5.1 Policy head (42 logits per own ship)** — KEEP; see 6.2. Run on all ship rows, mask in the sampler. a. High.

**5.2 Value head (4 per-ship levels x 77 symexp bins)**
- Purpose: categorical critic; one head per reward family so each can have its own discount.
- Standard?: the standard is one scalar value with normalisation (PopArt / value normalisation in MAPPO [FETCHED 2103.01955]). A two-hot categorical critic over symexp bins is published and argued to be better than regression (DreamerV3 [FETCHED 2301.04104]; Farebrother et al. [FETCHED 2403.03950]; Imani & White [FETCHED 1806.04613]). Separate heads per reward component with separate discounts are published as ideas (Hybrid Reward Architecture, van Seijen et al. 2017, arXiv:1706.06208 [MEMORY]) but are not standard and I know of no result showing a benefit in on-policy multi-agent PPO.
- Size generalisation: none at play (the critic does not act). In training at other sizes, the fixed support of +-19 reward units clips (12.16); DreamerV3's support is far wider.
- Verdict: SIMPLIFY: categorical two-hot kept; four per-ship levels collapse to one per-ship value with one discount, plus the team outcome head (5.3). Widen the support (symexp of linspace(-5, 5) reaches about +-147 at the same bin count). The 4-level version returns as an experiment.
- Provenance: b for the categorical form; c-needed for multiple levels. Confidence: medium.

**5.3 Outcome head (4-way: win, tie, loss, unresolved, on the global token)**
- Purpose: team-level win probability with discounting expressed as mass on "unresolved"; exactly a categorical critic for a terminal reward in {+1, 0, -1}.
- Standard?: a win-probability head is common (OpenAI Five [FETCHED 1912.06680]); using it as the value of the outcome reward is a clean special case of a categorical critic.
- Size generalisation: helps (one team-level output, independent of S; avoids each ship separately regressing a shared quantity).
- Verdict: KEEP. Provenance d. Confidence: medium-high.

**5.4 Next-state head (500 logits; 12 symlog HL-Gauss codes of the change from an extrapolation baseline, per-component knees and bounds, measured bias table)**
- Purpose: auxiliary dynamics prediction to shape the representation; formerly also the belief's forecaster, which is why it needed calibrated distributions and a decode.
- Standard?: auxiliary prediction losses are well supported in general (UNREAL, Jaderberg et al. [FETCHED 1611.05397]). The standard target form is a regression; DreamerV3 uses symlog squared error for vector predictions [FETCHED 2301.04104]. Categorical HL-Gauss targets are supported for value learning [FETCHED 2403.03950, 1806.04613]; for auxiliary dynamics, no source known. The extrapolation baseline, the per-component knees and the marginal bias table are all bespoke.
- Size generalisation: neutral (training only, per row).
- Verdict: SIMPLIFY. With no decode and no belief: target = change from the *current* state (persistence), in the ship's heading frame, symlog-scaled; one uniform bin layout (for example 12 x 33). That removes 4.3, 4.4 and both `EXTRAPOLATION` tables. Extend the same head to ghost tokens with privileged truth (where is the hidden ship now, relative to where it was last seen): this is the learned replacement for the belief and gives the recurrent memory a direct training signal. TEST later: a longer horizon (0.5 s) instead of one tick, since a one-tick change is mostly determined by the current velocity.
- Provenance: d (auxiliary prediction a/b in general; this form untested). Confidence: medium.

**5.5 Error (correction) head** — DROP (decided). It is also the most arena-dependent head (12.39). High.

**5.6 Enemy-action head**
- Purpose: opponent modelling; at play it fed 1.14.
- Standard?: opponent-action prediction as an auxiliary task is published (Hernandez-Leal et al. 2019, "Agent Modeling as Auxiliary Task for Deep RL", arXiv:1907.09597 [MEMORY]).
- Verdict: DROP from the baseline, since its play-time role is gone and for visible enemies it duplicates the next-state head (an enemy's next-tick change is its action). It is cheap, so it is a good first experiment to add back.
- Provenance: b (weak) / c-needed. Confidence: medium.

**5.7 Density head** — DROP (probable decision; I agree). It is the most size-dependent head: rates scale with fleet size, the bias is set from 10 ships, the normaliser uses own-ship count, the grid scales with the arena (12.10, 12.11, 12.41). The ghost-token target in 5.4 covers its purpose (knowing where unseen enemies are) per entity instead of per grid cell. High.

**5.8 Shot-outcome head (19 logits: hit enemy / hit ally / 8 + 8 near-miss bins / clear)**
- Purpose: a dense aiming signal: "if I fired now, what would happen".
- Standard?: No source known. Reasonable in principle. Its label is counterfactual and needs the bullet re-simulation (8.11), which is 311,664 of 353,330 ops in the learner's prepare phase: by far the most expensive thing in the learner, and it stays even when the trajectory head goes.
- Size generalisation: neutral (cost linear in N).
- Verdict: KEEP (owner's decision), but TEST the cheap label: train only on bullets actually fired, with hit / ally hit / miss from game events (3 classes, no re-simulation). If aiming quality is unchanged, the re-simulation leaves the baseline.
- Provenance: c-needed. Confidence: medium.

**5.9 Shot-trajectory head** — DROP (decided). High.

**5.10 Geometry head (pairs: range, 3 angles, closure; 241 logits; K = 8 partners cycling)**
- Purpose: force the trunk to make pairwise geometry decodable, because rotary attention does not deliver it in values.
- Standard?: No source known for this head. The standard way to give a set model pair geometry is as an input: pairwise edge features or a relative-position bias in attention (Shaw et al. 2018, arXiv:1803.02155 [MEMORY]).
- Size generalisation: neutral (training only; 12.12 is coverage per tick, harmless).
- Verdict: KEEP (owner's decision) as the cheaper of the two routes; SIMPLIFY nothing essential. TEST by ablation whether it earns its place: one arm without it. The 64 km range bound (12.40) is irrelevant at training sizes.
- Provenance: c-needed. Confidence: medium.

**5.11 Head evaluation at play** — becomes policy + value + outcome only. KEEP the principle (auxiliary heads are training-only). a. High.

### 6. Action space and sampling

**6.1 Action space (3 x 7 x 2)** — KEEP. Discretised controls are standard and work well with PPO (Tang & Agrawal, arXiv:1901.10500 [MEMORY]). a. High.

**6.2 One 42-way joint categorical per ship; ships independent**
- Standard?: the usual choice for multi-discrete actions is independent factor heads or autoregressive heads. A joint categorical over 42 is exact, strictly more expressive than independent factors, and trivially cheap. Independent sampling across ships with shared parameters is standard (MAPPO [FETCHED 2103.01955]; OpenAI Five [FETCHED 1912.06680]).
- Size generalisation: structurally none (12.23). Behaviourally there is a risk: with no coupling, the number of ships independently choosing the same target grows with N.
- Verdict: KEEP. Provenance d (simple and exact). Confidence: high.

**6.3 Inverse-CDF sampling with three keyed uniforms** — SIMPLIFY to one categorical draw (Gumbel-max or `multinomial`) from a seeded generator. The chain-rule sampling and decision-keyed uniforms serve bit-exact replay. a. High.

**6.4 No action masking** — KEEP. Masking is standard when many actions are invalid (Huang & Ontañón [FETCHED 2006.14171]); here nothing is strictly invalid and dead ships are masked in the loss. a. High.

**6.5 Temperature 1 in training** — KEEP. a. High.

**6.6 No action repeat (decisions at 30 Hz)**
- Standard?: acting every few frames is standard in real-time game RL (OpenAI Five acted every fourth frame [FETCHED 1912.06680, detail MEMORY]; v1 used repeat 2).
- Verdict: TEST repeat 2 (15 Hz decisions). It halves model calls per game second and halves the horizon the critic has to span. The gun fires at most every 2 ticks, so fire rate is not lost. One training arm; compare win rate against Imugi at equal wall time.
- Provenance: c-needed. Confidence: medium.

---

## 1. The baseline model I would build

Everything in metres; no normalisation by the arena. Tags in brackets.

**Observation contract**
- Ship rows: state columns as now minus `proper_speed` and the acceleration columns; plus explicit `team`, `exists` (replaces benched/spawned), `visible`, `alive`, `fired`, `respawn_wait`, `ship_type`, `time_since_seen`. Rows may be padded; teams may differ in size. [a]
- Field rows and zone rows: variable count with `present` used as a mask. Zone owner 3-way + is-spawn bit instead of the 5-role one-hot. [d]
- Global row: episode progress, score ratios, mode flags, front, front threshold, boundary fraction, clipped timers. No `fogged`, no raw score target, no arena radius. [d]

**Inputs and encoding**
- Ship token, about 70 inputs: heading cos/sin; radially log-compressed velocity (2) and speed; nose rate, angle of attack, shield and energy fractions, recharge wait, cooldown, log index, gradient (2), all as scaled scalars [a]; Fourier position features of the offset position, sin/cos per axis at 6 periods from 64 m to 16 km (24) [a]; random per-match position offset [d]; distance to boundary clipped at 2 km and unit vector to the map centre (3) [d]; team bit, type one-hot, alive/visible/fired flags [a]; symlog time since seen [a]; clipped respawn wait [a]; one-hot previous action for own rows, zeros for enemies [a].
- Hidden enemies: seen since spawn = ghost token (state inputs zero, `visible = 0`, time since seen, rotary position = last seen position, own recurrent state kept); never seen = masked. [d]
- Global token: about 15 scalars, placed at the map centre. [a]
- Field token: Fourier position + radius, band width, level. Zone token: Fourier position + radius, owner, is-spawn, progress, direction, clipped distances, clipped lock timers. Both memory-only. [d]
- Encoder: per kind Linear -> RMSNorm -> GELU -> shared Linear to 128 -> RMSNorm. No slot, index or count parameters. [a]

**Trunk** (width 128, 2 blocks, each: temporal sublayer then 2 spatial sublayers) [d for the order]
- Spatial: pre-norm attention of ship + global queries over themselves and the map memory, `scaled_dot_product_attention` with a key mask (padding, benched, absent, never-seen, permanently dead; respawning dead are queries only) [a]; rotary relative position on queries and keys with about 6 bands from 64 m to about 30 km [b by analogy]; no heading rotary; gated MLP [b].
- Temporal: Griffin recurrent block per token (norm, Linear, 4-tap causal conv, RG-LRU, gated branch) [b], decay initialised for retention 0.9 to 0.9995 per tick [d]; gated MLP.
- RMSNorm throughout, one final norm. [a]
- No custom kernels; framework default initialisation under a seed. [a]

**Heads** (Linear -> GELU -> Linear on the normed trunk output; last layer gain 0.01) [a]
- Policy: 42-way joint categorical per ship, computed on all ship rows, sampled for own living ships, one draw per ship, ships independent. [d for joint; a for independence]
- Value: one per-ship categorical critic, two-hot over symexp bins with a widened support. [b]
- Outcome: 4-way on the global token, used as the team-level value of the terminal reward. [d]
- Auxiliary, training only: next-state (change from the current state in the heading frame, uniform categorical layout; for ghost tokens, privileged true displacement from last seen) [d]; shot outcome [c-needed]; geometry [c-needed].
- Not in the baseline: correction, density, shot trajectory, enemy action, local presence, colour codes, arena code, heading rotary, absent-zone vector, per-level value heads.

**Training-distribution changes that the model changes exist to enable** (outside the model, but they are the point)
- Mixed sizes in one batch by padding and masks: team sizes drawn per environment (for example 2 to 6 per side, unequal allowed), field and zone counts drawn, arena scale drawn (`arena_scale`, `active_fields` exist and are off). Domain variation in the quantity one wants to extrapolate over is the standard requirement for extrapolating at all [d here; Xu et al. FETCHED 2009.11848 for why an unvaried input does not extrapolate].

**What this keeps from the existing design**: the entity-set formulation with no slot parameters, per-kind encoders, memory-only map tokens, the global token, rotary relative position, the random offset, the Griffin temporal block, RMSNorm, the joint 42-way action, the categorical critic and the outcome head. These are its good parts.

---

## 2. Size-generalisation risks, ranked

Each with the cheapest revealing test; "eval-only" means on an existing 5v5-trained model with no training.

1. **Unmasked non-existent keys stacked at the map centre** (dead ships, never-seen enemies, absent zones; 12.6, 12.29, 1.17, 3.4). Their number grows from a handful to dozens and they sit on the global token. Test (eval-only): at 20v20, run with a key mask over never-seen enemies, permanently dead ships and absent zones (the `absent_zones` ablation and `mask_benched` paths already exist) versus unmasked; also log, per head and layer, the attention mass that lands on such keys at 5v5, 20v20 and 50v50.
2. **Softmax dilution in dense attention** (12.5, 3.5; 26 keys in training, 116 at 50v50), including the map tokens' falling share in the shared softmax (3.2). Test (eval-only): at 20v20 and 50v50, restrict each query's ship keys to its 9 nearest ships and compare win rate with dense; log attention entropy and maximum weight per head against N. If the nearest-9 model plays clearly better, dilution is the cause.
3. **Unequal teams** (12.2, 12.3, 2.8): positional own/enemy split and a `benched` flag that is always 0 in training. Test (eval-only): 5v3 and 3v5 through benched slots with `mask_benched` on and off, against the 5v5 and 3v3 results.
4. **Arena size** (12.33-12.38, 1.28, 1.29, 1.26, 1.19): two radii in training, two encodings of the radius, and coarse rotary bands and position levels that barely turn in training. Test (eval-only, 5v5): arenas at 0.5x, 2x and 4x, each run twice: once with the true radius fed in and once with the radius inputs held at 7800 m. The difference between the two runs separates "the radius input is out of range" from "the geometry is out of range".
5. **Memory capacity for hidden enemies after the belief is removed** (my reading (A) above): fixed-width recurrent state must hold up to 50 hidden enemies, and the recurrence is initialised with sub-second memory (3.9). Test (on the rebuilt model; cheap probe, no policy change): a linear probe from trunk states to hidden enemies' true positions, error versus time since seen, at 5v5 and at 20v20; compare readings (A) and (B).
6. **Constants that scale with team size in the global inputs** (12.9): `symlog(score_target)` is 25 in training, 100 and 250 at scale. Test (eval-only): 20v20 deathmatch with the target input held at its training value versus the true value.
7. **Local presence** (12.8, 1.16): a sum whose range grows with density. Test (eval-only): clamp to the training maximum at 20v20; zero it at 5v5.
8. **Zone and field counts** (12.27-12.30): Z = 5 or 0 only, 10 fields always, absent fields read as zero-feature keys. Test (eval-only): frontline with 3 zones (two slots absent), and maps with 5 and 20 fields with `mask_absent_fields` on and off.
9. **Local density at scale**: if 50v50 arenas do not grow with fleet size, every ship has several times more neighbours within weapon range than it ever trained with, whatever the architecture. Test (eval-only): 20v20 on an arena scaled to keep ships per square km at the training value, versus the training arena.
10. **Uncoordinated independent sampling** (12.23): over-commitment of many ships to one target and more friendly fire as allies multiply. Test (eval-only): at 5v5, 20v20, 50v50 measure the distribution of attackers per target and the friendly-fire share of damage.
11. **Critic support, return scale, reward shares** (12.13-12.17): training-only, so no effect on zero-shot play; they matter the moment training includes other sizes. Test: histogram of per-ship returns from 20v20 rollouts against the +-19 support.
12. **Fused kernel tile** (12.7): removed with the kernel.

---

## 3. What the inventory does not tell me

Answers to these would change a verdict.

1. How do 20v20 and 50v50 arenas scale: fixed radius, or constant ship density? (Risks 4 and 9; whether nearest-K belongs in the baseline.)
2. Which runtime switches does the large-size evaluation already turn on (`mask_benched`, `mask_absent_fields`, absent-zone masking)? If masking is already on at evaluation but off in training, risk 1 is a train/test mismatch rather than an untested case.
3. What does the existing 5v5 model actually score at 20v20 and 50v50? I rank risks without any measured baseline.
4. Did the project measure colour code vs Fourier vs scalars as *inputs*, and with the belief on or off? A measurement with the belief off would move 1.8/1.19 from "replace" to "keep".
5. Bullet speed and effective range relative to ship speed and turn rate. If bullets can be dodged after launch, 1.6 (no bullet tokens) is wrong; if effective range is under a few hundred metres, the position ladder can be shorter still.
6. What position does a dead own ship's row carry, and is it a key near its death site? (Affects whether respawning dead ships should be keys.)
7. How objective rewards are split among a team's ships (inventory's own uncertainty). Decides whether one per-ship value is enough or the team-level head should also carry objectives.
8. Is team vision shared (a team view is the union of its ships' sight)? I assumed yes. If not, one forward per team is not a valid actor.
9. Whether ships respawn in all modes; determines the "permanently dead" mask rule.
10. The cost record's attribution inside the spatial block (`map_norm` 79.96 M FLOPs, sub-items summing above the block's 119.26 M) looks wrong; which build was traced is also unstated. My cost arguments rest on the op counts, not on that line.
11. Whether the per-level value heads were measured against a single value. If yes (category c), keep them.
12. Whether a human opponent's experience depends on 30 Hz agent decisions (6.6).
13. Whether row identity is stable across ticks for enemies as they leave and re-enter sight (ghost tokens and the per-token recurrence both require it).

---

## 4. Papers cited

Fetched this session (arXiv API by id; title confirmed, content from memory):
- [FETCHED] Su et al., "RoFormer: Enhanced Transformer with Rotary Position Embedding", arXiv:2104.09864.
- [FETCHED] Heo et al., "Rotary Position Embedding for Vision Transformer", arXiv:2403.13298.
- [FETCHED] Press, Smith, Lewis, "Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation", arXiv:2108.12409.
- [FETCHED] De et al., "Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models", arXiv:2402.19427.
- [FETCHED] Veličković et al., "Softmax is not Enough (for Sharp Size Generalisation)", arXiv:2410.01104.
- [FETCHED] Xu et al., "How Neural Networks Extrapolate: From Feedforward to Graph Neural Networks", arXiv:2009.11848.
- [FETCHED] Hafner et al., "Mastering Diverse Domains through World Models" (DreamerV3), arXiv:2301.04104.
- [FETCHED] Farebrother et al., "Stop Regressing: Training Value Functions via Classification for Scalable Deep RL", arXiv:2403.03950.
- [FETCHED] Imani & White, "Improving Regression Performance with Distributional Losses", arXiv:1806.04613.
- [FETCHED] Gorishniy et al., "On Embeddings for Numerical Features in Tabular Deep Learning", arXiv:2203.05556.
- [FETCHED] Tancik et al., "Fourier Features Let Networks Learn High Frequency Functions in Low Dimensional Domains", arXiv:2006.10739.
- [FETCHED] Lee et al., "Set Transformer", arXiv:1810.00825.
- [FETCHED] Baker et al., "Emergent Tool Use From Multi-Agent Autocurricula", arXiv:1909.07528.
- [FETCHED] Berner et al. (OpenAI), "Dota 2 with Large Scale Deep Reinforcement Learning", arXiv:1912.06680.
- [FETCHED] Yu et al., "The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games", arXiv:2103.01955.
- [FETCHED] Hu et al., "UPDeT: Universal Multi-agent Reinforcement Learning via Policy Decoupling with Transformers", arXiv:2101.08001.
- [FETCHED] Iqbal et al., "Randomized Entity-wise Factorization for Multi-Agent Reinforcement Learning", arXiv:2006.04222.
- [FETCHED] Andrychowicz et al., "What Matters In On-Policy Reinforcement Learning? A Large-Scale Empirical Study", arXiv:2006.05990.
- [FETCHED] Huang & Ontañón, "A Closer Look at Invalid Action Masking in Policy Gradient Algorithms", arXiv:2006.14171.
- [FETCHED] Jaderberg et al., "Reinforcement Learning with Unsupervised Auxiliary Tasks", arXiv:1611.05397.

From memory, not checked:
- [MEMORY] Vaswani et al., "Attention Is All You Need", arXiv:1706.03762.
- [MEMORY] Vinyals et al., "Grandmaster level in StarCraft II using multi-agent reinforcement learning", Nature 2019 (AlphaStar).
- [MEMORY] Zaheer et al., "Deep Sets", arXiv:1703.06114.
- [MEMORY] Espeholt et al., "IMPALA", arXiv:1802.01561.
- [MEMORY] Kapturowski et al., "Recurrent Experience Replay in Distributed Reinforcement Learning" (R2D2), ICLR 2019.
- [MEMORY] Zhang & Sennrich, "Root Mean Square Layer Normalization", arXiv:1910.07467.
- [MEMORY] Shazeer, "GLU Variants Improve Transformer", arXiv:2002.05202.
- [MEMORY] Shaw, Uszkoreit, Vaswani, "Self-Attention with Relative Position Representations", arXiv:1803.02155.
- [MEMORY] van Seijen et al., "Hybrid Reward Architecture for Reinforcement Learning", arXiv:1706.06208.
- [MEMORY] Hernandez-Leal, Kartal, Taylor, "Agent Modeling as Auxiliary Task for Deep Reinforcement Learning", arXiv:1907.09597.
- [MEMORY] Tang & Agrawal, "Discretizing Continuous Action Space for On-Policy Optimization", arXiv:1901.10500.
- [MEMORY] Huang et al., "The 37 Implementation Details of Proximal Policy Optimization", ICLR 2022 blog track.
- [MEMORY] Specific details recalled rather than checked: Griffin's decay initialisation range (a^c in [0.9, 0.999]) and its width-4 temporal convolution; OpenAI Five acting every fourth frame; DreamerV3's bin range.
