# Zero-shot generalisation across the number of entities/agents in MARL: focused literature review

Scope: evidence relevant to training a centralised entity-attention + per-entity-recurrent policy at about 4v4/5v5 and playing, with no fine-tuning, at 20v20 to 50v50 (4x to 10x), with unbalanced teams, different zone counts and different map sizes.

## How to read this document

Verification labels (one per citation):

- **[FULL]** I fetched the arXiv PDF, extracted the text, and read the sections that carry the claim (method, the relevant experiment, the relevant appendix). It does not mean cover to cover. Figures were not viewable; where a result exists only as a plotted curve I say so and give no number.
- **[ABSTRACT]** I fetched the paper and read only the abstract (sometimes plus the introduction or a few lines located by keyword). Claims are limited to what those lines say.
- **[MEMORY]** Cited from memory. Where noted "existence verified", I fetched the arXiv id and confirmed title and authors, but did not read the content for this review.

All arXiv ids below are ids I actually fetched (`https://arxiv.org/pdf/<id>`), except the one paper marked "not fetched".

Counts: 28 [FULL], 19 [ABSTRACT], 6 [MEMORY]. An index is at the end.

Two corrections to the starting points supplied:

- **REFIL** exists as described (Iqbal, Schroeder de Witt, Peng, Böhmer, Whiteson, Sha, ICML 2021, arXiv 2006.04222), but it is **not** an entity-dropout method and it does **not** test extrapolation in count. It randomly partitions entities into two groups and adds an auxiliary value-factorisation loss; training and testing use the same distribution of team sizes (3 to 8 per side).
- **UPDeT** (Hu, Zhu, Chang, Liang, ICLR 2021, arXiv 2101.08001) and **EPC** (Long, Zhou, Gupta, Fang, Wu, Wang, ICLR 2020, arXiv 2003.10423) exist as described. Both are primarily about transfer **with fine-tuning**; their zero-shot evidence is a few plotted points (UPDeT, 3 to 7 marines) and one figure at 2x the largest training size (EPC).

---

## 1. Summary: the most decision-relevant findings

1. No paper I read demonstrates the target claim (centralised joint attention over all entities, competitive two-team self-play, 4x to 10x zero-shot in count); every large zero-shot extrapolation (5x to 250x) I found is for decentralised, homogeneous, cooperative agents that each see a bounded local neighbourhood. [FULL: 2011.08055, 2109.07735, 1810.09202, 1907.03822, 1807.06613]
2. The one study that isolates the mechanism found that a self-attention policy trained on 1-4 agents x 1-4 targets degrades when given the full observation at larger sizes, and that masking each agent's input to its k nearest targets (k=4) at evaluation restores performance up to 1000 x 1000, with the map scaled to keep density constant. [FULL: Hsu et al., 2011.08055]
3. Softmax attention provably disperses as the number of keys grows; a single attention head trained to find the maximum over at most 16 items falls from 98.6% at 16 items to 94.3% at 64 (4x), 81.3% at 256 (16x) and 12.4% at 16,384, and inference-time temperature adaptation recovers at most about four points. [FULL: Veličković et al., 2410.01104]
4. Without bounded neighbourhoods, global-context entity models show only small zero-shot extrapolation: 5-agent policy success in formation control goes 100% at 5 agents to 91/21/1% at 6/7/8 agents; a 3v3 Spread policy goes 0.98 to 0.75 landmark coverage at 6v6; hide-and-seek reports only "slightly larger" counts (1-3 to 4-6 hiders) with gradual decline. [FULL: 1906.01202, 2301.05334, 1909.07528]
5. Training across a range of counts flattens the size curve (curriculum over 3 to 6 agents gives 0.87-0.88 coverage at every size, against 0.75 to 0.98 for single-size training), but training directly on a large range can itself fail: a policy trained on 1-20 x 1-20 did worse than a 1-4 x 1-4 policy plus masking. [FULL: 2301.05334, 2011.08055]
6. For readouts, sum pooling is the consistent loser on size extrapolation and max is the best single aggregator; softmax attention behaves like mean pooling at initialisation. [FULL: Xu et al. 2009.11848; Corso et al. 2004.05718; Richter and Wattenhofer 2005.09561]
7. Pre-training on several team sizes does not by itself give zero-shot wins at larger sizes with a centralised transformer: the Multi-Agent Transformer pre-trained on eight SMAC maps (including 25m) scores 0% zero-shot on 27m_vs_30m and 5m_vs_6m, and 100% on 8m. [FULL: Wen et al., 2205.14953]
8. Random masking of units during training (mask ratio drawn from [0,1]) is the strongest reported training-side trick for unseen maps in SMAC (77.8% average zero-shot win rate on 60 unseen maps from 11 training maps, against 43.7% for the baseline), but this is offline imitation and the unseen maps are not shown to exceed the training entity counts. [FULL: Liu et al., 2310.11846]
9. Return scale depends on team size unless the reward is designed otherwise; the large precedents made rewards per-agent and averaged (OpenAI Five subtracts the average of enemy rewards, blends own reward with the team mean, and normalises by a running standard deviation), and MAPPO reports value-target normalisation "never hurts". [FULL: 1912.06680, 2103.01955, 1810.02912]
10. The large-scale precedents did not solve variable counts by extrapolation: OpenAI Five pads to 189 unit slots and keeps the units nearest to allied heroes when over; AlphaStar caps at 512 entities and truncates; TiZero uses fixed 11-player slots with player-ID embeddings; SMACv2 fixes team size per scenario. [FULL: 1912.06680, 2308.03526, 2302.07515, 2212.07489]
11. There is almost no literature on per-entity recurrent state with entities appearing and disappearing; the nearest evidence is that per-entity and global hidden state reach the same final performance in UPDeT, that one GRU for all targets under-performs per-target belief filters, and that memory-less beat LSTM in AlphaStar behaviour cloning (84% against 70%). [FULL: 2101.08001, 2011.08055, 2308.03526]
12. Size-generalisation theory for GNNs says extrapolation fails when the local structure seen by a node changes with size, and can work when it does not. [ABSTRACT: Yehudai et al., 2010.08853]

---

## 2. Findings by question

Format per finding: claim; paper; what was tested; how our setting differs.

### Q1. Architectural choices

#### 1.1 Entity/attention encoders against fixed-slot concatenation

**Claim: permutation-invariant entity encoders are a precondition for running at a different count at all, and outperform concatenation even at the training size.**

- Batra et al., "Decentralized Control of Quadrotor Swarms with End-to-end Deep RL" [FULL, 2109.07735]. Quadrotor swarm, N=8, K=6 visible neighbours. A plain MLP that concatenates neighbour observations "fails"; deep-sets (mean) and attention encoders work, attention best on tasks that need collision avoidance.
  - Differs: decentralised, per-drone local observation, cooperative, no opponent, no fog.
- Baker et al., "Emergent Tool Use from Multi-Agent Autocurricula" (hide-and-seek) [FULL, 1909.07528]. Entity embeddings, masked residual self-attention, masked average pooling, then LSTM. Appendix ablation: self-attention is more sample-efficient than an embed-then-pool architecture of similar parameter count.
  - Differs: closest large precedent to ours in structure (entity attention, then recurrence, PPO self-play, visibility masking), but the LSTM sits after pooling (one state per agent), and team sizes are 1-3.
- Zambaldi et al., "Relational Deep RL" [FULL, 1806.01830]. Box-World: relational (attention) agent solves more than 88% of levels with longer solution chains than trained on (6, 8, 10 boxes); the non-relational baseline collapses to 5%, 0%, 0%. StarCraft II mini-game trained with 2 marines, tested with 5: "high variability", effect diminishes with larger models.
  - Differs: single agent; extrapolation is in chain length, and the 2 to 5 marine result (2.5x) is explicitly inconclusive.
- Hao et al., "Breaking the Curse of Dimensionality in Multiagent State Space" (API/HPN) [FULL for appendix A.8 and abstract, 2203.05285]. Permutation-invariant input and permutation-equivariant output modules; size transfer (5m to 12m, 5m_vs_6m to 8m_vs_10m) is shown only as a better initialisation for further training, not zero-shot.
- Counter-precedent: TiZero [FULL, 2302.07515] (Google Research Football 11v11) uses six MLPs over a fixed-layout observation, an LSTM and a player-ID embedding; OpenAI Five [FULL, 1912.06680] uses fixed padded slots. Fixed-slot designs reach superhuman play at a fixed count and say nothing about other counts.

#### 1.2 Pooling: sum, mean, max, attention

**Claim: sum readouts extrapolate worst in count; max is the best single aggregator for extrapolation; mean is count-invariant but discards count; softmax attention acts like mean at initialisation.**

- Xu et al., "How Neural Networks Extrapolate" [FULL, 2009.11848]. Theory and experiments on GNNs. Corollary 1: GNNs with sum aggregation and sum readout do not extrapolate on max-degree; with max readout they do (given suitable training graphs). ReLU MLPs become linear along directions away from the training support. On n-body, an improved input representation improves extrapolation.
  - Differs: supervised algorithmic tasks, not RL. The principle (the aggregator must match the task's own non-linearity) is general.
- Corso et al., "Principal Neighbourhood Aggregation" [FULL, 2004.05718]. Trained on graphs of 15-25 nodes, tested on 20-50 (2x). Performance of all models "gradually worsens, with some of them having feature explosions"; among single aggregators "the max tends to perform best when extrapolating to larger graphs". They argue linear (sum) amplification per layer compounds exponentially with depth and propose logarithmic degree scalers, with the normaliser computed over the training set.
  - Differs: 2x only; supervised.
- Richter and Wattenhofer, "Normalized Attention Without Probability Cage" [FULL, 2005.09561]. Synthetic sequence tasks. At initialisation softmax-attention outputs scale like mean pooling with sequence length. Models trained at length 128 and validated at length 64: "the sum architecture does not generalize well"; their normalised attention (softmax replaced by normalisation) "seems to be the most robust"; max pooling is a strong baseline that is rarely reported.
  - Differs: tests half-length, not longer; synthetic.
- Iqbal et al., REFIL [FULL, 2006.04222]. In multi-task SMAC (3-8 units per side, same distribution at train and test), attention beats mean pooling, and max pooling was "marginally worse than mean pooling".
  - Differs: no extrapolation; value-based, decentralised.
- Hüttenrauch et al., "Deep RL for Swarm Systems" [FULL, 1807.06613]. Mean embedding of neighbour features (TRPO, decentralised). Rendezvous: trained with 20 agents, evaluated with 100 (5x), the neural mean-embedding policy remains the best performer (curves only). Pursuit: trained with 10, evaluated with 5, 20, 50; more agents capture faster, but with 5 agents "all methods struggle" because the learned encirclement leaves gaps.
  - Differs: decentralised homogeneous swarm; note the failure was on *fewer* agents and was a strategy failure, not a representation failure.
- Wang et al., DyMA-CL [FULL, 1909.02790]. Curriculum 5 to 10 to 15 Immortals in StarCraft II with a GNN and model reload **with fine-tuning**: SUM aggregation gave the highest win rate, above MEAN and MAX.
  - Differs: this is transfer with continued training, where sum's scale shift can be re-learned; it does not contradict the zero-shot results above.
- Liu et al., PIC [FULL, 1911.00025]. Permutation-invariant critic (graph convolution with mean or max pooling) trained separately at N = 15, 30, 100, 200 in particle environments. It shows such critics train at large N; it does not test transfer across N.
- OpenAI Five [FULL, 1912.06680] max-pools unit embeddings; hide-and-seek [FULL, 1909.07528] uses masked average pooling. Neither compares pooling types across counts.
- Wagstaff et al., "On the Limitations of Representing Functions on Sets" [ABSTRACT, 1901.09006]. For sum-decomposition with continuous maps, the latent dimension must be at least the maximum set size for universal representation.

#### 1.3 Attention dilution and proposed fixes

**Claim: softmax attention weights necessarily flatten as keys are added, so behaviour that relies on sharp selection degrades with count; fixes are locality masks, logit scaling with log n, and normalisation.**

- Veličković et al., "Softmax is not Enough (for Sharp Size Generalisation)" [FULL, 2410.01104]. Lemma 2.1 and Theorem 2.2: with bounded logits, every softmax coefficient decays towards zero as n grows. Max-retrieval with one attention head, trained on at most 16 items: accuracy 98.6% (16), 97.1% (32), 94.3% (64), 89.7% (128), 81.3% (256), 70.1% (512), 53.8% (1,024), 12.4% (16,384). Entropy-adaptive temperature at inference: 94.5% (64), 82.1% (256), 57.7% (1,024).
  - Differs: a single head on a pure selection task. In our range (4x to 10x keys) the loss on this task is a few points to roughly ten points per head; how that compounds through several blocks and a control loop is not measured anywhere.
- Hsu et al. [FULL, 2011.08055]. See 2.2 below: the full observation at larger counts degrades a self-attention Q-network; a k-nearest mask at evaluation fixes it. This is the direct RL evidence for locality masking.
- Batra et al. [FULL, 2109.07735] and Jiang et al., DGN [FULL, 1810.09202] both fix the neighbourhood (K=6 neighbours; 3 nearest neighbours) at training and test time.
- Nakanishi, "Scalable-Softmax Is Superior for Attention" [ABSTRACT, 2501.19399]. Multiplies logits by s·log n with learned s; reports better loss at contexts beyond the training length and better key-information retrieval in language models.
- Anson, Wang, Aitchison, "Scale-invariant Attention" [ABSTRACT, 2505.17083]. Argues log-n scaling restores sharpness but "forfeits the ability to attend to the local context"; proposes a position-dependent logit transform; reports gains when zero-shot generalising to longer contexts.
- Richter and Wattenhofer [FULL, 2005.09561]: replacing softmax with normalisation (above).
  - Differs (all three language-model papers): 1-D sequences with a distance notion given by token position; none is tested on entity sets or in RL. No paper I found tests distance-biased attention logits for count extrapolation in MARL.

#### 1.4 Relative against absolute position encodings; identity and slot embeddings

**Claim: the papers that extrapolate in count use agent-relative geometry and no agent identity.**

- Hsu et al. [FULL, 2011.08055]: targets are encoded in polar coordinates relative to each pursuer; agents are homogeneous with no index.
- Batra et al. [FULL, 2109.07735]: relative positions and relative velocities of neighbours.
- Gallici et al., TransfQMix [FULL, 2301.05334]: each agent observes its own absolute position and velocity plus relative positions of all other entities; an IS_SELF flag replaces agent identity.
- Xu et al. [FULL, 2009.11848]: inputs outside the training support are extrapolated linearly by ReLU MLPs; a better input representation improved n-body extrapolation.
- Shi et al., MTR++ [ABSTRACT, 2306.17770]: motion prediction; replaces a focal-agent-centred global frame with symmetric "query-centric" encoding in each token's local frame, with local attention in the encoder.
- Kazemnejad et al., "The Impact of Positional Encoding on Length Generalization in Transformers" [ABSTRACT, 2305.19466]: learned absolute, ALiBi and rotary encodings are "not well suited" to length generalisation on their tasks; no positional encoding did best.
- Identity embeddings in practice: MAPPO feeds agent id and reports that keeping the agent id in the "death mask" state matters [FULL, 2103.01955]; TiZero appends a player-ID embedding [FULL, 2302.07515]. Both are fixed-count systems. I found no paper that tests learned identity or slot embeddings at indices beyond the training count.
  - Differs: our positions are in metres on maps of varying size, so absolute coordinates at a larger map are outside the training support (Xu et al.'s result applies to any MLP that embeds them).

#### 1.5 Normalisation layers

- Zhang et al., "Set Norm and Equivariant Skip Connections" [FULL for the definitions, 2206.11925]. Argues layer norm can hurt in set models; proposes set norm, which standardises over all elements and features of a set. Experiments use fixed sample sizes (1000, or 10); no test across set sizes.
- I found no paper that measures the effect of normalisation-layer choice on count extrapolation in MARL. Anything said about it is inference (Section 3).

#### 1.6 GNN size extrapolation

- Yehudai et al., "From Local Structures to Size Generalization in GNNs" [ABSTRACT plus introduction, 2010.08853]. Proves that when local structure (their "d-patterns") differs between small and large graphs there are global minima that fit small graphs and fail on large ones; empirically GNNs tend to converge to such solutions; a self-supervised task on large unlabelled graphs improves accuracy by 4% on average.
- Bevilacqua, Zhou, Ribeiro [ABSTRACT, 2103.05045]: size-invariant graph representations via a causal model for out-of-distribution graph classification.
- Khan et al., "Graph Policy Gradients" [FULL, 1907.03822]: graph filters trained on 3 robots (or 5) for formation flying, transferred without updates to swarms of 101 and 51. Reported qualitatively (figures). Hsu et al. report that this baseline "seemingly scales" but sits in a poor local optimum on their task.

### Q2. Training-side methods

**2.1 Fixed small training range plus locality at test time.** Hsu et al., "Scalable RL Policies for Multi-Agent Control" [FULL, 2011.08055].
- Tested: multi-agent target tracking; decentralised soft double Q-learning with shared self-attention Q-network; target beliefs from Kalman filters. Policies trained on tasks sampled from n in [1,4] agents and m in [1,4] targets ("4a4t"), also 8a8t and 20a20t. Evaluated without retraining at 8, 20, 100 and 1000 agents/targets, **with the map scaled to keep the density of the 4a4t task**.
- Result (text; exact values are in figures): with the full observation, performance degrades "as the task space expands past their training space". With a k-nearest mask at evaluation, 4a4t and 8a8t policies with k=4 "outperform all other policies"; abstract claims "minimal degradation" up to 1000 x 1000. The 20a20t policy "does not work well even on tasks inside its training space"; the authors conclude "simply using a heuristic on a well-trained 4a4t policy provides better performance than directly training large MARL systems".
- Differs: decentralised and cooperative; the mask is applied to each agent's own input, which is trivial there and in our model would be a per-query attention mask; targets follow a fixed policy (no adversary adapting); extrapolation 250x but on a task that decomposes into local sub-tasks.

**2.2 Fixed neighbourhood from the start.** Batra et al. [FULL, 2109.07735].
- Tested: trained N=8, K=6 neighbours (IPPO-style, decentralised). Zero-shot at N = 16, 32, 48, 64, 128 with K=6.
- Result (Table 1; collisions per minute per drone / distance to target in metres): N=8: 0.02 / 0.42. Zero-shot: 16: 0.09 / 0.57; 32: 0.90 / 0.86; 48: 2.43 / 1.11; 64: 5.36 / 1.70; 128: 8.63 / 4.32. After an extra 2x10^8 steps at the target N (20% of the original training): 16: 0.08 / 0.63; 32: 0.16 / 0.81; 48: 0.23 / 0.92; 64: 0.29 / 1.12; 128: 1.37 / 3.05. Authors: without additional training collisions rise "because the state distribution changes significantly".
- Differs: even with a fixed K, 4x was a 45-fold rise in collisions zero-shot. This is the most relevant caution: a bounded neighbourhood fixes the input statistics, not the state distribution (denser formations).

**2.3 Neighbour-limited graph attention.** Jiang et al., DGN [FULL, 1810.09202].
- Tested: routing; trained N=20 agents, 3 neighbours each; applied without retraining at N=40, 60 and up to 140.
- Result: DGN beats the shortest-path baseline up to N=140 (7x); beyond that the baseline wins.
- Differs: decentralised Q-learning; the comparison is against a heuristic, not against a policy trained at the larger size.

**2.4 Training over a distribution of sizes; curricula.**
- TransfQMix [FULL, 2301.05334]. Spread (cooperative navigation). Zero-shot landmark coverage (rows: training size; columns: test 3v3 / 4v4 / 5v5 / 6v6; read from a text-extracted table, column order inferred): trained 3v3: 0.98 / 0.88 / 0.80 / 0.75; trained 4v4: 0.96 / 0.93 / 0.90 / 0.86; trained 5v5: 0.88 / 0.85 / 0.82 / 0.82; trained 6v6: 0.91 / 0.88 / 0.85 / 0.84; curriculum 3,4,5,6 (500k steps each): 0.88 / 0.88 / 0.87 / 0.87. Authors: "worse performances for larger teams, except for the agents trained with CL".
  - Differs: 2x at most; QMIX-style; in Spread the optimal policy is the same at every n.
- Agarwal et al., "Learning Transferable Cooperative Behavior in Multi-Agent Teams" [FULL, 1906.01202]. Shared agent-entity graph, decentralised, curriculum 3 to 5 to 7 to 10 agents with fine-tuning. Zero-shot from M=5 (Table 4, success %, M-3 ... M+3): coverage (unrestricted comm.) 89, 95, 93, **98**, 83, 65, 41; coverage (restricted comm.) 84, 92, 99, **99**, 99, 95, 74; formation (unrestricted) 1, 9, 98, **100**, 91, 21, 1; formation (restricted) 1, 68, 99, **99**, 34, 30, 8; line (unrestricted) 0, 0, 57, **99**, 81, 45, 16; line (restricted) 0, 16, 76, **99**, 45, 16, 8.
  - Differs: 1.6x at most, and it already fails on tasks whose solution geometry depends on N (formation, line). Coverage, which decomposes, holds up best, and restricted (local) communication extrapolates better than unrestricted on coverage.
- Hide-and-seek [FULL, 1909.07528]. Training randomises team size (1-3 per team), boxes (3-9), ramps, walls; this randomisation "helps emergence". Zero-shot to 4-6 hiders, 3-4 ramps, more boxes: hider reward declines "very gradual[ly]" (figure only). The authors' own summary is "slightly larger number".
  - Differs: 2x; decentralised policies with a centralised omniscient critic.
- Long et al., EPC [FULL, 2003.10423]. MADDPG with attention in policy and critic; population doubled per stage with fine-tuning and evolutionary selection. Scales: Grassland 3-2 to 6-4 to 12-8 to 24-16; Adversarial Battle 4-4 to 8-8 to 16-16; Food Collection from 3 upward by doubling. Zero-shot: the largest-scale policies applied to a doubled population by self-cloning; EPC beats vanilla curriculum and attention-MADDPG (figure, normalised scores only).
  - Differs: the method's point is that agents good at small scale "are not necessarily the best candidates" at large scale, and that fine-tuning is needed; zero-shot evidence is 2x only.
- UPDeT [FULL, 2101.08001]. SMAC marines 3m, 5m, 7m; transformer with policy decoupling (each enemy token outputs its own attack logit). Fine-tuning converges "at least 10 times" faster than a GRU baseline; zero-shot performance is shown as points in Figure 5 and described as "strong generalization ability without finetuning" (no numbers in text). TransfQMix later reports UPDeT failing on 27m_vs_30m when trained there [FULL, 2301.05334].
- DyMA-CL [FULL, 1909.02790]: curriculum over agent number with fine-tuning (5 to 10 to 15 Immortals); not zero-shot.
- Neural MMO [ABSTRACT plus figure caption, 1903.00784]: agents trained at population caps 16, 32, 64, 128 and evaluated at a common population size; "agents trained in larger populations always perform better".
- Multi-Agent Transformer [FULL, 2205.14953]. Centralised encoder-decoder transformer over agents, PPO-style. Pre-trained on 3m, 8m_vs_9m, 10m_vs_11m, 25m, 3s_vs_3z, 2s3z, 3s5z, MMM (10M samples). Zero-shot median win rate (Table 2): 5m_vs_6m 0.0; 8m 100; 27m_vs_30m 0.0; 2s_vs_1sc 0.0; 1c3s5z 3.1; MMM2 0.0. With 10% of the data: 27m_vs_30m 71.2, MMM2 62.5. MAPPO pre-trained the same way: 27m_vs_30m 9.4 zero-shot.
  - Differs: this is the closest architecture family to ours (centralised transformer over agents, PPO), and its zero-shot transfer to a slightly larger and harder map than any seen was nil. Hard asymmetric maps demand precise micro, so this is a strict test, but so is self-play combat.

**2.5 Random masking / sub-group training.**
- REFIL [FULL, 2006.04222]: random two-way partition of entities implemented as attention masks; auxiliary loss that the value should factor across the two groups. Multi-task SMAC, 3-8 units per side (39 to 66 tasks), train and test on the same task distribution; REFIL beats QMIX with attention. Random partitions worked better than partitions from domain knowledge. No held-out sizes.
- Liu et al., COPA [ABSTRACT plus located lines, 2105.08692]: reports "REFIL does not always" help in their dynamic-team tasks, and tests on "unseen environments with more agents" (training 2-4 agents; exact test sizes not read).
- MaskMA [FULL, 2310.11846]: transformer trained by imitation on expert trajectories from 11 SMAC maps with units randomly masked in attention; mask ratio drawn uniformly from [0,1] per step was the best setting; action space split into intrinsic and per-target interactive actions. Zero-shot on 60 unseen maps: 79.71% (centralised execution) and 77.75% (decentralised), against 43.72% for MADT; the baseline drops on maps with more than 20 entities while MaskMA does not.
  - Differs: offline imitation, not RL; the training set includes large maps (for example corridor, MMM2), so this is generalisation across compositions more than extrapolation in count.

**2.6 Mean-field.** Yang et al., "Mean Field MARL" [ABSTRACT plus experiment lines, 1802.05438]: each agent's Q depends on its own action and the mean action of its neighbours; battle game trained and evaluated at 64 v 64. It is a way to train at large N, not evidence of transfer from small N. EPC reports "little differences between the mean-field approach and the original MADDPG" in its environments [FULL, 2003.10423].

### Q3. Value function and credit

**Claim: with summed or team-total rewards the return scale and the policy-gradient variance grow with team size; published systems handle this with per-agent averaged rewards, target normalisation, and per-agent or attention critics. Mixing networks tied to a fixed agent count do not transfer.**

- OpenAI Five [FULL, 1912.06680]. Per-hero reward; "team spirit" τ blends own reward with the team **mean** (annealed 0.3 to 1.0); zero-sum enforced "by subtracting from each hero's reward the average of the enemies' rewards"; rewards normalised by a running standard deviation. One value output per hero replica.
  - Differs: fixed 5v5, so they never needed count invariance; but every aggregation they chose is a mean, which is count-invariant by construction.
- MAAC, Iqbal and Sha [FULL, 1810.02912]. Per-agent critics that attend over other agents. Trained separately at 4, 8, 12 agents: improvement over MADDPG+SAC of 17%, 98%, 208%. They state that "differing the number of agents changes the nature of rewards" and therefore normalise by the reward range; in the task where each pair earns the same reward regardless of N, the scores are directly comparable and MAAC "does not deteriorate as agents are added".
  - Differs: no zero-shot test; off-policy.
- MAPPO, Yu et al. [FULL, 2103.01955]. Value targets standardised with running mean and standard deviation (PopArt is named in the algorithm listing); "never hurts training and often improves"; critical in Spread where returns range from below -200 to 0. Also: an agent-specific global state for the critic beats a plain concatenation of observations, "particularly in maps with many agents".
- PopArt [MEMORY, existence verified, 1602.07714] and multi-task PopArt [ABSTRACT, 1809.04474]: adapt the contribution of each task so that tasks with different reward magnitudes have similar impact on learning.
- Kuba et al., "Settling the Variance of Multi-Agent Policy Gradients" [ABSTRACT plus located lines, 2108.08612]: the excess variance of the multi-agent policy-gradient estimator "can grow linearly with the number of agents"; they derive an optimal baseline.
- VDN [ABSTRACT, 1706.05296]: joint value as a sum of per-agent values, learned from the team reward. QMIX [ABSTRACT, 1803.11485]: monotonic mixing network with weights from hypernetworks on the global state. REFIL's appendix [FULL, 2006.04222] documents that standard QMIX requires fixed-size state and a fixed number of utilities, and has to be rebuilt with attention to handle variable counts; TransfQMix [FULL, 2301.05334] notes UPDeT is "only partially transferable, because the mixer network must be re-trained".
- Hide-and-seek [FULL, 1909.07528]: a centralised omniscient value function per agent was necessary; with a visibility-masked value function training "never progressed past stage 3".
- Qu, Wierman, Li, "Scalable RL for Multi-Agent Networked Systems" [ABSTRACT, 1912.02906]: when interactions are local, a localised policy using κ-hop truncated value functions is an O(ρ^(κ+1)) approximation of a stationary point, with complexity depending on the largest κ-hop neighbourhood and not on the number of agents.
- HL-Gauss, Farebrother et al. [ABSTRACT plus located lines, 2403.03950]: the categorical value is supported on a fixed interval [v_min, v_max] with m bins.
  - Differs (whole section): no paper studies a categorical value head whose target scale shifts with team size. That interaction is inference (Section 3).

### Q4. Per-entity recurrent memory with variable entity sets

**Claim: there is very little direct evidence. What exists suggests per-entity state is viable but not clearly better than a global state at small scale, and that hand-built per-entity belief trackers beat a single learned recurrent state when the number of tracked entities grows.**

- UPDeT [FULL, 2101.08001]. Compares (a) a global hidden state fed as an extra transformer token and (b) an individual hidden state per entity token (each input carries its own state, each output emits a new one). Removing the hidden state hurts significantly; global converges faster; "the final performances are almost the same". The authors note the individual scheme adds a learning burden.
  - Differs: the per-entity state there is produced by the transformer itself, not by a dedicated recurrent cell; decentralised; small maps.
- Hsu et al. [FULL, 2011.08055]. Policies using Kalman-filter beliefs per target plus attention beat policies using one GRU: "the single GRU must maintain statistics for all the targets". They note one GRU per target is possible but did not pursue it.
- REFIL [FULL, 2006.04222] and hide-and-seek [FULL, 1909.07528] place recurrence **after** attention/pooling, one state per controlled agent. Hide-and-seek masks unseen entities out of attention and pooling.
- OpenAI Five [FULL, 1912.06680]: "When enemy heroes are in fog of war, we reuse the observation from the last time step when the unit was visible"; other units in fog are simply not observed. Memory is a single LSTM per hero after pooling.
- AlphaStar Unplugged, Mathieu et al. [FULL, 2308.03526]: for behaviour cloning, win rate against the very-hard bot was 70% with LSTM, 84% with no memory, 85-89% with a transformer memory; they found "no memory performs better than LSTM for behavior cloning".
  - Differs: full StarCraft II with a single global memory; relevant to our behaviour-cloning stage.
- AlignNet, Creswell et al. [ABSTRACT plus located lines, 2007.08973]: keeping per-object slots consistent over time requires solving alignment; the model handles "appearance of new entities and disappearance of entities" and reappearance after occlusion.
  - Differs: vision; in our simulator entity identity is given, so alignment is free for own ships and visible entities.
- Gap: I found no MARL paper that gives every entity token its own recurrent state, handles spawn/despawn, and tests at a different count.

### Q5. Large-scale precedents: what they did about variable counts

| System | Counts | Mechanism for variable counts | Tested at unseen counts? |
|---|---|---|---|
| OpenAI Five [FULL, 1912.06680] | 5v5 heroes; 189 unit slots | Per-unit embedding, **max-pool**; pad with zeros; if more units than slots, keep those **closest to allied heroes**; target selection by attention over unit embeddings; cross-hero max-pool over 25% of the vector | No |
| Hide-and-seek [FULL, 1909.07528] | 1-3 per team, 3-9 boxes, randomised | Entity embeddings, masked residual self-attention, **masked average pooling**, LSTM; omniscient critic | Yes, up to about 2x (figure only) |
| AlphaStar [MEMORY, not fetched: Vinyals et al., Nature 2019] | up to 512 entities | Entity transformer, pooled into a single LSTM core, pointer network over entities for targeting | Not that I recall |
| AlphaStar Unplugged [FULL, 2308.03526] | up to 512 entities | Units transformer; "in the rare event that more than 512 units/effects exist at once, the additional observations will be truncated"; pointer networks; mostly memory-less | No |
| SMAC / SMACv2 [FULL for SMACv2, 2212.07489; SMAC 1902.04043 existence verified] | SMACv2: 5v5, 10v10, 10v11, 20v20, 20v23, fixed per scenario | Randomised unit types and start positions; team size fixed per scenario | No; it tests generalisation over composition and positions |
| SMAC variable-count work | 3-8 (REFIL), 3/5/7 (UPDeT), 11 to 60 maps (MaskMA), 8 maps to 6 (MAT) | Attention masks; per-target action tokens; random masking | See Q2: small or nil zero-shot gains |
| Neural MMO [ABSTRACT, 1903.00784] | up to 128 per server | Independent agents with local observation | Cross-population evaluation at a common size |
| Google Research Football, TiZero [FULL, 2302.07515] | 11v11 fixed | Fixed-layout MLP encoders, LSTM, player-ID embedding, curriculum over scenario difficulty | No |
| Drone/swarm [FULL: 2109.07735, 2011.08055, 1807.06613, 1907.03822] | 3-20 trained, 50-1000 tested | k nearest neighbours or local graph; relative coordinates; mean/attention pooling over the neighbourhood | Yes; see Q2 |

Common pattern: the game-playing systems fix the count and bound the input with a slot cap plus a nearest-first rule; the swarm systems vary the count and bound the input with a neighbourhood.

### Q6. Limits, and conditions under which extrapolation works

- **Works when each unit's input distribution is unchanged by N.** Hsu (k-nearest mask, constant density), DGN (3 neighbours), GPG (local graph filters), Hüttenrauch (mean embedding of neighbours). Theory: Yehudai et al. [ABSTRACT] on local-structure discrepancy; Qu et al. [ABSTRACT] on κ-hop truncation under local interaction.
- **Fails or degrades when the local picture changes with N even if the input size is bounded.** Batra et al.: K fixed at 6, yet collisions rise 45-fold at 4x and over 400-fold at 16x zero-shot, repaired by fine-tuning.
- **Fails when the solution's global geometry depends on N.** Agarwal et al.: formation and line tasks collapse at +2 or +3 agents; Hüttenrauch: encirclement learned with 10 pursuers leaves gaps with 5.
- **Global attention degrades with key count.** Veličković et al. (theory plus numbers); Hsu et al. (full observation degrades).
- **Training on the large range directly can be worse.** Hsu et al. (20a20t). EPC's premise is the same: direct training at large N is hard, and small-N winners are not the best large-N candidates.
- **Transformers on algorithmic length generalisation.** Anil et al. [ABSTRACT, 2207.04901]: fine-tuned transformers show "significant generalization deficiencies independent of model scale". Zhou et al. [ABSTRACT, 2310.16028]: transformers tend to length-generalise when the task has a short RASP program that works at all lengths. Zhou et al., "Transformers Can Achieve Length Generalization But Not Robustly" [MEMORY, existence verified, 2402.09371]: as I recall, success depends on a specific positional-encoding and data-format combination and varies strongly across seeds.

---

## 3. Implications for this architecture

Everything in this section is my inference from the evidence above. Tags:
**[A]** supported by direct evidence at comparable extrapolation (4x or more, RL);
**[B]** supported at smaller extrapolation or outside RL;
**[C]** theoretical or inferred, should be tested.

1. **Make every attention query see a bounded, distance-defined neighbourhood (k nearest per entity type, or a metric radius with a cap), in training and at test.** [A] Hsu (4 to 1000 with k-nearest mask; full observation degrades), Batra, DGN. Caveat: all are decentralised; in a centralised forward pass the equivalent is a per-query attention mask, and information can still propagate across blocks, which none of these papers tests. [C] for that part.
2. **Do not expect a bounded neighbourhood alone to carry 4x to 10x.** [A] Batra: fixed K, yet strong zero-shot degradation from changed state distribution. The quantity to hold in range is what each ship sees locally: number of friends, enemies and bullets within its neighbourhood.
3. **Control local density, not just count.** Hsu scaled the map to keep density constant when scaling N; that is the condition under which their result holds. For us: 50v50 on a map sized so that per-ship neighbourhood occupancy matches training is the supported case; 50v50 on the 5v5 map is not supported by any evidence I found. [A] for the constant-density case, [C] for anything denser. Training should cover the range of local densities expected at test (for example, small maps with 5v5 to produce 50v50-like crowding). [C]
4. **Train on a distribution of team sizes, including unbalanced teams, not a single size.** [B] TransfQMix (flat 0.87-0.88 across 3 to 6 after curriculum, against 0.98 to 0.75), hide-and-seek (randomisation helps), REFIL (multi-task 3-8). But keep the range modest and well-learned before widening: Hsu's 1-20 range failed to train well, and EPC needed staged doubling. [B]
5. **Random entity masking during training is cheap and worth an ablation.** [B] MaskMA (random mask ratio in [0,1] best; imitation only), REFIL (random sub-groups beat oracle groups; in-distribution only). Not shown for extrapolation in count. With fog of war we already have a natural visibility mask; extra random masking of visible entities is the untested part. [C]
6. **No learned per-slot or per-index identity embeddings; entity type and team embeddings only.** [C] No paper tests unseen indices; the systems that extrapolate all omit identity (Hsu, Batra, TransfQMix's IS_SELF flag), and the fixed-count systems that use IDs (MAPPO, TiZero) never change count. If the model needs to break symmetry between ships, derive it from state (relative geometry), not from an index.
7. **Geometry as relative features in physical units; avoid raw absolute coordinates as the main positional signal.** [B] Hsu, Batra, TransfQMix use relative coordinates; Xu et al. show MLP embeddings extrapolate linearly outside the training support and that input representation drives extrapolation. On a larger map, absolute metres are out of range. Zone and boundary information should likewise be expressed relative to each ship (bearing and distance to nearest zones and to walls), with distances passed through a saturating transform so far-away values stay inside the training range. [C] for the saturating transform.
8. **If attention logits get a distance bias, treat it as a soft version of rule 1 and still cap k.** [C] No MARL evidence. The language-model results (SSMax, scale-invariant attention) show log-n logit scaling helps retrieval at longer lengths but can remove local focus; they are 1-D sequence results.
9. **Pooling for any team-level readout: prefer max, mean, or attention pooling over a bounded set; avoid un-normalised sum.** [B] Xu, PNA, NAP (sum fails; max best for extrapolation); OpenAI Five uses max, hide-and-seek masked mean. If count information is needed (it is: 5 against 8 matters), supply it explicitly as a bounded feature, for example log(1+n) per type or the ratio of forces, in the spirit of PNA's logarithmic scalers. Note PNA's normaliser is computed on the training set, so the feature still goes out of range at 10x; clip or saturate it. [C]
10. **Per-ship heads read only that ship's token (after local attention), not a global pooled vector.** [B] UPDeT's policy decoupling and MaskMA's per-target action logits are the pattern that transfers across counts; UPDeT's "aggregation transformer" (pool then project) under-performed. This makes the action path count-independent by construction, leaving only the attention inputs as the count-sensitive part.
11. **Value targets must not scale with team size.** [B] OpenAI Five (means everywhere, running-std reward normalisation), MAAC (reward nature changes with N), MAPPO (target normalisation). With a categorical head on a fixed support (HL-Gauss), a team-summed shaped reward at 20v20 would push targets off the support. Options consistent with the evidence: terminal win/loss in [-1, 1]; shaped terms averaged per ship and zero-summed against the enemy **average**; or per-ship value outputs with mean aggregation. [C] for the specific interaction with categorical bins (no paper tests it).
12. **The value head is not needed to act at test time, so its extrapolation matters only through training.** [C] Under PPO the zero-shot policy uses the actor only. The value head's behaviour at unseen sizes matters if (a) training covers several sizes (rule 4), where targets must be comparable across sizes, (b) value-like auxiliary heads share trunk features, or (c) the value is used for league matchmaking or evaluation at large sizes. A centralised critic may be given privileged, un-fogged state (hide-and-seek found this necessary), and it does not need to extrapolate.
13. **Per-entity recurrent state is compatible with variable counts but is unproven.** [C] The recurrent cell is shared across tokens and adds no count dependence itself. Evidence it helps is thin: UPDeT found per-entity and global state equal in final performance; Hsu found per-target filters better than one GRU as targets multiply, which supports per-entity memory in principle. Design consequences that follow from the precedents: initialise state at spawn; for enemies that enter fog, either keep the token alive with a "last seen" age feature (OpenAI Five reused the last observation for fogged heroes) or drop it, and test which; bullets probably need no memory.
14. **Check that memory helps the behaviour-cloning stage.** [B] AlphaStar Unplugged: memory-less beat LSTM in behaviour cloning (84% against 70%). If the scripted teacher is reactive, a recurrent block may learn to be ignored or may hurt; an ablation with the recurrent state zeroed is cheap.
15. **Normalisation: use per-token normalisation only; nothing that computes statistics across the entity axis.** [C] No direct evidence. Per-token layer norm is count-independent by construction; set-level statistics (Set Norm's per-set standardisation) and running observation statistics that pool across entities depend on composition and would shift with count.
16. **Plan for a short fine-tune as the fallback and report it separately.** [A] Batra: 20% extra training at the target size cut collisions from 5.36 to 0.29 per minute at 64 drones; MAT: 0% zero-shot to 71% with 10% data on 27m_vs_30m; EPC and UPDeT are built on this. The honest headline, on current evidence, is "zero-shot within the trained local-density range, plus cheap adaptation outside it".
17. **Evaluate with a size curve against a size-specific reference.** [B] The papers that convinced me report performance at each test size (Agarwal Table 4, Batra Table 1, TransfQMix Table 1, Veličković Table 1). DGN's and Hsu's comparisons to heuristics are weaker. For self-play combat, the meaningful tests are: the 5v5-trained policy against a fixed scripted opponent at each size; against itself with unbalanced teams (does 8 beat 5 at the expected rate?); and, where affordable, against a policy trained at that size.

Risks specific to our setting that the literature does not cover at all: (i) an adaptive opponent at large scale (all positive results are cooperative or against fixed opponents); (ii) tactics whose geometry changes with numbers (focus fire, flanking, zone allocation among 50 ships), which is the regime where Agarwal's formation task failed at +2 agents; (iii) multi-block attention over a centralised entity set with locality masks; (iv) categorical value heads under size shift.

---

## 4. Papers the owner should download for us

Full text I could not obtain or could not read adequately:

1. Vinyals, O., Babuschkin, I., Czarnecki, W. M., et al. (2019). "Grandmaster level in StarCraft II using multi-agent reinforcement learning." Nature 575, 350-354. https://www.nature.com/articles/s41586-019-1724-z (including the Methods and supplementary architecture description; not on arXiv). Needed for: exact entity-encoder pooling, entity cap, how the LSTM core receives entities.
2. Bagnell, J. A., and Ng, A. Y. (2005). "On Local Rewards and Scaling Distributed Reinforcement Learning." NeurIPS 18. https://papers.nips.cc/paper/2005 (search by title). Needed for: sample-complexity scaling of global against local rewards with agent count. I could not verify it from this machine; see Unverified leads.

Fetched but only skimmed; worth a proper read by whoever acts on the relevant rule (all are freely available at the arXiv ids given):

3. Nakanishi, K. M. (2025). "Scalable-Softmax Is Superior for Attention." https://arxiv.org/abs/2501.19399
4. Anson, B., Wang, X., Aitchison, L. (2025). "Scale-invariant Attention." https://arxiv.org/abs/2505.17083
5. Yehudai, G., Fetaya, E., Meirom, E., Chechik, G., Maron, H. (2021). "From Local Structures to Size Generalization in Graph Neural Networks." https://arxiv.org/abs/2010.08853
6. Liu, B., Liu, Q., Stone, P., Garg, A., Zhu, Y., Anandkumar, A. (2021). "Coach-Player Multi-Agent Reinforcement Learning for Dynamic Team Composition." https://arxiv.org/abs/2105.08692 (its Table 1 reports unseen environments with more agents; I did not extract the numbers).

Figures I could not view, which hold the actual numbers: UPDeT Figure 5 (zero-shot points 3m/5m/7m); hide-and-seek Figure A.8; EPC Figure 10; Hsu et al. Figures 3 and 8; Hüttenrauch Figures 5 and 8.

---

## 5. Unverified leads

Not used to support any claim above. Titles and details are from memory and may be wrong.

- Bagnell and Ng (2005), "On Local Rewards and Scaling Distributed Reinforcement Learning": as I recall, worst-case sample complexity grows linearly with the number of agents under a global reward and logarithmically under local rewards.
- "Gigastep" (Lechner et al., NeurIPS 2023 Datasets and Benchmarks track, as I recall): a JAX multi-agent environment with 3-D dynamics, fog of war, heterogeneous teams and scenarios such as 5v5 and 20v20. An arXiv title search returned nothing. If it exists as recalled it is the closest public environment to this project.
- ODIS, "Discovering Generalizable Multi-agent Coordination Skills from Multi-task Offline Data" (Zhang et al., ICLR 2023, as I recall): offline skill discovery with zero-shot tests on unseen SMAC marine team sizes. arXiv title search returned nothing.
- "Decompose a Task into Generalizable Subtasks in Multi-Agent Reinforcement Learning" (DT2GS, NeurIPS 2023, as I recall). Not found by title search.
- Qin et al., "Multi-Agent Policy Transfer via Task Relationship Modeling" (arXiv 2203.04482; id and title verified by search, content not read).
- Liu et al., "Multi-Agent Game Abstraction via Graph Attention Neural Network" (G2ANet, arXiv 1911.10715; fetched, not read).
- Schroeder de Witt et al., "Is Independent Learning All You Need in the StarCraft Multi-Agent Challenge?" (arXiv 2011.09533; fetched, not read).
- Terry et al., "Revisiting Parameter Sharing in Multi-Agent Deep RL" (arXiv 2005.13625; fetched, not read): agent-indication schemes for shared policies.
- Li et al., "Graph Neural Networks for Decentralized Multi-Robot Path Planning" (arXiv 1912.06095; found by search, not fetched): as I recall, trained on small teams and tested on larger ones.
- Tolstaya et al., "Learning Decentralized Controllers for Robot Swarms with Graph Neural Networks" (arXiv 1903.10527; found by search, not fetched).
- Buffelli, Liò, Vandin, "SizeShiftReg" (arXiv 2207.07888; found by search, not fetched): a regulariser for GNN size generalisation.
- Tang et al., "Towards Scale-Invariant Graph-related Problem Solving by Iterative Homogeneous GNNs" (arXiv 2010.13547; found by search, not fetched).
- Li et al., "Permutation Invariant Policy Optimization for Mean-Field MARL" (arXiv 2105.08268; found by search, not fetched).
- Semantic Scholar was not queried (reported rate-limited); the arXiv API title search was unreliable for several exact titles, so absence from search is not evidence of absence.

---

## Index of cited papers and labels

[FULL] (28)
- 2006.04222 Iqbal et al., REFIL, ICML 2021
- 2101.08001 Hu et al., UPDeT, ICLR 2021
- 2003.10423 Long et al., Evolutionary Population Curriculum, ICLR 2020
- 1909.07528 Baker et al., Emergent Tool Use from Multi-Agent Autocurricula, ICLR 2020
- 1912.06680 OpenAI (Berner et al.), Dota 2 with Large Scale Deep RL, 2019
- 1810.02912 Iqbal and Sha, Actor-Attention-Critic (MAAC), ICML 2019
- 2103.01955 Yu et al., The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games
- 2410.01104 Veličković et al., Softmax is not Enough (for Sharp Size Generalisation)
- 1906.01202 Agarwal, Kumar, Sycara, Learning Transferable Cooperative Behavior in Multi-Agent Teams
- 2109.07735 Batra et al., Decentralized Control of Quadrotor Swarms with End-to-end Deep RL
- 2011.08055 Hsu, Jeong, Pappas, Chaudhari, Scalable RL Policies for Multi-Agent Control
- 2301.05334 Gallici, Martin, Masmitja, TransfQMix
- 1810.09202 Jiang et al., Graph Convolutional RL (DGN), ICLR 2020
- 1807.06613 Hüttenrauch, Šošić, Neumann, Deep RL for Swarm Systems, JMLR 2019
- 2205.14953 Wen et al., Multi-Agent RL is a Sequence Modeling Problem (MAT)
- 2310.11846 Liu et al., MaskMA
- 1806.01830 Zambaldi et al., Relational Deep RL
- 2004.05718 Corso et al., Principal Neighbourhood Aggregation
- 2005.09561 Richter and Wattenhofer, Normalized Attention Without Probability Cage
- 2009.11848 Xu et al., How Neural Networks Extrapolate, ICLR 2021
- 1909.02790 Wang et al., From Few to More (DyMA-CL)
- 2212.07489 Ellis et al., SMACv2
- 1907.03822 Khan et al., Graph Policy Gradients for Large Scale Robot Control
- 2308.03526 Mathieu et al., AlphaStar Unplugged
- 2302.07515 Lin et al., TiZero
- 2203.05285 Hao et al., Breaking the Curse of Dimensionality in Multiagent State Space (API/HPN) (appendix A.8 and abstract)
- 1911.00025 Liu, Yeh, Schwing, PIC: Permutation Invariant Critic
- 2206.11925 Zhang et al., Set Norm and Equivariant Skip Connections (definitions only)

[ABSTRACT] (19)
- 2105.08692 Liu et al., COPA; 2010.08853 Yehudai et al.; 1903.00784 Suarez et al., Neural MMO; 2501.19399 Nakanishi; 2505.17083 Anson et al.; 1802.05438 Yang et al., Mean Field MARL; 2403.03950 Farebrother et al., HL-Gauss; 1809.04474 Hessel et al., Multi-task PopArt; 1706.05296 Sunehag et al., VDN; 1803.11485 Rashid et al., QMIX; 2108.08612 Kuba et al.; 1912.02906 Qu, Wierman, Li; 2305.19466 Kazemnejad et al.; 2207.04901 Anil et al.; 2310.16028 Zhou et al.; 2103.05045 Bevilacqua et al.; 1901.09006 Wagstaff et al.; 2007.08973 Creswell et al., AlignNet; 2306.17770 Shi et al., MTR++

[MEMORY] (6)
- 1602.07714 van Hasselt et al., PopArt (existence verified)
- 2402.09371 Zhou et al., Transformers Can Achieve Length Generalization But Not Robustly (existence verified)
- 2402.19427 De et al., Griffin (RG-LRU; existence verified, the text contains "RG-LRU"; not otherwise used)
- 1902.04043 Samvelyan et al., SMAC (existence verified)
- 2006.06626 Qu, Lin, Wierman, Scalable MARL for Networked Systems with Average Reward (existence verified; not otherwise used)
- Vinyals et al., AlphaStar, Nature 2019 (not fetched)
