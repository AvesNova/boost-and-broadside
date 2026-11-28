# Loss balance on a shared trunk: what the literature says

Literature check, 2026-11-28, read-only. It follows docs/rewrite/literature-recipe-1.md
section 5 (2026-11-23), which already argued for fixed coefficients on scale-free terms and
gradient shares as diagnostics only. This page adds the numbers strong systems actually
use, the auxiliary-interference results, the diagnostic practices with their thresholds,
the teacher schedule in more detail, and the per-column question. It does not repeat the
earlier page's table of library defaults.

Labels as in the earlier page: **measured** (a source ran an experiment), **default** (a
system uses it, no published ablation), **inference** (mine, not a paper's result).

The setup under review (as given in the brief, not re-read from code):

| Term | Stage | Coefficient |
|---|---|---|
| Policy cross-entropy to the teacher's 42-way joint | BC | 1.0 |
| PPO surrogate on normalised advantages (clip 0.2, 4 epochs, 32 minibatches) | RL | 1.0 |
| Joint 42-way entropy | RL | 0.003 |
| Teacher cross-entropy | RL | 0.5 x 0.95^k, gated off at league win rate 0.45 (and, as I read the brief, when the coefficient drops below 0.005) |
| Value, categorical HL-Gauss, 3-5 reward levels | both | 0.05 |
| Next-state, categorical | both | 0.1 per column |
| Hit prediction | both | 0.05 |
| Pairwise relative geometry (range, range rate, line-of-sight rate, energy difference, angles), categorical | both | 0.1 per column |
| Ship-field distance; map-edge distance | both | 0.05 each |

Adam at 3e-4, global gradient clip 1.0. History: next_state at 1.0 owned 76% of the trunk
gradient and hurt.

## Summary

1. **Fixed coefficients are the norm** in every strong system read here. None of them
   adapts weights online; they make terms scale-free instead (reward/return normalisation,
   categorical losses, normalised advantages) and then fix the weights (measured for
   DreamerV3's robustness, default elsewhere).
2. **Value on a shared trunk with a categorical critic: 0.25** is the common value in the
   MuZero family (MuZero Reanalyse, Muesli, EfficientZero), against a policy weight of 1 to
   3. MSE critics on shared trunks use 0.5 to 1 (IMPALA, PPO, OpenAI Five). The project's
   0.05 is 5x below the categorical convention (default, not ablated in those papers).
3. **Auxiliary tasks on a shared trunk help when they predict task-relevant quantities and
   are kept small or separated in time**; reconstruction-like tasks can hurt final
   performance (UNREAL, measured). Interference is real for value-vs-policy (PPG, measured;
   Andrychowicz et al., measured). OpenAI Five ran most auxiliary heads behind a stop-gradient
   (default).
4. **Adaptive balancers (GradNorm, uncertainty weighting, PCGrad) did not beat a tuned
   weighted sum** in the two large comparisons (measured, earlier page). The RL-specific
   adaptive methods (Du et al. 2018 cosine gating, Lin et al. 2019) were measured only at
   small scale.
5. **No source gives a gradient-share threshold.** The cosine test of Du et al. uses
   per-layer cosines on the shared parameters, an EMA of 0.999, and a gate at 0.01 to 0.02.
6. **Teacher term**: every measured schedule decays to zero or close to it (VPT, kickstarting,
   Wolczyk et al., QDagger); constants plateau at the teacher (measured). VPT and Wolczyk et
   al. turn the entropy bonus off while the teacher term is on.
7. **Per-column**: a k-column head at c per column has a trunk-gradient weight between
   c x sqrt(k) (columns' gradients orthogonal) and c x k (aligned). Correlated geometry
   columns sit near the top, so a 6-column head at 0.1 behaves like roughly 0.4 to 0.6, not
   0.1 (inference). MuZero normalises by count (1/K over unroll steps) on purpose (default).

## 1. How strong systems set the weights

### 1.1 The numbers

| System | Trunk | Value loss and weight | Other weights | Normalisation | Source |
|---|---|---|---|---|---|
| PPO (Atari) | shared CNN | MSE, c1 = 1 | entropy 0.01 | advantages per minibatch | [PPO, Table 5][ppo] (default) |
| Common PPO code (OpenAI baselines, CleanRL) | shared | MSE, 0.5 | entropy 0.01 | advantages | [37 implementation details][ppo37] (default) |
| IMPALA | shared | "baseline loss scaling 0.5" | entropy 0.01 fixed, or searched in [5e-5, 1e-2] by PBT | reward clipping | [IMPALA, appendix tables][impala] (default) |
| OpenAI Five | one LSTM; "our value function and action policy share a network and share gradients" | MSE, "value loss weight" 1.0 (0.25 to 1.0 in the Rerun) | entropy 0.01 to 0.001; PPO clip 0.2 | "We normalize rewards using a running estimate of the standard deviation, and the value loss weight is applied post-normalization" | [OpenAI Five, Table 1 and appendix][five] (default) |
| OpenAI Five auxiliary heads (win probability, net worth rank, team objectives) | small MLPs on the LSTM output | "win probability passes gradients to the main LSTM ... with a very small weight; the other auxiliary predictions use Tensorflow's stop_gradient method to train on their own" | | | [OpenAI Five, appendix][five] (default) |
| AlphaStar | shared | V-trace + UPGO + TD(lambda) baselines (weights below) | distillation (KL to the supervised policy) 2e-3 on all action arguments, 0.1 on action type in the first four minutes; entropy 1e-4 | | Pseudocode as quoted in [mini-AlphaStar's port][minias]; [Nature paper][alphastar]. The port uses winloss V-trace 10 / baseline 5 and 4-6 / 1 for the other baselines; I could not check these against the original pseudocode |
| MuZero Reanalyse | shared representation | categorical value, **0.25**, "compared to weights of 1.0 for policy and reward targets" | | categorical value/reward on a transformed scale; loss of each head scaled by 1/K over K unroll steps | [MuZero, appendix H][muzero] (default) |
| Muesli | shared | categorical value, **0.25**; reward 1.0 | total policy loss 3.0; entropy 0.003 (Atari) or 0.0003; CMPO 1.0 | normalised advantages; value/reward losses in a "non-linearly transformed space" "to avoid having to tune, per game, the weighting of these unnormalized components" | [Muesli, Tables 5-6, section 4.5][muesli] (default) |
| EfficientZero | shared | value **0.25** | policy 1; self-supervised consistency 2 | | [EfficientZero, Table 4][ez] (default) |
| UNREAL | shared CNN-LSTM | A3C value inside the base loss; value replay lambda_VR = 1 | reward prediction lambda_RP = 1; pixel control lambda_PC log-uniform in [0.01, 0.1] (Labyrinth) or [1e-4, 1e-2] (Atari); entropy in [5e-4, 1e-2] | pixel control's weight "we need to fit this normalization factor" per domain | [UNREAL, appendix][unreal] (searched by random sampling) |
| PPG | separate policy and value networks, or one network with the value gradient detached in the policy phase | value trained in its own phase; beta_clone 1 | entropy 0.01; 6 auxiliary epochs every 32 policy iterations | reward normalisation | [PPG, appendix A][ppg] (measured ablations of phase frequency and epochs) |
| VPT fine-tuning | one transformer; PPG phases | sleep-phase value 0.5, auxiliary value 0.5, sleep KL 1.0; value head zero-initialised at the start of RL | KL to the frozen BC policy rho = 0.2, decayed x0.9995 per iteration; no entropy | "we normalize the value-function target by subtracting the mean and dividing by the standard deviation ... exponentially weighted moving average" | [VPT, appendix G, Table 6][vpt] (KL measured; the rest default) |
| DreamerV3 | world model trained separately; actor and critic are their own networks on model states | critic beta_val = 1, replay critic 0.3 | prediction 1, dynamics 1, representation 0.1 (first arXiv version: dynamics 0.5); actor entropy 3e-4 | percentile return normalisation; categorical critic "decoupling the scale of gradients from the prediction targets"; adaptive gradient clipping "decouples the clipping threshold from the loss scales" | [DreamerV3][dreamer] (robustness across domains measured) |
| HL-Gauss ("Stop regressing") | IMPALA actor-critic among others | cross-entropy replaces MSE | sigma / bin width 0.75 | | [Farebrother et al. 2024, sections 3-4][hlgauss]. I found no retuned value coefficient reported for the IMPALA runs |

### 1.2 What this says

- **Fixed, not normalised-by-the-optimiser.** None of these systems balances terms with an
  online weighting rule. Their scale control is in the losses: normalised rewards or returns
  (Five, VPT, DreamerV3), categorical values (MuZero family, DreamerV3), normalised
  advantages (Muesli, PPO). Muesli says so directly: the categorical reparametrisation is
  there "to avoid having to tune, per game, the weighting" (default).
- **A categorical critic on a shared trunk sits at 0.25 in the MuZero family**, against a
  policy term of 1 (MuZero, EfficientZero) or 3 (Muesli) that is itself a cross-entropy or a
  normalised-advantage loss. MSE critics on shared trunks use 0.5 to 1, but their scale
  depends on reward normalisation, so the numbers do not transfer (inference).
- **Separated value training is the other route.** PPG, VPT and DreamerV3 all keep the value
  gradient out of the policy's parameters for most or all steps; PPG's single-network
  variant detaches the value gradient "at the last layer shared between the policy and value
  heads" during the policy phase and "performs almost as well" as two networks (measured).
- **Auxiliary heads of production systems were cheap or isolated.** OpenAI Five's auxiliary
  heads were stop-gradient except win probability at "a very small weight"; UNREAL's pixel
  control weight was 0.01 to 0.1 and searched (default).

### 1.3 Two optimiser facts that change how to read the coefficients (inference)

- **Adam cancels a coefficient on parameters only one term touches.** Adam is invariant to a
  diagonal rescaling of the gradient ([Kingma and Ba 2014][adam]), so a head's own
  parameters move at the same speed whether its coefficient is 0.05 or 1, as long as the
  scaled gradient stays well above epsilon (1e-5). The coefficients act only where terms
  share parameters (the trunk) and through the global clip. A low value coefficient
  therefore does not slow the value head's fitting; it only reduces how much the trunk is
  shaped for value. This is my reading of Adam's update rule, not a paper's measurement.
- **The global clip couples all terms.** When the global norm exceeds 1.0, every term is
  scaled down together, so a large auxiliary gradient shrinks the policy step too.
  DreamerV3 moved to per-tensor adaptive clipping for this reason ("decouples the clipping
  threshold from the loss scales"). How often the clip binds is worth logging.

## 2. Auxiliary tasks on a shared trunk: help or interference

| Source | What was measured | Result |
|---|---|---|
| [UNREAL, Jaderberg et al. 2016][unreal] | Pixel control, reward prediction, value replay on a shared A3C trunk (Labyrinth, Atari) | Large gains in data efficiency and final score. But "learning to reconstruct only led to faster initial learning and actually made the final scores worse when compared to vanilla A3C", attributed to focusing "on reconstructing irrelevant parts of the visual input" (measured) |
| [PPG, Cobbe et al. 2020][ppg] | Shared vs separate networks; auxiliary phase frequency and epochs (Procgen) | "Interference between policy and value function optimization can negatively impact performance when parameters are shared." Auxiliary phases too often hurt: "each auxiliary phase interferes with policy optimization"; 6 auxiliary epochs best (measured) |
| [Andrychowicz et al. 2020][wm] | 50+ PPO design choices on MuJoCo, including shared vs separate MLPs and a shared "baseline cost" swept over {0.001, 0.1, 1, 10, 100} | "Separate value and policy networks appear to lead to better performance on four out of five environments" (measured; low-dimensional control, not a game trunk) |
| [Raileanu and Fergus 2021 (IDAAC)][daac] | Procgen | A shared representation overfits because the value needs more instance-specific information than the policy; separate networks plus an invariance loss generalise better (measured) |
| [Du et al. 2018][du] | Auxiliary loss gated by gradient cosine (gridworld, Atari distillation from a sub-optimal teacher, Breakout + Ms. Pac-Man) | Gating helps when the auxiliary task is partly misaligned; with a perfect teacher it underperformed simply adding the losses (measured, small scale) |
| [Lin et al. 2019][lin] | Online weighting of several auxiliary tasks by whether they reduce the main loss over a horizon (robot and Atari-like tasks) | Faster than earlier heuristics; reviewers raised high variance and sparse-reward doubts (measured; I read the abstract and reviews, not the paper) |
| [Schaul et al. 2019, ray interference][ray] | Multi-component objectives in RL | On-policy coupling makes components learn one at a time through plateaus (analysis, small experiments) |
| [DreamerV3][dreamer] | Stopping reward/value vs reconstruction gradients into the world model | "Dreamer rests predominantly on the unsupervised objective of its world model" (measured; separate actor-critic, so no policy interference) |
| [Kurin et al. 2022][kurin], [Xin et al. 2022][xin] | GradNorm, PCGrad, uncertainty weighting and others against a weighted sum | None reliably better than the weighted sum with tuning and standard regularisation. Kurin's RL experiment is multi-task SAC on Meta-World, i.e. several RL tasks, not auxiliary heads for one task (measured) |

What this says:

- **Predictive auxiliaries that encode task-relevant state help; generic reconstruction can
  hurt the final score.** The project's pairwise geometry heads (range, closure, line-of-sight
  rate, energy difference) are task-relevant by construction; whether next-state is closer
  to "relevant" or "reconstruction" depends on which columns it predicts (inference).
- **The one interference consistently measured is value vs policy**, and the measured fixes
  are structural (separate networks, phases, detaching), not reweighting (PPG, Andrychowicz,
  IDAAC).
- **When adaptive methods pay** (inference from the table): when an auxiliary task is known
  to be partly misaligned and cheap per-step cosines are available (Du), or with many tasks
  of unknown value (Lin). With a handful of hand-designed, task-relevant heads, a tuned fixed
  weighted sum matched them in every large comparison. I found no RL paper showing GradNorm,
  uncertainty weighting or PCGrad beating fixed auxiliary weights for a single-task PPO
  agent. A related method I found but did not read, PAGrad (projecting out the component of
  an auxiliary gradient that conflicts with the PPO gradient), is in [Ko and Ok 2021][pagrad].
- Uncertainty weighting ([Kendall et al. 2018][kendall]) assumes each term is a likelihood
  with a learnable noise scale. For cross-entropies whose floor is the target's own entropy
  it rewards the heads that are inherently noisy with lower weight (inference from the
  method's form); it was designed for supervised multi-task vision, not RL.

## 3. Diagnostics and thresholds

| Practice | Who | Details |
|---|---|---|
| Per-term gradient norm on the shared parameters | [GradNorm][gradnorm] | Measures each task's gradient norm at the last shared layer and drives them towards a common target scaled by each task's relative training rate (one asymmetry hyperparameter alpha). The quantity is used as a control signal; GradNorm gives no "acceptable share" |
| Cosine between main and auxiliary gradients | [Du et al. 2018][du] | On the shared parameters only. Atari: computed "on a per-layer basis and then averaged", smoothed with an EMA (0.999 old, 0.001 new); auxiliary kept when cosine >= 0.02 (Breakout) or 0.01 (two-game). The authors note mini-batch cosines are noisy, the fixed threshold is a limitation, and a cosine near 1 is uninformative (the auxiliary term then acts as a larger learning rate) |
| Conflict = negative cosine | [PCGrad, Yu et al. 2020][pcgrad] | Projects a task gradient onto the normal plane of another when their cosine is negative; their analysis names conflicting directions, large magnitude differences and high curvature together as the harmful case (from the paper; not re-read today) |
| Value-vs-policy interference | [PPG][ppg] | Tested by structure (shared vs separate, detached vs not) and by phase frequency, not by a gradient statistic |
| Gradient share thresholds | none found | I found no paper that recommends a maximum share of the trunk gradient for an auxiliary term |

Practical reading (inference):

- Shares and cosines are worth logging but are noisy per minibatch; Du et al. needed an EMA
  of 0.999 before the cosine signal was usable, and on ImageNet the gate only began to act
  after about 15,000 steps.
- **A share tells you who drives the trunk; a cosine tells you whether it fights the
  policy.** A large share with a positive cosine is a bigger learning rate in the policy's
  direction; a large share with a near-zero cosine is a term shaping features the policy does
  not currently use (not necessarily harmful); a sustained negative EMA cosine is the case
  Du et al. gate out.
- Under Adam the gradient share is not the update share (section 1.3), so a share number
  should not be read as "76% of the learning", only as "76% of the raw gradient norm".
- A rule of thumb I would use, not from any paper: investigate any single auxiliary term
  that holds more than half the trunk gradient norm, or whose EMA cosine with the policy
  gradient stays below zero for several updates; act by an on/off arm or a stop-gradient,
  not by chasing the share.

## 4. BC then RL: the teacher term

### 4.1 Schedules

| Source | Form | Schedule | Result |
|---|---|---|---|
| [VPT][vpt] | rho x KL(pi_pretrained, pi_theta) on the learner's states; replaces the entropy bonus | rho 0.2, x0.9995 per iteration (half-life about 1,400 iterations); 0.4 in one variant | Without it the agent only got early items and needed a 7x lower learning rate; "a high coefficient rho ... would prevent the agent from properly optimizing the reward function while a low coefficient rho was ineffective" (measured) |
| [Kickstarting, Schmitt et al. 2018][kick] | lambda_k x H(pi_teacher, pi_student) on the student's own trajectories; shared policy/value trunk | Constant 1 or 2; linear 1 or 2 to 0 at 1B, 2B or 4B frames; PBT | "constant schedules can perform well early on, but ultimately cause the agent to plateau near the teacher's final performance"; linear to 0 better, "it is important to reduce this weight quickly"; PBT nearly as good as the best hand schedule; under PBT the weight fell to "almost negligible" after 2B of 10B frames (measured) |
| [AlphaStar][alphastar] ([pseudocode port][minias]) | KL to the supervised policy | 2e-3 on all arguments, permanent; 0.1 on action type for the first four minutes of each game | Kept for exploration and strategic diversity in a league (default) |
| [QDagger, Agarwal et al. 2022][qdagger] | Distillation weight on a value-based student | lambda_t = 1{t < t0} x max(1 - G_student / G_teacher, 0), updated every iteration; linear decay "also worked well" | Fixed coefficient saturates early; the decayed one keeps improving (measured, Atari) |
| [Wolczyk et al. 2024][forget] | Kickstarting on NetHack (APPO) | 0.5 x 0.99998 per train step (half-life about 35,000 steps); "we turn off entropy when employing knowledge retention methods in similar fashion to (Baker et al., 2022)"; encoders frozen | Kickstarting best on NetHack (5K to over 10K score); behaviour cloning on teacher data best on other tasks (measured) |
| [Jump-Start RL][jsrl] | Guide policy plays the first h steps, learner the rest; h shrinks on a performance gate | Gate on the learner's moving average | Measured on robot tasks; different mechanism (no loss term) |

What this says for 0.5 x 0.95^k with a win-rate gate:

- The starting weight 0.5 matches Wolczyk et al.'s kickstarting weight and sits between VPT
  (0.2) and kickstarting (1 to 2) (default). The half-life of 0.95^k is about 13.5 decay
  steps; whether that is fast or slow depends on what one k is (one update? one checkpoint?)
  relative to the RL budget. Kickstarting found faster better within the range tried
  (measured); VPT and Wolczyk used half-lives of hundreds of iterations or tens of thousands
  of steps (default).
- A performance gate has two precedents: QDagger's student-to-teacher return ratio (measured)
  and JSRL's moving-average gate (measured). Gating on the league win rate is close to
  QDagger's idea (inference).
- **Entropy while the teacher term is on.** VPT replaced the entropy bonus with the KL term
  and Wolczyk et al. followed it; kickstarting kept A3C's entropy cost. The cross-entropy to
  a full teacher distribution is mass-covering (it penalises the student for putting near-zero
  probability where the teacher has mass), so it already does part of entropy's job while it
  is large (inference). The 0.003 entropy matters mainly after the gate closes.
- **Value head at the hand-over.** VPT trained no value during BC and zero-initialised the
  value head "which appeared to prevent destructive updates early in training" (default);
  the earlier page notes JSRL measured a cloned policy collapsing with a fresh critic. The
  project trains the value during BC, which avoids both problems (inference).

### 4.2 Auxiliary heads during BC

- No source in this set ablates auxiliary heads on vs off during a BC stage that precedes RL.
- Indirect evidence for on: pre-training representations with latent dynamics and other
  self-supervised objectives improved later RL fine-tuning on Atari ([Schwarzer et al. 2021][sgi],
  measured); DreamerV3's representation "rests predominantly" on its unsupervised losses
  (measured); OpenAI Five trained its auxiliary heads throughout (default).
- Indirect evidence for caution: VPT's BC trained the policy only (default); Wolczyk et al.
  froze the encoder during fine-tuning to keep pre-trained features stable (default).
- My reading (inference): keeping the heads on during BC is the safer default because the
  trunk then reaches RL already encoding the predicted quantities, and switching them on
  only at the hand-over would add a new gradient source to the trunk exactly when the
  teacher term starts decaying. The 76% next-state history says the risk is the size of the
  auxiliary load, not the stage.

## 5. Per-column against per-head weighting

The arithmetic (inference). A head with k columns, each a cross-entropy L_j with trunk
gradient g_j, at c per column, adds c x (g_1 + ... + g_k) to the trunk. With equal column
norms |g|:

| Column gradients | Trunk-gradient norm of the head | k = 6, c = 0.1 behaves like |
|---|---|---|
| Orthogonal (independent targets) | c x sqrt(k) x \|g\| | 0.24 x \|g\| |
| Fully aligned (redundant targets) | c x k x \|g\| | 0.60 x \|g\| |

Range, range rate, line-of-sight rate and angles of one pair are computed from the same
relative state and are likely to have strongly aligned gradients, so the 6-column geometry
head probably behaves closer to 0.6 than to 0.24 (inference; measurable with the existing
grad diagnostics by logging the head's norm against the sum of its columns' norms). The same
holds for any reduction across pairs: if the pairwise loss sums over the 7 other ships
instead of averaging, the weight multiplies again.

Precedents:

- **Normalise by count**: MuZero scales the loss of each head by 1/K over K unroll steps so
  "the total gradient has similar magnitude irrespective of how many steps we unroll for"
  ([MuZero][muzero], default). UNREAL had to fit pixel control's weight per domain because it
  aggregates many cells ([UNREAL][unreal]).
- **Sum over dimensions**: DreamerV3's prediction loss is the negative log-likelihood of the
  whole observation, i.e. summed over its dimensions, at weight 1 next to the 1-dimensional
  reward and continuation terms ([DreamerV3][dreamer], equation 4); its reconstruction then
  dominates the world model. Its actor and critic are separate networks, so this does not
  touch the policy.
- GradNorm, uncertainty weighting and PCGrad all weight per task; what counts as a task is
  the user's choice.

What this means for the current numbers (inference): the auxiliary load on the trunk is the
sum over all heads of c x (effective column count). With the pairwise geometry head at 0.1
per column over 6 or more columns, the next-state head at 0.1 per column over its columns,
and hit, field and edge heads at 0.05, the total auxiliary weight can exceed the policy
term's 1.0 without any single coefficient looking large. If next-state has 10 or more
columns, 0.1 per column is roughly back at the 1.0 that owned 76% of the trunk gradient.
A per-head mean (coefficient divided by column count, and by pair count for pairwise
heads) makes a coefficient mean the same thing whatever the head's width; a per-column sum
is defensible only if each column is meant to count as its own task.

## Recommendations (mine, for this setup)

These are recommendations, not decisions; they follow from the sources above plus the
inferences marked in each section.

**Keep fixed**

1. Keep fixed coefficients and scale-free losses (normalised advantages, every other term a
   cross-entropy in nats). Every strong system read here works this way.
2. Keep the teacher term's form (full-distribution cross-entropy on the learner's states),
   its decay and its performance gate. Keep the auxiliary heads on during BC.

**Measure first (cheap, no new runs)**

3. Write down, per head, the column count and pair count and whether the loss sums or
   averages over them; compute the effective per-head weight and the total auxiliary weight
   against the policy's 1.0. This answers question 5 for the current config in minutes.
4. From the existing `grad_*` diagnostics, on the trunk only: each head's share of the trunk
   gradient norm; each head's EMA (0.999) cosine with the policy-surrogate gradient,
   averaged per layer as Du et al. did; the ratio of a multi-column head's norm to the sum of
   its columns' norms (tells whether the head acts like c x sqrt(k) or c x k); how often the
   global clip at 1.0 binds.
5. Say what one k is in 0.95^k, and compute the teacher term's half-life in updates and in
   environment steps against the RL budget.

**Try first (one change per arm, built on the current champion)**

6. Per-head normalisation: divide each multi-column head's coefficient by its column count
   (and pairwise heads by pair count), keeping the head-level numbers (0.1, 0.05) as the
   per-head weights. This is the most direct guard against a repeat of the 76% next-state
   case.
7. Value 0.05 against 0.25 (the MuZero-family value for a categorical critic on a shared
   trunk). Under Adam this changes only how much the trunk learns value features, not how
   fast the value head fits, so watch explained variance and Elo together.
8. If a head's EMA cosine with the policy gradient is persistently negative, or its share
   stays above half: an arm with that head behind a stop-gradient (OpenAI Five's choice), not
   a smaller coefficient. If value-policy interference shows: PPG-style detaching of the
   value gradient in the policy epochs plus a separate value pass is the measured fix.
9. Entropy 0 while the teacher coefficient is above some floor (VPT, Wolczyk et al.), 0.003
   after the gate closes, as one arm against the current constant 0.003.

**Do not**

10. Do not adopt GradNorm, uncertainty weighting or PCGrad as a first step; the large
    comparisons found no gain over a tuned weighted sum, and no RL result shows one for
    auxiliary heads on a single-task PPO trunk.

## What I could not verify

- PPO's Table 5 values (c1 = 1, entropy 0.01 on Atari), PCGrad's analysis, Kendall et al.'s
  method details and Lin et al.'s results are from my knowledge of those papers, not
  re-read today.
- AlphaStar's numbers come from the mini-AlphaStar port, which quotes the original
  pseudocode for the KL weights (2e-3, 0.1) and the entropy sign; its baseline cost
  constants (10 / 5, 4-6 / 1) are not quoted, and the Nature text was not read.
- DreamerV3's dynamics weight differs between paper versions (0.5 in the first arXiv
  version, 1 in the version read today).
- Whether the HL-Gauss paper retuned the value coefficient for its IMPALA runs: not stated
  in the text I read.
- PAGrad (Ko and Ok 2021) was found by search and not read.
- The setup table is from the brief; I did not read `lean/learn` to check column counts,
  reductions over pairs, or how the value loss reduces over levels.

## Sources

- [ppo]: Schulman et al. 2017, Proximal Policy Optimization Algorithms. https://arxiv.org/abs/1707.06347
- [ppo37]: Huang et al. 2022, The 37 Implementation Details of PPO. https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
- [impala]: Espeholt et al. 2018, IMPALA. https://arxiv.org/abs/1802.01561
- [five]: Berner et al. 2019, Dota 2 with Large Scale Deep RL (OpenAI Five). https://arxiv.org/abs/1912.06680
- [alphastar]: Vinyals et al. 2019, Grandmaster level in StarCraft II. https://www.nature.com/articles/s41586-019-1724-z
- [minias]: mini-AlphaStar, rl_loss.py. https://github.com/liuruoze/mini-AlphaStar/blob/main/alphastarmini/core/rl/rl_loss.py
- [muzero]: Schrittwieser et al. 2020, MuZero. https://arxiv.org/abs/1911.08265
- [muesli]: Hessel et al. 2021, Muesli. https://arxiv.org/abs/2104.06159
- [ez]: Ye et al. 2021, EfficientZero. https://arxiv.org/abs/2111.00210
- [unreal]: Jaderberg et al. 2016, UNREAL. https://arxiv.org/abs/1611.05397
- [ppg]: Cobbe et al. 2020, Phasic Policy Gradient. https://arxiv.org/abs/2009.04416
- [vpt]: Baker et al. 2022, Video PreTraining. https://arxiv.org/abs/2206.11795
- [dreamer]: Hafner et al. 2023, DreamerV3. https://arxiv.org/abs/2301.04104
- [hlgauss]: Farebrother et al. 2024, Stop Regressing. https://arxiv.org/abs/2403.03950
- [wm]: Andrychowicz et al. 2020, What Matters in On-Policy RL. https://arxiv.org/abs/2006.05990
- [daac]: Raileanu and Fergus 2021, Decoupling Value and Policy for Generalization in RL. https://arxiv.org/abs/2102.10330
- [du]: Du et al. 2018, Adapting Auxiliary Losses Using Gradient Similarity. https://arxiv.org/abs/1812.02224
- [lin]: Lin et al. 2019, Adaptive Auxiliary Task Weighting for RL. https://papers.nips.cc/paper/2019/hash/0e900ad84f63618452210ab8baae0218-Abstract.html
- [ray]: Schaul et al. 2019, Ray Interference. https://arxiv.org/abs/1904.11455
- [gradnorm]: Chen et al. 2018, GradNorm. https://arxiv.org/abs/1711.02257
- [kendall]: Kendall et al. 2018, Multi-Task Learning Using Uncertainty to Weigh Losses. https://arxiv.org/abs/1705.07115
- [pcgrad]: Yu et al. 2020, Gradient Surgery for Multi-Task Learning. https://arxiv.org/abs/2001.06782
- [kurin]: Kurin et al. 2022, In Defense of the Unitary Scalarization. https://arxiv.org/abs/2201.04122
- [xin]: Xin et al. 2022, Do Current Multi-Task Optimization Methods in Deep Learning Even Help? https://arxiv.org/abs/2209.11379
- [pagrad]: Ko and Ok 2021, Efficient Scheduling of Data Augmentation for Deep RL. https://arxiv.org/abs/2102.08581
- [kick]: Schmitt et al. 2018, Kickstarting Deep RL. https://arxiv.org/abs/1803.03835
- [qdagger]: Agarwal et al. 2022, Reincarnating RL. https://arxiv.org/abs/2206.01626
- [forget]: Wolczyk et al. 2024, Fine-tuning RL Models is Secretly a Forgetting Mitigation Problem. https://arxiv.org/abs/2402.02868
- [jsrl]: Uchendu et al. 2023, Jump-Start RL. https://arxiv.org/abs/2204.02372
- [sgi]: Schwarzer et al. 2021, Pretraining Representations for Data-Efficient RL. https://arxiv.org/abs/2106.04799
- [adam]: Kingma and Ba 2014, Adam. https://arxiv.org/abs/1412.6980

[ppo]: https://arxiv.org/abs/1707.06347
[ppo37]: https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
[impala]: https://arxiv.org/abs/1802.01561
[five]: https://arxiv.org/abs/1912.06680
[alphastar]: https://www.nature.com/articles/s41586-019-1724-z
[minias]: https://github.com/liuruoze/mini-AlphaStar/blob/main/alphastarmini/core/rl/rl_loss.py
[muzero]: https://arxiv.org/abs/1911.08265
[muesli]: https://arxiv.org/abs/2104.06159
[ez]: https://arxiv.org/abs/2111.00210
[unreal]: https://arxiv.org/abs/1611.05397
[ppg]: https://arxiv.org/abs/2009.04416
[vpt]: https://arxiv.org/abs/2206.11795
[dreamer]: https://arxiv.org/abs/2301.04104
[hlgauss]: https://arxiv.org/abs/2403.03950
[wm]: https://arxiv.org/abs/2006.05990
[daac]: https://arxiv.org/abs/2102.10330
[du]: https://arxiv.org/abs/1812.02224
[lin]: https://papers.nips.cc/paper/2019/hash/0e900ad84f63618452210ab8baae0218-Abstract.html
[ray]: https://arxiv.org/abs/1904.11455
[gradnorm]: https://arxiv.org/abs/1711.02257
[kendall]: https://arxiv.org/abs/1705.07115
[pcgrad]: https://arxiv.org/abs/2001.06782
[kurin]: https://arxiv.org/abs/2201.04122
[xin]: https://arxiv.org/abs/2209.11379
[pagrad]: https://arxiv.org/abs/2102.08581
[kick]: https://arxiv.org/abs/1803.03835
[qdagger]: https://arxiv.org/abs/2206.01626
[forget]: https://arxiv.org/abs/2402.02868
[jsrl]: https://arxiv.org/abs/2204.02372
[sgi]: https://arxiv.org/abs/2106.04799
[adam]: https://arxiv.org/abs/1412.6980

## Answers from the code (coordinator, 2026-11-28)

- Next-state has 6 columns (lean/model/policy.py NEXT_STATE_COLUMNS). Its loss is the mean over
  rows and columns (training.py: divided by next_count x columns), not a sum.
- The geometry heads are likewise nats per column, the sampled pairs weighted and divided by the
  minibatch's valid pair count (lean/learn/geometry.py geometry_losses): a mean over pairs.
- The value loss is the mean over reward levels (divided by own_count x levels).
- k in teacher_decay ** k is the RL update index (one update = 1,048,576 environment ticks): a
  half-life of 13.5 updates (~14M ticks); the weight falls below teacher_cutoff 0.005 at
  update ~90 (0.5 x 0.95^90 = 0.0049, ~94M ticks), unless the win-rate gate closes it earlier. The cutoff is
  on the coefficient, as read.
- Because every head averages over its columns (and pairs), section 5's c x sqrt(k) .. c x k
  becomes c / sqrt(k) .. c on the trunk gradient: a head's coefficient is an upper bound on its
  weight, not a multiple of it.
- There are no grad_* diagnostics in lean today (only the global clip's norm); the gradient-share
  measurement needs a small script.
