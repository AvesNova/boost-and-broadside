# References

Papers and published methods the project relies on or has used to decide something, grouped by topic. Code locations were checked against the tree on 2026-11-14; where a method is not in the code the entry says "planned" or "discussion only". Entries marked "(unverified)" were not checked against the publisher or arXiv in this pass (details taken from the repository's docs or from memory). The 2026-11-14 planning decisions are in `docs/rewrite/arms-todo.md`.

## Training algorithm

### PPO
- **Citation:** Schulman, Wolski, Dhariwal, Radford, Klimov, 2017, "Proximal Policy Optimization Algorithms", arXiv 1707.06347 (unverified).
- **What we take from it:** the clipped surrogate objective with several minibatch epochs per rollout.
- **Where it applies:** `src/bnb/learn/losses.py` (clipped surrogate), `src/bnb/learn/ppo.py`.
- **How it affected a decision:** it is the base RL objective; the trust-region variants below are arms against it.

### GAE
- **Citation:** Schulman, Moritz, Levine, Jordan, Abbeel, 2016, "High-Dimensional Continuous Control Using Generalized Advantage Estimation", ICLR 2016, arXiv 1506.02438 (unverified).
- **What we take from it:** lambda-weighted advantage estimates.
- **Where it applies:** `src/bnb/learn/returns.py` (`per_level_gae`, one discount and lambda per reward level).
- **How it affected a decision:** per-level discounts and lambdas live in the reward-function config (`docs/rewrite/yemong-review.md`).

### DAPO (clip-higher)
- **Citation:** Yu et al., 2025, "DAPO: An Open-Source LLM Reinforcement Learning System at Scale", arXiv 2503.14476 (unverified).
- **What we take from it:** an asymmetric clip, a larger upper margin than lower.
- **Where it applies:** `src/bnb/learn/losses.py` (`clip_high`), `src/bnb/learn/ppo.py` (`trust_region = "clip_higher"`, `clip_high` 0.28); `docs/explanation/learner.md`.
- **How it affected a decision:** an arm option against the symmetric clip and DPPO (`docs/rewrite/decisions.md`, DPPO item).

### DPPO
- **Citation:** Qi, Zhou, Liu, Pang, Du, Lin, Lee, 2026, "Rethinking the Trust Region in LLM Reinforcement Learning", ICML 2026, arXiv 2602.04879 (arXiv id and authors verified; venue as given in the repository's notes).
- **What we take from it:** replace ratio clipping with a mask on a direct divergence estimate (total variation, or its binary approximation) between old and new policies.
- **Where it applies:** `src/bnb/learn/losses.py` (`dppo_keep`, the DPPO surrogate), `src/bnb/learn/ppo.py` (`trust_region` = `dppo_tv`, `dppo_binary_tv`; `dppo_delta` 0.15); notes in `docs/rewrite/sections/dppo-notes.md`.
- **How it affected a decision:** to be tried in the RL fine-tunes against PPO and clip-higher, delta 0.10 / 0.15 / 0.20 (`docs/rewrite/decisions.md`). Open item: measure the total variation under today's clipping to see what 0.15 would mask (`docs/rewrite/arms-todo.md`).

### Time limits in RL
- **Citation:** Pardo, Tavakoli, Levdik, Kormushev, 2018, "Time Limits in Reinforcement Learning", ICML 2018, arXiv 1712.00378 (unverified).
- **What we take from it:** bootstrap through a time limit only when it is artificial and unobserved.
- **Where it applies:** `docs/rewrite/yemong-review.md` (time limit as termination); the learner's returns in `src/bnb/learn/returns.py`.
- **How it affected a decision:** the game's time limit is a rule and `time_remaining` is observed, so it is treated as a real termination (review, decided).

### Team spirit (OpenAI Five)
- **Citation:** see the OpenAI Five entry under league and self-play.
- **What we take from it:** per-level advantage = own + s x mean of living teammates' advantages.
- **Where it applies:** `src/bnb/learn/returns.py` (`team_mixed_levels`, `team_spirit`); explained in `docs/rewrite/yemong-review.md`.
- **How it affected a decision:** kept as config, 0 on every level.

## League and self-play

### AlphaStar
- **Citation:** Vinyals et al., 2019, "Grandmaster level in StarCraft II using multi-agent reinforcement learning", Nature 575, 350-354, doi 10.1038/s41586-019-1724-z (unverified).
- **What we take from it:** a league of past agents, prioritised fictitious self-play with a "hard" weighting (1-p)^2 on the probability p that the learner beats an opponent, and a fixed share of self-play for main agents; regularising toward the supervised policy; a value function that sees the opponent's information.
- **Where it applies:** league pool in `src/bnb/league/pool.py` (`sampling_weights` today weights by p(1-p) with a uniform floor); (1-p)^2 is decided but not implemented. Discussed in `docs/rewrite/decisions.md` (league sampling), `docs/rewrite/yemong-review.md` (KL to the BC policy; privileged critic).
- **How it affected a decision:** 2026-11-14: league sampling will be (1-p)^2 over all members, cheese agents included as ordinary members; the floor's size and latest's share stay open (`arms-todo.md`). Earlier (decisions.md) AlphaStar-style prioritised fictitious self-play was listed as "to consider".

### OpenAI Five
- **Citation:** Berner et al., 2019, "Dota 2 with Large Scale Deep Reinforcement Learning", arXiv 1912.06680 (unverified).
- **What we take from it:** training against past selves; team spirit mixing of advantages annealed from about 0.3 toward 1.
- **Where it applies:** league snapshots and averages in `src/bnb/league/pool.py`, `league.py`; team spirit in `src/bnb/learn/returns.py`.
- **How it affected a decision:** 2026-11-14 planning: the live policy trains against frozen past versions. Team spirit stays at 0 (`yemong-review.md`).

### AlphaZero
- **Citation:** Silver et al., 2017/2018, "Mastering Chess and Shogi by Self-Play with a General Reinforcement Learning Algorithm", arXiv 1712.01815; Science 362, 2018 (unverified).
- **What we take from it:** one network trained on the data of both players in self-play.
- **Where it applies:** discussion only for the citation; the code trains one policy on both teams' data (`src/bnb/learn/ppo.py`).
- **How it affected a decision:** 2026-11-14 planning: self-play on both sides' data is the training setup; the choice of one-sided training with a single league is deferred (`arms-todo.md`).

### Fictitious play and averaged agents
- **Citation:** none single; the repository's notes refer to "fictitious play" for training against a de-noised copy of oneself.
- **What we take from it:** the uniform and exponential averages of the live weights are rated and drawn as league opponents.
- **Where it applies:** `src/bnb/league/pool.py` (`UNIFORM_AVERAGE`, `EXPONENTIAL_AVERAGE`); `docs/rewrite/decisions.md` (league and evaluation).
- **How it affected a decision:** both averages beat the live policy about 3 games in 4 by 34M steps (run 764, decisions.md); kept in the league.

## Ratings and evaluation

### Bradley-Terry
- **Citation:** Bradley, Terry, 1952, "Rank Analysis of Incomplete Block Designs: I. The Method of Paired Comparisons", Biometrika 39(3/4), 324-345 (unverified).
- **What we take from it:** a logistic model of a win probability from a rating difference, fitted by maximum likelihood; its Fisher information is a graph Laplacian, which gives standard errors and A-optimal game allocation.
- **Where it applies:** `src/bnb/league/ratings.py` (fit with standard errors, `allocate_blocks`), `src/bnb/league/tournament.py`, `src/bnb/league/lanes.py`; `docs/explanation/league.md`.
- **How it affected a decision:** ratings are a Bradley-Terry fit on every game so far with the scripted agent pinned at 1000 (`docs/explanation/league.md`); draws count as half a win.

### Elo
- **Citation:** Elo, 1978, "The Rating of Chessplayers, Past and Present", Arco (unverified).
- **What we take from it:** the Elo scale (400-point logistic convention) for reporting ratings.
- **Where it applies:** reporting in `src/bnb/league/ratings.py` and the docs (scripted = 1000).
- **How it affected a decision:** ratings are reported in Elo units; no K-factor update is used (anchor-only continuous fit).

### Thurstone / probit link
- **Citation:** Thurstone, 1927, "A law of comparative judgment", Psychological Review 34(4), 273-286 (unverified).
- **What we take from it:** the probit link p = Phi(z) as an alternative to the logistic one.
- **Where it applies:** `src/bnb/league/cyclic.py` (`LINKS`, "probit"); `docs/explanation/game/real-scale-balance.md`.
- **How it affected a decision:** used as one model in the goodness-of-fit comparison of the real-scale balance rounds.

### Re-evaluating Evaluation (mElo, Nash averaging)
- **Citation:** Balduzzi, Tuyls, Perolat, Graepel, 2018, "Re-evaluating Evaluation", NeurIPS 2018, arXiv 1806.02643 (unverified).
- **What we take from it:** the multidimensional Elo (mElo-2k) for results with a cyclic part, and the Nash equilibrium of the meta-game as a balance criterion.
- **Where it applies:** `src/bnb/league/cyclic.py` (mElo-2k by penalised likelihood, held-out log-likelihood against Bradley-Terry); `docs/explanation/game/ship-design.md` (no dominant type); `docs/explanation/game/real-scale-balance.md`. The planned loop check in the rating pool (goodness of fit and a likelihood-ratio test against mElo) is in `arms-todo.md`; the file `src/bnb/league/goodness_of_fit.py` named in planning does not exist yet.
- **How it affected a decision:** 2026-11-14: the rating pool will include a loop check against mElo (planned). Balance work uses mElo to look for cycles.

### Learning-curve extrapolation and scaling laws
- **Citation:** Domhan, Springenberg, Hutter, 2015, "Speeding up Automatic Hyperparameter Optimization of Deep Neural Networks by Extrapolation of Learning Curves", IJCAI 2015 (unverified). Jones, 2021, "Scaling Scaling Laws with Board Games", arXiv 2104.03113 (verified).
- **What we take from it:** predicting a run's final strength from its early curve.
- **Where it applies:** discussion only.
- **How it affected a decision:** 2026-11-14: considered for judging arms early; decided against extrapolation as the decision rule. The rule is the endpoint Elo (mean of the last 2-3 snapshots) in one shared pool (`arms-todo.md`).

### Weight averaging (LAWA)
- **Citation:** Kaddour, 2022, "Stop Wasting My Time! Saving Days of ImageNet and BERT Training with Latest Weight Averaging", NeurIPS 2022 Workshops, arXiv 2209.14981 (verified).
- **What we take from it:** averaging the last k checkpoints (it finds k above 16 worse).
- **Where it applies:** planned. The uniform and exponential averages exist (`src/bnb/league/pool.py`); the sweep is not implemented.
- **How it affected a decision:** 2026-11-14: save the live weights every update in an early arm, then sweep averages (uniform, exponential at several decays, windowed uniform) in one tournament (`arms-todo.md`).

## Model architecture

### Griffin (RG-LRU)
- **Citation:** De et al., 2024, "Griffin: Mixing Gated Linear Recurrences with Local Attention for Efficient Language Models", arXiv 2402.19427 (unverified).
- **What we take from it:** the real-gated linear recurrent unit with a causal convolution as the temporal block; its decay initialisation (a^c in [0.9, 0.999]).
- **Where it applies:** `src/bnb/models/yemong/temporal.py` (RG-LRU, `TemporalSublayer`), Triton scan in `src/bnb/models/yemong/kernels/`; `docs/rewrite/yemong-review.md`.
- **How it affected a decision:** 2026-10-21: Griffin-only memory (no belief feedback) became the port's design; the paper's decay initialisation was kept (`docs/rewrite/decisions.md`). Griffin against GRU is idea 31 (`ideas.md`).

### Gated attention
- **Citation:** Qiu et al. (Qwen team), 2025, "Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free", NeurIPS 2025, arXiv 2505.06708 (verified).
- **What we take from it:** a sigmoid gate on each head's attention output, query-dependent; the elementwise head-specific gate did best of the forms compared.
- **Where it applies:** `YemongConfig.gate` ("none", "elementwise", "headwise"), `src/bnb/models/yemong/spatial.py`; `docs/explanation/yemong.md`.
- **How it affected a decision:** an arm option (`--set yemong.gate=elementwise`, about 1% cost; `ideas.md` item 32); default "none".

### Rotary position embeddings
- **Citation:** Su et al., 2021, "RoFormer: Enhanced Transformer with Rotary Position Embedding", arXiv 2104.09864 (unverified).
- **What we take from it:** rotating query/key pairs by position-dependent phases. Here the phases come from world x, y and attitude, with periods set by the map sizes.
- **Where it applies:** `src/bnb/models/yemong/layers.py` (rotary tables and rotation), `geometry.py`; `docs/rewrite/yemong-review.md`.
- **How it affected a decision:** absolute positions plus rotary kept over ego-centric inputs; rotary off collapses a trained model (ablation, decisions.md). Rotary against colour codes is idea 36.

### RMSNorm
- **Citation:** Zhang, Sennrich, 2019, "Root Mean Square Layer Normalization", NeurIPS 2019, arXiv 1910.07467 (unverified).
- **What we take from it:** normalisation by root mean square without centring.
- **Where it applies:** `src/bnb/models/yemong/layers.py` (`norm`, `nn.RMSNorm`).
- **How it affected a decision:** used throughout; RMSNorm against LayerNorm is idea 34.

### Gated MLP (GLU variants)
- **Citation:** Shazeer, 2020, "GLU Variants Improve Transformer", arXiv 2002.05202 (unverified).
- **What we take from it:** a gated feed-forward layer. Ours is `gelu(gate x) * (up x)`.
- **Where it applies:** `src/bnb/models/yemong/layers.py`, `kernels/gated.py`. The review docs call it SwiGLU; the kernel's gate is GELU.
- **How it affected a decision:** none recorded beyond the architecture choice.

### Transformer
- **Citation:** Vaswani et al., 2017, "Attention Is All You Need", NeurIPS 2017, arXiv 1706.03762 (unverified).
- **What we take from it:** pre-norm attention blocks over a set of tokens (one per ship, zone, and so on).
- **Where it applies:** `src/bnb/models/yemong/spatial.py`.
- **How it affected a decision:** base of Yemong; no decision recorded.

### Value and input encodings: DreamerV3, MuZero, HL-Gauss
- **Citation:** Hafner, Pasukonis, Ba, Lillicrap, 2023, "Mastering Diverse Domains through World Models" (DreamerV3), arXiv 2301.04104 (unverified). Schrittwieser et al., 2020, "Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model" (MuZero), Nature 588, arXiv 1911.08265 (unverified). Farebrother et al., 2024, "Stop Regressing: Training Value Functions via Classification for Scalable Deep RL", ICML 2024, arXiv 2403.03950 (unverified).
- **What we take from it:** symlog compression, two-hot / categorical targets with a cross-entropy loss and a percentile return normaliser (DreamerV3); an invertible squashing transform (MuZero); a Gaussian-smoothed histogram target (HL-Gauss, sigma 0.75 bin widths) over two-hot and regression.
- **Where it applies:** `src/bnb/models/yemong/codes.py` (`two_hot_at`, `hl_gauss`, `wrapped_hl_gauss`, `symlog_*`), `critic.py`, `state_code.py`; `docs/rewrite/yemong-review.md` (HL-Gauss discussion), `docs/rewrite/diagnostics.md` (return normaliser).
- **How it affected a decision:** HL-Gauss codes went into the port, and the critic uses symlog, then Gaussian (0.75 bins), then bin, read out with calibrated bin values (`yemong-review.md`, decided). Two-hot distributional value against scalar is idea 27; HL-Gauss against raw scalar inputs is idea 35.

### Centralised and asymmetric critics
- **Citation:** Pinto, Andrychowicz, Welinder, Zaremba, Abbeel, 2017, "Asymmetric Actor Critic for Image-Based Robot Learning", RSS 2018, arXiv 1710.06542 (unverified). Yu et al., 2022, "The Surprising Effectiveness of PPO in Cooperative, Multi-Agent Games" (MAPPO), NeurIPS 2022 Datasets and Benchmarks, arXiv 2103.01955 (unverified). Lyu, Baisero, Xiao, Amato, 2022, "A Deeper Understanding of State-Based Critics in Multi-Agent Reinforcement Learning", AAAI 2022, arXiv 2201.01221 (verified). Baisero, Amato, 2022, "Unbiased Asymmetric Reinforcement Learning under Partial Observability", AAMAS 2022, arXiv 2105.11674 (unverified).
- **What we take from it:** a critic with privileged state beside a fog-limited actor; shared weights across agents (MAPPO style); the caveat that a state-only critic biases the policy gradient in a partially observable game, so the critic wants history as well.
- **Where it applies:** discussion only for the privileged critic (`docs/rewrite/yemong-review.md`, literature note, about line 1372). Shared weights across ships (MAPPO-style) are in the model.
- **How it affected a decision:** 2026-11-14 planning: a centralised / no-fog critic was judged too much work for an arm. The values use the actor's own fogged view.

## Behaviour cloning

### DAgger
- **Citation:** Ross, Gordon, Bagnell, 2011, "A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning", AISTATS 2011, arXiv 1011.0686 (unverified).
- **What we take from it:** the teacher labels the states the student visits, which removes the covariate shift of plain cloning.
- **Where it applies:** the BC run: the student acts in both teams and the scripted teacher labels the visited states, with no teacher roll-in (`docs/explanation/behaviour-cloning.md`, `docs/rewrite/yemong-review.md`); `src/bnb/runtime/cloning.py`. Diagnostic I4 (loss on student against teacher-driven states) is not yet in place (`docs/rewrite/diagnostics.md`).
- **How it affected a decision:** BC uses on-policy imitation, the student's own state distribution (review, decided). Not full DAgger: no mixing of teacher actions.

## Other

### Regularising toward a supervised policy
- **Citation:** see AlphaStar above.
- **What we take from it:** a KL term to the frozen BC policy.
- **Where it applies:** `docs/rewrite/yemong-review.md` (R2, a step-decayed KL to the frozen BC policy); `docs/rewrite/decisions.md` (a step-based decay can replace the win-rate-gated BC term).
- **How it affected a decision:** discussed; the step-based schedule principle (no Elo-gated schedules) is decided; no claim here that the KL term is implemented.
