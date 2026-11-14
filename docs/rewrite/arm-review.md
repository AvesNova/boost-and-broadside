# Arm review (draft for the author)

This lists every candidate arm still open as of 2026-11-15, prepared for choosing the run budget and the order of arms.

**Fields:**
- **What / why:** what the arm changes, and the reason to test it.
- **Paper question:** the question a reader would ask that this arm answers.
- **Purpose:** a better agent, a game-design decision, or a scientific claim.
- **Why an arm:** why a cheaper test or theory can't settle it.
- **When:** early, mid or late in the chain, or anywhere.
- **Cost:** an **RL fork** (RL only, from the shared BC; about half an arm) or **BC+RL** (its own BC; a full arm).
- **Importance:** high, medium or low.

The challenger is normally the more complex or slower option. It wins only by beating the incumbent past the gating threshold on the overall score, without losing any mode by the gate's margin ([evaluation-suite.md](evaluation-suite.md)). Numbers in brackets point to the original list in [ideas.md](ideas.md).

## Early: choices that change every later arm

**A1. Map sizes: fixed against variable** [54]
- **What / why:** train on each mode's fixed map against maps drawn at 0.5–2× scale. rl-v20, trained on fixed maps, collapses in frontline above 1.4×.
- **Paper question:** does the agent generalise across map sizes, and does it need to train on them?
- **Purpose:** game design (variable maps stay or go), plus a generalisation claim.
- **Why an arm:** only training shows whether variable maps are learnable at the same strength. If they aren't, the colour codes and rotary frequencies get re-tuned for fixed maps.
- **When:** first.
- **Cost:** BC+RL, since BC's data comes from the maps.
- **Importance:** high.

**A2. Ship and field counts: fixed 5v5 against variable** [55]
- **What / why:** variable team sizes, with absent ships and fields masked. Deathmatch and elimination go down to 1v1, frontline to 2v2.
- **Paper question:** does one model generalise to fleet sizes it never trained on (2v2, 10v10, 20v20)? This is the paper's main claim.
- **Purpose:** a scientific claim, plus the agent the evaluation suite rewards.
- **Why an arm:** whether variable counts cost strength at 5v5 or improve generalisation can only be measured. rl-v20 is only even with Imugi at 2v2. Under the density rule, counts also vary the map size, so it runs after A1.
- **When:** second.
- **Cost:** BC+RL.
- **Importance:** high.

**A3. Seed pairs (2 arms × 2 seeds)** [68]
- **What / why:** the first pair of arms, each run again at a second seed.
- **Paper question:** are the differences between arms larger than seed noise?
- **Purpose:** process. It sets the sequential test's indifference zone, and shows whether seed isolation couples arms as intended.
- **Why an arm:** noise in RL can only be measured by running.
- **When:** very early. It doubles A1, and takes the place of a separate noise study.
- **Cost:** two extra runs of A1's size.
- **Importance:** high.

**A4. Minibatches: 32 against 256** [3]
- **What / why:** PPO minibatches per epoch, which changes the size of each gradient step.
- **Paper question:** are the PPO settings sound?
- **Purpose:** a better agent and faster training; 256 minibatches are smaller, which may change throughput.
- **Why an arm:** the effect on final strength isn't predictable. The author asked for it early, from one BC checkpoint.
- **When:** early.
- **Cost:** RL fork.
- **Importance:** medium.

**A5. Training mix: 6/5/5 against 8/4/4** (frontline/deathmatch/elimination)
- **What / why:** the share of games per mode.
- **Paper question:** how should modes share training when skills transfer between them?
- **Purpose:** a better agent.
- **Why an arm:** the evidence points both ways (deathmatch and elimination share skill; frontline is distinct and still climbing).
- **When:** early, because it affects every later RL run.
- **Cost:** RL fork.
- **Importance:** medium.

**A6. Self-play: train both sides against one side** (plus more environments, and "latest" as a league member)
- **What / why:** whether each self-play game trains both seats.
- **Paper question:** is training both seats of a self-play game an efficient use of compute?
- **Purpose:** a better agent per unit of compute.
- **Why an arm:** both are standard practice, and the trade-off (correlated rows against about 7% more time) has no theoretical answer.
- **When:** early, since it's part of the training setup.
- **Cost:** RL fork.
- **Importance:** medium.

**A7. Offensive premium: zero-sum 1:1, 2:1, and 2:1 annealed to 1:1** (three forks; damage and kill/death only)
- **What / why:** the author's hypothesis that passive play is an equilibrium of zero-sum self-play, and a premium makes it worth searching for the rare winning aggressive line.
- **Paper question:** does a non-zero-sum shaping term help self-play escape stalemates, and does annealing it back recover the true objective?
- **Purpose:** a better agent, plus a scientific claim.
- **Why an arm:** the only evidence is a recollected v1 result in an older game. The split critic makes this cheap.
- **When:** early to mid, since reward design affects later arms.
- **Cost:** two RL forks beyond the 1:1 baseline.
- **Importance:** high.

## Middle: training algorithm

**A8. Trust region: clipping against DPPO**
- **What / why:** PPO's ratio clip against DPPO's divergence mask, which is already implemented.
- **Paper question:** does a trust region based on the actual policy divergence beat ratio clipping in a small joint action space?
- **Purpose:** a better agent.
- **Why an arm:** the paper's evidence is from language models with huge vocabularies. Our clip binds in 22–26% of samples in uncertain states, so it could matter. A cheap check of what `dppo_delta` = 0.15 would mask comes first.
- **When:** mid, before A9.
- **Cost:** RL fork.
- **Importance:** medium.

**A9. Learning rate: ×1 against ×2** [4]
- **What / why:** the learning rate.
- **Paper question:** is the optimiser tuned?
- **Purpose:** a better agent.
- **Why an arm:** the low KL suggests room, but the clip already binds where it matters.
- **When:** mid, after A8. Re-checked at the width arm.
- **Cost:** RL fork.
- **Importance:** medium.

**A10. Entropy: today's bonus against a stronger bonus or an entropy floor** [7]
- **What / why:** joint entropy fell to 0.45 late in rl-v20.
- **Paper question:** is exploration adequate late in training?
- **Purpose:** a better agent.
- **Why an arm:** in comparable units, the coefficient is within the normal range, so only training shows whether late training is under-exploring.
- **When:** anywhere.
- **Cost:** RL fork.
- **Importance:** medium-low.

**A11. Steps per update: 2048 against 4096 environments, or a 128-tick against 256-tick rollout** [1, 2]
- **What / why:** how much data each update sees, and, for the rollout length, how far each estimate looks ahead before the critic takes over.
- **Paper question:** are the rollout settings sound? (128 was chosen by feel in v1.)
- **Purpose:** a better agent.
- **Why an arm:** the effects interact with the long-horizon discounts.
- **When:** anywhere. Choose one of the two variants.
- **Cost:** RL fork; 4096 environments may not fit in 32 GB.
- **Importance:** medium-low.

**A12. Dense rewards decayed to zero partway** [23]
- **What / why:** the damage and kill/death rewards fade out over RL, leaving the objective and outcome.
- **Paper question:** are the dense rewards scaffolding, or needed throughout?
- **Purpose:** a better agent, plus a reward-design claim.
- **Why an arm:** with per-level critics it's only a change of weights, so it's cheap, but its effect has to be measured.
- **When:** mid, after A7, which also changes the dense levels.
- **Cost:** RL fork.
- **Importance:** medium.

## Middle: model

**A13. No auxiliary heads** [44]
- **What / why:** remove every auxiliary head: next-state, correction, density, enemy action, shot and geometry. This also weakens the belief.
- **Paper question:** do the auxiliary heads, which are a large part of the design, actually help?
- **Purpose:** a scientific claim.
- **Why an arm:** the gradient-share diagnostics show what the heads cost, not what they're worth.
- **When:** mid.
- **Cost:** BC+RL.
- **Importance:** high for the paper.

**A14. Belief input: the next-step heads' outputs for hidden ships against the temporal layers alone** [42]
- **What / why:** whether hidden enemies are fed back as predicted states.
- **Paper question:** does an explicit predicted belief help under fog, beyond what the recurrent layers learn?
- **Purpose:** a scientific claim, and one of the more novel parts.
- **Why an arm:** never tested.
- **When:** mid.
- **Cost:** BC+RL.
- **Importance:** medium-high.

**A15. Layout: 2 Yemong blocks against 6 spatial blocks** [30]
- **What / why:** temporal and spatial blocks interleaved, against purely spatial depth.
- **Paper question:** does the temporal processing earn its compute?
- **Purpose:** a scientific claim, plus the architecture choice.
- **Why an arm:** an architecture effect, not predictable.
- **When:** mid.
- **Cost:** BC+RL.
- **Importance:** medium-high.

**A16. Temporal block: Griffin against GRU** [31]
- **What / why:** the recurrent block.
- **Paper question:** is the modern recurrent block better than a classic GRU here?
- **Purpose:** the architecture choice.
- **Why an arm:** not predictable at our scale.
- **When:** mid, after A15.
- **Cost:** BC+RL.
- **Importance:** medium.

**A17. Input encoding: the colour/HL-Gauss codes against raw scalars** [35]
- **What / why:** how observations enter the model.
- **Paper question:** are the codes, a distinctive part of the design, worth their complexity?
- **Purpose:** a scientific claim.
- **Why an arm:** the earlier Fourier-against-colour comparison (−212 Elo) was on an older model and game, and never against raw inputs.
- **When:** mid, after A1, because map sizes change the codes.
- **Cost:** BC+RL.
- **Importance:** medium-high.

**A18. Position encoding: rotary against the colour code against both** [36]
- **What / why:** how attention sees relative positions.
- **Paper question:** which position scheme does a set-based model need?
- **Purpose:** the architecture choice.
- **Why an arm:** rotary was never ablated; keeping both was decided by review.
- **When:** mid, after A1.
- **Cost:** BC+RL.
- **Importance:** medium.

**A19. Qwen gated attention (elementwise)** [32]
- **What / why:** a sigmoid gate on each head's output, about 1% cost.
- **Paper question:** does output gating help a small, short-context model that we know uses an attention sink?
- **Purpose:** a better agent.
- **Why an arm:** the paper's setting is large language models; our setting differs.
- **When:** anywhere.
- **Cost:** BC+RL.
- **Importance:** medium-low.

**A20. Attention heads: 2 against 4** [28]
- **What / why:** 64 against 32 dimensions per head.
- **Paper question:** a minor architecture detail.
- **Purpose:** a better agent.
- **Why an arm:** small, unpredictable effect.
- **When:** anywhere.
- **Cost:** BC+RL.
- **Importance:** low.

**A21. Value output: two-hot distributional against a symlog scalar** [27]
- **What / why:** the critic's output form.
- **Paper question:** does the distributional value earn its complexity?
- **Purpose:** simplification, or a claim.
- **Why an arm:** the literature is mixed at small scale.
- **When:** anywhere. Note the split critic doubles the critic outputs.
- **Cost:** BC+RL, since BC trains the critic.
- **Importance:** low-medium.

**A22. Shot head: full bullet trajectory against "if I shot now", damage to each enemy and ally** [74]
- **What / why:** which shot prediction to train.
- **Paper question:** which auxiliary target helps aiming?
- **Purpose:** a better agent.
- **Why an arm:** both are plausible; no data.
- **When:** late, after A13 shows whether auxiliary heads matter at all.
- **Cost:** BC+RL.
- **Importance:** low-medium.

## Late

**A23. Width: 128 against 192** [29]
- **What / why:** model capacity, compared at equal steps and at equal cost, with a learning-rate re-check.
- **Paper question:** is the model capacity-limited, and what does scale buy?
- **Purpose:** a better final agent, plus a scaling claim.
- **Why an arm:** capacity effects can only be measured, and the learning rate interacts with width.
- **When:** last, on the final configuration.
- **Cost:** BC+RL at about 1.5–2× the compute.
- **Importance:** medium-high.

**A24. BC→RL against RL from scratch** [20]
- **What / why:** the same rewards with no BC, at 2–3× the steps.
- **Paper question:** is the scripted-agent BC start necessary, and what does it save?
- **Purpose:** a scientific claim.
- **Why an arm:** only training shows it. It may fail to get going without BC.
- **When:** late, or in parallel. It doesn't feed the chain.
- **Cost:** about 2–3 RL arms.
- **Importance:** high for the paper.

## Reference runs (not pairwise)

**R1. MLP baseline** [43]
- **What / why:** a simple model trained the same way.
- **Paper question:** how much does the architecture matter at all?
- **Purpose:** a scientific baseline.
- **When:** anywhere, in parallel.
- **Cost:** BC+RL, cheaper per step.
- **Importance:** high for the paper.

**R2. Final model with 2–3 seeds** [69]
- **What / why:** the final configuration repeated at other seeds.
- **Paper question:** is the final result reproducible?
- **When:** end; the author decides then.
- **Cost:** 1–2 full runs.
- **Importance:** medium.

**R3. Exploiter against the frozen final model** [59]
- **What / why:** a new agent trained only to beat the final one.
- **Paper question:** how exploitable is the final agent?
- **When:** end, once the final run has plateaued.
- **Cost:** an RL run.
- **Importance:** medium.

**R4. Single-mode specialists (3 forks)** [60]
- **What / why:** one fork per mode from the final model.
- **Paper question:** does the generalist pay a capacity cost? rl-dm already beat the generalist at 75M.
- **When:** end.
- **Cost:** three RL forks.
- **Importance:** medium.

**R5. Averaging sweep** (no extra training)
- **What / why:** uniform, exponential at several decays, and windowed averages, built from per-update saves of an early arm, rated in one tournament.
- **Paper question:** which weight average should ship, and how much does averaging help?
- **When:** attached to the first baseline run.
- **Cost:** disk plus a tournament.
- **Importance:** medium.

**R6. Elimination-only and frontline-only forks** (optional)
- **What / why:** complete the mode-transfer measurement (deathmatch-only exists).
- **Paper question:** how much does each mode's skill transfer to the others?
- **When:** early, if it should inform the mode weights or A5.
- **Cost:** two RL forks.
- **Importance:** low-medium.

## Settled without an arm (for the record)

| Idea | Settled by |
|---|---|
| 5 epochs, 8 flat learning rate, 9 optimiser defaults, 33 feed-forward width, 34 RMSNorm, 39 presence radius | defaults kept |
| 6 clip and target KL | the early stop never fires; no arm |
| 10 value coefficient, 11 old log-probabilities | diagnostics and cheap checks |
| 12 return-scale decay, 13 unified γ/λ | dropped; unified γ/λ failed on 2026-11-13 |
| 14, 18 BC handoff | fixed at 100M |
| 15 KL to BC, 16 BC temperature, 17 BC data mix | dropped or kept as is |
| 19 RL start weights | exponential average |
| 21–22 reward weights | too large a search; A7 and A12 cover the useful parts |
| 24 deathmatch outcome | win/loss (kill lead) |
| 25–26 separate or centralised critic | too much work for an arm now (deferred) |
| 37 ego-centric frame, 38 CNN map, 45 heads one by one, 46 augmentation | dropped |
| 40 delta head, 41 belief decay | earlier v2 results |
| 47 seat symmetry, 71 bf16 spot check | cheap checks |
| 48 fog | always on |
| 49–53 league | decided structure; cheese agents built |
| 56 lineups | mirrored 50/50 coin flips |
| 58 clocks | game freeze |
| 61–67, 73 evaluation | the evaluation suite |
| 70, 72 process | equal-steps rule; seed isolation built |
| 75 bullet drag | game freeze |

## Counts

- **Early:** A1–A7. A1, A2 and A3 are full arms; A4–A7 are RL forks (A7 is two).
- **Middle:**
  - A8–A12, all RL forks;
  - A13–A22, all full arms.
- **Late:** A23 (a full arm at higher cost) and A24 (about 2–3 RL arms).
- **Reference runs:** R1–R6.

Next: estimate each class's cost on the 5090s, then choose the subset that fits the budget.
