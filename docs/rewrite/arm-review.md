# Arm review (draft for the author)

This lists every candidate arm still open as of 2026-11-15, with an estimated cost for each, prepared for choosing the run budget and the order of arms. The game is frozen as version 21 (`feat/imugi-boundary`).

**Fields:**
- **What / why:** what the arm changes, and the reason to test it.
- **Paper question:** the question a reader would ask that this arm answers.
- **Purpose:** a better agent, a game-design decision, or a scientific claim.
- **Why an arm:** why a cheaper test or theory can't settle it.
- **When:** early, mid or late in the chain, or anywhere.
- **Cost:** the class, an **RL fork** (RL only, from the shared BC) or **BC+RL** (its own BC), then 5090 GPU-hours and dollars at the rate below. It covers the challenger's training, its 5v5 ladder and its decision; the incumbent already exists.
- **Importance:** high, medium or low.

The challenger is normally the more complex or slower option. It wins only by beating the incumbent past the gating threshold on the overall score, without losing any mode by the gate's margin ([evaluation-suite.md](evaluation-suite.md)). Numbers in brackets point to the original list in [ideas.md](ideas.md).

## Cost basis

Measured on the rented 5090 (rl-v20, rl-dm, rl-kl, rl-mix-kl, bc-v19, bc-v20 and bc-v21, game version 20 or older) unless marked. The extraction and every number are in [sections/arm-costs/numbers.md](sections/arm-costs/numbers.md) (`extract.py` beside it).

| Item | Figure | 5090 hours | Source |
|---|---|---|---|
| BC throughput | 21,900 steps/s steady; 20,000 with compile | **1.4 h per 100M** | bc-v20: 84.9M in 1.18 h (`run.json`, segment 1) |
| RL throughput | 23,100–23,700 steps/s steady; 21,100–22,800 with compile | **1.3 h per 100M** | rl-v20: 150M in 1.98 h; rl-dm, rl-kl, rl-mix-kl: 75.5M in 0.92–0.97 h |
| Compile (first update) | 170–330 s with a warm cache, 400–755 s cold | included above | each log's update 1 |
| Laptop 4070 against 5090 | rollout 3.8×, learn 6.8×, wall 5.9× | 3.8 converts games | rl-v20-seed1 against rl-v20 |
| Evaluation per decision | 0.6–0.85 laptop engine-hours | **0.3 h** (0.16–0.22 h, plus the exploit check and margin) | [evaluation-suite.md](evaluation-suite.md#design-pilot) |
| Ladder, per snapshot (5v5, three modes, ±45 Elo) | about 400 laptop-s at ±25–35 Elo | **50 s** | live-Elo logs of rl-v20, rl-dm, rl-kl, rl-mix-kl |
| Ladder, RL fork (26 snapshots to 100M) | | **0.4 h** | RL snapshot schedule ([arms-todo.md](arms-todo.md), 2026-11-14) |
| Ladder, BC+RL (20 BC + 26 RL snapshots) | | **0.65 h** | as above |
| Full ladder (every condition) | about 4.2× the 5v5 ladder | **2.7 h** per BC+RL run | the pilot's seconds per game, conditions 1–13 against 1–3 |
| Rate | **$0.50 per 5090 GPU-hour** | | the author's rough estimate, 2026-11-15 |

**Per class, at 100M BC and 100M RL:**
- **RL fork:** 1.3 h training + 0.4 h ladder + 0.3 h decision = **2.0 h ($1.00).**
- **BC+RL:** 1.4 h BC + 1.3 h RL + 0.65 h ladder + 0.3 h decision = **3.7 h ($1.85).**
- **Rescaling:** each further 100M of RL adds about 1.6 h per run (1.3 h training, 0.3 h for its 20 snapshots).
- **Wall-clock:** one job per GPU, so a run's GPU-hours are its wall hours on one 5090.

**Assumptions to confirm:**
- **RL length per arm: 100M** (provisional, the author, 2026-11-15; to be confirmed from the dry run's curve). A24, A23, R2 and the publication run keep their own lengths.
- **BC at 100M** for every arm (decided 2026-11-14).
- **Throughput was measured before the freeze.** Game version 21, the split critic, the cheese agents in the league and keeping every member may change it; seed isolation didn't (the laptop speed check). The first run's log replaces these figures.
- **Games on the 5090** use the rollout ratio (3.8×) as a proxy; no evaluation was timed on a 5090.
- **The ladder figure is scaled** from ±25–35 Elo to the decided ±40–50 by games ∝ SE⁻² (0.44×). Unscaled, every ladder figure doubles (about 105 s per snapshot).
- **The pilot's costs** came from density maps at every size but 5v5; the re-run on preset maps replaces them. Every break-even game is now 20 a side.
- **BC snapshots** every 5M (20 per BC), rated for every arm with its own BC.
- **Per-step multipliers** marked "estimate" come from `sections/launch/arms.toml` (2026-11-12) and are unmeasured.

## Early: choices that change every later arm

**R0. Baseline (the first proper run)**
- **What / why:** the default configuration on the frozen game: fixed maps, fixed 5v5, the split critic at 1:1, cheese agents in the league, mode mix 6/5/5. Every early arm is read against it.
- **Purpose:** the incumbent, plus the paper's reference curves.
- **Extras:** per-update policy files (on only for this run; they feed R5), the full ladder in every condition, the exploit check (the frontline zone rusher) and the front-reversal re-check.
- **When:** first, beside A1.
- **Cost:** BC+RL; 2.7 h training + 2.7 h full ladder + 0.2 h checks = **5.6 h ($2.80)**. The per-update files take about 2 GB of disk (about 190 updates × 11 MB).
- **Importance:** high.

**A1. Map sizes: fixed against variable** [54]
- **What / why:** train on each mode's fixed map (the default since 2026-11-15) against maps drawn at 0.5–2× scale (opt-in `arena_scale`). rl-v20, trained on fixed maps, collapses in frontline above 1.4×.
- **Paper question:** does the agent generalise across map sizes, and does it need to train on them?
- **Purpose:** game design, plus a generalisation claim. The author's rule: every map stays fixed-size unless this arm trains successfully on variable maps; if it fails, fixed-size full stop.
- **Why an arm:** only training shows whether variable maps are learnable at the same strength. If they aren't, the colour codes and rotary frequencies get re-tuned for fixed maps.
- **When:** first. Its incumbent is R0.
- **Cost:** BC+RL, since BC's data comes from the maps; **3.9 h ($1.95)**. Its decision adds the map-scale conditions (0.5× and 2×), about 0.2 h.
- **Importance:** high.

**A2. Ship and field counts: fixed 5v5 against variable** [55]
- **What / why:** variable team sizes, with absent ships and fields masked (built on `feat/absent-fields`). Deathmatch and elimination go down to 1v1; frontline to 3v3, its frozen minimum.
- **Paper question:** does one model generalise to fleet sizes it never trained on (10v10, 20v20)? This is the paper's main claim.
- **Purpose:** a scientific claim, plus the agent the evaluation suite rewards.
- **Why an arm:** whether variable counts cost strength at 5v5 or improve generalisation can only be measured. rl-v20 is only even with Imugi at 2v2. On fixed maps the counts no longer change the map size; if A1 keeps variable maps, the density rule couples them again.
- **When:** second, after A1.
- **Cost:** BC+RL; **3.7 h ($1.85)**. A benched slot costs about as much as an active one, so counts up to 5 keep the cost per step; counts above 5 would raise it with the slot count (unmeasured).
- **Importance:** high.

**A3. Seed pairs (2 arms × 2 seeds)** [68]
- **What / why:** R0 and A1, each run again at a second seed. With R0 and A1 these are the box's first jobs.
- **Paper question:** are the differences between arms larger than seed noise?
- **Purpose:** process. It sets the sequential test's indifference zone δ (the pilot's 0.02 is provisional), and shows whether seed isolation couples arms as intended. The rl-v20 seed twin differed by 70–150 Elo, most likely seed noise.
- **Why an arm:** noise in RL can only be measured by running.
- **When:** very early, beside R0 and A1.
- **Cost:** two extra BC+RL runs with 5v5 ladders and the seed-against-seed decisions; **7.6 h ($3.80)**.
- **Importance:** high.

**A4. Minibatches: 32 against 256** [3]
- **What / why:** PPO minibatches per epoch, which changes the size of each gradient step.
- **Paper question:** are the PPO settings sound?
- **Purpose:** a better agent and faster training.
- **Why an arm:** the effect on final strength isn't predictable. The author asked for it early, from one BC checkpoint.
- **When:** early.
- **Cost:** RL fork; **2.2 h ($1.10)**. Smaller minibatches may learn slower per step: BC at 256 learns in 44.8 s per update against RL's 38.3 s at 32, though the losses differ too, so up to about +15%.
- **Importance:** medium.

**A5. Training mix: 6/5/5 against 8/4/4** (frontline/deathmatch/elimination)
- **What / why:** the share of games per mode. 6/5/5 stays the default until it reports.
- **Paper question:** how should modes share training when skills transfer between them?
- **Purpose:** a better agent.
- **Why an arm:** the evidence points both ways (deathmatch and elimination share skill; frontline is distinct and still climbing).
- **When:** early, because it affects every later RL run.
- **Cost:** RL fork; **2.0 h ($1.00)**. The mix doesn't change the speed: the deathmatch-only runs matched the all-mode ones.
- **Importance:** medium.

**A6. Self-play: train both sides against one side** (plus more environments, and "latest" as a league member)
- **What / why:** whether each self-play game trains both seats.
- **Paper question:** is training both seats of a self-play game an efficient use of compute?
- **Purpose:** a better agent per unit of compute.
- **Why an arm:** both are standard practice, and the trade-off (correlated rows against about 7% more time) has no theoretical answer.
- **When:** early, since it's part of the training setup.
- **Cost:** RL fork at about 1.07× per step; **2.1 h ($1.05)**.
- **Importance:** medium.

**A7. Offensive premium: zero-sum 1:1, 2:1, and 2:1 annealed to 1:1** (three forks; damage and kill/death only)
- **What / why:** the author's hypothesis that passive play is an equilibrium of zero-sum self-play, and a premium makes it worth searching for the rare winning aggressive line.
- **Paper question:** does a non-zero-sum shaping term help self-play escape stalemates, and does annealing it back recover the true objective?
- **Purpose:** a better agent, plus a scientific claim.
- **Why an arm:** the only evidence is a recollected v1 result in an older game. The split critic, now in every arm, keeps every critic target fixed, so the ratio is only a weight on the advantages.
- **When:** early to mid, since reward design affects later arms.
- **Cost:** two RL forks beyond the 1:1 baseline (R0); **4.0 h ($2.00)**.
- **Importance:** high.

## Middle: training algorithm

**A8. Trust region: clipping against DPPO**
- **What / why:** PPO's ratio clip against DPPO's divergence mask, which is already implemented.
- **Paper question:** does a trust region based on the actual policy divergence beat ratio clipping in a small joint action space?
- **Purpose:** a better agent.
- **Why an arm:** the paper's evidence is from language models with huge vocabularies. Our clip binds in 22–26% of samples in uncertain states, so it could matter. The DPPO divergence check comes first (below).
- **When:** mid, before A9.
- **Cost:** RL fork; **2.0 h ($1.00)**.
- **Importance:** medium.

**A9. Learning rate: ×1 against ×2** [4]
- **What / why:** the learning rate.
- **Paper question:** is the optimiser tuned?
- **Purpose:** a better agent.
- **Why an arm:** the low KL suggests room, but the clip already binds where it matters.
- **When:** mid, after A8. Re-checked at the width arm.
- **Cost:** RL fork; **2.0 h ($1.00)**.
- **Importance:** medium.

**A10. Entropy: today's bonus against a stronger bonus or an entropy floor** [7]
- **What / why:** joint entropy fell to 0.45 late in rl-v20.
- **Paper question:** is exploration adequate late in training?
- **Purpose:** a better agent.
- **Why an arm:** in comparable units, the coefficient is within the normal range, so only training shows whether late training is under-exploring.
- **When:** anywhere. At 100M of RL it sees less of the late phase than rl-v20's 150M did.
- **Cost:** RL fork; **2.0 h ($1.00)**.
- **Importance:** medium-low.

**A11. Steps per update: 2048 against 4096 environments, or a 128-tick against 256-tick rollout** [1, 2]
- **What / why:** how much data each update sees, and, for the rollout length, how far each estimate looks ahead before the critic takes over.
- **Paper question:** are the rollout settings sound? (128 was chosen by feel in v1.)
- **Purpose:** a better agent.
- **Why an arm:** the effects interact with the long-horizon discounts.
- **When:** anywhere. Choose one of the two variants.
- **Cost:** RL fork; **2.0 h ($1.00)** at about the same cost per step. 4096 environments may not fit: the RL runs allocated 19–20 GB of the 31 GB budget at 2048, and the rollout store doubles; a host store would slow it.
- **Importance:** medium-low.

**A12. Dense rewards decayed to zero partway** [23]
- **What / why:** the damage and kill/death rewards fade out over RL, leaving the objective and outcome.
- **Paper question:** are the dense rewards scaffolding, or needed throughout?
- **Purpose:** a better agent, plus a reward-design claim.
- **Why an arm:** with per-level critics it's only a change of weights, so it's cheap, but its effect has to be measured.
- **When:** mid, after A7, which also changes the dense levels.
- **Cost:** RL fork; **2.0 h ($1.00)**.
- **Importance:** medium.

## Middle: model

**A13. No auxiliary heads** [44]
- **What / why:** remove every auxiliary head: next-state, correction, density, enemy action, shot and geometry. This also weakens the belief.
- **Paper question:** do the auxiliary heads, which are a large part of the design, actually help?
- **Purpose:** a scientific claim.
- **Why an arm:** the gradient-share diagnostics show what the heads cost, not what they're worth.
- **When:** mid.
- **Cost:** BC+RL at about 0.8× per step (estimate); **3.1 h ($1.55)**.
- **Importance:** high for the paper.

**A14. Belief input: the next-step heads' outputs for hidden ships against the temporal layers alone** [42]
- **What / why:** whether hidden enemies are fed back as predicted states.
- **Paper question:** does an explicit predicted belief help under fog, beyond what the recurrent layers learn?
- **Purpose:** a scientific claim, and one of the more novel parts.
- **Why an arm:** never tested.
- **When:** mid.
- **Cost:** BC+RL; **3.7 h ($1.85)**.
- **Importance:** medium-high.

**A15. Layout: 2 Yemong blocks against 6 spatial blocks** [30]
- **What / why:** temporal and spatial blocks interleaved, against purely spatial depth.
- **Paper question:** does the temporal processing earn its compute?
- **Purpose:** a scientific claim, plus the architecture choice.
- **Why an arm:** an architecture effect, not predictable.
- **When:** mid.
- **Cost:** BC+RL at about 0.9× per step (estimate); **3.4 h ($1.70)**.
- **Importance:** medium-high.

**A16. Temporal block: Griffin against GRU** [31]
- **What / why:** the recurrent block.
- **Paper question:** is the modern recurrent block better than a classic GRU here?
- **Purpose:** the architecture choice.
- **Why an arm:** not predictable at our scale.
- **When:** mid, after A15.
- **Cost:** BC+RL; **3.7 h ($1.85)**, assuming a similar cost per step.
- **Importance:** medium.

**A17. Input encoding: the colour/HL-Gauss codes against raw scalars** [35]
- **What / why:** how observations enter the model.
- **Paper question:** are the codes, a distinctive part of the design, worth their complexity?
- **Purpose:** a scientific claim.
- **Why an arm:** the earlier Fourier-against-colour comparison (−212 Elo) was on an older model and game, and never against raw inputs.
- **When:** mid, after A1, because map sizes change the codes.
- **Cost:** BC+RL; **3.7 h ($1.85)**.
- **Importance:** medium-high.

**A18. Position encoding: rotary against the colour code against both** [36]
- **What / why:** how attention sees relative positions. Both is today's default.
- **Paper question:** which position scheme does a set-based model need?
- **Purpose:** the architecture choice.
- **Why an arm:** rotary was never ablated; keeping both was decided by review.
- **When:** mid, after A1.
- **Cost:** two BC+RL challengers against the default; **7.4 h ($3.70)**.
- **Importance:** medium.

**A19. Qwen gated attention (elementwise)** [32]
- **What / why:** a sigmoid gate on each head's output, about 1% cost.
- **Paper question:** does output gating help a small, short-context model that we know uses an attention sink?
- **Purpose:** a better agent.
- **Why an arm:** the paper's setting is large language models; our setting differs.
- **When:** anywhere.
- **Cost:** BC+RL; **3.7 h ($1.85)**.
- **Importance:** medium-low.

**A20. Attention heads: 2 against 4** [28]
- **What / why:** 64 against 32 dimensions per head.
- **Paper question:** a minor architecture detail.
- **Purpose:** a better agent.
- **Why an arm:** small, unpredictable effect.
- **When:** anywhere.
- **Cost:** BC+RL; **3.7 h ($1.85)**.
- **Importance:** low.

**A21. Value output: two-hot distributional against a symlog scalar** [27]
- **What / why:** the critic's output form.
- **Paper question:** does the distributional value earn its complexity?
- **Purpose:** simplification, or a claim.
- **Why an arm:** the literature is mixed at small scale.
- **When:** anywhere. The split critic doubles the critic outputs (10 instead of 5).
- **Cost:** BC+RL, since BC trains the critic; **3.7 h ($1.85)**.
- **Importance:** low-medium.

**A22. Shot head: full bullet trajectory against "if I shot now", damage to each enemy and ally** [74]
- **What / why:** which shot prediction to train.
- **Paper question:** which auxiliary target helps aiming?
- **Purpose:** a better agent.
- **Why an arm:** both are plausible; no data.
- **When:** late, after A13 shows whether auxiliary heads matter at all.
- **Cost:** BC+RL; **3.7 h ($1.85)**. The trajectory head took 24% of an update in the 2026-11-02 profile (an older model), so the other head may be cheaper.
- **Importance:** low-medium.

## Late

**A23. Width: 128 against 192** [29]
- **What / why:** model capacity, compared at equal steps and at equal compute, with a learning-rate re-check.
- **Paper question:** is the model capacity-limited, and what does scale buy?
- **Purpose:** a better final agent, plus a scaling claim.
- **Why an arm:** capacity effects can only be measured, and the learning rate interacts with width.
- **When:** last, on the final configuration.
- **Cost:** BC+RL at about 1.8× per step (estimate), in three parts; **11.7 h ($5.85)**:
  - the 192 run at equal steps: 6.0 h;
  - the learning-rate re-check, an RL fork at 192: 3.0 h;
  - equal compute: the 128 incumbent extended to the 192 run's GPU-hours (about 270M of RL), 2.7 h.
- **Importance:** medium-high.

**A24. BC→RL against RL from scratch** [20]
- **What / why:** the same rewards with no BC, at 2–3× the RL steps (200–300M, against the incumbent's 100M BC + 100M RL).
- **Paper question:** is Imugi BC start necessary, and what does it save?
- **Purpose:** a scientific claim.
- **Why an arm:** only training shows it. It may fail to get going without BC.
- **When:** late, or in parallel. It doesn't feed the chain.
- **Cost:** RL only, 200–300M, with 46–66 snapshots rated; **3.6–5.1 h ($1.80–2.55)**.
- **Importance:** high for the paper.

## Reference runs (not pairwise)

R0, the baseline, is listed with the early arms.

**R1. MLP baseline** [43]
- **What / why:** a simple model trained the same way (`--model nearest`).
- **Paper question:** how much does the architecture matter at all?
- **Purpose:** a scientific baseline.
- **When:** anywhere, in parallel.
- **Cost:** BC+RL at about 0.3× per step (estimate); **1.8 h ($0.90)** per seed.
- **Importance:** high for the paper.

**R2. Final model with 2–3 seeds** [69]
- **What / why:** the final configuration repeated at other seeds.
- **Paper question:** is the final result reproducible?
- **When:** end; the author decides then.
- **Cost:** 1–2 seeds. At the publication length (100M BC, 500M RL, 126 snapshots): about **10 h ($5) per seed at width 128, 16.7 h ($8.35) at 192**. At the arm length: 3.7 h or 6.0 h.
- **Importance:** medium.

**R3. Exploiter against the frozen final model** [59]
- **What / why:** a new agent trained only to beat the final one.
- **Paper question:** how exploitable is the final agent?
- **When:** end, once the final run has plateaued.
- **Cost:** an RL run of 100M from BC, and its games against the final; **1.6 h ($0.80)**. More if it must start from scratch.
- **Importance:** medium.

**R4. Single-mode specialists (3 forks)** [60]
- **What / why:** one fork per mode from the final model.
- **Paper question:** does the generalist pay a capacity cost? rl-dm already beat the generalist at 75M.
- **When:** end.
- **Cost:** three RL forks of 100M, each rated in its own mode; **5.3 h ($2.65)**.
- **Importance:** medium.

**R5. Averaging sweep** (no extra training)
- **What / why:** a large suite of averages (uniform, exponential at many decays, windowed), built after training from R0's per-update policy files and rated in one tournament. The variants are chosen then (the author, 2026-11-15).
- **Paper question:** which weight average should ship, and how much does averaging help?
- **When:** after R0.
- **Cost:** about 2 GB of disk, the averages built on the CPU, and the tournament: **0.5–2 h ($0.25–1.00)** for 20–30 variants, more if close variants need errors tighter than ±45 Elo.
- **Importance:** medium.

**R6. Elimination-only and frontline-only forks** (optional)
- **What / why:** complete the mode-transfer measurement (deathmatch-only exists).
- **Paper question:** how much does each mode's skill transfer to the others?
- **When:** early, if it should inform the mode weights or A5.
- **Cost:** two RL forks of 100M; **3.5 h ($1.75)**.
- **Importance:** low-medium.

## Checks and tasks

Before or beside the first runs; none needs an arm. Hours are 5090 hours.

| Check or task | What it takes | Cost |
|---|---|---|
| Seat symmetry (47) | a probe on the trunk's features, and a mirror match (a checkpoint against itself, by seat), about 1,000 games per mode at 5v5 | 0.2 h |
| Old log-probabilities (11) and DPPO divergence | one compile and a few logged updates from a checkpoint; both from the same run | 0.2 h |
| bf16 spot check (71) | a few updates in both precisions | 0.1 h |
| Design pilot re-run on the preset maps | rl-v20 against rl-mix-kl at 75M, 512 games per condition, every break-even scored (about 1 laptop engine-hour), then `bnb evaluate design` on the CPU | 0.3 h |
| Freeze measurements (`check/freeze-final`) | Imugi only, on the CPU: the shield check, deathmatch respawn, the modes and the Yongma–Jujak balance | no GPU |
| Seed-streams speed check on the 5090 | 4 updates on the old and the new tree (passed on the laptop) | 0.3 h |
| Rating layout probe on the 5090 | the laptop's blocks-per-batch probe, repeated on the box | 0.3 h |
| Box setup and compile cache | one cold compile fill per GPU model, exported to later boxes | 0.5 h |
| First jobs: 2 arms × 2 seeds | R0, A1 and A3 together | 17.1 h, in those entries |
| Exploit check and front reversal | the exploit check on every final checkpoint (in each decision's 0.3 h); front reversal on R0 | in the entries |
| Tie rule | per tie: extend both arms by 10–20M and rate their last 5 snapshots | about 0.5 h per tie |

Together, without the ties: **about 2 h ($1)**.

## Settled without an arm (for the record)

| Idea | Settled by |
|---|---|
| 5 epochs, 8 flat learning rate, 9 optimiser defaults, 33 feed-forward width, 34 RMSNorm, 39 presence radius | defaults kept |
| 6 clip and target KL | the early stop never fires; no arm |
| 10 value coefficient, 11 old log-probabilities | diagnostics and cheap checks |
| 12 return-scale decay, 13 unified γ/λ | dropped; unified γ/λ failed on 2026-11-13 |
| 14, 18 BC handoff | fixed at 100M; the plateau stop only logs |
| 15 KL to BC, 16 BC temperature, 17 BC data mix | dropped or kept as is |
| 19 RL start weights | exponential average |
| 21–22 reward weights | too large a search; A7 and A12 cover the useful parts |
| 24 deathmatch outcome | win/loss by kill lead |
| 25–26 separate or centralised critic | too much work for an arm now (deferred). The split offence/defence critic is in every arm. |
| 37 ego-centric frame, 38 CNN map, 45 heads one by one, 46 augmentation | dropped |
| 40 delta head, 41 belief decay | earlier v2 results |
| 47 seat symmetry, 71 bf16 spot check | cheap checks (above) |
| 48 fog | always on |
| 49–53 league | decided structure: self-play 50%, (1−p)² over every member, rating lanes; every member kept (no pruning); members every 5M to 50M, then the interval doubles with the steps; cheese agents in the league |
| 56 lineups | mirrored 50/50 coin flips |
| 57 mode mix | 6/5/5 by default; A5 tests 8/4/4 |
| 58 clocks | game frozen as version 21: deathmatch 4·N^(4/3) tickets with 10 minutes; elimination a linear 4-minute shrink from 0 s with a 248 s clock; frontline ±2 with 10 minutes and a 3v3 minimum |
| 61–67, 73 evaluation | the evaluation suite: win-probability scoring in thirds, a group-sequential test (K = 4, δ = 0.02 until A3), the no-regression gate; break-even with the weaker side fixed at 20, scored in every mode; preset maps |
| 63 averages | R5, chosen after training |
| 70, 72 process | equal-steps rule; seed isolation built and merged |
| 75 bullet drag | no change (freeze, 2026-11-15) |

## Counts and total cost

- **Early:** R0 and A1–A7. R0, A1, A2 and A3 are full runs (A3 is two); A4–A7 are RL forks (A7 is two). A1–A7: **25.5 h**; R0: 5.6 h.
- **Middle:**
  - A8–A12, all RL forks: **10.0 h**;
  - A13–A22, all BC+RL (A18 is two): **39.8 h**.
- **Late:** A23 (three parts, 11.7 h) and A24 (5.1 h at 3×): **16.8 h**.
- **Reference runs:** R0–R6, **39.8 h**, with R2 at two seeds of the 500M length at width 128.

| Set | 5090 hours | Dollars at $0.50 |
|---|---|---|
| Every arm (A1–A24) | 92 | 46 |
| Reference runs (R0–R6) | 40 | 20 |
| Checks and tasks | 2 | 1 |
| Ties (about a third of 30 decisions) | 5 | 2.50 |
| **Total** | **139** | **69.50** |
| Plus the 500M publication run (width 128 or 192; its full ladder about 7 h more) | 10–17 | 5–8.50 |
| **Minimum set:** R0, A1, A2, A3, A7, A13, A24, R1, the checks and ties | 38 | 19 |
| Minimum set plus the medium-high arms (A14, A15, A17, A23) | 60 | 30 |

On one 5090 the total is about 5.8 days end to end; on 8 GPUs the chain's order, not the sum, sets the wall time. Each further 100M of RL per run adds about 1.6 h.
