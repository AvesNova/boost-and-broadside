# Design history: learner side (rewards, returns, value targets, losses, PPO, BC, league)

Scope: inventory sections 7-11, the heads of section 5 as training targets, and the
learner-side rows of section 12. This reports what the record says. It does not judge the
choices.

## How to read this

- **Paths.** `D` = `docs/rewrite/decisions.md`; `YR` = `docs/rewrite/yemong-review.md`;
  `S/` = `docs/rewrite/sections/`; `SIMP` = `docs/rewrite/simplification.md`;
  `v1:` = a file on `origin/main` (read with `git show origin/main:<path>`).
- **Evidence labels.** MEASURED, CITED, ARGUED, INHERITED, NOT FOUND, as in the brief.
- **Three code generations, not two.** This matters for every "inherited" label below.
  1. **v1-main** (`origin/main`, code frozen 2026-09-07 at `d017f0b`; runs 682, 719, 731).
     Scalar value heads.
  2. **v1-branch** (`feat/counterfactual-shot-labels`, then `feat/resim-shot-labels`,
     2026-09 to 2026-10-30; runs 737-764). Categorical everything. **Not in this clone**:
     known only through `YR` (which reviewed it) and `D:1188-1242` (which summarised it).
  3. **v2** (this branch, from 2026-11-01). When v2's notes say "v1's coefficients" or
     "v1's math" they mean generation 2, not `origin/main`.
- **Dating limits.** This clone is shallow: history starts at `3b9adbd` (2026-11-15), so
  `git blame` dates nothing earlier. Dates before that come from the notes.
- **Seed noise.** "About 110 Elo" is from BC-only arms at 30M steps (item 6 below).

---

## 1. Answer to the owner's question

**Short answer.** The record does not support the premise as stated. The loss weights
(policy 27, value 0.007, auxiliaries 0.0033-0.11) were not built to bridge scalar and
cross-entropy losses. They were solved on the v1-branch **after** every output was already
categorical and in nats. The machinery that did exist to balance scalar outputs lived on
v1-main and was removed at the categorical switch. What survives today is a second layer,
added to make returns fit a **fixed categorical support**, plus a few carry-overs whose
reasons have since changed.

### Timeline

| When | State | Source |
|---|---|---|
| to 2026-09-07 (v1-main, run 731) | Value = scalar regression, one output per reward component (14, then 16). Next state = regression on 10 continuous deltas. | inventory "v1 equivalents"; `v1:docs/training.md:155-202` |
| same | Balancing machinery of that era: per-component `ReturnScaler` in symlog reward space; Huber value loss (delta 1.0); per-component unit-RMS advantages, then the aggregate divided by its own RMS; two floors; a lambda aggregation matrix; tier scales; a shaping taper; rewards divided by total ship count; scheduled coefficient tables (policy 1.0, value 1.0, entropy 0.005, next-state 0.2, windowed 0.1). | `v1:docs/training.md:176-202, 310-349`; `v1:src/boost_and_broadside/train/rl/buffer.py:220-262` |
| 2026-09 to 2026-10 (v1-branch) | Categorical critic (4 per-ship levels x 51 symlog bins, two-hot), 4-class outcome head, next-state codes, Poisson density, shot heads. One DreamerV3 return scale replaces the per-component scalers. Coefficients still value 1.0, entropy 3e-4, next-state 0.2, enemy action 0.1, density 1.0, shots 0.1. | `YR:273, 665-711, 1789-1796` |
| same | Known defect of that state: "the gradient clip fires on every step ... norms around 100x the limit", so the clip was a per-step normaliser and the coefficients' absolute scale was "nearly irrelevant". | `YR:1816-1821` (run 748) |
| 2026-10-06 to 10-30 (`feat/resim-shot-labels`) | "Loss units: every term in nats per token ... A weight is an exchange rate between nats and does not move with fleet size. Weights are then set against measured trunk-gradient shares, not by hand." Policy 27, value 0.007, etc. `max_grad_norm 1.0 as a spike guard only`. | `D:1199-1227` |
| 2026-11-01 (v2 port) | "No coefficient calibration here ... v1's coefficients are the starting values." | `S/learner.md:22-25`; `S/bc.md:14-16` |
| 2026-11-07/08 | Next-state head changes from absolute codes to delta codes. Coefficient re-matched 0.0041 -> 0.0033. Marginal bias tables added. | `docs/explanation/learner.md:231-248`; `S/delta-heads-brief.md:41-42` |
| 2026-11-11 | "Return units": every reward weight x10; value support 51 bins to +-2 -> 77 bins to +-3. | `S/compile-overhaul-notes.md:640-664` |
| 2026-11-21 | Audit lists reward scaling, the floor, the support alert and the bias tables under "R6. Categorical outputs over fixed, very wide ranges". | `SIMP:537-544, 618` |

### Where the loss-weight numbers came from

- **Derived, on the v1-branch, against target trunk-gradient shares.** Solved from pilots
  754, 755, 759; shares measured on run 760: bc 0.36, next_state 0.25, enemy_action 0.16,
  density 0.12, shot heads 0.06 / 0.05, value 0.009 (`D:1212-1216`).
- **Policy 27**: "about 0.36-0.40 of the trunk; 14 and 60 bracket it" (`D:1221-1222`).
- **Value 0.007**: deliberately tiny. "Solving value up to a 0.30 share was rejected (run
  756: live Elo 530 vs 913, worse cloning)" (`D:1216-1217`).
- **Enemy action 0.11**: "halving it, run 763, did worse" (`D:1223-1224`).
- **Entropy -0.0081**: "3e-4 of the policy weight (DreamerV3)" = 27 x 3e-4 (`D:1222-1223`).
- **Next state 0.0033, correction 0.0033, geometry 0.007**: set in v2, by matching trunk
  gradient norms on short CPU runs (rows 8.5, 8.6, 8.12 below).
- **Why the numbers look odd.** A next-state row's cross-entropy is about 30 nats
  (`docs/explanation/yemong.md:210-211, 325-326`); a policy token's is under ln 42 = 3.7.
  The weights are exchange rates between heads with very different nats per token, not
  between scalars and nats. Their absolute level (27, not 1) follows from making the
  gradient clip a spike guard instead of a normaliser.
- **None of the v1-branch pilots (754-764) is in this clone.** The numbers reach v2 only
  through `D:1188-1242`.

### Per piece of balancing machinery

| Piece | Built for | Re-examined after the categorical switch? | Still needed, per the record |
|---|---|---|---|
| Per-component return scalers, Huber loss, per-component advantage RMS, tier scales, lambda matrix (v1-main) | Scalar heads with different natural scales (`v1:docs/training.md:176-178`) | Removed at the switch | Gone |
| Loss weights 27 / 0.007 / 0.0033-0.11 | Trunk-gradient shares between categorical heads | Set after the switch. In v2: next-state, correction and geometry re-measured; the rest never (`S/learner.md:22-25`; `arm-review.md:300` lists "10 value coefficient" as "diagnostics and cheap checks") | The method stands. The values belong to the v1-branch head set and game. Most heads they balance are being removed |
| Return scale (5th-95th percentile EMA, floor 1.0) | Advantage units for the policy term; DreamerV3's normaliser (`YR:1792-1793`) | Kept through the port unchanged. Decay and floor dropped as an arm (`arm-review.md:301`) | Audit: "Goes: normalise advantages per batch and predict normalised returns" (`SIMP:541`). An independent review objects that per-batch advantage normalisation "does not replace return scaling for the value loss" (`review.md:387-390`) |
| Reward weights x10 | Returns were a tenth of the critic's bin units and the scale sat at 1.12-1.22, "just above its floor of 1" (`S/compile-overhaul-notes.md:644-650`) | This **is** post-switch machinery; it exists because the support and floor are fixed | Goes with a fixed support and floor (`SIMP:541`) |
| Value support 77 bins to +-3 | Keep bin spacing after x10 | Same | Open: "Value: what the simplified form is" (`SIMP:136`) |
| Symlog bins on the value | DreamerV3 two-hot critic (`YR:392-398`) | A21 (two-hot against symlog scalar) never ran (`arm-review.md:186-192`) | Open |
| Outcome as a separate 4-class level | Discounted outcome with an "unresolved" class; run 737 (`YR:691-695`) | Decided 2026-11-21: merged into value (`SIMP:126`) | Going |
| Per-level reward structure (5 levels, own gamma/lambda) | v1-main: per-component critics so different scales need not share a head. Today: per-level horizons | Tested once without (2026-11-13): the run stopped fighting | Record says still needed; the test was confounded (item 1 below) |
| Marginal bias tables (next-state, error heads) | A 500-logit categorical head starts about 14 nats above its marginal entropy without them (`S/delta-heads-brief.md:41-42`) | Post-switch; exists because the head is categorical | "Probably" goes (`SIMP:543`) |
| Zero entropy at warm start | The fixed -0.0081 outweighed the policy gradient at the clone's advantage scale | Listed by the audit under R6 (`SIMP:541`) | Tied to the return scale's units (item 2 below) |
| Gradient decomposition on a cadence | Setting weights from shares | "Mostly" goes, "mostly for heads being dropped" (`SIMP:586`) | Partly |
| Global grad clip 1.0 | v1-branch: de facto normaliser | Re-purposed 2026-10-30 as a spike guard; v2 clip fire rate 0 (`SIMP:599-600`) | Guard only |

**Two things the owner should know that the question does not ask.**

- **Some machinery was added because of the categorical switch, not despite it.** The x10
  reward factor, the wider support, the support-clipping alert, the bias tables, the
  grouped cross-entropy and the HL-Gauss decodes all exist to serve fixed-range categorical
  outputs. The audit's own root-cause ranking says so (`SIMP:618, 685-692`).
- **v1-main already recorded what a floor of 1.0 on a percentile span does.** Under the
  p5/p95 estimator "a floor of 1.0 ... bound 8 of 12 components on every update of run 719,
  suppressing `field_death`'s critic gradient by four orders of magnitude"
  (`v1:src/boost_and_broadside/train/rl/buffer.py:243-250`). v1-main then moved to a
  standard deviation with an epsilon floor. The v1-branch went back to p5/p95 with floor 1
  (on the summed return, one scalar) when it adopted the DreamerV3 form. No note records
  that the earlier finding was weighed.

---

## 2. The six specific items

### Item 1. The run that "learned to stop fighting"

- **What ran.** `rl-check`, 2026-11-13 17:40, on a 16 GB box: PPO warm-started from
  `bc-mb256`'s 100M BC exponential average; 1,024 environments; 32 minibatches; 256 lanes;
  60 updates (about 30M steps) (`S/overnight-calibration/calib.toml:78-88`).
- **Reward and discount design.** The five weighted levels, unchanged. One discount and one
  trace decay for all five: gamma 0.9999, lambda 0.99, in BC and RL alike, "the author,
  2026-11-12" (`calib.toml:10-16`). No entropy bonus (`diagnostics.md:242-243`).
- **What happened.** "stopped fighting within about 60 updates: team deathmatch hits per
  game 2,600 to 20, combat deaths 33 to 0, while Frontline kept capturing zones"
  (`diagnostics.md:243-245`).
- **What changed.** The code defaults went to per-level values: the 2026-11-09 champion's
  with the two combat levels squared for the x2 game (`src/bnb/learn/ppo.py:139-146`). A
  combat-collapse alert was added (`diagnostics.md:241-254`). Unified gamma/lambda was
  "dropped; unified gamma/lambda failed on 2026-11-13" (`arm-review.md:301`).
- **Evidence that per-level discounts fixed it.** One sentence: rl-v20 (2026-11-14,
  per-level) saw "deathmatch hits per game went from 417 to 927 ... the opposite of the
  failed old-game RL check" (`S/bc-v19/overnight-log.md:172`).
- **This is not a controlled comparison.** Between the two runs the game changed ("old-game";
  bc-v19 and v20 are a later game version), the BC clone changed, the environment count
  doubled, and the critic the check started from had been trained at 0.9999 / 0.99.
- **No analysis of the mechanism is recorded.** The notes give no advantage shares, no
  per-level returns and no reward trace for `rl-check`. The per-level advantage series were
  added the same day because "nothing showed which level the summed advantage follows"
  (`diagnostics.md:478-480`).
- **A different single-value design was never tested.** The `single-value` arm (one summed
  level, gamma 0.995, lambda 0.95) is in `S/launch/arms.toml:66-81` and never ran. The
  failed test was five heads at one extreme discount, not one head at a moderate one.
- **Still open in the record.** `core-brief.md:326-327` lists it as an open question;
  `review.md:84-86, 391-392` notes the plan's register and its open questions disagree.
  `SIMP:583` says "Fix the cause; the monitor can stay".

### Item 2. Entropy coefficient 0.0 after the BC warm start

- **Deliberate.** `WARM_START_ENTROPY = 0.0`, applied to every factor not set on the command
  line (`src/bnb/learn/ppo.py:207-212`; `src/bnb/cli.py:931-941`).
- **Stated reason.** "at its advantage scale the bonus outweighs the policy gradient on rare
  actions: from the first clone, v1's -0.0081 drove the shoot share from 1.7% to 6.8% and
  the score against Imugi from 0.23 to 0.03 in 24 updates, while 0 held it (0.25)."
- **Evidence.** MEASURED: one ablation pair on the first clone (2026-11-04,
  `fix/rl-degradation`). The branch's own notes are not in the repository.
- **Author's later confirmation.** 2026-11-06: "No entropy bonus or KL toward the clone in
  RL: 'the clone is just an Imugi and not a diverse pool of human experiences. We do need
  some way to control the kl though' (how: open)" (`D:1320-1322`).
- **Tension inside the record.**
  - The pre-fix metrics sweep said "the entropy bonus's gradient is about zero; the policy
    term dominates" (`S/rl-degradation-brief.md:24`). The fix then blamed the bonus.
  - The same investigation found the batch was a quarter of v1's and advantages were cut at
    every shard (`S/rollout-lanes-brief.md:3-5`). Those were fixed at the same time. The
    entropy result was not re-measured afterwards.
  - The comment's supporting clause, "the BC clone already samples every action (its
    labels' uniform floor)", is stale: the label floor is now 0.0
    (`src/bnb/learn/bc.py:75-81`), with Imugi's own soft gates giving the support.
  - `arm-review.md:97-103` (A10) describes "today's bonus" as "within the normal range".
    The RL stage runs with none.
- **Does the problem still exist?** PARTLY. The mismatch is between a fixed coefficient and
  an advantage whose units come from the return scale. Since 2026-11-11 the returns are 10x
  larger and the scale with them, so the measurement predates today's units.
- **Known consequence.** Joint entropy fell 1.44 -> 0.66 by 150M and 0.4 by 172M in rl-v20
  (`overnight-log.md:163, 215, 260`). Idea 7 / A10 never ran.

### Item 3. The trust region, and DAPO / DPPO

- **Rule today.** Symmetric clip 0.2 plus `target_kl` 0.02 early stop.
- **Origin.** v1-main: clip 0.15, `target_kl` 0.1 tightened to 0.02 above an 80% win rate.
  The win-rate gate went with all win-rate gates (`YR:1808-1811`). The v1-branch then chose
  "clip 0.2 and target_kl 0.02 early stopping, one value for the whole run" (`D:1224-1226`).
- **Evidence.** One pilot: "Pilot 762: 0.43 -> 0.89 score against scripted, live Elo ~1,000
  -> ~1,300 in 20M steps" (`D:1226-1227`). No comparison against another clip value is
  recorded. INHERITED into v2.
- **How it behaves in v2.** KL about 0.007 against 0.02; "the early stop never fires"
  (`arm-review.md:299`). Clip fraction 0.06-0.08 overall, but "our clip binds in 22-26% of
  samples in uncertain states" (`arm-review.md:87`). In the 2026-11-04 collapse the KL "sits
  at its 0.02 target every update, and 62% of updates stop their epochs early"
  (`S/rl-degradation-brief.md:21-22`), so it fired and did not prevent the collapse.
- **DAPO (`clip_higher`, upper margin 0.28) and DPPO (`dppo_tv`, `dppo_binary_tv`, delta
  0.15).** Implemented 2026-11-06 with tests (`S/rl-round-five-notes.md:89-110`). **Never
  run in training.** Decided only "to try in the RL fine-tunes" (`D:1347-1349`).
  - The cheap pre-check (total variation under today's clip) is unticked
    (`arms-todo.md:212`). Arm A8 is unrun (`arm-review.md:83-90`).
  - The notes' own caveats: the paper's evidence is from language models; "much of the
    motivation is training-inference mismatch ... which a single engine like ours barely
    has"; DAPO "treats the symptom"; no cap on r for unmasked rare actions was built
    (`S/dppo-notes.md:26-29, 42-47`; `rl-round-five-notes.md:110`).
  - `references.md:20` marks DAPO unverified; DPPO's arXiv id and authors verified.
- **Status.** Non-goal as switches: "alternatives (trust regions, model variants) live on
  branches and in arms" (`SIMP:83-84`); "Four trust regions ... Keep one" (`SIMP:584`).

### Item 4. The return scale, and the un-normalised advantage

- **Origin.** DreamerV3's return normaliser, adopted on the v1-branch with the categorical
  critic: "EMA of the 5th-95th percentile spread, decay 0.99, floor one win, no mean
  subtraction, no per-minibatch standardisation" (`YR:1789-1793`; also `YR:361-366,
  707-711`). CITED; `references.md:122` marks DreamerV3 unverified.
- **Floor 1.0** meant "one win" when the outcome weight was 1. The outcome weight is now 10
  and the floor is still 1.0. No note revisits the floor's meaning.
- **No bias correction.** Specified that way: "both start at 0" (`S/yemong-spec.md:486-489`).
  Recorded as a suspected defect: "early updates sit on the floor (`scale_floored 1` in the
  3090 logs)" (`SIMP:597-598`).
  - Arithmetic, mine and not in the record: at decay 0.99 per update, 95 BC updates (100M
    steps) bring the EMA to about 61% of the true gap. The scale handed to RL is still
    rising, so the effective advantage size drifts down through early RL.
- **No mean or std normalisation.** Reason recorded only as "the DreamerV3 normaliser
  divides, it does not clip" (`diagnostics.md:462-463`) and as the property that "a factor
  common to every reward weight leaves the policy's gradient as it was"
  (`docs/explanation/learner.md:129-131`). The review called it "non-standard but
  principled" (`YR:394-397`). ARGUED and CITED; never compared with per-batch normalisation.
- **Entropy pairing.** The fixed entropy coefficient was kept because "it is the tested
  DreamerV3 pairing with this normaliser" (`YR:1931-1933`).
- **Later doubts.** Idea 12 (decay and floor) "dropped" without a test (`arm-review.md:301`).
  The audit wants it gone (`SIMP:541`). The independent review wants a standard return
  normaliser kept for the value loss (`review.md:387-390`).

### Item 5. Why BC stops on budget

- **History of the stop rule.**
  - 2026-11-01: first of score 0.45, KL plateau, budget (`S/bc.md:35-39`).
  - 2026-11-02: score target 0.40 with a confidence bound, after the first rule fired on a
    lucky window (`S/bc.md:60-74`).
  - 2026-11-04: budget default 50M, "v1's time to parity" (`S/rollout-lanes-brief.md:60-62`).
  - 2026-11-09/10: plateau rule fitted offline to four BC runs; "The 5% was chosen after
    looking at these arms" (`docs/explanation/behaviour-cloning.md:161-165`).
  - 2026-11-12: capped at 100M "so that a BC stage whose plateau never fires still leaves RL
    50M" (`S/launch/launch.toml:40-45`).
  - 2026-11-14: "BC is fixed at 100M for every arm. The plateau stop only logs"
    (`arms-todo.md:32`; commit `85ffcd7`).
- **Stated reasons for the last step.**
  1. "so arms compare at equal BC training" (`behaviour-cloning.md:152-153`), and one shared
     BC serves every arm (`arms-todo.md:56`).
  2. The plateau does not track strength: bc-v20 stopped by plateau at 84.9M while "Elo was
     still rising at 70M ... The KL rule doesn't track strength, as the old-game handoff
     curve also showed" (`overnight-log.md:104-113`).
- **Evidence.** MEASURED, one run (bc-v20). The plateau rule itself fires 11-13M steps after
  the offline reading (`behaviour-cloning.md:183-185`).
- **Still exists?** The need for a stop exists. The plateau and score rules do not stop
  anything: "only the budget fires; about 240 lines ... Goes" (`SIMP:582`).
- **Never tested.** Ideas 14 and 18 (hand-off at 50M against plateau; plateau against
  budget) were "fixed at 100M" without an arm (`arm-review.md:302`).

### Item 6. Seed-to-seed variation: every recorded pair

| # | Pair | Stage, steps | Where | Gap | Source |
|---|---|---|---|---|---|
| 1 | C / C-seed1 | BC, 30M | laptop | "about 30 Elo in frontline (... played the same)"; outcome Brier 0.52 against 0.28 | `S/round2-plan.md:16-18`; `S/gpu-box-log.md:387` |
| 2 | absolute / absolute-seed1 | BC, 30M | 3090, tournament 3 | 657 against 532 mean = 125 (frontline 672 / 569, deathmatch 642 / 496) | `gpu-box-log.md:440-444` |
| 3 | same, re-rated | BC, 30M | tournament 4 | 657 against 545 = **112** | `gpu-box-log.md:588-591` |
| 4 | hiddenphysics / -seed1 | BC, 30M | tournament 4 | 811 against 696 = **115** | same |
| 5 | deadreckon-split / -seed1 | BC, 30M | tournament 4 | 796 against 686 = **110** | same |
| 6 | rl-v20 / rl-v20-seed1 | RL, 50-150M | 5090 against laptop 4070 | twin weaker by "roughly -70 to -150 Elo" from head-to-head scores | `arms-todo.md:108` |

- **Was the seed effect common across arms? Yes, on the record, for pairs 3-5.** "Seed 0 is
  ~110 above seed 1 for all three configs that have both (112, 115, 110): systematic (the
  seed sets the init and data order every seed-0 arm shares), so same-seed comparisons are
  paired and far tighter than 110; never compare across seeds" (`gpu-box-log.md:591-594`).
  Seed 1 was also worse than seed 0 on every training metric in all three arms
  (`gpu-box-log.md:526`). This matches the owner's recollection.
- **The paired difference held across seeds.** Hidden physics against its base: mean +15 at
  seed 0 and +10 at seed 1; frontline +101 / +53; deathmatch -73 / -34
  (`gpu-box-log.md:594-595`).
- **Limits of this evidence.**
  - Two seeds only (0 and 1). A common shift between two seeds is one draw of a seed effect.
  - All three arms are variants of one model, BC only, 30M steps, one box.
  - Frontline-only seed gaps in tournament 4 were 54-102 (`gpu-box-log.md:566`).
  - Pair 6 is the only RL pair and is confounded: micro-batch rows (96 against 480), host
    store, GPU, commit. The micro-batch check cleared the update itself, so the gap was
    judged "most likely seed noise" (`arms-todo.md:109-112`;
    `S/micro-batch-check/results.md:9-12`).
  - Arm A3 (2 arms x 2 seeds, clean) never ran (`arm-review.md:38-45`).
- **Since then.** Seed isolation was built so arms share random streams
  (`arms-todo.md:25, 104`). Whether it couples arms as intended is what A3 was to show.
- **Use in decisions.** "+27 Elo mean, under the +40/+50 bar; seed noise about 110 Elo" for
  the error head (`SIMP:289-291`). The 60M champion call (+64 over the null) is "seed 0
  only" (`gpu-box-log.md:611-613`), and by 120M BC the two were level (+10,
  `gpu-box-log.md:642`).

---

## 3. Row by row

Each row: **Added** / **Reason** / **Evidence** / **Still exists?** / **Doubts** / **Size**.
"Size" is omitted where nothing is recorded and nothing in the row depends on team size.

### Section 5 heads, as training targets

**5.2 Value head, per-ship levels, 77 symlog bins (with 7.11, 8.3)**
- Added: categorical critic on the v1-branch (51 bins, +-5 symlog at review time, then +-2:
  "the largest measured return is 4.55", `D:1210-1211`). 77 bins to +-3 on 2026-11-11.
- Reason: DreamerV3 two-hot critic; per-level so each level has its own horizon.
- Evidence: CITED (unverified). The move to categorical was not A/B tested against v1-main's
  scalar heads in any note here.
- Record against code: the review **decided** HL-Gauss (sigma 0.75 bins) with calibrated bin
  values and 2 padding bins for the critic (`YR:656-658, 630-633`; `references.md:124`). The
  code is plain two-hot with no padding (`src/bnb/models/yemong/critic.py:28-35`), because
  the port took "v1, exact" (`S/yemong-spec.md:471-480`). Never implemented.
- Still exists? PARTLY. Value "to be simplified" (`SIMP:126, 136`).
- Doubts: A21 never ran. Explained variance against lambda-returns "measures the critic's
  agreement with itself" (`S/compile-overhaul-notes.md:661-664`).
- Size: fixed support +-19 in reward units; per-ship returns that grow with fleet size meet
  it (inventory 12.16). No measurement at other sizes is recorded. The support-clip alert
  never fired at 5v5 (`compile-overhaul-notes.md:642-644`).

**5.3 Outcome head (with 7.6, 8.4)**
- Added: v1-branch. Targets are categorical lambda-returns with mass leaking to "unresolved".
- Reason: "The outcome gamma is not 1 on purpose ... at gamma 1 errors compounded (run 737:
  win explained variance fell from 0.994 to 0.42)" (`YR:691-695`).
- Evidence: MEASURED, one v1-branch run.
- Still exists? GONE by decision ("Outcome merges into value", `SIMP:126`). The bootstrap
  chain behind it still exists unless rollouts lengthen (`YR:776-782`).
- Doubts: its critic had "~0 trunk share in every arm" and falling Brier skill
  (`S/round2-log.md:129-135`); "removing the outcome level from the advantages until its
  critic works" was discussed, not decided (`D:1333-1334`). Yet outcome is the largest part
  of the advantage (share 0.44-0.49, `overnight-log.md:163, 189`). Calibration "only sees
  the last seconds of each game" (`diagnostics.md:492-496`).
- Size: one value per view added to every own ship; not divided by team size (12.14).

**5.4 / 8.5 Next-state (physics) head and loss, 0.0033**
- Added: absolute state codes on the v1-branch (0.0041, 22 groups). Delta codes 2026-11-07;
  extrapolation baseline and split 2026-11-08/09; default since 2026-11-12.
- Reason for the weight: "Nats per token change with the code, so it is set to match the
  next-state term's trunk gradient in the absolute arm" (`docs/explanation/learner.md:234-236`).
- Evidence: MEASURED on a short CPU run, seed 1 (trunk norm 0.0177 against 0.0143, ratio
  1.24; 0.0041 / 1.23 = 0.0033). Its own caveat: "The shares at this scale are noisy".
- Evidence for the head: deadreckon-split-hiddenphysics +64 over absolute at 60M BC, one
  seed; level at 120M (+10) (`gpu-box-log.md:608-613, 642`). In RL the champion led by +106
  to +190 at 121.6M, one seed each (`gpu-box-log.md:676-677`).
- Still exists? PARTLY. A next-state head is kept "simplest form"; "a plain regression on
  next-tick truth may do" (`SIMP:127, 137-138`). Its rollout use (the belief) is removed.
- Doubts: "The next-state code's bounds may predate the game the model trains on"
  (`SIMP:592-596`).

**5.5 / 8.6 Error (correction) head and loss, 0.0033**
- Added 2026-11-08/09 with `delta_split`. Coefficient equal per row to the physics head's;
  measured per-row trunk gradient 0.78x and 1.32x, "inside the 1.5x band"
  (`learner.md:250-262`).
- Still exists? GONE (decided 2026-11-21, follows from no belief).

**5.6 / 8.7 Enemy-action head and loss, 0.11**
- Added: v1-branch. Evidence for the weight: run 763 (halving did worse), single run.
- Still exists? UNKNOWN: "keep or drop ... its only rollout use was the belief" (`SIMP:134-135`).
- Size: mean over living enemy rows; scales with N by construction of the mean.

**5.7 / 8.8 Density head and loss, 0.021**
- Added: v1-branch; the review settled the form (HL-Gauss targets, Poisson head, `D:1525-1526`).
- Evidence for the weight: INHERITED from run 760's shares. In v2's bc-v19 its trunk norm was
  0.076-0.092 against about 0.02 for next-state and geometry (`overnight-log.md:24-25`); no
  action recorded.
- Still exists? Probably GONE ("maybe", `SIMP:125`).
- Size: targets are counts per cell, so rates scale with fleet size; the loss divides by own
  valued ships though cells count both sides (12.10, 12.11). "density its cells per living
  ship" was the design intent for size independence (`D:1200-1201`).

**5.8 / 8.10 Shot-outcome head and loss, 0.019**
- Added: v1-branch. "Shot-outcome gradient is mostly sampling noise" (`D:1235-1236`).
- Still exists? Kept (`SIMP:127`); labels may move to real shots (`SIMP:152`).

**5.9 / 8.9 / 8.11 Shot-trajectory head, loss 0.007, re-simulation**
- Added: v1-branch (0.0035, two ages summed); v2 averages 4 ages at 0.007
  (`S/learner-spec.md:87`). Re-simulation decided 2026-10-22 (`D:1436-1452`).
- Still exists? GONE (decided 2026-11-21).

**5.10 / 8.12 Geometry head and loss, 0.007**
- Added 2026-11-12, replacing the rotary read-out (2026-11-07 to 11-12). Weight "set in
  commit e334d032 ... so that its trunk gradient norm sits near 0.02" (`learner.md:207-212`).
- Evidence: MEASURED in bc-v19 and bc-v20 (0.0205-0.025). A CPU run ranked the five outputs
  in the opposite order to production: "The CPU run was misleading"
  (`overnight-log.md:117-136`). x3 was a planned Stage A arm; never ran
  (`S/launch/progress.md:18`).
- Still exists? Kept (`SIMP:127`).
- Size: 8 partners per anchor per tick whatever N (12.12).

### Section 7: rewards and returns

**7.1 Reward levels and weights (2.74, 2.83, 20, 20, 10)**
- Added: five levels on the v1-branch (from v1-main's 14-16 components). Base weights win 1,
  death 0.283, full shield 0.274, objectives 2 (`S/rewards.md:39-41`). x10 on 2026-11-11.
- Reason for the base numbers: v1-main "solved, not chosen": a least-squares fit in log
  space to "the one configuration measured that beat run 719, by +58 Elo at matched steps"
  (`v1:docs/training.md:284-292`).
- Reason for x10: see section 1.
- Evidence: MEASURED once on v1-main, in a different game (4v4, other physics, shaping on).
  +58 Elo is under the seed noise v2 later measured. INHERITED since.
- Still exists? PARTLY. The mix is a design choice. The x10 exists only for the fixed
  support and floor.
- Doubts: "21-22 reward weights: too large a search" (`arm-review.md:303`). A12 (dense
  rewards decayed) and A7 (offensive premium) never ran.
- Size: see 7.2-7.4.

**7.2 / 7.3 Damage and kill/death events, ledger, charge-back, unforced shares**
- Added: v1 math, ported 2026-11-01 with stated changes (`S/rewards.md:43-71`).
- Reasons, each ARGUED:
  - Zero-sum "so the critic's targets never move within a run or across a BC warm start"
    (`D:1205-1208`).
  - Damage in full shields: "it had been priced per health point, 27 wins per shield, hidden
    by the old advantage scaler" (`D:1208-1210`).
  - Charge-back: "Damage that heals did not change the game" (`S/rewards.md:58-60`).
  - Unforced shares to the opposing team: v1-main, "attributed by team rather than by
    proximity, so it survives a change of fleet size" (`v1:docs/training.md:262-266`).
- Still exists? STILL EXISTS.
- Doubts: the 2:1 offensive premium. The author "recalls an earlier test (v1 or v2.1) where
  it trained to a clearly higher Elo; v2 dropped the premium on 2026-10-30"
  (`arms-todo.md:220`). The split critic decided for it on 2026-11-14 (`arms-todo.md:56,
  189`) is unbuilt and now sits against "outcome merges into value; value to be simplified".
- Size: v1-main "divides component rewards by total ship count for team-size normalization"
  (`v1:docs/training.md:310`). v2 does not; only the unforced shares divide by team size.
  No note records dropping it.

**7.4 Objective and outcome events; `kill_lead`**
- Deathmatch outcome: kill share from `b2bea713`, then `kill_lead` default 2026-11-14.
- Evidence: MEASURED no difference (rl-mix-kl against rl-v20 within noise; deathmatch-only
  rl-kl about -350 at 75M, "around 1.6 sigma", one seed) (`overnight-log.md:370-418`).
- **Record contradicts itself.** `overnight-log.md:418`: "Settled: the deathmatch outcome
  stays kill share". `ideas.md` item 24: "kill share stays". `arms-todo.md:33`: `kill_lead`,
  "like a sports result ... Training showed no difference". Code: `kill_lead`.
- Size: how objective rewards split across a team was not traced by the inventory and no
  note covers it.

**7.5 Per-level GAE: discounts and trace decays**
- Added: per-component on v1-main (0.999 win, 0.995 kill/death, 0.991 damage, per 60 Hz
  tick). Five levels on the v1-branch. Combat levels squared 2026-11-12/13 for game v13.
- Reason: horizons follow "a credit-assignment ladder" (`v1:docs/training.md:330-333`).
  Outcome 0.9997: run 737. Squaring: "a fight takes half the ticks" (`ppo.py:141-142`).
- Evidence: INHERITED horizons; ARGUED squaring; the 2026-11-13 failure (item 1).
- BC's values (outcome 1.0 / 1.0, traces 0.98-0.99): MEASURED. With PPO's values outcome
  Brier skill was -0.23 at 6-10M; with BC's, +0.04 to +0.19 (`gpu-box-log.md:298-307`).
  These settings lived only in a queue command until an agent rebuilt runs without them.
- Still exists? STILL EXISTS as recorded; open (`core-brief.md:326`).
- Doubts: "128 was chosen by feel in v1" for the rollout length the outcome discount is tied
  to (`arm-review.md:107`). BC and RL train the critic on different targets, so the outcome
  target changes at the hand-over.
- Size: nothing recorded.

**7.7 Lookahead (128 untrained ticks) and chained advantages**
- Added 2026-11-04 (`S/rollout-lanes-brief.md:15-35`).
- Reason: "its advantages are cut at every shard"; the lookahead supplies the returns' tail
  and labels needing the future ("the bullet lives 12 ticks").
- Evidence: ARGUED from the first RL run's failure; no isolated measurement.
- Still exists? PARTLY. "Goes: bootstrap the tail from the value ... shot labels need at
  most 11 ticks" (`SIMP:558`). Chaining across shards is separate and its reason stands.

**7.8 Level sum and team spirit (0)**
- Added: v1 (OpenAI Five). "Team spirit stays as a config option, default 0" (`YR:829-830`).
- Evidence: CITED (unverified); never run above 0 in any note.
- Still exists? GONE in effect: "computes an identity (checked) ... Goes" (`SIMP:585`).
- Size: the author noted "team spirit may not stay at zero" (`diagnostics.md:473`).

**7.9 / 7.10 Return scale and advantage normalisation**: item 4.
- Size: one scalar over all modes and ship-steps (12.17, 12.45). MEASURED effect: deathmatch
  gets a 4-5x smaller advantage than the other modes; then ruled out as limiting it
  (`overnight-log.md:239-251, 316`).

**7.12 Valued mask**: v1-branch: "a downed ship still receives its team's rewards"
(`YR:702-704`). ARGUED. Still exists.

### Section 8: losses

**8.1 Policy surrogate, weight 27**: section 1. INHERITED, never re-examined in v2.

**8.2 Entropy terms**: item 2. Per-factor split added 2026-11-04 "so behaviour is unchanged
until someone sets them. Tuning comes later" (`S/rollout-lanes-brief.md:64-68`). It never
came.

**8.3 / 8.4 Value and outcome loss, 0.007**: section 1.
- Size: "Critic levels are averaged equally ... regardless of reward weight" (`YR:726-727`).
  Per-ship levels average over ship-steps, the outcome over view-steps, so their relative
  weight moves with N (12.19). The stated design claim is the opposite: a weight "does not
  move with fleet size" (`D:1202`). No measurement at another size.

**8.13 Label smoothing**: HL-Gauss sigma 0.75 from Farebrother et al. (unverified), checked
numerically for bias (`YR:572-578`). Applies to auxiliary codes only; see 5.2 for the critic.

**8.14 Grouped cross-entropy**: a memory and speed device for the coded heads
(`gpu-box-log` speed work). "Goes with the shot trajectory head" (`SIMP:544`).

**8.15 Non-finite guard**: decided in the review as "dealt with, not hidden"
(`YR:1907-1918`). Record: "The one real NaN came from the game (a 0/0); the learner never
produced one ... Goes: check once per update and stop the run" (`SIMP:581`;
`overnight-log.md:60-68`).

### Section 9: PPO and optimisation

| Row | Added | Reason / evidence | Still exists? / doubts |
|---|---|---|---|
| 9.1 Clip 0.2 | v1-branch, 2026-10-30 | Item 3 | Unrun arm A8 |
| 9.2 Rollout log-probabilities | v2, 2026-11-01, as a setting "measured in section 8" (`S/learner-spec.md:98-99`) | No result of that measurement found | Cheap check unticked (`arms-todo.md:211`) |
| 9.3 2,048 x 128 x 4 | 2026-11-05/06 (`D:1289-1290`) | "more independent games for the same trained ticks". First v2 runs were at a quarter of v1's batch (`S/rollout-lanes-brief.md:3`). v1: "small batches measurably hurt peak ratings" (`YR:1865-1866`) | A11 unrun |
| 9.4 4 epochs x 32 (RL), 256 (BC) | v1-main for RL. BC 256 from 2026-11-05/06 | RL: "inherited, never A/B tested" (`S/overnight-calibration/progress.md:6`). BC: MEASURED, "256 led 64 and 32 in frontline and deathmatch at 30M" (`calib.toml:39-41`), one seed. RL kept at 32 by "the build-on-the-winner rule" (`overnight-log.md:155`) | The RL minibatch arms were configured (`calib.toml:90-112`); no result found. A4 unrun |
| 9.5 Grad clip 1.0 | v1 | "spike guard only" (`D:1225`) | Fire rate 0 in v2; the old rationale is a stale note (`SIMP:599-600`) |
| 9.6 Micro-batches | v1 | Memory only. MEASURED independent of size to 6e-7 (float32) and 3e-3 (bf16) (`arms-todo.md:109`) | Stands |
| 9.7 KL stop 0.02 | v1-branch | Item 3 | "never fires"; "Goes" (`SIMP:582`) |
| 9.8 Adam, eps 1e-8 | v1 | "deliberately not 1e-5: measured, 1e-5 dominated the denominator" (`YR:1798-1799`) | v2 passes only `lr`; the eps is torch's default, by coincidence of the same value |
| 9.9 LR 3e-4, warm-up 10M, no decay | v1-branch 3e-4 with 1M warm-up. 10M on 2026-11-06 | Author: "The 10M warmup is also for the lr (or more accurately, primarily for the lr)" (`D:1313-1316`). No measurement. v1-main decayed to 1.5e-4; the decay was dropped without a note here ("LR-decay fork ... recorded" on the v1-branch, `D:1240`) | A9 unrun ("KL is about 0.007 against a target of 0.02", `ideas.md` item 4) |
| 9.11 bf16 | v2 | "Precision is decided by measurement" (`S/learner.md:26-35`); measured on the 3090 (`gpu-box-log`) | bf16 against fp32 spot check listed as a cheap check, not done (`arm-review.md:306`) |
| 9.13 Averaged policies | v1-branch | MEASURED there: "both averages beat the live policy about 3 games in 4 by 34M steps (run 764)" (`D:1237-1240`). EMA 0.95 is "v1's decay" (`S/league.md:20-22`). In v2: uniform best at 120M BC (+23, +30); live best late in RL (`gpu-box-log.md:642, 676-678`) | Averaging sweep R5 unrun |

### Section 10: BC

**10.1 Teacher (Imugi, T = 1, floor 0)**
- Added 2026-11-01. Floor 0.01 on 2026-11-02 so "the clone can sample every action when RL
  starts" (`S/bc.md:54-57`); later 0.
- Evidence: MEASURED cost of the floor: the floor-0.01 teacher scores 0.398 against raw
  Imugi; floor 0.05 scores 0.061 (`S/scripted-softness-notes.md:16-21`).
- Doubts: temperature target "choose after the first RL runs" (`S/bc.md:33-34`); never chosen.

**10.2 BC loss, 0.025**: v1-branch, share 0.36 on run 760. INHERITED.

**10.3 Critic and auxiliaries during BC**
- Reason: "A head start, not convergence ... the value must move anyway once RL changes the
  policy" (`S/bc.md:20-23`). Contradicts the same day's "BC must converge ... especially the
  value heads" (`S/learner.md:23-24`).

**10.4 BC opponents (half self-play, half teacher)**: review 4c, so "the student also learns
states where it faces the teacher" (`YR:1741-1742, 1755-1757`). ARGUED. Idea 17 dropped
without a test.

**10.5 Stop rule**: item 5.

**10.6 Hand-over**
- Warm start only: "No joint BC + RL paradigm" (`YR:1752-1755`). KL-to-clone dropped: "a
  single scripted teacher has no diversity to keep" (`arms-todo.md:96`).
- RL starts from the **exponential** average: "the author, 2026-11-12"
  (`behaviour-cloning.md:215-216`), decided with BC's lanes turned off. The only measurement
  favours the **uniform** average for a BC clone (989 against 966 and 965;
  `gpu-box-log.md:642`), and round four said "the uniform average for BC" (`D:1317`). No
  reason for the switch is recorded.
- Fresh optimiser and a second 10M warm-up: ARGUED (`D:1315-1316`).
- A24 (RL from scratch) never ran. In v2 nothing shows BC is necessary.

### Section 11: league

| Row | Added | Reason / evidence | Still exists? / doubts |
|---|---|---|---|
| 11.1 50% self-play; mix 6/5/5 | 0.5 "as in v1" (`S/league.md:40-41`); mix 2026-11-14 | Self-play: INHERITED, "The loop test finds no significant cycles" (`arms-todo.md:38`). Mix: MEASURED single-seed transfer table (`arms-todo.md:68-77`) | A5, A6 unrun. `launch.toml:4-5` still states the old 8/5/3 mix |
| 11.2 4 slots, one cut per 2 updates | 2026-11-01; clean cut 2026-11-06 | Simulation (`S/league-sampling-sim.md`) | Slot count never revisited (`D:1502-1503`) |
| 11.3 (1-p)^2, no floor | p(1-p) + 5% floor from review 4b; (1-p)^2 on 2026-11-14 | CITED AlphaStar (unverified). `arms-todo.md:39` says "no uniform floor", `:41` says "plus a uniform floor"; code has none. No training run compares the samplers. The p(1-p) problems in calm-fern were mainly a rating-fit bug (`league-sampling-sim.md`, cause (a)) | Untested in training |
| 11.3 Snapshot schedule | 2026-11-14 | "about 100-200 Elo between league members, read off the RL curve" of rl-v20; "re-derive it after the game freeze" (`arms-todo.md:42`) | Retention still prunes scheduled members (`arms-todo.md:17`) |
| 11.3 Cheese agents | 2026-11-14 | MEASURED need: rl-v20 scores 0.55 against the frontline zone rusher, which Imugi beats 0.75 (`arms-todo.md:126-128`) | Not yet trained with |
| 11.4 Carry kept at a cut | 2026-11-06 | MEASURED: KL to truth 4-14x lower than reset in the first half second (`S/rl-round-five-notes.md:67-78`) | Stands |
| 11.5 Rating lanes | 2026-11-04 | First runs' ratings "rest on 0-6 games per window" (`rollout-lanes-brief.md:4-5`) | "Leaning: rate from training games" (`SIMP:572`) |
| 11.7 Lineup | Fixed YYJJJ 2026-11-12; coin-flip default 2026-11-14 | Decision: coin flips (`arms-todo.md:35`). `launch.toml` still fixes YYJJJ, which overrides the draw | Launch file not updated to the decision |

### Section 12, learner-side rows

- **12.13-12.15 (reward by team size).** The only stated design reasoning is v1-main's
  (7.2 above). v2 dropped v1-main's division by ship count without a note.
- **12.16 (critic support), 12.17 (return scale).** No measurement off 5v5. The scale is
  training-only, so it does not affect zero-shot play directly; the value head's output does
  not enter the policy at play either.
- **12.19 (loss denominators).** Design claim `D:1199-1203`; no check at another N.
- **12.45 (scale over the mode mix).** MEASURED and judged harmless for deathmatch (7.9).
- **What is recorded about size generalisation itself** (all evaluation, none about the
  learner's settings):
  - v1-main run 731, trained 4v4: advantage over scripted 1.75x at training size, 1.19x at
    64v64 (`v1:docs/training-runs.md`).
  - v2 champion (2026-11-10, trained 5v5): deathmatch transfers to 20 ships (about 70% wins
    outnumbered 1.5:1) and breaks at 40; frontline 1997 / 1875 / 1134 / 526 at 5 / 10 / 20 /
    40 (`gpu-box-log.md:914-922`).
  - rl-v20: wins 26 of 64 at 20v20 frontline on the preset map; "only even with Imugi at
    2v2" (`evaluation-suite.md:137`; `arm-review.md:34`).
  - A2 (variable team sizes in training) never ran.

---

## 4. Chains of consequence

1. **Fixed categorical value support -> return units.** The support is fixed (+-2, then
   +-3 symlog), so returns must land on its bins. That produced the x10 reward factor, the
   77-bin widening, the support-clip alert and the floor's units. Remove the fixed support
   (predict normalised returns, or a scalar) and all four go.
2. **Return scale in reward units -> entropy coefficient.** The advantage's size is set by
   the scale, so a fixed entropy coefficient is only meaningful relative to it. That pairing
   produced -0.0081 (3e-4 x 27) and then its replacement by 0 at warm start. Change the
   advantage normalisation and the entropy coefficient must be set again from nothing.
3. **Return scale state -> hand-over machinery.** The scale is carried in policy files and
   checkpoints only because it is an EMA with state. A per-batch statistic removes that.
4. **Hand-built belief -> error head -> correction loss, its coefficient, its bias table,
   the physics head's hidden-row decay, the 0.0033 match.** All follow the belief out.
5. **Shot-trajectory head -> shot capture -> re-simulation -> most of the lookahead's
   label use -> grouped cross-entropy's memory tricks.** Keeping the shot-outcome head keeps
   the re-simulation unless its labels change (`SIMP:428-431`).
6. **Many heads -> gradient-share balancing -> the weights and the decomposition
   diagnostic.** With three or four heads left, the method's reason shrinks to "policy
   against the rest".
7. **Per-level horizons -> five critics -> the outcome's own 4-class head -> BC needing
   different discounts -> two critic targets across the hand-over.** Merging outcome into
   value removes the last two links only if the merged value has one target in both stages.
8. **Trunk-gradient weights + grad clip.** The weights' absolute level was chosen so the
   clip stops firing. Rescale the weights (or drop heads) and the clip may fire again; it
   then silently becomes a normaliser, as on the v1-branch.
9. **Plateau stop -> score windows -> BC lanes -> which average the stop reads.** With a
   fixed budget and BC lanes off, none of these acts.
10. **Zero-sum rewards -> no premium -> (planned) split critic.** The split critic exists in
    the plan only to make the premium cheap to vary.

## 5. Reasons that have expired

- **The gradient clip as a normaliser** (v1-branch). Fire rate 0 in v2.
- **The return-scale floor as "one win".** The outcome weight is 10; the floor is 1.
- **"The clone already samples every action (its labels' uniform floor)"** as support for
  zero entropy. The floor is 0.
- **The plateau and score stop rules.** Only the budget fires.
- **The KL early stop** in normal training. It never fires. (It did fire in the 2026-11-04
  collapse, without stopping it.)
- **Team spirit.** Identity.
- **The outcome head's separate class structure**, by decision.
- **Coefficients for the correction, shot-trajectory and (probably) density heads.**
- **The lookahead as a label source**, once the shot-trajectory head goes.
- **The four trust-region switches and the rollout's full-distribution record.**
- **The non-finite guard's skip-and-count design.** The learner never produced one.
- **Exactness as a reason** for anything: "bit for bit" trust-region default, exact resume
  of the return scale, the micro-batch denominators' exact equality.
- **Run 760's gradient shares** as the basis for today's weights: another game, another
  head set, 4v4, before the delta heads, the geometry head, the x10 and the 77 bins.

## 6. Reasons that still hold and are easy to forget

1. **A critic given a large share of the trunk hurt play.** Value solved to a 0.30 share:
   live Elo 530 against 913 and worse cloning (run 756, `D:1216-1217`). This is why value is
   0.007. Any redesign that puts the value loss on the same footing as the policy risks it.
2. **A floor that binds silently rescales a signal.** Floor 1.0 on a percentile span bound
   8 of 12 components and cut one critic gradient by four orders of magnitude (run 719,
   `v1:.../buffer.py:243-250`). The 0.1 advantage floor down-weighted the win signal about
   thirteenfold (`v1:docs/training.md:193-197`). v2's scale sat at 1.12-1.22 against its
   floor before the x10.
3. **Percentile spans fail on sparse returns.** "p5/p95 measures the width of the spike":
   0.0059 against a full range of 0.137 (`v1:docs/training.md:179-182`). v2's scale is a
   p5/p95 span of a summed return dominated by a sparse outcome level.
4. **A long outcome horizon with a 128-tick bootstrap drifts at gamma 1** (run 737,
   `YR:691-695`): explained variance 0.994 -> 0.42. Relevant to any merged value.
5. **In BC, a discounted, bootstrapped outcome target gave negative Brier skill**
   (`gpu-box-log.md:298-307`). A default nobody wrote down caused a whole round to run on
   the wrong base.
6. **An entropy bonus sized for one advantage scale can swamp the policy gradient at
   another** (shoot share 1.7% -> 6.8%, score 0.23 -> 0.03 in 24 updates;
   `ppo.py:207-212`).
7. **A quarter-size batch and advantages cut at each shard accompanied the first RL
   collapse** (`S/rollout-lanes-brief.md:3-5`). v1: small batches "measurably hurt peak
   ratings" (`YR:1865-1866`).
8. **RL can unlearn fighting in about 60 updates while objectives keep being captured**
   (item 1). The alert exists; the cause was not isolated.
9. **Seeds shift every arm together by about 110 Elo; compare like seeds only** (item 6).
10. **Scoring changes look like skill changes.** Deathmatch "lagged" at about 1200 on kill
    share and led at about 2800 on win/loss, same policy (`overnight-log.md:289-304`).
11. **A rating-fit bug, not the sampler, made the league play beaten opponents**
    (`S/league-sampling-sim.md`, cause (a)): snapshot ratings of 7,000-65,000.
12. **Short CPU runs mislead on gradient shares** (`overnight-log.md:128`). Three of today's
    weights were set on such runs.
13. **BC KL does not track strength** (`overnight-log.md:113`).
14. **Single-mode training forgets the other modes**: deathmatch-only RL took frontline from
    1548 to 914 between 50M and 75M (`overnight-log.md:318-327`).
15. **Damage priced per health point was 27 wins per shield, "hidden by the old advantage
    scaler"** (`D:1208-1210`). A normaliser can hide a reward-unit error.

## 7. Never tested

**Adopted without a measurement in this record**
- Categorical value against scalar value (A21). The switch itself.
- The return scale's form, decay and floor; un-normalised advantages against per-batch.
- Reward weights in v2's game; the x10 was a unit change, not a test.
- Per-level discounts' values; the x2-game squaring.
- Clip 0.2 against anything; 4 epochs; RL's 32 minibatches; learning rate; no LR decay; the
  10M warm-up; the second warm-up after the hand-over.
- Value 0.007, policy 27, enemy action 0.11, density 0.021, shot outcome 0.019 **in v2**.
- Starting RL from the exponential average.
- (1-p)^2 against p(1-p) in training; the 50% self-play share; 4 slots.
- BC opponents' half-and-half split; the teacher's temperature.
- Whether the auxiliary heads help at all (A13), and whether BC is needed (A24).

**Planned and never run**
- A3 seed pairs; A4 minibatches; A5 mix; A6 one-sided self-play; A7 offensive premium (and
  its split critic); A8 DPPO, with its cheap TV pre-check; A9 learning rate; A10 entropy;
  A11 rollout size; A12 dense-reward decay; A13 no auxiliaries; A21 value form; A24 from
  scratch; R5 averaging sweep.
- The `single-value` arm (gamma 0.995, lambda 0.95) and geometry x3 (`S/launch/arms.toml`).
- The RL minibatch arms of 2026-11-13 (`calib.toml:90-112`): configured; no result found.
- Old log-probabilities check; seat symmetry; bf16 against fp32 spot check
  (`arms-todo.md:209-212`).
- Carry-staleness (burn-in) diagnostic; privileged critic (`D:1499-1505`).
- Monte Carlo returns by rolling out to the episode's end (`S/learner.md:55-59`).
- HL-Gauss with calibrated bin values for the critic: decided, not built.
- The teacher's entropy target: "choose after the first RL runs".

## 8. Could not find

- **Anything from the v1-branch itself**: pilots 754-764, run 737, run 748,
  `docs/internal/*-oct2026.json`, `benchmarks/gradient_decomposition.py`. The branch is not
  in this clone. Every number attributed to it here is second-hand from `D` or `YR`.
- **The `fix/rl-degradation` notes** (`/tmp/bnb-scratch-rl-degradation/progress.md`) behind
  the entropy decision: the ablation table and which other suspects were cleared.
- **Any analysis of `rl-check`** (the stop-fighting run) beyond two sentences, and the
  results of the BC and RL minibatch arms run that night
  (`S/overnight-calibration/progress.md` stops at the first launch).
- **Why the v1-branch returned to p5/p95 with floor 1** after v1-main had moved to a
  standard deviation with an epsilon floor.
- **Why v2 does not divide rewards by ship count** as v1-main did.
- **Why the learning-rate decay was dropped**, and why RL starts from the exponential
  average.
- **A measured gradient share for the value term in v2**, before or after the x10.
- **A result for "old log-probabilities: measured in section 8"**.
- **Commit dates before 2026-11-15** (shallow clone).
- **Any record on how objective rewards divide among a team's ships.**

**Where the record contradicts itself or the code**
- Deathmatch outcome: "stays kill share" against `kill_lead` (7.4).
- League floor: "no uniform floor" against "plus a uniform floor" (11.3).
- Critic targets: HL-Gauss decided, two-hot in code (5.2).
- BC's purpose: "must converge ... especially the value heads" against "a head start, not
  convergence" (10.3).
- BC's product: "uniform average for BC" against exponential at the hand-over (10.6).
- Entropy: A10 speaks of "today's bonus"; the RL stage has none (item 2).
- `launch.toml`: stale mode mix and a fixed lineup, against the 2026-11-14 decisions.
- The inventory's v1 summary gives the `ReturnScaler` as an EMA of the 5th/95th
  percentiles; `origin/main`'s code and docs use an EMA of mean and standard deviation, the
  percentile form being the one it replaced.
- The plan: one value head and one discount "settled" in a register, "open" in its
  questions (`review.md:84-86`).
